"""PennyLane adapter: differentiable simulation and variational control.

Two capabilities:

1. ``execute_circuit`` — differentiable execution of a QNRS circuit program.
2. ``optimize`` — variational training of a hardware-efficient ansatz toward a
   target state (this is the "PennyLane ML" leg of the stack: gradient-based
   pulse/parameter search for nanorobot state preparation).

When PennyLane is unavailable, :class:`_NumpyVariational` provides a
finite-difference optimizer over the same ansatz using the built-in
state-vector engine, so the ML path still works offline.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from qnrs.adapters.base import (
    AdapterError,
    CircuitProgram,
    ExecutionResult,
    GateOperation,
    QuantumBackendAdapter,
)
from qnrs.core.capabilities import (
    ControlProtocol,
    DiscoveryMessage,
    QuantumBackendType,
    QuantumDeviceCapabilities,
)
from qnrs.sim.statevector import StateVectorSimulator

try:  # pragma: no cover
    import pennylane as qml

    PENNYLANE_AVAILABLE = True
except Exception:  # pragma: no cover
    PENNYLANE_AVAILABLE = False


_PL_SINGLE = {
    "H": "Hadamard", "X": "PauliX", "Y": "PauliY", "Z": "PauliZ", "S": "S", "T": "T",
    "SX": "SX", "SDG": "S", "TDG": "T",
}
_PL_PARAM = {"RX": "RX", "RY": "RY", "RZ": "RZ", "PHASE": "PhaseShift", "P": "PhaseShift"}
_PL_TWO = {"CNOT": "CNOT", "CX": "CNOT", "CZ": "CZ", "SWAP": "SWAP"}


@dataclass
class VariationalTemplate:
    """Hardware-efficient ansatz: RZ-RY-RZ per wire, then a CNOT ring."""

    num_qubits: int
    layers: int = 2
    entangler: str = "CNOT"

    def __post_init__(self) -> None:
        if self.num_qubits < 1 or self.layers < 1:
            raise AdapterError("num_qubits and layers must be >= 1")
        if self.entangler.upper() not in ("CNOT", "CX", "CZ"):
            raise AdapterError(f"unsupported entangler {self.entangler!r}")
        if self.num_qubits < 2 and self.layers > 1:
            raise AdapterError("entangling ansatz with >1 layer needs at least 2 qubits")

    @property
    def parameter_shape(self) -> tuple:
        return (self.layers, self.num_qubits, 3)

    def initial_parameters(self, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.uniform(-math.pi, math.pi, size=self.parameter_shape)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VariationalResult:
    optimal_parameters: List[float]
    parameter_shape: tuple
    best_cost: float
    final_fidelity: float
    history: List[float] = field(default_factory=list)
    steps: int = 0
    optimizer: str = ""
    engine: str = ""
    duration_us: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "optimal_parameters": self.optimal_parameters,
            "parameter_shape": list(self.parameter_shape),
            "best_cost": self.best_cost,
            "final_fidelity": self.final_fidelity,
            "history": self.history,
            "steps": self.steps,
            "optimizer": self.optimizer,
            "engine": self.engine,
            "duration_us": self.duration_us,
        }


class PennyLaneAdapter(QuantumBackendAdapter):
    """Differentiable simulator + variational optimizer."""

    name = "pennylane"

    DEVICE_ID = "pennylane:default.qubit"

    def __init__(self, bus_key: str = "", wires: int = 16) -> None:
        super().__init__(bus_key=bus_key)
        self._wires = wires
        self._caps = QuantumDeviceCapabilities(
            backend_type=QuantumBackendType.SIMULATOR,
            num_qubits=wires,
            native_gates=sorted(set(_PL_SINGLE) | set(_PL_PARAM) | set(_PL_TWO)),
            supported_protocols=[ControlProtocol.GATE_SEQUENCE],
            max_circuit_depth=10_000,
            max_shots=1_000_000,
            gate_time_1q_ns=20.0,
            gate_time_2q_ns=120.0,
            metadata={
                "engine": "pennylane.default.qubit" if PENNYLANE_AVAILABLE else "numpy.fallback",
                "differentiable": True,
            },
        )

    # ------------------------------------------------------------------ discovery
    async def discover(self) -> List[DiscoveryMessage]:
        message = self._advertise(self.DEVICE_ID, "PennyLane default.qubit (differentiable)", self._caps)
        return self._remember([message])

    async def verify(self, device_id: str) -> bool:
        return device_id == self.DEVICE_ID

    async def capabilities(self, device_id: str) -> QuantumDeviceCapabilities:
        if device_id != self.DEVICE_ID:
            raise AdapterError(f"unknown PennyLane device {device_id!r}")
        return self._caps

    # ------------------------------------------------------------------ execution
    async def execute_circuit(self, device_id: str, program: CircuitProgram) -> ExecutionResult:
        start = time.perf_counter()
        job_id = f"pl-{uuid.uuid4().hex[:12]}"
        if program.num_qubits > self._wires:
            raise AdapterError(f"program needs {program.num_qubits} wires, device has {self._wires}")
        try:
            state = self._execute_statevector(program)
            probabilities = np.abs(state) ** 2
            rng = np.random.default_rng(program.seed)
            outcomes = rng.choice(len(state), size=program.shots, p=probabilities / probabilities.sum())
            counts: Dict[str, int] = {}
            for outcome in outcomes:
                key = format(int(outcome), f"0{program.num_qubits}b")
                counts[key] = counts.get(key, 0) + 1
            return ExecutionResult(
                job_id=job_id,
                device_id=device_id,
                adapter=self.name,
                status="completed",
                counts=counts,
                statevector=state.tolist(),
                probabilities=probabilities.tolist(),
                shots=program.shots,
                duration_us=(time.perf_counter() - start) * 1e6,
                engine="pennylane.default.qubit" if PENNYLANE_AVAILABLE else "qnrs.sim.statevector",
                metadata={"depth": program.depth, "gate_counts": program.gate_counts},
            )
        except Exception as exc:
            return ExecutionResult(
                job_id=job_id,
                device_id=device_id,
                adapter=self.name,
                status="failed",
                shots=program.shots,
                duration_us=(time.perf_counter() - start) * 1e6,
                engine="pennylane",
                error=f"{type(exc).__name__}: {exc}",
            )

    def _execute_statevector(self, program: CircuitProgram) -> np.ndarray:
        if not PENNYLANE_AVAILABLE:
            sim = StateVectorSimulator(program.num_qubits)
            for op in program.operations:
                sim.apply(op.name, op.qubits, op.params)
            return sim.state

        dev = qml.device("default.qubit", wires=program.num_qubits)

        @qml.qnode(dev)
        def circuit() -> Any:
            for op in program.operations:
                self._apply_op(op)
            return qml.state()

        return np.asarray(circuit(), dtype=np.complex128).reshape(-1)

    @staticmethod
    def _apply_op(op: GateOperation) -> None:
        key = op.key
        if key in _PL_SINGLE:
            getattr(qml, _PL_SINGLE[key])(wires=op.qubits)
        elif key in _PL_PARAM:
            getattr(qml, _PL_PARAM[key])(op.params[0], wires=op.qubits[0])
        elif key in _PL_TWO:
            getattr(qml, _PL_TWO[key])(wires=op.qubits)
        else:  # pragma: no cover
            raise AdapterError(f"PennyLane cannot express gate {op.name!r}")

    # ------------------------------------------------------------------ variational ML
    async def optimize(
        self,
        target_state: Sequence[complex],
        template: Optional[VariationalTemplate] = None,
        steps: int = 60,
        learning_rate: float = 0.2,
        seed: int = 0,
        device_id: Optional[str] = None,
    ) -> VariationalResult:
        """Train the ansatz to maximise fidelity with ``target_state``."""
        target = np.asarray(target_state, dtype=np.complex128).reshape(-1)
        num_qubits = int(round(math.log2(target.size)))
        if 2 ** num_qubits != target.size:
            raise AdapterError("target_state length must be a power of two")
        template = template or VariationalTemplate(num_qubits=num_qubits)
        if template.num_qubits != num_qubits:
            raise AdapterError("template width does not match target state")
        norm = np.linalg.norm(target)
        if norm == 0:
            raise AdapterError("target_state has zero norm")
        target = target / norm
        params0 = template.initial_parameters(seed=seed)
        start = time.perf_counter()

        if PENNYLANE_AVAILABLE:
            params, history, best = self._optimize_pennylane(template, target, params0, steps, learning_rate)
            engine = "pennylane.Adam"
        else:
            params, history, best = self._optimize_numpy(template, target, params0, steps, learning_rate)
            engine = "numpy.finite-difference"

        fidelity = self._ansatz_fidelity(template, params, target)
        return VariationalResult(
            optimal_parameters=np.asarray(params).reshape(-1).tolist(),
            parameter_shape=template.parameter_shape,
            best_cost=best,
            final_fidelity=fidelity,
            history=history,
            steps=steps,
            optimizer=engine,
            engine=engine,
            duration_us=(time.perf_counter() - start) * 1e6,
        )

    def _optimize_pennylane(
        self,
        template: VariationalTemplate,
        target: np.ndarray,
        params0: np.ndarray,
        steps: int,
        learning_rate: float,
    ) -> tuple:
        dev = qml.device("default.qubit", wires=template.num_qubits)

        @qml.qnode(dev)
        def ansatz(weights: Any) -> Any:
            self._apply_ansatz(template, weights)
            return qml.state()

        target_tensor = qml.math.stack([qml.math.cast(v, "complex128") for v in target])

        def cost(weights: Any) -> Any:
            psi = ansatz(weights)
            overlap = qml.math.sum(qml.math.conj(target_tensor) * psi)
            return 1.0 - qml.math.real(overlap * qml.math.conj(overlap))

        weights = qml.numpy.array(params0, requires_grad=True)
        optimizer = qml.AdamOptimizer(stepsize=learning_rate)
        history: List[float] = []
        best = float("inf")
        best_weights = params0.copy()
        for _ in range(steps):
            updated = optimizer.step(cost, weights)
            # PennyLane returns the new argument(s); unwrap for a single tensor.
            if isinstance(updated, (tuple, list)):
                weights = updated[0]
            else:
                weights = updated
            value = float(np.real(cost(weights)))
            history.append(value)
            if value < best:
                best = value
                best_weights = np.asarray(qml.math.toarray(weights), dtype=float).copy()
        return best_weights, history, best

    def _optimize_numpy(
        self,
        template: VariationalTemplate,
        target: np.ndarray,
        params0: np.ndarray,
        steps: int,
        learning_rate: float,
    ) -> tuple:
        params = params0.copy()
        epsilon = 1e-3
        history: List[float] = []
        best = float("inf")
        best_params = params.copy()
        for _ in range(steps):
            base = self._ansatz_fidelity(template, params, target)
            cost = 1.0 - base
            history.append(cost)
            if cost < best:
                best = cost
                best_params = params.copy()
            grad = np.zeros_like(params)
            flat = params.reshape(-1)
            for i in range(flat.size):
                shifted = flat.copy()
                shifted[i] += epsilon
                plus = self._ansatz_fidelity(template, shifted.reshape(params.shape), target)
                grad.reshape(-1)[i] = (1.0 - plus - cost) / epsilon
            params = params - learning_rate * grad
        return best_params, history, best

    def _ansatz_fidelity(self, template: VariationalTemplate, params: np.ndarray,
                         target: np.ndarray) -> float:
        state = self._ansatz_state(template, params)
        return float(abs(np.vdot(target, state)) ** 2)

    def _ansatz_state(self, template: VariationalTemplate, params: np.ndarray) -> np.ndarray:
        weights = np.asarray(params, dtype=float).reshape(template.parameter_shape)
        if PENNYLANE_AVAILABLE:
            dev = qml.device("default.qubit", wires=template.num_qubits)

            @qml.qnode(dev)
            def circuit() -> Any:
                self._apply_ansatz(template, weights)
                return qml.state()

            return np.asarray(circuit(), dtype=np.complex128).reshape(-1)

        sim = StateVectorSimulator(template.num_qubits)
        self._apply_ansatz_numpy(sim, template, weights)
        return sim.state

    @staticmethod
    def _apply_ansatz(template: VariationalTemplate, weights: Any) -> None:
        for layer in range(template.layers):
            for wire in range(template.num_qubits):
                qml.RZ(weights[layer, wire, 0], wires=wire)
                qml.RY(weights[layer, wire, 1], wires=wire)
                qml.RZ(weights[layer, wire, 2], wires=wire)
            if template.num_qubits > 1:
                entangler = getattr(qml, "CNOT" if template.entangler.upper() in ("CNOT", "CX") else "CZ")
                for wire in range(template.num_qubits - 1):
                    entangler(wires=[wire, wire + 1])
                if template.num_qubits > 2:
                    entangler(wires=[template.num_qubits - 1, 0])

    @staticmethod
    def _apply_ansatz_numpy(sim: StateVectorSimulator, template: VariationalTemplate,
                            weights: np.ndarray) -> None:
        entangler = "CNOT" if template.entangler.upper() in ("CNOT", "CX") else "CZ"
        for layer in range(template.layers):
            for wire in range(template.num_qubits):
                sim.apply("RZ", [wire], [weights[layer, wire, 0]])
                sim.apply("RY", [wire], [weights[layer, wire, 1]])
                sim.apply("RZ", [wire], [weights[layer, wire, 2]])
            if template.num_qubits > 1:
                for wire in range(template.num_qubits - 1):
                    sim.apply(entangler, [wire, wire + 1])
                if template.num_qubits > 2:
                    sim.apply(entangler, [template.num_qubits - 1, 0])

    async def read_state(self, device_id: str) -> Dict[str, Any]:
        return {"device_id": device_id, "differentiable": True, "pennyLane": PENNYLANE_AVAILABLE}
