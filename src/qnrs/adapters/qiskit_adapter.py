"""Qiskit adapter: gate-level execution against local or IBM backends.

Discovery is intentionally conservative:

* ``qiskit:statevector`` and ``qiskit:aer`` are published whenever the local
  simulator stack is importable (offline, deterministic, used by CI);
* IBM Runtime backends are published only when ``QISKIT_IBM_TOKEN`` (or a
  pre-configured service) is present, so the control plane never blocks on the
  network at import time.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, List, Optional

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
from qnrs.core.pulses import PulseSequence

try:  # pragma: no cover - exercised implicitly by the environment
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import Statevector

    QISKIT_AVAILABLE = True
except Exception:  # pragma: no cover
    QISKIT_AVAILABLE = False

try:  # pragma: no cover
    from qiskit_aer import AerSimulator

    AER_AVAILABLE = True
except Exception:  # pragma: no cover
    AER_AVAILABLE = False

try:  # pragma: no cover
    from qiskit.primitives import StatevectorSampler

    SAMPLER_AVAILABLE = True
except Exception:  # pragma: no cover
    SAMPLER_AVAILABLE = False


#: Generic gate name -> Qiskit method name.
_GATE_METHODS = {
    "H": "h", "X": "x", "Y": "y", "Z": "z", "S": "s", "SDG": "sdg",
    "T": "t", "TDG": "tdg", "SX": "sx", "RX": "rx", "RY": "ry", "RZ": "rz",
    "PHASE": "p", "P": "p", "CNOT": "cx", "CX": "cx", "CZ": "cz", "SWAP": "swap",
}

LOCAL_BACKENDS = {
    "qiskit:statevector": {
        "backend_type": QuantumBackendType.SIMULATOR,
        "num_qubits": 12,
        "native_gates": ["id", "rz", "sx", "x", "cx", "measure"],
        "engine": "qiskit.primitives.StatevectorSampler",
    },
    "qiskit:aer": {
        "backend_type": QuantumBackendType.SIMULATOR,
        "num_qubits": 24,
        "native_gates": ["id", "rz", "sx", "x", "cx", "measure", "reset"],
        "engine": "qiskit_aer.AerSimulator",
    },
}


def _counts_from_data_bin(data: Any, num_qubits: int) -> Dict[str, int]:
    """Extract counts from a Qiskit 2.x ``DataBin``.

    Qiskit names the measurement field after the classical register (``c`` by
    default, ``meas`` in the older API), so locate it by capability instead of
    by name.
    """
    keys = data.keys() if hasattr(data, "keys") else []
    for key in keys:
        field = getattr(data, key, None)
        if hasattr(field, "get_counts"):
            return {str(k): int(v) for k, v in field.get_counts().items()}
    raise AdapterError("Qiskit sampler returned no measurement field")


class QiskitAdapter(QuantumBackendAdapter):
    """Drives Qiskit circuits; degrades gracefully if Qiskit is absent."""

    name = "qiskit"

    def __init__(self, bus_key: str = "", service: Any = None, include_ibm: Optional[bool] = None) -> None:
        super().__init__(bus_key=bus_key)
        if not QISKIT_AVAILABLE:
            raise AdapterError("qiskit is not importable; install qnrs[qiskit]")
        self._service = service
        self._ibm_backends: Dict[str, Any] = {}
        self._caps: Dict[str, QuantumDeviceCapabilities] = {}
        self._include_ibm = include_ibm if include_ibm is not None else bool(os.environ.get("QISKIT_IBM_TOKEN"))
        self._aer = AerSimulator() if AER_AVAILABLE else None

    # ------------------------------------------------------------------ discovery
    async def discover(self) -> List[DiscoveryMessage]:
        messages: List[DiscoveryMessage] = []
        available = {"qiskit:statevector": SAMPLER_AVAILABLE, "qiskit:aer": self._aer is not None}
        if not any(available.values()):
            return []
        for device_id, spec in LOCAL_BACKENDS.items():
            if not available[device_id]:
                continue
            caps = self._localhost_capabilities(device_id, spec)
            self._caps[device_id] = caps
            messages.append(self._advertise(device_id, f"Qiskit {device_id.split(':')[1]}", caps))
            if device_id == "qiskit:aer":
                noisy = self._noisy_capabilities()
                self._caps["qiskit:aer-noisy"] = noisy
                messages.append(self._advertise("qiskit:aer-noisy", "Qiskit aer (noisy)", noisy))

        if self._include_ibm:
            messages.extend(await self._discover_ibm())

        return self._remember(messages)

    def _localhost_capabilities(self, device_id: str, spec: Dict[str, Any]) -> QuantumDeviceCapabilities:
        return QuantumDeviceCapabilities(
            backend_type=spec["backend_type"],
            num_qubits=spec["num_qubits"],
            native_gates=list(spec["native_gates"]),
            supported_protocols=[
                ControlProtocol.GATE_SEQUENCE,
                ControlProtocol.MEASUREMENT_FEEDBACK,
            ],
            max_circuit_depth=10_000,
            max_shots=1_000_000,
            t1_us=0.0,   # 0 => ideal, unlimited coherence window
            t2_us=0.0,
            gate_time_1q_ns=35.0,
            gate_time_2q_ns=300.0,
            single_qubit_fidelity=1.0,
            two_qubit_fidelity=1.0,
            readout_fidelity=1.0,
            supports_mid_circuit_measurement=True,
            metadata={"engine": spec["engine"], "ideal": True},
        )

    def _noisy_capabilities(self) -> QuantumDeviceCapabilities:
        """A representative superconducting device model for planner testing."""
        return QuantumDeviceCapabilities(
            backend_type=QuantumBackendType.SUPERCONDUCTING_QUBIT,
            num_qubits=12,
            native_gates=["id", "rz", "sx", "x", "cx", "measure"],
            supported_protocols=[
                ControlProtocol.GATE_SEQUENCE,
                ControlProtocol.MEASUREMENT_FEEDBACK,
            ],
            max_circuit_depth=1_000,
            max_shots=200_000,
            t1_us=180.0,
            t2_us=140.0,
            gate_time_1q_ns=35.0,
            gate_time_2q_ns=320.0,
            single_qubit_fidelity=0.9993,
            two_qubit_fidelity=0.9890,
            readout_fidelity=0.9720,
            supports_mid_circuit_measurement=True,
            metadata={"engine": "qiskit_aer+noise-model", "ideal": False},
        )

    async def _discover_ibm(self) -> List[DiscoveryMessage]:
        messages: List[DiscoveryMessage] = []
        try:  # pragma: no cover - requires credentials
            if self._service is None:
                from qiskit_ibm_runtime import QiskitRuntimeService

                self._service = QiskitRuntimeService()
            for backend in self._service.backends():
                device_id = f"ibm:{backend.name}"
                self._ibm_backends[device_id] = backend
                caps = self._ibm_capabilities(device_id, backend)
                self._caps[device_id] = caps
                messages.append(
                    self._advertise(device_id, f"IBM {backend.name}", caps, endpoint="qiskit-ibm-runtime")
                )
        except Exception:
            return messages
        return messages

    def _ibm_capabilities(self, device_id: str, backend: Any) -> QuantumDeviceCapabilities:  # pragma: no cover
        num_qubits = int(getattr(backend, "num_qubits", 0) or 0)
        props = None
        try:
            props = backend.properties()
        except Exception:
            props = None
        t1 = t2 = 0.0
        readout = 1.0
        if props is not None:
            t1_values = [props.t1(q) for q in range(num_qubits)]
            t2_values = [props.t2(q) for q in range(num_qubits)]
            t1 = float(np.median([v for v in t1_values if v])) if any(t1_values) else 0.0
            t2 = float(np.median([v for v in t2_values if v])) if any(t2_values) else 0.0
            readouts = [props.readout_error(q) for q in range(num_qubits)]
            readout = float(1.0 - np.median(readouts)) if readouts else 1.0
        return QuantumDeviceCapabilities(
            backend_type=QuantumBackendType.SUPERCONDUCTING_QUBIT,
            num_qubits=num_qubits or 1,
            native_gates=["id", "rz", "sx", "x", "cx", "reset", "measure"],
            supported_protocols=[ControlProtocol.GATE_SEQUENCE, ControlProtocol.MEASUREMENT_FEEDBACK],
            max_circuit_depth=10_000,
            max_shots=100_000,
            t1_us=t1,
            t2_us=min(t2, 2 * t1) if t1 else t2,
            gate_time_1q_ns=35.0,
            gate_time_2q_ns=500.0,
            single_qubit_fidelity=0.9995,
            two_qubit_fidelity=0.99,
            readout_fidelity=readout,
            supports_mid_circuit_measurement=True,
            metadata={"engine": "qiskit-ibm-runtime", "ideal": False},
        )

    async def verify(self, device_id: str) -> bool:
        if device_id in LOCAL_BACKENDS:
            return self._caps.get(device_id) is not None
        if device_id in self._ibm_backends:  # pragma: no cover
            try:
                return bool(self._ibm_backends[device_id].status().operational)
            except Exception:
                return False
        return False

    async def capabilities(self, device_id: str) -> QuantumDeviceCapabilities:
        if device_id not in self._caps:
            await self.discover()
        if device_id not in self._caps:
            raise AdapterError(f"unknown Qiskit device {device_id!r}")
        return self._caps[device_id]

    # ------------------------------------------------------------------ execution
    async def execute_circuit(self, device_id: str, program: CircuitProgram) -> ExecutionResult:
        start = time.perf_counter()
        job_id = f"qk-{uuid.uuid4().hex[:12]}"
        caps = await self.capabilities(device_id)
        if device_id == "qiskit:aer-noisy":
            device_key = "qiskit:aer"
        else:
            device_key = device_id

        circuit = None
        engine = caps.metadata.get("engine", "qiskit")

        try:
            circuit = self._build_circuit(program)
            counts: Dict[str, int] = {}
            statevector: Optional[List[complex]] = None
            if device_key == "qiskit:aer" and self._aer is not None:
                result = self._aer.run(circuit, shots=program.shots, seed_simulator=program.seed).result()
                counts = {str(k): int(v) for k, v in result.get_counts().items()}
            elif SAMPLER_AVAILABLE:
                sampler = StatevectorSampler(default_shots=program.shots, seed=program.seed)
                result = sampler.run([circuit], shots=program.shots).result()
                data = result[0].data
                counts = _counts_from_data_bin(data, program.num_qubits)
            else:  # pragma: no cover
                raise AdapterError("no Qiskit sampler available")

            bare = self._build_circuit(program, measure=False)
            statevector = np.asarray(Statevector.from_instruction(bare).data, dtype=np.complex128).tolist()

            return ExecutionResult(
                job_id=job_id,
                device_id=device_id,
                adapter=self.name,
                status="completed",
                counts=counts,
                statevector=statevector,
                probabilities=(np.abs(np.asarray(statevector)) ** 2).tolist(),
                shots=program.shots,
                duration_us=(time.perf_counter() - start) * 1e6,
                engine=str(engine),
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
                engine=str(engine),
                error=f"{type(exc).__name__}: {exc}",
            )

    async def execute_pulse(self, device_id: str, sequence: PulseSequence) -> ExecutionResult:
        """Qiskit pulse was removed in Qiskit 1.x; route pulses to schedule info.

        The returned payload contains the resolved schedule so a Labber/QCoDeS
        bridge can execute it. We report ``status="delegated"`` rather than
        pretending the pulse ran on the Qiskit backend.
        """
        schedule = sequence.schedule()
        return ExecutionResult(
            job_id=f"qk-pulse-{uuid.uuid4().hex[:8]}",
            device_id=device_id,
            adapter=self.name,
            status="delegated",
            shots=sequence.repetitions,
            duration_us=sequence.compute_total_duration() / 1000.0,
            engine="qnrs.pulse-ir",
            metadata={
                "program": sequence.analog_program(),
                "n_pulses": len(schedule),
                "required_channels": sequence.required_channels,
            },
        )

    async def read_state(self, device_id: str) -> Dict[str, Any]:
        caps = await self.capabilities(device_id)
        return {
            "device_id": device_id,
            "ideal": bool(caps.metadata.get("ideal", True)),
            "queue_depth": 0,
            "calibration_timestamp": caps.calibration_timestamp,
        }

    # ------------------------------------------------------------------ internals
    def _build_circuit(self, program: CircuitProgram, measure: Optional[bool] = None) -> Any:
        """Build a Qiskit circuit.

        ``measure=None`` (the sampling path) always appends terminal
        measurements: shots are only meaningful with readout, and the QNRS
        circuit IR does not carry explicit measurement operations, so "measure
        nothing" is indistinguishable from "measure everything" here.
        ``measure=False`` builds the bare circuit used for the state vector.
        """
        circuit = QuantumCircuit(program.num_qubits, program.num_qubits)
        for op in program.operations:
            self._append_op(circuit, op)
        do_measure = True if measure is None else measure
        if do_measure:
            circuit.measure(range(program.num_qubits), range(program.num_qubits))
        return circuit

    def _append_op(self, circuit: Any, op: GateOperation) -> None:
        method_name = _GATE_METHODS.get(op.key)
        if method_name is None:
            raise AdapterError(f"Qiskit adapter cannot express gate {op.name!r}")
        method = getattr(circuit, method_name, None)
        if method is None:  # pragma: no cover - very old Qiskit
            raise AdapterError(f"Qiskit backend lacks gate method {method_name!r}")
        if op.params:
            method(*op.params, *op.qubits)
        else:
            method(*op.qubits)
