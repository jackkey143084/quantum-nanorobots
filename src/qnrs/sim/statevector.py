"""NumPy state-vector simulator used as the always-available fallback backend.

It exists so the whole stack (discovery -> negotiation -> job execution ->
telemetry) is runnable and testable with no vendor SDK installed. When Qiskit or
PennyLane is present the adapters prefer them, but nothing in the control plane
depends on that.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_SQRT2 = math.sqrt(2.0)

GATES: Dict[str, np.ndarray] = {
    "I": np.eye(2, dtype=np.complex128),
    "H": np.array([[1, 1], [1, -1]], dtype=np.complex128) / _SQRT2,
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
    "S": np.array([[1, 0], [0, 1j]], dtype=np.complex128),
    "SDG": np.array([[1, 0], [0, -1j]], dtype=np.complex128),
    "T": np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=np.complex128),
    "TDG": np.array([[1, 0], [0, np.exp(-1j * math.pi / 4)]], dtype=np.complex128),
    "SX": np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=np.complex128) / 2.0,
}

TWO_QUBIT_GATES = {"CNOT", "CX", "CZ", "SWAP"}


class SimulationError(RuntimeError):
    """Raised for malformed circuits or unsupported operations."""


def _rotation_gate(axis: np.ndarray, theta: float) -> np.ndarray:
    return math.cos(theta / 2) * np.eye(2, dtype=np.complex128) - 1j * math.sin(theta / 2) * axis


def parametrised_gate(name: str, params: Sequence[float]) -> np.ndarray:
    """Return a 2x2 matrix for a parametrised single-qubit gate."""
    upper = name.upper()
    if upper == "RX":
        return _rotation_gate(GATES["X"], float(params[0]))
    if upper == "RY":
        return _rotation_gate(GATES["Y"], float(params[0]))
    if upper == "RZ":
        theta = float(params[0])
        return np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=np.complex128)
    if upper == "PHASE" or upper == "P":
        return np.diag([1.0, np.exp(1j * float(params[0]))]).astype(np.complex128)
    raise SimulationError(f"unsupported parametrised gate {name!r}")


def gate_matrix(name: str, params: Sequence[float] = ()) -> np.ndarray:
    upper = name.upper()
    if params:
        return parametrised_gate(upper, params)
    if upper in GATES:
        return GATES[upper]
    raise SimulationError(f"unknown gate {name!r}")


def _apply_single(state: np.ndarray, matrix: np.ndarray, target: int, num_qubits: int) -> np.ndarray:
    tensor = state.reshape([2] * num_qubits)
    tensor = np.tensordot(matrix, tensor, axes=([1], [target]))
    tensor = np.moveaxis(tensor, 0, target)
    return tensor.reshape(-1)


def _apply_two_qubit(state: np.ndarray, matrix: np.ndarray, q0: int, q1: int, num_qubits: int) -> np.ndarray:
    tensor = state.reshape([2] * num_qubits)
    # Reshape the two target axes into one, apply, then restore.
    tensor = np.moveaxis(tensor, [q0, q1], [0, 1])
    shape = tensor.shape
    flat = tensor.reshape(4, -1)
    flat = matrix @ flat
    tensor = flat.reshape(shape)
    tensor = np.moveaxis(tensor, [0, 1], [q0, q1])
    return tensor.reshape(-1)


def _two_qubit_matrix(name: str) -> np.ndarray:
    upper = name.upper()
    if upper in ("CNOT", "CX"):
        return np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex128
        )
    if upper == "CZ":
        return np.diag([1, 1, 1, -1]).astype(np.complex128)
    if upper == "SWAP":
        return np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.complex128
        )
    raise SimulationError(f"unknown two-qubit gate {name!r}")


class StateVectorSimulator:
    """Minimal, dependency-free state-vector engine."""

    def __init__(self, num_qubits: int) -> None:
        if num_qubits < 1:
            raise SimulationError("num_qubits must be >= 1")
        self.num_qubits = num_qubits
        self.state = np.zeros(2 ** num_qubits, dtype=np.complex128)
        self.state[0] = 1.0

    def reset(self) -> None:
        self.state = np.zeros(2 ** self.num_qubits, dtype=np.complex128)
        self.state[0] = 1.0

    def apply(self, name: str, qubits: Sequence[int], params: Sequence[float] = ()) -> None:
        self._check_qubits(qubits)
        if len(qubits) == 1:
            matrix = gate_matrix(name, params)
            self.state = _apply_single(self.state, matrix, qubits[0], self.num_qubits)
        elif len(qubits) == 2:
            if params:
                raise SimulationError("two-qubit gates do not take parameters here")
            matrix = _two_qubit_matrix(name)
            self.state = _apply_two_qubit(self.state, matrix, qubits[0], qubits[1], self.num_qubits)
        else:
            raise SimulationError(f"unsupported gate arity {len(qubits)} for {name!r}")

    def probabilities(self) -> np.ndarray:
        return np.abs(self.state) ** 2

    def sample_counts(self, shots: int, seed: Optional[int] = None) -> Dict[str, int]:
        if shots < 1:
            raise SimulationError("shots must be >= 1")
        rng = np.random.default_rng(seed)
        outcomes = rng.choice(len(self.state), size=shots, p=self.probabilities() / self.probabilities().sum())
        counts: Dict[str, int] = {}
        for outcome in outcomes:
            bitstring = format(int(outcome), f"0{self.num_qubits}b")
            counts[bitstring] = counts.get(bitstring, 0) + 1
        return counts

    def fidelity(self, target: Optional[np.ndarray] = None) -> float:
        if target is None:
            return 1.0
        target = np.asarray(target, dtype=np.complex128).reshape(-1)
        return float(abs(np.vdot(target, self.state)) ** 2)

    def _check_qubits(self, qubits: Sequence[int]) -> None:
        for q in qubits:
            if q < 0 or q >= self.num_qubits:
                raise SimulationError(f"qubit index {q} out of range for {self.num_qubits} qubits")


def run_circuit(num_qubits: int, operations: Sequence[Tuple[str, Sequence[int], Sequence[float]]],
                shots: int = 1024, seed: Optional[int] = None) -> Dict[str, object]:
    """Execute a flat operation list and return state vector + sampled counts."""
    sim = StateVectorSimulator(num_qubits)
    for name, qubits, params in operations:
        sim.apply(name, qubits, params)
    return {
        "statevector": sim.state.tolist(),
        "probabilities": sim.probabilities().tolist(),
        "counts": sim.sample_counts(shots, seed=seed),
        "engine": "qnrs.statevector",
    }
