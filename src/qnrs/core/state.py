"""Quantum state tracking with an explicit decoherence model.

Pure numpy so the runtime can track, checkpoint and predict state fidelity on a
classical host without any quantum SDK. The evolution and noise channel code uses
exact linear algebra (eigendecomposition of Hermitian generators, Kraus
operators) rather than a truncated series, so it stays correct for strong drives.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class StateType(str, Enum):
    PURE = "pure"
    MIXED = "mixed"


class CoherenceLevel(str, Enum):
    """Bucketed health of a state relative to its target, by fidelity."""

    EXCELLENT = "excellent"   # >= 0.95
    GOOD = "good"             # >= 0.85
    ACCEPTABLE = "acceptable" # >= 0.70
    DEGRADED = "degraded"     # >= 0.50
    LOST = "lost"             # <  0.50

    @classmethod
    def classify(cls, fidelity: float) -> "CoherenceLevel":
        if fidelity >= 0.95:
            return cls.EXCELLENT
        if fidelity >= 0.85:
            return cls.GOOD
        if fidelity >= 0.70:
            return cls.ACCEPTABLE
        if fidelity >= 0.50:
            return cls.DEGRADED
        return cls.LOST


class StateError(RuntimeError):
    """Raised for malformed states or invalid evolution parameters."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_complex_1d(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.complex128).reshape(-1)
    if arr.size == 0:
        raise StateError("state vector must be non-empty")
    return arr


def partial_trace(rho: np.ndarray, num_qubits: int, keep: Tuple[int, ...]) -> np.ndarray:
    """Trace out all qubits except ``keep`` (big-endian qubit ordering, q0 leftmost)."""
    if rho.shape != (2 ** num_qubits, 2 ** num_qubits):
        raise StateError(f"rho shape {rho.shape} inconsistent with {num_qubits} qubits")
    if any(q < 0 or q >= num_qubits for q in keep):
        raise StateError(f"qubits to keep out of range: {keep}")
    if len(set(keep)) != len(keep):
        raise StateError(f"duplicate qubits in keep={keep}")
    tensor = rho.reshape([2] * (2 * num_qubits))
    remaining = list(range(num_qubits))
    # Trace out the highest-index qubits first so the lower axes stay aligned.
    for qubit in sorted((q for q in remaining if q not in keep), reverse=True):
        axis = remaining.index(qubit)
        tensor = np.trace(tensor, axis1=axis, axis2=axis + len(remaining))
        remaining.pop(axis)
    dim = 2 ** len(keep)
    return tensor.reshape(dim, dim)


@dataclass
class DecoherenceModel:
    """T1/T2 relaxation + pure dephasing, plus a surface-code logical error model.

    Fidelity decay is estimated as the *worse* of two independent bounds:

    * coherence (dephasing):  ``F_phi(t) = exp(-t / T2)``
    * relaxation:             ``F_rel(t) = exp(-t / (2*T1))``

    For the common T2 <= T1 regime the dephasing term dominates, which matches
    the conservative assumption we want for control planning.
    """

    t1_us: float
    t2_us: float
    temperature_mk: float = 15.0
    #: Surface-code constants: logical error rate p_L = A * (p / p_th) ** ((d+1)/2)
    surface_code_threshold: float = 0.01
    surface_code_prefactor: float = 0.03

    def __post_init__(self) -> None:
        if self.t1_us <= 0 or self.t2_us <= 0:
            raise StateError("T1 and T2 must be positive")
        if self.t2_us > 2.0 * self.t1_us + 1e-9:
            raise StateError(f"unphysical T2={self.t2_us} > 2*T1={2 * self.t1_us}")

    @property
    def pure_dephasing_time_us(self) -> float:
        """T_phi from 1/T2 = 1/(2*T1) + 1/T_phi."""
        inv = 1.0 / self.t2_us - 1.0 / (2.0 * self.t1_us)
        if inv <= 0:
            return math.inf
        return 1.0 / inv

    def fidelity_decay(self, time_us: float) -> float:
        """Predicted fidelity of an idle state after ``time_us`` microseconds."""
        if time_us < 0:
            raise StateError("time_us must be >= 0")
        dephasing = math.exp(-time_us / self.t2_us)
        relaxation = math.exp(-time_us / (2.0 * self.t1_us))
        return min(dephasing, relaxation)

    def logical_error_rate(self, physical_error_rate: float, distance: int) -> float:
        """Surface-code logical error rate for a given code distance ``d``.

        Valid for odd ``d >= 3``; below threshold the rate is suppressed as
        ``(p/p_th) ** ((d+1)/2)``. Above threshold the code provides no benefit.
        """
        if distance < 3 or distance % 2 == 0:
            raise StateError("code distance must be an odd integer >= 3")
        if physical_error_rate <= 0:
            return 0.0
        if physical_error_rate >= self.surface_code_threshold:
            return min(0.5, physical_error_rate)
        ratio = physical_error_rate / self.surface_code_threshold
        return self.surface_code_prefactor * ratio ** ((distance + 1) // 2)

    def correction_gain(self, physical_error_rate: float, distance: int = 3) -> float:
        """Fidelity multiplier from one correction cycle (>= 1.0, capped at 1/F).

        Returns the factor by which residual error is reduced, i.e.
        ``min(1/F, 1 / max(p_L, eps))``.
        """
        residual = self.logical_error_rate(physical_error_rate, distance)
        if residual <= 0:
            return math.inf if physical_error_rate > 0 else 1.0
        return min(1.0 / max(residual, 1e-12), 1.0 / max(physical_error_rate, 1e-12))


@dataclass
class QuantumState:
    """A tracked quantum state (vector for pure, matrix for mixed)."""

    state_vector: Optional[np.ndarray] = None
    density_matrix: Optional[np.ndarray] = None
    state_type: StateType = StateType.PURE
    num_qubits: int = 0
    fidelity: float = 1.0
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        dim = 2 ** self.num_qubits if self.num_qubits else 0
        if self.state_vector is not None:
            self.state_vector = _as_complex_1d(self.state_vector)
            if dim and self.state_vector.size != dim:
                raise StateError(f"state vector has {self.state_vector.size} amplitudes, expected {dim}")
            if not dim:
                self.num_qubits = int(round(math.log2(self.state_vector.size)))
            norm = np.linalg.norm(self.state_vector)
            if norm == 0:
                raise StateError("state vector has zero norm")
            self.state_vector = self.state_vector / norm
            self.state_type = StateType.PURE
            if self.density_matrix is None:
                self.density_matrix = np.outer(self.state_vector, self.state_vector.conj())
        if self.density_matrix is not None:
            self.density_matrix = np.asarray(self.density_matrix, dtype=np.complex128)
            if self.density_matrix.shape[0] != self.density_matrix.shape[1]:
                raise StateError("density matrix must be square")
            if not self.num_qubits:
                self.num_qubits = int(round(math.log2(self.density_matrix.shape[0])))
            trace = float(np.real(np.trace(self.density_matrix)))
            if not math.isclose(trace, 1.0, abs_tol=1e-6):
                raise StateError(f"density matrix trace must be 1, got {trace:.6f}")
        if self.state_vector is None and self.density_matrix is None:
            raise StateError("a state needs either a state_vector or a density_matrix")
        if self.state_type is StateType.PURE and self.state_vector is None:
            # Mixed state handed in: classify it for the caller.
            self.state_type = StateType.MIXED

    # ------------------------------------------------------------------ queries
    @property
    def dim(self) -> int:
        return 2 ** self.num_qubits

    def age_us(self, now: Optional[datetime] = None) -> float:
        now = now or _utcnow()
        return (now - self.created_at).total_seconds() * 1e6

    def is_pure(self) -> bool:
        if self.density_matrix is None:
            return True
        purity = float(np.real(np.trace(self.density_matrix @ self.density_matrix)))
        return math.isclose(purity, 1.0, abs_tol=1e-6)

    def coherence_level(self, threshold: float = 0.85) -> CoherenceLevel:
        return CoherenceLevel.classify(self.fidelity)

    def is_coherent(self, threshold: float = 0.85) -> bool:
        return self.fidelity >= threshold

    def reduced(self, keep: Tuple[int, ...]) -> np.ndarray:
        """Reduced density matrix over ``keep`` qubits."""
        return partial_trace(self.density_matrix, self.num_qubits, keep)

    def clone(self) -> "QuantumState":
        return QuantumState(
            state_vector=None if self.state_vector is None else self.state_vector.copy(),
            density_matrix=None if self.density_matrix is None else self.density_matrix.copy(),
            state_type=self.state_type,
            num_qubits=self.num_qubits,
            fidelity=self.fidelity,
            metadata=dict(self.metadata),
        )


class QuantumStateManager:
    """Owns the live state, checkpoints, and the decoherence bookkeeping."""

    def __init__(
        self,
        num_qubits: int,
        coherence_time_us: Optional[float] = None,
        decoherence_model: Optional[DecoherenceModel] = None,
    ) -> None:
        if num_qubits < 1:
            raise StateError("num_qubits must be >= 1")
        self.num_qubits = num_qubits
        self.model = decoherence_model or DecoherenceModel(
            t1_us=coherence_time_us or 100.0,
            t2_us=coherence_time_us or 100.0,
        )
        self.current: Optional[QuantumState] = None
        self.history: List[QuantumState] = []
        self._checkpoints: Dict[str, QuantumState] = {}
        self._target: Optional[QuantumState] = None

    @property
    def dim(self) -> int:
        return 2 ** self.num_qubits

    # ------------------------------------------------------------------ lifecycle
    def initialize(self, state_vector: Optional[np.ndarray] = None,
                   density_matrix: Optional[np.ndarray] = None) -> QuantumState:
        if state_vector is None and density_matrix is None:
            state_vector = np.zeros(2 ** self.num_qubits, dtype=np.complex128)
            state_vector[0] = 1.0
        self.current = QuantumState(
            state_vector=state_vector,
            density_matrix=density_matrix if state_vector is None else None,
            num_qubits=self.num_qubits,
            fidelity=1.0,
        )
        # If a target was registered, fidelity is measured against it from the start.
        self.current.fidelity = self._fidelity_vs_target(self.current.density_matrix)
        self.history.append(self.current)
        return self.current

    def checkpoint(self, checkpoint_id: str) -> QuantumState:
        if self.current is None:
            raise StateError("no live state to checkpoint")
        snapshot = self.current.clone()
        snapshot.metadata["checkpoint_id"] = checkpoint_id
        self._checkpoints[checkpoint_id] = snapshot
        return snapshot

    def restore(self, checkpoint_id: str) -> QuantumState:
        if checkpoint_id not in self._checkpoints:
            raise StateError(f"unknown checkpoint {checkpoint_id!r}")
        self.current = self._checkpoints[checkpoint_id].clone()
        self.current.metadata["restored_from"] = checkpoint_id
        self.history.append(self.current)
        return self.current

    def list_checkpoints(self) -> List[str]:
        return sorted(self._checkpoints)

    # ------------------------------------------------------------------ evolution
    def evolve(self, hamiltonian: np.ndarray, time_us: float) -> QuantumState:
        """Exact unitary evolution ``U = exp(-i H t)`` for Hermitian ``H``.

        ``hamiltonian`` is in units of rad/us. Uses the eigendecomposition of H
        (exact for any Hermitian matrix, no series truncation).
        """
        if self.current is None:
            raise StateError("initialize a state before evolving it")
        h = np.asarray(hamiltonian, dtype=np.complex128)
        if h.shape != (self.dim, self.dim):
            raise StateError(f"H must be {self.dim}x{self.dim}, got {h.shape}")
        if not np.allclose(h, h.conj().T, atol=1e-9):
            raise StateError("H must be Hermitian")
        energies, vectors = np.linalg.eigh(h)
        phases = np.exp(-1j * energies * time_us)
        u = (vectors * phases) @ vectors.conj().T
        return self.apply_unitary(u, label="evolve")

    def apply_unitary(self, unitary: np.ndarray, label: str = "") -> QuantumState:
        if self.current is None:
            raise StateError("initialize a state before applying a unitary")
        u = np.asarray(unitary, dtype=np.complex128)
        if u.shape != (self.dim, self.dim):
            raise StateError(f"unitary must be {self.dim}x{self.dim}, got {u.shape}")
        rho = u @ self.current.density_matrix @ u.conj().T
        return self._adopt(rho, label or "unitary")

    def apply_kraus(self, kraus_ops: List[np.ndarray], label: str = "") -> QuantumState:
        """Apply a CPTP map sum_i K_i rho K_i^dagger, checking completeness."""
        if self.current is None:
            raise StateError("initialize a state before applying a channel")
        if not kraus_ops:
            raise StateError("kraus_ops must be non-empty")
        completeness = sum(k.conj().T @ k for k in kraus_ops)
        if not np.allclose(completeness, np.eye(self.dim), atol=1e-6):
            raise StateError("Kraus operators are not trace-preserving")
        rho = np.zeros_like(self.current.density_matrix)
        for k in kraus_ops:
            rho += k @ self.current.density_matrix @ k.conj().T
        return self._adopt(rho, label or "kraus")

    def apply_noise(self, duration_us: float, label: str = "idle-noise") -> QuantumState:
        """Apply amplitude + phase damping for an idle period of ``duration_us``."""
        return self.apply_kraus(self.noise_kraus(duration_us), label=label)

    def noise_kraus(self, duration_us: float) -> List[np.ndarray]:
        """Tensor product of amplitude- and phase-damping channels per qubit."""
        if duration_us < 0:
            raise StateError("duration_us must be >= 0")
        gamma = 1.0 - math.exp(-duration_us / self.model.t1_us)  # relaxation
        lam = 1.0 - math.exp(-duration_us / max(self.model.pure_dephasing_time_us, 1e-12))

        amp = [
            np.array([[1.0, 0.0], [0.0, math.sqrt(1 - gamma)]], dtype=np.complex128),
            np.array([[0.0, math.sqrt(gamma)], [0.0, 0.0]], dtype=np.complex128),
        ]
        phase = [
            np.array([[1.0, 0.0], [0.0, math.sqrt(1 - lam)]], dtype=np.complex128),
            np.array([[0.0, 0.0], [0.0, math.sqrt(lam)]], dtype=np.complex128),
        ]

        ops: List[np.ndarray] = [np.array([[1.0]], dtype=np.complex128)]
        for q in range(self.num_qubits):
            # Compose relaxation then dephasing as K = P_j @ A_i so that
            # sum K^dagger K = I (a valid CPTP channel).
            channel = [p @ a for a in amp for p in phase]
            new_ops = []
            for existing in ops:
                for k in channel:
                    new_ops.append(np.kron(existing, k))
            ops = new_ops
        return ops

    # ------------------------------------------------------------------ prediction
    def predict_fidelity(self, operation_duration_us: float) -> Tuple[float, CoherenceLevel]:
        """Fidelity after an operation, and its bucket."""
        decay = self.model.fidelity_decay(operation_duration_us)
        base = self.current.fidelity if self.current else 1.0
        fidelity = base * decay
        return fidelity, CoherenceLevel.classify(fidelity)

    def coherence_status(self) -> Dict[str, object]:
        if self.current is None:
            return {"status": "uninitialized", "num_qubits": self.num_qubits}
        age = self.current.age_us()
        predicted, level = self.predict_fidelity(age)
        return {
            "status": "tracking",
            "num_qubits": self.num_qubits,
            "fidelity": round(self.current.fidelity, 6),
            "predicted_fidelity": round(predicted, 6),
            "age_us": round(age, 3),
            "t1_us": self.model.t1_us,
            "t2_us": self.model.t2_us,
            "coherence_level": level.value,
            "requires_correction": level in (CoherenceLevel.DEGRADED, CoherenceLevel.LOST),
        }

    # ------------------------------------------------------------------ internals
    def _adopt(self, density_matrix: np.ndarray, label: str) -> QuantumState:
        state = QuantumState(
            density_matrix=density_matrix,
            num_qubits=self.num_qubits,
            fidelity=self._fidelity_vs_target(density_matrix),
            metadata={"operation": label},
        )
        self.current = state
        self.history.append(state)
        return state

    def set_target(self, target: np.ndarray) -> None:
        self._target = QuantumState(state_vector=_as_complex_1d(target), num_qubits=self.num_qubits)

    def _fidelity_vs_target(self, rho: np.ndarray) -> float:
        """Uhlmann fidelity vs the stored target (1.0 when no target is set)."""
        if self._target is None:
            return 1.0
        sigma = self._target.density_matrix
        sqrt_rho = _matrix_sqrt(rho)
        inner = sqrt_rho @ sigma @ sqrt_rho
        return float(np.real(np.trace(_matrix_sqrt(inner))) ** 2)


def _matrix_sqrt(matrix: np.ndarray) -> np.ndarray:
    """Hermitian PSD square root via eigh (numpy-only)."""
    matrix = np.asarray(matrix, dtype=np.complex128)
    matrix = (matrix + matrix.conj().T) / 2.0
    values, vectors = np.linalg.eigh(matrix)
    values = np.clip(values, 0.0, None)
    return (vectors * np.sqrt(values)) @ vectors.conj().T
