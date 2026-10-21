"""Adapter contract + the vendor-neutral circuit IR.

An adapter is simultaneously:

* a :class:`~qnrs.core.discovery.ProtocolNegotiator` — it can advertise and
  health-check the devices it owns; and
* an executor — it can turn a :class:`CircuitProgram` or
  :class:`~qnrs.core.pulses.PulseSequence` into results.

Keeping both on one object means the gateway can route by ``device_id`` without
a separate registry of "who can run what".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from qnrs.core.capabilities import DiscoveryMessage, QuantumDeviceCapabilities
from qnrs.core.discovery import ProtocolNegotiator
from qnrs.core.pulses import PulseSequence


class AdapterError(RuntimeError):
    """Raised when a backend adapter cannot service a request."""


@dataclass
class GateOperation:
    """One logical operation in a circuit program."""

    name: str
    qubits: List[int] = field(default_factory=list)
    params: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.name:
            raise AdapterError("gate name must not be empty")
        if not self.qubits:
            raise AdapterError(f"gate {self.name!r} needs at least one qubit")
        self.qubits = [int(q) for q in self.qubits]
        self.params = [float(p) for p in self.params]

    @property
    def key(self) -> str:
        return self.name.upper()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GateOperation":
        return cls(name=data["name"], qubits=list(data.get("qubits", [])), params=list(data.get("params", [])))


@dataclass
class CircuitProgram:
    """Vendor-neutral circuit description submitted by a client."""

    num_qubits: int
    operations: List[GateOperation] = field(default_factory=list)
    shots: int = 1024
    measure_all: bool = True
    seed: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_qubits < 1:
            raise AdapterError("num_qubits must be >= 1")
        if self.shots < 1:
            raise AdapterError("shots must be >= 1")
        for op in self.operations:
            for q in op.qubits:
                if q < 0 or q >= self.num_qubits:
                    raise AdapterError(f"gate {op.name!r} targets qubit {q} out of range")
        self.operations = [
            op if isinstance(op, GateOperation) else GateOperation.from_dict(op) for op in self.operations
        ]

    @property
    def depth(self) -> int:
        """Naive ASAP depth: a layer per op that reuses an already-used qubit."""
        layers: List[set] = []
        for op in self.operations:
            placed = False
            for layer in layers:
                if layer.isdisjoint(op.qubits):
                    layer.update(op.qubits)
                    placed = True
                    break
            if not placed:
                layers.append(set(op.qubits))
        return len(layers)

    @property
    def gate_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for op in self.operations:
            counts[op.key] = counts.get(op.key, 0) + 1
        return counts

    def operation_tuples(self) -> List[tuple]:
        return [(op.name, tuple(op.qubits), tuple(op.params)) for op in self.operations]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_qubits": self.num_qubits,
            "operations": [op.to_dict() for op in self.operations],
            "shots": self.shots,
            "measure_all": self.measure_all,
            "seed": self.seed,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CircuitProgram":
        return cls(
            num_qubits=data["num_qubits"],
            operations=[GateOperation.from_dict(o) for o in data.get("operations", [])],
            shots=data.get("shots", 1024),
            measure_all=data.get("measure_all", True),
            seed=data.get("seed"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class SubmitResult:
    """Handle returned the moment a job is accepted by an instrument."""

    job_id: str
    device_id: str
    adapter: str
    accepted_at: float
    estimated_duration_us: float = 0.0
    queue_position: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionResult:
    """Outcome of a circuit or pulse job."""

    job_id: str
    device_id: str
    adapter: str
    status: str = "completed"
    counts: Dict[str, int] = field(default_factory=dict)
    statevector: Optional[List[complex]] = None
    probabilities: Optional[List[float]] = None
    fidelity: Optional[float] = None
    readout: List[Dict[str, Any]] = field(default_factory=list)
    shots: int = 0
    duration_us: float = 0.0
    engine: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_statevector: bool = True) -> Dict[str, Any]:
        payload = asdict(self)
        if not include_statevector:
            payload.pop("statevector", None)
        return payload


class QuantumBackendAdapter(ProtocolNegotiator, ABC):
    """Base class for every vendor adapter."""

    #: Human-readable adapter name (e.g. ``"qiskit"``).
    name: str = "adapter"

    def __init__(self, bus_key: str = "") -> None:
        super().__init__()
        self.bus_key = bus_key

    # ------------------------------------------------------------------ queries
    @abstractmethod
    async def capabilities(self, device_id: str) -> QuantumDeviceCapabilities:
        """Current (possibly re-calibrated) capabilities for a device."""

    # ------------------------------------------------------------------ execution
    @abstractmethod
    async def execute_circuit(self, device_id: str, program: CircuitProgram) -> ExecutionResult:
        """Run a logical circuit program."""

    async def execute_pulse(self, device_id: str, sequence: PulseSequence) -> ExecutionResult:
        raise AdapterError(f"{self.name} does not accept pulse-level programs")

    async def read_state(self, device_id: str) -> Dict[str, Any]:
        """Best-effort live device state (temperature, drift, queue depth...)."""
        return {}

    async def close(self) -> None:
        """Release network/hardware handles."""

    # ------------------------------------------------------------------ helpers
    def _advertise(
        self,
        device_id: str,
        device_name: str,
        capabilities: QuantumDeviceCapabilities,
        endpoint: str = "",
    ) -> DiscoveryMessage:
        if self.bus_key:
            return DiscoveryMessage.create(device_id, device_name, capabilities, self.bus_key, endpoint=endpoint)
        return DiscoveryMessage(
            device_id=device_id,
            device_name=device_name,
            capabilities=capabilities,
            endpoint=endpoint,
        )
