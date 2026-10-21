"""Device capability model for the Quantum Nanorobot Runtime (QNRS).

Every quantum instrument the runtime can drive is described by a
:class:`QuantumDeviceCapabilities` record. Adapters produce these records during
discovery; the negotiation engine consumes them to pick a control protocol.

Nothing here imports a quantum SDK — the model is pure Python so that discovery
works even when no vendor library is installed.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence


class ProtocolError(RuntimeError):
    """Raised when a device, protocol or capability contract is violated."""


class QuantumBackendType(str, Enum):
    """Physical realisation of the quantum processing unit."""

    SUPERCONDUCTING_QUBIT = "superconducting_qubit"
    TRAPPED_ION = "trapped_ion"
    PHOTONIC = "photonic"
    NEUTRAL_ATOM = "neutral_atom"
    NV_CENTER = "nv_center"
    SIMULATOR = "simulator"
    HYBRID = "hybrid"


class ControlProtocol(str, Enum):
    """How the runtime is allowed to talk to a device."""

    GATE_SEQUENCE = "gate_sequence"          # logical gates, device transpiles
    PULSE_SEQUENCE = "pulse_sequence"        # explicit analog waveforms
    ANALOG_CONTROL = "analog_control"        # continuous drive / DC bias
    MEASUREMENT_FEEDBACK = "measurement_feedback"  # mid-circuit readout + branch
    HYBRID_CONTROL = "hybrid_control"        # gates now, pulses for calibration


#: Lower value == preferred when several protocols are mutually supported.
PROTOCOL_PRIORITY: Dict[ControlProtocol, int] = {
    ControlProtocol.MEASUREMENT_FEEDBACK: 0,
    ControlProtocol.PULSE_SEQUENCE: 1,
    ControlProtocol.HYBRID_CONTROL: 2,
    ControlProtocol.GATE_SEQUENCE: 3,
    ControlProtocol.ANALOG_CONTROL: 4,
}


@dataclass
class QuantumDeviceCapabilities:
    """Static + calibrated description of one quantum instrument.

    Times are microseconds unless the field name says otherwise. All fidelities
    are in ``[0, 1]``.
    """

    backend_type: QuantumBackendType
    num_qubits: int
    native_gates: List[str] = field(default_factory=list)
    supported_protocols: List[ControlProtocol] = field(
        default_factory=lambda: [ControlProtocol.GATE_SEQUENCE]
    )

    max_circuit_depth: int = 0
    max_shots: int = 100_000

    # Calibration data
    t1_us: float = 0.0
    t2_us: float = 0.0
    gate_time_1q_ns: float = 0.0
    gate_time_2q_ns: float = 0.0
    single_qubit_fidelity: float = 1.0
    two_qubit_fidelity: float = 1.0
    readout_fidelity: float = 1.0
    supports_mid_circuit_measurement: bool = False

    calibration_timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Coerce raw strings coming off the wire into enums.
        if isinstance(self.backend_type, str):
            self.backend_type = QuantumBackendType(self.backend_type)
        self.supported_protocols = [
            p if isinstance(p, ControlProtocol) else ControlProtocol(p)
            for p in self.supported_protocols
        ]
        if not self.supported_protocols:
            raise ProtocolError("device must advertise at least one control protocol")
        if self.num_qubits < 1:
            raise ProtocolError(f"num_qubits must be >= 1, got {self.num_qubits}")
        if self.max_circuit_depth < 0:
            raise ProtocolError("max_circuit_depth must be >= 0")
        for name in ("single_qubit_fidelity", "two_qubit_fidelity", "readout_fidelity"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ProtocolError(f"{name} must be within [0, 1], got {value}")
        for name in ("t1_us", "t2_us", "gate_time_1q_ns", "gate_time_2q_ns"):
            if getattr(self, name) < 0:
                raise ProtocolError(f"{name} must be >= 0")
        if self.t1_us and self.t2_us and self.t2_us > 2.0 * self.t1_us:
            # Physical bound: T2 <= 2*T1. Tolerate tiny calibration overshoot.
            if self.t2_us > 2.05 * self.t1_us:
                raise ProtocolError(
                    f"unphysical calibration: T2={self.t2_us:g}us > 2*T1={2 * self.t1_us:g}us"
                )
        if self.supports_mid_circuit_measurement and (
            ControlProtocol.MEASUREMENT_FEEDBACK not in self.supported_protocols
        ):
            self.supported_protocols.append(ControlProtocol.MEASUREMENT_FEEDBACK)

    # ------------------------------------------------------------------ helpers
    @property
    def effective_coherence_us(self) -> float:
        """Coherence window usable for a computation (bounded by T2)."""
        candidates = [c for c in (self.t2_us, self.t1_us) if c > 0]
        return min(candidates) if candidates else math.inf

    def supports(self, protocol: ControlProtocol) -> bool:
        return protocol in self.supported_protocols

    def estimated_duration_us(self, circuit_depth: int, two_qubit_fraction: float = 0.3) -> float:
        """Rough wall-clock estimate for a circuit of the given depth.

        A layer is assumed to cost a weighted mix of 1q and 2q gate times.
        """
        if circuit_depth < 0:
            raise ProtocolError("circuit_depth must be >= 0")
        frac = min(max(two_qubit_fraction, 0.0), 1.0)
        layer_ns = (1 - frac) * self.gate_time_1q_ns + frac * self.gate_time_2q_ns
        return circuit_depth * layer_ns / 1000.0

    def fits_coherence(self, circuit_depth: int, safety_factor: float = 0.5) -> bool:
        """True if the circuit provably fits inside the coherence window."""
        window = self.effective_coherence_us
        if math.isinf(window):
            return True
        return self.estimated_duration_us(circuit_depth) <= safety_factor * window

    def merged(self, **overrides: Any) -> "QuantumDeviceCapabilities":
        """Return a copy with calibration fields overridden (live re-calibration)."""
        data = asdict(self)
        data.pop("calibration_timestamp", None)
        data.update(overrides)
        data["calibration_timestamp"] = self.calibration_timestamp
        return QuantumDeviceCapabilities(**data)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["backend_type"] = self.backend_type.value
        data["supported_protocols"] = [p.value for p in self.supported_protocols]
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "QuantumDeviceCapabilities":
        return cls(**dict(data))


@dataclass
class DiscoveryMessage:
    """Signed advertisement broadcast by an instrument on the control bus.

    The signature is an HMAC-SHA256 over a canonical JSON encoding of the
    message body (excluding the ``signature`` field itself) using a pre-shared
    bus key. This gives integrity + authenticity without a PKI on the bench.
    """

    device_id: str
    device_name: str
    capabilities: QuantumDeviceCapabilities
    endpoint: str = ""
    nonce: str = ""
    timestamp: float = field(default_factory=time.time)
    signature: str = ""

    def _body(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "device_name": self.device_name,
            "endpoint": self.endpoint,
            "nonce": self.nonce,
            "timestamp": self.timestamp,
            "capabilities": self.capabilities.to_dict(),
        }

    def canonical_bytes(self) -> bytes:
        return json.dumps(self._body(), sort_keys=True, separators=(",", ":")).encode("utf-8")

    def sign(self, key: str) -> str:
        self.signature = hmac.new(key.encode("utf-8"), self.canonical_bytes(), hashlib.sha256).hexdigest()
        return self.signature

    def verify(self, key: str, max_age_s: float = 300.0, clock_skew_s: float = 5.0) -> bool:
        """Constant-time signature check plus replay-window check."""
        if not self.signature:
            return False
        expected = hmac.new(key.encode("utf-8"), self.canonical_bytes(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, self.signature):
            return False
        age = time.time() - self.timestamp
        return -clock_skew_s <= age <= max_age_s

    @classmethod
    def create(
        cls,
        device_id: str,
        device_name: str,
        capabilities: QuantumDeviceCapabilities,
        key: str,
        endpoint: str = "",
        nonce: str = "",
    ) -> "DiscoveryMessage":
        msg = cls(
            device_id=device_id,
            device_name=device_name,
            capabilities=capabilities,
            endpoint=endpoint,
            nonce=nonce or hashlib.sha256(f"{device_id}{time.time_ns()}".encode()).hexdigest()[:16],
        )
        msg.sign(key)
        return msg
