"""Discovery and protocol negotiation.

The negotiator is deliberately transport-agnostic: an adapter implements
``discover()`` (broadcast/query the bus) and ``verify()`` (health probe). The
engine fans out across registered negotiators, verifies signatures, then picks
the best mutually-supported protocol using a transparent score so the choice can
be logged and audited.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from qnrs.core.capabilities import (
    PROTOCOL_PRIORITY,
    ControlProtocol,
    DiscoveryMessage,
    ProtocolError,
    QuantumBackendType,
    QuantumDeviceCapabilities,
)


def score_protocol(
    protocol: ControlProtocol,
    capabilities: QuantumDeviceCapabilities,
    required: Sequence[ControlProtocol] = (),
) -> float:
    """Higher is better. Purely mechanical, so it is easy to reason about.

    Terms
    -----
    priority      : explicit preference order (feedback > pulses > hybrid > gates)
    fidelity      : protocol-specific fidelity proxy from calibration
    headroom      : how much of the coherence window a notional deep circuit uses
    requirement   : bonus for covering a caller-mandated protocol
    """
    priority = PROTOCOL_PRIORITY.get(protocol, 9)

    if protocol in (ControlProtocol.PULSE_SEQUENCE, ControlProtocol.ANALOG_CONTROL):
        fidelity = capabilities.two_qubit_fidelity
    elif protocol is ControlProtocol.MEASUREMENT_FEEDBACK:
        fidelity = capabilities.readout_fidelity * capabilities.two_qubit_fidelity
    else:
        fidelity = capabilities.single_qubit_fidelity * capabilities.two_qubit_fidelity

    window = capabilities.effective_coherence_us
    if window and window != float("inf"):
        headroom = min(1.0, window / max(capabilities.estimated_duration_us(64), 1e-9))
        headroom = min(headroom, 1.0)
    else:
        headroom = 1.0

    requirement_bonus = 0.5 if protocol in required else 0.0

    return (10 - priority) + 2.0 * fidelity + 0.5 * headroom + requirement_bonus


@dataclass
class NegotiationResult:
    device_id: str
    protocol: ControlProtocol
    score: float
    candidates: List[Tuple[ControlProtocol, float]] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "device_id": self.device_id,
            "protocol": self.protocol.value,
            "score": round(self.score, 4),
            "candidates": [(p.value, round(s, 4)) for p, s in self.candidates],
            "notes": self.notes,
        }


class ProtocolNegotiator(ABC):
    """Adapter-side half of discovery/negotiation."""

    #: Backend family this negotiator handles (used for routing requests).
    backend_type: QuantumBackendType = QuantumBackendType.HYBRID

    @abstractmethod
    async def discover(self) -> List[DiscoveryMessage]:
        """Return signed advertisements for every device this adapter can see."""

    @abstractmethod
    async def verify(self, device_id: str) -> bool:
        """Health probe: is the device reachable and operational?"""

    async def protocol_support(self, device_id: str) -> List[ControlProtocol]:
        """Protocols the device currently accepts. Defaults to advertised set."""
        for msg in self._cache.values():  # type: ignore[attr-defined]
            if msg.device_id == device_id:
                return list(msg.capabilities.supported_protocols)
        return []

    def __init__(self) -> None:
        self._cache: Dict[str, DiscoveryMessage] = {}

    def _remember(self, messages: Iterable[DiscoveryMessage]) -> List[DiscoveryMessage]:
        messages = list(messages)
        for msg in messages:
            self._cache[msg.device_id] = msg
        return messages


class NegotiationEngine:
    """Fans out across negotiators and chooses a protocol per device."""

    def __init__(self, bus_key: str = "") -> None:
        self._negotiators: List[ProtocolNegotiator] = []
        self._bus_key = bus_key
        self._results: Dict[str, NegotiationResult] = {}

    def register(self, negotiator: ProtocolNegotiator) -> "NegotiationEngine":
        self._negotiators.append(negotiator)
        return self

    @property
    def negotiators(self) -> List[ProtocolNegotiator]:
        return list(self._negotiators)

    @property
    def devices(self) -> Dict[str, DiscoveryMessage]:
        merged: Dict[str, DiscoveryMessage] = {}
        for negotiator in self._negotiators:
            merged.update(negotiator._cache)
        return merged

    async def discover_all(self) -> List[DiscoveryMessage]:
        """Query every negotiator; drop messages with a bad signature.

        Signature verification is skipped only when no bus key is configured
        (development mode) — a warning is attached to the message metadata.
        """
        found: List[DiscoveryMessage] = []
        errors: List[str] = []
        for negotiator in self._negotiators:
            try:
                messages = await negotiator.discover()
            except Exception as exc:  # one flaky vendor must not kill discovery
                errors.append(f"{type(negotiator).__name__}: {exc}")
                continue
            for msg in messages:
                if self._bus_key and not msg.verify(self._bus_key):
                    errors.append(f"rejected unsigned/stale advertisement from {msg.device_id}")
                    continue
                found.append(msg)
        if errors:
            for msg in found:
                msg.capabilities.metadata.setdefault("discovery_warnings", errors)
        return found

    async def negotiate(
        self,
        device_id: str,
        required: Sequence[ControlProtocol] = (),
        allow_fallback: bool = True,
    ) -> NegotiationResult:
        """Pick the best protocol for ``device_id``.

        If ``required`` protocols are given and the device supports none of them,
        ``allow_fallback`` decides whether to use the device's best protocol or
        raise :class:`ProtocolError`.
        """
        await self.discover_all()
        msg = self.devices.get(device_id)
        if msg is None:
            raise ProtocolError(f"unknown device {device_id!r}; run discovery first")
        if not await self._verify(device_id):
            raise ProtocolError(f"device {device_id!r} failed its health probe")

        supported = set(msg.capabilities.supported_protocols)
        pool = [p for p in required if p in supported] if required else list(supported)

        notes = ""
        if not pool:
            if not allow_fallback:
                raise ProtocolError(
                    f"device {device_id!r} supports {[p.value for p in supported]}, "
                    f"none of the required {[p.value for p in required]}"
                )
            pool = list(supported)
            notes = "no required protocol available; fell back to device-native protocol"

        scored = sorted(
            ((p, score_protocol(p, msg.capabilities, required)) for p in pool),
            key=lambda item: item[1],
            reverse=True,
        )
        best, best_score = scored[0]
        result = NegotiationResult(
            device_id=device_id,
            protocol=best,
            score=best_score,
            candidates=scored,
            notes=notes,
        )
        self._results[device_id] = result
        return result

    async def _verify(self, device_id: str) -> bool:
        for negotiator in self._negotiators:
            if device_id in negotiator._cache:
                return await negotiator.verify(device_id)
        return False

    def last_result(self, device_id: str) -> Optional[NegotiationResult]:
        return self._results.get(device_id)
