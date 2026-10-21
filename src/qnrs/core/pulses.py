"""Pulse-level intermediate representation.

This is the wire format between the control plane and Labber/QCoDeS-style
instrument servers: a validated list of IQ drive envelopes with explicit timing.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional


class PulseValidationError(ValueError):
    """Raised when a pulse or sequence is physically/temporally invalid."""


class TimingError(PulseValidationError):
    """Raised when a sequence violates timing constraints (overlap, gap)."""


class Envelope(str, Enum):
    GAUSSIAN = "gaussian"
    SQUARE = "square"
    RAMP = "ramp"
    DRAG = "drag"
    BLACKMAN = "blackman"


@dataclass
class PulseParameters:
    """One shaped drive pulse on one channel."""

    frequency_ghz: float
    amplitude: float
    duration_ns: float
    phase_rad: float = 0.0
    envelope: Envelope = Envelope.GAUSSIAN
    target: str = ""            # device channel / qubit label
    drag_beta: float = 0.0      # only used with Envelope.DRAG
    start_ns: Optional[float] = None  # absolute offset; None => append

    def __post_init__(self) -> None:
        if isinstance(self.envelope, str):
            self.envelope = Envelope(self.envelope)
        if self.duration_ns <= 0:
            raise PulseValidationError("duration_ns must be > 0")
        if not 0.0 <= self.amplitude <= 1.0:
            raise PulseValidationError("amplitude must be within [0, 1]")
        if self.frequency_ghz < 0:
            raise PulseValidationError("frequency_ghz must be >= 0")
        if self.envelope is Envelope.DRAG and self.drag_beta == 0.0:
            raise PulseValidationError("DRAG envelope requires a non-zero drag_beta")
        if self.start_ns is not None and self.start_ns < 0:
            raise PulseValidationError("start_ns must be >= 0")

    @property
    def samples(self) -> int:
        """Sample count at a nominal 1 GSa/s control clock."""
        return int(round(self.duration_ns))

    def envelope_value(self, t_ns: float) -> float:
        """Normalised envelope amplitude at ``t_ns`` (0 <= t <= duration)."""
        if t_ns < 0 or t_ns > self.duration_ns:
            return 0.0
        x = t_ns / self.duration_ns
        if self.envelope is Envelope.SQUARE:
            return 1.0
        if self.envelope is Envelope.RAMP:
            return x
        if self.envelope is Envelope.GAUSSIAN:
            sigma = 0.2
            centre = 0.5
            return math.exp(-((x - centre) ** 2) / (2 * sigma ** 2))
        if self.envelope is Envelope.BLACKMAN:
            return 0.42 - 0.5 * math.cos(2 * math.pi * x) + 0.08 * math.cos(4 * math.pi * x)
        if self.envelope is Envelope.DRAG:
            sigma = 0.2
            gauss = math.exp(-((x - 0.5) ** 2) / (2 * sigma ** 2))
            derivative = -((x - 0.5) / sigma ** 2) * gauss
            return gauss + self.drag_beta * derivative
        raise PulseValidationError(f"unsupported envelope {self.envelope}")

    def waveform(self, sample_rate_gsa: float = 1.0) -> List[complex]:
        """Sampled complex IQ waveform."""
        if sample_rate_gsa <= 0:
            raise PulseValidationError("sample_rate_gsa must be > 0")
        n = max(1, int(round(self.duration_ns * sample_rate_gsa)))
        dt = self.duration_ns / n
        omega = 2 * math.pi * self.frequency_ghz  # GHz * ns -> rad/us*1e-3; kept relative
        out: List[complex] = []
        for i in range(n):
            t = i * dt
            env = self.envelope_value(t)
            phase = omega * t * 1e-3 + self.phase_rad
            out.append(self.amplitude * env * complex(math.cos(phase), math.sin(phase)))
        return out

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["envelope"] = self.envelope.value
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PulseParameters":
        return cls(**dict(data))


@dataclass
class PulseSequence:
    """Ordered pulses plus a repetition count and optional inter-pulse gaps."""

    sequence_id: str
    pulses: List[PulseParameters] = field(default_factory=list)
    repetitions: int = 1
    gap_ns: float = 0.0
    required_channels: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.repetitions < 1:
            raise PulseValidationError("repetitions must be >= 1")
        if self.gap_ns < 0:
            raise PulseValidationError("gap_ns must be >= 0")

    def add(self, pulse: PulseParameters) -> "PulseSequence":
        self.pulses.append(pulse)
        if pulse.target and pulse.target not in self.required_channels:
            self.required_channels.append(pulse.target)
        return self

    def schedule(self) -> List[Dict[str, Any]]:
        """Resolve absolute start times and detect per-channel overlap.

        Pulses are appended in order; a pulse carrying ``start_ns`` is placed at
        that offset (and may overlap a neighbouring channel, but never itself).
        """
        schedule: List[Dict[str, Any]] = []
        cursor = 0.0
        occupied: Dict[str, List[tuple]] = {}
        for pulse in self.pulses:
            start = cursor if pulse.start_ns is None else pulse.start_ns
            end = start + pulse.duration_ns
            if pulse.target:
                for other_start, other_end in occupied.get(pulse.target, []):
                    if start < other_end and other_start < end:
                        raise TimingError(
                            f"pulse overlap on channel {pulse.target!r}: "
                            f"[{start:.3f}, {end:.3f}) intersects [{other_start:.3f}, {other_end:.3f})"
                        )
                occupied.setdefault(pulse.target, []).append((start, end))
            schedule.append({"pulse": pulse, "start_ns": start, "end_ns": end})
            cursor = end + self.gap_ns
        return schedule

    def compute_total_duration(self) -> float:
        """Total wall-clock duration in ns, including gaps and repetitions."""
        schedule = self.schedule()
        if not schedule:
            return 0.0
        single = max(entry["end_ns"] for entry in schedule)
        return (single + self.gap_ns) * self.repetitions

    def analog_program(self) -> List[Dict[str, Any]]:
        """Flatten to ``[{channel, start_ns, duration_ns, iq: [[I,Q],...]}]``."""
        program: List[Dict[str, Any]] = []
        for entry in self.schedule():
            pulse: PulseParameters = entry["pulse"]
            waveform = pulse.waveform()
            program.append({
                "channel": pulse.target or "default",
                "start_ns": entry["start_ns"],
                "duration_ns": pulse.duration_ns,
                "iq": [[c.real, c.imag] for c in waveform],
                "envelope": pulse.envelope.value,
                "frequency_ghz": pulse.frequency_ghz,
            })
        return program

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "repetitions": self.repetitions,
            "gap_ns": self.gap_ns,
            "required_channels": self.required_channels,
            "metadata": self.metadata,
            "pulses": [p.to_dict() for p in self.pulses],
            "total_duration_ns": self.compute_total_duration(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PulseSequence":
        payload = dict(data)
        pulses = [PulseParameters.from_dict(p) for p in payload.pop("pulses", [])]
        payload.pop("total_duration_ns", None)
        return cls(pulses=pulses, **payload)
