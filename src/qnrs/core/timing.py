"""Real-time budget arithmetic for the control plane.

Quantum control loop deadlines live in the microsecond range; this module makes
the budget explicit so the gateway can reject work that cannot be serviced, and
so the ROS 2 bridge can report overruns.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional


class DeadlineMiss(RuntimeError):
    """Raised when a control step overruns its allocated budget."""


@dataclass
class TimingBudget:
    """Breakdown of one control step's microsecond budget."""

    name: str
    total_us: float
    allocation: Dict[str, float] = field(default_factory=dict)
    _started_ns: Optional[int] = None
    _elapsed: Dict[str, float] = field(default_factory=dict)
    overruns: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.total_us <= 0:
            raise ValueError("total_us must be > 0")
        used = sum(self.allocation.values())
        if used > self.total_us + 1e-9:
            raise ValueError(f"allocations ({used}us) exceed budget ({self.total_us}us)")

    @property
    def headroom_us(self) -> float:
        return self.total_us - sum(self._elapsed.values())

    def start(self) -> "TimingBudget":
        self._started_ns = time.perf_counter_ns()
        return self

    @contextmanager
    def stage(self, label: str) -> Iterator[None]:
        """Time a named stage and record an overrun if it blows its allocation."""
        began = time.perf_counter_ns()
        try:
            yield
        finally:
            elapsed_us = (time.perf_counter_ns() - began) / 1000.0
            self._elapsed[label] = self._elapsed.get(label, 0.0) + elapsed_us
            allowed = self.allocation.get(label)
            if allowed is not None and self._elapsed[label] > allowed:
                self.overruns.append(label)

    def elapsed_us(self) -> float:
        return sum(self._elapsed.values())

    def check(self) -> None:
        """Raise if the budget was exceeded."""
        elapsed = self.elapsed_us()
        if elapsed > self.total_us:
            raise DeadlineMiss(
                f"step {self.name!r} used {elapsed:.2f}us of {self.total_us:.2f}us"
            )

    def report(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "budget_us": self.total_us,
            "elapsed_us": round(self.elapsed_us(), 3),
            "stages": {k: round(v, 3) for k, v in self._elapsed.items()},
            "overruns": list(self.overruns),
            "headroom_us": round(self.headroom_us, 3),
        }
