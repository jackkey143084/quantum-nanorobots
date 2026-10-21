"""Error taxonomy, recovery strategies and the quantum error handler.

Recovery is modelled as registered strategies keyed by error class, with a
policy that decides when to retry, restore a checkpoint, or give up loudly. The
default strategies use the surface-code model from
:class:`qnrs.core.state.DecoherenceModel` instead of an ad-hoc fidelity bump.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional

import numpy as np

from qnrs.core.state import CoherenceLevel, QuantumStateManager


class QuantumErrorKind(str, Enum):
    DECOHERENCE = "decoherence"
    MEASUREMENT = "measurement"
    GATE = "gate"
    READOUT = "readout"
    TRANSPORT = "transport"
    PROTOCOL = "protocol"


@dataclass
class ErrorRecord:
    kind: QuantumErrorKind
    detail: str
    context: Dict[str, Any] = field(default_factory=dict)
    at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "detail": self.detail,
            "context": self.context,
            "at": self.at.isoformat(),
            "resolved": self.resolved,
        }


@dataclass
class ErrorPolicy:
    """When to act on an error."""

    #: Retries per error class before escalating to a checkpoint restore.
    max_retries: int = 2
    #: Fidelity floor; below it the handler restores the last checkpoint.
    restore_below_fidelity: float = 0.5
    #: Measurement error probability used for symmetric readout correction.
    measurement_error_rate: float = 0.05
    #: Gate fidelity we consider nominal; deviations are logged.
    nominal_gate_fidelity: float = 0.99
    #: Code distance used when estimating correction gain.
    code_distance: int = 3


#: A strategy receives (handler, record) and returns True when it recovered.
RecoveryStrategy = Callable[["QuantumErrorHandler", ErrorRecord], Awaitable[bool]]


class QuantumErrorHandler:
    """Central error log + recovery dispatcher."""

    def __init__(self, state_manager: QuantumStateManager, policy: Optional[ErrorPolicy] = None) -> None:
        self.state_manager = state_manager
        self.policy = policy or ErrorPolicy()
        self.log: List[ErrorRecord] = []
        self._strategies: Dict[QuantumErrorKind, RecoveryStrategy] = {}
        self._attempts: Counter = Counter()
        self._register_defaults()

    # ------------------------------------------------------------------ registry
    def _register_defaults(self) -> None:
        self.register(QuantumErrorKind.DECOHERENCE, _recover_decoherence)
        self.register(QuantumErrorKind.MEASUREMENT, _recover_measurement)
        self.register(QuantumErrorKind.GATE, _recover_gate)
        self.register(QuantumErrorKind.READOUT, _recover_readout)
        self.register(QuantumErrorKind.TRANSPORT, _recover_transport)
        self.register(QuantumErrorKind.PROTOCOL, _recover_protocol)

    def register(self, kind: QuantumErrorKind, strategy: RecoveryStrategy) -> None:
        self._strategies[kind] = strategy

    # ------------------------------------------------------------------ handling
    async def handle(self, kind: QuantumErrorKind, detail: str = "", **context: Any) -> bool:
        record = ErrorRecord(kind=kind, detail=detail, context=context)
        self.log.append(record)
        self._attempts[kind] += 1

        strategy = self._strategies.get(kind)
        if strategy is None:
            return False

        recovered = await strategy(self, record)
        record.resolved = recovered
        if not recovered and self._attempts[kind] > self.policy.max_retries:
            await self._escalate(record)
        return recovered

    async def _escalate(self, record: ErrorRecord) -> None:
        """Last resort: restore the newest checkpoint, if any."""
        checkpoints = self.state_manager.list_checkpoints()
        if checkpoints:
            self.state_manager.restore(checkpoints[-1])
            record.context["escalated_to_checkpoint"] = checkpoints[-1]

    # ------------------------------------------------------------------ convenience
    async def handle_decoherence(self) -> bool:
        status = self.state_manager.coherence_status()
        if status.get("requires_correction"):
            return await self.handle(
                QuantumErrorKind.DECOHERENCE,
                f"coherence level {status['coherence_level']}",
                fidelity=status.get("fidelity"),
            )
        return False

    def statistics(self) -> Dict[str, Any]:
        by_kind = Counter(r.kind.value for r in self.log)
        resolved = sum(1 for r in self.log if r.resolved)
        return {
            "total": len(self.log),
            "resolved": resolved,
            "unresolved": len(self.log) - resolved,
            "by_kind": dict(by_kind),
            "attempts": {k.value: v for k, v in self._attempts.items()},
            "last": self.log[-1].to_dict() if self.log else None,
        }


# ---------------------------------------------------------------------- strategies
async def _recover_decoherence(handler: QuantumErrorHandler, record: ErrorRecord) -> bool:
    """Estimate the residual error and, if the window allows, restore a checkpoint."""
    model = handler.state_manager.model
    physical = 1.0 - float(record.context.get("fidelity", 1.0) or 1.0)
    if physical <= 0:
        return True
    logical = model.logical_error_rate(physical, handler.policy.code_distance) if physical < model.surface_code_threshold else physical
    record.context["estimated_logical_error"] = logical
    if logical < physical:
        # Correction helps: re-anchor on the newest checkpoint if one exists.
        checkpoints = handler.state_manager.list_checkpoints()
        if checkpoints:
            handler.state_manager.restore(checkpoints[-1])
            return True
        return True  # the estimate itself is the recovery signal for the planner
    return False


async def _recover_measurement(handler: QuantumErrorHandler, record: ErrorRecord) -> bool:
    """Maximum-likelihood bit flip for a symmetric readout error."""
    rate = handler.policy.measurement_error_rate
    result = record.context.get("result")
    if result in (0, 1) and rate > 0.5:
        record.context["corrected_result"] = 1 - result
        return True
    # Re-shoot with majority vote when the caller allows repetitions.
    repetitions = int(record.context.get("repetitions", 3) or 3)
    record.context["recommended_repetitions"] = max(repetitions, 3)
    return rate <= 0.5


async def _recover_gate(handler: QuantumErrorHandler, record: ErrorRecord) -> bool:
    fidelity = float(record.context.get("fidelity", handler.policy.nominal_gate_fidelity))
    deviation = abs(fidelity - handler.policy.nominal_gate_fidelity)
    record.context["deviation"] = deviation
    if fidelity >= handler.policy.nominal_gate_fidelity - 0.01:
        return True
    # Recalibrate the gate rather than silently retrying an unchanged pulse.
    record.context["action"] = "recalibrate"
    return False


async def _recover_readout(handler: QuantumErrorHandler, record: ErrorRecord) -> bool:
    """Invert a calibrated confusion matrix when one is supplied."""
    matrix = record.context.get("confusion_matrix")
    counts = record.context.get("counts")
    if matrix is None or counts is None:
        return False
    confusion = np.asarray(matrix, dtype=float)
    try:
        corrected = np.linalg.solve(confusion, np.asarray(counts, dtype=float))
    except np.linalg.LinAlgError:
        return False
    corrected = np.clip(corrected, 0.0, None)
    total = corrected.sum()
    record.context["corrected_counts"] = (corrected / total).tolist() if total else corrected.tolist()
    return True


async def _recover_transport(handler: QuantumErrorHandler, record: ErrorRecord) -> bool:
    """Transport errors are retried upstream; log and let the caller back off."""
    record.context["action"] = "retry_with_backoff"
    record.context["backoff_s"] = min(2.0 ** handler._attempts[QuantumErrorKind.TRANSPORT], 30.0)
    return False


async def _recover_protocol(handler: QuantumErrorHandler, record: ErrorRecord) -> bool:
    """A protocol mismatch must be renegotiated, never retried blindly."""
    record.context["action"] = "renegotiate"
    return False
