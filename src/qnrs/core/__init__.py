"""Core, dependency-light (numpy-only) abstractions shared by every adapter."""

from qnrs.core.capabilities import (
    ControlProtocol,
    DiscoveryMessage,
    QuantumBackendType,
    QuantumDeviceCapabilities,
    ProtocolError,
)
from qnrs.core.discovery import (
    NegotiationEngine,
    NegotiationResult,
    ProtocolNegotiator,
    score_protocol,
)
from qnrs.core.errors import (
    ErrorPolicy,
    QuantumErrorHandler,
    RecoveryStrategy,
)
from qnrs.core.pulses import (
    Envelope,
    PulseParameters,
    PulseSequence,
    PulseValidationError,
    TimingError,
)
from qnrs.core.state import (
    CoherenceLevel,
    DecoherenceModel,
    QuantumState,
    QuantumStateManager,
    StateType,
)
from qnrs.core.timing import DeadlineMiss, TimingBudget

__all__ = [
    "ControlProtocol",
    "DiscoveryMessage",
    "QuantumBackendType",
    "QuantumDeviceCapabilities",
    "ProtocolError",
    "NegotiationEngine",
    "NegotiationResult",
    "ProtocolNegotiator",
    "score_protocol",
    "ErrorPolicy",
    "QuantumErrorHandler",
    "RecoveryStrategy",
    "Envelope",
    "PulseParameters",
    "PulseSequence",
    "PulseValidationError",
    "TimingError",
    "CoherenceLevel",
    "DecoherenceModel",
    "QuantumState",
    "QuantumStateManager",
    "StateType",
    "DeadlineMiss",
    "TimingBudget",
]
