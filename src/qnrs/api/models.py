"""Pydantic request/response models for the QNRS gateway."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class CircuitRequest(BaseModel):
    program: Dict[str, Any] = Field(..., description="CircuitProgram payload")
    device_id: Optional[str] = Field(None, description="Pin execution to a device; omit to auto-plan")

    @field_validator("program")
    @classmethod
    def _needs_qubits(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        if "num_qubits" not in value:
            raise ValueError("program.num_qubits is required")
        return value


class PulseRequest(BaseModel):
    sequence: Dict[str, Any]
    device_id: Optional[str] = None


class SilqRequest(BaseModel):
    source: str
    shots: Optional[int] = None
    device_id: Optional[str] = None


class TargetState(BaseModel):
    real: List[float]
    imag: Optional[List[float]] = None

    def as_complex(self) -> List[complex]:
        if self.imag is None:
            return [complex(r, 0.0) for r in self.real]
        if len(self.imag) != len(self.real):
            raise ValueError("real and imag must have the same length")
        return [complex(r, i) for r, i in zip(self.real, self.imag)]


class VariationalRequest(BaseModel):
    target_state: TargetState
    num_qubits: Optional[int] = None
    layers: int = 2
    steps: int = 60
    learning_rate: float = 0.2
    seed: int = 0


class NegotiateRequest(BaseModel):
    required: List[str] = Field(default_factory=list)
    allow_fallback: bool = True


class PlanRequest(BaseModel):
    program: Dict[str, Any]
    required: List[str] = Field(default_factory=list)


class DeviceSummary(BaseModel):
    device_id: str
    device_name: str
    adapter: str
    backend_type: str
    num_qubits: int
    protocols: List[str]
    t1_us: float
    t2_us: float
    two_qubit_fidelity: float
    readout_fidelity: float
    ideal: bool
