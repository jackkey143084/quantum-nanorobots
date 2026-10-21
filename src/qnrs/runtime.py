"""Runtime facade: builds the adapter set, routes jobs, keeps a job store.

This is the single object the gateway and the ROS 2 node talk to. It owns:

* the negotiation engine (discovery + protocol choice),
* a device -> adapter routing table,
* a device-selection planner (coherence fit, depth, fidelity budget),
* an in-memory job store with results and partial-failure records.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from qnrs.adapters.base import (
    AdapterError,
    CircuitProgram,
    ExecutionResult,
    QuantumBackendAdapter,
)
from qnrs.adapters.qcodes_adapter import QCoDeSAdapter
from qnrs.adapters.labber_adapter import LabberAdapter
from qnrs.adapters.pennylane_adapter import PennyLaneAdapter
from qnrs.adapters.qiskit_adapter import QiskitAdapter
from qnrs.adapters.silq_adapter import SilqAdapter
from qnrs.adapters.simulator_adapter import NumpySimulatorAdapter
from qnrs.core.capabilities import (
    ControlProtocol,
    DiscoveryMessage,
    ProtocolError,
    QuantumDeviceCapabilities,
)
from qnrs.core.discovery import NegotiationEngine, NegotiationResult
from qnrs.core.errors import ErrorPolicy, QuantumErrorHandler
from qnrs.core.pulses import PulseSequence
from qnrs.core.state import QuantumStateManager


@dataclass
class RuntimeConfig:
    """Feature flags so a deployment can run a strict subset of the stack."""

    bus_key: str = ""
    enable_qiskit: bool = True
    enable_pennylane: bool = True
    enable_labber: bool = True
    enable_qcodes: bool = True
    enable_silq: bool = True
    mock_labber: bool = True
    simulator_qubits: int = 16
    error_policy: ErrorPolicy = field(default_factory=ErrorPolicy)


@dataclass
class JobRecord:
    job_id: str
    device_id: str
    kind: str            # "circuit" | "pulse" | "silq" | "variational"
    status: str
    submitted_at: float
    completed_at: Optional[float] = None
    result: Optional[ExecutionResult] = None
    request: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self, include_statevector: bool = True) -> Dict[str, Any]:
        payload = {
            "job_id": self.job_id,
            "device_id": self.device_id,
            "kind": self.kind,
            "status": self.status,
            "submitted_at": self.submitted_at,
            "completed_at": self.completed_at,
            "request": self.request,
            "error": self.error,
            "result": self.result.to_dict(include_statevector=include_statevector) if self.result else None,
        }
        return payload


class QuantumRuntime:
    """Composes adapters into one addressable control plane."""

    def __init__(self, config: Optional[RuntimeConfig] = None) -> None:
        self.config = config or RuntimeConfig()
        self.engine = NegotiationEngine(bus_key=self.config.bus_key)
        self.adapters: Dict[str, QuantumBackendAdapter] = {}
        self._device_adapter: Dict[str, str] = {}
        self._devices: Dict[str, DiscoveryMessage] = {}
        self._jobs: Dict[str, JobRecord] = {}
        self._state_managers: Dict[str, QuantumStateManager] = {}
        self._error_handlers: Dict[str, QuantumErrorHandler] = {}
        self._started_at = time.time()
        self._build_adapters()

    # ------------------------------------------------------------------ wiring
    def _build_adapters(self) -> None:
        self._add(NumpySimulatorAdapter(bus_key=self.config.bus_key, num_qubits=self.config.simulator_qubits))

        if self.config.enable_qiskit:
            try:
                self._add(QiskitAdapter(bus_key=self.config.bus_key))
            except AdapterError:
                pass

        if self.config.enable_pennylane:
            try:
                self._add(PennyLaneAdapter(bus_key=self.config.bus_key, wires=self.config.simulator_qubits))
            except AdapterError:
                pass

        if self.config.enable_silq:
            silq = SilqAdapter(bus_key=self.config.bus_key, delegate=self.adapters.get("numpy"))
            self._add(silq)

        if self.config.enable_labber:
            labber = LabberAdapter(bus_key=self.config.bus_key, use_mock=self.config.mock_labber)
            self._add(labber)

        if self.config.enable_qcodes:
            self._add(QCoDeSAdapter(bus_key=self.config.bus_key))

        # Labber and QCoDeS are pulse-level; give them a gate-level delegate so
        # circuit submissions to them degrade gracefully instead of failing.
        delegate = self.adapters.get("numpy")
        if delegate is not None:
            for name in ("labber", "qcodes"):
                adapter = self.adapters.get(name)
                if isinstance(adapter, (LabberAdapter, QCoDeSAdapter)):
                    adapter.set_delegate(delegate)

    def _add(self, adapter: QuantumBackendAdapter) -> None:
        self.adapters[adapter.name] = adapter
        self.engine.register(adapter)

    @property
    def started_at(self) -> float:
        return self._started_at

    # ------------------------------------------------------------------ discovery
    async def discover(self, refresh: bool = True) -> List[DiscoveryMessage]:
        if refresh:
            messages = await self.engine.discover_all()
        else:
            messages = list(self._devices.values())
        for message in messages:
            self._devices[message.device_id] = message
            self._device_adapter[message.device_id] = self._adapter_for_message(message)
            self._state_managers.setdefault(
                message.device_id,
                QuantumStateManager(
                    num_qubits=message.capabilities.num_qubits,
                    coherence_time_us=message.capabilities.t2_us or 100.0,
                ),
            )
            self._error_handlers.setdefault(
                message.device_id,
                QuantumErrorHandler(self._state_managers[message.device_id], self.config.error_policy),
            )
        return list(self._devices.values())

    def _adapter_for_message(self, message: DiscoveryMessage) -> str:
        for name, adapter in self.adapters.items():
            if message.device_id in adapter._cache:
                return name
        # Fall back to matching by device-id prefix.
        prefix = message.device_id.split(":", 1)[0]
        return prefix if prefix in self.adapters else "numpy"

    @property
    def devices(self) -> Dict[str, DiscoveryMessage]:
        return dict(self._devices)

    async def describe(self, device_id: str) -> Dict[str, Any]:
        message = self._devices.get(device_id)
        if message is None:
            await self.discover()
            message = self._devices.get(device_id)
        if message is None:
            raise ProtocolError(f"unknown device {device_id!r}")
        return {
            "device_id": message.device_id,
            "device_name": message.device_name,
            "endpoint": message.endpoint,
            "adapter": self._device_adapter.get(device_id, "unknown"),
            "capabilities": message.capabilities.to_dict(),
        }

    async def negotiate(
        self, device_id: str, required: Sequence[ControlProtocol] = (), allow_fallback: bool = True
    ) -> NegotiationResult:
        await self.discover(refresh=False)
        return await self.engine.negotiate(device_id, required=required, allow_fallback=allow_fallback)

    # ------------------------------------------------------------------ planning
    def plan(self, program: CircuitProgram, required: Sequence[ControlProtocol] = ()) -> Dict[str, Any]:
        """Rank devices that can run ``program`` and explain the choice.

        Score = fidelity proxy * coherence fit * headroom, so a plan is auditable
        rather than a black box.
        """
        candidates: List[Dict[str, Any]] = []
        for device_id, message in self._devices.items():
            caps = message.capabilities
            if caps.num_qubits < program.num_qubits:
                continue
            if required and not all(caps.supports(p) for p in required):
                continue
            if caps.max_circuit_depth and program.depth > caps.max_circuit_depth:
                continue
            duration = caps.estimated_duration_us(program.depth)
            window = caps.effective_coherence_us
            fits = caps.fits_coherence(program.depth)
            fidelity_proxy = caps.two_qubit_fidelity * caps.readout_fidelity
            coherence_score = 1.0 if window == float("inf") else min(1.0, window / max(duration, 1e-9))
            score = fidelity_proxy * (0.5 + 0.5 * coherence_score)
            candidates.append(
                {
                    "device_id": device_id,
                    "adapter": self._device_adapter.get(device_id, "unknown"),
                    "score": round(score, 6),
                    "estimated_duration_us": round(duration, 4),
                    "coherence_window_us": None if window == float("inf") else window,
                    "fits_coherence": fits,
                    "protocols": [p.value for p in caps.supported_protocols],
                }
            )
        candidates.sort(key=lambda c: c["score"], reverse=True)
        return {
            "requested_qubits": program.num_qubits,
            "requested_depth": program.depth,
            "recommended": candidates[0]["device_id"] if candidates else None,
            "candidates": candidates,
        }

    # ------------------------------------------------------------------ submission
    async def submit_circuit(self, program: CircuitProgram, device_id: Optional[str] = None) -> JobRecord:
        await self.discover()
        if not self._devices:
            raise ProtocolError("no devices discovered")
        if device_id is None:
            plan = self.plan(program)
            if not plan["recommended"]:
                raise ProtocolError(
                    f"no discovered device can host a {program.num_qubits}-qubit circuit of depth {program.depth}"
                )
            device_id = plan["recommended"]

        adapter = self._adapter_for(device_id)
        record = self._new_job("circuit", device_id, {"program": program.to_dict()})
        result = await adapter.execute_circuit(device_id, program)
        self._finish(record, result)
        return record

    async def submit_pulse(self, sequence: PulseSequence, device_id: Optional[str] = None) -> JobRecord:
        await self.discover()
        if device_id is None:
            device_id = self._first_pulse_device()
        if device_id is None:
            raise ProtocolError("no pulse-capable device discovered")
        adapter = self._adapter_for(device_id)
        caps = self._devices[device_id].capabilities
        if not any(caps.supports(p) for p in (ControlProtocol.PULSE_SEQUENCE, ControlProtocol.ANALOG_CONTROL)):
            raise ProtocolError(f"device {device_id!r} does not accept pulse programs")
        record = self._new_job("pulse", device_id, {"sequence": sequence.to_dict()})
        result = await adapter.execute_pulse(device_id, sequence)
        self._finish(record, result)
        return record

    def _first_pulse_device(self) -> Optional[str]:
        """Prefer an instrument driver that natively executes pulses.

        Labber/QCoDeS are the real pulse paths; simulators (and gate-level
        backends that only export a schedule) come after them.
        """
        candidates = [
            device_id
            for device_id, message in self._devices.items()
            if any(
                message.capabilities.supports(p)
                for p in (ControlProtocol.PULSE_SEQUENCE, ControlProtocol.ANALOG_CONTROL)
            )
        ]
        if not candidates:
            return None
        instrument = [
            d for d in candidates if self._device_adapter.get(d) in ("labber", "qcodes")
        ]
        if instrument:
            return instrument[0]
        real = [
            d
            for d in candidates
            if self._devices[d].capabilities.backend_type.value != "simulator"
        ]
        return (real or candidates)[0]

    def _adapter_for(self, device_id: str) -> QuantumBackendAdapter:
        name = self._device_adapter.get(device_id)
        if name is None:
            prefix = device_id.split(":", 1)[0]
            name = prefix if prefix in self.adapters else None
        if name is None or name not in self.adapters:
            raise ProtocolError(f"no adapter registered for device {device_id!r}")
        return self.adapters[name]

    def _new_job(self, kind: str, device_id: str, request: Dict[str, Any]) -> JobRecord:
        job_id = f"job-{uuid.uuid4().hex[:12]}"
        record = JobRecord(
            job_id=job_id,
            device_id=device_id,
            kind=kind,
            status="running",
            submitted_at=time.time(),
            request=request,
        )
        self._jobs[job_id] = record
        return record

    def _finish(self, record: JobRecord, result: ExecutionResult) -> None:
        record.result = result
        record.status = result.status
        record.error = result.error
        record.completed_at = time.time()

    # ------------------------------------------------------------------ jobs
    def get_job(self, job_id: str) -> Optional[JobRecord]:
        return self._jobs.get(job_id)

    def list_jobs(self, limit: int = 50) -> List[JobRecord]:
        return sorted(self._jobs.values(), key=lambda j: j.submitted_at, reverse=True)[:limit]

    # ------------------------------------------------------------------ telemetry
    def telemetry(self) -> Dict[str, Any]:
        return {
            "uptime_s": round(time.time() - self._started_at, 3),
            "adapters": sorted(self.adapters),
            "devices": len(self._devices),
            "jobs": {
                "total": len(self._jobs),
                "running": sum(1 for j in self._jobs.values() if j.status == "running"),
                "failed": sum(1 for j in self._jobs.values() if j.status == "failed"),
            },
            "coherence": {
                device_id: manager.coherence_status()
                for device_id, manager in self._state_managers.items()
            },
            "errors": {
                device_id: handler.statistics()
                for device_id, handler in self._error_handlers.items()
                if handler.log
            },
        }

    async def close(self) -> None:
        for adapter in self.adapters.values():
            try:
                await adapter.close()
            except Exception:
                continue
