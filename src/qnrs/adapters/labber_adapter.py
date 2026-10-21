"""Labber adapter: pulse-level control over the Labber bridge protocol.

Labber instruments are analog: they take pulse programs and return I/Q readout,
so this adapter serves the ``PULSE_SEQUENCE`` / ``ANALOG_CONTROL`` protocols and
delegates logical circuits to a gate-level backend.

If no bridge is reachable, ``use_mock=True`` starts an in-process
:class:`~qnrs.protocols.labber_rpc.MockLabberServer`, which lets the whole stack
run (and CI pass) with no instrument workstation present.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

from qnrs.adapters.base import (
    AdapterError,
    CircuitProgram,
    ExecutionResult,
    QuantumBackendAdapter,
)
from qnrs.core.capabilities import (
    ControlProtocol,
    DiscoveryMessage,
    QuantumBackendType,
    QuantumDeviceCapabilities,
)
from qnrs.core.pulses import PulseSequence
from qnrs.protocols.labber_rpc import LabberRpcClient, LabberRpcError, MockLabberServer

_KIND_TO_BACKEND = {
    "qubit_drive": QuantumBackendType.SUPERCONDUCTING_QUBIT,
    "photonic_drive": QuantumBackendType.PHOTONIC,
    "ion_drive": QuantumBackendType.TRAPPED_ION,
}


class LabberAdapter(QuantumBackendAdapter):
    """Speaks the Labber JSON-RPC bridge; returns shot-resolved readout."""

    name = "labber"

    def __init__(
        self,
        bus_key: str = "",
        host: str = "127.0.0.1",
        port: int = 5590,
        client: Optional[LabberRpcClient] = None,
        use_mock: bool = True,
        delegate: Optional[QuantumBackendAdapter] = None,
    ) -> None:
        super().__init__(bus_key=bus_key)
        self._client = client or LabberRpcClient(host=host, port=port)
        self._use_mock = use_mock and client is None
        self._mock: Optional[MockLabberServer] = None
        self._connected = False
        self._caps: Dict[str, QuantumDeviceCapabilities] = {}
        self._descriptions: Dict[str, Dict[str, Any]] = {}
        self._delegate = delegate

    def set_delegate(self, delegate: QuantumBackendAdapter) -> None:
        self._delegate = delegate

    # ------------------------------------------------------------------ lifecycle
    async def connect(self) -> "LabberAdapter":
        if self._connected:
            return self
        if self._use_mock:
            self._mock = await MockLabberServer(host=self._client.host, port=0).start()
            self._client.port = self._mock.port
        try:
            await self._client.connect()
            await self._client.call("system.health")
        except LabberRpcError:
            if self._use_mock:
                raise
            raise
        self._connected = True
        return self

    async def close(self) -> None:
        await self._client.close()
        if self._mock is not None:
            await self._mock.stop()
            self._mock = None
        self._connected = False

    # ------------------------------------------------------------------ discovery
    async def discover(self) -> List[DiscoveryMessage]:
        await self.connect()
        listing = await self._client.call("device.list")
        messages: List[DiscoveryMessage] = []
        for entry in listing.get("devices", []):
            device_id = entry["id"]
            description = await self._client.call("device.describe", {"device": device_id})
            self._descriptions[device_id] = description
            caps = self._capabilities_from_description(description)
            self._caps[device_id] = caps
            messages.append(
                self._advertise(device_id, description.get("name", device_id), caps, endpoint="labber-bridge")
            )
        return self._remember(messages)

    def _capabilities_from_description(self, description: Dict[str, Any]) -> QuantumDeviceCapabilities:
        kind = description.get("kind", "qubit_drive")
        t1 = float(description.get("t1_us", 0.0) or 0.0)
        t2 = float(description.get("t2_us", 0.0) or 0.0)
        return QuantumDeviceCapabilities(
            backend_type=_KIND_TO_BACKEND.get(kind, QuantumBackendType.HYBRID),
            num_qubits=int(description.get("num_qubits", 1)),
            native_gates=[],
            supported_protocols=[
                ControlProtocol.PULSE_SEQUENCE,
                ControlProtocol.ANALOG_CONTROL,
            ],
            max_circuit_depth=0,
            max_shots=1_000_000,
            t1_us=t1,
            t2_us=t2,
            gate_time_1q_ns=20.0,
            gate_time_2q_ns=0.0,
            single_qubit_fidelity=float(description.get("readout_fidelity", 1.0)),
            two_qubit_fidelity=float(description.get("two_qubit_fidelity", 1.0)),
            readout_fidelity=float(description.get("readout_fidelity", 1.0)),
            supports_mid_circuit_measurement=False,
            metadata={
                "engine": "labber-bridge",
                "kind": kind,
                "channels": list(description.get("channels", [])),
                "sample_rate_gsa": description.get("sample_rate_gsa"),
            },
        )

    async def verify(self, device_id: str) -> bool:
        try:
            await self.connect()
            health = await self._client.call("system.health")
            return health.get("status") == "ok" and device_id in self._caps
        except (LabberRpcError, OSError):
            return False

    async def capabilities(self, device_id: str) -> QuantumDeviceCapabilities:
        if device_id not in self._caps:
            await self.discover()
        if device_id not in self._caps:
            raise AdapterError(f"unknown Labber device {device_id!r}")
        return self._caps[device_id]

    # ------------------------------------------------------------------ execution
    async def execute_pulse(self, device_id: str, sequence: PulseSequence) -> ExecutionResult:
        start = time.perf_counter()
        job_id = f"lb-{uuid.uuid4().hex[:12]}"
        await self.connect()
        caps = await self.capabilities(device_id)
        program = sequence.analog_program()
        try:
            response = await self._client.call(
                "pulse.execute",
                {
                    "device": device_id,
                    "program": program,
                    "repetitions": sequence.repetitions,
                    "shots": sequence.repetitions,
                },
            )
        except LabberRpcError as exc:
            return ExecutionResult(
                job_id=job_id,
                device_id=device_id,
                adapter=self.name,
                status="failed",
                shots=sequence.repetitions,
                duration_us=(time.perf_counter() - start) * 1e6,
                engine="labber-bridge",
                error=str(exc),
            )
        return ExecutionResult(
            job_id=job_id,
            device_id=device_id,
            adapter=self.name,
            status="completed",
            counts={str(k): int(v) for k, v in (response.get("counts") or {}).items()},
            readout=list(response.get("readout", [])),
            shots=int(response.get("shots", sequence.repetitions)),
            duration_us=(time.perf_counter() - start) * 1e6,
            engine="labber-bridge",
            metadata={
                "excitation": response.get("excitation"),
                "repetitions": response.get("repetitions"),
                "sample_rate_gsa": caps.metadata.get("sample_rate_gsa"),
            },
        )

    async def execute_circuit(self, device_id: str, program: CircuitProgram) -> ExecutionResult:
        """Labber is pulse-level; run logical circuits on the delegate backend."""
        if self._delegate is None:
            raise AdapterError(
                "Labber device accepts pulse programs only; configure a gate-level delegate "
                "or submit to a pulse-capable device"
            )
        return await self._delegate.execute_circuit("numpy:simulator", program)

    # ------------------------------------------------------------------ instrument ops
    async def set_instrument(self, name: str, value: float) -> float:
        await self.connect()
        response = await self._client.call("instrument.set", {"name": name, "value": value})
        return float(response["value"])

    async def get_instrument(self, name: str) -> float:
        await self.connect()
        response = await self._client.call("instrument.get", {"name": name})
        return float(response["value"])

    async def tune_mixer(self, channel: str) -> Dict[str, Any]:
        await self.connect()
        return await self._client.call("mixer.tune", {"channel": channel})

    async def read_state(self, device_id: str) -> Dict[str, Any]:
        await self.connect()
        health = await self._client.call("system.health")
        return {
            "device_id": device_id,
            "bridge_status": health.get("status"),
            "temperature_mk": health.get("temperature_mk"),
            "uptime_s": health.get("uptime_s"),
            "description": self._descriptions.get(device_id, {}),
        }
