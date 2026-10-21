"""ROS 2 bridge logic, decoupled from ``rclpy`` so it is unit-testable.

The rclpy node in ``ros2_ws/src/qnrs_ros2`` is a thin shell: it converts these
plain dicts into ROS messages/services. Keeping the logic here means the bridge
can be tested (and used) with no ROS installation.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from qnrs.adapters.base import AdapterError, CircuitProgram
from qnrs.core.capabilities import ControlProtocol, ProtocolError
from qnrs.core.pulses import PulseSequence, PulseValidationError
from qnrs.runtime import QuantumRuntime


@dataclass
class BridgeStatus:
    success: bool
    message: str = ""


class QuantumBridge:
    """Transport-agnostic bridge between ROS 2 and the QNRS runtime."""

    def __init__(self, runtime: Optional[QuantumRuntime] = None) -> None:
        self.runtime = runtime or QuantumRuntime()
        self._started_at = time.time()

    # ------------------------------------------------------------------ topics
    async def state_message(self, device_id: str) -> Dict[str, Any]:
        """Payload for ``qnrs_ros2/QuantumState``."""
        await self.runtime.discover(refresh=False)
        manager = self.runtime._state_managers.get(device_id)
        if manager is None:
            raise ProtocolError(f"unknown device {device_id!r}")
        status = manager.coherence_status()
        caps = (await self.runtime.describe(device_id))["capabilities"]
        return {
            "device_id": device_id,
            "fidelity": float(status.get("fidelity", 1.0)),
            "predicted_fidelity": float(status.get("predicted_fidelity", 1.0)),
            "coherence_level": str(status.get("coherence_level", "unknown")),
            "age_us": float(status.get("age_us", 0.0)),
            "t1_us": float(caps.get("t1_us", 0.0)),
            "t2_us": float(caps.get("t2_us", 0.0)),
            "num_qubits": int(caps.get("num_qubits", 0)),
            "stamp": time.time(),
        }

    async def capabilities_message(self, device_id: str) -> Dict[str, Any]:
        """Payload for ``qnrs_ros2/DeviceCapabilities``."""
        await self.runtime.discover(refresh=False)
        description = await self.runtime.describe(device_id)
        caps = description["capabilities"]
        return {
            "device_id": device_id,
            "device_name": description["device_name"],
            "backend_type": caps["backend_type"],
            "num_qubits": int(caps["num_qubits"]),
            "native_gates": list(caps.get("native_gates", [])),
            "protocols": list(caps.get("supported_protocols", [])),
            "t1_us": float(caps.get("t1_us", 0.0)),
            "t2_us": float(caps.get("t2_us", 0.0)),
            "two_qubit_fidelity": float(caps.get("two_qubit_fidelity", 1.0)),
            "readout_fidelity": float(caps.get("readout_fidelity", 1.0)),
        }

    def telemetry_json(self) -> str:
        return json.dumps(self.runtime.telemetry(), default=str)

    def device_ids(self) -> List[str]:
        return sorted(self.runtime.devices)

    # ------------------------------------------------------------------ services
    async def negotiate(self, device_id: str, required: Sequence[str] = (),
                        allow_fallback: bool = True) -> Dict[str, Any]:
        """Handler for ``NegotiateProtocol.srv``."""
        try:
            protocols = [ControlProtocol(name) for name in required]
        except ValueError as exc:
            return {"success": False, "protocol": "", "score": 0.0, "message": f"bad protocol: {exc}"}
        try:
            result = await self.runtime.negotiate(device_id, protocols, allow_fallback=allow_fallback)
        except ProtocolError as exc:
            return {"success": False, "protocol": "", "score": 0.0, "message": str(exc)}
        return {
            "success": True,
            "protocol": result.protocol.value,
            "score": result.score,
            "message": result.notes or "ok",
        }

    async def execute_pulse(self, sequence: Dict[str, Any],
                            device_id: Optional[str] = None) -> Dict[str, Any]:
        """Handler for ``ExecutePulseSequence.srv``."""
        try:
            parsed = PulseSequence.from_dict(sequence)
        except (PulseValidationError, KeyError, TypeError) as exc:
            return {"success": False, "job_id": "", "device_id": device_id or "",
                    "counts_json": "{}", "excitation": 0.0, "message": f"invalid sequence: {exc}"}
        try:
            record = await self.runtime.submit_pulse(parsed, device_id=device_id)
        except (ProtocolError, AdapterError) as exc:
            return {"success": False, "job_id": "", "device_id": device_id or "",
                    "counts_json": "{}", "excitation": 0.0, "message": str(exc)}
        result = record.result
        return {
            "success": record.status == "completed",
            "job_id": record.job_id,
            "device_id": record.device_id,
            "counts_json": json.dumps(result.counts if result else {}),
            "excitation": float(result.metadata.get("excitation", 0.0)) if result else 0.0,
            "message": record.error or "ok",
        }

    async def submit_circuit(self, program: Dict[str, Any],
                             device_id: Optional[str] = None) -> Dict[str, Any]:
        """Handler for ``SubmitCircuit.srv``."""
        try:
            parsed = CircuitProgram.from_dict(program)
        except (AdapterError, KeyError, TypeError) as exc:
            return {"success": False, "job_id": "", "device_id": device_id or "",
                    "counts_json": "{}", "fidelity": 0.0, "message": f"invalid program: {exc}"}
        try:
            record = await self.runtime.submit_circuit(parsed, device_id=device_id)
        except (ProtocolError, AdapterError) as exc:
            return {"success": False, "job_id": "", "device_id": device_id or "",
                    "counts_json": "{}", "fidelity": 0.0, "message": str(exc)}
        result = record.result
        return {
            "success": record.status == "completed",
            "job_id": record.job_id,
            "device_id": record.device_id,
            "counts_json": json.dumps(result.counts if result else {}),
            "fidelity": float(result.fidelity or 0.0) if result else 0.0,
            "message": record.error or "ok",
        }

    async def close(self) -> None:
        await self.runtime.close()
