"""Labber bridge protocol (JSON-RPC 2.0 over newline-delimited TCP).

Labber's own API is closed-source and ships with the instrument workstation, so
the runtime talks to it through a thin, documented bridge: a newline-delimited
JSON-RPC 2.0 service that a ~100-line script on the Labber host exposes. The
methods are:

===================================  ===========================================
``system.health``                    liveness, uptime, fridge temperature
``device.list``                      enumerate addressable drive/acquisition units
``device.describe``                  channels, sample rate, calibration
``pulse.execute``                    run an analog program, return readout
``instrument.set`` / ``instrument.get``  scalar parameter access
``mixer.tune``                       IQ mixer calibration
===================================  ===========================================

:class:`MockLabberServer` implements the same surface in-process so the control
plane (and CI) can exercise the full path with no hardware.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np


class LabberRpcError(RuntimeError):
    """Raised for transport failures or JSON-RPC error responses."""

    def __init__(self, message: str, code: Optional[int] = None, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.data = data


class LabberRpcClient:
    """Async JSON-RPC 2.0 client speaking newline-delimited JSON over TCP."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5590, timeout: float = 5.0) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._lock = asyncio.Lock()
        self._pending: Dict[str, asyncio.Future] = {}
        self._reader_task: Optional[asyncio.Task] = None
        self._request_id = 0

    async def connect(self) -> "LabberRpcClient":
        if self._writer is not None:
            return self
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port), timeout=self.timeout
            )
        except (OSError, asyncio.TimeoutError) as exc:
            raise LabberRpcError(f"cannot reach Labber bridge at {self.host}:{self.port}: {exc}") from exc
        self._reader_task = asyncio.create_task(self._read_loop())
        return self

    async def call(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        if self._writer is None:
            await self.connect()
        self._request_id += 1
        request_id = str(self._request_id)
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending[request_id] = future
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}
        async with self._lock:
            assert self._writer is not None
            self._writer.write((json.dumps(payload) + "\n").encode("utf-8"))
            await self._writer.drain()
        try:
            return await asyncio.wait_for(future, timeout=self.timeout)
        except asyncio.TimeoutError as exc:
            self._pending.pop(request_id, None)
            raise LabberRpcError(f"Labber bridge timed out on {method!r}") from exc

    async def _read_loop(self) -> None:
        assert self._reader is not None
        while True:
            try:
                line = await self._reader.readline()
            except (asyncio.CancelledError, ConnectionError):
                break
            if not line:
                break
            try:
                message = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            request_id = str(message.get("id"))
            future = self._pending.pop(request_id, None)
            if future is None or future.done():
                continue
            if "error" in message:
                error = message["error"]
                future.set_exception(
                    LabberRpcError(error.get("message", "unknown error"), error.get("code"), error.get("data"))
                )
            else:
                future.set_result(message.get("result"))

    async def close(self) -> None:
        if self._reader_task:
            self._reader_task.cancel()
        if self._writer is not None:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:
                pass
        self._writer = None
        self._reader = None

    async def __aenter__(self) -> "LabberRpcClient":
        return await self.connect()

    async def __aexit__(self, *_exc: Any) -> None:
        await self.close()


@dataclass
class _MockDevice:
    device_id: str
    name: str
    kind: str
    num_qubits: int
    channels: List[str]
    sample_rate_gsa: float = 2.4
    t1_us: float = 85.0
    t2_us: float = 60.0
    readout_fidelity: float = 0.96
    two_qubit_fidelity: float = 0.985
    seed: int = 7


class MockLabberServer:
    """In-process implementation of the Labber bridge surface.

    Drive model (documented, deterministic-ish): a pulse program excites a qubit
    proportional to pulse area, with a Gaussian readout distribution. Good enough
    to exercise scheduling, error handling and telemetry end-to-end.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 0,
                 devices: Optional[List[_MockDevice]] = None) -> None:
        self.host = host
        self.port = port
        self._server: Optional[asyncio.AbstractServer] = None
        self._started_at = time.time()
        self._params: Dict[str, float] = {}
        self.devices = devices or [
            _MockDevice(
                device_id="labber:nanorobot-drive-1",
                name="Nanorobot Drive Unit 1",
                kind="qubit_drive",
                num_qubits=6,
                channels=["drive-0", "drive-1", "drive-2", "flux-0", "readout"],
            ),
            _MockDevice(
                device_id="labber:photonic-bus",
                name="Photonic Bus Controller",
                kind="photonic_drive",
                num_qubits=4,
                channels=["laser-a", "laser-b", "phase-mod", "detector"],
                sample_rate_gsa=10.0,
                t1_us=1_000.0,
                t2_us=800.0,
                readout_fidelity=0.93,
                two_qubit_fidelity=0.97,
                seed=11,
            ),
        ]
        self._handlers: Dict[str, Callable[[Dict[str, Any]], Any]] = {
            "system.health": self._health,
            "device.list": self._device_list,
            "device.describe": self._device_describe,
            "pulse.execute": self._pulse_execute,
            "instrument.set": self._instrument_set,
            "instrument.get": self._instrument_get,
            "mixer.tune": self._mixer_tune,
        }

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> "MockLabberServer":
        self._server = await asyncio.start_server(self._handle_client, self.host, self.port)
        sockets = self._server.sockets or []
        if sockets:
            self.port = sockets[0].getsockname()[1]
        return self

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    @property
    def address(self) -> tuple:
        return self.host, self.port

    # ------------------------------------------------------------------ transport
    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                response = self._dispatch(line)
                writer.write((json.dumps(response) + "\n").encode("utf-8"))
                await writer.drain()
        except (ConnectionError, asyncio.CancelledError):
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    def _dispatch(self, line: bytes) -> Dict[str, Any]:
        try:
            request = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            return {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
        request_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params") or {}
        handler = self._handlers.get(method)
        if handler is None:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"method not found: {method}"},
            }
        try:
            result = handler(params)
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32000, "message": f"{type(exc).__name__}: {exc}"},
            }
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    # ------------------------------------------------------------------ methods
    def _health(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": "ok",
            "uptime_s": round(time.time() - self._started_at, 3),
            "temperature_mk": 12.5,
            "devices": len(self.devices),
        }

    def _device_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "devices": [
                {"id": d.device_id, "name": d.name, "kind": d.kind, "num_qubits": d.num_qubits}
                for d in self.devices
            ]
        }

    def _device_describe(self, params: Dict[str, Any]) -> Dict[str, Any]:
        device = self._find(params.get("device", ""))
        return {
            "id": device.device_id,
            "name": device.name,
            "kind": device.kind,
            "num_qubits": device.num_qubits,
            "channels": list(device.channels),
            "sample_rate_gsa": device.sample_rate_gsa,
            "t1_us": device.t1_us,
            "t2_us": device.t2_us,
            "readout_fidelity": device.readout_fidelity,
            "two_qubit_fidelity": device.two_qubit_fidelity,
        }

    def _pulse_execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        device = self._find(params.get("device", ""))
        program = params.get("program") or []
        repetitions = int(params.get("repetitions", 1))
        shots = int(params.get("shots", 1024))
        rng = np.random.default_rng(device.seed + len(program) + repetitions)

        # Pulse area -> excitation of the primary drive channel.
        total_area = 0.0
        peak_amplitude = 0.0
        for entry in program:
            iq = entry.get("iq") or []
            duration = float(entry.get("duration_ns", 0.0))
            if not iq:
                continue
            amplitudes = [math.hypot(float(pair[0]), float(pair[1])) for pair in iq]
            peak_amplitude = max(peak_amplitude, max(amplitudes))
            total_area += sum(amplitudes) * (duration / max(len(iq), 1))

        excitation = 0.5 * (1.0 - math.cos(min(total_area / 400.0, math.pi)))
        noise = rng.normal(0.0, 0.02)
        measured = min(max(excitation + noise, 0.0), 1.0)
        thresholded = int(measured > 0.5)
        p1 = measured
        counts = {
            "1" * device.num_qubits: int(round(shots * p1)),
            "0" * device.num_qubits: shots - int(round(shots * p1)),
        }

        readout = [
            {
                "channel": entry.get("channel", "default"),
                "start_ns": entry.get("start_ns", 0.0),
                "duration_ns": entry.get("duration_ns", 0.0),
                "excitation": round(measured, 6),
                "peak_amplitude": round(peak_amplitude, 6),
                "single_shot": thresholded,
            }
            for entry in program
        ]
        return {
            "status": "ok",
            "device": device.device_id,
            "shots": shots,
            "repetitions": repetitions,
            "counts": counts,
            "readout": readout,
            "excitation": round(measured, 6),
            "executed_at": time.time(),
        }

    def _instrument_set(self, params: Dict[str, Any]) -> Dict[str, Any]:
        name = params.get("name")
        if not name:
            raise ValueError("instrument.set requires 'name'")
        self._params[name] = float(params.get("value", 0.0))
        return {"name": name, "value": self._params[name]}

    def _instrument_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        name = params.get("name")
        if name not in self._params:
            raise KeyError(f"parameter {name!r} has not been set")
        return {"name": name, "value": self._params[name]}

    def _mixer_tune(self, params: Dict[str, Any]) -> Dict[str, Any]:
        channel = params.get("channel", "drive-0")
        rng = np.random.default_rng(hash(channel) % 2 ** 32)
        return {
            "channel": channel,
            "if_freq_mhz": round(float(rng.uniform(80, 160)), 4),
            "lo_freq_ghz": round(float(rng.uniform(4.5, 6.5)), 6),
            "phase_offset_rad": round(float(rng.uniform(-0.05, 0.05)), 6),
            "tuned_at": time.time(),
        }

    def _find(self, device_id: str) -> _MockDevice:
        for device in self.devices:
            if device.device_id == device_id or device.name == device_id:
                return device
        raise KeyError(f"unknown Labber device {device_id!r}")
