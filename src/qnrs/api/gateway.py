"""Unified HTTP + WebSocket gateway (the "API interface" layer).

One surface, many instruments: clients speak QNRS JSON, the gateway routes to
Qiskit / PennyLane / Labber / QCoDeS adapters behind the unified abstraction.

Run it::

    uvicorn qnrs.api.gateway:app --host 0.0.0.0 --port 8080
    # or
    qnrs serve --port 8080

Endpoints
---------
GET  /healthz                          liveness
GET  /v1/telemetry                     uptime, jobs, coherence, error stats
GET  /v1/devices                       discovery (refreshes adapter caches)
GET  /v1/devices/{device_id}           one device's capabilities
POST /v1/devices/{device_id}/negotiate choose a control protocol
POST /v1/plan                          rank devices for a circuit
POST /v1/jobs/circuit                  submit a circuit program
POST /v1/jobs/pulse                    submit a pulse sequence
POST /v1/jobs/silq                     compile + run Silq-subset source
POST /v1/jobs/variational              PennyLane variational optimization
GET  /v1/jobs                          recent jobs
GET  /v1/jobs/{job_id}                 one job (+ result)
WS   /v1/stream                        telemetry stream
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from qnrs.adapters.base import AdapterError, CircuitProgram
from qnrs.adapters.pennylane_adapter import PennyLaneAdapter, VariationalTemplate
from qnrs.adapters.silq_adapter import SilqSyntaxError
from qnrs.api.models import (
    CircuitRequest,
    DeviceSummary,
    NegotiateRequest,
    PlanRequest,
    PulseRequest,
    SilqRequest,
    VariationalRequest,
)
from qnrs.core.capabilities import ControlProtocol, ProtocolError
from qnrs.core.pulses import PulseSequence, PulseValidationError
from qnrs.core.timing import DeadlineMiss
from qnrs.runtime import QuantumRuntime, RuntimeConfig


def _parse_protocols(names: List[str]) -> List[ControlProtocol]:
    protocols: List[ControlProtocol] = []
    for name in names:
        try:
            protocols.append(ControlProtocol(name))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"unknown protocol {name!r}") from exc
    return protocols


def _summaries(runtime: QuantumRuntime) -> List[DeviceSummary]:
    summaries: List[DeviceSummary] = []
    for device_id, message in runtime.devices.items():
        caps = message.capabilities
        summaries.append(
            DeviceSummary(
                device_id=device_id,
                device_name=message.device_name,
                adapter=runtime._device_adapter.get(device_id, "unknown"),
                backend_type=caps.backend_type.value,
                num_qubits=caps.num_qubits,
                protocols=[p.value for p in caps.supported_protocols],
                t1_us=caps.t1_us,
                t2_us=caps.t2_us,
                two_qubit_fidelity=caps.two_qubit_fidelity,
                readout_fidelity=caps.readout_fidelity,
                ideal=bool(caps.metadata.get("ideal", False)),
            )
        )
    return summaries


def create_app(runtime: Optional[QuantumRuntime] = None) -> FastAPI:
    """Build the FastAPI app around a runtime (one is created if omitted)."""
    rt = runtime or QuantumRuntime(RuntimeConfig())

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            await rt.discover()
        except Exception:
            # Discovery failures are surfaced per-device, not fatal at boot.
            pass
        try:
            yield
        finally:
            await rt.close()

    app = FastAPI(
        title="QNRS — Quantum Nanorobot Runtime",
        version="0.2.0",
        summary="Unified API layer for quantum instruments (Qiskit, Silq, PennyLane, Labber, QCoDeS)",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.runtime = rt

    # ------------------------------------------------------------------ meta
    @app.get("/healthz", tags=["meta"])
    async def healthz() -> Dict[str, Any]:
        return {"status": "ok", "adapters": sorted(rt.adapters)}

    @app.get("/v1/telemetry", tags=["meta"])
    async def telemetry() -> Dict[str, Any]:
        return rt.telemetry()

    # ------------------------------------------------------------------ devices
    @app.get("/v1/devices", response_model=List[DeviceSummary], tags=["devices"])
    async def list_devices() -> List[DeviceSummary]:
        await rt.discover(refresh=True)
        return _summaries(rt)

    @app.get("/v1/devices/{device_id}", tags=["devices"])
    async def get_device(device_id: str) -> Dict[str, Any]:
        try:
            return await rt.describe(device_id)
        except ProtocolError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/v1/devices/{device_id}/negotiate", tags=["devices"])
    async def negotiate(device_id: str, request: NegotiateRequest) -> Dict[str, Any]:
        try:
            result = await rt.negotiate(
                device_id,
                required=_parse_protocols(request.required),
                allow_fallback=request.allow_fallback,
            )
        except ProtocolError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return result.to_dict()

    # ------------------------------------------------------------------ planning
    @app.post("/v1/plan", tags=["planning"])
    async def plan(request: PlanRequest) -> Dict[str, Any]:
        try:
            program = CircuitProgram.from_dict(request.program)
        except (AdapterError, KeyError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=f"invalid circuit program: {exc}") from exc
        await rt.discover(refresh=False)
        return rt.plan(program, required=_parse_protocols(request.required))

    # ------------------------------------------------------------------ jobs
    @app.post("/v1/jobs/circuit", tags=["jobs"])
    async def submit_circuit(request: CircuitRequest) -> Dict[str, Any]:
        try:
            program = CircuitProgram.from_dict(request.program)
        except (AdapterError, KeyError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=f"invalid circuit program: {exc}") from exc
        try:
            record = await rt.submit_circuit(program, device_id=request.device_id)
        except (ProtocolError, AdapterError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return record.to_dict()

    @app.post("/v1/jobs/pulse", tags=["jobs"])
    async def submit_pulse(request: PulseRequest) -> Dict[str, Any]:
        try:
            sequence = PulseSequence.from_dict(request.sequence)
        except (PulseValidationError, KeyError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=f"invalid pulse sequence: {exc}") from exc
        try:
            record = await rt.submit_pulse(sequence, device_id=request.device_id)
        except (ProtocolError, AdapterError, PulseValidationError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return record.to_dict()

    @app.post("/v1/jobs/silq", tags=["jobs"])
    async def submit_silq(request: SilqRequest) -> Dict[str, Any]:
        silq = rt.adapters.get("silq")
        if silq is None:
            raise HTTPException(status_code=503, detail="Silq front-end is not enabled")
        try:
            compiled = silq.compile(request.source, shots=request.shots)
        except SilqSyntaxError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            record = await rt.submit_circuit(compiled.program, device_id=request.device_id)
        except (ProtocolError, AdapterError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        payload = record.to_dict()
        payload["frontend"] = {
            "qubit_names": compiled.qubit_names,
            "measurements": compiled.measurements,
            "source_lines": compiled.source_lines,
        }
        return payload

    @app.post("/v1/jobs/variational", tags=["jobs"])
    async def submit_variational(request: VariationalRequest) -> Dict[str, Any]:
        adapter = next((a for a in rt.adapters.values() if isinstance(a, PennyLaneAdapter)), None)
        if adapter is None:
            raise HTTPException(status_code=503, detail="PennyLane adapter is not enabled")
        try:
            state = request.target_state.as_complex()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        num_qubits = request.num_qubits or (len(state).bit_length() - 1)
        if 2 ** num_qubits != len(state):
            raise HTTPException(
                status_code=422,
                detail=f"target_state length {len(state)} is not 2**{num_qubits}",
            )
        template = VariationalTemplate(num_qubits=num_qubits, layers=request.layers)
        try:
            result = await adapter.optimize(
                state, template=template, steps=request.steps,
                learning_rate=request.learning_rate, seed=request.seed,
            )
        except AdapterError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        record = rt._new_job("variational", adapter.DEVICE_ID, {"layers": request.layers})
        record.status = "completed"
        record.completed_at = asyncio.get_event_loop().time()
        payload = record.to_dict()
        payload["variational"] = result.to_dict()
        return payload

    @app.get("/v1/jobs", tags=["jobs"])
    async def list_jobs(limit: int = Query(50, ge=1, le=500)) -> List[Dict[str, Any]]:
        return [job.to_dict(include_statevector=False) for job in rt.list_jobs(limit=limit)]

    @app.get("/v1/jobs/{job_id}", tags=["jobs"])
    async def get_job(job_id: str, include_statevector: bool = True) -> Dict[str, Any]:
        record = rt.get_job(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"unknown job {job_id!r}")
        return record.to_dict(include_statevector=include_statevector)

    # ------------------------------------------------------------------ streaming
    @app.websocket("/v1/stream")
    async def stream(websocket: WebSocket) -> None:
        await websocket.accept()
        interval = 1.0
        try:
            while True:
                await websocket.send_json({"type": "telemetry", "payload": rt.telemetry()})
                try:
                    message = await asyncio.wait_for(websocket.receive_json(), timeout=interval)
                    if isinstance(message, dict) and "interval" in message:
                        interval = max(0.1, min(float(message["interval"]), 60.0))
                except asyncio.TimeoutError:
                    pass
        except WebSocketDisconnect:
            return
        except Exception:
            return

    return app


app = create_app()
