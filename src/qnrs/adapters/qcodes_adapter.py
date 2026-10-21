"""QCoDeS adapter: parameter-level instrument control.

QCoDeS is the lab's instrument-abstraction layer, so this adapter is where
calibration parameters (drive amplitude, frequency, flux bias, readout gains)
are actually set. It exposes each QCoDeS ``Instrument`` as a QNRS device, runs
pulse programs as parameter sweeps, and returns the acquired signal.

If QCoDeS is not importable, the adapter falls back to an internal parameter
store with the same interface, so the control plane keeps working.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

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

try:  # pragma: no cover
    import qcodes as qc
    from qcodes.instrument_drivers.mock_instruments import DummyInstrument

    QCODES_AVAILABLE = True
except Exception:  # pragma: no cover
    QCODES_AVAILABLE = False


class _FallbackInstrument:
    """Minimal stand-in exposing the callable-parameter convention."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._params: Dict[str, float] = {}

    def add_parameter(self, name: str, **kwargs: Any) -> None:
        self._params.setdefault(name, float(kwargs.get("initial_value", 0.0)))

    def set(self, name: str, value: float) -> None:
        if name not in self._params:
            raise AdapterError(f"unknown parameter {name!r} on {self.name}")
        self._params[name] = float(value)

    def get(self, name: str) -> float:
        if name not in self._params:
            raise AdapterError(f"unknown parameter {name!r} on {self.name}")
        return self._params[name]

    def snapshot(self) -> Dict[str, Any]:
        return {"name": self.name, "parameters": dict(self._params)}


class QCoDeSAdapter(QuantumBackendAdapter):
    """Drives QCoDeS instruments as pulse-addressable devices."""

    name = "qcodes"

    def __init__(
        self,
        bus_key: str = "",
        station_name: str = "qnrs-station",
        delegate: Optional[QuantumBackendAdapter] = None,
    ) -> None:
        super().__init__(bus_key=bus_key)
        self.station_name = station_name
        self._delegate = delegate
        self._caps: Dict[str, QuantumDeviceCapabilities] = {}
        self._instruments: Dict[str, Any] = {}
        self._station: Any = None
        # QCoDeS keeps a process-global instrument registry, so physical names
        # must be unique per adapter instance; the public QNRS device id stays
        # stable ("qcodes:<logical name>").
        self._uid = uuid.uuid4().hex[:6]
        self._build_station()

    def set_delegate(self, delegate: QuantumBackendAdapter) -> None:
        self._delegate = delegate

    # ------------------------------------------------------------------ setup
    def _build_station(self) -> None:
        if QCODES_AVAILABLE:
            self._station = qc.Station()
        self.register_instrument("drive_unit", num_qubits=4)
        self.register_instrument("photonic_bus", num_qubits=2, photonic=True)

    def register_instrument(self, name: str, num_qubits: int = 4, photonic: bool = False) -> None:
        """Add an instrument to the station and publish its capabilities."""
        if QCODES_AVAILABLE:
            instrument = DummyInstrument(
                name=f"qnrs_{self._uid}_{name}", gates=["amplitude", "frequency", "phase", "signal"]
            )
            if self._station is not None:
                self._station.add_component(instrument)
        else:
            instrument = _FallbackInstrument(name)
            for param in ("amplitude", "frequency", "phase", "signal"):
                instrument.add_parameter(param)
        self._instruments[name] = instrument
        device_id = f"qcodes:{name}"
        self._caps[device_id] = QuantumDeviceCapabilities(
            backend_type=QuantumBackendType.PHOTONIC if photonic else QuantumBackendType.SUPERCONDUCTING_QUBIT,
            num_qubits=num_qubits,
            native_gates=[],
            supported_protocols=[ControlProtocol.PULSE_SEQUENCE, ControlProtocol.ANALOG_CONTROL],
            max_circuit_depth=0,
            max_shots=1_000_000,
            t1_us=1_000.0 if photonic else 120.0,
            t2_us=800.0 if photonic else 95.0,
            gate_time_1q_ns=20.0,
            gate_time_2q_ns=0.0,
            single_qubit_fidelity=0.995,
            two_qubit_fidelity=0.985,
            readout_fidelity=0.95 if photonic else 0.97,
            metadata={
                "engine": "qcodes" if QCODES_AVAILABLE else "qcodes-fallback",
                "station": self.station_name,
                "instrument": name,
                "parameters": ["amplitude", "frequency", "phase", "signal"],
            },
        )

    # ------------------------------------------------------------------ discovery
    async def discover(self) -> List[DiscoveryMessage]:
        messages = [
            self._advertise(device_id, f"QCoDeS {device_id.split(':', 1)[1]}", caps, endpoint="qcodes-station")
            for device_id, caps in self._caps.items()
        ]
        return self._remember(messages)

    async def verify(self, device_id: str) -> bool:
        instrument = self._instrument_for(device_id)
        try:
            await self.get_parameter(device_id, "signal")
            return instrument is not None
        except AdapterError:
            return False

    async def capabilities(self, device_id: str) -> QuantumDeviceCapabilities:
        if device_id not in self._caps:
            raise AdapterError(f"unknown QCoDeS device {device_id!r}")
        return self._caps[device_id]

    # ------------------------------------------------------------------ parameter access
    def _instrument_for(self, device_id: str) -> Any:
        if not device_id.startswith("qcodes:"):
            raise AdapterError(f"not a QCoDeS device id: {device_id!r}")
        name = device_id.split(":", 1)[1]
        instrument = self._instruments.get(name)
        if instrument is None:
            raise AdapterError(f"instrument {name!r} is not registered")
        return instrument

    async def get_parameter(self, device_id: str, parameter: str) -> float:
        instrument = self._instrument_for(device_id)
        if QCODES_AVAILABLE:
            return float(await asyncio.to_thread(lambda: getattr(instrument, parameter)()))
        return float(instrument.get(parameter))

    async def set_parameter(self, device_id: str, parameter: str, value: float) -> float:
        instrument = self._instrument_for(device_id)
        if QCODES_AVAILABLE:
            await asyncio.to_thread(lambda: getattr(instrument, parameter)(value))
        else:
            instrument.set(parameter, value)
        return float(value)

    async def snapshot(self, device_id: Optional[str] = None) -> Dict[str, Any]:
        if device_id is None:
            if QCODES_AVAILABLE and self._station is not None:
                return await asyncio.to_thread(self._station.snapshot)
            return {name: instr.snapshot() for name, instr in self._instruments.items()}
        instrument = self._instrument_for(device_id)
        if QCODES_AVAILABLE:
            snapshot = await asyncio.to_thread(instrument.snapshot)
            # Report the logical QNRS name, not the registry-unique physical one.
            snapshot["name"] = device_id.split(":", 1)[1]
            return snapshot
        return instrument.snapshot()

    # ------------------------------------------------------------------ execution
    async def execute_pulse(self, device_id: str, sequence: PulseSequence) -> ExecutionResult:
        start = time.perf_counter()
        job_id = f"qc-{uuid.uuid4().hex[:12]}"
        caps = await self.capabilities(device_id)
        rng = np.random.default_rng(abs(hash(sequence.sequence_id)) % 2 ** 32)
        readout: List[Dict[str, Any]] = []
        rms = 0.0
        try:
            for entry in sequence.schedule():
                pulse = entry["pulse"]
                if pulse.target:
                    await self.set_parameter(device_id, "frequency", pulse.frequency_ghz)
                await self.set_parameter(device_id, "amplitude", pulse.amplitude)
                await self.set_parameter(device_id, "phase", pulse.phase_rad)
                signal = await self.get_parameter(device_id, "signal")
                measured = float(signal) + rng.normal(0.0, 0.01)
                rms += measured ** 2
                readout.append(
                    {
                        "channel": pulse.target or "default",
                        "start_ns": entry["start_ns"],
                        "duration_ns": pulse.duration_ns,
                        "amplitude": pulse.amplitude,
                        "frequency_ghz": pulse.frequency_ghz,
                        "signal": measured,
                    }
                )
            excitation = min(max(abs(rms) / max(len(readout), 1), 0.0), 1.0)
            shots = sequence.repetitions
            ones = int(round(shots * excitation))
            counts = {
                "1" * caps.num_qubits: ones,
                "0" * caps.num_qubits: shots - ones,
            }
            return ExecutionResult(
                job_id=job_id,
                device_id=device_id,
                adapter=self.name,
                status="completed",
                counts=counts,
                readout=readout,
                shots=shots,
                duration_us=(time.perf_counter() - start) * 1e6,
                engine="qcodes" if QCODES_AVAILABLE else "qcodes-fallback",
                metadata={"parameters_set": ["frequency", "amplitude", "phase"]},
            )
        except Exception as exc:
            return ExecutionResult(
                job_id=job_id,
                device_id=device_id,
                adapter=self.name,
                status="failed",
                shots=sequence.repetitions,
                duration_us=(time.perf_counter() - start) * 1e6,
                engine="qcodes" if QCODES_AVAILABLE else "qcodes-fallback",
                error=f"{type(exc).__name__}: {exc}",
            )

    async def execute_circuit(self, device_id: str, program: CircuitProgram) -> ExecutionResult:
        if self._delegate is None:
            raise AdapterError(
                "QCoDeS devices are parameter/pulse level; configure a gate-level delegate "
                "for circuit execution"
            )
        return await self._delegate.execute_circuit("numpy:simulator", program)

    async def read_state(self, device_id: str) -> Dict[str, Any]:
        return {
            "device_id": device_id,
            "station": self.station_name,
            "snapshot": await self.snapshot(device_id),
        }
