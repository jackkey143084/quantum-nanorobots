"""Always-available NumPy simulator adapter.

Acts as the reference device for tests, CI and demos: no vendor SDK, no network,
deterministic given a seed. Also the semantic baseline other adapters are
validated against.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List

import numpy as np

from qnrs.adapters.base import AdapterError, CircuitProgram, ExecutionResult, QuantumBackendAdapter
from qnrs.core.capabilities import (
    ControlProtocol,
    DiscoveryMessage,
    QuantumBackendType,
    QuantumDeviceCapabilities,
)
from qnrs.core.pulses import PulseSequence
from qnrs.sim.statevector import StateVectorSimulator


class NumpySimulatorAdapter(QuantumBackendAdapter):
    """Dependency-free state-vector backend."""

    name = "numpy"

    def __init__(self, bus_key: str = "", num_qubits: int = 16) -> None:
        super().__init__(bus_key=bus_key)
        self._num_qubits = num_qubits
        self._caps = QuantumDeviceCapabilities(
            backend_type=QuantumBackendType.SIMULATOR,
            num_qubits=num_qubits,
            native_gates=["H", "X", "Y", "Z", "S", "T", "RX", "RY", "RZ", "CNOT", "CZ", "SWAP"],
            supported_protocols=[
                ControlProtocol.GATE_SEQUENCE,
                ControlProtocol.PULSE_SEQUENCE,
                ControlProtocol.MEASUREMENT_FEEDBACK,
            ],
            max_circuit_depth=10_000,
            max_shots=10_000_000,
            t1_us=0.0,
            t2_us=0.0,
            gate_time_1q_ns=20.0,
            gate_time_2q_ns=120.0,
            single_qubit_fidelity=1.0,
            two_qubit_fidelity=1.0,
            readout_fidelity=1.0,
            supports_mid_circuit_measurement=True,
            metadata={"engine": "qnrs.sim.statevector", "ideal": True},
        )

    async def discover(self) -> List[DiscoveryMessage]:
        message = self._advertise("numpy:simulator", "QNRS NumPy state-vector simulator", self._caps)
        return self._remember([message])

    async def verify(self, device_id: str) -> bool:
        return device_id == "numpy:simulator"

    async def capabilities(self, device_id: str) -> QuantumDeviceCapabilities:
        if device_id != "numpy:simulator":
            raise AdapterError(f"unknown NumPy device {device_id!r}")
        return self._caps

    async def execute_circuit(self, device_id: str, program: CircuitProgram) -> ExecutionResult:
        start = time.perf_counter()
        job_id = f"np-{uuid.uuid4().hex[:12]}"
        if device_id != "numpy:simulator":
            raise AdapterError(f"unknown NumPy device {device_id!r}")
        if program.num_qubits > self._num_qubits:
            raise AdapterError(
                f"circuit needs {program.num_qubits} qubits but simulator exposes {self._num_qubits}"
            )
        try:
            sim = StateVectorSimulator(program.num_qubits)
            for op in program.operations:
                sim.apply(op.name, op.qubits, op.params)
            counts = sim.sample_counts(program.shots, seed=program.seed)
            return ExecutionResult(
                job_id=job_id,
                device_id=device_id,
                adapter=self.name,
                status="completed",
                counts=counts,
                statevector=sim.state.tolist(),
                probabilities=sim.probabilities().tolist(),
                shots=program.shots,
                duration_us=(time.perf_counter() - start) * 1e6,
                engine="qnrs.sim.statevector",
                metadata={"depth": program.depth, "gate_counts": program.gate_counts},
            )
        except Exception as exc:
            return ExecutionResult(
                job_id=job_id,
                device_id=device_id,
                adapter=self.name,
                status="failed",
                shots=program.shots,
                duration_us=(time.perf_counter() - start) * 1e6,
                engine="qnrs.sim.statevector",
                error=f"{type(exc).__name__}: {exc}",
            )

    async def execute_pulse(self, device_id: str, sequence: PulseSequence) -> ExecutionResult:
        """Simulated pulse execution: verify the schedule and echo the program."""
        start = time.perf_counter()
        schedule = sequence.schedule()
        return ExecutionResult(
            job_id=f"np-pulse-{uuid.uuid4().hex[:8]}",
            device_id=device_id,
            adapter=self.name,
            status="completed",
            shots=sequence.repetitions,
            duration_us=(time.perf_counter() - start) * 1e6,
            engine="qnrs.pulse-ir",
            readout=[
                {"channel": entry["pulse"].target or "default", "start_ns": entry["start_ns"]}
                for entry in schedule
            ],
            metadata={"total_duration_ns": sequence.compute_total_duration()},
        )

    async def read_state(self, device_id: str) -> Dict[str, Any]:
        return {"device_id": device_id, "ideal": True, "queue_depth": 0}
