import math

import numpy as np
import pytest

from qnrs.adapters.base import CircuitProgram, GateOperation
from qnrs.core.pulses import Envelope, PulseParameters, PulseSequence


def bell_program(shots=2048, seed=5):
    return CircuitProgram(
        num_qubits=2,
        operations=[GateOperation("H", [0]), GateOperation("CNOT", [0, 1])],
        shots=shots,
        seed=seed,
    )


@pytest.mark.asyncio
async def test_numpy_adapter_runs_bell_circuit():
    from qnrs.adapters.simulator_adapter import NumpySimulatorAdapter

    adapter = NumpySimulatorAdapter()
    await adapter.discover()
    result = await adapter.execute_circuit("numpy:simulator", bell_program())
    assert result.status == "completed"
    assert sum(result.counts.values()) == 2048
    assert set(result.counts) <= {"00", "11"}
    # statevector should be the Bell state
    state = np.asarray(result.statevector)
    assert abs(state[0]) == pytest.approx(1 / math.sqrt(2), abs=1e-9)
    assert abs(state[3]) == pytest.approx(1 / math.sqrt(2), abs=1e-9)


@pytest.mark.asyncio
async def test_numpy_adapter_depth_and_gate_counts():
    from qnrs.adapters.simulator_adapter import NumpySimulatorAdapter

    adapter = NumpySimulatorAdapter()
    program = CircuitProgram(
        num_qubits=2,
        operations=[
            GateOperation("H", [0]),
            GateOperation("X", [0]),
            GateOperation("H", [1]),
            GateOperation("CNOT", [0, 1]),
        ],
    )
    result = await adapter.execute_circuit("numpy:simulator", program)
    assert result.metadata["depth"] == 3
    assert result.metadata["gate_counts"]["H"] == 2


@pytest.mark.asyncio
async def test_numpy_adapter_rejects_out_of_range_qubits():
    from qnrs.adapters.base import AdapterError
    from qnrs.adapters.simulator_adapter import NumpySimulatorAdapter

    adapter = NumpySimulatorAdapter(num_qubits=2)
    with pytest.raises(AdapterError):
        await adapter.execute_circuit(
            "numpy:simulator",
            CircuitProgram(num_qubits=5, operations=[GateOperation("H", [4])]),
        )


@pytest.mark.asyncio
async def test_qiskit_adapter_matches_bell_statistics():
    qiskit = pytest.importorskip("qiskit")
    from qnrs.adapters.qiskit_adapter import QiskitAdapter

    adapter = QiskitAdapter()
    messages = await adapter.discover()
    assert any(m.device_id == "qiskit:statevector" for m in messages)
    result = await adapter.execute_circuit("qiskit:statevector", bell_program(shots=4096, seed=1))
    assert result.status == "completed", result.error
    assert set(result.counts) <= {"00", "11"}
    fraction = result.counts.get("00", 0) / 4096
    assert 0.4 < fraction < 0.6


@pytest.mark.asyncio
async def test_qiskit_adapter_rejects_unknown_gate():
    pytest.importorskip("qiskit")
    from qnrs.adapters.qiskit_adapter import QiskitAdapter

    adapter = QiskitAdapter()
    await adapter.discover()
    result = await adapter.execute_circuit(
        "qiskit:statevector",
        CircuitProgram(num_qubits=1, operations=[GateOperation("MAGIC", [0])]),
    )
    assert result.status == "failed"
    assert "MAGIC" in (result.error or "")


@pytest.mark.asyncio
async def test_silq_frontend_compiles_and_runs():
    from qnrs.adapters.silq_adapter import SilqAdapter, SilqSyntaxError
    from qnrs.adapters.simulator_adapter import NumpySimulatorAdapter

    adapter = SilqAdapter(delegate=NumpySimulatorAdapter())
    compiled = adapter.compile(
        """
        H(q0);
        CNOT(q0, q1);
        x = measure(q0);
        """,
        shots=1024,
    )
    assert compiled.program.num_qubits == 2
    assert [op.name for op in compiled.program.operations] == ["H", "CNOT"]
    assert compiled.qubit_names == {"q0": 0, "q1": 1}

    result = await adapter.execute_source("X(q0);", shots=256)
    assert result.status == "completed"
    assert result.counts.get("01", 0) + result.counts.get("10", 0) > 0 or "1" in result.counts


def test_silq_frontend_handles_pi_constants():
    from qnrs.adapters.silq_adapter import SilqAdapter

    adapter = SilqAdapter()
    compiled = adapter.compile("RZ(pi/4, q0); RX(-pi, q1);")
    assert compiled.program.operations[0].params[0] == pytest.approx(math.pi / 4)
    assert compiled.program.operations[1].params[0] == pytest.approx(-math.pi)


def test_silq_frontend_rejects_unknown_operation():
    from qnrs.adapters.silq_adapter import SilqAdapter, SilqSyntaxError

    adapter = SilqAdapter()
    with pytest.raises(SilqSyntaxError):
        adapter.compile("FROBNICATE(q0);")


@pytest.mark.asyncio
async def test_pennylane_adapter_bell_fidelity():
    pytest.importorskip("pennylane")
    from qnrs.adapters.pennylane_adapter import PennyLaneAdapter

    adapter = PennyLaneAdapter()
    await adapter.discover()
    result = await adapter.execute_circuit("pennylane:default.qubit", bell_program(shots=512, seed=2))
    assert result.status == "completed", result.error
    state = np.asarray(result.statevector)
    assert abs(state[0]) == pytest.approx(1 / math.sqrt(2), abs=1e-6)


@pytest.mark.asyncio
async def test_pennylane_variational_optimizer_finds_single_qubit_state():
    from qnrs.adapters.pennylane_adapter import PennyLaneAdapter, VariationalTemplate

    adapter = PennyLaneAdapter()
    target = [0.0, 1.0]  # |1>
    result = await adapter.optimize(
        target, template=VariationalTemplate(num_qubits=1, layers=1), steps=80,
        learning_rate=0.3, seed=1,
    )
    assert result.final_fidelity > 0.99
    assert len(result.optimal_parameters) == 3


@pytest.mark.asyncio
async def test_qiskit_aer_runs_and_matches_statevector_backend():
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")
    from qnrs.adapters.qiskit_adapter import QiskitAdapter

    adapter = QiskitAdapter()
    await adapter.discover()
    result = await adapter.execute_circuit("qiskit:aer", bell_program(shots=4096, seed=4))
    assert result.status == "completed", result.error
    assert set(result.counts) <= {"00", "11"}
    fraction = result.counts.get("00", 0) / 4096
    assert 0.4 < fraction < 0.6


@pytest.mark.asyncio
async def test_qiskit_noisy_backend_is_executable():
    pytest.importorskip("qiskit")
    from qnrs.adapters.qiskit_adapter import QiskitAdapter

    adapter = QiskitAdapter()
    await adapter.discover()
    result = await adapter.execute_circuit("qiskit:aer-noisy", bell_program(shots=1024, seed=6))
    assert result.status == "completed", result.error
    assert sum(result.counts.values()) == 1024


@pytest.mark.asyncio
async def test_silq_program_with_measurement_does_not_collapse_to_zero():
    """Regression: a program with an explicit measure() must still sample all qubits."""
    from qnrs.adapters.silq_adapter import SilqAdapter
    from qnrs.adapters.simulator_adapter import NumpySimulatorAdapter

    adapter = SilqAdapter(delegate=NumpySimulatorAdapter())
    target = adapter._delegate
    compiled = adapter.compile("H(q0); CNOT(q0, q1); H(q2); x := measure(q0);", shots=2048)
    result = await target.execute_circuit("numpy:simulator", compiled.program)
    assert result.status == "completed"
    assert len(result.counts) > 1
    assert all(len(key) == 3 for key in result.counts)


@pytest.mark.asyncio
async def test_labber_mock_bridge_pulse_execution():
    from qnrs.adapters.labber_adapter import LabberAdapter

    adapter = LabberAdapter(use_mock=True)
    try:
        messages = await adapter.discover()
        assert any(m.device_id == "labber:nanorobot-drive-1" for m in messages)
        sequence = PulseSequence(sequence_id="s1", repetitions=512)
        sequence.add(
            PulseParameters(frequency_ghz=5.1, amplitude=0.6, duration_ns=40.0,
                            envelope=Envelope.GAUSSIAN, target="drive-0")
        )
        result = await adapter.execute_pulse("labber:nanorobot-drive-1", sequence)
        assert result.status == "completed", result.error
        assert sum(result.counts.values()) == 512
        assert result.readout[0]["channel"] == "drive-0"
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_labber_mixer_and_instrument_ops():
    from qnrs.adapters.labber_adapter import LabberAdapter

    adapter = LabberAdapter(use_mock=True)
    try:
        await adapter.set_instrument("drive.amplitude", 0.42)
        assert await adapter.get_instrument("drive.amplitude") == pytest.approx(0.42)
        tuning = await adapter.tune_mixer("drive-0")
        assert "if_freq_mhz" in tuning
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_qcodes_adapter_parameter_roundtrip():
    from qnrs.adapters.qcodes_adapter import QCoDeSAdapter

    adapter = QCoDeSAdapter()
    await adapter.set_parameter("qcodes:drive_unit", "amplitude", 0.33)
    assert await adapter.get_parameter("qcodes:drive_unit", "amplitude") == pytest.approx(0.33)
    snapshot = await adapter.snapshot("qcodes:drive_unit")
    assert snapshot["name"] == "drive_unit"


@pytest.mark.asyncio
async def test_qcodes_adapter_executes_pulse():
    from qnrs.adapters.qcodes_adapter import QCoDeSAdapter

    adapter = QCoDeSAdapter()
    sequence = PulseSequence(sequence_id="qc1", repetitions=128)
    sequence.add(PulseParameters(frequency_ghz=5.0, amplitude=0.5, duration_ns=20.0, target="d0"))
    result = await adapter.execute_pulse("qcodes:drive_unit", sequence)
    assert result.status == "completed"
    assert sum(result.counts.values()) == 128
