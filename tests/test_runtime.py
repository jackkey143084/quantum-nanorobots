import pytest

from qnrs.adapters.base import CircuitProgram, GateOperation
from qnrs.core.capabilities import ControlProtocol
from qnrs.core.pulses import Envelope, PulseParameters, PulseSequence
from qnrs.runtime import QuantumRuntime, RuntimeConfig


def bell():
    return CircuitProgram(
        num_qubits=2,
        operations=[GateOperation("H", [0]), GateOperation("CNOT", [0, 1])],
        shots=1024,
        seed=9,
    )


@pytest.fixture
async def runtime():
    rt = QuantumRuntime(RuntimeConfig())
    await rt.discover()
    yield rt
    await rt.close()


@pytest.mark.asyncio
async def test_discovery_finds_every_adapter_family(runtime):
    device_ids = set(runtime.devices)
    assert "numpy:simulator" in device_ids
    assert any(d.startswith("qiskit:") for d in device_ids)
    assert any(d.startswith("pennylane:") for d in device_ids)
    assert any(d.startswith("labber:") for d in device_ids)
    assert any(d.startswith("qcodes:") for d in device_ids)


@pytest.mark.asyncio
async def test_plan_prefers_ideal_simulator_and_reports_candidates(runtime):
    plan = runtime.plan(bell())
    assert plan["recommended"] is not None
    ranked = [c["score"] for c in plan["candidates"]]
    assert ranked == sorted(ranked, reverse=True)
    assert all("fits_coherence" in c for c in plan["candidates"])


@pytest.mark.asyncio
async def test_submit_circuit_autoplans_and_records_job(runtime):
    record = await runtime.submit_circuit(bell())
    assert record.status == "completed"
    assert record.result is not None
    assert sum(record.result.counts.values()) == 1024
    assert runtime.get_job(record.job_id) is record


@pytest.mark.asyncio
async def test_submit_circuit_on_named_numpy_device(runtime):
    record = await runtime.submit_circuit(bell(), device_id="numpy:simulator")
    assert record.device_id == "numpy:simulator"
    assert set(record.result.counts) <= {"00", "11"}


@pytest.mark.asyncio
async def test_submit_pulse_routes_to_a_pulse_capable_device(runtime):
    sequence = PulseSequence(sequence_id="rt-pulse", repetitions=256)
    sequence.add(PulseParameters(frequency_ghz=5.0, amplitude=0.5, duration_ns=24.0,
                                 envelope=Envelope.GAUSSIAN, target="drive-0"))
    record = await runtime.submit_pulse(sequence)
    assert record.status == "completed"
    caps = runtime.devices[record.device_id].capabilities
    assert any(caps.supports(p) for p in (ControlProtocol.PULSE_SEQUENCE, ControlProtocol.ANALOG_CONTROL))


@pytest.mark.asyncio
async def test_negotiate_returns_scored_result(runtime):
    result = await runtime.negotiate("numpy:simulator", required=[ControlProtocol.GATE_SEQUENCE])
    assert result.protocol is ControlProtocol.GATE_SEQUENCE
    assert result.candidates


@pytest.mark.asyncio
async def test_describe_reports_capabilities(runtime):
    description = await runtime.describe("numpy:simulator")
    assert description["capabilities"]["num_qubits"] >= 2
    assert description["adapter"] == "numpy"


@pytest.mark.asyncio
async def test_telemetry_tracks_jobs_and_coherence(runtime):
    await runtime.submit_circuit(bell(), device_id="numpy:simulator")
    telemetry = runtime.telemetry()
    assert telemetry["jobs"]["total"] >= 1
    assert "numpy:simulator" in telemetry["coherence"]


@pytest.mark.asyncio
async def test_unknown_device_raises(runtime):
    from qnrs.core.capabilities import ProtocolError

    with pytest.raises(ProtocolError):
        await runtime.describe("does-not-exist")
