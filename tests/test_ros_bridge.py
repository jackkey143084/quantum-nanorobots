import json

import pytest

from qnrs.core.pulses import Envelope, PulseParameters, PulseSequence
from qnrs.ros2.bridge_logic import QuantumBridge
from qnrs.runtime import QuantumRuntime, RuntimeConfig


@pytest.fixture
async def bridge():
    runtime = QuantumRuntime(RuntimeConfig())
    await runtime.discover()
    bridge = QuantumBridge(runtime)
    yield bridge
    await bridge.close()


@pytest.mark.asyncio
async def test_state_message_shape(bridge):
    message = await bridge.state_message("numpy:simulator")
    for key in ("device_id", "fidelity", "predicted_fidelity", "coherence_level", "age_us", "t1_us", "t2_us"):
        assert key in message
    assert message["device_id"] == "numpy:simulator"


@pytest.mark.asyncio
async def test_capabilities_message_shape(bridge):
    message = await bridge.capabilities_message("numpy:simulator")
    assert message["num_qubits"] >= 2
    assert "gate_sequence" in message["protocols"]


@pytest.mark.asyncio
async def test_negotiate_service(bridge):
    response = await bridge.negotiate("numpy:simulator", ["gate_sequence"])
    assert response["success"] is True
    assert response["protocol"] == "gate_sequence"


@pytest.mark.asyncio
async def test_negotiate_service_bad_protocol(bridge):
    response = await bridge.negotiate("numpy:simulator", ["warp_drive"])
    assert response["success"] is False
    assert "bad protocol" in response["message"]


@pytest.mark.asyncio
async def test_execute_pulse_service(bridge):
    sequence = PulseSequence(sequence_id="ros-pulse", repetitions=64)
    sequence.add(PulseParameters(frequency_ghz=5.0, amplitude=0.4, duration_ns=16.0,
                                 envelope=Envelope.SQUARE, target="drive-0"))
    response = await bridge.execute_pulse(sequence.to_dict())
    assert response["success"] is True
    assert json.loads(response["counts_json"])


@pytest.mark.asyncio
async def test_execute_pulse_service_rejects_garbage(bridge):
    response = await bridge.execute_pulse({"not": "a sequence"})
    assert response["success"] is False


@pytest.mark.asyncio
async def test_submit_circuit_service(bridge):
    program = {
        "num_qubits": 2,
        "operations": [{"name": "H", "qubits": [0]}, {"name": "CNOT", "qubits": [0, 1]}],
        "shots": 128,
    }
    response = await bridge.submit_circuit(program, "numpy:simulator")
    assert response["success"] is True
    assert sum(json.loads(response["counts_json"]).values()) == 128


@pytest.mark.asyncio
async def test_telemetry_json(bridge):
    payload = json.loads(bridge.telemetry_json())
    assert "devices" in payload
