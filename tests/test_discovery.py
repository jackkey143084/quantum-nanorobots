import pytest

from qnrs.core.capabilities import (
    ControlProtocol,
    DiscoveryMessage,
    ProtocolError,
    QuantumBackendType,
    QuantumDeviceCapabilities,
)
from qnrs.core.discovery import NegotiationEngine, ProtocolNegotiator, score_protocol


def make_caps(**overrides):
    base = dict(
        backend_type=QuantumBackendType.SUPERCONDUCTING_QUBIT,
        num_qubits=5,
        native_gates=["x", "cx"],
        supported_protocols=[
            ControlProtocol.GATE_SEQUENCE,
            ControlProtocol.PULSE_SEQUENCE,
            ControlProtocol.MEASUREMENT_FEEDBACK,
        ],
        max_circuit_depth=100,
        t1_us=120.0,
        t2_us=90.0,
        gate_time_1q_ns=35.0,
        gate_time_2q_ns=300.0,
        single_qubit_fidelity=0.999,
        two_qubit_fidelity=0.99,
        readout_fidelity=0.97,
        supports_mid_circuit_measurement=True,
    )
    base.update(overrides)
    return QuantumDeviceCapabilities(**base)


class FakeNegotiator(ProtocolNegotiator):
    backend_type = QuantumBackendType.SUPERCONDUCTING_QUBIT

    def __init__(self, messages, healthy=True, bus_key=""):
        super().__init__()
        self._messages = messages
        self._healthy = healthy
        self.bus_key = bus_key

    async def discover(self):
        return self._remember(self._messages)

    async def verify(self, device_id):
        return self._healthy


def test_capabilities_reject_bad_qubit_count():
    with pytest.raises(ProtocolError):
        make_caps(num_qubits=0)


def test_capabilities_reject_unphysical_calibration():
    with pytest.raises(ProtocolError):
        make_caps(t1_us=10.0, t2_us=100.0)


def test_mid_circuit_flag_implies_protocol():
    caps = make_caps(
        supports_mid_circuit_measurement=True,
        supported_protocols=[ControlProtocol.GATE_SEQUENCE],
    )
    assert ControlProtocol.MEASUREMENT_FEEDBACK in caps.supported_protocols


def test_estimated_duration_and_coherence_fit():
    caps = make_caps()
    duration = caps.estimated_duration_us(10, two_qubit_fraction=1.0)
    assert duration == pytest.approx(10 * 300 / 1000)
    # 400 layers * 300ns = 120us > 0.5 * 90us window -> does not fit
    assert caps.fits_coherence(400, safety_factor=0.5) is False
    assert caps.fits_coherence(2, safety_factor=0.5) is True


def test_discovery_signature_roundtrip():
    caps = make_caps()
    message = DiscoveryMessage.create("dev-1", "Device One", caps, key="secret")
    assert message.verify("secret") is True
    assert message.verify("wrong-key") is False


def test_discovery_rejects_tampering():
    caps = make_caps()
    message = DiscoveryMessage.create("dev-1", "Device One", caps, key="secret")
    message.device_name = "Evil"
    assert message.verify("secret") is False


def test_discovery_rejects_stale_message():
    caps = make_caps()
    message = DiscoveryMessage.create("dev-1", "Device One", caps, key="secret")
    message.timestamp -= 10_000
    assert message.verify("secret") is False


def test_score_prefers_measurement_feedback():
    caps = make_caps()
    feedback = score_protocol(ControlProtocol.MEASUREMENT_FEEDBACK, caps)
    gates = score_protocol(ControlProtocol.GATE_SEQUENCE, caps)
    assert feedback > gates


@pytest.mark.asyncio
async def test_engine_negotiates_best_protocol():
    caps = make_caps()
    message = DiscoveryMessage.create("dev-1", "Device One", caps, key="k")
    engine = NegotiationEngine(bus_key="k").register(FakeNegotiator([message], bus_key="k"))
    result = await engine.negotiate("dev-1")
    assert result.protocol is ControlProtocol.MEASUREMENT_FEEDBACK
    assert result.candidates[0][1] >= result.candidates[-1][1]


@pytest.mark.asyncio
async def test_engine_strict_required_protocol_fails_without_fallback():
    caps = make_caps(supported_protocols=[ControlProtocol.GATE_SEQUENCE])
    message = DiscoveryMessage.create("dev-1", "Device One", caps, key="k")
    engine = NegotiationEngine(bus_key="k").register(FakeNegotiator([message], bus_key="k"))
    with pytest.raises(ProtocolError):
        await engine.negotiate(
            "dev-1", required=[ControlProtocol.PULSE_SEQUENCE], allow_fallback=False
        )


@pytest.mark.asyncio
async def test_engine_falls_back_when_allowed():
    caps = make_caps(
        supported_protocols=[ControlProtocol.GATE_SEQUENCE],
        supports_mid_circuit_measurement=False,
    )
    message = DiscoveryMessage.create("dev-1", "Device One", caps, key="k")
    engine = NegotiationEngine(bus_key="k").register(FakeNegotiator([message], bus_key="k"))
    result = await engine.negotiate(
        "dev-1", required=[ControlProtocol.PULSE_SEQUENCE], allow_fallback=True
    )
    assert result.protocol is ControlProtocol.GATE_SEQUENCE
    assert "fell back" in result.notes


@pytest.mark.asyncio
async def test_engine_drops_bad_signature():
    caps = make_caps()
    good = DiscoveryMessage.create("dev-1", "Good", caps, key="k")
    bad = DiscoveryMessage(device_id="dev-2", device_name="Bad", capabilities=caps, signature="deadbeef")
    engine = NegotiationEngine(bus_key="k").register(FakeNegotiator([good, bad], bus_key="k"))
    found = await engine.discover_all()
    assert [m.device_id for m in found] == ["dev-1"]


@pytest.mark.asyncio
async def test_engine_rejects_unhealthy_device():
    caps = make_caps()
    message = DiscoveryMessage.create("dev-1", "Device One", caps, key="k")
    engine = NegotiationEngine(bus_key="k").register(FakeNegotiator([message], healthy=False, bus_key="k"))
    with pytest.raises(ProtocolError):
        await engine.negotiate("dev-1")
