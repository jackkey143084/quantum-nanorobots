import pytest

from qnrs.core.pulses import (
    Envelope,
    PulseParameters,
    PulseSequence,
    PulseValidationError,
    TimingError,
)


def test_pulse_rejects_bad_amplitude():
    with pytest.raises(PulseValidationError):
        PulseParameters(frequency_ghz=5.0, amplitude=1.5, duration_ns=20.0)


def test_drag_requires_beta():
    with pytest.raises(PulseValidationError):
        PulseParameters(frequency_ghz=5.0, amplitude=0.5, duration_ns=20.0, envelope=Envelope.DRAG)


def test_envelope_shapes_peak_in_the_middle_for_gaussian():
    pulse = PulseParameters(frequency_ghz=5.0, amplitude=1.0, duration_ns=100.0,
                            envelope=Envelope.GAUSSIAN)
    assert pulse.envelope_value(50.0) == pytest.approx(1.0)
    assert pulse.envelope_value(0.0) < 0.1


def test_waveform_sample_count():
    pulse = PulseParameters(frequency_ghz=5.0, amplitude=0.5, duration_ns=32.0,
                            envelope=Envelope.SQUARE)
    assert len(pulse.waveform(sample_rate_gsa=1.0)) == 32


def test_sequence_total_duration_accounts_for_gaps_and_repetitions():
    sequence = PulseSequence(sequence_id="s", repetitions=3, gap_ns=5.0)
    sequence.add(PulseParameters(frequency_ghz=5.0, amplitude=0.4, duration_ns=20.0, target="d0"))
    sequence.add(PulseParameters(frequency_ghz=5.0, amplitude=0.4, duration_ns=10.0, target="d1"))
    # 20 + 5 + 10 = 35, plus trailing gap 5 => 40, times 3 repetitions
    assert sequence.compute_total_duration() == pytest.approx(120.0)


def test_sequence_detects_overlap_on_same_channel():
    sequence = PulseSequence(sequence_id="overlap")
    sequence.add(PulseParameters(frequency_ghz=5.0, amplitude=0.4, duration_ns=20.0,
                                 target="d0", start_ns=0.0))
    sequence.add(PulseParameters(frequency_ghz=5.0, amplitude=0.4, duration_ns=20.0,
                                 target="d0", start_ns=10.0))
    with pytest.raises(TimingError):
        sequence.schedule()


def test_sequence_allows_simultaneous_pulses_on_different_channels():
    sequence = PulseSequence(sequence_id="parallel")
    sequence.add(PulseParameters(frequency_ghz=5.0, amplitude=0.4, duration_ns=20.0,
                                 target="d0", start_ns=0.0))
    sequence.add(PulseParameters(frequency_ghz=6.0, amplitude=0.4, duration_ns=20.0,
                                 target="d1", start_ns=0.0))
    schedule = sequence.schedule()
    assert len(schedule) == 2
    assert all(entry["start_ns"] == 0.0 for entry in schedule)


def test_roundtrip_serialisation():
    sequence = PulseSequence(sequence_id="rt", repetitions=2)
    sequence.add(PulseParameters(frequency_ghz=5.0, amplitude=0.4, duration_ns=20.0,
                                 target="d0", envelope=Envelope.RAMP))
    restored = PulseSequence.from_dict(sequence.to_dict())
    assert restored.sequence_id == "rt"
    assert restored.pulses[0].envelope is Envelope.RAMP
    assert restored.compute_total_duration() == pytest.approx(sequence.compute_total_duration())


def test_analog_program_shape():
    sequence = PulseSequence(sequence_id="ap")
    sequence.add(PulseParameters(frequency_ghz=5.0, amplitude=0.5, duration_ns=8.0,
                                 envelope=Envelope.SQUARE, target="d0"))
    program = sequence.analog_program()
    assert program[0]["channel"] == "d0"
    assert len(program[0]["iq"]) == 8
