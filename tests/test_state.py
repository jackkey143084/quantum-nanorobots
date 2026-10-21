import math

import numpy as np
import pytest

from qnrs.core.state import (
    CoherenceLevel,
    DecoherenceModel,
    QuantumState,
    QuantumStateManager,
    StateError,
    partial_trace,
)


def test_state_normalises_and_builds_density_matrix():
    state = QuantumState(state_vector=[1, 1], num_qubits=1)
    assert state.num_qubits == 1
    assert math.isclose(np.linalg.norm(state.state_vector), 1.0)
    assert np.allclose(state.density_matrix, np.outer(state.state_vector, state.state_vector.conj()))


def test_state_rejects_non_unit_trace():
    with pytest.raises(StateError):
        QuantumState(density_matrix=np.eye(2), num_qubits=1)


def test_partial_trace_bell_state_is_maximally_mixed():
    bell = np.zeros(4, dtype=complex)
    bell[0] = bell[3] = 1 / math.sqrt(2)
    rho = np.outer(bell, bell.conj())
    reduced = partial_trace(rho, 2, (0,))
    assert np.allclose(reduced, np.eye(2) / 2)


def test_evolve_under_pauli_x_hamiltonian_flips_state():
    manager = QuantumStateManager(num_qubits=1)
    manager.initialize()  # |0>
    # H = (pi/2) X ; exp(-i H t) with t=1 gives exp(-i pi/2 X) = -i X
    manager.evolve(np.array([[0, math.pi / 2], [math.pi / 2, 0]], dtype=complex), time_us=1.0)
    probabilities = np.real(np.diag(manager.current.density_matrix))
    assert probabilities[1] == pytest.approx(1.0, abs=1e-9)


def test_evolve_rejects_non_hermitian():
    manager = QuantumStateManager(num_qubits=1)
    manager.initialize()
    with pytest.raises(StateError):
        manager.evolve(np.array([[0, 1j], [0, 0]], dtype=complex), time_us=1.0)


def test_noise_kraus_is_trace_preserving():
    manager = QuantumStateManager(num_qubits=2)
    ops = manager.noise_kraus(50.0)
    completeness = sum(k.conj().T @ k for k in ops)
    assert np.allclose(completeness, np.eye(4), atol=1e-9)


def test_noise_reduces_population_of_excited_state():
    model = DecoherenceModel(t1_us=50.0, t2_us=40.0)
    manager = QuantumStateManager(num_qubits=1, decoherence_model=model)
    manager.initialize(state_vector=[0, 1])  # |1>
    manager.apply_noise(50.0)
    excited = np.real(manager.current.density_matrix[1, 1])
    assert excited == pytest.approx(math.exp(-1.0), abs=1e-6)


def test_fidelity_decay_matches_closed_form():
    model = DecoherenceModel(t1_us=200.0, t2_us=100.0)
    assert model.fidelity_decay(0.0) == pytest.approx(1.0)
    assert model.fidelity_decay(100.0) == pytest.approx(math.exp(-1.0), rel=1e-9)


def test_pure_dephasing_time_identity():
    model = DecoherenceModel(t1_us=100.0, t2_us=50.0)
    # 1/T2 = 1/(2 T1) + 1/T_phi => 1/T_phi = 1/50 - 1/200 = 0.015 => T_phi = 66.67us
    assert model.pure_dephasing_time_us == pytest.approx(1.0 / 0.015, rel=1e-9)


def test_unphysical_t2_is_rejected():
    with pytest.raises(StateError):
        DecoherenceModel(t1_us=10.0, t2_us=30.0)


def test_surface_code_suppresses_logical_error():
    model = DecoherenceModel(t1_us=100.0, t2_us=100.0)
    physical = 0.001
    d3 = model.logical_error_rate(physical, 3)
    d5 = model.logical_error_rate(physical, 5)
    assert 0 < d5 < d3 < physical


def test_surface_code_requires_odd_distance():
    model = DecoherenceModel(t1_us=100.0, t2_us=100.0)
    with pytest.raises(StateError):
        model.logical_error_rate(0.001, 4)


def test_checkpoint_restore_roundtrip():
    manager = QuantumStateManager(num_qubits=1)
    manager.initialize(state_vector=[1, 0])
    manager.checkpoint("initial")
    manager.apply_unitary(np.array([[0, 1], [1, 0]], dtype=complex))
    assert np.real(manager.current.density_matrix[1, 1]) == pytest.approx(1.0)
    manager.restore("initial")
    assert np.real(manager.current.density_matrix[0, 0]) == pytest.approx(1.0)
    assert manager.list_checkpoints() == ["initial"]


def test_fidelity_vs_target():
    manager = QuantumStateManager(num_qubits=1)
    manager.set_target(np.array([0, 1], dtype=complex))
    manager.initialize(state_vector=[1, 0])
    assert manager.current.fidelity == pytest.approx(0.0, abs=1e-9)
    manager.apply_unitary(np.array([[0, 1], [1, 0]], dtype=complex))
    assert manager.current.fidelity == pytest.approx(1.0, abs=1e-9)


def test_coherence_level_classification():
    assert CoherenceLevel.classify(0.99) is CoherenceLevel.EXCELLENT
    assert CoherenceLevel.classify(0.8) is CoherenceLevel.ACCEPTABLE
    assert CoherenceLevel.classify(0.2) is CoherenceLevel.LOST
