import numpy as np
import pytest
from scipy.signal import lfilter
from demo1.dsp import fxlms, anc_example, filtering_example
from demo1.quantum import problem, statevector, benchmark


def test_secondary_path_is_stateful_and_sign_matches():
    x = np.random.default_rng(1).normal(0, 0.3, 5000)
    s = [0, 0, 0.7, 0.2]
    result = fxlms(x, [0,0,0,0,0.8,0.1], s)
    np.testing.assert_allclose(result.secondary_output, lfilter(s, [1], result.control), atol=1e-14)
    np.testing.assert_allclose(result.error, result.disturbance-result.secondary_output)


def test_zero_learning_is_uncancelled_baseline():
    result = fxlms(np.ones(100), [0,1], [0,0.5], mu=0)
    np.testing.assert_array_equal(result.error, result.disturbance)


@pytest.mark.parametrize("kind", ["tonal", "broadband"])
def test_fxlms_reduces_noise(kind):
    _, result, metrics = anc_example(kind)
    assert np.all(np.isfinite(result.error))
    assert metrics["noise_reduction_db"] > 10
    assert np.mean(result.error[-8000:]**2) < np.mean(result.error[:8000]**2)


def test_secondary_estimate_can_exceed_controller_length():
    result = fxlms(np.ones(100), [0,0,1], [0,1], [0,1,0.1,0.1,0.1], length=2)
    assert result.error.shape == (100,)


def test_filter_snr_and_known_component_linearity():
    _, _, _, _, metrics = filtering_example()
    assert metrics["snr_improvement_db"] > 15
    assert np.isfinite(metrics["clean_reference_sdr_db"])


def test_qaoa_unitary_and_exact_reference():
    states, energies = problem()
    for angles in [[0,0], [0.3,0.7], [2,1.2]]:
        assert np.sum(np.abs(statevector(angles, energies))**2) == pytest.approx(1)
    assert states[energies.argmin()].tolist() == [1,1,0,0]
    _, _, _, metrics = benchmark()
    assert metrics["qaoa_expected_cost"] >= metrics["exact_cost"] - 1e-10
    assert 0 <= metrics["optimal_probability"] <= 1


def test_preferences_change_exact_choice():
    states, energies = problem([-1, -1, 1, 1])
    assert states[energies.argmin()].tolist() == [0,0,1,1]
