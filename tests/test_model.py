import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import nelson_siegel, fit_nelson_siegel


def test_nelson_siegel_recovers_known_parameters():
    """If we generate synthetic yields from known params, the fit should recover them closely."""
    true_beta0, true_beta1, true_beta2, true_lam = 4.5, -1.2, 0.5, 1.8
    maturities = np.array([1/12, 3/12, 6/12, 1, 2, 5, 10, 30])

    synthetic_yields = nelson_siegel(maturities, true_beta0, true_beta1, true_beta2, true_lam)

    fitted_beta0, fitted_beta1, fitted_beta2, fitted_lam = fit_nelson_siegel(maturities, synthetic_yields)

    assert abs(fitted_beta0 - true_beta0) < 0.1
    assert abs(fitted_beta1 - true_beta1) < 0.1


def test_nelson_siegel_curve_is_smooth():
    """The formula shouldn't produce NaN or wild values across a normal maturity range."""
    tau = np.linspace(0.01, 30, 50)
    result = nelson_siegel(tau, 4.5, -1.2, 0.5, 1.8)
    assert not np.isnan(result).any()
    assert result.max() < 20  # sanity bound, yields shouldn't be absurd
    