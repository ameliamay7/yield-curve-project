import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import norm


def nelson_siegel(tau, beta0, beta1, beta2, lam):
    """Nelson-Siegel yield curve formula. tau = maturity in years."""
    term1 = (1 - np.exp(-tau / lam)) / (tau / lam)
    term2 = term1 - np.exp(-tau / lam)
    return beta0 + beta1 * term1 + beta2 * term2


def fit_nelson_siegel(maturities_years, yields, initial_guess=None):
    """
    Fit level (beta0), slope (beta1), curvature (beta2), decay (lam).
    Returns fitted params, their 95% CI half-widths, and R^2 for the fit.
    """
    if initial_guess is None:
        initial_guess = [yields[-1], yields[0] - yields[-1], 0.0, 1.5]
    bounds = ([0, -15, -15, 0.1], [15, 15, 15, 5])

    params, pcov = curve_fit(
        nelson_siegel, maturities_years, yields,
        p0=initial_guess, bounds=bounds, maxfev=5000
    )

    # Standard errors from the covariance matrix, 95% CI (alpha = 0.05)
    perr = np.sqrt(np.diag(pcov))
    ci95 = norm.ppf(0.975) * perr  # ~1.96 * std error

    # R^2: how well the fitted curve explains observed yields
    predicted = nelson_siegel(maturities_years, *params)
    ss_res = np.sum((yields - predicted) ** 2)
    ss_tot = np.sum((yields - np.mean(yields)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    return params, ci95, r_squared


def fit_curve_history(curve_history: pd.DataFrame, maturity_years: dict, warm_start: bool = False) -> pd.DataFrame:
    """
    Fit Nelson-Siegel to every day in curve_history.
    warm_start=False -> Strategy A: independent daily fits
    warm_start=True  -> Strategy B: each day starts from previous day's params
    """
    maturities = list(curve_history.columns)
    tau = np.array([maturity_years[m] for m in maturities])

    rows = []
    prev_params = None

    for date, row in curve_history.iterrows():
        y = row.values.astype(float)
        if np.isnan(y).any():
            continue  # skip days with missing maturities

        guess = prev_params if (warm_start and prev_params is not None) else None
        try:
            params, ci95, r2 = fit_nelson_siegel(tau, y, initial_guess=guess)
        except RuntimeError:
            continue  # optimizer failed to converge on this day, skip it

        prev_params = params
        rows.append({
            "date": date,
            "beta0": params[0], "beta1": params[1],
            "beta2": params[2], "lam": params[3],
            "beta1_ci95": ci95[1],
            "r_squared": r2,
        })

    return pd.DataFrame(rows).set_index("date")
