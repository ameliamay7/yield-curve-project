import numpy as np
import matplotlib.pyplot as plt
from src.fetch import get_curve_snapshot, get_curve_history, MATURITY_YEARS
from src.model import nelson_siegel, fit_nelson_siegel

df = get_curve_snapshot()
print(df)

plt.figure(figsize=(8, 5))
plt.plot(df["maturity"], df["yield"], marker="o", linewidth=2)
plt.title(f"US Treasury Yield Curve — {df['date'].iloc[0]}")
plt.xlabel("Maturity")
plt.ylabel("Yield (%)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/yield_curve.png")

curve_history = get_curve_history()
print("\nCurve history shape:", curve_history.shape)

tau = np.array([MATURITY_YEARS[m] for m in df["maturity"]])
y = df["yield"].values
params, ci95, r2 = fit_nelson_siegel(tau, y)
beta0, beta1, beta2, lam = params
print(f"Single-day fit R^2: {r2:.4f}")
print(f"\nNelson-Siegel fit: beta0={beta0:.3f}, beta1={beta1:.3f}, beta2={beta2:.3f}, lambda={lam:.3f}")

tau_smooth = np.linspace(0.01, 30, 200)
fitted = nelson_siegel(tau_smooth, beta0, beta1, beta2, lam)

plt.figure(figsize=(8, 5))
plt.scatter(tau, y, color="black", zorder=5, label="Actual yields")
plt.plot(tau_smooth, fitted, color="steelblue", linewidth=2, label="Nelson-Siegel fit")
plt.title(f"Nelson-Siegel Fit — {df['date'].iloc[0]}")
plt.xlabel("Maturity (years)")
plt.ylabel("Yield (%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/ns_fit.png")
print("Saved plots to outputs/")

from src.model import fit_curve_history

print("\nFitting Strategy A: independent daily fits...")
hist_a = fit_curve_history(curve_history, MATURITY_YEARS, warm_start=False)

print("Fitting Strategy B: warm-started daily fits...")
hist_b = fit_curve_history(curve_history, MATURITY_YEARS, warm_start=True)

print(f"\nStrategy A — mean R^2: {hist_a['r_squared'].mean():.4f}, beta1 std dev: {hist_a['beta1'].std():.4f}")
print(f"Strategy B — mean R^2: {hist_b['r_squared'].mean():.4f}, beta1 std dev: {hist_b['beta1'].std():.4f}")

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

axes[0].plot(hist_a.index, hist_a["beta1"], label="Strategy A (independent)", alpha=0.7)
axes[0].plot(hist_b.index, hist_b["beta1"], label="Strategy B (warm-started)", alpha=0.7)
axes[0].fill_between(hist_a.index, hist_a["beta1"] - hist_a["beta1_ci95"],
                      hist_a["beta1"] + hist_a["beta1_ci95"], alpha=0.15, label="95% CI (Strategy A)")
axes[0].set_ylabel("beta1 (slope)")
axes[0].set_title("Nelson-Siegel slope (beta1) over the past year — two fitting strategies")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(hist_a.index, hist_a["r_squared"], label="Strategy A R²", alpha=0.7)
axes[1].plot(hist_b.index, hist_b["r_squared"], label="Strategy B R²", alpha=0.7)
axes[1].set_ylabel("R²")
axes[1].set_xlabel("Date")
axes[1].set_title("Daily fit quality")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/ns_history_comparison.png")
print("Saved plot to outputs/ns_history_comparison.png")

from src.forecast import run_forecast_comparison

print("\nRunning forecast comparison (random walk vs ARIMA vs regression)...")
results = run_forecast_comparison(curve_history, hist_a)

for name in ["random_walk", "arima", "regression"]:
    m = results[name]["metrics"]
    print(f"{name:12s}  RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(results["test_dates"], results["actual"], color="black", linewidth=2, label="Actual")
plt.plot(results["test_dates"], results["random_walk"]["pred"], "--", alpha=0.7, label="Random walk")
plt.plot(results["test_dates"], results["arima"]["pred"], "--", alpha=0.7, label="ARIMA(1,1,1)")
plt.plot(results["test_dates"], results["regression"]["pred"], "--", alpha=0.7, label="Regression (lag + slope)")
plt.title("10Y yield — forecast comparison on held-out test period")
plt.xlabel("Date")
plt.ylabel("Yield (%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/forecast_comparison.png")
print("Saved plot to outputs/forecast_comparison.png")
