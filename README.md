# US Treasury Yield Curve Modeling

Fetches live Treasury yield data across 8 maturities from the FRED API, fits a Nelson-Siegel model to estimate the curve's level, slope, and curvature over time, and benchmarks three forecasting approaches for the 10-year yield against a naive baseline.

## What it does

- Pulls current and historical daily yields (1M through 30Y) via the FRED API
- Fits the Nelson-Siegel parametric curve model using nonlinear least squares, decomposing the curve into 4 interpretable parameters: level (β0), slope (β1), curvature (β2), and decay (λ)
- Reports a 95% confidence interval and R² for every daily fit, not just point estimates
- Compares two fitting strategies across ~1 year of history: independent daily fits vs. warm-started fits that carry forward the previous day's parameters
- Forecasts the 10-year yield three ways — random walk baseline, ARIMA, and linear regression on lagged yield + curve slope — evaluated on a chronological train/test split
- Validates the Nelson-Siegel implementation against synthetic data with known ground-truth parameters (pytest)

## Why Nelson-Siegel

Raw yield data across 8 maturities doesn't tell you much on its own, so Nelson-Siegel compresses that into a handful of numbers that map to real economic meaning — β1, for example, is closely related to the 2s10s spread that markets watch for recession signals, but expressed as a smooth, continuous model rather than a single hand-picked difference.

## Why two fitting strategies

Independent daily fits are the "honest" baseline, but yield curves move gradually day to day, so re-fitting from scratch each time can introduce noise from the optimizer landing in slightly different places. Warm-starting each day from the previous day's parameters tests whether that noise is real — measured directly as the standard deviation of β1 across the full history under each strategy.

## Why benchmark against a random walk

Yields are close to a random walk empirically, which makes "tomorrow = today" a surprisingly hard baseline to beat. Any forecasting model here is judged against that baseline, not in isolation — a model that "works" but doesn't beat the naive forecast isn't actually adding value.

## Setup

\`\`\`bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
\`\`\`

Create a `.env` file with a free [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html):

\`\`\`
FRED_API_KEY=your_key_here
\`\`\`

## Run

\`\`\`bash
python main.py
\`\`\`

Outputs land in `outputs/`:
- `yield_curve.png` — today's raw curve snapshot
- `ns_fit.png` — Nelson-Siegel fit vs. actual yields, single day
- `ns_history_comparison.png` — β1 over time under both fitting strategies, with 95% CI band and daily R²
- `forecast_comparison.png` — actual vs. predicted 10Y yield, all three forecasting methods, on the held-out test period

## Tests

\`\`\`bash
pytest tests/
\`\`\`

## Sample output

![Nelson-Siegel fit](assets/ns_fit.png)

## Project structure

\`\`\`
src/
  fetch.py     — FRED API calls, snapshot + historical pulls
  model.py     — Nelson-Siegel formula, fitting with CI and R², historical fit loop
  forecast.py  — random walk / ARIMA / regression forecasting, train/test split, evaluation
tests/
  test_model.py — validates NS fit recovers known parameters from synthetic data
main.py        — orchestrates fetch -> fit -> forecast -> plot
\`\`\`

## Roadmap

- 2s10s and 10Y-3M spread analysis, inversion detection
- Backtest curve inversion as a recession signal against NBER dates
- Expand regression features (β0, β2, multi-day momentum) to test whether they can beat the random-walk baseline
