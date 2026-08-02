import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression


def build_lagged_dataset(curve_history: pd.DataFrame, ns_history: pd.DataFrame) -> pd.DataFrame:
    """Combine 10Y yield and NS beta1 (slope) into a lagged feature set."""
    df = pd.DataFrame({
        "yield_10y": curve_history["10Y"],
        "beta1": ns_history["beta1"],
    }).dropna()

    df["yield_10y_lag1"] = df["yield_10y"].shift(1)
    df["beta1_lag1"] = df["beta1"].shift(1)
    return df.dropna()


def time_split(df: pd.DataFrame, test_frac: float = 0.2):
    split_idx = int(len(df) * (1 - test_frac))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def evaluate(y_true, y_pred) -> dict:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    return {"rmse": rmse, "mae": mae}


def forecast_random_walk(test: pd.DataFrame) -> np.ndarray:
    """Naive baseline: tomorrow = today. Uses actual lagged value each step."""
    return test["yield_10y_lag1"].values


def forecast_regression(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Linear regression: predict yield_10y from yesterday's yield + yesterday's curve slope."""
    features = ["yield_10y_lag1", "beta1_lag1"]
    model = LinearRegression()
    model.fit(train[features], train["yield_10y"])
    return model.predict(test[features])


def forecast_arima(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """ARIMA(1,1,1) fit once on train, forecast the full test horizon blind (multi-step)."""
    model = ARIMA(train["yield_10y"], order=(1, 1, 1))
    fitted = model.fit()
    forecast = fitted.forecast(steps=len(test))
    return forecast.values


def run_forecast_comparison(curve_history: pd.DataFrame, ns_history: pd.DataFrame) -> dict:
    df = build_lagged_dataset(curve_history, ns_history)
    train, test = time_split(df)

    rw_pred = forecast_random_walk(test)
    reg_pred = forecast_regression(train, test)
    arima_pred = forecast_arima(train, test)

    results = {
        "test_dates": test.index,
        "actual": test["yield_10y"].values,
        "random_walk": {"pred": rw_pred, "metrics": evaluate(test["yield_10y"], rw_pred)},
        "regression": {"pred": reg_pred, "metrics": evaluate(test["yield_10y"], reg_pred)},
        "arima": {"pred": arima_pred, "metrics": evaluate(test["yield_10y"], arima_pred)},
    }
    return results
