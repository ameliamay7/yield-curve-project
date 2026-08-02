import requests
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("FRED_API_KEY")

MATURITIES = {
    "1M": "DGS1MO",
    "3M": "DGS3MO",
    "6M": "DGS6MO",
    "1Y": "DGS1",
    "2Y": "DGS2",
    "5Y": "DGS5",
    "10Y": "DGS10",
    "30Y": "DGS30",
}

MATURITY_YEARS = {
    "1M": 1/12, "3M": 3/12, "6M": 6/12, "1Y": 1,
    "2Y": 2, "5Y": 5, "10Y": 10, "30Y": 30,
}

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"


def get_latest_yield(series_id: str) -> tuple[str, float]:
    """Fetch the most recent observation for a FRED series."""
    params = {
        "series_id": series_id, "api_key": API_KEY,
        "file_type": "json", "sort_order": "desc", "limit": 1
    }
    response = requests.get(FRED_URL, params=params)
    obs = response.json()["observations"][0]
    return obs["date"], float(obs["value"])


def get_yield_history(series_id: str, limit: int = 250) -> pd.DataFrame:
    """Fetch up to `limit` recent daily observations for a FRED series."""
    params = {
        "series_id": series_id, "api_key": API_KEY,
        "file_type": "json", "sort_order": "desc", "limit": limit
    }
    response = requests.get(FRED_URL, params=params)
    data = response.json()
    history = [(o["date"], o["value"]) for o in data["observations"] if o["value"] != "."]
    return pd.DataFrame(history, columns=["date", "yield"])


def get_curve_snapshot() -> pd.DataFrame:
    """Fetch the latest yield for every maturity in MATURITIES."""
    rows = []
    for label, series_id in MATURITIES.items():
        date, value = get_latest_yield(series_id)
        rows.append({"maturity": label, "date": date, "yield": value})
    return pd.DataFrame(rows)


def get_curve_history() -> pd.DataFrame:
    """Fetch historical yields for every maturity, wide format (date x maturity)."""
    all_history = {}
    for label, series_id in MATURITIES.items():
        hist = get_yield_history(series_id)
        hist["date"] = pd.to_datetime(hist["date"])
        hist["yield"] = hist["yield"].astype(float)
        all_history[label] = hist.set_index("date")["yield"]
    return pd.DataFrame(all_history).sort_index()