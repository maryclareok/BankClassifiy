# server/forecast.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import pandas as pd
from server.config import USER_DATA_ROOT

DATA_ROOT = Path(USER_DATA_ROOT).resolve()

# Optional ARIMA (falls back to EWM if not available)
try:
    from pmdarima import auto_arima  # type: ignore
    PMDARIMA_OK = True
except Exception:
    PMDARIMA_OK = False

@dataclass
class ForecastResult:
    as_of: date
    horizon: str
    by_category: dict
    total: float

def _user_csv(user_id: str) -> Path:
    (DATA_ROOT / user_id).mkdir(parents=True, exist_ok=True)
    return DATA_ROOT / user_id / "AllData.csv"

def _load_user_df(user_id: str) -> pd.DataFrame:
    p = _user_csv(user_id)
    if not p.exists():
        return pd.DataFrame(columns=["date","desc","amount","cat"])
    df = pd.read_csv(p)
    df.columns = [c.lower() for c in df.columns]
    assert {"date","desc","amount","cat"}.issubset(df.columns), "AllData.csv needs date,desc,amount,cat"
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    return df

def _resample(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    g = df.groupby("cat").resample(freq, on="date")["amount"].sum().reset_index()
    g = g.sort_values(["cat","date"]).reset_index(drop=True)
    return g

def _ewm_forecast(series: pd.Series, span: int = 3) -> float:
    return float(series.ewm(span=span).mean().iloc[-1])

def _arima_forecast(series: pd.Series) -> float:
    m = auto_arima(series, seasonal=False, stepwise=True, suppress_warnings=True, error_action="ignore")
    return float(m.predict(1)[0])

def forecast_next_period(user_id: str, horizon: str = "next_month") -> ForecastResult:
    df = _load_user_df(user_id)
    if df.empty:
        return ForecastResult(date.today(), horizon, {}, 0.0)

    freq = "W" if horizon == "next_week" else "M"
    agg = _resample(df, freq=freq)
    preds = {}

    for cat in agg["cat"].unique():
        s = agg.loc[agg["cat"] == cat, "amount"].astype(float).reset_index(drop=True)
        if len(s) == 0:
            continue
        if len(s) < 4:
            pred = float(s.iloc[-1])
        elif PMDARIMA_OK and len(s) >= 20:
            try:
                pred = _arima_forecast(s)
            except Exception:
                pred = _ewm_forecast(s)
        else:
            pred = _ewm_forecast(s)
        preds[cat] = pred

    return ForecastResult(date.today(), horizon, preds, float(sum(preds.values())))
