"""Price data loading: MT5 CSV exports, generic OHLC CSVs, live MT5, and synthetic data for tests.

All prices are assumed to be BID prices (what MT5 stores). Ask = bid + spread.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ["open", "high", "low", "close"]


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load either an MT5 "Export bars" file (<DATE> <TIME> <OPEN> ... tab separated)
    or a generic CSV with a time/datetime column and open/high/low/close[/volume/spread]."""
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip("<>").strip().lower() for c in df.columns]
    if "date" in df.columns and "time" in df.columns:
        ts = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
    else:
        tcol = next(c for c in ("time", "datetime", "timestamp", "date") if c in df.columns)
        ts = pd.to_datetime(df[tcol])
    df.index = pd.DatetimeIndex(ts, name="time")
    if "tickvol" in df.columns and "volume" not in df.columns:
        df["volume"] = df["tickvol"]
    if "tick_volume" in df.columns and "volume" not in df.columns:
        df["volume"] = df["tick_volume"]
    return clean(df)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    keep = REQUIRED + [c for c in ("volume", "spread") if c in df.columns]
    df = df[keep].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=REQUIRED)
    bad = (df["high"] < df[["open", "close"]].max(axis=1)) | (df["low"] > df[["open", "close"]].min(axis=1))
    return df[~bad]


def spread_series(df: pd.DataFrame, default_spread: float, point: float = 0.01) -> pd.Series:
    """Spread in price units. MT5 exports spread in points (0.01 for XAUUSD)."""
    if "spread" in df.columns:
        s = df["spread"] * point
        return s.where(s > 0, default_spread)
    return pd.Series(default_spread, index=df.index)


def fetch_mt5(symbol: str = "XAUUSD", timeframe: str = "H1", bars: int = 50000) -> pd.DataFrame:
    """Pull history straight from a running MetaTrader 5 terminal (Windows only)."""
    import MetaTrader5 as mt5  # noqa: lazy import, optional dependency

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    tf = getattr(mt5, f"TIMEFRAME_{timeframe}")
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
    if rates is None:
        raise RuntimeError(f"copy_rates_from_pos failed: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    # MT5 returns server time encoded as a unix timestamp; keep it as naive server time.
    df.index = pd.DatetimeIndex(pd.to_datetime(df["time"], unit="s"), name="time")
    df = df.rename(columns={"tick_volume": "volume"})
    return clean(df)


def synthetic_gold(n_bars: int = 40000, seed: int = 0, start: str = "2019-01-01") -> pd.DataFrame:
    """Synthetic H1 gold-like series with intraday volatility seasonality, volatility clustering and
    weak, regime-switching momentum. ONLY for smoke tests -- results on it say nothing about real markets."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=int(n_bars * 1.45), freq="h")
    idx = idx[idx.dayofweek < 5][:n_bars]  # drop weekends
    hours = idx.hour.to_numpy()
    season = 0.6 + 0.9 * np.exp(-((hours - 15.5) ** 2) / 8) + 0.4 * np.exp(-((hours - 10) ** 2) / 4)
    vol = np.empty(n_bars)
    drift = np.empty(n_bars)
    v, mom, regime = 0.0018, 0.0, 1.0
    rets = np.empty(n_bars)
    for i in range(n_bars):
        if rng.random() < 0.002:
            regime = -regime
        shock = rng.standard_normal()
        v = np.sqrt(0.0018**2 * 0.05 + 0.90 * v**2 + 0.05 * (v * shock) ** 2)
        vol[i] = v * season[i] * 0.55
        mom = 0.97 * mom + 0.03 * (rets[i - 1] if i else 0.0)
        drift[i] = 0.08 * regime * mom + 0.00002
        rets[i] = drift[i] + vol[i] * rng.standard_t(5) / np.sqrt(5 / 3)
    close = 1300 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[1300.0], close[:-1]]) * (1 + rng.normal(0, 0.00005, n_bars))
    wick = np.abs(rng.normal(0, 1, (2, n_bars))) * vol * close * 0.6
    high = np.maximum(open_, close) + wick[0]
    low = np.minimum(open_, close) - wick[1]
    volume = (1000 * season * (1 + 30 * np.abs(rets))).round()
    spread = np.where((hours >= 22) | (hours < 2), 45, 22).astype(float)  # in points
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume, "spread": spread},
        index=pd.DatetimeIndex(idx, name="time"),
    )
