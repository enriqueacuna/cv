"""Feature engineering. Every feature at bar t uses only data available at the close of bar t."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .patterns import flag_features


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


def build_features(df: pd.DataFrame, atr_period: int = 14, flag_params: dict | None = None) -> pd.DataFrame:
    c, h, l, o = df["close"], df["high"], df["low"], df["open"]
    a = atr(df, atr_period)
    logc = np.log(c)
    f = pd.DataFrame(index=df.index)

    for n in (1, 3, 6, 12, 24, 48, 120):
        f[f"ret_{n}"] = logc.diff(n) / (a / c)  # returns in ATR units, comparable across price levels
    r1 = logc.diff()
    f["vol_24"] = r1.rolling(24).std()
    f["vol_120"] = r1.rolling(120).std()
    f["vol_ratio"] = f["vol_24"] / f["vol_120"]
    f["atr_pct"] = a / c
    f["atr_ratio"] = a / a.rolling(240).mean()

    f["rsi_14"] = rsi(c, 14)
    f["rsi_4"] = rsi(c, 4)
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / a
    for span in (20, 50, 200):
        ema = c.ewm(span=span, adjust=False, min_periods=span).mean()
        f[f"dist_ema{span}"] = (c - ema) / a
        f[f"slope_ema{span}"] = ema.diff(5) / a

    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_pctb"] = (c - (mid - 2 * sd)) / (4 * sd)
    for n in (24, 120):
        hh, ll = h.rolling(n).max(), l.rolling(n).min()
        f[f"range_pos_{n}"] = (c - ll) / (hh - ll)
        f[f"dist_high_{n}"] = (hh - c) / a
        f[f"dist_low_{n}"] = (c - ll) / a

    rng = (h - l).replace(0, np.nan)
    f["body"] = (c - o) / rng
    f["upper_wick"] = (h - np.maximum(o, c)) / rng
    f["lower_wick"] = (np.minimum(o, c) - l) / rng
    f["bar_range_atr"] = (h - l) / a

    if "volume" in df.columns:
        v = df["volume"].replace(0, np.nan)
        f["vol_z"] = (v - v.rolling(120).mean()) / v.rolling(120).std()
        f["vol_rel_hour"] = v / v.groupby(df.index.hour).transform(lambda s: s.rolling(20, min_periods=5).mean())

    # Previous-day levels (shifted one day so nothing from the current day leaks in).
    day = df.index.normalize()
    daily = df.groupby(day).agg(high=("high", "max"), low=("low", "min"), close=("close", "last"), open=("open", "first"))
    prev = daily.shift(1).reindex(day)
    prev.index = df.index
    f["dist_prev_high"] = (c - prev["high"]) / a
    f["dist_prev_low"] = (c - prev["low"]) / a
    f["dist_prev_close"] = (c - prev["close"]) / a
    f["prev_day_ret"] = (prev["close"] - prev["open"]) / a
    today_open = daily["open"].reindex(day)
    today_open.index = df.index
    f["dist_today_open"] = (c - today_open) / a

    hour = df.index.hour + df.index.minute / 60
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    f["dow"] = df.index.dayofweek

    f = f.join(flag_features(df, a, flag_params))
    return f.replace([np.inf, -np.inf], np.nan).astype("float32")
