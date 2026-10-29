"""Bull and bear flag detection, causal at every bar.

A flag at bar t is:
  pole  -- a strong, mostly one-directional move over `pole_bars` bars ending just before the flag
  flag  -- the `flag_bars` bars ending at t-1: a tight consolidation that retraces only part of
           the pole and drifts flat or against it
  breakout -- bar t closes beyond the flag's range in the pole's direction

Besides yes/no flags, the measurements themselves (pole strength, tightness, retracement, drift,
breakout distance) are returned as signed features so the model can learn which flags work.
Positive = bullish geometry, negative = bearish.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULTS = {
    "pole_bars": 10,
    "flag_bars": [5, 10],
    "min_pole_atr": 3.0,       # pole must move at least this many ATRs
    "min_pole_efficiency": 0.5,  # |net move| / sum of |bar moves|: 1.0 = perfectly straight
    "max_retrace": 0.5,        # flag may give back at most 50% of the pole
    "max_tightness": 0.6,      # flag range at most 60% of the pole height
    "max_drift_atr": 0.15,     # flag may drift with the pole by at most this many ATRs per bar
}


def _rolling_slope(s: pd.Series, n: int) -> pd.Series:
    t = pd.Series(np.arange(len(s), dtype=float), index=s.index)
    return s.rolling(n).cov(t) / t.rolling(n).var()


def flag_features(df: pd.DataFrame, a: pd.Series, params: dict | None = None) -> pd.DataFrame:
    p = {**DEFAULTS, **(params or {})}
    c, h, l = df["close"], df["high"], df["low"]
    P = p["pole_bars"]
    out = pd.DataFrame(index=df.index)
    any_flag = pd.Series(0.0, index=df.index)
    any_breakout = pd.Series(0.0, index=df.index)

    for F in p["flag_bars"]:
        pole_end, pole_start = c.shift(F + 1), c.shift(F + 1 + P)
        a_pole = a.shift(F + 1)
        move = pole_end - pole_start
        direction = np.sign(move)
        height = move.abs()
        pole_atr = height / a_pole
        efficiency = height / c.diff().abs().rolling(P).sum().shift(F + 1)

        flag_hi = h.shift(1).rolling(F).max()
        flag_lo = l.shift(1).rolling(F).min()
        tightness = (flag_hi - flag_lo) / height
        retrace = np.where(direction > 0, pole_end - flag_lo, flag_hi - pole_end) / height
        drift = direction * _rolling_slope(c.shift(1), F) / a  # >0 = drifting with the pole
        breakout = np.where(direction > 0, c - flag_hi, flag_lo - c) / a

        is_flag = (
            (pole_atr >= p["min_pole_atr"])
            & (efficiency >= p["min_pole_efficiency"])
            & (tightness <= p["max_tightness"])
            & (retrace <= p["max_retrace"])
            & (drift <= p["max_drift_atr"])
        )
        is_breakout = is_flag & (breakout > 0)

        out[f"flag{F}_pole_atr"] = direction * pole_atr
        out[f"flag{F}_pole_eff"] = efficiency
        out[f"flag{F}_tightness"] = tightness
        out[f"flag{F}_retrace"] = retrace
        out[f"flag{F}_drift"] = drift
        out[f"flag{F}_breakout_atr"] = breakout
        out[f"flag{F}_signal"] = direction * is_flag
        out[f"flag{F}_breakout"] = direction * is_breakout
        any_flag = any_flag.where(any_flag != 0, direction * is_flag)
        any_breakout = any_breakout.where(any_breakout != 0, direction * is_breakout)

    out["flag_any"] = any_flag          # +1 bull flag in progress, -1 bear flag, 0 none
    out["flag_breakout_dir"] = any_breakout  # +1 bull flag breakout this bar, -1 bear, 0 none
    return out.replace([np.inf, -np.inf], np.nan)
