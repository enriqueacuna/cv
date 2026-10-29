"""Shared steps: data -> features -> labels."""
from __future__ import annotations

from .data import spread_series
from .features import atr, build_features
from .labels import triple_barrier


def prepare(df, cfg):
    lab = cfg.labels
    a = atr(df, lab.atr_period)
    spread = spread_series(df, cfg.costs.spread)
    X = build_features(df, lab.atr_period)
    y = triple_barrier(df, a, spread, lab.tp_atr, lab.sl_atr, lab.horizon_bars)
    return X, y, a, spread
