"""Shared steps: data -> features -> labels."""
from __future__ import annotations

from .config import to_dict
from .data import spread_series
from .features import atr, build_features
from .labels import triple_barrier


def flag_params(cfg) -> dict | None:
    pat = getattr(cfg, "patterns", None)
    return to_dict(pat.flags) if pat is not None else None


def prepare(df, cfg):
    lab = cfg.labels
    a = atr(df, lab.atr_period)
    spread = spread_series(df, cfg.costs.spread)
    X = build_features(df, lab.atr_period, flag_params(cfg))
    y = triple_barrier(df, a, spread, lab.tp_atr, lab.sl_atr, lab.horizon_bars)
    return X, y, a, spread
