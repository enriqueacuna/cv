"""Gradient-boosted classifiers for P(long TP before SL) and P(short TP before SL),
trained with purged walk-forward validation so every prediction is out-of-sample."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

WARMUP_BARS = 250  # longest feature lookback (EMA200 / 240-bar ATR mean)


@dataclass
class DirectionalModel:
    params: dict
    feature_names: list[str] = field(default_factory=list)
    long_clf: HistGradientBoostingClassifier | None = None
    short_clf: HistGradientBoostingClassifier | None = None

    def fit(self, X: pd.DataFrame, y_long: pd.Series, y_short: pd.Series) -> "DirectionalModel":
        self.feature_names = list(X.columns)
        self.long_clf = HistGradientBoostingClassifier(**self.params).fit(X, y_long.astype(int))
        self.short_clf = HistGradientBoostingClassifier(**self.params).fit(X, y_short.astype(int))
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[self.feature_names]
        return pd.DataFrame(
            {"p_long": self.long_clf.predict_proba(X)[:, 1], "p_short": self.short_clf.predict_proba(X)[:, 1]},
            index=X.index,
        )

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "DirectionalModel":
        return joblib.load(path)


def training_mask(X: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    mask = labels["y_long"].notna() & labels["y_short"].notna()
    mask.iloc[:WARMUP_BARS] = False
    return mask


def walk_forward(X: pd.DataFrame, labels: pd.DataFrame, cfg) -> pd.DataFrame:
    """Rolling (or expanding) retrain every `test_bars`. Training rows whose label window
    overlaps the test block are purged (embargo = horizon + 1 bars)."""
    m = cfg.model
    embargo = cfg.labels.horizon_bars + 1
    valid = training_mask(X, labels).to_numpy()
    n = len(X)
    first_test = WARMUP_BARS + m.train_bars + embargo
    if first_test >= n:
        raise ValueError(f"need more than {first_test} bars for walk-forward, have {n}")

    preds = []
    for test_start in range(first_test, n, m.test_bars):
        test_end = min(n, test_start + m.test_bars)
        train_end = test_start - embargo
        train_start = WARMUP_BARS if m.expanding_window else max(WARMUP_BARS, train_end - m.train_bars)
        rows = np.arange(train_start, train_end)
        rows = rows[valid[rows]]
        model = DirectionalModel(vars(m.params) if not isinstance(m.params, dict) else m.params)
        model.fit(X.iloc[rows], labels["y_long"].iloc[rows], labels["y_short"].iloc[rows])
        preds.append(model.predict_proba(X.iloc[test_start:test_end]))
    return pd.concat(preds)


def signals_from_proba(proba: pd.DataFrame, threshold: float, min_edge: float) -> pd.Series:
    pl, ps = proba["p_long"], proba["p_short"]
    sig = np.where((pl >= threshold) & (pl - ps >= min_edge), 1, 0)
    sig = np.where((ps >= threshold) & (ps - pl >= min_edge), -1, sig)
    return pd.Series(sig, index=proba.index, name="signal")


def apply_pattern_filter(signals: pd.Series, X: pd.DataFrame, cfg) -> pd.Series:
    """With patterns.trade_only_flags, keep a signal only on a flag breakout in the same direction."""
    pat = getattr(cfg, "patterns", None)
    if pat is None or not pat.trade_only_flags:
        return signals
    flag_dir = X["flag_breakout_dir"].reindex(signals.index).fillna(0)
    return signals.where(signals == flag_dir, 0)


def classification_report(proba: pd.DataFrame, labels: pd.DataFrame) -> dict:
    lab = labels.reindex(proba.index).dropna()
    p = proba.loc[lab.index]
    out = {}
    for side in ("long", "short"):
        y, s = lab[f"y_{side}"].astype(int), p[f"p_{side}"]
        out[f"{side}_base_rate"] = float(y.mean())
        out[f"{side}_auc"] = float(roc_auc_score(y, s)) if y.nunique() == 2 else float("nan")
        top = s >= s.quantile(0.9)
        out[f"{side}_hit_rate_top_decile"] = float(y[top].mean())
    return out
