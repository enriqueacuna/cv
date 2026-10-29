import numpy as np
import pandas as pd
import pytest

from ftmo_ml.backtest import run_backtest
from ftmo_ml.config import load_config
from ftmo_ml.data import synthetic_gold
from ftmo_ml.features import atr, build_features
from ftmo_ml.labels import triple_barrier
from ftmo_ml.model import classification_report, walk_forward
from ftmo_ml.pipeline import prepare
from ftmo_ml.risk import FTMORiskManager


@pytest.fixture
def cfg():
    c = load_config()
    c.costs.slippage = 0.0
    c.costs.commission_per_lot = 0.0
    c.session.trade_hours = [0, 24]
    return c


def bars(rows, start="2024-01-02 08:00"):
    """rows: list of (open, high, low, close) hourly bars on weekdays."""
    idx = pd.date_range(start, periods=len(rows), freq="h")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


def test_features_have_no_lookahead():
    df = synthetic_gold(3000, seed=1)
    full = build_features(df)
    cut = 2000
    part = build_features(df.iloc[:cut])
    pd.testing.assert_frame_equal(full.iloc[:cut], part, check_exact=False, rtol=1e-5)


def test_triple_barrier_long_tp_and_short_sl():
    df = bars([(100, 100.5, 99.5, 100)] + [(100, 100.5, 99.8, 100.2)] + [(100.2, 103, 100, 102.8)] * 3)
    a = pd.Series(1.0, index=df.index)
    sp = pd.Series(0.1, index=df.index)
    y = triple_barrier(df, a, sp, tp_atr=2.0, sl_atr=1.0, horizon=3)
    # entry at open of bar 1 = 100.1 ask; TP 102.1 hit in bar 2. Short SL at 101 (ask) hit in bar 2.
    assert y.y_long.iloc[0] == 1
    assert y.y_short.iloc[0] == 0
    assert y.y_long.iloc[-1] != y.y_long.iloc[-1]  # incomplete windows are NaN


def test_same_bar_touch_counts_as_stop():
    df = bars([(100, 100, 100, 100), (100, 103, 98, 100), (100, 100, 100, 100), (100, 100, 100, 100)])
    y = triple_barrier(df, pd.Series(1.0, index=df.index), pd.Series(0.0, index=df.index), 2.0, 1.0, 2)
    assert y.y_long.iloc[0] == 0 and y.y_short.iloc[0] == 0


def test_lot_size_respects_risk_and_ftmo_buffer(cfg):
    rm = FTMORiskManager(cfg)
    lots = rm.lot_size(100_000, stop_distance=10.0)  # $500 risk / ($10 * 100oz) = 0.5 lots
    assert lots == pytest.approx(0.5)
    rm.start_day(100_000, 100_000)
    # Equity 95,800: only $300 of room above the 95,000 daily floor + $500 buffer -> size shrinks
    assert rm.lot_size(95_800, 10.0) == pytest.approx(0.3)
    assert rm.lot_size(95_400, 10.0) == 0.0
    assert rm.breach(94_999) == "daily_loss"
    assert rm.breach(89_999) == "max_loss"
    rm.loss_streak = 2
    assert rm.lot_size(100_000, 10.0) == pytest.approx(0.25)


def test_backtest_long_take_profit(cfg):
    cfg.labels.horizon_bars = 5
    df = bars([(2000, 2001, 1999, 2000)] * 2 + [(2000, 2030, 1999, 2025)] + [(2025, 2026, 2024, 2025)] * 3)
    sig = pd.Series([1, 0, 0, 0, 0, 0], index=df.index)
    a = pd.Series(10.0, index=df.index)
    sp = pd.Series(0.0, index=df.index)
    r = run_backtest(df, sig, a, sp, cfg)
    t = r.trades.iloc[0]
    assert t.reason == "tp" and t.dir == 1
    assert t.lots == pytest.approx(0.5)
    assert t.pnl == pytest.approx(20 * 100 * 0.5)  # TP = 2 ATR = $20/oz
    assert r.stats["final_equity"] == pytest.approx(101_000)


def test_backtest_stops_at_ftmo_breach(cfg):
    cfg.risk.risk_per_trade_pct = 50  # deliberately absurd; sizing is still capped by the buffer
    cfg.risk.max_lots = 1000
    # Huge gap through the stop: loss far beyond plan -> FTMO daily-loss breach must end the run.
    df = bars([(2000, 2000, 2000, 2000)] * 2 + [(1800, 1800, 1700, 1750)] + [(1750, 1750, 1750, 1750)] * 2)
    sig = pd.Series([1, 0, 0, 0, 0], index=df.index)
    r = run_backtest(df, sig, pd.Series(10.0, index=df.index), pd.Series(0.0, index=df.index), cfg)
    assert r.status.startswith("failed")


def planted_edge_series(n=6000, seed=3):
    """Random walk where a down bar at 15:00 is followed by four up-drifting bars."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=int(n * 1.5), freq="h")
    idx = idx[idx.dayofweek < 5][:n]
    r = rng.normal(0, 2.0, n)
    for i in range(n - 5):
        if idx[i].hour == 15 and r[i] < 0:
            r[i + 1:i + 5] += 1.5
    close = 2000 + np.cumsum(r)
    open_ = np.concatenate([[2000.0], close[:-1]])
    wick = np.abs(rng.normal(0, 0.8, (2, n)))
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + wick[0],
                         "low": np.minimum(open_, close) - wick[1], "close": close}, index=idx)


def test_walk_forward_finds_planted_edge_and_not_noise(cfg):
    cfg.model.train_bars, cfg.model.test_bars = 3000, 1000
    df = planted_edge_series()
    X, y, _, _ = prepare(df, cfg)
    proba = walk_forward(X, y, cfg)
    setup = ((df.index.hour == 15) & (df.close < df.open))[proba.index]
    assert proba.p_long[setup].mean() > proba.p_long[~setup].mean() + 0.08

    X, y, _, _ = prepare(synthetic_gold(6000, seed=5), cfg)
    assert abs(classification_report(walk_forward(X, y, cfg), y)["long_auc"] - 0.5) < 0.05


def flag_bars(direction=1):
    """Flat base, 10-bar straight pole of +/-2 per bar, 6-bar tight pullback, then a breakout bar."""
    closes = [2000.0] * 30
    for _ in range(10):
        closes.append(closes[-1] + 2 * direction)
    top = closes[-1]
    closes += [top - 0.6 * direction, top - 1.0 * direction, top - 0.8 * direction,
               top - 1.2 * direction, top - 0.9 * direction, top - 1.1 * direction]
    closes.append(top + 2.0 * direction)  # breakout
    rows = [(c - 0.3 * direction if i else c, max(c, c - 0.3 * direction) + 0.2,
             min(c, c - 0.3 * direction) - 0.2, c) for i, c in enumerate(closes)]
    return bars(rows, start="2024-01-02 00:00")


@pytest.mark.parametrize("direction", [1, -1])
def test_flag_breakout_detected(direction):
    df = flag_bars(direction)
    f = build_features(df, atr_period=5, flag_params={"flag_bars": [5]})
    assert f["flag_breakout_dir"].iloc[-1] == direction
    assert f["flag_any"].iloc[-1] == direction
    assert (f["flag_breakout_dir"].iloc[:-1] != direction).all()  # nothing fires before the breakout


def test_pattern_filter_keeps_only_matching_breakouts(cfg):
    from types import SimpleNamespace

    from ftmo_ml.model import apply_pattern_filter

    idx = pd.date_range("2024-01-02", periods=4, freq="h")
    sig = pd.Series([1, -1, 1, -1], index=idx)
    X = pd.DataFrame({"flag_breakout_dir": [1, 1, 0, -1]}, index=idx)
    cfg.patterns = SimpleNamespace(trade_only_flags=True)
    assert apply_pattern_filter(sig, X, cfg).tolist() == [1, 0, 0, -1]
