"""Bar-by-bar backtester that executes the model's signals under FTMO rules.

Signals computed at the close of bar t are executed at the open of bar t+1, matching the labels.
Prices are bid; longs fill at ask (bid + spread), shorts are closed at ask.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .risk import FTMORiskManager


@dataclass
class BacktestResult:
    status: str                      # running/completed, passed, failed:<reason>, timeout
    trades: pd.DataFrame
    equity: pd.Series
    start: pd.Timestamp
    end: pd.Timestamp
    trading_days: int
    initial: float
    stats: dict = field(default_factory=dict)


def _is_last_bar_of_week(idx: pd.DatetimeIndex, t: int) -> bool:
    if t + 1 >= len(idx):
        return False
    return idx[t + 1].dayofweek < idx[t].dayofweek or (idx[t + 1] - idx[t]) > pd.Timedelta(days=2)


def run_backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    atr_values: pd.Series,
    spread: pd.Series,
    cfg,
    challenge: bool = False,
    max_days: int | None = None,
) -> BacktestResult:
    lab, ftmo, costs = cfg.labels, cfg.ftmo, cfg.costs
    rm = FTMORiskManager(cfg)
    initial = rm.initial
    target = initial * (1 + ftmo.profit_target_pct / 100)
    h0, h1 = cfg.session.trade_hours
    contract, commission, slip = costs.contract_size, costs.commission_per_lot, costs.slippage

    idx = df.index
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    sp = spread.reindex(idx).to_numpy()
    # Values known at the close of the previous bar drive the decision at this bar's open.
    sig = signals.reindex(idx).fillna(0).shift(1).fillna(0).to_numpy().astype(int)
    a_prev = atr_values.reindex(idx).shift(1).to_numpy()

    balance = initial
    pos = None
    trades, eq_times, eq_vals = [], [], []
    trading_days: set = set()
    cur_day = None
    status = "completed"
    target_hit_day = None
    micro_days: set = set()

    def mtm(price_bid: float, ask_spread: float) -> float:
        if pos is None:
            return 0.0
        exit_px = price_bid if pos["dir"] == 1 else price_bid + ask_spread
        return pos["dir"] * (exit_px - pos["entry"]) * pos["lots"] * contract - commission * pos["lots"]

    def close(t: int, px: float, reason: str) -> None:
        nonlocal balance, pos
        pnl = pos["dir"] * (px - pos["entry"]) * pos["lots"] * contract - commission * pos["lots"]
        balance += pnl
        rm.on_close(pnl)
        trades.append({**pos, "exit_time": idx[t], "exit": px, "pnl": pnl, "reason": reason,
                       "r_multiple": pnl / pos["risk_usd"] if pos["risk_usd"] else np.nan})
        pos = None

    for t in range(1, len(idx)):
        ts = idx[t]
        day = ts.normalize()
        if day != cur_day:
            rm.start_day(balance, balance + mtm(o[t], sp[t]))
            cur_day = day
            if target_hit_day is not None and day.dayofweek < 5:
                micro_days.add(day)  # after hitting target we place one 0.01-lot trade/day to log trading days

        # 1) entries at the open
        if pos is None and sig[t] != 0 and target_hit_day is None and np.isfinite(a_prev[t]):
            friday_late = ftmo.flatten_before_weekend and ts.dayofweek == 4 and ts.hour >= ftmo.friday_flatten_hour - 1
            ok, _ = rm.can_open(balance)
            if ok and h0 <= ts.hour < h1 and not friday_late and not _is_last_bar_of_week(idx, t):
                sl_d, tp_d = lab.sl_atr * a_prev[t], lab.tp_atr * a_prev[t]
                lots = rm.lot_size(balance, sl_d)
                if lots > 0:
                    d = int(sig[t])
                    entry = o[t] + sp[t] + slip if d == 1 else o[t] - slip
                    pos = {"entry_time": ts, "dir": d, "lots": lots, "entry": entry,
                           "sl": entry - d * sl_d, "tp": entry + d * tp_d, "bars": 0,
                           "risk_usd": (sl_d + slip) * contract * lots + commission * lots}
                    rm.on_open()
                    trading_days.add(day)

        # 2) intrabar stop/target and FTMO breach check (stop assumed before target)
        if pos is not None:
            d, sl, tp = pos["dir"], pos["sl"], pos["tp"]
            exit_px = reason = None
            if d == 1:
                worst = l[t]
                if o[t] <= sl:
                    exit_px, reason = o[t] - slip, "sl_gap"
                elif l[t] <= sl:
                    exit_px, reason = sl - slip, "sl"
                elif h[t] >= tp:
                    exit_px, reason = max(tp, o[t]), "tp"
                worst_px = exit_px if reason and reason.startswith("sl") else worst
                worst_eq = balance + d * (worst_px - pos["entry"]) * pos["lots"] * contract - commission * pos["lots"]
            else:
                ao, ah, al = o[t] + sp[t], h[t] + sp[t], l[t] + sp[t]
                if ao >= sl:
                    exit_px, reason = ao + slip, "sl_gap"
                elif ah >= sl:
                    exit_px, reason = sl + slip, "sl"
                elif al <= tp:
                    exit_px, reason = min(tp, ao), "tp"
                worst_px = exit_px if reason and reason.startswith("sl") else ah
                worst_eq = balance + d * (worst_px - pos["entry"]) * pos["lots"] * contract - commission * pos["lots"]
            breach = rm.breach(worst_eq)
            if breach:
                close(t, worst_px, f"breach_{breach}")
                status = f"failed:{breach}"
            elif exit_px is not None:
                close(t, exit_px, reason)

        if status.startswith("failed"):
            eq_times.append(ts)
            eq_vals.append(balance)
            break

        # 3) exits at the close: time stop and weekend flatten
        if pos is not None:
            pos["bars"] += 1
            weekend = ftmo.flatten_before_weekend and (
                (ts.dayofweek == 4 and ts.hour >= ftmo.friday_flatten_hour) or _is_last_bar_of_week(idx, t))
            if pos["bars"] >= lab.horizon_bars or weekend:
                px = c[t] if pos["dir"] == 1 else c[t] + sp[t]
                close(t, px, "weekend" if weekend else "time")

        equity = balance + mtm(c[t], sp[t])
        eq_times.append(ts)
        eq_vals.append(equity)

        if challenge:
            if target_hit_day is None and pos is None and balance >= target:
                target_hit_day = day
            if target_hit_day is not None and len(trading_days | micro_days) >= ftmo.min_trading_days:
                status = "passed"
                break
            if max_days is not None and (ts - idx[0]).days >= max_days:
                status = "timeout"
                break

    if challenge and status == "completed":
        status = "timeout"  # data ran out before pass/fail
    if pos is not None and not status.startswith("failed"):
        close(t, c[t] if pos["dir"] == 1 else c[t] + sp[t], "end")
        eq_vals[-1] = balance

    equity = pd.Series(eq_vals, index=pd.DatetimeIndex(eq_times), name="equity")
    tr = pd.DataFrame(trades)
    res = BacktestResult(status, tr, equity, idx[0], equity.index[-1] if len(equity) else idx[0],
                         len(trading_days | micro_days), initial)
    res.stats = compute_stats(res)
    return res


def compute_stats(res: BacktestResult) -> dict:
    eq, tr, initial = res.equity, res.trades, res.initial
    s = {"status": res.status, "start": str(res.start), "end": str(res.end), "trading_days": res.trading_days}
    if eq.empty:
        return s
    s["final_equity"] = float(eq.iloc[-1])
    s["return_pct"] = float((eq.iloc[-1] / initial - 1) * 100)
    peak = eq.cummax()
    s["max_drawdown_pct_of_initial"] = float(((peak - eq) / initial).max() * 100)
    daily_close = eq.groupby(eq.index.normalize()).last()
    day_start = daily_close.shift(1).fillna(initial)
    day_min = eq.groupby(eq.index.normalize()).min()
    s["worst_daily_loss_pct_of_initial"] = float(((day_start - day_min) / initial).max() * 100)
    dr = daily_close.pct_change().dropna()
    s["sharpe_daily_annualised"] = float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0.0
    s["n_trades"] = int(len(tr))
    if len(tr):
        wins, losses = tr.loc[tr.pnl > 0, "pnl"], tr.loc[tr.pnl <= 0, "pnl"]
        s["win_rate_pct"] = float(len(wins) / len(tr) * 100)
        s["profit_factor"] = float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf")
        s["avg_r_multiple"] = float(tr["r_multiple"].mean())
        s["long_trades"] = int((tr.dir == 1).sum())
        s["short_trades"] = int((tr.dir == -1).sum())
        s["exit_reasons"] = tr["reason"].value_counts().to_dict()
    return s


def simulate_challenges(df, signals, atr_values, spread, cfg) -> tuple[pd.DataFrame, dict]:
    """Start a fresh FTMO challenge every `start_every_days` across the out-of-sample period and
    report how many pass, fail or time out."""
    ch = cfg.challenge
    starts = pd.date_range(df.index[0].normalize(), df.index[-1] - pd.Timedelta(days=ch.max_calendar_days),
                           freq=f"{ch.start_every_days}D")
    rows = []
    for s in starts:
        i0 = df.index.searchsorted(s)
        end = df.index[i0] + pd.Timedelta(days=ch.max_calendar_days + 1)
        seg = df.iloc[i0:df.index.searchsorted(end)]
        if len(seg) < 50:
            continue
        r = run_backtest(seg, signals, atr_values, spread, cfg, challenge=True, max_days=ch.max_calendar_days)
        rows.append({"start": seg.index[0], "status": r.status, "days": (r.end - r.start).days,
                     "return_pct": r.stats.get("return_pct", 0.0), "trades": r.stats.get("n_trades", 0)})
    res = pd.DataFrame(rows)
    if res.empty:
        return res, {}
    summary = {
        "n_challenges": int(len(res)),
        "pass_rate_pct": float((res.status == "passed").mean() * 100),
        "fail_rate_pct": float(res.status.str.startswith("failed").mean() * 100),
        "timeout_rate_pct": float((res.status == "timeout").mean() * 100),
        "median_days_to_pass": float(res.loc[res.status == "passed", "days"].median()) if (res.status == "passed").any() else None,
        "fail_reasons": res.loc[res.status.str.startswith("failed"), "status"].value_counts().to_dict(),
    }
    return res, summary
