"""Command line entry point: python -m ftmo_ml.cli <command> ..."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from .backtest import run_backtest, simulate_challenges
from .config import load_config, to_dict
from .data import fetch_mt5, load_csv, synthetic_gold
from .model import (DirectionalModel, apply_pattern_filter, classification_report, signals_from_proba,
                    training_mask, walk_forward)
from .pipeline import prepare


def cmd_synth(args):
    df = synthetic_gold(args.bars, seed=args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out)
    print(f"wrote {len(df)} synthetic bars to {args.out} (for smoke tests only)")


def cmd_fetch(args):
    cfg = load_config(args.config)
    df = fetch_mt5(cfg.symbol, cfg.timeframe, args.bars)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out)
    print(f"wrote {len(df)} bars {df.index[0]} -> {df.index[-1]} to {args.out}")


def cmd_train(args):
    cfg = load_config(args.config)
    df = load_csv(args.data)
    X, y, _, _ = prepare(df, cfg)
    mask = training_mask(X, y)
    if cfg.model.train_bars and not args.all_history:
        mask &= mask.cumsum() > mask.sum() - cfg.model.train_bars  # same window length as walk-forward
    model = DirectionalModel(to_dict(cfg.model.params)).fit(X[mask], y.y_long[mask], y.y_short[mask])
    model.save(args.out)
    print(f"trained on {int(mask.sum())} bars ending {X.index[mask][-1]}; saved {args.out}")


def flag_report(X, y) -> dict:
    """How often flag breakouts occur out-of-sample and how often their trade hit TP before SL."""
    out = {}
    for name, d in (("bull", 1), ("bear", -1)):
        m = (X["flag_breakout_dir"] == d) & y.y_long.reindex(X.index).notna()
        lab = y.loc[m[m].index, "y_long" if d == 1 else "y_short"]
        out[f"{name}_flag_breakouts"] = int(m.sum())
        out[f"{name}_flag_tp_rate"] = float(lab.mean()) if len(lab) else None
    out["base_tp_rate_long"] = float(y.y_long.reindex(X.index).mean())
    out["base_tp_rate_short"] = float(y.y_short.reindex(X.index).mean())
    return out


def cmd_backtest(args):
    cfg = load_config(args.config)
    out = Path(args.reports)
    out.mkdir(parents=True, exist_ok=True)
    df = load_csv(args.data)
    print(f"data: {len(df)} bars {df.index[0]} -> {df.index[-1]}")
    X, y, a, spread = prepare(df, cfg)
    proba = walk_forward(X, y, cfg)
    oos = df.loc[proba.index]
    print(f"out-of-sample: {len(oos)} bars {oos.index[0]} -> {oos.index[-1]}")

    report = {"classification": classification_report(proba, y)}

    # Threshold sensitivity: if only one magic threshold works, the edge is probably not real.
    sweep = []
    for thr in args.thresholds:
        s = apply_pattern_filter(signals_from_proba(proba, thr, cfg.model.min_edge), X, cfg)
        r = run_backtest(oos, s, a, spread, cfg)
        sweep.append({"threshold": thr, **{k: r.stats.get(k) for k in
                      ("status", "return_pct", "max_drawdown_pct_of_initial", "n_trades", "win_rate_pct", "profit_factor")}})
    report["threshold_sweep"] = sweep

    signals = apply_pattern_filter(signals_from_proba(proba, cfg.model.prob_threshold, cfg.model.min_edge), X, cfg)
    report["flags"] = flag_report(X.loc[proba.index], y)
    res = run_backtest(oos, signals, a, spread, cfg)
    report["backtest"] = res.stats
    ch_table, ch_summary = simulate_challenges(oos, signals, a, spread, cfg)
    report["challenges"] = ch_summary

    res.trades.to_csv(out / "trades.csv", index=False)
    res.equity.to_csv(out / "equity.csv")
    ch_table.to_csv(out / "challenges.csv", index=False)
    proba.to_csv(out / "oos_probabilities.csv")
    (out / "summary.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    print(f"\nreports written to {out}/")


def cmd_live(args):
    from .live import LiveTrader

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler("live.log")])
    LiveTrader(load_config(args.config), args.model, dry_run=not args.live).run()


def main(argv=None):
    p = argparse.ArgumentParser(prog="ftmo_ml")
    p.add_argument("--config", default=None, help="path to config.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="generate synthetic data for smoke tests")
    s.add_argument("--bars", type=int, default=40000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="data/synthetic_xauusd_h1.csv")
    s.set_defaults(func=cmd_synth)

    s = sub.add_parser("fetch-mt5", help="download history from a running MT5 terminal")
    s.add_argument("--bars", type=int, default=60000)
    s.add_argument("--out", default="data/xauusd_h1.csv")
    s.set_defaults(func=cmd_fetch)

    s = sub.add_parser("backtest", help="walk-forward train + out-of-sample backtest + FTMO challenge sim")
    s.add_argument("--data", required=True)
    s.add_argument("--reports", default="reports")
    s.add_argument("--thresholds", type=float, nargs="+", default=[0.40, 0.45, 0.50, 0.55])
    s.set_defaults(func=cmd_backtest)

    s = sub.add_parser("train", help="fit the production model on the most recent data")
    s.add_argument("--data", required=True)
    s.add_argument("--out", default="models/xauusd_model.joblib")
    s.add_argument("--all-history", action="store_true", help="train on all bars instead of the last train_bars")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("live", help="run on an MT5 account (dry-run unless --live)")
    s.add_argument("--model", default="models/xauusd_model.joblib")
    s.add_argument("--live", action="store_true", help="actually send orders")
    s.set_defaults(func=cmd_live)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
