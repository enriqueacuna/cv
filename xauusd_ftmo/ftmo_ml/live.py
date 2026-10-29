"""Live/demo trader for an FTMO MetaTrader 5 account.

Requirements: Windows, the MT5 terminal installed and logged in to your FTMO account,
`pip install MetaTrader5`, and "Algo Trading" enabled in the terminal.
ALWAYS run on an FTMO Free Trial / demo first, with --dry-run, then without it.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from .data import clean
from .features import atr, build_features
from .model import DirectionalModel, signals_from_proba
from .risk import FTMORiskManager

log = logging.getLogger("ftmo_ml.live")
MAGIC = 240811
HISTORY_BARS = 3000


class LiveTrader:
    def __init__(self, cfg, model_path: str, dry_run: bool = True, state_path: str = "live_state.json"):
        import MetaTrader5 as mt5  # noqa: optional dependency, Windows only

        self.mt5 = mt5
        self.cfg = cfg
        self.dry_run = dry_run
        self.model = DirectionalModel.load(model_path)
        self.state_path = Path(state_path)
        if not mt5.initialize():
            raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
        info = mt5.symbol_info(cfg.symbol)
        if info is None or (not info.visible and not mt5.symbol_select(cfg.symbol, True)):
            raise RuntimeError(f"symbol {cfg.symbol} not available")
        # Trust the broker's contract spec over the config file.
        cfg.costs.contract_size = info.trade_contract_size
        cfg.costs.lot_step = info.volume_step
        cfg.costs.min_lot = info.volume_min
        self.info = info
        self.tf = getattr(mt5, f"TIMEFRAME_{cfg.timeframe}")
        self.tf_seconds = mt5_timeframe_seconds(cfg.timeframe)
        self.rm = FTMORiskManager(cfg)
        self.state = self._load_state()
        self.last_bar_time = None

    # --- persistence -----------------------------------------------------
    def _load_state(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text())
        return {"day": None, "day_start": None, "halted": False}

    def _save_state(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=2))

    # --- helpers ---------------------------------------------------------
    def server_now(self) -> datetime:
        tick = self.mt5.symbol_info_tick(self.cfg.symbol)
        return datetime.utcfromtimestamp(tick.time)  # MT5 tick time is server time encoded as epoch

    def positions(self):
        return [p for p in (self.mt5.positions_get(symbol=self.cfg.symbol) or []) if p.magic == MAGIC]

    def bars(self) -> pd.DataFrame:
        rates = self.mt5.copy_rates_from_pos(self.cfg.symbol, self.tf, 0, HISTORY_BARS)
        df = pd.DataFrame(rates)
        df.index = pd.DatetimeIndex(pd.to_datetime(df["time"], unit="s"), name="time")
        df = df.rename(columns={"tick_volume": "volume"})
        return clean(df).iloc[:-1]  # drop the still-forming bar

    def roll_day(self, now: datetime, acct) -> None:
        today = now.date().isoformat()
        if self.state["day"] != today:
            self.state.update(day=today, day_start=max(acct.balance, acct.equity))
            self.rm.trades_today = 0
            log.info("new trading day %s, day-start level %.2f", today, self.state["day_start"])
            self._save_state()
        self.rm.day_start = self.state["day_start"]

    def close_position(self, p, reason: str) -> None:
        tick = self.mt5.symbol_info_tick(self.cfg.symbol)
        is_buy = p.type == self.mt5.POSITION_TYPE_BUY
        req = {
            "action": self.mt5.TRADE_ACTION_DEAL, "symbol": p.symbol, "volume": p.volume, "position": p.ticket,
            "type": self.mt5.ORDER_TYPE_SELL if is_buy else self.mt5.ORDER_TYPE_BUY,
            "price": tick.bid if is_buy else tick.ask, "deviation": 30, "magic": MAGIC,
            "comment": f"close:{reason}"[:31], "type_filling": self.mt5.ORDER_FILLING_IOC,
        }
        log.warning("closing #%s (%s)", p.ticket, reason)
        if not self.dry_run:
            r = self.mt5.order_send(req)
            if r is None or r.retcode != self.mt5.TRADE_RETCODE_DONE:
                log.error("close failed: %s", r)

    def open_position(self, direction: int, lots: float, sl_dist: float, tp_dist: float) -> None:
        tick = self.mt5.symbol_info_tick(self.cfg.symbol)
        price = tick.ask if direction == 1 else tick.bid
        digits = self.info.digits
        req = {
            "action": self.mt5.TRADE_ACTION_DEAL, "symbol": self.cfg.symbol, "volume": lots,
            "type": self.mt5.ORDER_TYPE_BUY if direction == 1 else self.mt5.ORDER_TYPE_SELL,
            "price": price, "sl": round(price - direction * sl_dist, digits),
            "tp": round(price + direction * tp_dist, digits), "deviation": 30, "magic": MAGIC,
            "comment": "ftmo_ml", "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self.mt5.ORDER_FILLING_IOC,
        }
        log.info("%s %s %.2f lots @ %.2f sl %.2f tp %.2f%s", "BUY" if direction == 1 else "SELL",
                 self.cfg.symbol, lots, price, req["sl"], req["tp"], " [dry-run]" if self.dry_run else "")
        if not self.dry_run:
            r = self.mt5.order_send(req)
            if r is None or r.retcode != self.mt5.TRADE_RETCODE_DONE:
                log.error("order failed: %s", r)
                return
        self.rm.on_open()

    # --- main loop -------------------------------------------------------
    def guard(self, now: datetime, acct) -> None:
        """Runs every few seconds: protects the FTMO limits regardless of model output."""
        ftmo = self.cfg.ftmo
        if acct.equity <= self.rm.hard_floor + self.rm.buffer or acct.equity <= self.rm.internal_total_floor:
            for p in self.positions():
                self.close_position(p, "equity_guard")
            if acct.equity <= self.rm.internal_total_floor:
                self.state["halted"] = True
                self._save_state()
        if ftmo.flatten_before_weekend and now.weekday() == 4 and now.hour >= ftmo.friday_flatten_hour:
            for p in self.positions():
                self.close_position(p, "weekend")
        for p in self.positions():
            held_bars = (now.timestamp() - p.time) // self.tf_seconds
            if held_bars >= self.cfg.labels.horizon_bars:
                self.close_position(p, "time")

    def on_new_bar(self, now: datetime, acct) -> None:
        cfg = self.cfg
        df = self.bars()
        X = build_features(df, cfg.labels.atr_period).iloc[[-1]]
        proba = self.model.predict_proba(X)
        sig = int(signals_from_proba(proba, cfg.model.prob_threshold, cfg.model.min_edge).iloc[-1])
        a = float(atr(df, cfg.labels.atr_period).iloc[-1])
        log.info("bar %s p_long=%.3f p_short=%.3f signal=%d atr=%.2f equity=%.2f",
                 df.index[-1], proba.p_long.iloc[-1], proba.p_short.iloc[-1], sig, a, acct.equity)
        if sig == 0 or self.positions() or self.state["halted"]:
            return
        h0, h1 = cfg.session.trade_hours
        if not (h0 <= now.hour < h1):
            return
        if cfg.ftmo.flatten_before_weekend and now.weekday() == 4 and now.hour >= cfg.ftmo.friday_flatten_hour - 1:
            return
        target = self.rm.initial * (1 + cfg.ftmo.profit_target_pct / 100)
        if acct.balance >= target:
            log.info("profit target reached -- not opening new risk")
            return
        ok, why = self.rm.can_open(acct.equity)
        if not ok:
            log.info("entry blocked: %s", why)
            return
        sl_d, tp_d = cfg.labels.sl_atr * a, cfg.labels.tp_atr * a
        lots = self.rm.lot_size(acct.equity, sl_d)
        if lots > 0:
            self.open_position(sig, lots, sl_d, tp_d)

    def run(self, poll_seconds: int = 5) -> None:
        log.info("live trader started (dry_run=%s)", self.dry_run)
        while True:
            try:
                acct = self.mt5.account_info()
                now = self.server_now()
                self.roll_day(now, acct)
                self.guard(now, acct)
                rates = self.mt5.copy_rates_from_pos(self.cfg.symbol, self.tf, 0, 1)
                bar_time = int(rates[0]["time"])
                if self.last_bar_time is not None and bar_time != self.last_bar_time:
                    self.on_new_bar(now, acct)
                self.last_bar_time = bar_time
            except Exception:  # keep the loop alive; the broker-side SL still protects open trades
                log.exception("loop error")
            time.sleep(poll_seconds)


def mt5_timeframe_seconds(tf: str) -> int:
    unit, n = tf[0], int(tf[1:])
    return n * {"M": 60, "H": 3600, "D": 86400}[unit]
