"""FTMO rule enforcement and position sizing, shared by the backtester and the live trader."""
from __future__ import annotations

import math


class FTMORiskManager:
    """Tracks FTMO's hard limits plus our tighter internal limits.

    FTMO daily loss: equity must stay above (day-start level - 5% of initial balance).
    We use max(balance, equity) at the day rollover as the day-start level, which is the
    stricter interpretation. FTMO max loss: equity must stay above 90% of initial balance.
    """

    def __init__(self, cfg):
        self.initial = float(cfg.account.initial_balance)
        f, r, c = cfg.ftmo, cfg.risk, cfg.costs
        self.max_daily_loss = f.max_daily_loss_pct / 100 * self.initial
        self.total_floor = self.initial * (1 - f.max_total_loss_pct / 100)
        self.internal_daily_loss = r.internal_daily_stop_pct / 100 * self.initial
        self.internal_total_floor = self.initial * (1 - r.internal_total_stop_pct / 100)
        self.buffer = r.safety_buffer_pct / 100 * self.initial
        self.risk_per_trade = r.risk_per_trade_pct / 100 * self.initial
        self.max_trades_per_day = r.max_trades_per_day
        self.reduce_after = r.reduce_risk_after_losses
        self.max_lots = r.max_lots
        self.contract = c.contract_size
        self.lot_step = c.lot_step
        self.min_lot = c.min_lot
        self.commission = c.commission_per_lot
        self.slippage = c.slippage
        self.day_start = self.initial
        self.trades_today = 0
        self.loss_streak = 0

    # --- state updates -------------------------------------------------
    def start_day(self, balance: float, equity: float) -> None:
        self.day_start = max(balance, equity)
        self.trades_today = 0

    def on_open(self) -> None:
        self.trades_today += 1

    def on_close(self, pnl: float) -> None:
        self.loss_streak = self.loss_streak + 1 if pnl < 0 else 0

    # --- limits --------------------------------------------------------
    @property
    def daily_floor(self) -> float:
        return self.day_start - self.max_daily_loss

    @property
    def hard_floor(self) -> float:
        return max(self.daily_floor, self.total_floor)

    def breach(self, equity: float) -> str | None:
        if equity <= self.total_floor:
            return "max_loss"
        if equity <= self.daily_floor:
            return "daily_loss"
        return None

    def can_open(self, equity: float) -> tuple[bool, str]:
        if equity <= self.internal_total_floor:
            return False, "internal_total_stop"
        if equity <= self.day_start - self.internal_daily_loss:
            return False, "internal_daily_stop"
        if self.trades_today >= self.max_trades_per_day:
            return False, "max_trades_per_day"
        return True, ""

    def lot_size(self, equity: float, stop_distance: float) -> float:
        """Lots such that a stop-out (plus slippage and commission) loses at most the per-trade
        risk, and can never push equity within `buffer` of an FTMO limit."""
        risk = self.risk_per_trade
        if self.reduce_after and self.loss_streak >= self.reduce_after:
            risk /= 2
        risk = min(risk, equity - self.hard_floor - self.buffer)
        loss_per_lot = (stop_distance + self.slippage) * self.contract + self.commission
        if risk <= 0 or loss_per_lot <= 0:
            return 0.0
        lots = math.floor(risk / loss_per_lot / self.lot_step + 1e-9) * self.lot_step
        lots = min(round(lots, 2), self.max_lots)
        return lots if lots >= self.min_lot else 0.0
