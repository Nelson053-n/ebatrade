"""Оркестратор: связывает данные → индикаторы → сигнал → (подтверждение) → виртуальная биржа.

Вход требует approve() от оператора (human-in-the-loop). Выход/стоп исполняются авто.
Класс не зависит от способа доставки данных — ему скармливают готовые IndicatorRow.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .config import AppConfig
from .indicators import build_indicators
from .models import Action, IndicatorRow, Recommendation, Trade
from .strategy import SignalEngine
from .virtual_exchange import VirtualExchange


@dataclass
class StepResult:
    rec: Recommendation
    trade: Optional[Trade] = None      # появляется при закрытии
    awaiting_approval: bool = False    # ждём решения оператора по входу


class Engine:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.signal = SignalEngine(cfg.strategy)
        self.exch = VirtualExchange(cfg.paper)
        self.pending: Optional[Recommendation] = None
        self._bars_held = 0
        self._last_row: Optional[IndicatorRow] = None

    @staticmethod
    def rows_from_df(df: pd.DataFrame, cfg) -> list[IndicatorRow]:
        ind = build_indicators(df, cfg)
        rows: list[IndicatorRow] = []
        for ts, r in ind.iterrows():
            rows.append(IndicatorRow(
                ts=int(ts), price_a=r["price_a"], price_b=r["price_b"], spread=r["spread"],
                beta=r["beta"], mid=r["mid"], upper=r["upper"], lower=r["lower"],
                std=r["std"], z=r["z"], width_pct=r["width_pct"],
            ))
        return rows

    def step(self, row: IndicatorRow) -> StepResult:
        self._last_row = row
        if self.exch.position is not None:
            self._bars_held += 1

        rec = self.signal.evaluate(row, self.exch.position, self._bars_held)

        if rec.action in (Action.EXIT, Action.STOP):
            trade = self.exch.close_pair(row.price_a, row.price_b, row.ts, rec.action.value, row.z)
            self._bars_held = 0
            self.pending = None
            return StepResult(rec=rec, trade=trade)

        if rec.action == Action.ENTER:
            self.pending = rec
            if self.cfg.auto_approve:
                self.approve()
                return StepResult(rec=rec)
            return StepResult(rec=rec, awaiting_approval=True)

        return StepResult(rec=rec)

    def approve(self) -> None:
        """Оператор подтверждает вход по последней рекомендации."""
        if not self.pending or self.exch.position is not None:
            return
        rec = self.pending
        notional = self.exch.balance * (self.cfg.paper.risk_pct / 100.0)
        # σ спреда на входе: (upper − mid)/bb_k — фиксируем вместе с β и mid,
        # чтобы z открытой позиции считался по неизменной базе
        std = (rec.upper - rec.mid) / self.cfg.strategy.bb_k if self.cfg.strategy.bb_k else 1.0
        self.exch.open_pair(
            direction=rec.direction, notional=notional,
            price_a=rec.price_a, price_b=rec.price_b, beta=rec.beta,
            ts=rec.ts, entry_z=rec.z,
            symbol_a=self.cfg.strategy.symbol_a, symbol_b=self.cfg.strategy.symbol_b,
            mid=rec.mid, std=std,
        )
        self._bars_held = 0
        self.pending = None

    def reject(self) -> None:
        self.pending = None

    # --- сводка ---
    def summary(self) -> dict:
        trades = self.exch.trades
        wins = [t for t in trades if t.net_pnl > 0]
        eq = self.exch.equity(
            self._last_row.price_a if self._last_row else 0,
            self._last_row.price_b if self._last_row else 0,
        )
        net = sum(t.net_pnl for t in trades)
        return {
            "trades": len(trades),
            "win_rate_pct": round(100 * len(wins) / len(trades), 1) if trades else 0.0,
            "net_pnl": round(net, 2),
            "fees_paid": round(sum(t.fees for t in trades), 2),
            "balance": round(self.exch.balance, 2),
            "equity": round(eq, 2),
            "return_pct": round(100 * (eq - self.cfg.paper.start_balance) / self.cfg.paper.start_balance, 2),
        }
