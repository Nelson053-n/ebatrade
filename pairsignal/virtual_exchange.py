"""Виртуальная биржа (paper).

Не ходит на реальную биржу — симулирует исполнение пары ног, считает комиссии,
проскальзывание, нереализованный и реализованный P&L. Это «исполнитель» phase 1.
"""
from __future__ import annotations

import math
from typing import Optional

from .config import PaperConfig
from .models import Leg, PairPosition, SpreadDirection, Trade


class VirtualExchange:
    def __init__(self, cfg: PaperConfig):
        self.cfg = cfg
        self.balance = cfg.start_balance      # реализованный капитал
        self.position: Optional[PairPosition] = None
        self.trades: list[Trade] = []

    # --- helpers ---
    def _fill_price(self, ref: float, side: str) -> float:
        s = self.cfg.slippage_pct
        return ref * (1 + s) if side == "long" else ref * (1 - s)

    def equity(self, price_a: float, price_b: float) -> float:
        return self.balance + self.unrealized(price_a, price_b)

    def unrealized(self, price_a: float, price_b: float) -> float:
        if not self.position:
            return 0.0
        p = self.position
        pnl_a = p.leg_a.qty * (price_a - p.leg_a.entry_price) * (1 if p.leg_a.side == "long" else -1)
        pnl_b = p.leg_b.qty * (price_b - p.leg_b.entry_price) * (1 if p.leg_b.side == "long" else -1)
        return pnl_a + pnl_b

    # --- основные операции ---
    def open_pair(
        self,
        direction: SpreadDirection,
        notional: float,
        price_a: float,
        price_b: float,
        beta: float,
        ts: int,
        entry_z: float,
        symbol_a: str = "A",
        symbol_b: str = "B",
        mid: float = 0.0,
        std: float = 1.0,
    ) -> PairPosition:
        assert self.position is None, "позиция уже открыта"
        beta = max(abs(beta), 1e-6)
        # доллар-нейтральное распределение: notional_b / notional_a = beta
        notional_a = notional / (1.0 + beta)
        notional_b = notional - notional_a

        if direction == SpreadDirection.LONG_SPREAD:
            side_a, side_b = "long", "short"
        else:
            side_a, side_b = "short", "long"

        fa = self._fill_price(price_a, side_a)
        fb = self._fill_price(price_b, side_b)
        leg_a = Leg(symbol_a, side_a, notional_a / fa, fa, notional_a)
        leg_b = Leg(symbol_b, side_b, notional_b / fb, fb, notional_b)

        fee = (notional_a + notional_b) * self.cfg.taker_fee
        self.balance -= fee
        spread0 = math.log(price_a) - beta * math.log(price_b)
        self.position = PairPosition(
            direction, leg_a, leg_b, ts, entry_z, entry_fee=fee,
            beta0=beta, mid0=mid, std0=max(std, 1e-9), spread0=spread0,
        )
        return self.position

    def close_pair(self, price_a: float, price_b: float, ts: int, reason: str, exit_z: float) -> Trade:
        assert self.position is not None, "нет открытой позиции"
        p = self.position
        # выход — закрытие каждой ноги в противоположную сторону
        fa = self._fill_price(price_a, "short" if p.leg_a.side == "long" else "long")
        fb = self._fill_price(price_b, "short" if p.leg_b.side == "long" else "long")

        pnl_a = p.leg_a.qty * (fa - p.leg_a.entry_price) * (1 if p.leg_a.side == "long" else -1)
        pnl_b = p.leg_b.qty * (fb - p.leg_b.entry_price) * (1 if p.leg_b.side == "long" else -1)
        gross = pnl_a + pnl_b

        exit_notional = p.leg_a.qty * fa + p.leg_b.qty * fb
        exit_fee = exit_notional * self.cfg.taker_fee
        fees = p.entry_fee + exit_fee
        net = gross - exit_fee  # entry_fee уже списан при открытии

        self.balance += gross - exit_fee
        trade = Trade(
            direction=p.direction,
            entry_ts=p.entry_ts,
            exit_ts=ts,
            entry_z=p.entry_z,
            exit_z=exit_z,
            notional=p.leg_a.notional + p.leg_b.notional,
            gross_pnl=gross,
            fees=fees,
            net_pnl=net,
            reason=reason,
            a_side=p.leg_a.side, b_side=p.leg_b.side,
            a_entry=p.leg_a.entry_price, a_exit=fa,
            b_entry=p.leg_b.entry_price, b_exit=fb,
            beta0=p.beta0,
            spread_entry=p.spread0,
            spread_exit=math.log(price_a) - p.beta0 * math.log(price_b),
        )
        self.trades.append(trade)
        self.position = None
        return trade
