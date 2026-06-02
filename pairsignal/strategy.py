"""Сигнальный движок.

Чистая логика: по текущему срезу индикаторов и наличию позиции возвращает
рекомендацию. Вход — только рекомендация (решает оператор), выход/стоп — авто.
"""
from __future__ import annotations

import math
from typing import Optional

from .config import StrategyConfig
from .models import Action, IndicatorRow, PairPosition, Recommendation, SpreadDirection


class SignalEngine:
    def __init__(self, cfg: StrategyConfig, taker_fee: float = 0.0006):
        self.cfg = cfg
        self._taker_fee = taker_fee   # для расчёта цели прибыли (round-trip издержки)

    @staticmethod
    def _valid(row: IndicatorRow) -> bool:
        vals = [row.z, row.mid, row.std, row.width_pct, row.beta]
        return all(v is not None and not math.isnan(v) for v in vals) and row.std > 0

    def evaluate(
        self, row: IndicatorRow, position: Optional[PairPosition], bars_held: int = 0,
        unrealized_gross: float = 0.0, notional: float = 0.0,
    ) -> Recommendation:
        c = self.cfg
        base = dict(
            ts=row.ts, z=row.z, spread=row.spread, mid=row.mid,
            upper=row.upper, lower=row.lower, width_pct=row.width_pct,
            beta=row.beta, price_a=row.price_a, price_b=row.price_b,
        )
        if not self._valid(row):
            return Recommendation(action=Action.NONE, reason="прогрев индикаторов", **base)

        # --- управление открытой позицией (авто) ---
        if position is not None:
            d = position.direction
            # ВЫХОД по ЦЕЛИ ПРИБЫЛИ. Закрываем по возврату, когда нереализованный валовый
            # P&L (по ногам, в β входа) достиг цели = profit_target_fees × round-trip
            # комиссий. Это гарантирует gross ≥ 0 на выходе — устраняет «z вернулся, а
            # P&L < 0» из-за дрейфа β и уехавшей скользящей средней (обе причины обходятся,
            # т.к. решение принимается по реальному P&L позиции, а не по z спреда).
            target = c.profit_target_fees * (2.0 * self._taker_fee) * notional
            reverted = unrealized_gross >= target
            # стоп по z (защита от ухода против позиции) — без изменений
            z = row.z
            stopped = (d == SpreadDirection.LONG_SPREAD and z <= -c.stop_z) or (
                d == SpreadDirection.SHORT_SPREAD and z >= c.stop_z
            )
            if reverted:
                return Recommendation(action=Action.EXIT, reason="цель прибыли", direction=d, **base)
            if stopped:
                return Recommendation(action=Action.STOP, reason="стоп по z", direction=d, **base)
            if bars_held >= c.max_bars_in_trade:
                return Recommendation(action=Action.STOP, reason="тайм-стоп", direction=d, **base)
            return Recommendation(action=Action.NONE, reason="удержание", direction=d, **base)

        # --- поиск входа (рекомендация оператору) ---
        if row.width_pct < c.min_width_pct:
            return Recommendation(action=Action.NONE, reason=f"флэт: канал {row.width_pct:.2f}% < {c.min_width_pct}%", **base)

        if -c.stop_z < row.z <= -c.entry_z:
            return Recommendation(
                action=Action.ENTER, direction=SpreadDirection.LONG_SPREAD,
                reason=f"спред у нижней полосы (z={row.z:.2f}) → лонг {c.symbol_a} / шорт {c.symbol_b}", **base,
            )
        if c.allow_short_spread and c.entry_z <= row.z < c.stop_z:
            return Recommendation(
                action=Action.ENTER, direction=SpreadDirection.SHORT_SPREAD,
                reason=f"спред у верхней полосы (z={row.z:.2f}) → шорт {c.symbol_a} / лонг {c.symbol_b}", **base,
            )
        return Recommendation(action=Action.NONE, reason=f"нет сигнала (z={row.z:.2f})", **base)
