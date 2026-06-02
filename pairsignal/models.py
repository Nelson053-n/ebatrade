"""Доменные модели: перечисления и dataclass'ы.

Никакой логики — только структуры данных, которыми обмениваются слои.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SpreadDirection(str, Enum):
    """Направление парной позиции относительно спреда."""
    LONG_SPREAD = "long_spread"    # лонг BTC + шорт ETH (ставка: спред вырастет)
    SHORT_SPREAD = "short_spread"  # шорт BTC + лонг ETH (ставка: спред упадёт)


class Action(str, Enum):
    NONE = "none"
    ENTER = "enter"   # рекомендация на вход (требует подтверждения оператора)
    EXIT = "exit"     # выход по возврату к средней (авто)
    STOP = "stop"     # выход по стопу z / времени (авто)


@dataclass(slots=True)
class IndicatorRow:
    """Срез индикаторов на момент закрытия свечи."""
    ts: int               # unix ms
    price_a: float        # цена первого инструмента (BTC)
    price_b: float        # цена второго инструмента (ETH)
    spread: float
    beta: float
    mid: float
    upper: float
    lower: float
    std: float
    z: float
    width_pct: float      # относительная полуширина канала, %


@dataclass(slots=True)
class Recommendation:
    """Сигнал/рекомендация, которую система отдаёт оператору или исполнителю."""
    ts: int
    action: Action
    reason: str
    direction: Optional[SpreadDirection] = None
    z: float = 0.0
    spread: float = 0.0
    mid: float = 0.0
    upper: float = 0.0
    lower: float = 0.0
    width_pct: float = 0.0
    beta: float = 1.0
    price_a: float = 0.0
    price_b: float = 0.0


@dataclass(slots=True)
class Leg:
    symbol: str
    side: str          # "long" | "short"
    qty: float         # в единицах базового актива
    entry_price: float
    notional: float    # USDT на момент входа


@dataclass(slots=True)
class PairPosition:
    direction: SpreadDirection
    leg_a: Leg
    leg_b: Leg
    entry_ts: int
    entry_z: float
    entry_fee: float = 0.0
    # параметры спреда, зафиксированные на входе — по ним считаем z для управления
    # позицией (иначе дрейф β/средней даёт ложные сигналы возврата к среднему)
    beta0: float = 1.0      # β в момент входа
    mid0: float = 0.0       # средняя спреда (BB) в момент входа
    std0: float = 1.0       # σ спреда в момент входа
    spread0: float = 0.0    # спред (ln A − β0·ln B) в момент входа — уровень для выхода


@dataclass(slots=True)
class Trade:
    """Закрытая виртуальная сделка."""
    direction: SpreadDirection
    entry_ts: int
    exit_ts: int
    entry_z: float
    exit_z: float
    notional: float
    gross_pnl: float
    fees: float
    net_pnl: float
    reason: str        # "exit" | "stop"
    bars_held: int = 0
    # цены исполнения ног (вход/выход) — «по чём купил/продал»
    a_side: str = ""           # сторона ноги A: "long" | "short"
    b_side: str = ""
    a_entry: float = 0.0       # цена входа ноги A
    a_exit: float = 0.0        # цена выхода ноги A
    b_entry: float = 0.0
    b_exit: float = 0.0
    # спред (с β входа) в моменты входа/выхода — для отметок на графике точно на уровне
    beta0: float = 1.0
    spread_entry: float = 0.0
    spread_exit: float = 0.0
