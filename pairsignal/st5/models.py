"""Доменные модели st5 (VWAP-reversion на одиночном инструменте). Только структуры данных.

st5 — НЕ парная стратегия: один фьючерс, отклонение цены от внутридневного VWAP.
Position — одна нога (лонг/шорт), не пара. Терминология намеренно отличается от st4.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class BotState(str, Enum):
    """Конечный автомат движка st5."""
    FLAT = "flat"
    ENTERING = "entering"
    LONG = "long"               # купили инструмент (ставка на рост цены к VWAP снизу)
    SHORT = "short"             # продали инструмент (ставка на падение к VWAP сверху)
    HALTED = "halted"           # авария — только ручной разбор


class Signal(str, Enum):
    NONE = "none"
    BUY = "buy"     # лонг: цена ниже нижней полосы VWAP, ждём возврата вверх
    SELL = "sell"   # шорт: цена выше верхней полосы VWAP, ждём возврата вниз
    EXIT = "exit"   # возврат к VWAP / тейк / стоп / конец сессии


@dataclass(slots=True)
class InstrumentSpec:
    """Справочник инструмента: тик, стоимость шага, лот, экспирация."""
    code: str                  # SECID серии, напр. SRM6
    tick_size: float           # MINSTEP
    tick_value_rub: float      # STEPPRICE — рублёвая стоимость шага целого контракта
    lot: int                   # LOTVOLUME
    expiry: Optional[str] = None


@dataclass(slots=True)
class PriceBar:
    """Закрытая свеча инструмента (OHLCV). ts — open_time бара, UTC unix ms."""
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def typical(self) -> float:
        """Типичная цена (H+L+C)/3 — вес в VWAP."""
        return (self.high + self.low + self.close) / 3.0


@dataclass(slots=True)
class VwapReading:
    """Срез VWAP на закрытый бар: VWAP дня + коридор ±k·σ отклонений цены от VWAP."""
    ts: int
    price: float               # close бара
    vwap: float
    sigma: float               # σ отклонений (price − vwap) с начала дня
    upper: float
    lower: float
    is_ready: bool             # накоплено ≥ min_bars баров текущего дня


@dataclass(slots=True)
class Position:
    """Открытая позиция (одна нога)."""
    state: BotState            # LONG | SHORT
    side: str                  # "buy" | "sell"
    lots: int
    entry_price: float
    entry_ts: int
    entry_vwap: float          # VWAP на входе (для freeze-выхода)
    entry_fee_rub: float = 0.0


@dataclass(slots=True)
class Trade:
    """Закрытая сделка (журнал)."""
    state: BotState            # направление: LONG | SHORT
    entry_ts: int
    exit_ts: int
    entry_price: float
    exit_price: float
    lots: int
    gross_pnl_rub: float
    fees_rub: float
    net_pnl_rub: float
    reason: str                # "exit" | "take" | "stop" | "eod" | "time_stop" | "flat_all"
    bars_held: int = 0
    side: str = ""
    slippage_ticks: float = 0.0


@dataclass(slots=True)
class EngineEvent:
    """Событие движка для журнала/UI."""
    ts: int
    kind: str                  # signal | position | exit | halt | warn | info
    message: str
    data: dict = field(default_factory=dict)
