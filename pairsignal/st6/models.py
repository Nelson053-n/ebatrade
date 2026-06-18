"""Доменные модели st6 — только структуры данных для paper-учёта.

FSM и сайзинг живут в core.py (Side/Position/Signal/decide/leg_quantities).
Здесь — журнал закрытых сделок пары и события для UI; вся «торговая» логика
вынесена в ядро, чтобы переиспользоваться в бэктесте и тестах.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class PairTrade:
    """Закрытая сделка пары (журнал paper-исполнения)."""
    side: str                  # "LONG_SPREAD" | "SHORT_SPREAD"
    ticker_a: str
    ticker_b: str
    entry_ts: int
    exit_ts: int
    entry_a: float
    exit_a: float
    entry_b: float
    exit_b: float
    qty_a: int                 # лотов ноги A
    qty_b: int                 # лотов ноги B
    beta: float
    entry_z: float
    exit_z: float
    bars_held: int
    net_pnl_rub: float
    reason: str                # "target" | "stop_z" | "corr_break" | "time_stop"


@dataclass(slots=True)
class EngineEvent:
    """Событие для журнала/UI."""
    ts: float
    kind: str                  # signal | position | exit | select | warn | info
    message: str
    data: dict = field(default_factory=dict)
