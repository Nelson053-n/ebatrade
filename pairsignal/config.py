"""Конфигурация системы (pydantic v2). Без хардкода — всё параметризуется."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class StrategyConfig(BaseModel):
    # --- инструменты и данные ---
    symbol_a: str = "XRP/USDT:USDT"     # формат CCXT для перпетуала
    symbol_b: str = "ADA/USDT:USDT"
    data_exchange: str = "mexc"          # биржа ТОЛЬКО для чтения котировок
    timeframe: str = "5m"

    # --- расчёт спреда ---
    spread_mode: Literal["ratio", "log"] = "log"
    beta_window: int = 240               # окно rolling-OLS для динамической беты (log-режим)

    # --- индикатор (Боллинджер поверх спреда) ---
    # пресет КОНСЕРВАТИВНЫЙ — лучше обобщается на всех 9 парах (бэктест 6 мес: +66,
    # 6/9 прибыльных пар), тогда как «сбалансированный» переобучился на 4 парах.
    bb_period: int = 240
    bb_k: float = 2.0

    # --- пороги сигналов (в единицах z-score) ---
    entry_z: float = 2.5                 # вход при |z| >= entry_z
    exit_z: float = 0.2                  # (не используется при profit-target выходе)
    stop_z: float = 5.0                  # стоп при |z| >= stop_z (движение против)
    min_width_pct: float = 0.5           # анти-флэт фильтр: полуширина канала, %
    max_bars_in_trade: int = 576         # тайм-стоп
    allow_short_spread: bool = True      # разрешить и обратную сторону
    # выход по ЦЕЛИ ПРИБЫЛИ: закрываем по возврату, когда нереализованный gross достиг
    # profit_target_fees × (round-trip комиссий). Гарантирует gross ≥ 0 на выходе,
    # устраняя «z вернулся, но P&L < 0» из-за дрейфа β и скользящей средней.
    profit_target_fees: float = 8.0      # цель = 8× round-trip комиссий


class PaperConfig(BaseModel):
    start_balance: float = 10_000.0      # стартовый виртуальный баланс USDT
    risk_pct: float = 2.0                # нотационал позиции, % от баланса
    taker_fee: float = 0.0006            # 0.06% за ногу
    slippage_pct: float = 0.0002         # 0.02% проскальзывание на исполнение


class AppConfig(BaseModel):
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    paper: PaperConfig = Field(default_factory=PaperConfig)
    poll_seconds: int = 15               # период опроса в live-режиме
    auto_approve: bool = False           # human-in-the-loop по умолчанию
