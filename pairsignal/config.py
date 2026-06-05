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
    spread_mode: Literal["ratio", "log", "cross_pct"] = "log"
    beta_window: int = 240               # окно rolling-OLS для динамической беты (log-режим)

    # --- кросс-биржевой режим (cross_pct): один инструмент на двух биржах ---
    # Спред = P_a − P_b (линейная разница цен одной монеты на двух биржах). Полосы —
    # фиксированный % от ЦЕНЫ (band_pct·price_a), НЕ от σ и НЕ от |SMA| (SMA спреда ≈ 0,
    # процент от неё схлопывается). SMA(sma_period) — средняя линия, выход к ней.
    band_pct: float = 0.03               # полуширина коридора, доля от цены (±3%)
    sma_period: int = 200                # окно SMA спреда (отдельно от bb_period)
    exchange_a: str = "bitmex"           # биржа ноги A (для read_ohlcv_cross)
    exchange_b: str = "okx"              # биржа ноги B
    symbol_cross: str = "SUI/USDT:USDT"  # единый символ на обеих биржах

    # --- индикатор (Боллинджер поверх спреда) ---
    # пресет КОНСЕРВАТИВНЫЙ — лучше обобщается на всех 9 парах (бэктест 6 мес: +66,
    # 6/9 прибыльных пар), тогда как «сбалансированный» переобучился на 4 парах.
    bb_period: int = 240
    bb_k: float = 2.0

    # --- пороги сигналов (в единицах z-score) ---
    entry_z: float = 2.5                 # вход при |z| >= entry_z
    exit_z: float = 0.2                  # (не используется при profit-target выходе)
    stop_z: float = 8.0                  # стоп при |z| >= stop_z (шире = меньше ложных стоп-аутов)
    min_width_pct: float = 0.5           # анти-флэт фильтр: полуширина канала, %
    max_bars_in_trade: int = 576         # тайм-стоп
    allow_short_spread: bool = True      # разрешить и обратную сторону
    # выход по ЦЕЛИ ПРИБЫЛИ: закрываем по возврату, когда нереализованный gross достиг
    # profit_target_fees × (round-trip комиссий). Гарантирует gross ≥ 0 на выходе,
    # устраняя «z вернулся, но P&L < 0» из-за дрейфа β и скользящей средней.
    profit_target_fees: float = 6.0      # цель = 6× round-trip комиссий (win 73%, out-of-sample подтв.)


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
