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
    # Спред = P_a − P_b (линейная разница цен одной монеты на двух биржах).
    # SMA(sma_period) — средняя линия, выход к ней. Полуширина полос (band) — два режима:
    #   "vol" (дефолт): band = bb_k·σ(спреда) — адаптивные Боллинджеры. На реальных данных
    #         спред 0.01–0.09%, фикс. процент не пробивается → нужна привязка к волатильности.
    #   "pct": band = band_pct·price_a — фиксированный % от ЦЕНЫ (для синтетики/демо).
    # В обоих режимах z=(spread−mid)/band, |z|=1 ровно на полосе (вход entry_z=1.0).
    band_mode: Literal["pct", "vol"] = "vol"
    band_pct: float = 0.03               # полуширина для band_mode="pct" (доля от цены)
    sma_period: int = 200                # окно SMA и σ спреда (отдельно от bb_period)
    exchange_a: str = "gate"             # биржа ноги A (ccxt ≥4.5.58: "gate", не "gateio")
    exchange_b: str = "mexc"             # биржа ноги B (BitMEX/OKX недоступны с сервера)
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
