"""Конфигурация st5 (VWAP-reversion) — pydantic v2, без хардкода.

Один инструмент (фьючерс FORTS). Стратегия внутридневная: VWAP сбрасывается каждый
торговый день, позиция не держится через ночь (принудительный выход к концу сессии).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class InstrumentConfig(BaseModel):
    """Инструмент и роллировер квартальной серии."""
    asset: str = "SBRF"                   # ASSETCODE базового актива
    leg_code: str = ""                    # явный SECID серии; пусто → авто-подбор по ISS
    auto_rollover: bool = True
    rollover_days_before_expiry: int = 3  # за сколько дней до экспирации закрывать/роллить
    rollover_no_new_entry_days_before: int = 5


class StrategyConfig(BaseModel):
    """VWAP-reversion: коридор ±k·σ вокруг внутридневного VWAP."""
    candle_interval_minutes: int = 10     # MOEX ISS: 5m нет, дефолт 10m
    # полуширина коридора в σ. Дефолт 1.5 по гриду 30д SRM6 (14.06.2026): band1.5 дал
    # лучший результат (win 63%, net +524₽) против band2.0 (51%, −96₽) — узкий коридор
    # ловит больше возвратов. ВНИМАНИЕ: VWAP-reversion на одиночном фьючерсе — слабый эдж
    # (net около нуля), в отличие от коинтегрированного спреда st4; нужен форвард-контроль.
    band_sigma: float = 1.5
    min_bars_in_day: int = 6              # минимум баров дня для готовности VWAP (анти-шум утра)
    std_mode: Literal["Population", "Sample"] = "Population"
    # вход: Breakout — пробой коридора наружу; ReEntry — возврат в коридор (защита от тренда)
    entry_trigger: Literal["Breakout", "ReEntry"] = "Breakout"
    # выход к VWAP: живой VWAP (дрейфует за день) или зафиксированный на входе
    freeze_vwap_on_exit: bool = False
    # тейк-профит: закрыть, когда цена вернулась к VWAP на take_profit_sigma·σ ВНУТРИ коридора
    take_profit_sigma: float = 0.0        # 0 = выкл (выход по пересечению VWAP)
    # защитный стоп: цена ушла против позиции дальше stop_sigma·σ от VWAP
    stop_sigma: float = 3.0
    max_bars_in_trade: int = 0            # тайм-стоп (0 = выкл)
    pending_ttl_bars: int = 3             # TTL неподтверждённой рекомендации (ручной режим)
    # объёмный фильтр входа: bar.volume ≥ volume_filter_mult · SMA(объёма дня). 0 = выкл
    volume_filter_mult: float = 0.0
    max_data_lag_min: float = 0.0         # гейт свежести (только live). 0 = выкл
    flat_at_session_end: bool = True      # принудительное закрытие к концу сессии (овернайт-риск)


class ExecutionConfig(BaseModel):
    """Исполнение одиночного ордера (paper-модель)."""
    entry_style: Literal["MarketableLimit", "Passive"] = "MarketableLimit"
    tick_offset: int = 1                  # ± N тиков от лучшего бид/аск
    quantity_lots: int = 1
    paper_book_halfspread_ticks: float = 1.0   # полуширина paper-стакана (см. st4)
    deviation_protection_ticks: int = 5


class RiskConfig(BaseModel):
    max_daily_loss_rub: float = 50_000.0
    max_consecutive_errors: int = 5
    trading_enabled: bool = True


class SessionConfig(BaseModel):
    """Торговая сессия FORTS."""
    timezone: str = "Europe/Moscow"
    skip_clearing_windows: bool = True
    clearing_windows: list[tuple[int, int]] = Field(
        default_factory=lambda: [(14 * 60, 14 * 60 + 5), (18 * 60 + 45, 19 * 60 + 5)]
    )
    # конец основной/вечерней сессии (мин от полуночи MSK) — для flat_at_session_end.
    # FORTS вечерняя до 23:50; закрываем за бар до конца.
    session_end_minute: int = 23 * 60 + 40


class Paper(BaseModel):
    start_balance_rub: float = 1_000_000.0
    taker_fee_rub_per_lot: float = 1.0    # сбор за лот за сделку (одна нога, не пара)


class St5Config(BaseModel):
    instrument: InstrumentConfig = Field(default_factory=InstrumentConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    paper: Paper = Field(default_factory=Paper)
    poll_seconds: int = 20
    auto_approve: bool = True
