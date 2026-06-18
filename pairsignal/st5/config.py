"""Конфигурация st5 (directional momentum) — pydantic v2, без хардкода.

Один инструмент (фьючерс FORTS). Стратегия внутридневная: directional momentum —
сравнение close с close[-lookback], удержание ровно holding баров, ранний стоп по
stop_pct. Позиция не держится через ночь (принудительный выход к концу сессии).

VWAP-reversion (старая логика) убран: на одиночном фьючерсе эджа не имел (оверфит —
параметры прибыльные на SRU6 теряли на GZU6). Momentum lb48/h18 обобщается на оба
инструмента (бэктест: SRU6 +97, GZU6 +420, оба в плюсе после издержек).

Совместимость: старые session_state_5_*.json содержат VWAP-параметры (band_sigma,
take_profit_sigma, entry_trigger, …). Они сохранены как DEPRECATED-поля (логика их
не читает) — чтобы load_session не падал и /st5/config продолжал принимать их без 500.
extra="ignore" страхует от вовсе незнакомых ключей.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class InstrumentConfig(BaseModel):
    """Инструмент и роллировер квартальной серии."""
    asset: str = "SBRF"                   # ASSETCODE базового актива
    leg_code: str = ""                    # явный SECID серии; пусто → авто-подбор по ISS
    auto_rollover: bool = True
    rollover_days_before_expiry: int = 3  # за сколько дней до экспирации закрывать/роллить
    rollover_no_new_entry_days_before: int = 5


class StrategyConfig(BaseModel):
    """Directional momentum: close vs close[-lookback], удержание holding баров."""
    model_config = ConfigDict(extra="ignore")   # незнакомые ключи старых сессий — молча отбросить

    candle_interval_minutes: int = 10     # MOEX ISS: 5m нет, дефолт 10m

    # --- параметры momentum (бэктест lb48/h18: SRU6 +97, GZU6 +420, оба в плюсе) ---
    lookback: int = 48                    # бар сравнения: signal = sign(close[i] − close[i−lookback])
    holding: int = 18                     # держим ровно столько баров, затем выход по времени
    stop_pct: float = 0.02                # ранний выход: цена против позиции > stop_pct (2%)

    # принудительное закрытие к концу сессии (овернайт-риск) — сохранено из инфраструктуры
    flat_at_session_end: bool = True

    # общая инфраструктура (используется движком/фильтрами/live)
    pending_ttl_bars: int = 3             # TTL неподтверждённой рекомендации (ручной режим)
    # объёмный фильтр входа: bar.volume ≥ volume_filter_mult · SMA(объёма дня). 0 = выкл
    volume_filter_mult: float = 0.0
    max_data_lag_min: float = 0.0         # гейт свежести (только live). 0 = выкл

    # --- DEPRECATED (логика momentum их не читает) ---
    # сохранены, чтобы /st5/config и старые session_state_5_*.json не ломались на присвоении.
    band_sigma: float = 1.5
    min_bars_in_day: int = 6
    std_mode: Literal["Population", "Sample"] = "Population"
    entry_trigger: Literal["Breakout", "ReEntry"] = "Breakout"
    freeze_vwap_on_exit: bool = False
    take_profit_sigma: float = 0.0
    stop_sigma: float = 3.0
    max_bars_in_trade: int = 0


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
