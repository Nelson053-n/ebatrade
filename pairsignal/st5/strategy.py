"""Сигнальная логика st5: пробой коридора VWAP → вход, возврат к VWAP → выход.

Чистые функции от (prev, cur) → сигнал. Не знают про позиции/исполнение. Сигналы — только
на закрытии бара и только когда VWAP.is_ready (накоплено ≥ min_bars дня).
"""
from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from .config import SessionConfig, StrategyConfig
from .models import BotState, Signal, VwapReading


def entry_signal(prev: VwapReading, cur: VwapReading, cfg: StrategyConfig) -> Signal:
    """Сигнал входа на возврат к VWAP.

    Breakout: цена пробила коридор НАРУЖУ → ставка на возврат.
      BUY: цена ниже нижней полосы (prev > lower, cur <= lower) — ждём роста к VWAP.
      SELL: цена выше верхней полосы (prev < upper, cur >= upper) — ждём падения к VWAP.
    ReEntry: цена была снаружи и ВЕРНУЛАСЬ в коридор (защита от трендового пробоя).
    """
    if not (cur.is_ready and prev.is_ready):
        return Signal.NONE
    if cfg.entry_trigger == "ReEntry":
        # была выше верхней, вернулась внутрь → SELL (ждём дальнейшего возврата к VWAP)
        if prev.price >= prev.upper and cur.price < cur.upper:
            return Signal.SELL
        if prev.price <= prev.lower and cur.price > cur.lower:
            return Signal.BUY
        return Signal.NONE
    # Breakout
    if prev.price < prev.upper and cur.price >= cur.upper:
        return Signal.SELL
    if prev.price > prev.lower and cur.price <= cur.lower:
        return Signal.BUY
    return Signal.NONE


def exit_signal(state: BotState, prev: VwapReading, cur: VwapReading,
                vwap_level: float) -> bool:
    """Выход по пересечению VWAP. vwap_level — живой VWAP или зафиксированный на входе.

    LONG (вошли снизу): цена пересекла VWAP снизу вверх (prev < VWAP, cur >= VWAP).
    SHORT (вошли сверху): цена пересекла VWAP сверху вниз (prev > VWAP, cur <= VWAP).
    """
    if state == BotState.LONG:
        return prev.price < vwap_level and cur.price >= vwap_level
    if state == BotState.SHORT:
        return prev.price > vwap_level and cur.price <= vwap_level
    return False


def in_clearing_window(ts_ms: int, cfg: SessionConfig) -> bool:
    """Попадает ли момент в клиринговое окно/аукцион. Время — в TZ сессии."""
    if not cfg.skip_clearing_windows:
        return False
    try:
        local = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).astimezone(
            ZoneInfo(cfg.timezone))
    except Exception:  # noqa: BLE001
        return False
    minutes = local.hour * 60 + local.minute
    return any(lo <= minutes < hi for lo, hi in cfg.clearing_windows)


def is_session_end(ts_ms: int, cfg: SessionConfig) -> bool:
    """Достигнут ли конец сессии (для принудительного закрытия позиции — овернайт-риск)."""
    try:
        local = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).astimezone(
            ZoneInfo(cfg.timezone))
    except Exception:  # noqa: BLE001
        return False
    return local.hour * 60 + local.minute >= cfg.session_end_minute
