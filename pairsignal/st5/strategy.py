"""Сигнальная логика st5: directional momentum → вход по тренду, выход по времени/стопу.

Чистые функции. Не знают про позиции/исполнение. Сигнал входа — только на закрытии бара
и только когда индикатор готов (накоплено > lookback закрытых баров).
"""
from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from .config import SessionConfig
from .models import BotState, MomentumReading, Position, Signal


def entry_signal(cur: MomentumReading) -> Signal:
    """Сигнал входа по направлению momentum.

    close > close[-lookback] (signal +1) → BUY (вход в тренд вверх).
    close < close[-lookback] (signal −1) → SELL (вход в тренд вниз).
    Равенство (signal 0) или индикатор не готов → нет сигнала.
    """
    if not cur.is_ready:
        return Signal.NONE
    if cur.signal > 0:
        return Signal.BUY
    if cur.signal < 0:
        return Signal.SELL
    return Signal.NONE


def exit_signal(pos: Position, bars_held: int, price: float, holding: int,
                stop_pct: float) -> tuple[bool, str]:
    """Условие выхода из позиции (по закрытому бару).

    Возвращает (выходить?, причина). Приоритет: стоп → holding.
      stop: цена ушла ПРОТИВ позиции более чем на stop_pct от цены входа.
        LONG  — убыток при падении: (entry − price)/entry > stop_pct.
        SHORT — убыток при росте:   (price − entry)/entry > stop_pct.
      time: держим ровно holding баров (bars_held >= holding) → выход по времени.
    """
    entry = pos.entry_price
    if stop_pct > 0 and entry > 0:
        if pos.state == BotState.LONG and (entry - price) / entry > stop_pct:
            return True, "stop"
        if pos.state == BotState.SHORT and (price - entry) / entry > stop_pct:
            return True, "stop"
    if holding > 0 and bars_held >= holding:
        return True, "time_stop"
    return False, ""


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
