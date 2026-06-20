"""Сигнальная логика st5: directional momentum → вход по тренду, выход по времени/стопу.

Чистые функции. Не знают про позиции/исполнение. Сигнал входа — только на закрытии бара
и только когда индикатор готов (накоплено > lookback закрытых баров).
"""
from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from .config import SessionConfig, StrategyConfig
from .models import BotState, MomentumReading, Position, Signal, ZScoreReading


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


def entry_signal_mr(cur: ZScoreReading, cfg: StrategyConfig) -> Signal:
    """Сигнал входа mean-reversion по z-score (фейдим отклонение от средней).

    z ≤ −entry_z (но не за стопом z > −stop_z) → BUY (ждём возврата вверх).
    z ≥ +entry_z (но не за стопом z < +stop_z) → SELL.
    Иначе или индикатор не готов → нет сигнала. (Сессионный фильтр — отдельно, в движке.)
    """
    if not cur.is_ready:
        return Signal.NONE
    z = cur.z
    if z <= -cfg.mr_entry_z and z > -cfg.mr_stop_z:
        return Signal.BUY
    if z >= cfg.mr_entry_z and z < cfg.mr_stop_z:
        return Signal.SELL
    return Signal.NONE


def exit_signal_mr(pos: Position, z: float, bars_held: int,
                   cfg: StrategyConfig) -> tuple[bool, str]:
    """Условие выхода mean-reversion (по закрытому бару). Приоритет: TP → стоп → время.

    LONG  (ставка на рост z к 0): |z| вернулся → z ≥ −exit_z → TP; ушёл глубже → z ≤ −stop_z → стоп.
    SHORT (ставка на падение z):  z ≤ +exit_z → TP; z ≥ +stop_z → стоп.
    Затем bars_held ≥ max_hold → time. Совпадает с research.meanrev_z.
    """
    if pos.state == BotState.LONG:
        if z >= -cfg.mr_exit_z:
            return True, "take"
        if z <= -cfg.mr_stop_z:
            return True, "stop"
    else:  # SHORT
        if z <= cfg.mr_exit_z:
            return True, "take"
        if z >= cfg.mr_stop_z:
            return True, "stop"
    if cfg.mr_max_hold > 0 and bars_held >= cfg.mr_max_hold:
        return True, "time_stop"
    return False, ""


def in_session_window(ts_ms: int, cfg: StrategyConfig,
                      session: SessionConfig) -> bool:
    """Попадает ли момент в окно входа основной сессии [session_lo_min; session_hi_min) MSK.

    Mean-reversion входит только в основную сессию (research: вечерняя/ночная вредит GAZR).
    Минуты считаются в TZ сессии (как in_clearing_window/is_session_end).
    """
    try:
        local = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).astimezone(
            ZoneInfo(session.timezone))
    except Exception:  # noqa: BLE001
        return False
    minutes = local.hour * 60 + local.minute
    return cfg.session_lo_min <= minutes < cfg.session_hi_min


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
