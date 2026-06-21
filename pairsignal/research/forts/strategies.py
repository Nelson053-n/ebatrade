"""Семейства стратегий для FORTS-фьючерсов. Каждая возвращает (pos, reason) ряды:
pos[i] in {-1,0,+1} — желаемая позиция НА ЗАКРЫТИИ бара i (исполняется по close[i]).

NO LOOK-AHEAD: все индикаторы считаются по close[..i] (текущий закрытый бар включительно);
решение о позиции на баре i использует только бары <= i. Вход реально исполняется по
close[i] (консервативно — реально по open[i+1], разница в проскальзывании).

Состояние позиции (holding/stop/TP) ведётся явным проходом, чтобы выходы были корректны
по приоритету и без заглядывания вперёд.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _atr(high, low, close, n: int) -> np.ndarray:
    """ATR по закрытым барам (Wilder-подобный SMA TR). Сдвинут — на баре i доступен ATR[i]."""
    high = np.asarray(high, float); low = np.asarray(low, float); close = np.asarray(close, float)
    prev_close = np.roll(close, 1); prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = pd.Series(tr).rolling(n, min_periods=n).mean().to_numpy()
    return atr


def _session_minutes(ts_ms: np.ndarray) -> np.ndarray:
    """Минуты от полуночи MSK для каждого бара (для сессионного фильтра)."""
    # ts уже UTC ms; MSK = UTC+3
    return ((ts_ms // 60000 + 180) % 1440).astype(int)


def momentum_atr(df: pd.DataFrame, lookback: int, holding: int,
                 atr_n: int = 14, atr_stop: float = 2.0, atr_tp: float = 0.0,
                 vol_floor: float = 0.0, time_lo: int = 0, time_hi: int = 1440):
    """Time-series momentum + ATR-стоп + опц. ATR-тейк + сессионный/вол фильтр.

    Вход: sign(close[i]-close[i-lookback]) в открытое окно времени.
    Выход (приоритет): ATR-стоп -> ATR-TP -> holding баров.
    vol_floor: минимальный ATR (в тиках) для входа (отсекаем мёртвые периоды). 0=выкл.
    time_lo/time_hi: окно входа в минутах MSK (фильтр клиринга/краёв). [lo,hi).
    """
    close = df["close"].to_numpy(float)
    high = df["high"].to_numpy(float); low = df["low"].to_numpy(float)
    ts = df.index.to_numpy()
    n = len(close)
    atr = _atr(high, low, close, atr_n)
    mins = _session_minutes(ts)

    pos = np.zeros(n, int); reason = np.array([""] * n, object)
    cur = 0; entry_px = 0.0; held = 0; entry_atr = 0.0
    for i in range(n):
        if cur != 0:
            held += 1
            exit_now = ""
            # ATR-стоп
            if atr_stop > 0 and entry_atr > 0:
                adverse = (entry_px - close[i]) if cur > 0 else (close[i] - entry_px)
                if adverse > atr_stop * entry_atr:
                    exit_now = "stop"
            # ATR take-profit
            if not exit_now and atr_tp > 0 and entry_atr > 0:
                favor = (close[i] - entry_px) if cur > 0 else (entry_px - close[i])
                if favor > atr_tp * entry_atr:
                    exit_now = "tp"
            if not exit_now and holding > 0 and held >= holding:
                exit_now = "time"
            if exit_now:
                pos[i] = 0; reason[i] = exit_now; cur = 0; held = 0
                continue
            pos[i] = cur
            continue
        # flat: ищем вход
        if i < lookback or np.isnan(atr[i]):
            continue
        if not (time_lo <= mins[i] < time_hi):
            continue
        if vol_floor > 0 and atr[i] < vol_floor:
            continue
        sig = 1 if close[i] > close[i - lookback] else (-1 if close[i] < close[i - lookback] else 0)
        if sig != 0:
            cur = sig; entry_px = close[i]; held = 0; entry_atr = atr[i]
            pos[i] = sig; reason[i] = "entry"
    return pos, reason


def meanrev_z(df: pd.DataFrame, ma_n: int, entry_z: float, exit_z: float,
              stop_z: float, max_hold: int, atr_n: int = 14,
              time_lo: int = 0, time_hi: int = 1440,
              trend_n: int = 0, trend_max: float = 0.0):
    """Внутридневной mean-reversion по z-score отклонения close от SMA(ma_n).

    z = (close - SMA)/rolling_std. Вход: z<=-entry_z -> long (ждём возврата вверх);
    z>=+entry_z -> short. Выход: |z|<=exit_z (возврат к среднему) -> прибыль;
    |z|>=stop_z в ту же сторону -> стоп; max_hold баров -> тайм-аут.
    Все скользящие по закрытым барам (включая текущий i).
    """
    close = df["close"].to_numpy(float)
    n = len(close)
    s = pd.Series(close)
    ma = s.rolling(ma_n, min_periods=ma_n).mean().to_numpy()
    sd = s.rolling(ma_n, min_periods=ma_n).std(ddof=0).to_numpy()
    ts = df.index.to_numpy(); mins = _session_minutes(ts)
    z = np.full(n, np.nan)
    valid = (~np.isnan(ma)) & (sd > 0)
    z[valid] = (close[valid] - ma[valid]) / sd[valid]

    # трендовый гейт: не фейдить, если |долгосрочный momentum| высок. trend = |close-close[-trend_n]|
    # нормированный на цену (доля). trend_max>0 включает фильтр (вход запрещён при trend>trend_max).
    if trend_n > 0 and trend_max > 0:
        ref = np.roll(close, trend_n)
        trend = np.abs(close - ref) / np.where(ref != 0, ref, np.nan)
        trend[:trend_n] = np.inf  # пока нет истории — блокируем
    else:
        trend = np.zeros(n)

    pos = np.zeros(n, int); reason = np.array([""] * n, object)
    cur = 0; held = 0
    for i in range(n):
        if np.isnan(z[i]):
            continue
        if cur != 0:
            held += 1
            exit_now = ""
            if cur > 0:  # long, ждём роста z к 0
                if z[i] >= -exit_z:
                    exit_now = "tp"
                elif z[i] <= -stop_z:
                    exit_now = "stop"
            else:        # short
                if z[i] <= exit_z:
                    exit_now = "tp"
                elif z[i] >= stop_z:
                    exit_now = "stop"
            if not exit_now and max_hold > 0 and held >= max_hold:
                exit_now = "time"
            if exit_now:
                pos[i] = 0; reason[i] = exit_now; cur = 0; held = 0
                continue
            pos[i] = cur
            continue
        if not (time_lo <= mins[i] < time_hi):
            continue
        if trend_max > 0 and trend[i] > trend_max:
            continue  # сильный тренд — не фейдим
        # вход против отклонения (но не за стопом)
        if z[i] <= -entry_z and z[i] > -stop_z:
            cur = 1; held = 0; pos[i] = 1; reason[i] = "entry"
        elif z[i] >= entry_z and z[i] < stop_z:
            cur = -1; held = 0; pos[i] = -1; reason[i] = "entry"
    return pos, reason


def donchian_breakout(df: pd.DataFrame, channel: int, exit_channel: int,
                      atr_n: int = 14, atr_stop: float = 2.0, max_hold: int = 0,
                      time_lo: int = 0, time_hi: int = 1440):
    """Брейкаут канала Дончяна: вход при пробое экстремума прошлых `channel` баров.

    Long: close[i] > max(close[i-channel..i-1]); short: close[i] < min(...).
    Выход: обратный канал exit_channel ИЛИ ATR-стоп ИЛИ max_hold.
    Экстремумы — строго по прошлым барам (shift 1) → no look-ahead.
    """
    close = df["close"].to_numpy(float)
    high = df["high"].to_numpy(float); low = df["low"].to_numpy(float)
    n = len(close)
    s = pd.Series(close)
    hi = s.rolling(channel, min_periods=channel).max().shift(1).to_numpy()
    lo = s.rolling(channel, min_periods=channel).min().shift(1).to_numpy()
    ex_hi = s.rolling(exit_channel, min_periods=exit_channel).max().shift(1).to_numpy()
    ex_lo = s.rolling(exit_channel, min_periods=exit_channel).min().shift(1).to_numpy()
    atr = _atr(high, low, close, atr_n)
    ts = df.index.to_numpy(); mins = _session_minutes(ts)

    pos = np.zeros(n, int); reason = np.array([""] * n, object)
    cur = 0; entry_px = 0.0; held = 0; entry_atr = 0.0
    for i in range(n):
        if cur != 0:
            held += 1
            exit_now = ""
            if atr_stop > 0 and entry_atr > 0:
                adverse = (entry_px - close[i]) if cur > 0 else (close[i] - entry_px)
                if adverse > atr_stop * entry_atr:
                    exit_now = "stop"
            if not exit_now:
                if cur > 0 and not np.isnan(ex_lo[i]) and close[i] < ex_lo[i]:
                    exit_now = "chan"
                elif cur < 0 and not np.isnan(ex_hi[i]) and close[i] > ex_hi[i]:
                    exit_now = "chan"
            if not exit_now and max_hold > 0 and held >= max_hold:
                exit_now = "time"
            if exit_now:
                pos[i] = 0; reason[i] = exit_now; cur = 0; held = 0
                continue
            pos[i] = cur
            continue
        if np.isnan(hi[i]) or np.isnan(atr[i]):
            continue
        if not (time_lo <= mins[i] < time_hi):
            continue
        if close[i] > hi[i]:
            cur = 1; entry_px = close[i]; held = 0; entry_atr = atr[i]
            pos[i] = 1; reason[i] = "entry"
        elif close[i] < lo[i]:
            cur = -1; entry_px = close[i]; held = 0; entry_atr = atr[i]
            pos[i] = -1; reason[i] = "entry"
    return pos, reason
