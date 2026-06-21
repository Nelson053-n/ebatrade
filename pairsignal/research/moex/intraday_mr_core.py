"""Ядро single-name intraday mean-reversion (z-score) для акций MOEX, часовой ТФ.

Логика 1:1 с эталоном research.forts.strategies.meanrev_z и боевым st5
(ZScoreIndicator + entry_signal_mr/exit_signal_mr), но для ОДНОГО инструмента
(не пара): z=(close−SMA(ma_n))/std(ddof=0) по закрытым барам.

Вход:  z ≤ −entry_z (и z > −stop_z)  → LONG  (фейдим провал, ждём возврата вверх)
       z ≥ +entry_z (и z < +stop_z)  → SHORT
Выход: TP   |z| ≤ exit_z (возврат к средней)
       stop |z| ≥ stop_z (ушло глубже против)
       time  bars_held ≥ max_hold
Издержки: round-trip = 2·(commission+slippage) от нотионала на сделку.

No look-ahead: z[i] считается по close[0..i] (закрытые бары), сделка моделируется
по close[i]. Издержки симметричны на вход и выход. Доходность сделки — на нотионал
(направленная позиция в одной акции).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class MRParams:
    ma_n: int = 36
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 3.5
    max_hold: int = 24
    cost_oneway: float = 0.0005   # комиссия+slippage на одну сторону (доля нотионала)


@dataclass
class MRTrade:
    side: int          # +1 long, -1 short
    entry_i: int
    exit_i: int
    bars: int
    ret: float         # чистая доходность на нотионал (после издержек)
    reason: str


@dataclass
class MRResult:
    trades: list[MRTrade] = field(default_factory=list)
    n_bars: int = 0

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def rets(self) -> np.ndarray:
        return np.array([t.ret for t in self.trades], dtype=float)

    @property
    def total_ret(self) -> float:
        # сумма лог-эквивалентна нет, берём арифметическую сумму на нотионал (малые числа)
        return float(self.rets.sum())

    @property
    def win_rate(self) -> float:
        r = self.rets
        return float((r > 0).mean()) if len(r) else float("nan")

    def sharpe_pertrade(self) -> float:
        r = self.rets
        if len(r) < 2 or r.std() == 0:
            return float("nan")
        return float(r.mean() / r.std() * np.sqrt(len(r)))


def _zscore(close: np.ndarray, ma_n: int) -> np.ndarray:
    """z=(close−SMA)/std(ddof=0) скользящим окном ma_n, по закрытым барам [i-ma_n+1..i].

    Векторизовано через кумулятивные суммы (no look-ahead: окно заканчивается на i).
    """
    n = len(close)
    z = np.full(n, np.nan)
    if n < ma_n:
        return z
    csum = np.concatenate([[0.0], np.cumsum(close)])
    csum2 = np.concatenate([[0.0], np.cumsum(close * close)])
    # окна [lo..i] длиной ma_n, для i в [ma_n-1 .. n-1]
    hi = np.arange(ma_n - 1, n)          # индекс i
    lo = hi - ma_n + 1
    s = csum[hi + 1] - csum[lo]
    s2 = csum2[hi + 1] - csum2[lo]
    mean = s / ma_n
    var = s2 / ma_n - mean * mean
    ok = var > 1e-12
    zz = np.full(len(hi), np.nan)
    zz[ok] = (close[hi][ok] - mean[ok]) / np.sqrt(var[ok])
    z[ma_n - 1:] = zz
    return z


def simulate(close: np.ndarray, p: MRParams,
             start_i: int = 0) -> MRResult:
    """Прогон single-name MR по серии closes. Сделки, чей ВХОД на баре >= start_i,
    учитываются (warmup-бары до start_i дают z, но входы там игнорируются —
    для walk-forward, где z виден из train-истории, а торгуем только в test).
    """
    n = len(close)
    z = _zscore(close, p.ma_n)
    res = MRResult(n_bars=n)
    cur = 0
    entry_px = 0.0
    entry_i = 0
    held = 0
    c = 2.0 * p.cost_oneway   # round-trip
    for i in range(n):
        if np.isnan(z[i]):
            continue
        if cur != 0:
            held += 1
            reason = ""
            if cur > 0:
                if z[i] >= -p.exit_z:
                    reason = "tp"
                elif z[i] <= -p.stop_z:
                    reason = "stop"
            else:
                if z[i] <= p.exit_z:
                    reason = "tp"
                elif z[i] >= p.stop_z:
                    reason = "stop"
            if not reason and p.max_hold > 0 and held >= p.max_hold:
                reason = "time"
            if reason:
                gross = cur * (close[i] - entry_px) / entry_px
                res.trades.append(MRTrade(side=cur, entry_i=entry_i, exit_i=i,
                                          bars=held, ret=gross - c, reason=reason))
                cur = 0
                held = 0
            continue
        if i < start_i:
            continue
        if z[i] <= -p.entry_z and z[i] > -p.stop_z:
            cur = 1
            entry_px = close[i]
            entry_i = i
            held = 0
        elif z[i] >= p.entry_z and z[i] < p.stop_z:
            cur = -1
            entry_px = close[i]
            entry_i = i
            held = 0
    return res


def buyhold_ret(close: np.ndarray, start_i: int = 0) -> float:
    seg = close[start_i:]
    if len(seg) < 2 or seg[0] <= 0:
        return float("nan")
    return float((seg[-1] - seg[0]) / seg[0])
