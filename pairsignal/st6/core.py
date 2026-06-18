"""
st6 · Correlation-Gated Pairs — ядро стратегии (без сети, тестируемо).

Чистая математика парного mean-reversion с корреляционным гейтом и
динамическим отбором пары. Никаких импортов tinkoff здесь — этот модуль
гоняется юнит-тестами и переиспользуется и в live, и в бэктесте.

Идея (3 корреляционных приёма в одном):
  1. Отбор пары (ranking): из корзины коррелированных бумаг периодически
     выбираем лучшую — высокая |corr| лог-доходностей + коинтеграция
     (стационарность спреда) + быстрый half-life возврата.
  2. Корреляционный гейт: входим только пока скользящая корреляция ног
     выше порога; если связь распадается (corr < exit-порога) — аварийный
     выход. Это страховка от классического слома пар-трейда.
  3. Mean-reversion спреда: spread = ln(A) - β·ln(B), β по OLS (доллар-
     нейтральный хедж). Торгуем z-score: вход на |z|≥entry, выход на
     |z|≤exit, стоп на |z|≥stop. Только по закрытым свечам (no repaint).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np


# --------------------------------------------------------------------------
# Параметры стратегии
# --------------------------------------------------------------------------
@dataclass
class Params:
    # окна
    beta_window: int = 240        # бар для оценки β (OLS) и средней спреда
    z_window: int = 240           # окно z-score
    corr_window: int = 120        # окно скользящей корреляции лог-доходностей
    # пороги входа/выхода по z
    z_entry: float = 2.0
    z_exit: float = 0.3
    z_stop: float = 3.5
    # корреляционный гейт
    corr_enter: float = 0.80      # вход разрешён только если |corr| >= этого
    corr_break: float = 0.55      # если |corr| падает ниже — аварийный выход
    # отбор пары (скан корзины)
    select_min_corr: float = 0.80
    select_max_pvalue: float = 0.10   # ADF p-value для коинтеграции спреда
    select_max_halflife: float = 240.0
    # риск/сайзинг
    risk_fraction: float = 0.02   # доля equity на сделку (нотионал каждой ноги)
    time_stop_bars: int = 0       # 0 = выкл; иначе принудительный выход по барам
    # издержки (на оборот, доля от нотионала ноги)
    fee_rate: float = 0.0006
    slippage_rate: float = 0.0005


# --------------------------------------------------------------------------
# Базовая статистика
# --------------------------------------------------------------------------
def log_returns(prices: Sequence[float]) -> np.ndarray:
    p = np.asarray(prices, dtype=float)
    return np.diff(np.log(p))


def rolling_correlation(a: Sequence[float], b: Sequence[float], window: int) -> float:
    """Корреляция Пирсона лог-доходностей за последние `window` баров."""
    ra = log_returns(a)
    rb = log_returns(b)
    n = min(len(ra), len(rb))
    if n < 3:
        return float("nan")
    w = min(window, n)
    ra, rb = ra[-w:], rb[-w:]
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def hedge_ratio(log_a: np.ndarray, log_b: np.ndarray) -> tuple[float, float]:
    """OLS: log_a = alpha + beta*log_b. Возвращает (beta, alpha)."""
    x = np.asarray(log_b, dtype=float)
    y = np.asarray(log_a, dtype=float)
    xm, ym = x.mean(), y.mean()
    denom = np.sum((x - xm) ** 2)
    if denom == 0:
        return 0.0, ym
    beta = float(np.sum((x - xm) * (y - ym)) / denom)
    alpha = float(ym - beta * xm)
    return beta, alpha


def spread_series(prices_a: Sequence[float], prices_b: Sequence[float],
                  beta: float) -> np.ndarray:
    """spread = ln(A) - beta*ln(B)."""
    la = np.log(np.asarray(prices_a, dtype=float))
    lb = np.log(np.asarray(prices_b, dtype=float))
    return la - beta * lb


def zscore(series: Sequence[float], window: int) -> float:
    """z последнего значения относительно скользящего окна."""
    s = np.asarray(series, dtype=float)
    w = min(window, len(s))
    if w < 2:
        return float("nan")
    win = s[-w:]
    sd = np.std(win)
    if sd == 0:
        return 0.0
    return float((win[-1] - np.mean(win)) / sd)


# Критические значения Dickey-Fuller (модель с константой, без тренда) и
# соответствующие приблизительные p-value — для лёгкого fallback без statsmodels.
_DF_T = [-4.00, -3.43, -2.86, -2.57, -1.95, -1.62, -1.00, -0.50, 0.00, 1.00]
_DF_P = [0.005, 0.010, 0.050, 0.100, 0.250, 0.500, 0.750, 0.900, 0.950, 0.990]


def adf_pvalue(series: Sequence[float]) -> float:
    """
    Тест на стационарность спреда (коинтеграция через Engle-Granger).
    Если установлен statsmodels — настоящий ADF. Иначе — самостоятельный
    Dickey-Fuller (lag 0): t-статистика наклона регрессии Δs_t на s_{t-1},
    отображённая в p-value по критическим значениям DF. Чем сильнее возврат
    к среднему, тем отрицательнее t и меньше p. Возвращает p-value.
    """
    s = np.asarray(series, dtype=float)
    if len(s) < 20:
        return 1.0
    try:
        from statsmodels.tsa.stattools import adfuller  # type: ignore
        return float(adfuller(s, autolag="AIC")[1])
    except Exception:
        ds = np.diff(s)
        lag = s[:-1]
        n = len(ds)
        lm = lag - lag.mean()
        sxx = np.sum(lm ** 2)
        if sxx == 0:
            return 1.0
        beta = np.sum(lm * (ds - ds.mean())) / sxx       # ~ (phi - 1)
        resid = (ds - ds.mean()) - beta * lm
        s2 = np.sum(resid ** 2) / max(1, n - 2)
        se = math.sqrt(s2 / sxx) if sxx > 0 else float("inf")
        if se == 0 or not np.isfinite(se):
            return 1.0
        t_stat = beta / se
        # интерполяция t -> p по таблице DF (вне диапазона — зажимаем к краям)
        return float(np.interp(t_stat, _DF_T, _DF_P))


def half_life(series: Sequence[float]) -> float:
    """Half-life возврата к среднему (бары) по модели Орнштейна-Уленбека."""
    s = np.asarray(series, dtype=float)
    if len(s) < 10:
        return float("inf")
    ds = np.diff(s)
    lag = s[:-1]
    lm = lag - lag.mean()
    denom = np.sum(lm ** 2)
    if denom == 0:
        return float("inf")
    theta = np.sum(lm * (ds - ds.mean())) / denom
    if theta >= 0:
        return float("inf")  # не возвращается к среднему
    return float(-math.log(2) / theta)


# --------------------------------------------------------------------------
# Отбор пары из корзины
# --------------------------------------------------------------------------
@dataclass
class PairStat:
    a: str
    b: str
    corr: float
    beta: float
    pvalue: float
    halflife: float
    score: float


def rank_pairs(series_by_ticker: dict[str, Sequence[float]],
               p: Params) -> list[PairStat]:
    """
    Перебираем все пары корзины, считаем корреляцию/β/коинтеграцию/half-life,
    отбираем годные и сортируем по композитному score (выше — лучше).
    series_by_ticker: тикер -> ряд цен (закрытия, выровненные по времени).
    """
    tickers = [t for t, s in series_by_ticker.items() if len(s) > p.beta_window]
    out: list[PairStat] = []
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            ta, tb = tickers[i], tickers[j]
            a = np.asarray(series_by_ticker[ta], dtype=float)
            b = np.asarray(series_by_ticker[tb], dtype=float)
            n = min(len(a), len(b))
            a, b = a[-n:], b[-n:]
            corr = rolling_correlation(a, b, p.corr_window)
            if not np.isfinite(corr) or abs(corr) < p.select_min_corr:
                continue
            beta, _ = hedge_ratio(np.log(a[-p.beta_window:]),
                                  np.log(b[-p.beta_window:]))
            if beta <= 0:
                continue  # ждём положительно связанные ноги
            spr = spread_series(a[-p.beta_window:], b[-p.beta_window:], beta)
            pval = adf_pvalue(spr)
            hl = half_life(spr)
            if pval > p.select_max_pvalue:
                continue
            if not np.isfinite(hl) or hl <= 1 or hl > p.select_max_halflife:
                continue
            # score: сильная связь + стационарность + быстрый возврат
            score = abs(corr) * (1.0 - pval) * (1.0 / math.log1p(hl))
            out.append(PairStat(ta, tb, corr, beta, pval, hl, score))
    out.sort(key=lambda s: s.score, reverse=True)
    return out


# --------------------------------------------------------------------------
# Машина состояний сделки (FSM)
# --------------------------------------------------------------------------
class Side(Enum):
    FLAT = 0
    LONG_SPREAD = 1    # купили спред: long A, short B (ждём роста спреда)
    SHORT_SPREAD = -1  # продали спред: short A, long B (ждём падения спреда)


class ExitReason(Enum):
    NONE = "—"
    TAKE = "target"            # вернулись к средней (|z|<=z_exit)
    STOP_Z = "stop_z"          # спред ушёл против (|z|>=z_stop)
    CORR_BREAK = "corr_break"  # корреляция распалась
    TIME = "time_stop"         # тайм-стоп


@dataclass
class Position:
    side: Side = Side.FLAT
    entry_z: float = 0.0
    beta: float = 0.0
    bars_held: int = 0
    qty_a: int = 0
    qty_b: int = 0
    entry_a: float = 0.0
    entry_b: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.side != Side.FLAT


@dataclass
class Signal:
    action: str  # "ENTER_LONG", "ENTER_SHORT", "EXIT", "HOLD"
    reason: ExitReason = ExitReason.NONE
    z: float = 0.0
    corr: float = 0.0
    beta: float = 0.0


def decide(prices_a: Sequence[float], prices_b: Sequence[float],
           pos: Position, p: Params) -> Signal:
    """
    Главное правило по закрытым свечам. На вход — выровненные ряды цен
    (последний элемент = только что закрытая свеча). Возвращает сигнал.
    """
    a = np.asarray(prices_a, dtype=float)
    b = np.asarray(prices_b, dtype=float)
    n = min(len(a), len(b))
    if n < max(p.beta_window, p.z_window, p.corr_window) + 1:
        return Signal("HOLD")
    a, b = a[-n:], b[-n:]

    corr = rolling_correlation(a, b, p.corr_window)
    beta = pos.beta if pos.is_open else hedge_ratio(
        np.log(a[-p.beta_window:]), np.log(b[-p.beta_window:]))[0]
    spr = spread_series(a, b, beta)
    z = zscore(spr, p.z_window)

    if not (np.isfinite(corr) and np.isfinite(z) and beta > 0):
        return Signal("HOLD", z=0.0, corr=corr if np.isfinite(corr) else 0.0)

    # ----- открытая позиция: проверяем выходы -----
    if pos.is_open:
        if abs(corr) < p.corr_break:
            return Signal("EXIT", ExitReason.CORR_BREAK, z, corr, beta)
        if abs(z) >= p.z_stop:
            return Signal("EXIT", ExitReason.STOP_Z, z, corr, beta)
        if abs(z) <= p.z_exit:
            return Signal("EXIT", ExitReason.TAKE, z, corr, beta)
        if p.time_stop_bars and pos.bars_held >= p.time_stop_bars:
            return Signal("EXIT", ExitReason.TIME, z, corr, beta)
        return Signal("HOLD", ExitReason.NONE, z, corr, beta)

    # ----- позиции нет: проверяем вход (с корреляционным гейтом) -----
    if abs(corr) < p.corr_enter:
        return Signal("HOLD", z=z, corr=corr, beta=beta)
    if z >= p.z_entry:
        # спред высоко -> продаём спред: short A, long B
        return Signal("ENTER_SHORT", ExitReason.NONE, z, corr, beta)
    if z <= -p.z_entry:
        # спред низко -> покупаем спред: long A, short B
        return Signal("ENTER_LONG", ExitReason.NONE, z, corr, beta)
    return Signal("HOLD", z=z, corr=corr, beta=beta)


# --------------------------------------------------------------------------
# Сайзинг ног (доллар-нейтральность по β)
# --------------------------------------------------------------------------
def leg_quantities(equity: float, price_a: float, price_b: float,
                   beta: float, lot_a: int, lot_b: int, p: Params
                   ) -> tuple[int, int]:
    """
    Нотионал на ногу = risk_fraction * equity. Нога B масштабируется на β
    для доллар-нейтральности. Возвращает число лотов (целое, >=0) по каждой.
    """
    notional = max(0.0, p.risk_fraction * equity)
    qa = int(notional // (price_a * lot_a)) if price_a > 0 and lot_a > 0 else 0
    nb = notional * beta
    qb = int(nb // (price_b * lot_b)) if price_b > 0 and lot_b > 0 else 0
    return qa, qb


def trade_pnl(side: Side, entry_a: float, exit_a: float,
              entry_b: float, exit_b: float,
              units_a: float, units_b: float, p: Params) -> float:
    """
    Net P&L пары в деньгах. LONG_SPREAD = long A / short B.
    units_* — это лоты*размер_лота (число штук бумаги).
    """
    if side == Side.LONG_SPREAD:
        gross = (exit_a - entry_a) * units_a + (entry_b - exit_b) * units_b
    elif side == Side.SHORT_SPREAD:
        gross = (entry_a - exit_a) * units_a + (exit_b - entry_b) * units_b
    else:
        return 0.0
    # издержки: вход+выход по обеим ногам
    turnover = (entry_a * units_a + exit_a * units_a
                + entry_b * units_b + exit_b * units_b)
    costs = turnover * (p.fee_rate + p.slippage_rate)
    return float(gross - costs)
