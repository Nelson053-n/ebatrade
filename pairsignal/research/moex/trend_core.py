"""trend_core — time-series trend-following на ОТДЕЛЬНЫХ акциях MOEX (single-name).

НЕ cross-sectional ранжирование (это отвергнуто). Здесь каждый тикер торгуется
независимо своим трендовым сигналом; equity по корзине — среднее equity по тикерам
(равновес, диверсификация). ОДИН набор параметров на все тикеры.

Семейства сигналов (позиция на бар t рассчитана по данным ≤ t, исполнена на t+1):
  - ma_cross:  pos = +1 если close>SMA(n) [long-only], либо sign(SMAf-SMAs) [long/short]
  - donchian:  лонг при пробое N-макс, выход при пробое M-мин (или флип в шорт)
  - tsmom:     знак доходности за lookback (Moskowitz), опц. vol-targeting

No look-ahead: сигнал s_t считается по close ≤ t, доходность применяется r_{t+1}.
Издержки: |Δpos| * (fee+slip) на смену позиции (вход/выход/флип).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parent / "cache"
_BARS_PER_YEAR = {"1h": 365 * 24, "1d": 252}  # для акций торговых дней ~252


def load_closes(interval: str, min_cov: float = 0.95) -> pd.DataFrame:
    """Закрытия всех тикеров (union), оставляем те, у кого coverage>=min_cov."""
    df = pd.read_csv(CACHE / f"union_{interval}.csv", index_col=0)
    df.index = df.index.astype("int64")
    if "union" in df.columns:
        df = df.drop(columns=["union"])
    cov = df.notna().mean()
    keep = cov[cov >= min_cov].index.tolist()
    df = df[keep].sort_index()
    df.attrs["interval"] = interval
    return df


def d(ts: int) -> str:
    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


# --- сигналы (позиция в [-1,0,1] или непрерывная для vol-target) ---------------

def sig_ma_cross(close: pd.Series, n: int, fast: int = 0, long_short: bool = False) -> pd.Series:
    """MA-cross. fast=0 → close vs SMA(n). fast>0 → SMA(fast) vs SMA(n)."""
    if fast > 0:
        ref = close.rolling(fast).mean()
        base = close.rolling(n).mean()
    else:
        ref = close
        base = close.rolling(n).mean()
    raw = np.sign(ref - base)
    if not long_short:
        raw = raw.clip(lower=0.0)
    return raw


def sig_donchian(close: pd.Series, entry_n: int, exit_n: int, long_short: bool = False) -> pd.Series:
    """Donchian breakout: лонг при close>=макс(entry_n предыдущих), выход при
    close<=мин(exit_n). long_short → флип в шорт по нижнему пробою."""
    hi = close.rolling(entry_n).max().shift(1)   # макс БЕЗ текущего бара (no look-ahead)
    lo = close.rolling(exit_n).min().shift(1)
    lo_e = close.rolling(entry_n).min().shift(1)  # для шорт-входа
    hi_x = close.rolling(exit_n).max().shift(1)   # для шорт-выхода
    pos = pd.Series(np.nan, index=close.index)
    state = 0.0
    arr_c = close.to_numpy(); arr_hi = hi.to_numpy(); arr_lo = lo.to_numpy()
    arr_loe = lo_e.to_numpy(); arr_hix = hi_x.to_numpy()
    out = np.zeros(len(close))
    for i in range(len(close)):
        c = arr_c[i]
        if np.isnan(arr_hi[i]):
            out[i] = 0.0; continue
        if state <= 0 and c >= arr_hi[i]:
            state = 1.0
        elif state >= 0 and long_short and c <= arr_loe[i]:
            state = -1.0
        elif state > 0 and c <= arr_lo[i]:
            state = 0.0
        elif state < 0 and c >= arr_hix[i]:
            state = 0.0
        out[i] = state
    return pd.Series(out, index=close.index)


def sig_tsmom(close: pd.Series, lookback: int, long_short: bool = False) -> pd.Series:
    """Time-series momentum: знак доходности за lookback закрытых баров."""
    raw = np.sign(close / close.shift(lookback) - 1.0)
    if not long_short:
        raw = raw.clip(lower=0.0)
    return raw


# --- бэктест одного тикера ----------------------------------------------------

def backtest_one(close: pd.Series, pos: pd.Series, fee: float, slip: float,
                 vol_target: float = 0.0, vol_win: int = 20) -> pd.Series:
    """Чистая по-бар доходность одного тикера.

    pos_t — сигнал по close≤t. Применяется к доходности r_{t+1}: используем pos.shift(1).
    vol_target>0 → масштаб позиции = target_daily_vol / realized_vol(vol_win) (cap 1.0),
    реализованная волатильность по доходностям ≤ t (shift внутри).
    """
    r = close.pct_change().fillna(0.0)
    if vol_target > 0:
        rv = r.rolling(vol_win).std().shift(1)
        scale = (vol_target / rv).clip(upper=3.0).fillna(0.0)
        eff = (pos * scale)
    else:
        eff = pos
    held = eff.shift(1).fillna(0.0)              # позиция, державшаяся в течение бара t
    gross = held * r
    turn = held.diff().abs().fillna(held.abs())  # смена экспозиции
    cost = turn * (fee + slip)
    return gross - cost


def metrics(net: pd.Series, interval: str) -> dict:
    eq = (1.0 + net).cumprod()
    bpy = _BARS_PER_YEAR[interval]
    mu, sd = net.mean(), net.std(ddof=0)
    sharpe = float(mu / sd * np.sqrt(bpy)) if sd > 0 else 0.0
    dd = float((eq / eq.cummax() - 1.0).min() * 100)
    return {"equity": eq, "ret_pct": float(eq.iloc[-1] - 1.0) * 100,
            "sharpe": sharpe, "max_dd_pct": dd}


# --- корзина: один набор параметров на все тикеры -----------------------------

def basket_net(closes: pd.DataFrame, sig_fn, fee: float, slip: float,
               vol_target: float = 0.0) -> pd.Series:
    """Средняя (равновес) по-бар доходность корзины: каждый тикер торгуется
    одним и тем же сигналом, equity усредняется. sig_fn(close)->pos."""
    cols = []
    for t in closes.columns:
        c = closes[t].dropna()
        if len(c) < 60:
            continue
        pos = sig_fn(c)
        net = backtest_one(c, pos, fee, slip, vol_target)
        cols.append(net.reindex(closes.index).fillna(0.0))
    if not cols:
        return pd.Series(0.0, index=closes.index)
    return pd.concat(cols, axis=1).mean(axis=1)


def basket_metrics(closes: pd.DataFrame, sig_fn, fee: float, slip: float,
                   interval: str, vol_target: float = 0.0) -> dict:
    net = basket_net(closes, sig_fn, fee, slip, vol_target)
    return metrics(net, interval)


def buyhold_ew(closes: pd.DataFrame, interval: str, fee: float = 0.0) -> dict:
    """Buy&Hold равновзвешенный по корзине (ребаланс не моделируем — просто среднее r)."""
    r = closes.pct_change().fillna(0.0)
    net = r.mean(axis=1)
    return metrics(net, interval)
