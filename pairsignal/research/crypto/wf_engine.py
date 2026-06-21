"""Walk-forward движок и семейства стратегий для крипто-исследования.

Все бэктесты — векторные на pandas, no look-ahead: на баре t решение использует
только данные по бар t-1 включительно (вход исполняется по open/close следующего
бара через shift(1) весов). Издержки = оборот · (fee + slippage) на каждую смену веса.

Семейства:
  csmom   — cross-sectional momentum (long/short top-k vs bottom-k или long-only+MA-фильтр)
  tsmom   — time-series momentum (по каждой монете отдельно: знак прошлой доходности / MA-cross)
  donch   — Donchian breakout (TS-momentum через пробой канала)

Метрики Sharpe аннуализируются по числу баров в году для таймфрейма.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

_BARS_PER_YEAR = {"5m": 365 * 288, "15m": 365 * 96, "1h": 365 * 24, "4h": 365 * 6, "1d": 365}


def bars_per_year(tf: str) -> int:
    return _BARS_PER_YEAR.get(tf, 365 * 24)


# ============================================================================
# Метрики по серии чистых барных доходностей портфеля
# ============================================================================

def metrics_from_net(net: pd.Series, tf: str) -> dict:
    """net — серия чистых (после издержек) барных доходностей портфеля."""
    equity = (1.0 + net).cumprod()
    bpy = bars_per_year(tf)
    mean, std = net.mean(), net.std(ddof=0)
    sharpe = float(mean / std * np.sqrt(bpy)) if std > 0 else 0.0
    dd = float((equity / equity.cummax() - 1.0).min() * 100)
    ret = float(equity.iloc[-1] - 1.0) * 100
    return {"equity": equity, "return_pct": ret, "sharpe": sharpe, "max_dd_pct": dd}


# ============================================================================
# CROSS-SECTIONAL MOMENTUM
# ============================================================================

def csmom_weights(prices: pd.DataFrame, lookback: int, holding: int, k: int,
                  long_short: bool, market_ma: int = 0) -> pd.DataFrame:
    """Веса портфеля (ffill между ребалансами). Ранг по доходности за lookback баров.

    long_short=True: топ-k → +1/k, дно-k → −1/k (доллар-нейтраль).
    long_short=False: топ-k → +1/k (long-only). market_ma>0 → в кэш, если индекс < SMA.
    No repaint: на баре ребаланса i momentum считается по prices[i]/prices[i-lookback].
    """
    mom = prices / prices.shift(lookback) - 1.0
    use_filter = market_ma > 0 and not long_short
    if use_filter:
        market = prices.mean(axis=1)
        market_sma = market.rolling(market_ma).mean()
    weights = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    for i in range(lookback, len(prices), holding):
        if use_filter and (np.isnan(market_sma.iloc[i]) or market.iloc[i] < market_sma.iloc[i]):
            weights.iloc[i] = np.zeros(prices.shape[1])
            continue
        row = mom.iloc[i].dropna()
        need = 2 * k if long_short else k
        if len(row) < need:
            continue
        ranked = row.sort_values(ascending=False)
        wrow = pd.Series(0.0, index=prices.columns)
        wrow[ranked.index[:k]] = 1.0 / k
        if long_short:
            wrow[ranked.index[-k:]] = -1.0 / k
        weights.iloc[i] = wrow.to_numpy()
    return weights.ffill().fillna(0.0)


def backtest_csmom(prices: pd.DataFrame, lookback: int, holding: int, k: int,
                   long_short: bool, market_ma: int, fee: float, slippage: float) -> dict:
    rets = prices.pct_change().fillna(0.0)
    w = csmom_weights(prices, lookback, holding, k, long_short, market_ma)
    port_ret = (w.shift(1) * rets).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).fillna(0.0)
    costs = turnover * (fee + slippage)
    net = port_ret - costs
    tf = prices.attrs.get("timeframe", "1h")
    m = metrics_from_net(net, tf)
    m["n_rebalances"] = int((turnover > 0).sum())
    m["costs_pct"] = round(float(costs.sum()) * 100, 2)
    m["net"] = net
    return m


# ============================================================================
# TIME-SERIES MOMENTUM (per-coin, equal-weight среди активных)
# ============================================================================

def tsmom_weights(prices: pd.DataFrame, lookback: int, holding: int,
                  long_short: bool, vol_target: bool = False) -> pd.DataFrame:
    """TS-momentum: для каждой монеты сигнал = знак доходности за lookback.

    long → +1, (если long_short) отрицательный momentum → −1, иначе 0 (кэш по этой ноге).
    Веса нормируются на число активных позиций (равный вес, суммарная экспозиция = 1
    в каждую сторону). Ребаланс каждые holding баров, ffill между.
    """
    mom = prices / prices.shift(lookback) - 1.0
    weights = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    for i in range(lookback, len(prices), holding):
        row = mom.iloc[i].dropna()
        if len(row) == 0:
            continue
        sig = pd.Series(0.0, index=prices.columns)
        longs = row[row > 0].index
        if long_short:
            shorts = row[row < 0].index
            nl, ns = len(longs), len(shorts)
            if nl:
                sig[longs] = 1.0 / nl
            if ns:
                sig[shorts] = -1.0 / ns
        else:
            if len(longs):
                sig[longs] = 1.0 / len(longs)
        weights.iloc[i] = sig.to_numpy()
    return weights.ffill().fillna(0.0)


def backtest_tsmom(prices: pd.DataFrame, lookback: int, holding: int,
                   long_short: bool, fee: float, slippage: float) -> dict:
    rets = prices.pct_change().fillna(0.0)
    w = tsmom_weights(prices, lookback, holding, long_short)
    port_ret = (w.shift(1) * rets).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).fillna(0.0)
    costs = turnover * (fee + slippage)
    net = port_ret - costs
    tf = prices.attrs.get("timeframe", "1h")
    m = metrics_from_net(net, tf)
    m["n_rebalances"] = int((turnover > 0).sum())
    m["costs_pct"] = round(float(costs.sum()) * 100, 2)
    m["net"] = net
    return m


# ============================================================================
# DONCHIAN BREAKOUT (TS-momentum через пробой канала, daily-стиль на 1h)
# ============================================================================

def backtest_donchian(prices: pd.DataFrame, entry: int, exit_n: int,
                      long_short: bool, fee: float, slippage: float) -> dict:
    """Пробой Donchian: long когда close = max(entry прошлых), выход когда close=min(exit_n).

    Состояние на монету: +1 после пробоя вверх, держим пока не пробьём нижний exit-канал.
    long_short: симметрично шорт по пробою вниз. Веса = равный вес активных, ffill по позиции.
    No repaint: каналы по shift(1) (только закрытые бары).
    """
    rets = prices.pct_change().fillna(0.0)
    hi = prices.shift(1).rolling(entry).max()
    lo_exit = prices.shift(1).rolling(exit_n).min()
    lo = prices.shift(1).rolling(entry).min()
    hi_exit = prices.shift(1).rolling(exit_n).max()

    pos = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for col in prices.columns:
        p = prices[col].to_numpy()
        h, le = hi[col].to_numpy(), lo_exit[col].to_numpy()
        l, he = lo[col].to_numpy(), hi_exit[col].to_numpy()
        state = 0.0
        out = np.zeros(len(p))
        for t in range(len(p)):
            if state == 0.0:
                if not np.isnan(h[t]) and p[t] >= h[t]:
                    state = 1.0
                elif long_short and not np.isnan(l[t]) and p[t] <= l[t]:
                    state = -1.0
            elif state == 1.0:
                if not np.isnan(le[t]) and p[t] <= le[t]:
                    state = 0.0
            elif state == -1.0:
                if not np.isnan(he[t]) and p[t] >= he[t]:
                    state = 0.0
            out[t] = state
        pos[col] = out

    # нормировка: равный вес среди активных позиций (по модулю), суммарная экспозиция ≤ 1
    active = pos.abs().sum(axis=1).replace(0.0, np.nan)
    w = pos.div(active, axis=0).fillna(0.0)
    port_ret = (w.shift(1) * rets).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).fillna(0.0)
    costs = turnover * (fee + slippage)
    net = port_ret - costs
    tf = prices.attrs.get("timeframe", "1h")
    m = metrics_from_net(net, tf)
    m["n_rebalances"] = int((turnover > 0).sum())
    m["costs_pct"] = round(float(costs.sum()) * 100, 2)
    m["net"] = net
    return m


# ============================================================================
# MEAN-REVERSION — cross-sectional (монета vs равновзвешенный индекс рынка)
# ============================================================================

def backtest_xmr(prices: pd.DataFrame, lookback: int, entry_z: float, stop_z: float,
                 max_hold: int, fee: float, slippage: float) -> dict:
    """Cross-sectional mean-reversion с ЧЕСТНЫМ симметричным стопом (без profit_target).

    Для каждой монеты считаем относительную силу r = ln(price) − ln(market_index),
    z = (r − rolling_mean(r, lookback)) / rolling_std(r, lookback). No repaint: индикаторы
    по закрытым барам, решение на t исполняется t+1 (shift(1) весов).

    Вход: z <= −entry_z → LONG монету (она отстала, ждём возврат); z >= +entry_z → SHORT.
    Выход: z пересёк 0 (возврат к среднему) — ВОЗВРАТ, прибыль.
    Стоп: |z| >= stop_z против позиции (отклонение усилилось) — СИММЕТРИЧНЫЙ риск-стоп.
    Тайм-стоп: max_hold баров.
    Доллар-нейтраль: позиции равновзвешены, суммарная экспозиция нормирована на 1 в сторону.
    """
    logp = np.log(prices)
    market = logp.mean(axis=1)
    rel = logp.sub(market, axis=0)                       # относит. сила каждой монеты
    mu = rel.rolling(lookback).mean()
    sd = rel.rolling(lookback).std(ddof=0)
    z = (rel - mu) / sd.replace(0.0, np.nan)

    cols = prices.columns
    znp = z.to_numpy()
    pos = np.zeros_like(znp)
    n, m = znp.shape
    for j in range(m):
        state = 0.0
        held = 0
        for t in range(n):
            zt = znp[t, j]
            if np.isnan(zt):
                pos[t, j] = state
                continue
            if state == 0.0:
                if zt <= -entry_z:
                    state, held = 1.0, 0
                elif zt >= entry_z:
                    state, held = -1.0, 0
            else:
                held += 1
                # возврат к среднему (z пересёк 0) → выход с прибылью
                if (state > 0 and zt >= 0.0) or (state < 0 and zt <= 0.0):
                    state = 0.0
                # симметричный стоп: отклонение усилилось за stop_z
                elif (state > 0 and zt <= -stop_z) or (state < 0 and zt >= stop_z):
                    state = 0.0
                elif held >= max_hold:
                    state = 0.0
            pos[t, j] = state

    pos = pd.DataFrame(pos, index=prices.index, columns=cols)
    rets = prices.pct_change().fillna(0.0)
    active = pos.abs().sum(axis=1).replace(0.0, np.nan)
    w = pos.div(active, axis=0).fillna(0.0)               # равный вес активных, нейтраль
    port_ret = (w.shift(1) * rets).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).fillna(0.0)
    costs = turnover * (fee + slippage)
    net = port_ret - costs
    tf = prices.attrs.get("timeframe", "1h")
    mdict = metrics_from_net(net, tf)
    mdict["n_rebalances"] = int((turnover > 0).sum())
    mdict["costs_pct"] = round(float(costs.sum()) * 100, 2)
    mdict["net"] = net
    return mdict


# ============================================================================
# BUY & HOLD бенчмарк (равновзвешенный индекс юниверса, с издержками входа)
# ============================================================================

def backtest_buyhold(prices: pd.DataFrame, fee: float, slippage: float) -> dict:
    rets = prices.pct_change().fillna(0.0)
    n = prices.shape[1]
    port_ret = rets.mean(axis=1)  # равный вес, ребаланс каждый бар (даёт rebalanced index)
    net = port_ret.copy()
    net.iloc[0] -= (fee + slippage)  # разовый вход
    tf = prices.attrs.get("timeframe", "1h")
    m = metrics_from_net(net, tf)
    m["net"] = net
    return m


# ============================================================================
# WALK-FORWARD: подбор параметров на train по Sharpe, прогон на test
# ============================================================================

def walk_forward(prices: pd.DataFrame, train_bars: int, test_bars: int,
                 grid: list[tuple], backtest_fn, warm_max: int,
                 fee: float, slippage: float) -> list[dict]:
    """Скользящие окна. grid — список кортежей параметров; backtest_fn(prices, *params,
    fee, slippage) → метрики. warm_max — макс. прогрев (баров истории до test для no-repaint).

    Параметры выбираются по Sharpe на train; OOS меряется на следующем test-окне с
    прогревом из хвоста train. Возвращает список окон с oos-метриками и net-серией.
    """
    out: list[dict] = []
    tf = prices.attrs.get("timeframe", "1h")
    start = 0
    while start + train_bars + test_bars <= len(prices):
        tr0, tr1 = start, start + train_bars
        te0, te1 = tr1, tr1 + test_bars
        train = prices.iloc[tr0:tr1]
        train.attrs["timeframe"] = tf

        best, best_sharpe = None, -np.inf
        for params in grid:
            m = backtest_fn(train, *params, fee, slippage)
            if np.isfinite(m["sharpe"]) and m["sharpe"] > best_sharpe:
                best_sharpe, best = m["sharpe"], params
        if best is None:
            start += test_bars
            continue

        warm = prices.iloc[max(0, te0 - warm_max):te1]
        warm.attrs["timeframe"] = tf
        oos = backtest_fn(warm, *best, fee, slippage)
        oos_net = oos["net"].iloc[-test_bars:]
        oos_m = metrics_from_net(oos_net, tf)
        out.append({
            "test_start": int(prices.index[te0]), "test_end": int(prices.index[te1 - 1]),
            "params": best,
            "oos_return_pct": round(oos_m["return_pct"], 2),
            "oos_sharpe": round(oos_m["sharpe"], 2),
            "oos_max_dd_pct": round(oos_m["max_dd_pct"], 2),
            "_net": oos_net,
        })
        start += test_bars
    return out


def stitch(rows: list[dict], tf: str) -> dict:
    """Склейка OOS net-серий всех окон в один портфель + агрегатные метрики."""
    if not rows:
        return {}
    net = pd.concat([r["_net"] for r in rows])
    m = metrics_from_net(net, tf)
    wins = sum(1 for r in rows if r["oos_return_pct"] > 0)
    return {
        "return_pct": round(m["return_pct"], 2),
        "sharpe": round(m["sharpe"], 2),
        "max_dd_pct": round(m["max_dd_pct"], 2),
        "n_windows": len(rows),
        "win_windows": wins,
        "equity": m["equity"],
    }
