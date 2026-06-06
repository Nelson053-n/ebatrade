"""Сравнение equity-кривых всех стратегий на ОДНОМ периоде/таймфрейме (честно).

Прогоняет на общем универсуме (топ-N mexc, 1h, год) и одних издержках:
  - CSMomentum (cross-sectional momentum, наш лучший OOS)
  - PairsCoint (pairs-trading по коинтеграции, split 50/50)
  - Buy&Hold (равновзвешенный портфель монет — бенчмарк рынка)
Рисует ASCII-график нормированных equity-кривых (старт = 1.0) в консоль.

  python -m pairsignal.compare_strategies --top 30 --days 365 --timeframe 1h
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from .main import run_backtest
from .momentum import momentum_weights
from .pairs_coint import load_universe, make_pair_cfg, select_pairs
from .scan_year import top_symbols

EXCHANGE = "mexc"


def eq_buy_hold(prices: pd.DataFrame) -> pd.Series:
    """Равновзвешенный buy&hold: среднее нормированных цен монет."""
    norm = prices / prices.iloc[0]
    return norm.mean(axis=1)


def eq_momentum(prices: pd.DataFrame, fee: float, slip: float) -> pd.Series:
    """Equity momentum на ВСЁМ периоде (фикс. параметры — для сопоставимой кривой)."""
    rets = prices.pct_change().fillna(0.0)
    w = momentum_weights(prices, lookback=48, holding=24, k=3, long_short=True)
    net = (w.shift(1) * rets).sum(axis=1) - w.diff().abs().sum(axis=1).fillna(0.0) * (fee + slip)
    return (1.0 + net).cumprod()


def eq_pairs(prices: pd.DataFrame, fee: float, slip: float, timeframe: str) -> pd.Series | None:
    """Equity pairs-trading: отбор коинтегрированных пар на 1-й половине, средняя equity
    портфеля пар на 2-й (out-of-sample). Возвращает кривую на test-периоде."""
    mid = len(prices) // 2
    train, test = prices.iloc[:mid], prices.iloc[mid:]
    selected = select_pairs(train, pvalue_max=0.05, corr_min=0.6,
                            max_half_life=600, top_pairs=10, workers=5)
    if not selected:
        return None
    curves = []
    for pair in selected:
        a, b = pair["sym_a"], pair["sym_b"]
        df = (test[[a, b]].rename(columns={a: "price_a", b: "price_b"}).dropna())
        if len(df) < 560:
            continue
        cfg = make_pair_cfg(a, b, timeframe, slip, fee)
        eng = run_backtest(cfg, df, verbose=False)
        # equity по сделкам: старт 1.0, каждая сделка двигает на net_pnl/notional
        bal = 1.0
        pts = {df.index[0]: 1.0}
        for t in eng.exch.trades:
            bal *= (1.0 + t.net_pnl / (t.notional or 1.0))
            pts[t.exit_ts] = bal
        curves.append(pd.Series(pts))
    if not curves:
        return None
    # выравниваем по test-индексу, ffill, усредняем (равный вес пар)
    aligned = [c.reindex(test.index, method="ffill").fillna(1.0) for c in curves]
    return pd.concat(aligned, axis=1).mean(axis=1)


def ascii_plot(series: dict[str, pd.Series], width: int = 78, height: int = 20) -> str:
    """ASCII-график нескольких нормированных кривых на общей временной оси."""
    # общая ось: объединяем индексы, ffill каждую кривую
    idx = sorted(set().union(*[set(s.index) for s in series.values()]))
    cols = {name: s.reindex(idx).ffill().bfill() for name, s in series.items()}
    grid = pd.DataFrame(cols)

    # ресэмпл по width колонок
    step = max(1, len(grid) // width)
    g = grid.iloc[::step]
    vals = g.to_numpy()
    lo, hi = np.nanmin(vals), np.nanmax(vals)
    rng = hi - lo or 1.0

    marks = {name: m for name, m in zip(series, "#*o+x")}
    canvas = [[" "] * len(g) for _ in range(height)]
    for ci, name in enumerate(series):
        col = g[name].to_numpy()
        for xi, v in enumerate(col):
            if np.isnan(v):
                continue
            yi = int((v - lo) / rng * (height - 1))
            canvas[height - 1 - yi][xi] = marks[name]

    out = []
    for r in range(height):
        lvl = hi - r / (height - 1) * rng
        out.append(f"{lvl:5.2f} |" + "".join(canvas[r]))
    out.append("      +" + "-" * len(g))
    legend = "  ".join(f"{m} {name}" for name, m in marks.items())
    out.append("      " + legend)
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Сравнение equity-кривых стратегий на одном периоде")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--fee", type=float, default=0.0006)
    ap.add_argument("--slippage", type=float, default=0.0002)
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()

    until = int(time.time() * 1000)
    since = until - args.days * 24 * 3600 * 1000
    print(f"Загрузка топ-{args.top} монет ({args.days}д, {args.timeframe})…")
    symbols = top_symbols(args.top)
    prices = load_universe(symbols, args.timeframe, since, until, args.workers, min_coverage=0.9)
    prices.attrs["timeframe"] = args.timeframe
    print(f"Юниверс: {prices.shape[1]} монет, {len(prices)} баров.")

    series: dict[str, pd.Series] = {}
    series["BuyHold"] = eq_buy_hold(prices)
    series["Momentum"] = eq_momentum(prices, args.fee, args.slippage)
    pairs = eq_pairs(prices, args.fee, args.slippage, args.timeframe)
    if pairs is not None:
        series["Pairs"] = pairs

    # нормируем каждую к старту = 1.0 (на её первой валидной точке)
    for k in series:
        s = series[k].dropna()
        series[k] = s / s.iloc[0]

    print("\n=== EQUITY-КРИВЫЕ (нормированы к старту = 1.00) ===")
    print(ascii_plot(series))
    print("\n=== ИТОГ ===")
    for name, s in series.items():
        ret = (s.iloc[-1] - 1.0) * 100
        print(f"  {name:10} {ret:+7.1f}%")


if __name__ == "__main__":
    main()
