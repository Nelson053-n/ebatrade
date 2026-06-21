"""Cross-sectional momentum walk-forward на ЧАСОВЫХ акциях MOEX (27 тикеров, ~6 лет).

Дневки дали лишь 9 OOS окон (Sharpe ≤0.38, jackknife убивал) — мало данных, чтобы
честно судить. Часовой ISS (interval=60) ~24k баров/тикер → десятки-сотни окон.
Тот же честный walk-forward что для дневок (momentum_wf), но в часах:
train подбирает (lookback, holding) по Sharpe, test невиданный меряет OOS.

Биржа MOEX работает ~9-10 ч/день, не 24 — поэтому bars/year ≈ фактическому числу
часовых баров в году (а не 365*24). Считаем его из самого ряда для корректного Sharpe.

Переиспользует чистое ядро backtest_momentum из pairsignal.momentum. Данные —
cache/union_1h.csv (momentum_h1_load.py, только чтение ISS).

  python -m pairsignal.research.moex.momentum_h1_wf
  python -m pairsignal.research.moex.momentum_h1_wf --long-only --market-ma 200
"""
from __future__ import annotations

import argparse
import itertools
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pairsignal.momentum import backtest_momentum  # noqa: E402

CACHE = Path(__file__).resolve().parent / "cache"
COV_MIN = 0.95  # порог покрытия тикера (отбрасываем недавно листнувшиеся / с дырами)


def _bars_per_year(index: pd.Index) -> float:
    """Фактических часовых баров в году по диапазону ряда (MOEX ~ 4000/год, не 365*24)."""
    span_days = (int(index[-1]) - int(index[0])) / 1000 / 86400
    if span_days <= 0:
        return 252 * 9.0  # фолбэк
    return len(index) / span_days * 365.0


def load_prices(cov_min: float = COV_MIN) -> pd.DataFrame:
    """Часовые закрытия выровненного юниверса MOEX. attrs['bpy'] — баров в году."""
    df = pd.read_csv(CACHE / "union_1h.csv", index_col=0)
    df.index = df.index.astype("int64")
    cov = df.notna().sum()
    keep = [t for t in df.columns if cov[t] >= cov_min * len(df)]
    sub = df[keep].dropna().sort_index()
    sub.attrs["timeframe"] = "1h"          # для _BARS_PER_YEAR внутри backtest (не критично)
    sub.attrs["bpy"] = _bars_per_year(sub.index)
    return sub


def _d(ts: int) -> str:
    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def _sharpe(net: pd.Series, bpy: float) -> float:
    s = net.std(ddof=0)
    return float(net.mean() / s * np.sqrt(bpy)) if s > 0 else 0.0


def walk_forward(prices: pd.DataFrame, train_bars: int, test_bars: int,
                 lookbacks: list[int], holdings: list[int], k: int,
                 long_short: bool, fee: float, slippage: float,
                 market_ma: int = 0, bpy: float | None = None) -> list[dict]:
    """Скользящие окна: train подбирает (lookback, holding) по Sharpe, test меряет OOS.

    Возвращает список окон с OOS-метриками + хвост equity-кривой test-сегмента.
    Sharpe аннуализируется по bpy (часовых баров в году). market_ma>0 (long-only):
    рыночный фильтр применяется и на train, и на test одинаково.
    """
    bpy = bpy or prices.attrs.get("bpy", 252 * 9.0)
    out: list[dict] = []
    grid = list(itertools.product(lookbacks, holdings))
    start = 0
    while start + train_bars + test_bars <= len(prices):
        train = prices.iloc[start:start + train_bars].copy()
        train.attrs["bpy"] = bpy
        test = prices.iloc[start + train_bars:start + train_bars + test_bars]

        best, best_sharpe = None, -np.inf
        for lb, hd in grid:
            if lb >= len(train):
                continue
            m = backtest_momentum(train, lb, hd, k, long_short, fee, slippage, market_ma)
            # пересчёт Sharpe на правильный bpy (backtest использует свой _BARS_PER_YEAR)
            eq = m["equity"]
            net = eq.pct_change().dropna()
            sh = _sharpe(net, bpy)
            if sh > best_sharpe:
                best_sharpe, best = sh, (lb, hd)
        if best is None:
            start += test_bars
            continue

        lb, hd = best
        warm_n = max(lb, market_ma)
        warm = prices.iloc[start + train_bars - warm_n:start + train_bars + test_bars]
        oos = backtest_momentum(warm, lb, hd, k, long_short, fee, slippage, market_ma)
        oos_eq = oos["equity"].iloc[-len(test):]
        oos_ret = float(oos_eq.iloc[-1] / oos_eq.iloc[0] - 1.0) * 100
        oos_net = oos_eq.pct_change().dropna()
        oos_sharpe = round(_sharpe(oos_net, bpy), 2)
        oos_dd = round(float((oos_eq / oos_eq.cummax() - 1.0).min() * 100), 2)

        out.append({
            "test_start": int(test.index[0]), "test_end": int(test.index[-1]),
            "lookback": lb, "holding": hd,
            "oos_return_pct": round(oos_ret, 2),
            "oos_sharpe": oos_sharpe, "oos_max_dd_pct": oos_dd,
            "avg_turnover": oos["avg_turnover"], "costs_pct": oos["costs_pct"],
            "_oos_eq": oos_eq,
        })
        start += test_bars
    return out


def stitch(rows: list[dict], bpy: float) -> dict:
    """Склейка OOS equity всех окон в одну кривую → агрегированные метрики."""
    if not rows:
        return {}
    eqs, cum = [], 1.0
    for r in rows:
        seg = r["_oos_eq"] / r["_oos_eq"].iloc[0] * cum
        eqs.append(seg)
        cum = float(seg.iloc[-1])
    full_eq = pd.concat(eqs)
    total_ret = (cum - 1.0) * 100
    fr = full_eq.pct_change().dropna()
    sharpe = _sharpe(fr, bpy)
    maxdd = float((full_eq / full_eq.cummax() - 1.0).min() * 100)
    wins = sum(1 for r in rows if r["oos_return_pct"] > 0)
    return {"total_ret": total_ret, "sharpe": sharpe, "maxdd": maxdd,
            "wins": wins, "n": len(rows), "equity": full_eq}


def _print_table(rows: list[dict], limit: int = 40) -> None:
    hdr = ["window", "lb", "hd", "oos_ret%", "sharpe", "maxDD%", "turn", "cost%"]
    show = rows if len(rows) <= limit else rows[:limit // 2] + rows[-limit // 2:]
    w = {h: len(h) for h in hdr}
    disp = []
    for r in show:
        d = {"window": f"{_d(r['test_start'])}…{_d(r['test_end'])}",
             "lb": r["lookback"], "hd": r["holding"], "oos_ret%": r["oos_return_pct"],
             "sharpe": r["oos_sharpe"], "maxDD%": r["oos_max_dd_pct"],
             "turn": r["avg_turnover"], "cost%": r["costs_pct"]}
        disp.append(d)
        for h in hdr:
            w[h] = max(w[h], len(str(d[h])))
    line = "  ".join(h.ljust(w[h]) for h in hdr)
    print(line)
    print("-" * len(line))
    for i, d in enumerate(disp):
        if len(rows) > limit and i == limit // 2:
            print(f"... ({len(rows) - limit} окон скрыто) ...")
        print("  ".join(str(d[h]).ljust(w[h]) for h in hdr))


def run_mode(prices, train_bars, test_bars, lookbacks, holdings, k,
             long_short, fee, slippage, market_ma, label, bpy):
    rows = walk_forward(prices, train_bars, test_bars, lookbacks, holdings, k,
                        long_short, fee, slippage, market_ma, bpy)
    print(f"\n=== {label} ===")
    if not rows:
        print("Недостаточно истории для walk-forward.")
        return None
    _print_table(rows)
    agg = stitch(rows, bpy)
    print(f"\nСклеенный OOS: return {agg['total_ret']:+.2f}%, Sharpe {agg['sharpe']:.2f}, "
          f"maxDD {agg['maxdd']:.2f}%, прибыльных окон {agg['wins']}/{agg['n']}")
    return rows, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--long-only", action="store_true")
    ap.add_argument("--market-ma", type=int, default=0)
    # окна в ЧАСАХ. MOEX ~9-10 торговых ч/день → 504 ч ≈ ~55 дней, 252 ч ≈ ~27 дней
    ap.add_argument("--train-bars", type=int, default=2520)  # ~1 год торговых часов
    ap.add_argument("--test-bars", type=int, default=504)    # ~2.5 месяца
    ap.add_argument("--fee", type=float, default=0.0005)     # 0.05% комиссия/нога
    ap.add_argument("--slippage", type=float, default=0.0003)  # 0.03% слиппедж/нога
    args = ap.parse_args()

    prices = load_prices()
    bpy = prices.attrs["bpy"]
    print(f"Юниверс: {prices.shape[1]} тикеров, {len(prices)} часовых баров "
          f"({_d(int(prices.index[0]))} … {_d(int(prices.index[-1]))}); bars/year≈{bpy:.0f}")
    rets = prices.pct_change().dropna()
    ew = rets.mean(axis=1)
    bh_eq = (1 + ew).cumprod()
    print(f"BuyHold EW корзины: {(bh_eq.iloc[-1]-1)*100:+.2f}%, "
          f"Sharpe {_sharpe(ew, bpy):.2f}")

    lookbacks = [24, 48, 96, 168, 336]
    holdings = [12, 24, 48]
    print(f"\nСетка: lookbacks(ч)={lookbacks}, holdings(ч)={holdings}, k={args.k}")
    print(f"Walk-forward: train={args.train_bars} / test={args.test_bars} часов, "
          f"издержки fee={args.fee} slip={args.slippage}/нога")

    long_short = not args.long_only
    label = (f"L/S k={args.k}" if long_short else
             f"LONG-ONLY k={args.k}" + (f" +MA{args.market_ma}" if args.market_ma else ""))
    run_mode(prices, args.train_bars, args.test_bars, lookbacks, holdings, args.k,
             long_short, args.fee, args.slippage, args.market_ma, label, bpy)


if __name__ == "__main__":
    main()
