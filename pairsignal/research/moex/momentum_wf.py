"""Cross-sectional momentum walk-forward на дневных акциях MOEX (27 тикеров, ~6 лет).

Контрапункт к st6 (парный mean-reversion дал t≈0.3, эджа нет). Здесь — momentum:
ранжируем акции по доходности за lookback ЗАКРЫТЫХ дней, держим топ-k (и при L/S
шортим дно-k), ребаланс каждые holding дней. Честный walk-forward: train подбирает
(lookback, holding) по Sharpe, test невиданный меряет, скольжение на test_bars.

Переиспользует чистое векторное ядро из pairsignal.momentum (momentum_weights,
backtest_momentum). Данные — готовый cache/union_1d.csv (только чтение ISS).

  python -m pairsignal.research.moex.momentum_wf
  python -m pairsignal.research.moex.momentum_wf --long-only --market-ma 50
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
COV_MIN = 0.98  # порог покрытия тикера (отбрасываем недавно листнувшиеся)


def load_prices(cov_min: float = COV_MIN) -> pd.DataFrame:
    """Дневные закрытия выровненного юниверса MOEX. timeframe='1d' в attrs (для Sharpe)."""
    df = pd.read_csv(CACHE / "union_1d.csv", index_col=0)
    cov = df.notna().sum()
    keep = [t for t in df.columns if cov[t] >= cov_min * len(df)]
    sub = df[keep].dropna().sort_index()
    sub.attrs["timeframe"] = "1d"
    return sub


def _d(ts: int) -> str:
    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def walk_forward(prices: pd.DataFrame, train_bars: int, test_bars: int,
                 lookbacks: list[int], holdings: list[int], k: int,
                 long_short: bool, fee: float, slippage: float,
                 market_ma: int = 0) -> list[dict]:
    """Скользящие окна: train подбирает (lookback, holding) по Sharpe, test меряет OOS.

    Возвращает список окон с OOS-метриками + хвост equity-кривой test-сегмента.
    market_ma>0 (long-only): рыночный фильтр (индекс<SMA → кэш) применяется и на train,
    и на test одинаково.
    """
    out: list[dict] = []
    grid = list(itertools.product(lookbacks, holdings))
    start = 0
    while start + train_bars + test_bars <= len(prices):
        train = prices.iloc[start:start + train_bars]
        test = prices.iloc[start + train_bars:start + train_bars + test_bars]

        best, best_sharpe = None, -np.inf
        for lb, hd in grid:
            if lb >= len(train):
                continue
            m = backtest_momentum(train, lb, hd, k, long_short, fee, slippage, market_ma)
            if m["sharpe"] > best_sharpe:
                best_sharpe, best = m["sharpe"], (lb, hd)
        if best is None:
            start += test_bars
            continue

        lb, hd = best
        # warmup: хвост train (нужен и для lookback, и для market_ma SMA)
        warm_n = max(lb, market_ma)
        warm = prices.iloc[start + train_bars - warm_n:start + train_bars + test_bars]
        oos = backtest_momentum(warm, lb, hd, k, long_short, fee, slippage, market_ma)
        oos_eq = oos["equity"].iloc[-len(test):]
        oos_ret = float(oos_eq.iloc[-1] / oos_eq.iloc[0] - 1.0) * 100

        out.append({
            "test_start": int(test.index[0]), "test_end": int(test.index[-1]),
            "lookback": lb, "holding": hd,
            "oos_return_pct": round(oos_ret, 2),
            "oos_sharpe": oos["sharpe"], "oos_max_dd_pct": oos["max_dd_pct"],
            "avg_turnover": oos["avg_turnover"], "costs_pct": oos["costs_pct"],
            "_oos_eq": oos_eq,
        })
        start += test_bars
    return out


def stitch(rows: list[dict]) -> dict:
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
    sharpe = float(fr.mean() / fr.std(ddof=0) * np.sqrt(252)) if fr.std(ddof=0) > 0 else 0.0
    maxdd = float((full_eq / full_eq.cummax() - 1.0).min() * 100)
    wins = sum(1 for r in rows if r["oos_return_pct"] > 0)
    return {"total_ret": total_ret, "sharpe": sharpe, "maxdd": maxdd,
            "wins": wins, "n": len(rows), "equity": full_eq}


def _print_table(rows: list[dict]) -> None:
    hdr = ["window", "lb", "hd", "oos_ret%", "sharpe", "maxDD%", "turn", "cost%"]
    w = {h: len(h) for h in hdr}
    disp = []
    for r in rows:
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
    for d in disp:
        print("  ".join(str(d[h]).ljust(w[h]) for h in hdr))


def run_mode(prices, train_bars, test_bars, lookbacks, holdings, k,
             long_short, fee, slippage, market_ma, label):
    rows = walk_forward(prices, train_bars, test_bars, lookbacks, holdings, k,
                        long_short, fee, slippage, market_ma)
    print(f"\n=== {label} ===")
    if not rows:
        print("Недостаточно истории для walk-forward.")
        return None
    _print_table(rows)
    agg = stitch(rows)
    print(f"\nСклеенный OOS: return {agg['total_ret']:+.2f}%, Sharpe {agg['sharpe']:.2f}, "
          f"maxDD {agg['maxdd']:.2f}%, прибыльных окон {agg['wins']}/{agg['n']}")
    return rows, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--long-only", action="store_true")
    ap.add_argument("--market-ma", type=int, default=0)
    ap.add_argument("--train-days", type=int, default=378)  # ~1.5y торговых
    ap.add_argument("--test-days", type=int, default=126)   # ~0.5y торговых
    ap.add_argument("--fee", type=float, default=0.0005)    # 0.05% комиссия/нога
    ap.add_argument("--slippage", type=float, default=0.0003)  # 0.03% слиппедж/нога
    args = ap.parse_args()

    prices = load_prices()
    print(f"Юниверс: {prices.shape[1]} тикеров, {len(prices)} баров "
          f"({_d(int(prices.index[0]))} … {_d(int(prices.index[-1]))})")
    rets = prices.pct_change().dropna()
    ew = rets.mean(axis=1)
    bh_eq = (1 + ew).cumprod()
    print(f"BuyHold EW корзины: {(bh_eq.iloc[-1]-1)*100:+.2f}%, "
          f"Sharpe {ew.mean()/ew.std()*np.sqrt(252):.2f}")

    lookbacks = [20, 40, 60, 120, 240]
    holdings = [5, 10, 20]
    print(f"\nСетка: lookbacks={lookbacks}, holdings={holdings}, k={args.k}")
    print(f"Walk-forward: train={args.train_days} / test={args.test_days} баров, "
          f"издержки fee={args.fee} slip={args.slippage}/нога")

    long_short = not args.long_only
    label = (f"L/S k={args.k}" if long_short else
             f"LONG-ONLY k={args.k}" + (f" +MA{args.market_ma}" if args.market_ma else ""))
    run_mode(prices, args.train_days, args.test_days, lookbacks, holdings, args.k,
             long_short, args.fee, args.slippage, args.market_ma, label)


if __name__ == "__main__":
    main()
