"""Прогон всех семейств алгоритмов через walk-forward на закэшированных данных.

  python -m pairsignal.research.crypto.run_experiments --exchange gate --top 40 \
      --train-days 90 --test-days 30 --fee 0.0006 --slippage 0.0002

Печатает агрегированные OOS-метрики (склеенный портфель по всем окнам) для каждого
семейства + BuyHold-бенчмарк. Издержки реальные (тейкер 0.06%/нога + slippage).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import pandas as pd

from .load_data import get_prices
from . import wf_engine as wf


def _d(ts: int) -> str:
    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def run_family(name: str, prices, train_bars, test_bars, grid, fn, warm_max,
               fee, slippage, tf) -> dict:
    rows = wf.walk_forward(prices, train_bars, test_bars, grid, fn, warm_max, fee, slippage)
    agg = wf.stitch(rows, tf)
    agg["family"] = name
    agg["_rows"] = rows
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchange", default="gate")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--days", type=int, default=400)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--train-days", type=int, default=90)
    ap.add_argument("--test-days", type=int, default=30)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--fee", type=float, default=0.0006)
    ap.add_argument("--slippage", type=float, default=0.0002)
    args = ap.parse_args()

    tf = args.timeframe
    bph = wf.bars_per_year(tf) / 365
    train_bars = int(args.train_days * bph)
    test_bars = int(args.test_days * bph)

    prices = get_prices(args.exchange, tf, args.days, args.top)
    prices.attrs["timeframe"] = tf
    print(f"Юниверс: {prices.shape[1]} монет, {len(prices)} баров "
          f"({_d(int(prices.index[0]))} … {_d(int(prices.index[-1]))})")
    print(f"Walk-forward: train={train_bars} ({args.train_days}д) / "
          f"test={test_bars} ({args.test_days}д) баров | fee={args.fee} slip={args.slippage}\n")

    k = args.k
    # сетки параметров (подбираются на train по Sharpe)
    lookbacks = [12, 24, 48, 96, 168, 240]
    holdings = [6, 12, 24, 48]
    mas = [0, 48, 96, 168]

    results = []

    # 1. CS-momentum long/short
    grid_ls = [(lb, hd, k, True, 0) for lb in lookbacks for hd in holdings]
    results.append(run_family("csmom_LS", prices, train_bars, test_bars, grid_ls,
                              wf.backtest_csmom, max(lookbacks), args.fee, args.slippage, tf))

    # 2. CS-momentum long-only + market filter
    grid_lo = [(lb, hd, k, False, ma) for lb in lookbacks for hd in holdings for ma in mas]
    results.append(run_family("csmom_LO_filt", prices, train_bars, test_bars, grid_lo,
                              wf.backtest_csmom, max(max(lookbacks), max(mas)),
                              args.fee, args.slippage, tf))

    # 3. TS-momentum long/short
    grid_ts_ls = [(lb, hd, True) for lb in lookbacks for hd in holdings]
    results.append(run_family("tsmom_LS", prices, train_bars, test_bars, grid_ts_ls,
                              wf.backtest_tsmom, max(lookbacks), args.fee, args.slippage, tf))

    # 4. TS-momentum long-only
    grid_ts_lo = [(lb, hd, False) for lb in lookbacks for hd in holdings]
    results.append(run_family("tsmom_LO", prices, train_bars, test_bars, grid_ts_lo,
                              wf.backtest_tsmom, max(lookbacks), args.fee, args.slippage, tf))

    # 5. Donchian breakout long/short
    grid_dc_ls = [(e, x, True) for e in [24, 48, 96, 168] for x in [12, 24, 48]]
    results.append(run_family("donchian_LS", prices, train_bars, test_bars, grid_dc_ls,
                              wf.backtest_donchian, 168, args.fee, args.slippage, tf))

    # 6. Donchian breakout long-only
    grid_dc_lo = [(e, x, False) for e in [24, 48, 96, 168] for x in [12, 24, 48]]
    results.append(run_family("donchian_LO", prices, train_bars, test_bars, grid_dc_lo,
                              wf.backtest_donchian, 168, args.fee, args.slippage, tf))

    # 7. Cross-sectional mean-reversion (честный симметричный стоп, без profit_target)
    grid_mr = [(lb, ez, sz, mh) for lb in [24, 48, 96, 168]
               for ez in [1.5, 2.0, 2.5] for sz in [3.0, 4.0] for mh in [24, 48, 96]]
    results.append(run_family("xmr", prices, train_bars, test_bars, grid_mr,
                              wf.backtest_xmr, 168, args.fee, args.slippage, tf))

    # BuyHold бенчмарк (полный период, не WF — для контекста)
    bh = wf.backtest_buyhold(prices, args.fee, args.slippage)
    bh_m = wf.metrics_from_net(bh["net"], tf)

    # вывод
    hdr = ["family", "OOS_return%", "OOS_sharpe", "OOS_maxDD%", "win_win", "n_win"]
    print(f"{'family':<16}{'OOS_ret%':>10}{'Sharpe':>9}{'maxDD%':>9}{'win/tot':>10}")
    print("-" * 54)
    for r in sorted(results, key=lambda x: x.get("sharpe", -99), reverse=True):
        if not r:
            continue
        print(f"{r['family']:<16}{r['return_pct']:>10.2f}{r['sharpe']:>9.2f}"
              f"{r['max_dd_pct']:>9.2f}{r['win_windows']:>6}/{r['n_windows']:<3}")
    print("-" * 54)
    print(f"{'BuyHold(EW)':<16}{bh_m['return_pct']:>10.2f}{bh_m['sharpe']:>9.2f}"
          f"{bh_m['max_dd_pct']:>9.2f}")

    # детали по лучшему семейству
    best = max((r for r in results if r), key=lambda x: x.get("sharpe", -99))
    print(f"\n=== ЛУЧШЕЕ: {best['family']} — окна OOS ===")
    print(f"{'window':<26}{'params':<24}{'ret%':>8}{'sharpe':>8}{'maxDD%':>8}")
    for r in best["_rows"]:
        win = f"{_d(r['test_start'])}…{_d(r['test_end'])}"
        print(f"{win:<26}{str(r['params']):<24}{r['oos_return_pct']:>8.2f}"
              f"{r['oos_sharpe']:>8.2f}{r['oos_max_dd_pct']:>8.2f}")

    # частоты выбранных параметров (на что чаще всего оседает подбор)
    from collections import Counter
    cnt = Counter(str(r["params"]) for r in best["_rows"])
    print("\nЧастоты выбранных параметров (train→test):")
    for p, c in cnt.most_common():
        print(f"  {p}: {c}")


if __name__ == "__main__":
    main()
