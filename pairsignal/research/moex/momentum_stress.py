"""Стресс-тест и jackknife momentum walk-forward на акциях MOEX.

(1) сетка режимов: L/S vs long-only vs long-only+MA-фильтр, разные train/test окна,
    разные издержки;
(2) jackknife: выбиваем лучшее OOS-окно — держится ли результат на остальных;
(3) разбиение период: первая/вторая половина;
(4) разные lookback-сетки (узкая/широкая).

Цель — отделить устойчивый эдж от «3-5 окон вытащили всё».
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pairsignal.research.moex.momentum_wf import (  # noqa: E402
    load_prices, walk_forward, stitch, _d,
)

GRID_DEFAULT = ([20, 40, 60, 120, 240], [5, 10, 20])
GRID_WIDE = ([10, 20, 40, 60, 90, 120, 180, 240], [3, 5, 10, 20, 40])
GRID_NARROW = ([60, 120, 240], [10, 20])


def summarize(prices, train, test, grid, k, long_short, fee, slip, ma, name):
    lbs, hds = grid
    rows = walk_forward(prices, train, test, lbs, hds, k, long_short, fee, slip, ma)
    if not rows:
        print(f"  {name:<42} | (нет окон)")
        return None
    agg = stitch(rows)
    print(f"  {name:<42} | ret {agg['total_ret']:+7.2f}%  Sharpe {agg['sharpe']:+5.2f}  "
          f"DD {agg['maxdd']:6.1f}%  win {agg['wins']}/{agg['n']}")
    return rows, agg


def jackknife(rows, name):
    """Выбиваем по одному окну (худшее/лучшее) — устойчив ли агрегат."""
    if not rows or len(rows) < 3:
        print(f"  {name}: мало окон для jackknife")
        return
    base = stitch(rows)
    rets = [(i, r["oos_return_pct"]) for i, r in enumerate(rows)]
    best_i = max(rets, key=lambda x: x[1])[0]
    worst_i = min(rets, key=lambda x: x[1])[0]
    drop_best = stitch([r for i, r in enumerate(rows) if i != best_i])
    drop_worst = stitch([r for i, r in enumerate(rows) if i != worst_i])
    print(f"\n  Jackknife [{name}] (склейка через произведение окон):")
    print(f"    база              : ret {base['total_ret']:+7.2f}%  Sharpe {base['sharpe']:+5.2f}  "
          f"win {base['wins']}/{base['n']}")
    print(f"    − лучшее окно ({_d(rows[best_i]['test_start'])}, "
          f"{rows[best_i]['oos_return_pct']:+.1f}%): "
          f"ret {drop_best['total_ret']:+7.2f}%  Sharpe {drop_best['sharpe']:+5.2f}")
    print(f"    − худшее окно ({_d(rows[worst_i]['test_start'])}, "
          f"{rows[worst_i]['oos_return_pct']:+.1f}%): "
          f"ret {drop_worst['total_ret']:+7.2f}%  Sharpe {drop_worst['sharpe']:+5.2f}")
    # топ-3 окна по вкладу
    contrib = sorted(rows, key=lambda r: r["oos_return_pct"], reverse=True)
    top3 = contrib[:3]
    pos_sum = sum(max(r["oos_return_pct"], 0) for r in rows)
    top3_sum = sum(r["oos_return_pct"] for r in top3)
    top3_str = ", ".join(f"{_d(r['test_start'])}:{r['oos_return_pct']:+.1f}%" for r in top3)
    print(f"    топ-3 окна по доходности: {top3_str}")
    print(f"    сумма топ-3 = {top3_sum:+.1f}% из суммы положительных {pos_sum:+.1f}%")


def main():
    prices = load_prices()
    fee, slip = 0.0005, 0.0003
    print(f"Юниверс: {prices.shape[1]} тикеров, {len(prices)} баров "
          f"({_d(int(prices.index[0]))} … {_d(int(prices.index[-1]))})")
    rets = prices.pct_change().dropna()
    ew = rets.mean(axis=1)
    print(f"BuyHold EW: {((1+ew).cumprod().iloc[-1]-1)*100:+.2f}%, "
          f"Sharpe {ew.mean()/ew.std()*np.sqrt(252):.2f}\n")

    # ---- (1) основные режимы, окно по умолчанию train=378/test=126 ----
    print("=== РЕЖИМЫ (train=378 / test=126, grid default) ===")
    modes = {
        "L/S k=3":              dict(k=3, long_short=True, ma=0),
        "L/S k=4":              dict(k=4, long_short=True, ma=0),
        "L/S k=5":              dict(k=5, long_short=True, ma=0),
        "long-only k=4":        dict(k=4, long_short=False, ma=0),
        "long-only k=4 +MA50":  dict(k=4, long_short=False, ma=50),
        "long-only k=4 +MA100": dict(k=4, long_short=False, ma=100),
        "long-only k=6 +MA50":  dict(k=6, long_short=False, ma=50),
    }
    stored = {}
    for name, cfg in modes.items():
        r = summarize(prices, 378, 126, GRID_DEFAULT, cfg["k"], cfg["long_short"],
                      fee, slip, cfg["ma"], name)
        if r:
            stored[name] = r[0]

    # ---- (2) разные train/test окна (для двух самых интересных режимов) ----
    print("\n=== РАЗНЫЕ ОКНА train/test ===")
    for (tr, te) in [(252, 63), (252, 126), (378, 126), (504, 126), (504, 252)]:
        print(f"  -- train={tr} test={te} --")
        summarize(prices, tr, te, GRID_DEFAULT, 4, True, fee, slip, 0, "  L/S k=4")
        summarize(prices, tr, te, GRID_DEFAULT, 4, False, fee, slip, 50,
                  "  long-only k=4 +MA50")

    # ---- (3) разные lookback-сетки ----
    print("\n=== РАЗНЫЕ LOOKBACK-СЕТКИ (train=378/test=126) ===")
    for gname, grid in [("narrow", GRID_NARROW), ("default", GRID_DEFAULT), ("wide", GRID_WIDE)]:
        summarize(prices, 378, 126, grid, 4, True, fee, slip, 0, f"  L/S k=4 [{gname}]")
        summarize(prices, 378, 126, grid, 4, False, fee, slip, 50,
                  f"  long-only +MA50 [{gname}]")

    # ---- (4) разные издержки ----
    print("\n=== РАЗНЫЕ ИЗДЕРЖКИ (train=378/test=126, long-only+MA50) ===")
    for f, s, lbl in [(0.0, 0.0, "0 (идеал)"), (0.0004, 0.0002, "комби 0.06%"),
                      (0.0005, 0.0003, "комби 0.08%"), (0.001, 0.0005, "комби 0.15%")]:
        summarize(prices, 378, 126, GRID_DEFAULT, 4, False, f, s, 50,
                  f"  costs={lbl}")
        summarize(prices, 378, 126, GRID_DEFAULT, 4, True, f, s, 0,
                  f"  L/S costs={lbl}")

    # ---- (5) первая/вторая половина ----
    print("\n=== ПОЛОВИНЫ ПЕРИОДА (long-only+MA50, L/S) ===")
    half = len(prices) // 2
    for hname, sub in [("1-я половина", prices.iloc[:half]),
                       ("2-я половина", prices.iloc[half:])]:
        sub.attrs["timeframe"] = "1d"
        print(f"  -- {hname} ({_d(int(sub.index[0]))}…{_d(int(sub.index[-1]))}, {len(sub)} баров) --")
        summarize(sub, 252, 126, GRID_DEFAULT, 4, False, fee, slip, 50, "  long-only +MA50")
        summarize(sub, 252, 126, GRID_DEFAULT, 4, True, fee, slip, 0, "  L/S k=4")

    # ---- (6) jackknife для каждого «выживающего» режима ----
    print("\n=== JACKKNIFE топ-окон ===")
    for name in ["L/S k=4", "long-only k=4 +MA50", "long-only k=6 +MA50"]:
        if name in stored:
            jackknife(stored[name], name)


if __name__ == "__main__":
    main()
