"""Стресс-тест и jackknife momentum walk-forward на ЧАСОВЫХ акциях MOEX.

Десятки OOS окон (часовик) позволяют честно судить об устойчивости:
(1) сетка режимов: L/S vs long-only vs long-only+MA-фильтр, разные k;
(2) разные train/test окна;
(3) разные lookback-сетки (узкая/широкая);
(4) разные издержки;
(5) первая/вторая половина периода;
(6) jackknife: выбиваем лучшее/худшее OOS-окно + вклад топ-окон.

Цель — отделить устойчивый эдж от «несколько окон вытащили всё».

  python -m pairsignal.research.moex.momentum_h1_stress
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pairsignal.research.moex.momentum_h1_wf import (  # noqa: E402
    load_prices, walk_forward, stitch, _d, _sharpe,
)

# lookbacks/holdings в ЧАСАХ
GRID_DEFAULT = ([24, 48, 96, 168, 336], [12, 24, 48])
GRID_WIDE = ([12, 24, 48, 72, 96, 120, 168, 240, 336], [6, 12, 24, 48, 96])
GRID_NARROW = ([48, 96, 168], [24, 48])


def summarize(prices, train, test, grid, k, long_short, fee, slip, ma, name, bpy):
    lbs, hds = grid
    rows = walk_forward(prices, train, test, lbs, hds, k, long_short, fee, slip, ma, bpy)
    if not rows:
        print(f"  {name:<44} | (нет окон)")
        return None
    agg = stitch(rows, bpy)
    print(f"  {name:<44} | ret {agg['total_ret']:+9.2f}%  Sharpe {agg['sharpe']:+5.2f}  "
          f"DD {agg['maxdd']:7.1f}%  win {agg['wins']}/{agg['n']}")
    return rows, agg


def jackknife(rows, name, bpy):
    """Выбиваем по одному окну (лучшее/худшее) + вклад топ-окон — устойчив ли агрегат."""
    if not rows or len(rows) < 3:
        print(f"  {name}: мало окон для jackknife")
        return
    base = stitch(rows, bpy)
    rets = [(i, r["oos_return_pct"]) for i, r in enumerate(rows)]
    best_i = max(rets, key=lambda x: x[1])[0]
    worst_i = min(rets, key=lambda x: x[1])[0]
    drop_best = stitch([r for i, r in enumerate(rows) if i != best_i], bpy)
    drop_worst = stitch([r for i, r in enumerate(rows) if i != worst_i], bpy)
    print(f"\n  Jackknife [{name}] ({base['n']} окон):")
    print(f"    база              : ret {base['total_ret']:+9.2f}%  Sharpe {base['sharpe']:+5.2f}  "
          f"win {base['wins']}/{base['n']}")
    print(f"    − лучшее окно ({_d(rows[best_i]['test_start'])}, "
          f"{rows[best_i]['oos_return_pct']:+.1f}%): "
          f"ret {drop_best['total_ret']:+9.2f}%  Sharpe {drop_best['sharpe']:+5.2f}")
    print(f"    − худшее окно ({_d(rows[worst_i]['test_start'])}, "
          f"{rows[worst_i]['oos_return_pct']:+.1f}%): "
          f"ret {drop_worst['total_ret']:+9.2f}%  Sharpe {drop_worst['sharpe']:+5.2f}")
    contrib = sorted(rows, key=lambda r: r["oos_return_pct"], reverse=True)
    topn = contrib[:max(3, len(rows) // 10)]
    pos_sum = sum(max(r["oos_return_pct"], 0) for r in rows)
    topn_sum = sum(r["oos_return_pct"] for r in topn)
    print(f"    топ-{len(topn)} окон сумма = {topn_sum:+.1f}% из суммы положительных "
          f"{pos_sum:+.1f}% ({100*topn_sum/pos_sum:.0f}% вклада от {100*len(topn)/len(rows):.0f}% окон)")
    # доля прибыльных окон уже в win; распределение знака
    pos = sum(1 for r in rows if r["oos_return_pct"] > 0)
    print(f"    прибыльных окон: {pos}/{len(rows)} ({100*pos/len(rows):.0f}%)")


def main():
    prices = load_prices()
    bpy = prices.attrs["bpy"]
    fee, slip = 0.0005, 0.0003
    print(f"Юниверс: {prices.shape[1]} тикеров, {len(prices)} часовых баров "
          f"({_d(int(prices.index[0]))} … {_d(int(prices.index[-1]))}); bars/year≈{bpy:.0f}")
    rets = prices.pct_change().dropna()
    ew = rets.mean(axis=1)
    print(f"BuyHold EW: {((1+ew).cumprod().iloc[-1]-1)*100:+.2f}%, "
          f"Sharpe {_sharpe(ew, bpy):.2f}\n")

    TR, TE = 2520, 504  # ~1 год train, ~2.5 мес test (в торговых часах)

    # ---- (1) основные режимы ----
    print(f"=== РЕЖИМЫ (train={TR} / test={TE} ч, grid default) ===")
    modes = {
        "L/S k=3":               dict(k=3, long_short=True, ma=0),
        "L/S k=4":               dict(k=4, long_short=True, ma=0),
        "L/S k=5":               dict(k=5, long_short=True, ma=0),
        "L/S k=6":               dict(k=6, long_short=True, ma=0),
        "long-only k=4":         dict(k=4, long_short=False, ma=0),
        "long-only k=4 +MA200":  dict(k=4, long_short=False, ma=200),
        "long-only k=4 +MA400":  dict(k=4, long_short=False, ma=400),
        "long-only k=6 +MA200":  dict(k=6, long_short=False, ma=200),
    }
    stored = {}
    for name, cfg in modes.items():
        r = summarize(prices, TR, TE, GRID_DEFAULT, cfg["k"], cfg["long_short"],
                      fee, slip, cfg["ma"], name, bpy)
        if r:
            stored[name] = r[0]

    # ---- (2) разные train/test окна ----
    print("\n=== РАЗНЫЕ ОКНА train/test (часы) ===")
    for (tr, te) in [(1260, 252), (2520, 252), (2520, 504), (3780, 504), (5040, 1008)]:
        print(f"  -- train={tr} test={te} --")
        summarize(prices, tr, te, GRID_DEFAULT, 4, True, fee, slip, 0, "  L/S k=4", bpy)
        summarize(prices, tr, te, GRID_DEFAULT, 4, False, fee, slip, 200,
                  "  long-only k=4 +MA200", bpy)

    # ---- (3) разные lookback-сетки ----
    print(f"\n=== РАЗНЫЕ LOOKBACK-СЕТКИ (train={TR}/test={TE}) ===")
    for gname, grid in [("narrow", GRID_NARROW), ("default", GRID_DEFAULT), ("wide", GRID_WIDE)]:
        summarize(prices, TR, TE, grid, 4, True, fee, slip, 0, f"  L/S k=4 [{gname}]", bpy)
        summarize(prices, TR, TE, grid, 4, False, fee, slip, 200,
                  f"  long-only +MA200 [{gname}]", bpy)

    # ---- (4) разные издержки ----
    print(f"\n=== РАЗНЫЕ ИЗДЕРЖКИ (train={TR}/test={TE}) ===")
    for f, s, lbl in [(0.0, 0.0, "0 (идеал)"), (0.0004, 0.0002, "комби 0.06%"),
                      (0.0005, 0.0003, "комби 0.08%"), (0.001, 0.0005, "комби 0.15%")]:
        summarize(prices, TR, TE, GRID_DEFAULT, 4, True, f, s, 0, f"  L/S costs={lbl}", bpy)
        summarize(prices, TR, TE, GRID_DEFAULT, 4, False, f, s, 200,
                  f"  long-only+MA200 costs={lbl}", bpy)

    # ---- (5) первая/вторая половина ----
    print("\n=== ПОЛОВИНЫ ПЕРИОДА ===")
    half = len(prices) // 2
    for hname, sub in [("1-я половина", prices.iloc[:half]),
                       ("2-я половина", prices.iloc[half:])]:
        sub = sub.copy()
        sub.attrs["bpy"] = bpy
        print(f"  -- {hname} ({_d(int(sub.index[0]))}…{_d(int(sub.index[-1]))}, {len(sub)} баров) --")
        summarize(sub, 2520, 504, GRID_DEFAULT, 4, True, fee, slip, 0, "  L/S k=4", bpy)
        summarize(sub, 2520, 504, GRID_DEFAULT, 4, False, fee, slip, 200,
                  "  long-only +MA200", bpy)

    # ---- (6) jackknife ----
    print("\n=== JACKKNIFE топ-окон ===")
    for name in ["L/S k=4", "L/S k=5", "long-only k=4 +MA200", "long-only k=6 +MA200"]:
        if name in stored:
            jackknife(stored[name], name, bpy)


if __name__ == "__main__":
    main()
