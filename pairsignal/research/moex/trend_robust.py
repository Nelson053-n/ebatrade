"""trend_robust — стресс-тесты trend-following (КРИТИЧНО для тренд-стратегий).

1. half-split: WF отдельно на ПЕРВОЙ и ВТОРОЙ половине истории. Тренд-фолловинг
   часто живёт только в трендовый режим (2020-23) — если вторая половина <0, эджа нет.
2. jackknife по тикерам: выкидываем по одному тикер, смотрим разброс склеенного OOS
   Sharpe (нет ли одного-двух тикеров, которые тащат весь результат).
3. cost stress: WF при разных издержках (0.0004 .. 0.0010 на сторону).
4. fixed-param half: фиксируем ОДИН набор параметров (победитель IS) и меряем equity
   в каждой половине отдельно — без всякого подбора, чистый режимный тест.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from trend_core import (_BARS_PER_YEAR, basket_metrics, buyhold_ew, d, load_closes,
                        metrics, sig_donchian, sig_ma_cross, sig_tsmom)
from trend_wf import make_grid, walk_forward


def _bph(interval):
    return _BARS_PER_YEAR[interval] / (252 if interval == "1d" else 365)


def half_split_wf(closes, family, interval, train_days, test_days, fee, slip, warm):
    bph = _bph(interval)
    tb, xb = int(train_days * bph), int(test_days * bph)
    mid = len(closes) // 2
    halves = {"H1 (первая)": closes.iloc[:mid], "H2 (вторая)": closes.iloc[mid:]}
    print(f"\n### HALF-SPLIT WF [{family} {interval}] ###")
    for name, sub in halves.items():
        r0 = d(int(sub.index[0])); r1 = d(int(sub.index[-1]))
        res = walk_forward(sub, family, interval, tb, xb, fee, slip, warm, verbose=False)
        if not res:
            print(f"{name} [{r0}..{r1}]: недостаточно окон")
            continue
        print(f"{name} [{r0}..{r1}]: ret {res['ret_pct']:+.1f}%  Sharpe {res['sharpe']:.2f}  "
              f"DD {res['max_dd_pct']:.1f}%  окна {res['wins']}/{res['n_windows']}")


def fixed_param_half(closes, family, interval, params, fee, slip):
    """Фиксированный набор параметров (без подбора) — equity в каждой половине."""
    _, mkfn = make_grid(family, interval)
    sig_fn = mkfn(params)
    mid = len(closes) // 2
    print(f"\n### FIXED-PARAM по половинам [{family} {params}] (без подбора) ###")
    for name, sub in {"FULL": closes, "H1": closes.iloc[:mid], "H2": closes.iloc[mid:]}.items():
        m = basket_metrics(sub, sig_fn, fee, slip, interval)
        r0 = d(int(sub.index[0])); r1 = d(int(sub.index[-1]))
        bh = buyhold_ew(sub, interval)
        print(f"{name:5} [{r0}..{r1}]: strat ret {m['ret_pct']:+7.1f}%  Sharpe {m['sharpe']:>5.2f}  "
              f"DD {m['max_dd_pct']:>6.1f}%   | BH ret {bh['ret_pct']:+6.1f}% Sh {bh['sharpe']:.2f}")


def jackknife(closes, family, interval, train_days, test_days, fee, slip, warm):
    bph = _bph(interval)
    tb, xb = int(train_days * bph), int(test_days * bph)
    base = walk_forward(closes, family, interval, tb, xb, fee, slip, warm, verbose=False)
    print(f"\n### JACKKNIFE по тикерам [{family} {interval}] (база Sharpe {base['sharpe']:.2f}) ###")
    sharpes = []
    for t in closes.columns:
        sub = closes.drop(columns=[t])
        res = walk_forward(sub, family, interval, tb, xb, fee, slip, warm, verbose=False)
        sharpes.append((t, res["sharpe"], res["ret_pct"]))
    sharpes.sort(key=lambda x: x[1])
    arr = np.array([s for _, s, _ in sharpes])
    print(f"OOS Sharpe при выкидывании 1 тикера: min {arr.min():.2f} ({sharpes[0][0]}), "
          f"max {arr.max():.2f} ({sharpes[-1][0]}), median {np.median(arr):.2f}")
    print("наиболее влиятельные (выкид сильнее всего роняет Sharpe — значит тащили результат):")
    for t, sh, rp in sorted(sharpes, key=lambda x: x[1])[:5]:
        print(f"  без {t}: Sharpe {sh:.2f} (ret {rp:+.1f}%)")


def cost_stress(closes, family, interval, train_days, test_days, warm):
    bph = _bph(interval)
    tb, xb = int(train_days * bph), int(test_days * bph)
    print(f"\n### COST STRESS [{family} {interval}] ###")
    print(f"{'costs/side':>11} {'OOS ret%':>9} {'Sharpe':>7} {'DD%':>7} {'wins':>6}")
    for tot in [0.0004, 0.0006, 0.0008, 0.0010, 0.0014]:
        fee, slip = tot * 0.6, tot * 0.4
        res = walk_forward(closes, family, interval, tb, xb, fee, slip, warm, verbose=False)
        print(f"{tot:>11.4f} {res['ret_pct']:>+9.1f} {res['sharpe']:>7.2f} "
              f"{res['max_dd_pct']:>7.1f} {res['wins']}/{res['n_windows']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=["donchian", "ma", "tsmom"])
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--train-days", type=int, default=365)
    ap.add_argument("--test-days", type=int, default=90)
    ap.add_argument("--fee", type=float, default=0.0005)
    ap.add_argument("--slip", type=float, default=0.0003)
    ap.add_argument("--fixed", default="", help="фикс параметры 'en,ex,ls' donchian / 'f,s,ls' ma / 'lb,ls' tsmom")
    args = ap.parse_args()

    closes = load_closes(args.interval)
    warm = 800 if args.interval == "1h" else 200
    print(f"=== ROBUST {args.family} {args.interval}: {closes.shape[1]} тикеров, {closes.shape[0]} баров ===")

    half_split_wf(closes, args.family, args.interval, args.train_days, args.test_days,
                  args.fee, args.slip, warm)
    if args.fixed:
        parts = args.fixed.split(",")
        if args.family == "tsmom":
            params = (int(parts[0]), parts[1].lower() in ("1", "true", "t"))
        else:
            params = (int(parts[0]), int(parts[1]), parts[2].lower() in ("1", "true", "t"))
        fixed_param_half(closes, args.family, args.interval, params, args.fee, args.slip)
    cost_stress(closes, args.family, args.interval, args.train_days, args.test_days, warm)
    jackknife(closes, args.family, args.interval, args.train_days, args.test_days,
              args.fee, args.slip, warm)


if __name__ == "__main__":
    main()
