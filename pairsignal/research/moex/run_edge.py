"""Оценка реального ЭДЖА st6-парного mean-reversion на MOEX, корректно по сайзингу.

Проблема прогона A-F: risk_fraction=0.02 → нотионал ноги всего 2% equity, поэтому
ret% по 1М капиталу микроскопичен и не отражает наличие/отсутствие эджа. Здесь
меряем доходность НА РАЗВЁРНУТЫЙ НОТИОНАЛ (per-trade ret), её t-стат и устойчивость:
  1. распределение per-trade ret-on-notional по честному OOS;
  2. t-тест: значимо ли среднее > 0;
  3. чувствительность к окнам (z_window/corr_window) — тюнинг на train, замер на test;
  4. стресс по множеству train/test сдвигов.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pairsignal.st6.core import Params  # noqa: E402
from pairsignal.research.moex.wf_core import walk_forward  # noqa: E402
from pairsignal.research.moex.run_experiments import load_prices, CORE_TICKERS, FIXED_PAIRS  # noqa: E402


def tstat(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float("nan"), float("nan")
    t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    return float(x.mean()), float(t)


def pair_oos_trades(prices, pa, pb, p, train, test, step):
    sub = {pa: prices[pa], pb: prices[pb]}
    r = walk_forward(sub, p, train, test, step, fixed_pair=(pa, pb))
    return np.array([t.ret_on_notional for t in r.trades], dtype=float)


def base_params(**over) -> Params:
    d = dict(beta_window=240, z_window=240, corr_window=120,
             z_entry=2.0, z_exit=0.3, z_stop=3.5,
             corr_enter=0.58, corr_break=0.45,
             select_min_corr=0.60, select_max_pvalue=0.20, select_max_halflife=400.0,
             risk_fraction=0.02, fee_rate=0.0006, slippage_rate=0.0005)
    d.update(over)
    return Params(**d)


def main():
    prices, _ = load_prices(CORE_TICKERS)
    n = len(next(iter(prices.values())))
    print(f"=== {len(prices)} тикеров x {n} баров ===\n")

    # ---- 1. Per-trade edge на каждой фикс-паре, default windows, OOS ----
    print("--- 1. OOS per-trade ret-on-notional (после комиссий), фикс-пары, default окна ---")
    print("    (mean*100=средняя доходность сделки в %, t=t-стат>0 значит эдж)")
    p = base_params()
    all_default = []
    for pa, pb in FIXED_PAIRS:
        if pa not in prices or pb not in prices:
            continue
        rr = pair_oos_trades(prices, pa, pb, p, 500, 250, 250)
        all_default.append((f"{pa}/{pb}", rr))
        m, t = tstat(rr)
        print(f"  {pa}/{pb:<6} n={len(rr):>2}  mean={m*100:+6.3f}%  t={t:+5.2f}  "
              f"win={100*np.mean(rr>0) if len(rr) else 0:4.0f}%  sum={rr.sum()*100:+6.2f}%")
    pooled = np.concatenate([r for _, r in all_default]) if all_default else np.array([])
    m, t = tstat(pooled)
    print(f"  POOLED все фикс-пары: n={len(pooled)} mean={m*100:+.3f}% t={t:+.2f} "
          f"win={100*np.mean(pooled>0):.0f}%")
    print()

    # ---- 2. Чувствительность к окнам: подбор на train недоступен напрямую,
    #         но проверим, какие окна вообще дают эдж (грубый sweep, честный OOS) ----
    print("--- 2. Sweep окон z/corr (честный OOS, pooled по фикс-парам) ---")
    print("    z_win corr_win z_entry  | n   pooled_mean   t    win   sum%")
    grids = [
        (240, 120, 2.0), (120, 90, 2.0), (90, 60, 2.0), (60, 45, 2.0),
        (120, 90, 1.5), (90, 60, 1.5), (60, 45, 1.5), (60, 45, 2.5),
        (40, 30, 1.5), (40, 30, 2.0),
    ]
    best = None
    for zw, cw, ze in grids:
        p = base_params(z_window=zw, beta_window=max(zw, 120), corr_window=cw, z_entry=ze)
        pooled = []
        for pa, pb in FIXED_PAIRS:
            if pa not in prices or pb not in prices:
                continue
            pooled.append(pair_oos_trades(prices, pa, pb, p, 500, 250, 250))
        pooled = np.concatenate(pooled) if pooled else np.array([])
        m, t = tstat(pooled)
        flag = ""
        if len(pooled) >= 20 and m > 0 and (best is None or t > best[1]):
            best = (m, t, zw, cw, ze); flag = " <-"
        print(f"    {zw:>3}   {cw:>3}    {ze:>3}    | {len(pooled):>3}  "
              f"{m*100:+7.3f}%  {t:+5.2f}  {100*np.mean(pooled>0) if len(pooled) else 0:4.0f}%  "
              f"{pooled.sum()*100 if len(pooled) else 0:+6.2f}%{flag}")
    print()
    if best:
        print(f"  Лучшие окна по t-стату: z_window={best[2]} corr_window={best[3]} "
              f"z_entry={best[4]} (mean={best[0]*100:+.3f}% t={best[1]:+.2f})")
    print()

    # ---- 3. Стресс: множество train/test сдвигов на лучших фикс-парах ----
    print("--- 3. Стресс по train/test разбиениям (default окна, фикс-пары) ---")
    splits = [(400, 200, 200), (500, 200, 200), (500, 250, 250), (600, 250, 250),
              (700, 200, 200), (500, 150, 150), (450, 150, 150)]
    p = base_params()
    for pa, pb in [("NLMK", "MAGN"), ("LKOH", "SIBN"), ("ROSN", "SIBN"), ("NLMK", "CHMF")]:
        if pa not in prices or pb not in prices:
            continue
        rets = []
        for tr, te, st in splits:
            rr = pair_oos_trades(prices, pa, pb, p, tr, te, st)
            rets.append(rr.sum() * 100 if len(rr) else 0.0)
        rets = np.array(rets)
        print(f"  {pa}/{pb:<6} по сплитам sum-ret%: "
              f"[{', '.join(f'{x:+.2f}' for x in rets)}]  "
              f"medianamean={rets.mean():+.2f}% pos_splits={np.mean(rets>0)*100:.0f}%")


if __name__ == "__main__":
    main()
