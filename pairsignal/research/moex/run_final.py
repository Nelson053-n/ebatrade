"""Финальные проверки: (1) можно ли спасти эдж сменой z_stop/time_stop (OOS);
(2) чистое сравнение in-sample vs OOS — насколько отбор завышает.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pairsignal.st6.core import Params, rank_pairs  # noqa: E402
from pairsignal.st6.backtest import backtest_pair  # noqa: E402
from pairsignal.research.moex.wf_core import walk_forward  # noqa: E402
from pairsignal.research.moex.run_experiments import load_prices, CORE_TICKERS, FIXED_PAIRS  # noqa: E402


def bp(**over):
    d = dict(beta_window=240, z_window=240, corr_window=120,
             z_entry=2.0, z_exit=0.3, z_stop=3.5, corr_enter=0.58, corr_break=0.45,
             select_min_corr=0.60, select_max_pvalue=0.20, select_max_halflife=400.0,
             risk_fraction=0.02, fee_rate=0.0006, slippage_rate=0.0005)
    d.update(over); return Params(**d)


def pooled_oos(prices, p, tr=500, te=250, st=250):
    out = []
    for pa, pb in FIXED_PAIRS:
        if pa not in prices or pb not in prices:
            continue
        r = walk_forward({pa: prices[pa], pb: prices[pb]}, p, tr, te, st, fixed_pair=(pa, pb))
        out.extend(t.ret_on_notional for t in r.trades)
    return np.array(out)


def tinfo(x):
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float("nan"), float("nan")
    return x.mean(), x.mean()/(x.std(ddof=1)/np.sqrt(len(x)))


def main():
    prices, _ = load_prices(CORE_TICKERS)

    print("--- 1. Спасает ли смена z_stop / time_stop / z_exit? (pooled OOS, фикс-пары) ---")
    print("    cfg                        | n   mean      t    win   sum%")
    cfgs = {
        "default z_stop3.5": bp(),
        "z_stop4.5 (шире стоп)": bp(z_stop=4.5),
        "z_stop3.0 (уже стоп)": bp(z_stop=3.0),
        "z_exit0.0 (до средней)": bp(z_exit=0.0),
        "z_exit1.0 (ранний тейк)": bp(z_exit=1.0),
        "time_stop20 + z_stop4.5": bp(z_stop=4.5, time_stop_bars=20),
        "z_entry1.5 z_stop4.5": bp(z_entry=1.5, z_stop=4.5),
        "corr_enter0.7 (строже гейт)": bp(corr_enter=0.70),
    }
    for name, p in cfgs.items():
        r = pooled_oos(prices, p)
        m, t = tinfo(r)
        print(f"    {name:<26} | {len(r):>3} {m*100:+6.3f}%  {t:+5.2f}  "
              f"{100*np.mean(r>0) if len(r) else 0:4.0f}%  {r.sum()*100 if len(r) else 0:+7.2f}%")

    print("\n--- 2. IN-SAMPLE vs OOS на одних и тех же фикс-парах (фактор завышения) ---")
    p = bp()
    print("    pair        | IS_ret(full)  OOS_sum   IS-факт")
    for pa, pb in FIXED_PAIRS:
        if pa not in prices or pb not in prices:
            continue
        # in-sample: прогон по всей истории (как боевой backtest_pair)
        ris = backtest_pair(prices[pa], prices[pb], p, start_equity=1_000_000.0)
        # для сравнимости считаем сумму per-trade ret-on-notional in-sample
        is_rets = []
        for t in ris.trades:
            units_a = t.entry_a  # qty=1 в backtest_pair при lot=1; нотионал=entry_a*units
            # backtest_pair не отдаёт notional, аппроксимируем через net/ (нет qty) — пропустим,
            # сравним просто по числу сделок и направлению
            pass
        roos = walk_forward({pa: prices[pa], pb: prices[pb]}, p, 500, 250, 250, fixed_pair=(pa, pb))
        oos_sum = sum(t.ret_on_notional for t in roos.trades) * 100
        print(f"    {pa}/{pb:<6} | IS trades={ris.n_trades:>2} win={ris.win_rate*100 if ris.n_trades else 0:.0f}%"
              f"  | OOS sum-ret={oos_sum:+6.2f}% trades={roos.n_trades}")

    print("\n--- 3. IN-SAMPLE отбор лучшей пары по rank_pairs на ВСЕЙ истории vs её OOS ---")
    p = bp()
    ranked = rank_pairs(prices, p)
    print("    Топ пары по in-sample score и их ЧЕСТНЫЙ OOS sum-ret:")
    for s in ranked[:6]:
        roos = walk_forward({s.a: prices[s.a], s.b: prices[s.b]}, p, 500, 250, 250,
                            fixed_pair=(s.a, s.b))
        oos = sum(t.ret_on_notional for t in roos.trades) * 100
        print(f"    {s.a}/{s.b:<6} IS_score={s.score:.3f} corr={s.corr:.2f} p={s.pvalue:.3f}"
              f"  ->  OOS sum-ret={oos:+6.2f}% ({roos.n_trades} сделок)")


if __name__ == "__main__":
    main()
