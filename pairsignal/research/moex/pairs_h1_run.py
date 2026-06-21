"""Главный прогон: часовой парный mean-reversion на реальных данных MOEX.

  .venv/bin/python -m pairsignal.research.moex.pairs_h1_run

Вывод flush'ится построчно. Блоки:
  A. Gross vs Net на фикс-парах (есть ли эдж до издержек и переживает ли он косты).
  B. Net-оптимальный выход (широкий z_exit) на фикс-парах.
  C. Честный OOS walk-forward (отбор пары на train → торговля на test).
  D. Бонферрони при OOS-отборе.
  E. Робастность лучшей фикс-пары: half-split + jackknife.
  F. BuyHold бенч.
Реальные тейкер-косты (0.05%+0.03% slip на ногу) и лоу-кост (маркет-мейкер) сценарий.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pairsignal.st6.core import Params  # noqa: E402
from pairsignal.research.moex.wf_core import walk_forward  # noqa: E402
from pairsignal.research.moex.pairs_h1_core import (  # noqa: E402
    BARS_PER_YEAR, CORE_TICKERS, load_union, load_pair,
    summarize, half_split, jackknife_tstat,
)

# фикс-пары, у которых на дневках/часовках вообще был гросс-эдж (отсортируем эмпирически)
FOCUS_PAIRS = [
    ("TATN", "TATNP"), ("SBER", "SBERP"),       # обычка/преф (сильнейшая коинтеграция)
    ("NLMK", "CHMF"), ("MAGN", "CHMF"), ("NLMK", "MAGN"),  # сталь
    ("LKOH", "ROSN"), ("ROSN", "SIBN"), ("SIBN", "TATN"),  # нефть
]

TAKER = (0.0005, 0.0003)   # реальные тейкер-косты на ногу
MAKER = (0.0001, 0.0001)   # оптимистичный лимит/маркет-мейкер сценарий


def P(beta=480, z=480, corr=240, ze=2.0, zx=0.5, cost=TAKER) -> Params:
    return Params(
        beta_window=beta, z_window=z, corr_window=corr,
        z_entry=ze, z_exit=zx, z_stop=3.5,
        corr_enter=0.58, corr_break=0.45,
        select_min_corr=0.60, select_max_pvalue=0.20, select_max_halflife=720.0,
        risk_fraction=0.02, fee_rate=cost[0], slippage_rate=cost[1],
    )


def L(label, s):
    return (f"{label:<26} tr={s['trades']:>4} ret={s['ret_pct']:+7.2f}% "
            f"win={s['win']:4.0f}% DD={s['dd']:5.1f}% Sh_ann={s['sharpe_ann']:+5.2f} "
            f"Sh_tr={s['sharpe_trade']:+5.2f} t={s['tstat']:+5.2f} "
            f"bps={s['mean_ret_bps']:+6.1f} LOO+={s['loo_pos_frac']:4.2f}")


def pr(*a):
    print(*a, flush=True)


TB, TE, ST = 3780, 945, 945  # train≈1г, test≈3мес часовых баров


def main() -> None:
    pr(f"=== Часовой парный MR на MOEX. BARS_PER_YEAR={BARS_PER_YEAR:.0f}, "
       f"train={TB}ч test={TE}ч ===\n")

    # -------- A. GROSS vs NET на фикс-парах (окна 480/480/240, ze2.0 zx0.5) ----
    pr("--- A. GROSS vs NET (фикс-пары, w=480/240, ze=2.0 zx=0.5) ---")
    pr(f"    {'pair':<12} {'GROSS bps/t':>11} {'GROSS t':>8} | "
       f"{'NET bps/t':>9} {'NET t':>7} {'NET ret%':>9}")
    a_rank = []
    for pa, pb in FOCUS_PAIRS:
        sub, _ = load_pair(pa, pb)
        rg = walk_forward(sub, P(cost=(0.0, 0.0)), TB, TE, ST, fixed_pair=(pa, pb))
        rn = walk_forward(sub, P(cost=TAKER), TB, TE, ST, fixed_pair=(pa, pb))
        sg, sn = summarize(rg), summarize(rn)
        a_rank.append(((pa, pb), sn))
        pr(f"    {pa+'/'+pb:<12} {sg['mean_ret_bps']:>11.1f} {sg['tstat']:>8.2f} | "
           f"{sn['mean_ret_bps']:>9.1f} {sn['tstat']:>7.2f} {sn['ret_pct']:>+8.2f}%")
    pr("")

    # -------- B. Net-оптимальный выход: широкий z_exit снижает оборот ----------
    pr("--- B. NET по выходу z_exit (фикс-пары, ze=2.0, taker) ---")
    for pa, pb in FOCUS_PAIRS[:4]:
        sub, _ = load_pair(pa, pb)
        pr(f"  {pa}/{pb}:")
        for zx in (0.3, 0.5, 1.0, 1.5):
            r = walk_forward(sub, P(ze=2.0, zx=zx, cost=TAKER), TB, TE, ST,
                             fixed_pair=(pa, pb))
            pr("    zx=%.1f " % zx + L("", summarize(r)).strip())
    pr("")

    # -------- C. Честный OOS walk-forward (отбор пары на train) ----------------
    pr("--- C. ЧЕСТНЫЙ OOS walk-forward (отбор на train, торговля на test) ---")
    prices_u, _ = load_union(CORE_TICKERS)
    for (bw, cw) in [(240, 120), (480, 240), (720, 360)]:
        for zx in (0.5, 1.0):
            r = walk_forward(prices_u, P(bw, bw, cw, zx=zx, cost=TAKER), TB, TE, ST)
            pr("  " + L(f"WF w={bw}/{cw} zx={zx}", summarize(r)))
    pr("")

    # -------- D. Бонферрони при отборе ----------------------------------------
    pr("--- D. OOS отбор с Бонферрони (n_pairs тестов) ---")
    n_pairs = len(CORE_TICKERS) * (len(CORE_TICKERS) - 1) // 2
    r = walk_forward(prices_u, P(480, 480, 240, zx=0.5, cost=TAKER), TB, TE, ST,
                     bonferroni_n=n_pairs)
    pr("  " + L("WF bonf w=480/240", summarize(r)))
    pr("")

    # -------- E. Робастность лучшей net-фикс-пары (half-split + jackknife) -----
    pr("--- E. РОБАСТНОСТЬ топ-2 net фикс-пар (taker, half-split + jackknife) ---")
    a_rank.sort(key=lambda kv: (kv[1]['tstat'] if np.isfinite(kv[1]['tstat'])
                                else -9), reverse=True)
    for (pa, pb), _ in a_rank[:2]:
        sub, _ = load_pair(pa, pb)
        pp = P(ze=2.0, zx=0.5, cost=TAKER)
        rfull = walk_forward(sub, pp, TB, TE, ST, fixed_pair=(pa, pb))
        r1, r2 = half_split(sub, pp, TB, TE, ST, fixed_pair=(pa, pb))
        s, s1, s2 = summarize(rfull), summarize(r1), summarize(r2)
        rets = np.array([t.ret_on_notional for t in rfull.trades])
        jk = jackknife_tstat(rets)
        pr(f"  {pa}/{pb}:")
        pr("    full  " + L("", s).strip())
        pr("    half1 " + L("", s1).strip())
        pr("    half2 " + L("", s2).strip())
        pr(f"    jackknife LOO mean range [{jk['loo_min']:+.5f},{jk['loo_max']:+.5f}] "
           f"pos_frac={jk['loo_pos_frac']:.2f}")
    pr("")

    # -------- F. BuyHold бенч --------------------------------------------------
    pr("--- F. BuyHold equal-weight корзины ---")
    n = len(next(iter(prices_u.values())))
    rets = [np.asarray(v)[-1] / np.asarray(v)[0] - 1 for v in prices_u.values()]
    mat = np.array([np.asarray(v, dtype=float) for v in prices_u.values()])
    dret = np.diff(mat, axis=1) / mat[:, :-1]
    port = dret.mean(axis=0)
    sh = port.mean() / port.std() * np.sqrt(BARS_PER_YEAR)
    pr(f"  EW BuyHold: total_ret={np.mean(rets)*100:+.1f}% за ~{n/BARS_PER_YEAR:.1f}лет "
       f"Sharpe_ann={sh:+.2f}")

    # -------- G. Лоу-кост сценарий (маркет-мейкер) на лучшей паре --------------
    pr("\n--- G. LOW-COST (maker 0.01%+0.01%) лучшая фикс-пара ---")
    (pa, pb), _ = a_rank[0]
    sub, _ = load_pair(pa, pb)
    for zx in (0.5, 1.0):
        r = walk_forward(sub, P(ze=2.0, zx=zx, cost=MAKER), TB, TE, ST,
                         fixed_pair=(pa, pb))
        pr(f"  {pa}/{pb} maker zx={zx} " + L("", summarize(r)).strip())


if __name__ == "__main__":
    main()
