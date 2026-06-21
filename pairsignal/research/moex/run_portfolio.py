"""Финальная проверка: можно ли торговать КОРЗИНУ фикс-пар без cherry-picking.

Если отдельные пары нестабильны и значимость низкая, единственный честный путь —
торговать ВСЕ экономически-обоснованные пары как портфель (диверсификация),
без выбора пары по результату. Меряем агрегированный OOS-Sharpe портфеля сделок
и проверяем, не держится ли результат на 1-2 везучих сделках (jackknife).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pairsignal.st6.core import Params  # noqa: E402
from pairsignal.research.moex.wf_core import walk_forward  # noqa: E402
from pairsignal.research.moex.run_experiments import load_prices, CORE_TICKERS, FIXED_PAIRS  # noqa: E402


def base_params(**over) -> Params:
    d = dict(beta_window=240, z_window=240, corr_window=120,
             z_entry=2.0, z_exit=0.3, z_stop=3.5,
             corr_enter=0.58, corr_break=0.45,
             select_min_corr=0.60, select_max_pvalue=0.20, select_max_halflife=400.0,
             risk_fraction=0.02, fee_rate=0.0006, slippage_rate=0.0005)
    d.update(over); return Params(**d)


def main():
    prices, _ = load_prices(CORE_TICKERS)
    p = base_params()
    tr, te, st = 500, 250, 250

    # Собираем все сделки всех фикс-пар (OOS) с временной меткой выхода
    rows = []  # (exit_i, pair, ret_on_notional, net)
    for pa, pb in FIXED_PAIRS:
        if pa not in prices or pb not in prices:
            continue
        sub = {pa: prices[pa], pb: prices[pb]}
        r = walk_forward(sub, p, tr, te, st, fixed_pair=(pa, pb))
        for t in r.trades:
            rows.append((t.exit_i, f"{pa}/{pb}", t.ret_on_notional, t.bars, t.reason))
    rows.sort(key=lambda x: x[0])
    rets = np.array([x[2] for x in rows])
    print(f"=== Портфель ВСЕХ {len(FIXED_PAIRS)} фикс-пар, OOS ===")
    print(f"  всего сделок: {len(rets)}")
    print(f"  mean per-trade ret-on-notional: {rets.mean()*100:+.3f}%")
    print(f"  t-стат (>0): {rets.mean()/(rets.std(ddof=1)/np.sqrt(len(rets))):+.2f}")
    print(f"  win-rate: {100*np.mean(rets>0):.0f}%")
    print(f"  лучшая/худшая сделка: {rets.max()*100:+.2f}% / {rets.min()*100:+.2f}%")
    print(f"  суммарный ret-on-notional (если 1 пара в моменте): {rets.sum()*100:+.2f}%")

    # jackknife: удаляем топ-k лучших сделок — держится ли плюс?
    print("\n  Jackknife (убрать k лучших сделок):")
    srt = np.sort(rets)[::-1]
    for k in [0, 1, 2, 3, 5]:
        kept = srt[k:]
        m = kept.mean() if len(kept) else float("nan")
        t = (kept.mean()/(kept.std(ddof=1)/np.sqrt(len(kept)))) if len(kept) > 1 and kept.std() > 0 else float("nan")
        print(f"    -{k}: mean={m*100:+.3f}%  t={t:+.2f}  sum={kept.sum()*100:+.2f}%")

    # Реальная портфельная equity: equal-risk на каждую открытую сделку,
    # дневная переоценка через walk_forward на всех парах одновременно нетривиальна;
    # приближаем: средний per-trade ret * число одновременных слотов = 1 (последовательно).
    # Аннуализированный Sharpe по сделкам (грубо):
    n_years = 1564 / 252
    trades_per_year = len(rets) / n_years
    sh_trade = rets.mean() / rets.std(ddof=1) * np.sqrt(trades_per_year) if rets.std() > 0 else float("nan")
    print(f"\n  Аннуализ. Sharpe по сделкам (~{trades_per_year:.0f} сделок/год): {sh_trade:+.2f}")

    print("\n  Разбивка по парам (вклад в портфель):")
    bypair = {}
    for _, pair, ret, _, _ in rows:
        bypair.setdefault(pair, []).append(ret)
    for pair, rs in sorted(bypair.items(), key=lambda kv: -np.sum(kv[1])):
        rs = np.array(rs)
        print(f"    {pair:<12} n={len(rs):>2} sum={rs.sum()*100:+7.2f}% mean={rs.mean()*100:+6.3f}%")

    # Распределение причин выхода
    reasons = {}
    for _, _, _, _, reason in rows:
        reasons[reason] = reasons.get(reason, 0) + 1
    print(f"\n  Причины выхода: {reasons}")


if __name__ == "__main__":
    main()
