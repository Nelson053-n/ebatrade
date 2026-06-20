"""Главный прогон: честный walk-forward st6 на реальных дневных данных MOEX.

Запуск:  .venv/bin/python -m pairsignal.research.moex.run_experiments
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pairsignal.st6.core import Params, rank_pairs  # noqa: E402
from pairsignal.st6.backtest import backtest_pair  # noqa: E402
from pairsignal.research.moex.wf_core import walk_forward, WFResult  # noqa: E402

CACHE = Path(__file__).resolve().parent / "cache"

# тикеры с полной 1614-историей (отбрасываем YDEX/FIVE/FIXP — частичное покрытие)
CORE_TICKERS = [
    "LKOH", "ROSN", "SIBN", "TATN", "GAZP", "NVTK", "TATNP", "BANE", "RNFT",
    "GMKN", "NLMK", "MAGN", "CHMF", "PLZL", "RUAL", "ALRS",
    "SBER", "SBERP", "VTBR", "CBOM",
    "HYDR", "IRAO", "FEES", "UPRO", "MTSS", "RTKM", "MGNT",
]

# Экономически-обоснованные кандидаты (один драйвер / близкие бумаги)
FIXED_PAIRS = [
    ("NLMK", "CHMF"),    # сталь
    ("NLMK", "MAGN"),    # сталь
    ("MAGN", "CHMF"),    # сталь
    ("LKOH", "ROSN"),    # нефть
    ("LKOH", "SIBN"),    # нефть
    ("ROSN", "SIBN"),    # нефть
    ("TATN", "TATNP"),   # обычка/префы одного эмитента (сильнейшая коинтеграция)
    ("SBER", "SBERP"),   # обычка/префы одного эмитента
    ("ROSN", "TATN"),    # нефть
]


def load_prices(tickers: list[str]) -> tuple[dict[str, list[float]], pd.Index]:
    df = pd.read_csv(CACHE / "union_1d.csv", index_col=0)
    sub = df[[t for t in tickers if t in df.columns]].dropna()
    return {t: sub[t].tolist() for t in sub.columns}, sub.index


def fmt(r: WFResult, label: str) -> str:
    wr = r.win_rate * 100 if r.n_trades else 0
    return (f"{label:<34} folds_traded={sum(1 for f in r.fold_log if f['pair']):>2}  "
            f"trades={r.n_trades:>3}  ret={r.return_pct:+7.2f}%  win={wr:4.0f}%  "
            f"DD={r.max_drawdown_pct:5.2f}%  Sh_ann={r.sharpe_annual():+5.2f}  "
            f"Sh_trade={r.sharpe_pertrade:+5.2f}")


def base_params() -> Params:
    """Текущие боевые пороги st6 (config.StrategyConfig дефолты)."""
    return Params(
        beta_window=240, z_window=240, corr_window=120,
        z_entry=2.0, z_exit=0.3, z_stop=3.5,
        corr_enter=0.58, corr_break=0.45,
        select_min_corr=0.60, select_max_pvalue=0.20, select_max_halflife=400.0,
        risk_fraction=0.02, fee_rate=0.0006, slippage_rate=0.0005,
    )


def main() -> None:
    prices, idx = load_prices(CORE_TICKERS)
    n = len(next(iter(prices.values())))
    print(f"=== Данные: {len(prices)} тикеров x {n} дневных баров "
          f"({idx[0]} .. {idx[-1]} ms) ===")
    print(f"    Период ~{n/252:.1f} торговых лет. warmup={max(240,240,120)+1} баров.\n")

    p = base_params()

    # ---------------------------------------------------------------
    # A. IN-SAMPLE baseline (как боевой st6): отбор + торговля на ОДНОМ окне
    # ---------------------------------------------------------------
    print("--- A. IN-SAMPLE (боевой st6: отбор пары и торговля на одном окне) ---")
    ranked = rank_pairs(prices, p)
    if ranked:
        best = ranked[0]
        print(f"  rank_pairs выбрал: {best.a}/{best.b}  corr={best.corr:.3f} "
              f"p={best.pvalue:.4f} hl={best.halflife:.0f} score={best.score:.4f}")
        r_is = backtest_pair(prices[best.a], prices[best.b], p, start_equity=1_000_000.0)
        wr = r_is.win_rate * 100 if r_is.n_trades else 0
        print(f"  IN-SAMPLE backtest {best.a}/{best.b}: trades={r_is.n_trades} "
              f"ret={r_is.return_pct:+.2f}% win={wr:.0f}% DD={r_is.max_drawdown_pct:.2f}% "
              f"net={r_is.total_net:+,.0f}")
        print(f"  Топ-5 пар по score:")
        for s in ranked[:5]:
            print(f"    {s.a}/{s.b}: corr={s.corr:.3f} p={s.pvalue:.4f} "
                  f"hl={s.halflife:.0f} score={s.score:.4f}")
    else:
        print("  rank_pairs: годной пары нет")
    print()

    # ---------------------------------------------------------------
    # B. ЧЕСТНЫЙ OOS walk-forward, динамический отбор на train
    # ---------------------------------------------------------------
    print("--- B. ЧЕСТНЫЙ OOS walk-forward (отбор пары на train, торговля на test) ---")
    for train_bars, test_bars, step in [(500, 250, 250), (600, 200, 200),
                                         (750, 250, 250), (500, 125, 125)]:
        r = walk_forward(prices, p, train_bars, test_bars, step)
        print("  " + fmt(r, f"WF train={train_bars} test={test_bars}"))
    print()

    # ---------------------------------------------------------------
    # C. ФИКСИРОВАННЫЕ экономически-обоснованные пары (без перебора 55)
    # ---------------------------------------------------------------
    print("--- C. ФИКСИРОВАННЫЕ пары (OOS, без data-snooping отбора) ---")
    train_bars, test_bars, step = 500, 250, 250
    for pa, pb in FIXED_PAIRS:
        if pa not in prices or pb not in prices:
            continue
        sub = {pa: prices[pa], pb: prices[pb]}
        r = walk_forward(sub, p, train_bars, test_bars, step, fixed_pair=(pa, pb))
        print("  " + fmt(r, f"{pa}/{pb}"))
    print()

    # ---------------------------------------------------------------
    # D. Бонферрони-поправка при отборе (55 тестов)
    # ---------------------------------------------------------------
    print("--- D. OOS отбор с Бонферрони p<0.20/55 ---")
    n_pairs = len(CORE_TICKERS) * (len(CORE_TICKERS) - 1) // 2
    for train_bars, test_bars, step in [(500, 250, 250), (600, 200, 200)]:
        r = walk_forward(prices, p, train_bars, test_bars, step, bonferroni_n=n_pairs)
        print("  " + fmt(r, f"WF bonf train={train_bars} test={test_bars}"))
    print()

    # ---------------------------------------------------------------
    # F. BuyHold корзины
    # ---------------------------------------------------------------
    print("--- F. BuyHold equal-weight корзины (тот же период) ---")
    rets = [np.asarray(v)[-1] / np.asarray(v)[0] - 1 for v in prices.values()]
    mat = np.array([np.asarray(v, dtype=float) for v in prices.values()])
    dret = np.diff(mat, axis=1) / mat[:, :-1]
    port = dret.mean(axis=0)
    sh = port.mean() / port.std() * np.sqrt(252)
    print(f"  EW BuyHold: ret={np.mean(rets)*100:+.2f}% Sharpe_ann={sh:+.2f} "
          f"(на капитал, не на нотионал ноги)")


if __name__ == "__main__":
    main()
