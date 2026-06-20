"""Стресс-тест победителя (cross-sectional momentum L/S) на устойчивость эджа.

Варьируем: издержки (тейкер / выше), юниверс (gate top40 / top80 / mexc), число ног k,
train/test окна, half-период (first/second half). Эдж считается устойчивым, если Sharpe
держится >1 и return положителен при разумных вариациях.

  python -m pairsignal.research.crypto.stress_csmom
"""
from __future__ import annotations

from datetime import datetime, timezone

from .load_data import get_prices
from . import wf_engine as wf


def _d(ts: int) -> str:
    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def run_csmom_wf(prices, train_days, test_days, k, fee, slippage, tf="1h",
                 long_short=True, lookbacks=None, holdings=None, mas=None,
                 slice_=None):
    prices = prices.copy()
    if slice_:
        a, b = slice_
        prices = prices.iloc[a:b]
    prices.attrs["timeframe"] = tf
    bph = wf.bars_per_year(tf) / 365
    tr, te = int(train_days * bph), int(test_days * bph)
    lookbacks = lookbacks or [12, 24, 48, 96, 168, 240]
    holdings = holdings or [6, 12, 24, 48]
    if long_short:
        grid = [(lb, hd, k, True, 0) for lb in lookbacks for hd in holdings]
        warm = max(lookbacks)
    else:
        mas = mas or [0, 48, 96, 168]
        grid = [(lb, hd, k, False, ma) for lb in lookbacks for hd in holdings for ma in mas]
        warm = max(max(lookbacks), max(mas))
    rows = wf.walk_forward(prices, tr, te, grid, wf.backtest_csmom, warm, fee, slippage)
    return wf.stitch(rows, tf), rows


def line(label, agg):
    if not agg:
        print(f"{label:<42} — недостаточно окон")
        return
    print(f"{label:<42}{agg['return_pct']:>10.2f}{agg['sharpe']:>8.2f}"
          f"{agg['max_dd_pct']:>9.2f}{agg['win_windows']:>5}/{agg['n_windows']:<3}")


def main() -> None:
    g40 = get_prices("gate", "1h", 400, 40)
    g80 = get_prices("gate", "1h", 400, 80)
    m40 = get_prices("mexc", "1h", 400, 40)

    print(f"gate40: {g40.shape[1]}coins×{len(g40)}bars | gate80: {g80.shape[1]}×{len(g80)} "
          f"| mexc40: {m40.shape[1]}×{len(m40)}\n")
    print(f"{'config':<42}{'ret%':>10}{'Sharpe':>8}{'maxDD%':>9}{'win':>6}")
    print("-" * 76)

    # --- базовый ---
    agg, _ = run_csmom_wf(g40, 90, 30, 3, 0.0006, 0.0002)
    line("BASE gate40 k3 taker(0.06%) 90/30", agg)

    # --- издержки ---
    line("  cost: maker 0.02%/0 slip", run_csmom_wf(g40, 90, 30, 3, 0.0002, 0.0)[0])
    line("  cost: taker 0.06%+0.05% slip", run_csmom_wf(g40, 90, 30, 3, 0.0006, 0.0005)[0])
    line("  cost: HIGH 0.10%+0.10% slip", run_csmom_wf(g40, 90, 30, 3, 0.0010, 0.0010)[0])

    # --- юниверс ---
    line("  univ: gate80 k4", run_csmom_wf(g80, 90, 30, 4, 0.0006, 0.0002)[0])
    line("  univ: gate80 k6", run_csmom_wf(g80, 90, 30, 6, 0.0006, 0.0002)[0])
    line("  univ: mexc40 k3", run_csmom_wf(m40, 90, 30, 3, 0.0006, 0.0002)[0])

    # --- окна WF ---
    line("  window: train60/test30", run_csmom_wf(g40, 60, 30, 3, 0.0006, 0.0002)[0])
    line("  window: train120/test30", run_csmom_wf(g40, 120, 30, 3, 0.0006, 0.0002)[0])
    line("  window: train90/test14", run_csmom_wf(g40, 90, 14, 3, 0.0006, 0.0002)[0])

    # --- k ног ---
    line("  k=2", run_csmom_wf(g40, 90, 30, 2, 0.0006, 0.0002)[0])
    line("  k=5", run_csmom_wf(g40, 90, 30, 5, 0.0006, 0.0002)[0])

    # --- подпериоды (first/second half) ---
    h = len(g40) // 2
    line("  period: first half", run_csmom_wf(g40, 90, 30, 3, 0.0006, 0.0002,
                                              slice_=(0, h))[0])
    line("  period: second half", run_csmom_wf(g40, 90, 30, 3, 0.0006, 0.0002,
                                               slice_=(h, len(g40)))[0])

    # --- long-only + market filter (для сравнения нейтральности) ---
    line("  variant: gate40 long-only+MA", run_csmom_wf(g40, 90, 30, 3, 0.0006, 0.0002,
                                                        long_short=False)[0])
    print("-" * 76)


if __name__ == "__main__":
    main()
