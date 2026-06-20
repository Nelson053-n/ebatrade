"""trend_scan — широкий IN-SAMPLE обзор семейств trend-following по корзине MOEX.

Это НЕ честный OOS — сначала смотрим, есть ли вообще сигнал на всей истории при
реальных издержках. Если на всей выборке всё <0, дальше WF не имеет смысла.
ОДИН набор параметров на все тикеры; equity усредняется по корзине.
"""
from __future__ import annotations

import sys

from trend_core import (basket_metrics, buyhold_ew, load_closes, sig_donchian,
                        sig_ma_cross, sig_tsmom)


def run(interval: str, fee: float, slip: float):
    closes = load_closes(interval)
    print(f"=== {interval}: {closes.shape[1]} тикеров, {closes.shape[0]} баров, "
          f"fee={fee} slip={slip} (costs/side={fee+slip}) ===")
    bh = buyhold_ew(closes, interval)
    print(f"BuyHold EW: ret {bh['ret_pct']:+.1f}%  Sharpe {bh['sharpe']:.2f}  DD {bh['max_dd_pct']:.1f}%\n")

    rows = []
    # MA-cross close vs SMA(n), long-only и long/short
    for n in ([20, 50, 100, 150, 200] if interval == "1d" else [48, 100, 200, 400, 800]):
        for ls in (False, True):
            m = basket_metrics(closes, lambda c, n=n, ls=ls: sig_ma_cross(c, n, 0, ls),
                               fee, slip, interval)
            rows.append((f"MAcross n={n} {'L/S' if ls else 'LO'}", m))
    # MA fast/slow
    for fast, slow in ([(10, 50), (20, 100), (50, 200)] if interval == "1d"
                       else [(24, 100), (48, 200), (100, 400)]):
        for ls in (False, True):
            m = basket_metrics(closes, lambda c, f=fast, s=slow, ls=ls: sig_ma_cross(c, s, f, ls),
                               fee, slip, interval)
            rows.append((f"MA {fast}/{slow} {'L/S' if ls else 'LO'}", m))
    # Donchian
    for en in ([20, 50, 100] if interval == "1d" else [100, 200, 400]):
        ex = max(5, en // 2)
        for ls in (False, True):
            m = basket_metrics(closes, lambda c, en=en, ex=ex, ls=ls: sig_donchian(c, en, ex, ls),
                               fee, slip, interval)
            rows.append((f"Donchian {en}/{ex} {'L/S' if ls else 'LO'}", m))
    # TSMOM
    for lb in ([20, 50, 100, 200] if interval == "1d" else [100, 200, 400, 800]):
        for ls in (False, True):
            m = basket_metrics(closes, lambda c, lb=lb, ls=ls: sig_tsmom(c, lb, ls),
                               fee, slip, interval)
            rows.append((f"TSMOM lb={lb} {'L/S' if ls else 'LO'}", m))
    # TSMOM + vol-target (long-only)
    vt = 0.02 if interval == "1d" else 0.005
    for lb in ([50, 100, 200] if interval == "1d" else [200, 400, 800]):
        m = basket_metrics(closes, lambda c, lb=lb: sig_tsmom(c, lb, False),
                           fee, slip, interval, vol_target=vt)
        rows.append((f"TSMOM lb={lb} LO +volT", m))

    rows.sort(key=lambda x: x[1]["sharpe"], reverse=True)
    print(f"{'strategy':28} {'ret%':>9} {'Sharpe':>7} {'maxDD%':>8}")
    print("-" * 56)
    for name, m in rows:
        print(f"{name:28} {m['ret_pct']:>+9.1f} {m['sharpe']:>7.2f} {m['max_dd_pct']:>8.1f}")


if __name__ == "__main__":
    interval = sys.argv[1] if len(sys.argv) > 1 else "1d"
    fee = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0005
    slip = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0003
    run(interval, fee, slip)
