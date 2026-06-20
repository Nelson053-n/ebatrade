"""Широкий скрин семейств стратегий на полной истории (in-sample sanity, не WF).

Цель: понять какие режимы вообще дают положительный net с реальными издержками на
ОБОИХ инструментах. Это НЕ доказательство (in-sample); честная проверка — walk_forward.py.
"""
from __future__ import annotations


from .bt import Costs, buy_hold, metrics, run_position_series
from .load_data import build_continuous
from . import strategies as S

COSTS = Costs(fee_per_lot=1.0, slip_ticks=1.0)
ASSETS = ["SBRF", "GAZR"]


def _run(df, pos, reason):
    close = df["close"].to_numpy(float)
    seam = df["code"].to_numpy()
    trades = run_position_series(close, pos, COSTS, seam=seam, reasons=reason)
    return metrics(trades, close)


def main():
    data = {a: build_continuous(a) for a in ASSETS}
    for a in ASSETS:
        df = data[a]
        bh = buy_hold(df["close"].to_numpy(float), df["code"].to_numpy(), COSTS)
        print(f"\n===== {a}  bars={len(df)}  BuyHold(roll,1lot)={bh:+.0f}₽ =====")

    print("\n--- MOMENTUM (current st5 lb48/h18/stop2% ~ atr off) ---")
    for lb, h in [(48, 18), (24, 12), (12, 6), (72, 24)]:
        line = f"  lb{lb}/h{h}: "
        for a in ASSETS:
            pos, r = S.momentum_atr(data[a], lookback=lb, holding=h, atr_stop=0.0, atr_tp=0.0)
            m = _run(data[a], pos, r)
            line += f"{a} net{m['net']:+.0f} sh{m['sharpe']} tr{m['trades']} wr{m['winrate']}  "
        print(line)

    print("\n--- MOMENTUM + ATR stop/TP ---")
    for lb, h, st, tp in [(48, 36, 2.0, 4.0), (24, 24, 1.5, 3.0), (48, 48, 3.0, 6.0),
                          (12, 18, 2.0, 3.0), (36, 30, 2.5, 5.0)]:
        line = f"  lb{lb}/h{h}/st{st}/tp{tp}: "
        for a in ASSETS:
            pos, r = S.momentum_atr(data[a], lookback=lb, holding=h, atr_stop=st, atr_tp=tp)
            m = _run(data[a], pos, r)
            line += f"{a} net{m['net']:+.0f} sh{m['sharpe']} tr{m['trades']} wr{m['winrate']}  "
        print(line)

    print("\n--- MEAN-REVERSION z-score ---")
    for ma, ez, xz, sz, mh in [(48, 2.0, 0.5, 3.5, 36), (24, 1.5, 0.3, 3.0, 24),
                               (96, 2.0, 0.0, 3.0, 48), (48, 2.5, 0.5, 4.0, 30),
                               (36, 1.5, 0.5, 3.0, 18)]:
        line = f"  ma{ma}/ez{ez}/xz{xz}/sz{sz}/mh{mh}: "
        for a in ASSETS:
            pos, r = S.meanrev_z(data[a], ma_n=ma, entry_z=ez, exit_z=xz, stop_z=sz, max_hold=mh)
            m = _run(data[a], pos, r)
            line += f"{a} net{m['net']:+.0f} sh{m['sharpe']} tr{m['trades']} wr{m['winrate']}  "
        print(line)

    print("\n--- DONCHIAN breakout ---")
    for ch, xc, st, mh in [(48, 12, 2.0, 0), (24, 8, 2.0, 0), (96, 24, 3.0, 0),
                           (36, 12, 2.5, 48), (60, 20, 2.0, 0)]:
        line = f"  ch{ch}/xc{xc}/st{st}/mh{mh}: "
        for a in ASSETS:
            pos, r = S.donchian_breakout(data[a], channel=ch, exit_channel=xc, atr_stop=st, max_hold=mh)
            m = _run(data[a], pos, r)
            line += f"{a} net{m['net']:+.0f} sh{m['sharpe']} tr{m['trades']} wr{m['winrate']}  "
        print(line)


if __name__ == "__main__":
    main()
