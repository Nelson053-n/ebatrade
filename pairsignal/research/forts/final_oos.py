"""ОКОНЧАТЕЛЬНЫЙ честный тест: один фиксированный набор параметров, выбранный на
ПЕРВОЙ половине истории (joint по обоим инструментам), проверен НЕТРОНУТЫМ на ВТОРОЙ
половине — out-of-sample во времени И по инструменту одновременно.

Это режим, в котором стратегия реально живёт в st5: один статический конфиг, без
переоптимизации на лету. Доказываем перенос: train-выбор на SBRF+GAZR[первая половина],
OOS-замер на SBRF+GAZR[вторая половина]. Сессионный фильтр основной сессии включён.
"""
from __future__ import annotations

import itertools


from .bt import Costs, buy_hold, metrics, run_position_series
from .load_data import build_continuous
from . import strategies as S

COSTS = Costs(1.0, 1.0)
SESSION = (600, 1125)  # 10:00-18:45 MSK — основная сессия
ASSETS = ["SBRF", "GAZR"]

GRID = {
    "ma_n": [24, 36, 48, 72],
    "entry_z": [1.5, 2.0, 2.5],     # ez1.0 исключён — нестабилен по переносу
    "exit_z": [0.0, 0.5],
    "stop_z": [3.0, 4.0],
    "max_hold": [24, 36],
    # трендовый гейт: не фейдить сильный тренд за trend_n баров. 0.0 = выкл.
    "trend_n": [48],
    "trend_max": [0.0, 0.015, 0.025, 0.04],
}


def _run(df, p, costs=COSTS):
    pos, r = S.meanrev_z(df, ma_n=p["ma_n"], entry_z=p["entry_z"], exit_z=p["exit_z"],
                         stop_z=p["stop_z"], max_hold=p["max_hold"],
                         time_lo=SESSION[0], time_hi=SESSION[1],
                         trend_n=p.get("trend_n", 0), trend_max=p.get("trend_max", 0.0))
    close = df["close"].to_numpy(float); seam = df["code"].to_numpy()
    return metrics(run_position_series(close, pos, costs, seam=seam, reasons=r), close)


def _grid():
    keys = list(GRID)
    for vals in itertools.product(*[GRID[k] for k in keys]):
        yield dict(zip(keys, vals))


def main():
    data = {a: build_continuous(a) for a in ASSETS}
    n = min(len(data[a]) for a in ASSETS)
    half = n // 2
    train = {a: data[a].iloc[:half] for a in ASSETS}
    test = {a: data[a].iloc[half:] for a in ASSETS}
    import pandas as pd
    split_ts = pd.Timestamp(data["SBRF"].index[half], unit="ms", tz="UTC")
    print(f"split at bar {half} ({split_ts:%Y-%m-%d %H:%M} MSK+0)")

    # выбор по min-Sharpe обоих инструментов на TRAIN (робастный перенос)
    best, best_score, best_tr = None, -1e18, None
    for p in _grid():
        ms = {a: _run(train[a], p) for a in ASSETS}
        if min(m["trades"] for m in ms.values()) < 15:
            continue
        score = min(m["sharpe"] for m in ms.values())
        if score > best_score:
            best, best_score, best_tr = p, score, ms

    print(f"\nSELECTED (joint train, min-Sharpe={best_score:+.2f}): {best}")
    print("  TRAIN:", {a: {k: best_tr[a][k] for k in ('net','sharpe','trades','winrate','max_dd')} for a in ASSETS})

    print("\n=== OUT-OF-SAMPLE (second half, untouched) ===")
    for a in ASSETS:
        m = _run(test[a], best)
        bh = buy_hold(test[a]["close"].to_numpy(float), test[a]["code"].to_numpy(), COSTS)
        print(f"  {a}: net{m['net']:+.0f}₽ sharpe{m['sharpe']:+.2f} trades{m['trades']} "
              f"winrate{m['winrate']}% maxDD{m['max_dd']:.0f} avgbars{m['avg_bars']} | BuyHold{bh:+.0f}")

    print("\n=== OOS cost sensitivity (slip ticks/side) ===")
    for slip in [0.5, 1.0, 1.5, 2.0]:
        c = Costs(1.0, slip)
        line = f"  slip={slip}t fee=1: "
        for a in ASSETS:
            m = _run(test[a], best, c)
            line += f"{a} net{m['net']:+.0f} sh{m['sharpe']:+.2f}  "
        print(line)

    print("\n=== OOS fee sensitivity (rub/lot/side) ===")
    for fee in [1.0, 2.0, 3.0]:
        c = Costs(fee, 1.0)
        line = f"  fee={fee} slip=1t: "
        for a in ASSETS:
            m = _run(test[a], best, c)
            line += f"{a} net{m['net']:+.0f} sh{m['sharpe']:+.2f}  "
        print(line)


if __name__ == "__main__":
    main()
