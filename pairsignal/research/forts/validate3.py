"""Валидация на 3 инструментах (SBRF/GAZR/LKOH) + улучшения для GAZR:
сессионный фильтр времени и трендовый фильтр (не фейдить сильный тренд).

LKOH — третий, не участвовавший в подборе инструмент: если эдж держится и на нём,
режим mean-reversion — общее свойство внутридневного FORTS, а не оверфит SBER.
"""
from __future__ import annotations


from .bt import Costs, metrics, run_position_series
from .load_data import build_continuous
from . import strategies as S

COSTS = Costs(1.0, 1.0)
ASSETS = ["SBRF", "GAZR", "LKOH"]


def run(df, pos, r):
    close = df["close"].to_numpy(float); seam = df["code"].to_numpy()
    return metrics(run_position_series(close, pos, COSTS, seam=seam, reasons=r), close)


def main():
    data = {a: build_continuous(a) for a in ASSETS}

    print("=== fixed MR candidates on ALL THREE instruments ===")
    cands = {
        "ma36_ez2.5_mh36": dict(ma_n=36, entry_z=2.5, exit_z=0.0, stop_z=3.0, max_hold=36),
        "ma48_ez2.0_mh30": dict(ma_n=48, entry_z=2.0, exit_z=0.5, stop_z=3.0, max_hold=30),
        "ma48_ez1.5_mh24": dict(ma_n=48, entry_z=1.5, exit_z=0.0, stop_z=3.0, max_hold=24),
    }
    for name, p in cands.items():
        line = f"  {name}: "
        for a in ASSETS:
            pos, r = S.meanrev_z(data[a], **p)
            m = run(data[a], pos, r)
            line += f"{a} net{m['net']:+.0f} sh{m['sharpe']:+.2f}  "
        print(line)

    print("\n=== session-time filter (only mid-session, MSK minutes) ===")
    # FORTS: основная 10:00-18:45 (600-1125), вечерняя 19:05-23:50. Пробуем разные окна.
    p = dict(ma_n=48, entry_z=2.0, exit_z=0.5, stop_z=3.0, max_hold=30)
    for lo, hi in [(0, 1440), (600, 1125), (660, 1080), (600, 1410), (720, 1410)]:
        line = f"  win[{lo}:{hi}]: "
        for a in ASSETS:
            pos, r = S.meanrev_z(data[a], time_lo=lo, time_hi=hi, **p)
            m = run(data[a], pos, r)
            line += f"{a} net{m['net']:+.0f} sh{m['sharpe']:+.2f} tr{m['trades']}  "
        print(line)


if __name__ == "__main__":
    main()
