"""Тест ФИКСИРОВАННОГО набора параметров mean-reversion на всей истории и по фолдам.

Боевой st5 использует один статический конфиг (не переоптимизирует на лету). Поэтому
итоговый эдж должен держаться на ЕДИНОМ наборе параметров, выбранном консервативно по
результатам joint-WF. Проверяем именно это: один набор → оба инструмента, помесячные срезы.
"""
from __future__ import annotations


from .bt import Costs, buy_hold, metrics, run_position_series
from .load_data import build_continuous
from . import strategies as S

# Кандидаты — робастные наборы из joint-WF (исключаем ez1.0 как нестабильный по переносу).
CANDIDATES = {
    "C1_ma36_ez2.5": dict(ma_n=36, entry_z=2.5, exit_z=0.0, stop_z=3.0, max_hold=36),
    "C2_ma48_ez1.5": dict(ma_n=48, entry_z=1.5, exit_z=0.0, stop_z=3.0, max_hold=24),
    "C3_ma48_ez2.0": dict(ma_n=48, entry_z=2.0, exit_z=0.5, stop_z=3.0, max_hold=30),
    "C4_ma72_ez2.0": dict(ma_n=72, entry_z=2.0, exit_z=0.0, stop_z=3.0, max_hold=24),
    "C5_ma36_ez2.0": dict(ma_n=36, entry_z=2.0, exit_z=0.5, stop_z=3.0, max_hold=24),
}


def run(df, p, costs):
    pos, r = S.meanrev_z(df, ma_n=p["ma_n"], entry_z=p["entry_z"], exit_z=p["exit_z"],
                         stop_z=p["stop_z"], max_hold=p["max_hold"])
    close = df["close"].to_numpy(float); seam = df["code"].to_numpy()
    return metrics(run_position_series(close, pos, costs, seam=seam, reasons=r), close)


def main():
    data = {a: build_continuous(a) for a in ["SBRF", "GAZR"]}
    base = Costs(1.0, 1.0)

    print("=== FIXED-PARAM full-history (both instruments, costs fee=1 slip=1t) ===")
    for name, p in CANDIDATES.items():
        line = f"  {name:16s}: "
        ok = True
        for a in ["SBRF", "GAZR"]:
            m = run(data[a], p, base)
            line += f"{a} net{m['net']:+.0f} sh{m['sharpe']:+.2f} tr{m['trades']} wr{m['winrate']} dd{m['max_dd']:.0f}  "
            if m["net"] <= 0:
                ok = False
        line += "<-- both+" if ok else ""
        print(line)

    # выбираем C1 (наиболее консервативный, прошёл joint-WF) для детального стресса
    p = CANDIDATES["C1_ma36_ez2.5"]
    print(f"\n=== STRESS C1 {p} ===")
    print("-- per-quarter (по контракту-донору) --")
    for a in ["SBRF", "GAZR"]:
        df = data[a]
        for code in sorted(df["code"].unique()):
            sub = df[df["code"] == code]
            if len(sub) < 200:
                continue
            m = run(sub, p, base)
            print(f"  {a} {code} bars{len(sub)}: net{m['net']:+.0f} sh{m['sharpe']:+.2f} tr{m['trades']} wr{m['winrate']}")

    print("-- cost sensitivity (slip ticks per side) --")
    for slip in [0.5, 1.0, 1.5, 2.0, 3.0]:
        c = Costs(1.0, slip)
        line = f"  slip={slip}t: "
        for a in ["SBRF", "GAZR"]:
            m = run(data[a], p, c)
            line += f"{a} net{m['net']:+.0f} sh{m['sharpe']:+.2f}  "
        print(line)

    print("-- BuyHold baseline --")
    for a in ["SBRF", "GAZR"]:
        print(f"  {a}: {buy_hold(data[a]['close'].to_numpy(float), data[a]['code'].to_numpy(), base):+.0f}₽")


if __name__ == "__main__":
    main()
