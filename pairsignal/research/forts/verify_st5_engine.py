"""Проверка переноса: боевой engine st5 (meanrev) на той же истории, что research.

Гоняет pairsignal.st5.engine.TradingEngine в режиме meanrev по непрерывной серии
SBRF/GAZR (per-contract сегменты — как seam-форс в research.bt) и печатает net/Sharpe.
Издержки выставлены под research (1 тик/сторона, 1₽/лот/сторона):
  paper_book_halfspread_ticks=0, tick_offset=1 → slip 1 тик; taker_fee_rub_per_lot=1.

Цель — подтвердить, что боевой код даёт OOS-числа, близкие к research.final_oos
(SBER Sharpe ~1.5). Не трогает session_state, не ходит в сеть (читает кэш research).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pairsignal.st5.config import St5Config
from pairsignal.st5.engine import TradingEngine
from pairsignal.st5.models import InstrumentSpec, PriceBar

from .load_data import build_continuous

ASSETS = ["SBRF", "GAZR"]
SESSION = (600, 1125)
BARS_PER_DAY = 84.0


def _cfg() -> St5Config:
    c = St5Config()
    c.strategy.strategy_mode = "meanrev"
    c.strategy.mr_ma_n = 36
    c.strategy.mr_entry_z = 2.5
    c.strategy.mr_exit_z = 0.5
    c.strategy.mr_stop_z = 4.0
    c.strategy.mr_max_hold = 36
    c.strategy.session_lo_min = SESSION[0]
    c.strategy.session_hi_min = SESSION[1]
    c.strategy.candle_interval_minutes = 10
    # издержки под research: 1 тик/сторона, 1₽/лот/сторона; без EOD-flat (meanrev)
    c.execution.paper_book_halfspread_ticks = 0.0
    c.execution.tick_offset = 1
    c.paper.taker_fee_rub_per_lot = 1.0
    c.execution.quantity_lots = 1
    return c


def _spec() -> InstrumentSpec:
    return InstrumentSpec(code="ST", tick_size=1.0, tick_value_rub=1.0, lot=1, expiry=None)


def _run_engine(df: pd.DataFrame) -> tuple[list, int]:
    """Прогон по сегментам контракта (seam = смена code). Возвращает (net-серия по сделкам, n_trades)."""
    nets: list[tuple[int, float]] = []   # (exit_global_idx, net)
    cfg = _cfg()
    spec = _spec()
    codes = df["code"].to_numpy()
    base = 0
    for code in pd.unique(codes):
        sub = df[df["code"] == code]
        eng = TradingEngine(cfg, spec)
        for ts, row in sub.iterrows():
            bar = PriceBar(int(ts), float(row["open"]), float(row["high"]),
                           float(row["low"]), float(row["close"]),
                           float(row.get("volume", 0.0)))
            eng.step(bar)
        # принудительный выход на стыке (как research seam)
        eng.flat_all("seam")
        # глобальные индексы выхода для дневной агрегации Sharpe
        sub_pos = {int(t): i for i, t in enumerate(sub.index.to_numpy())}
        for t in eng.trades:
            gi = base + sub_pos.get(int(t.exit_ts), len(sub) - 1)
            nets.append((gi, t.net_pnl_rub))
        base += len(sub)
    return nets, len(nets)


def _sharpe(nets: list[tuple[int, float]], n_bars: int) -> tuple[float, float, float]:
    if not nets:
        return 0.0, 0.0, 0.0
    pnl = np.zeros(n_bars)
    for gi, net in nets:
        pnl[min(gi, n_bars - 1)] += net
    day = (np.arange(n_bars) // BARS_PER_DAY).astype(int)
    daily = pd.Series(pnl).groupby(day).sum()
    mu, sd = daily.mean(), daily.std(ddof=1)
    sharpe = (mu / sd * np.sqrt(252)) if sd > 0 else 0.0
    total = sum(net for _, net in nets)
    wins = sum(1 for _, net in nets if net > 0)
    wr = 100 * wins / len(nets)
    return float(sharpe), float(total), float(wr)


def main() -> None:
    data = {a: build_continuous(a) for a in ASSETS}
    n = min(len(data[a]) for a in ASSETS)
    half = n // 2
    print("=== БОЕВОЙ engine st5 (meanrev) — full history (издержки research) ===")
    for a in ASSETS:
        nets, ntr = _run_engine(data[a])
        sh, total, wr = _sharpe(nets, len(data[a]))
        print(f"  {a}: net{total:+.0f}₽ sharpe{sh:+.2f} trades{ntr} winrate{wr:.1f}%")

    print("\n=== OUT-OF-SAMPLE (вторая половина, как research.final_oos) ===")
    for a in ASSETS:
        test = data[a].iloc[half:]
        nets, ntr = _run_engine(test)
        sh, total, wr = _sharpe(nets, len(test))
        print(f"  {a}: net{total:+.0f}₽ sharpe{sh:+.2f} trades{ntr} winrate{wr:.1f}%")


if __name__ == "__main__":
    main()
