"""Honest walk-forward для single-name intraday MR на акциях MOEX (часовой ТФ).

ГЛАВНОЕ против оверфита:
  ОДИН общий набор параметров (ma_n, entry_z, exit_z, stop_z, max_hold) подбирается
  на TRAIN (часть тикеров × раннее окно) по агрегированному Sharpe ПО СДЕЛКАМ всех
  train-тикеров, затем без изменений применяется к TEST-тикерам и/или TEST-окну,
  которых оптимизатор не видел. Эдж обязан переноситься (как SBER→GAZP в st5).

Режимы валидации:
  A) cross-ticker: train на половине тикеров (раннее+позднее окно), test на ДРУГОЙ
     половине — параметры не настраивались под test-тикеры.
  B) time-split: первая половина периода train (все тикеры), вторая половина test.
  Затем jackknife (выкинуть топ-тикеры/окна), sweep издержек.

no look-ahead: z по закрытым барам; издержки реальные (round-trip 2·cost_oneway).

  python -m pairsignal.research.moex.intraday_mr_wf
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pairsignal.research.moex.intraday_mr_core import (  # noqa: E402
    MRParams, MRResult, MRTrade, buyhold_ret, simulate,
)

CACHE = Path(__file__).resolve().parent / "cache"

# ликвидные имена (избегаем тонких: FEES/HYDR/UPRO/IRAO/CBOM/RNFT/BANE/FIXP)
LIQUID = ["SBER", "GAZP", "LKOH", "GMKN", "ROSN", "NLMK", "CHMF", "MAGN",
          "NVTK", "TATN", "PLZL", "ALRS", "MGNT", "RUAL", "SIBN", "MTSS",
          "SBERP", "RTKM", "TATNP", "SBER"]
LIQUID = list(dict.fromkeys(LIQUID))  # уникальные, сохраняя порядок

# часов в торговом году MOEX: ~250 дней × ~14 баров (10:00–23:50) ≈ 3500
BARS_PER_YEAR = 3500.0


def load_closes(ticker: str) -> np.ndarray:
    fp = CACHE / f"{ticker}_1h.csv"
    s = pd.read_csv(fp)
    return s["close"].to_numpy(dtype=float)


def agg_sharpe(rets: np.ndarray) -> float:
    """Sharpe по пулу доходностей сделок (не аннуализированный)."""
    if len(rets) < 5 or rets.std() == 0:
        return float("nan")
    return float(rets.mean() / rets.std() * np.sqrt(len(rets)))


def annual_sharpe(rets: np.ndarray, total_bars: int) -> float:
    """Грубая аннуализация: средняя доходность на бар × bars_per_year / std·sqrt(...).
    Считаем по сделкам, перевод в годовой через число сделок/баров."""
    if len(rets) < 5 or rets.std() == 0 or total_bars <= 0:
        return float("nan")
    trades_per_year = len(rets) / total_bars * BARS_PER_YEAR
    return float(rets.mean() / rets.std() * np.sqrt(trades_per_year))


# сетка перебора (намеренно небольшая — меньше степеней свободы → меньше оверфита)
GRID_MA = [24, 36, 48, 72]
GRID_ENTRY = [1.5, 2.0, 2.5]
GRID_EXIT = [0.0, 0.5]
GRID_STOP = [3.0, 4.0]
GRID_HOLD = [12, 24, 48]


def optimize(closes_by_t: dict[str, np.ndarray], cost: float,
             window: tuple[float, float] | None = None) -> tuple[MRParams, float]:
    """Подбор ОДНОГО набора параметров по агрегированному per-trade Sharpe на пуле
    train-тикеров. window=(lo_frac,hi_frac) — доля серии каждого тикера для train."""
    best_p = None
    best_s = -1e9
    combos = list(itertools.product(GRID_MA, GRID_ENTRY, GRID_EXIT, GRID_STOP, GRID_HOLD))
    for ma, ez, xz, sz, hold in combos:
        if xz >= ez:
            continue
        p = MRParams(ma_n=ma, entry_z=ez, exit_z=xz, stop_z=sz, max_hold=hold,
                     cost_oneway=cost)
        pool = []
        for c in closes_by_t.values():
            seg = c
            if window is not None:
                lo = int(len(c) * window[0])
                hi = int(len(c) * window[1])
                seg = c[lo:hi]
            r = simulate(seg, p)
            pool.extend(t.ret for t in r.trades)
        if len(pool) < 30:
            continue
        s = agg_sharpe(np.array(pool))
        if not np.isnan(s) and s > best_s:
            best_s = s
            best_p = p
    return best_p, best_s


def evaluate(closes_by_t: dict[str, np.ndarray], p: MRParams,
             window: tuple[float, float] | None = None) -> dict:
    """Прогон фиксированных параметров: пул сделок + per-ticker метрики."""
    pool = []
    total_bars = 0
    per_ticker = {}
    bh_pool = []
    for t, c in closes_by_t.items():
        seg = c
        if window is not None:
            lo = int(len(c) * window[0])
            hi = int(len(c) * window[1])
            seg = c[lo:hi]
        r = simulate(seg, p)
        rets = r.rets
        pool.extend(rets.tolist())
        total_bars += len(seg)
        per_ticker[t] = {
            "n": len(rets),
            "ret": float(rets.sum()) if len(rets) else 0.0,
            "sharpe": agg_sharpe(rets),
            "win": float((rets > 0).mean()) if len(rets) else float("nan"),
        }
        bh = buyhold_ret(seg)
        bh_pool.append(bh)
    pool = np.array(pool)
    return {
        "n_trades": len(pool),
        "total_ret": float(pool.sum()),
        "sharpe_pertrade": agg_sharpe(pool),
        "sharpe_annual": annual_sharpe(pool, total_bars),
        "win": float((pool > 0).mean()) if len(pool) else float("nan"),
        "avg_ret_bps": float(pool.mean() * 1e4) if len(pool) else float("nan"),
        "per_ticker": per_ticker,
        "pool": pool,
        "buyhold_avg": float(np.nanmean(bh_pool)),
        "total_bars": total_bars,
    }


def fmt(x, d=2):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def main():
    tickers = [t for t in LIQUID if (CACHE / f"{t}_1h.csv").exists()]
    closes = {t: load_closes(t) for t in tickers}
    print(f"Тикеров: {len(tickers)}  ({', '.join(tickers)})")
    for t in tickers[:3]:
        print(f"  {t}: {len(closes[t])} баров")

    COST = 0.0005  # 0.05% одна сторона (≈комиссия 0.04% + slippage)

    # ========================================================================
    # РЕЖИМ A: cross-ticker. Train на одной группе, test на ДРУГОЙ.
    # Делим по алфавиту индекса, чтобы группы были «случайны» по сектору.
    # ========================================================================
    print("\n" + "=" * 72)
    print("РЕЖИМ A: CROSS-TICKER (параметры с train-тикеров → test-тикеры)")
    print("=" * 72)
    half = len(tickers) // 2
    # чередующееся разбиение (не по секторам подряд)
    train_t = tickers[0::2]
    test_t = tickers[1::2]
    print(f"TRAIN тикеры: {train_t}")
    print(f"TEST  тикеры: {test_t}")

    train_closes = {t: closes[t] for t in train_t}
    test_closes = {t: closes[t] for t in test_t}

    best_p, train_s = optimize(train_closes, COST)
    print(f"\nЛучшие параметры на TRAIN-тикерах: {best_p}")
    print(f"  train per-trade Sharpe (in-sample): {fmt(train_s)}")

    ev_train = evaluate(train_closes, best_p)
    ev_test = evaluate(test_closes, best_p)
    print(f"\nTRAIN-тикеры (in-sample): n={ev_train['n_trades']} "
          f"Sharpe_pt={fmt(ev_train['sharpe_pertrade'])} "
          f"Sharpe_ann={fmt(ev_train['sharpe_annual'])} "
          f"win={fmt(ev_train['win'],3)} ret={fmt(ev_train['total_ret']*100,1)}%")
    print(f"TEST-тикеры (OUT-OF-SAMPLE): n={ev_test['n_trades']} "
          f"Sharpe_pt={fmt(ev_test['sharpe_pertrade'])} "
          f"Sharpe_ann={fmt(ev_test['sharpe_annual'])} "
          f"win={fmt(ev_test['win'],3)} ret={fmt(ev_test['total_ret']*100,1)}% "
          f"avg={fmt(ev_test['avg_ret_bps'],1)}bps")
    print(f"  BuyHold avg по test-тикерам: {fmt(ev_test['buyhold_avg']*100,1)}%")

    print("\n  Per-ticker OOS (test-группа):")
    for t, m in sorted(ev_test["per_ticker"].items(), key=lambda kv: -(kv[1]["sharpe"] if not np.isnan(kv[1]["sharpe"]) else -9)):
        print(f"    {t:6s} n={m['n']:4d} Sharpe={fmt(m['sharpe']):>6s} "
              f"win={fmt(m['win'],2):>5s} ret={fmt(m['ret']*100,1):>7s}%")

    # ========================================================================
    # РЕЖИМ B: TIME-SPLIT. Параметры с первой половины ВСЕХ тикеров → вторая половина.
    # ========================================================================
    print("\n" + "=" * 72)
    print("РЕЖИМ B: TIME-SPLIT (1-я половина периода → 2-я половина, все тикеры)")
    print("=" * 72)
    bp_time, ts_time = optimize(closes, COST, window=(0.0, 0.5))
    print(f"Лучшие параметры на 1-й половине: {bp_time}  (in-sample Sharpe {fmt(ts_time)})")
    ev_h1 = evaluate(closes, bp_time, window=(0.0, 0.5))
    ev_h2 = evaluate(closes, bp_time, window=(0.5, 1.0))
    print(f"1-я половина (IS):  n={ev_h1['n_trades']} Sharpe_pt={fmt(ev_h1['sharpe_pertrade'])} "
          f"Sharpe_ann={fmt(ev_h1['sharpe_annual'])} ret={fmt(ev_h1['total_ret']*100,1)}%")
    print(f"2-я половина (OOS): n={ev_h2['n_trades']} Sharpe_pt={fmt(ev_h2['sharpe_pertrade'])} "
          f"Sharpe_ann={fmt(ev_h2['sharpe_annual'])} ret={fmt(ev_h2['total_ret']*100,1)}% "
          f"avg={fmt(ev_h2['avg_ret_bps'],1)}bps")

    # ========================================================================
    # JACKKNIFE по тикерам на полном OOS (режим A test): убрать топ-1/2/3 тикера.
    # ========================================================================
    print("\n" + "=" * 72)
    print("JACKKNIFE OOS (test-группа реж.A): выкидываем топ-вкладчиков")
    print("=" * 72)
    contrib = sorted(ev_test["per_ticker"].items(),
                     key=lambda kv: -kv[1]["ret"])
    print("  Топ по вкладу в total_ret:", [f"{t}({fmt(m['ret']*100,1)}%)" for t, m in contrib[:4]])
    for k in (1, 2, 3):
        drop = {t for t, _ in contrib[:k]}
        sub = {t: c for t, c in test_closes.items() if t not in drop}
        ev = evaluate(sub, best_p)
        print(f"  без топ-{k} ({', '.join(sorted(drop))}): "
              f"n={ev['n_trades']} Sharpe_pt={fmt(ev['sharpe_pertrade'])} "
              f"Sharpe_ann={fmt(ev['sharpe_annual'])} ret={fmt(ev['total_ret']*100,1)}%")

    # ========================================================================
    # SWEEP ИЗДЕРЖЕК на OOS (test-группа) с теми же train-параметрами.
    # ========================================================================
    print("\n" + "=" * 72)
    print("SWEEP ИЗДЕРЖЕК (OOS test-группа, параметры с train)")
    print("=" * 72)
    for cost in (0.0, 0.0004, 0.0005, 0.0006, 0.0008):
        pc = MRParams(**{**best_p.__dict__, "cost_oneway": cost})
        ev = evaluate(test_closes, pc)
        print(f"  cost_oneway={cost*100:.2f}%  (rt {cost*200:.2f}%): "
              f"Sharpe_pt={fmt(ev['sharpe_pertrade'])} "
              f"Sharpe_ann={fmt(ev['sharpe_annual'])} "
              f"ret={fmt(ev['total_ret']*100,1)}% avg={fmt(ev['avg_ret_bps'],1)}bps")

    # сводный вердикт-числа
    print("\n" + "=" * 72)
    print("СВОДКА (для вердикта)")
    print("=" * 72)
    print(f"OOS cross-ticker Sharpe_annual = {fmt(ev_test['sharpe_annual'])}")
    print(f"OOS time-split  Sharpe_annual = {fmt(ev_h2['sharpe_annual'])}")
    print(f"Параметры       = {best_p}")


if __name__ == "__main__":
    main()
