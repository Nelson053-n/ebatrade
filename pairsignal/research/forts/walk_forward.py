"""Честный walk-forward для mean-reversion на FORTS-фьючерсах.

Принцип переноса эджа (главное требование): параметры подбираются на TRAIN-окне ОДНОГО
инструмента (SBRF), затем применяются НЕВИДАННО к TEST-окну ОБОИХ инструментов. Эдж
обязан переноситься и во времени (train→test), и по инструменту (SBRF→GAZR).

Скольжение: окна train/test движутся по истории. Метрика OOS — агрегат net и Sharpe
по всем test-кускам, отдельно по каждому инструменту.

Издержки реальные (Costs). Все индикаторы — по закрытым барам (strategies.py гарантирует
no look-ahead). Сделки не переносятся через стык контракта (seam в bt).
"""
from __future__ import annotations

import itertools

import numpy as np

from .bt import Costs, metrics, run_position_series
from .load_data import build_continuous
from . import strategies as S

COSTS = Costs(fee_per_lot=1.0, slip_ticks=1.0)
BARS_PER_DAY = 84

# сетка параметров mean-reversion (z-score reversion к SMA)
GRID = {
    "ma_n": [24, 36, 48, 72],
    "entry_z": [1.0, 1.5, 2.0, 2.5],
    "exit_z": [0.0, 0.5],
    "stop_z": [3.0, 4.0],
    "max_hold": [18, 24, 36],
}


# окно основной сессии MSK (10:00-18:45). Фиксируем — не оптимизируем (исключает
# вечернюю/ночную сессию, где у GAZR эдж деградирует; найдено на validate3).
SESSION = (600, 1125)


def _eval(df, params) -> dict:
    pos, r = S.meanrev_z(df, ma_n=params["ma_n"], entry_z=params["entry_z"],
                         exit_z=params["exit_z"], stop_z=params["stop_z"],
                         max_hold=params["max_hold"],
                         time_lo=SESSION[0], time_hi=SESSION[1])
    close = df["close"].to_numpy(float)
    seam = df["code"].to_numpy()
    trades = run_position_series(close, pos, COSTS, seam=seam, reasons=r)
    m = metrics(trades, close, BARS_PER_DAY)
    return m


def _grid():
    keys = list(GRID)
    for vals in itertools.product(*[GRID[k] for k in keys]):
        yield dict(zip(keys, vals))


def optimize(df_train, min_trades: int = 20) -> tuple[dict, dict]:
    """Лучший набор по Sharpe на train (с фильтром по числу сделок)."""
    best, best_m, best_score = None, None, -1e18
    for p in _grid():
        m = _eval(df_train, p)
        if m["trades"] < min_trades:
            continue
        score = m["sharpe"]
        if score > best_score:
            best, best_m, best_score = p, m, score
    return best, best_m


def optimize_joint(data, tr_lo, tr_hi, min_trades: int = 20):
    """Подбор по СОВМЕСТНОЙ робастности: score = min(Sharpe) по ОБОИМ инструментам на
    train. Так параметры обязаны работать на обоих сразу → перенос не «случайный»."""
    best, best_score = None, -1e18
    metas = {}
    for p in _grid():
        ms = {a: _eval(data[a].iloc[tr_lo:tr_hi], p) for a in ["SBRF", "GAZR"]}
        if min(m["trades"] for m in ms.values()) < min_trades:
            continue
        score = min(m["sharpe"] for m in ms.values())
        if score > best_score:
            best, best_score, metas = p, score, ms
    return best, metas


def walk_forward(train_bars=3500, test_bars=1500, step=1500,
                 opt_asset="SBRF", joint=False, verbose=True):
    """Скользящий WF. opt_asset[train] → тест ОБОИХ[test]; joint=True — отбор по min-Sharpe
    обоих инструментов на train (робастный перенос)."""
    data = {a: build_continuous(a) for a in ["SBRF", "GAZR"]}
    n = min(len(data["SBRF"]), len(data["GAZR"]))
    folds = []
    start = 0
    while start + train_bars + test_bars <= n:
        tr_lo, tr_hi = start, start + train_bars
        te_lo, te_hi = tr_hi, tr_hi + test_bars
        if joint:
            params, metas = optimize_joint(data, tr_lo, tr_hi)
            m_tr = metas.get(opt_asset) if params else None
        else:
            df_tr = data[opt_asset].iloc[tr_lo:tr_hi]
            params, m_tr = optimize(df_tr)
        if params is None:
            start += step
            continue
        res = {"fold": len(folds), "tr": (tr_lo, tr_hi), "te": (te_lo, te_hi),
               "params": params, "train": m_tr, "test": {}}
        for a in ["SBRF", "GAZR"]:
            df_te = data[a].iloc[te_lo:te_hi]
            res["test"][a] = _eval(df_te, params)
        folds.append(res)
        if verbose:
            p = params
            tag = (f"ma{p['ma_n']} ez{p['entry_z']} xz{p['exit_z']} "
                   f"sz{p['stop_z']} mh{p['max_hold']}")
            sb, gz = res["test"]["SBRF"], res["test"]["GAZR"]
            print(f"fold{res['fold']} te[{te_lo}:{te_hi}] {tag:32s} | "
                  f"trainSh {m_tr['sharpe']:+.2f} | "
                  f"OOS SBRF net{sb['net']:+.0f} sh{sb['sharpe']:+.2f} tr{sb['trades']} | "
                  f"GAZR net{gz['net']:+.0f} sh{gz['sharpe']:+.2f} tr{gz['trades']}")
        start += step
    return folds


def summarize(folds):
    for a in ["SBRF", "GAZR"]:
        net = sum(f["test"][a]["net"] for f in folds)
        trs = sum(f["test"][a]["trades"] for f in folds)
        shs = [f["test"][a]["sharpe"] for f in folds]
        pos_folds = sum(1 for f in folds if f["test"][a]["net"] > 0)
        print(f"\n{a} OOS aggregate: net {net:+.0f}₽  trades {trs}  "
              f"mean-fold-Sharpe {np.mean(shs):+.2f}  "
              f"positive folds {pos_folds}/{len(folds)}")


if __name__ == "__main__":
    print("=== WF-A: optimize on SBRF train only, OOS SBRF & GAZR ===")
    summarize(walk_forward(opt_asset="SBRF", joint=False))
    print("\n=== WF-B: optimize on GAZR train only, OOS SBRF & GAZR ===")
    summarize(walk_forward(opt_asset="GAZR", joint=False))
    print("\n=== WF-C: JOINT optimize (min-Sharpe both on train), OOS SBRF & GAZR ===")
    summarize(walk_forward(joint=True))
