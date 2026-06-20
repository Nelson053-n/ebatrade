"""trend_wf — ЧЕСТНЫЙ walk-forward time-series trend-following по корзине MOEX.

Скользящие окна: на train подбираем ОДИН набор параметров (по Sharpe корзины),
прогоняем на следующем невиданном test. ОДИН набор на все тикеры. Параметры
выбираются только на train. Equity усредняется по корзине (диверсификация).

Семейство выбирается флагом. Прогрев индикатора берём из хвоста train (no leak:
параметры уже зафиксированы по train, прогрев лишь даёт корректные значения SMA/канала
на старте test — это не подглядывание в метрику test).

Также: half-split (первая/вторая половина истории отдельным WF), jackknife
(выкидываем по одному тикеру), стресс по издержкам.
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd

from trend_core import (_BARS_PER_YEAR, backtest_one, buyhold_ew, d, load_closes,
                        metrics, sig_donchian, sig_ma_cross, sig_tsmom)


# --- фабрики сигналов по семейству + сетки параметров -------------------------

def make_grid(family: str, interval: str):
    """Возвращает (список параметров, функция params->sig_fn)."""
    if family == "donchian":
        ens = [50, 100, 200, 400] if interval == "1h" else [20, 50, 100]
        params = [(en, max(5, en // 2), ls) for en in ens for ls in (False, True)]
        fn = lambda p: (lambda c: sig_donchian(c, p[0], p[1], p[2]))
    elif family == "ma":
        if interval == "1h":
            combos = [(0, 100), (0, 200), (0, 400), (24, 100), (48, 200), (100, 400)]
        else:
            combos = [(0, 20), (0, 50), (0, 100), (0, 200), (10, 50), (20, 100), (50, 200)]
        params = [(f, s, ls) for (f, s) in combos for ls in (False, True)]
        fn = lambda p: (lambda c: sig_ma_cross(c, p[1], p[0], p[2]))
    elif family == "tsmom":
        lbs = [100, 200, 400, 800] if interval == "1h" else [20, 50, 100, 200]
        params = [(lb, ls) for lb in lbs for ls in (False, True)]
        fn = lambda p: (lambda c: sig_tsmom(c, p[0], p[1]))
    else:
        raise ValueError(family)
    return params, fn


def basket_net_warm(closes: pd.DataFrame, sig_fn, fee: float, slip: float,
                    warm: int) -> pd.Series:
    """Чистая доходность корзины, но метрики берём с warm-го бара (прогрев индикатора)."""
    cols = []
    for t in closes.columns:
        c = closes[t].dropna()
        if len(c) < warm + 10:
            continue
        pos = sig_fn(c)
        net = backtest_one(c, pos, fee, slip)
        cols.append(net.reindex(closes.index))
    if not cols:
        return pd.Series(0.0, index=closes.index)
    return pd.concat(cols, axis=1).mean(axis=1, skipna=True).fillna(0.0)


def basket_sharpe(closes: pd.DataFrame, sig_fn, fee: float, slip: float, interval: str) -> float:
    net = basket_net_warm(closes, sig_fn, fee, slip, 0)
    return metrics(net, interval)["sharpe"]


def walk_forward(closes: pd.DataFrame, family: str, interval: str,
                 train_bars: int, test_bars: int, fee: float, slip: float,
                 warm: int, verbose: bool = True) -> dict:
    """Скользящий WF. Возвращает склеенную OOS equity + метрики + список окон."""
    params, mkfn = make_grid(family, interval)
    idx = closes.index
    oos_segments = []
    win_rows = []
    start = 0
    while start + train_bars + test_bars <= len(idx):
        train = closes.iloc[start:start + train_bars]
        # подбор лучших параметров на train по Sharpe корзины
        best, best_sh = None, -np.inf
        for p in params:
            sh = basket_sharpe(train, mkfn(p), fee, slip, interval)
            if sh > best_sh:
                best_sh, best = sh, p
        # OOS: считаем на test, но с прогревом хвостом train (warm баров до test)
        seg_lo = start + train_bars - warm
        seg = closes.iloc[seg_lo:start + train_bars + test_bars]
        net = basket_net_warm(seg, mkfn(best), fee, slip, warm)
        oos_net = net.iloc[warm:]                       # только test-часть
        oos_segments.append(oos_net)
        m = metrics(oos_net, interval)
        win_rows.append({"start": idx[start + train_bars], "end": idx[min(start + train_bars + test_bars - 1, len(idx) - 1)],
                         "params": best, "ret_pct": m["ret_pct"], "sharpe": m["sharpe"]})
        start += test_bars

    if not oos_segments:
        return {}
    full = pd.concat(oos_segments)
    M = metrics(full, interval)
    wins = sum(1 for r in win_rows if r["ret_pct"] > 0)
    res = {"net": full, "ret_pct": M["ret_pct"], "sharpe": M["sharpe"],
           "max_dd_pct": M["max_dd_pct"], "n_windows": len(win_rows),
           "wins": wins, "windows": win_rows}
    if verbose:
        print(f"\n{'window':23} {'params':18} {'ret%':>8} {'Sharpe':>7}")
        print("-" * 60)
        for r in win_rows:
            print(f"{d(r['start'])}..{d(r['end']):11} {str(r['params']):18} "
                  f"{r['ret_pct']:>+8.1f} {r['sharpe']:>7.2f}")
        print("-" * 60)
        print(f"СКЛЕЕННЫЙ OOS: ret {M['ret_pct']:+.1f}%  Sharpe {M['sharpe']:.2f}  "
              f"DD {M['max_dd_pct']:.1f}%  прибыльных окон {wins}/{len(win_rows)}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=["donchian", "ma", "tsmom"])
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--train-days", type=int, default=365)
    ap.add_argument("--test-days", type=int, default=90)
    ap.add_argument("--fee", type=float, default=0.0005)
    ap.add_argument("--slip", type=float, default=0.0003)
    args = ap.parse_args()

    closes = load_closes(args.interval)
    bph = _BARS_PER_YEAR[args.interval] / (252 if args.interval == "1d" else 365)
    train_bars = int(args.train_days * bph)
    test_bars = int(args.test_days * bph)
    warm = 800 if args.interval == "1h" else 200
    print(f"=== WF {args.family} {args.interval}: {closes.shape[1]} тикеров, {closes.shape[0]} баров, "
          f"train={train_bars} test={test_bars} warm={warm} fee={args.fee} slip={args.slip} ===")
    bh = buyhold_ew(closes, args.interval)
    print(f"BuyHold EW: ret {bh['ret_pct']:+.1f}%  Sharpe {bh['sharpe']:.2f}  DD {bh['max_dd_pct']:.1f}%")
    walk_forward(closes, args.family, args.interval, train_bars, test_bars,
                 args.fee, args.slip, warm)


if __name__ == "__main__":
    main()
