"""Эксперименты walk-forward на реальных данных MOEX для st6.

Запускает набор вариантов и печатает реальные OOS-числа:
  A. Текущий боевой st6 in-sample (отбор пары и торговля на одном окне) — baseline,
     показывает завышение.
  B. Честный OOS walk-forward с динамическим отбором пары на train.
  C. Фиксированные экономически-обоснованные пары (без перебора 55).
  D. Бонферрони-поправка при отборе.
  E. Стресс: разные train/test разбиения.
  F. BuyHold корзины за тот же период.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pairsignal.st6.core import Params, rank_pairs  # noqa: E402
from pairsignal.st6.backtest import backtest_pair  # noqa: E402
from pairsignal.research.moex.wf_core import walk_forward  # noqa: E402

CACHE = Path(__file__).resolve().parent / "cache"


def load_union(min_bars: int = 300) -> pd.DataFrame:
    df = pd.read_csv(CACHE / "union_1d.csv", index_col=0)
    # отбрасываем тикеры с большими дырами: оставляем те, что покрывают >= min_bars
    # из самого глубокого пересечения
    return df


def aligned_basket(df: pd.DataFrame, tickers: list[str]) -> dict[str, list[float]]:
    sub = df[[t for t in tickers if t in df.columns]].dropna()
    return {t: sub[t].tolist() for t in sub.columns}, sub.index.to_list()  # type: ignore


def fmt(r) -> str:
    return (f"trades={r.n_trades:>3}  ret={r.return_pct:+7.2f}%  "
            f"win={r.win_rate*100 if r.n_trades else 0:4.0f}%  "
            f"DD={r.max_drawdown_pct:5.2f}%  "
            f"Sh_trade={r.sharpe_pertrade:+5.2f}  Sh_ann={r.sharpe_annual():+5.2f}  "
            f"net={r.total_net:+,.0f}")


def buyhold_basket(prices: dict[str, list[float]]) -> dict:
    """Equal-weight buy&hold корзины за весь период (для сравнения)."""
    rets = []
    for t, v in prices.items():
        a = np.asarray(v, dtype=float)
        rets.append(a[-1] / a[0] - 1.0)
    eqw = float(np.mean(rets))
    # дневной Sharpe equal-weight
    mat = np.array([np.asarray(v, dtype=float) for v in prices.values()])
    dret = np.diff(mat, axis=1) / mat[:, :-1]
    port = dret.mean(axis=0)
    sh = float(port.mean() / port.std() * np.sqrt(252)) if port.std() > 0 else float("nan")
    return {"buyhold_ret_pct": eqw * 100, "buyhold_sharpe_ann": sh, "n_assets": len(prices)}
