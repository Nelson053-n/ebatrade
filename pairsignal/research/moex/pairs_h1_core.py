"""Часовой парный mean-reversion на MOEX: общие утилиты + статистика.

Переиспользует ЧИСТОЕ ядро st6.core (decide/rank_pairs/leg_quantities/trade_pnl)
и честный walk-forward из wf_core.py. Здесь — только часовой загрузчик данных,
корректная аннуализация (≈15 баров/торговый день) и робастность-тесты
(jackknife по сделкам, half-split, bootstrap t-стат на серии сделок).

⛔ только чтение CSV-кэша, никаких ордеров. Боевой st6 не трогаем.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pairsignal.st6.core import Params  # noqa: E402
from pairsignal.research.moex.wf_core import walk_forward, WFResult  # noqa: E402

CACHE = Path(__file__).resolve().parent / "cache"

# ~15 часовых баров в торговом дне на MOEX (median по кэшу), 252 торг. дня
BARS_PER_DAY = 15.0
BARS_PER_YEAR = BARS_PER_DAY * 252.0  # ≈3780

CORE_TICKERS = [
    "LKOH", "ROSN", "SIBN", "TATN", "GAZP", "NVTK", "TATNP", "BANE", "RNFT",
    "GMKN", "NLMK", "MAGN", "CHMF", "PLZL", "RUAL", "ALRS",
    "SBER", "SBERP", "VTBR", "CBOM",
    "HYDR", "IRAO", "FEES", "UPRO", "MTSS", "RTKM", "MGNT",
]

# Экономически-обоснованные внутрисекторные пары (минимум data-snooping)
FIXED_PAIRS = [
    ("NLMK", "CHMF"),    # сталь
    ("NLMK", "MAGN"),    # сталь
    ("MAGN", "CHMF"),    # сталь
    ("LKOH", "ROSN"),    # нефть
    ("LKOH", "SIBN"),    # нефть
    ("ROSN", "SIBN"),    # нефть
    ("SIBN", "TATN"),    # нефть
    ("TATN", "ROSN"),    # нефть
    ("TATN", "TATNP"),   # обычка/преф один эмитент
    ("SBER", "SBERP"),   # обычка/преф один эмитент
]


def load_union(tickers: list[str]) -> tuple[dict[str, list[float]], pd.Index]:
    """Все тикеры по общему индексу (dropna по пересечению)."""
    df = pd.read_csv(CACHE / "union_1h.csv", index_col=0)
    sub = df[[t for t in tickers if t in df.columns]].dropna()
    return {t: sub[t].tolist() for t in sub.columns}, sub.index


def load_pair(a: str, b: str) -> tuple[dict[str, list[float]], pd.Index]:
    """Только две ноги, dropna по их пересечению (максимум баров)."""
    df = pd.read_csv(CACHE / "union_1h.csv", index_col=0)
    sub = df[[a, b]].dropna()
    return {a: sub[a].tolist(), b: sub[b].tolist()}, sub.index


# --------------------------------------------------------------------------
# Статистика по серии сделок
# --------------------------------------------------------------------------
def trade_tstat(rets: np.ndarray) -> float:
    """t-стат среднего по доходностям сделок (H0: mean=0)."""
    r = np.asarray(rets, dtype=float)
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r))))


def jackknife_tstat(rets: np.ndarray) -> dict:
    """Leave-one-out: насколько среднее держится без любой одной сделки.
    Возвращает min/max среднего по LOO и долю положительных LOO-средних."""
    r = np.asarray(rets, dtype=float)
    n = len(r)
    if n < 5:
        return {"loo_min": float("nan"), "loo_max": float("nan"),
                "loo_pos_frac": float("nan")}
    total = r.sum()
    loo_means = (total - r) / (n - 1)
    return {
        "loo_min": float(loo_means.min()),
        "loo_max": float(loo_means.max()),
        "loo_pos_frac": float((loo_means > 0).mean()),
    }


def top_trade_share(nets: np.ndarray) -> float:
    """Доля совокупного профита, приходящаяся на топ-3 сделки (концентрация)."""
    n = np.asarray(nets, dtype=float)
    pos = n[n > 0]
    if pos.sum() <= 0:
        return float("nan")
    k = min(3, len(pos))
    return float(np.sort(pos)[-k:].sum() / n.sum()) if n.sum() > 0 else float("nan")


def half_split(prices: dict, p: Params, train_bars: int, test_bars: int,
               step: int, **kw) -> tuple[WFResult, WFResult]:
    """WF на первой и второй половине ряда отдельно."""
    n = min(len(v) for v in prices.values())
    mid = n // 2
    p1 = {t: v[:mid] for t, v in prices.items()}
    p2 = {t: v[mid:] for t, v in prices.items()}
    r1 = walk_forward(p1, p, train_bars, test_bars, step, **kw)
    r2 = walk_forward(p2, p, train_bars, test_bars, step, **kw)
    return r1, r2


def summarize(r: WFResult) -> dict:
    """Сводка по WF-результату с часовой аннуализацией и робастностью."""
    rets = np.array([t.ret_on_notional for t in r.trades], dtype=float)
    nets = np.array([t.net for t in r.trades], dtype=float)
    jk = jackknife_tstat(rets)
    return {
        "trades": r.n_trades,
        "folds": sum(1 for f in r.fold_log if f["pair"]),
        "ret_pct": r.return_pct,
        "win": r.win_rate * 100 if r.n_trades else float("nan"),
        "dd": r.max_drawdown_pct,
        "sharpe_ann": r.sharpe_annual(bars_per_year=BARS_PER_YEAR),
        "sharpe_trade": r.sharpe_pertrade,
        "tstat": trade_tstat(rets),
        "mean_ret_bps": rets.mean() * 1e4 if len(rets) else float("nan"),
        "top3_share": top_trade_share(nets),
        "loo_pos_frac": jk["loo_pos_frac"],
    }
