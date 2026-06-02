"""Индикаторы: динамическая бета, спред, полосы Боллинджера, z-score, ширина канала.

Все расчёты — по закрытым свечам (no repaint). Возвращаем DataFrame с колонками,
готовый к скармливанию в SignalEngine построчно.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StrategyConfig


def rolling_beta(log_a: pd.Series, log_b: pd.Series, window: int) -> pd.Series:
    """Хедж-β по ЛОГ-ДОХОДНОСТЯМ, РАСШИРЯЮЩИМСЯ окном: cov(Δa,Δb)/var(Δb) по всей
    истории ДО текущего бара (expanding, no-repaint).

    Почему не rolling: при скользящей β коэффициент дрейфует от бара к бару, и знак
    «сходимости спреда» в момент входа (β входа, по ней считается P&L) отличается от
    знака в момент выхода (другая β) — сделка может выйти «у средней», но в убыток.
    Расширяющаяся β меняется крайне медленно и практически постоянна на горизонте
    одной сделки → вход, выход, средняя и P&L согласованы в одной β, и выход у
    средней даёт прибыль геометрически. `window` задаёт минимум баров для оценки.
    """
    ra, rb = log_a.diff(), log_b.diff()
    cov = ra.expanding(min_periods=window).cov(rb)
    var = rb.expanding(min_periods=window).var()
    beta = cov / var
    return beta.replace([np.inf, -np.inf], np.nan)


def build_indicators(
    df: pd.DataFrame, cfg: StrategyConfig
) -> pd.DataFrame:
    """df: индекс — ts(ms), колонки 'price_a','price_b' (close обеих ног, выровненные)."""
    out = df.copy()

    if cfg.spread_mode == "ratio":
        out["beta"] = 1.0
        out["spread"] = out["price_a"] / out["price_b"]
    else:  # log
        log_a, log_b = np.log(out["price_a"]), np.log(out["price_b"])
        out["beta"] = rolling_beta(log_a, log_b, cfg.beta_window)
        out["spread"] = log_a - out["beta"] * log_b

    mid = out["spread"].rolling(cfg.bb_period).mean()
    std = out["spread"].rolling(cfg.bb_period).std(ddof=0)
    out["mid"] = mid
    out["std"] = std
    out["upper"] = mid + cfg.bb_k * std
    out["lower"] = mid - cfg.bb_k * std
    out["z"] = (out["spread"] - mid) / std.replace(0, np.nan)

    # относительная полуширина канала, % (страховка от деления на ~0)
    denom = mid.abs().replace(0, np.nan)
    out["width_pct"] = (cfg.bb_k * std) / denom * 100.0

    return out
