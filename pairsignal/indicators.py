"""Индикаторы: динамическая бета, спред, полосы Боллинджера, z-score, ширина канала.

Все расчёты — по закрытым свечам (no repaint). Возвращаем DataFrame с колонками,
готовый к скармливанию в SignalEngine построчно.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StrategyConfig


def rolling_beta(log_a: pd.Series, log_b: pd.Series, window: int) -> pd.Series:
    """Хедж-β как OLS-наклон по ЛОГ-ДОХОДНОСТЯМ: cov(Δa, Δb) / var(Δb).

    Раньше β считалась по УРОВНЯМ лог-цен (cov(a,b)/var(b)). На реальных BTC/ETH
    это вырождается: в боковике var(уровней) → 0, β скачет (std ~0.5) и в ~20%
    баров уходит в минус (бессмыслица для доллар-нейтрального хеджа), а сам спред
    ln(A) − β·ln(B) теряет возврат к среднему (ADF γ > 0 на 3 из 4 окон истории).
    По доходностям β стабильна (std ~0.02, без отрицательных), а спред становится
    mean-reverting (γ < 0 во всех окнах) — то, на чём держится strategy.py.
    """
    ra, rb = log_a.diff(), log_b.diff()
    cov = ra.rolling(window).cov(rb)
    var = rb.rolling(window).var()
    beta = cov / var
    return beta.replace([np.inf, -np.inf], np.nan)


def build_indicators(
    df: pd.DataFrame, cfg: StrategyConfig
) -> pd.DataFrame:
    """df: индекс — ts(ms), колонки 'price_a','price_b' (close обеих ног, выровненные)."""
    out = df.copy()

    if cfg.spread_mode == "cross_pct":
        # кросс-биржевой: линейный спред, полосы = SMA ± band. z нормирован так, что
        # |z|=1 ровно на полосе, z=0 на SMA — existing SignalEngine (entry_z=1.0) даёт
        # вход по пробою, выход — по возврату z к 0. band — два режима (см. config):
        #   vol: bb_k·σ(спреда) — адаптивно (реальные данные); pct: band_pct·price (демо).
        out["beta"] = 1.0
        out["spread"] = out["price_a"] - out["price_b"]
        mid = out["spread"].rolling(cfg.sma_period).mean()
        if cfg.band_mode == "vol":
            sd = out["spread"].rolling(cfg.sma_period).std(ddof=0)
            band = (cfg.bb_k * sd).clip(lower=1e-9)
            out["std"] = sd                  # настоящая σ спреда (для approve/стопа)
        else:  # pct
            band = (cfg.band_pct * out["price_a"]).clip(lower=1e-9)
            out["std"] = band / cfg.bb_k     # псевдо-σ: (upper−mid)/bb_k = band/bb_k
        out["mid"] = mid
        out["upper"] = mid + band
        out["lower"] = mid - band
        out["z"] = (out["spread"] - mid) / band
        out["width_pct"] = 100.0            # анти-флэт фильтр не применим к кросс-режиму
        return out

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
