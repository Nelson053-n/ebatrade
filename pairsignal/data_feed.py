"""Источник данных.

read_ohlcv_ccxt — реальные котировки (только чтение, ключи не нужны для публичных свечей).
generate_synthetic — коинтегрированная пара для офлайн-демо/тестов без сети.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StrategyConfig


def read_ohlcv_ccxt(cfg: StrategyConfig, limit: int = 1000) -> pd.DataFrame:
    """Тянем close обеих ног через CCXT и выравниваем по времени.

    Публичные OHLCV не требуют API-ключей. Сетевой доступ к бирже нужен в рантайме.
    """
    import ccxt  # импорт внутри, чтобы офлайн-демо работало без установленного ccxt

    ex = getattr(ccxt, cfg.data_exchange)({"enableRateLimit": True})
    ex.options["defaultType"] = "swap"

    def _close(symbol: str) -> pd.Series:
        raw = ex.fetch_ohlcv(symbol, timeframe=cfg.timeframe, limit=limit)
        s = pd.Series(
            [c[4] for c in raw], index=[int(c[0]) for c in raw], dtype="float64"
        )
        s.index.name = "ts"
        return s

    a = _close(cfg.symbol_a)
    b = _close(cfg.symbol_b)
    df = pd.DataFrame({"price_a": a, "price_b": b}).dropna()
    df = df.sort_index()
    return df


def generate_synthetic(n: int = 3000, seed: int = 7) -> pd.DataFrame:
    """Пара с возвращающимся к среднему лог-спредом — чтобы демо реально давало сигналы."""
    rng = np.random.default_rng(seed)
    # общий рыночный фактор (случайное блуждание)
    factor = np.cumsum(rng.normal(0, 0.004, n))
    p_eth = 3000 * np.exp(factor)

    # спред = ln(BTC) - ln(ETH) колеблется вокруг ~2.4 (BTC/ETH ≈ 11) по OU-процессу
    spread = np.zeros(n)
    spread[0] = 2.4
    theta, mu, sigma = 0.02, 2.4, 0.01
    for i in range(1, n):
        spread[i] = spread[i - 1] + theta * (mu - spread[i - 1]) + rng.normal(0, sigma)

    p_btc = np.exp(np.log(p_eth) + spread)
    ts = (np.arange(n) * 300_000 + 1_700_000_000_000).astype("int64")  # шаг 5m
    df = pd.DataFrame({"price_a": p_btc, "price_b": p_eth}, index=ts)
    df.index.name = "ts"
    return df
