"""Тесты cross-sectional momentum (CSMomentum)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pairsignal.momentum import backtest_momentum, momentum_weights


def _ts(n: int) -> np.ndarray:
    return (np.arange(n) * 3_600_000 + 1_700_000_000_000).astype("int64")


def test_long_only_catches_trend():
    """Long-only momentum ловит стабильно растущую монету → return > 0."""
    n = 500
    rng = np.random.default_rng(0)
    # 5 монет: одна с устойчивым трендом вверх, остальные — флэт-шум
    trend = 100 * np.exp(np.cumsum(np.full(n, 0.002)))      # стабильный рост
    flats = {f"F{j}": 100 * np.exp(np.cumsum(rng.normal(0, 0.005, n))) for j in range(4)}
    prices = pd.DataFrame({"TREND": trend, **flats}, index=_ts(n))
    prices.attrs["timeframe"] = "1h"
    m = backtest_momentum(prices, lookback=24, holding=12, k=1,
                          long_short=False, fee=0.0, slippage=0.0)
    assert m["return_pct"] > 0
    assert m["sharpe"] > 0


def test_no_repaint_weight_uses_only_past():
    """Вес на баре t зависит только от данных ≤ t (no repaint).

    Меняем цены строго ПОСЛЕ бара t — веса до t включительно не меняются.
    """
    n = 300
    rng = np.random.default_rng(1)
    prices = pd.DataFrame(
        {c: 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))) for c in ("A", "B", "C", "D")},
        index=_ts(n),
    )
    w1 = momentum_weights(prices, lookback=24, holding=12, k=1, long_short=True)
    t = 200
    perturbed = prices.copy()
    perturbed.iloc[t + 1:] *= 1.5  # меняем будущее относительно t
    w2 = momentum_weights(perturbed, lookback=24, holding=12, k=1, long_short=True)
    # веса на барах ≤ t не должны измениться
    pd.testing.assert_frame_equal(w1.iloc[: t + 1], w2.iloc[: t + 1])


def test_costs_monotonic():
    """Рост комиссии → ниже итоговый equity (издержки реальны)."""
    n = 500
    rng = np.random.default_rng(2)
    prices = pd.DataFrame(
        {c: 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))) for c in
         ("A", "B", "C", "D", "E", "F")},
        index=_ts(n),
    )
    prices.attrs["timeframe"] = "1h"
    low = backtest_momentum(prices, 24, 12, k=2, long_short=True, fee=0.0, slippage=0.0)
    high = backtest_momentum(prices, 24, 12, k=2, long_short=True, fee=0.002, slippage=0.0)
    assert high["return_pct"] < low["return_pct"]
    assert high["costs_pct"] > low["costs_pct"]


def test_dollar_neutral_weights_sum_zero():
    """Long/short веса на барах ребаланса доллар-нейтральны: Σ ≈ 0."""
    n = 300
    rng = np.random.default_rng(3)
    prices = pd.DataFrame(
        {c: 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))) for c in
         ("A", "B", "C", "D", "E", "F")},
        index=_ts(n),
    )
    w = momentum_weights(prices, lookback=24, holding=12, k=2, long_short=True)
    active = w[(w != 0).any(axis=1)]
    assert np.allclose(active.sum(axis=1), 0.0, atol=1e-9)
