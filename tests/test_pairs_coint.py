"""Тесты отбора пар по коинтеграции (PairsCoint)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pairsignal.pairs_coint import _half_life, select_pairs


def _coint_universe(n: int = 1500, seed: int = 3) -> pd.DataFrame:
    """Юниверс из 4 монет: COINTA/COINTB коинтегрированы (общий фактор + стац. спред),
    RWX/RWY — независимые рандом-уолки (не коинтегрированы)."""
    rng = np.random.default_rng(seed)
    factor = np.cumsum(rng.normal(0, 0.01, n))           # общий рыночный фактор
    base = 100 * np.exp(factor)
    # коинтегрированная пара: общий уровень + малый стационарный (OU) разброс
    ou = np.zeros(n)
    for i in range(1, n):
        ou[i] = ou[i - 1] * 0.9 + rng.normal(0, 0.005)   # mean-reverting
    a = base * np.exp(ou)
    b = base * np.exp(-ou)
    # независимые рандом-уолки
    x = 50 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    y = 30 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    ts = (np.arange(n) * 300_000 + 1_700_000_000_000).astype("int64")
    return pd.DataFrame(
        {"COINTA/USDT:USDT": a, "COINTB/USDT:USDT": b,
         "RWX/USDT:USDT": x, "RWY/USDT:USDT": y},
        index=ts,
    )


def test_select_pairs_finds_cointegrated():
    """Коинтегрированная пара отбирается, независимые рандом-уолки — нет."""
    prices = _coint_universe()
    pairs = select_pairs(prices, pvalue_max=0.05, corr_min=0.5,
                         max_half_life=10_000, top_pairs=10, workers=2)
    found = {tuple(sorted((p["sym_a"], p["sym_b"]))) for p in pairs}
    coint_pair = tuple(sorted(("COINTA/USDT:USDT", "COINTB/USDT:USDT")))
    rw_pair = tuple(sorted(("RWX/USDT:USDT", "RWY/USDT:USDT")))
    assert coint_pair in found            # настоящая коинтеграция найдена
    assert rw_pair not in found           # ложная (рандом-уолки) отвергнута


def test_half_life_mean_reverting_finite():
    """Half-life стационарного OU-ряда конечен и положителен."""
    rng = np.random.default_rng(0)
    s = np.zeros(2000)
    for i in range(1, 2000):
        s[i] = s[i - 1] * 0.8 + rng.normal(0, 1)  # сильный возврат
    hl = _half_life(s)
    assert 0 < hl < 100


def test_half_life_random_walk_large():
    """Half-life рандом-уолка много больше, чем у mean-reverting ряда.

    На конечной выборке λ может оказаться слабо отрицательным из-за шума (не строго inf),
    поэтому фильтрация идёт по порогу max_half_life, а не по равенству inf. Главное —
    half-life RW на порядок длиннее, чем у настоящего возврата.
    """
    rng = np.random.default_rng(1)
    rw = np.cumsum(rng.normal(0, 1, 2000))
    ou = np.zeros(2000)
    for i in range(1, 2000):
        ou[i] = ou[i - 1] * 0.8 + rng.normal(0, 1)
    assert _half_life(rw) > 10 * _half_life(ou)
