"""Тесты live paper-портфеля momentum (без сети)."""
from __future__ import annotations

import json

from pairsignal.momentum import (
    PaperPortfolio,
    load_state,
    save_state,
    target_weights_now,
)


def test_rebalance_charges_turnover_cost():
    """Ребаланс с нуля списывает издержки оборота с equity."""
    p = PaperPortfolio(start_equity=1000.0, fee=0.001, slippage=0.0)
    target = {"A/USDT:USDT": 0.5, "B/USDT:USDT": -0.5}
    prices = {"A/USDT:USDT": 10.0, "B/USDT:USDT": 20.0}
    rec = p.rebalance(target, prices, ts=1000)
    # оборот = |0.5−0| + |−0.5−0| = 1.0 → cost = 1.0·0.001 = 0.001 → equity 999.0
    assert rec["turnover"] == 1.0
    assert abs(p.equity - 999.0) < 1e-9
    assert rec["longs"] == ["A/USDT:USDT"]
    assert rec["shorts"] == ["B/USDT:USDT"]


def test_mark_to_market_moves_equity():
    """Mark-to-market меняет equity по движению цен (long прибылен при росте)."""
    p = PaperPortfolio(start_equity=1000.0, fee=0.0, slippage=0.0)
    p.rebalance({"A/USDT:USDT": 1.0}, {"A/USDT:USDT": 10.0}, ts=1000)  # лонг A
    p.mark({"A/USDT:USDT": 11.0}, ts=2000)                            # A +10%
    assert abs(p.equity - 1100.0) < 1e-6


def test_dollar_neutral_market_move_cancels():
    """Доллар-нейтраль long/short: равное движение обеих ног почти не двигает equity."""
    p = PaperPortfolio(start_equity=1000.0, fee=0.0, slippage=0.0)
    p.rebalance({"A/USDT:USDT": 0.5, "B/USDT:USDT": -0.5},
                {"A/USDT:USDT": 10.0, "B/USDT:USDT": 10.0}, ts=1000)
    p.mark({"A/USDT:USDT": 11.0, "B/USDT:USDT": 11.0}, ts=2000)  # обе +10%
    # +0.5·10% − 0.5·10% = 0 → equity без изменений
    assert abs(p.equity - 1000.0) < 1e-6


def test_state_roundtrip(tmp_path):
    """save_state/load_state сохраняют и восстанавливают портфель один-в-один."""
    p = PaperPortfolio(start_equity=1234.5, fee=0.0006, slippage=0.0002)
    p.rebalance({"X/USDT:USDT": 0.5, "Y/USDT:USDT": -0.5},
                {"X/USDT:USDT": 5.0, "Y/USDT:USDT": 8.0}, ts=42)
    path = tmp_path / "state.json"
    save_state(path, p, {"bars_since_reb": 3, "lookback": 24})
    loaded = load_state(path)
    assert loaded is not None
    p2, meta = loaded
    assert abs(p2.equity - p.equity) < 1e-9
    assert p2.weights == p.weights
    assert meta["lookback"] == 24
    assert len(p2.rebalances) == 1


def test_save_state_atomic(tmp_path):
    """Запись атомарна: итоговый файл — валидный JSON, временного .tmp не остаётся."""
    p = PaperPortfolio(start_equity=1000.0, fee=0.0, slippage=0.0)
    path = tmp_path / "s.json"
    save_state(path, p, {})
    assert path.exists()
    assert not path.with_suffix(".json.tmp").exists()
    json.loads(path.read_text())  # валидный JSON


def test_target_weights_now_ranks():
    """target_weights_now лонгует лидера по доходности, шортит аутсайдера."""
    import numpy as np
    import pandas as pd
    n = 50
    # A растёт сильнее всех, D падает — на последнем баре A=лидер, D=аутсайдер
    data = {
        "A/USDT:USDT": np.linspace(100, 130, n),
        "B/USDT:USDT": np.linspace(100, 110, n),
        "C/USDT:USDT": np.linspace(100, 105, n),
        "D/USDT:USDT": np.linspace(100, 90, n),
    }
    prices = pd.DataFrame(data)
    w = target_weights_now(prices, lookback=24, k=1, long_short=True)
    assert w["A/USDT:USDT"] > 0   # лидер — лонг
    assert w["D/USDT:USDT"] < 0   # аутсайдер — шорт
