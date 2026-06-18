# -*- coding: utf-8 -*-
"""
Юнит-тесты st6 — контроль качества торговой логики (без сети).
Перенос test_st6.py из референса на импорты пакета pairsignal.st6.

Запуск:  pytest tests/test_st6.py -q
"""

import numpy as np

from pairsignal.st6.core import (
    ExitReason, Params, Position, Side,
    decide, half_life, hedge_ratio, leg_quantities, rank_pairs,
    rolling_correlation, trade_pnl, zscore,
)
from pairsignal.st6.backtest import backtest_pair, make_synthetic_pair


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


# -------------------------------------------------------------- математика
def test_hedge_ratio_recovers_beta():
    rng = np.random.default_rng(1)
    lb = np.cumsum(rng.normal(0, 0.01, 500)) + 5
    la = 1.7 * lb + 0.3 + rng.normal(0, 1e-6, 500)
    beta, alpha = hedge_ratio(la, lb)
    assert approx(beta, 1.7, 1e-2), beta
    assert approx(alpha, 0.3, 1e-2), alpha


def test_correlation_high_for_linked_series():
    rng = np.random.default_rng(2)
    b = np.exp(np.cumsum(rng.normal(0, 0.01, 400)) + 4)
    a = b * 1.5
    assert rolling_correlation(a, b, 120) > 0.99


def test_correlation_low_for_independent():
    rng = np.random.default_rng(3)
    a = np.exp(np.cumsum(rng.normal(0, 0.01, 400)))
    b = np.exp(np.cumsum(rng.normal(0, 0.01, 400)))
    assert abs(rolling_correlation(a, b, 200)) < 0.3


def test_zscore_zero_at_mean():
    s = np.array([-2.0, -1.0, 1.0, 2.0, 0.0])
    assert approx(zscore(s, 5), 0.0, 1e-9)


def test_zscore_sign():
    s = np.concatenate([np.zeros(99), [5.0]])
    assert zscore(s, 100) > 2


def test_halflife_positive_for_mean_reverting():
    rng = np.random.default_rng(4)
    s = np.zeros(1000)
    for t in range(1, 1000):
        s[t] = 0.9 * s[t - 1] + rng.normal(0, 0.1)
    hl = half_life(s)
    assert 0 < hl < 50, hl


# -------------------------------------------------------------- знак P&L
def test_pnl_long_spread_profits_when_a_outperforms():
    p = Params(fee_rate=0, slippage_rate=0)
    net = trade_pnl(Side.LONG_SPREAD, entry_a=100, exit_a=110,
                    entry_b=100, exit_b=90, units_a=10, units_b=10, p=p)
    assert net > 0, net


def test_pnl_short_spread_profits_when_a_underperforms():
    p = Params(fee_rate=0, slippage_rate=0)
    net = trade_pnl(Side.SHORT_SPREAD, entry_a=100, exit_a=90,
                    entry_b=100, exit_b=110, units_a=10, units_b=10, p=p)
    assert net > 0, net


def test_costs_reduce_pnl():
    a = trade_pnl(Side.LONG_SPREAD, 100, 110, 100, 90, 10, 10,
                  Params(fee_rate=0, slippage_rate=0))
    b = trade_pnl(Side.LONG_SPREAD, 100, 110, 100, 90, 10, 10,
                  Params(fee_rate=0.001, slippage_rate=0.001))
    assert b < a


# -------------------------------------------------------------- sizing
def test_leg_quantities_dollar_neutral_by_beta():
    p = Params(risk_fraction=0.10)
    qa, qb = leg_quantities(1_000_000, price_a=100, price_b=200,
                            beta=1.0, lot_a=1, lot_b=1, p=p)
    assert qa == 1000 and qb == 500, (qa, qb)


# -------------------------------------------------------------- FSM
def _mr_series(n=1200, seed=11):
    return make_synthetic_pair(n=n, seed=seed)


def test_decide_enters_on_extreme_z():
    a, b = _mr_series()
    p = Params()
    sig = decide(a, b, Position(), p)
    if abs(sig.z) >= p.z_entry and abs(sig.corr) >= p.corr_enter:
        assert sig.action in ("ENTER_LONG", "ENTER_SHORT")
        if sig.z >= p.z_entry:
            assert sig.action == "ENTER_SHORT"
        else:
            assert sig.action == "ENTER_LONG"


def test_decide_no_entry_when_corr_gate_blocks():
    a, b = _mr_series()
    p = Params(corr_enter=1.01)  # impossible gate -> entry always blocked
    sig = decide(a, b, Position(), p)
    assert sig.action == "HOLD"


def test_decide_exit_on_corr_break():
    a, b = _mr_series()
    p = Params(corr_break=1.01)  # any corr below -> emergency exit
    pos = Position(side=Side.LONG_SPREAD, beta=1.0, qty_a=1, qty_b=1,
                   entry_a=a[-1], entry_b=b[-1])
    sig = decide(a, b, pos, p)
    assert sig.action == "EXIT" and sig.reason == ExitReason.CORR_BREAK


def test_decide_take_profit_near_mean():
    a, b = _mr_series()
    p = Params(z_exit=10.0)  # any |z|<=10 -> take profit
    pos = Position(side=Side.LONG_SPREAD, beta=1.0, qty_a=1, qty_b=1,
                   entry_a=a[-1], entry_b=b[-1])
    sig = decide(a, b, pos, p)
    assert sig.action == "EXIT" and sig.reason == ExitReason.TAKE


# -------------------------------------------------------------- pair scan
def test_rank_pairs_finds_cointegrated_pair():
    a, b = make_synthetic_pair(n=1500, beta=1.0, seed=5)
    rng = np.random.default_rng(99)
    noise = np.exp(np.cumsum(rng.normal(0, 0.01, 1500)) + 4)  # independent
    series = {"AA": a, "BB": b, "NN": noise}
    ranked = rank_pairs(series, Params())
    assert ranked, "should find at least one valid pair"
    top = ranked[0]
    assert {top.a, top.b} == {"AA", "BB"}, (top.a, top.b)


# -------------------------------------------------------------- backtest
def test_backtest_runs_and_trades():
    a, b = make_synthetic_pair(n=2500, seed=7)
    r = backtest_pair(a, b, Params(), start_equity=1_000_000.0)
    assert r.n_trades > 0, "cointegrated pair should produce trades"
    assert len(r.equity_curve) > 0
    assert min(r.equity_curve) > 0


def test_backtest_no_lookahead_determinism():
    a, b = make_synthetic_pair(n=2000, seed=3)
    r1 = backtest_pair(a, b, Params())
    r2 = backtest_pair(a, b, Params())
    assert r1.total_net == r2.total_net


# -------------------------------------------------------------- St6Session
def test_st6_session_creates_and_snapshots():
    from pairsignal.st6.service import St6Session
    s = St6Session("test")
    snap = s.snapshot(0.0)
    # снапшот не падает и содержит ключевые поля
    for key in ("sid", "live", "summary", "pair", "wait_reason", "params",
                "history", "trades", "events"):
        assert key in snap, key
    assert snap["summary"]["trades"] == 0


def test_st6_session_player_trades_on_synthetic():
    """Прогон синтетики через FSM сессии: выбирает пару, делает paper-сделки."""
    import asyncio
    from pairsignal.st6 import data_feed as feed
    from pairsignal.st6.service import St6Session
    s = St6Session("test_player")
    s.player_closes = feed.synthetic_basket(n=1200, seed=7)
    s.state["player"] = True
    asyncio.run(s.run_player())
    snap = s.snapshot(0.0)
    assert snap["pair"] is not None, "должна выбраться коинтегрированная пара"
    assert snap["summary"]["trades"] > 0, "синтетика должна дать сделки"
