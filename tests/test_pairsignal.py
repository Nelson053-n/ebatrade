"""Тесты phase 1: индикаторы, логика сигналов, paper-P&L, human-in-the-loop, API."""
from __future__ import annotations

from pairsignal.config import AppConfig, StrategyConfig
from pairsignal.data_feed import generate_synthetic
from pairsignal.engine import Engine
from pairsignal.indicators import build_indicators
from pairsignal.models import Action, IndicatorRow, SpreadDirection
from pairsignal.strategy import SignalEngine
from pairsignal.virtual_exchange import VirtualExchange


def _row(z, width=5.0, std=1.0, mid=0.0, pa=100.0, pb=50.0, beta=1.0, ts=0):
    spread = mid + z * std
    return IndicatorRow(
        ts=ts, price_a=pa, price_b=pb, spread=spread, beta=beta,
        mid=mid, upper=mid + 2 * std, lower=mid - 2 * std, std=std,
        z=z, width_pct=width,
    )


# --- индикаторы ---
def test_indicators_columns_and_warmup():
    cfg = StrategyConfig()
    df = generate_synthetic(n=600)
    ind = build_indicators(df, cfg)
    for col in ("spread", "beta", "mid", "upper", "lower", "z", "width_pct"):
        assert col in ind.columns
    # после прогрева z определён
    assert ind["z"].dropna().shape[0] > 0


# --- логика сигналов ---
def test_entry_long_spread_at_low_z():
    eng = SignalEngine(StrategyConfig())
    rec = eng.evaluate(_row(z=-2.5), position=None)
    assert rec.action == Action.ENTER
    assert rec.direction == SpreadDirection.LONG_SPREAD


def test_no_entry_beyond_stop():
    eng = SignalEngine(StrategyConfig())  # stop_z=3.5
    rec = eng.evaluate(_row(z=-4.0), position=None)
    assert rec.action == Action.NONE


def test_width_filter_blocks_entry():
    eng = SignalEngine(StrategyConfig())  # min_width_pct=2
    rec = eng.evaluate(_row(z=-2.5, width=1.0), position=None)
    assert rec.action == Action.NONE


def test_short_spread_at_high_z():
    eng = SignalEngine(StrategyConfig())
    rec = eng.evaluate(_row(z=2.5), position=None)
    assert rec.action == Action.ENTER
    assert rec.direction == SpreadDirection.SHORT_SPREAD


# --- виртуальная биржа ---
def test_long_spread_pnl_sign():
    ve = VirtualExchange(AppConfig().paper)
    ve.open_pair(SpreadDirection.LONG_SPREAD, notional=1000, price_a=100, price_b=50,
                 beta=1.0, ts=0, entry_z=-2.0)
    # A растёт, B без изменений → лонг A в плюсе
    up = ve.unrealized(price_a=110, price_b=50)
    assert up > 0
    trade = ve.close_pair(price_a=110, price_b=50, ts=1, reason="exit", exit_z=0.0)
    assert trade.gross_pnl > 0
    assert ve.position is None


def test_fees_charged():
    ve = VirtualExchange(AppConfig().paper)
    ve.open_pair(SpreadDirection.LONG_SPREAD, notional=1000, price_a=100, price_b=50,
                 beta=1.0, ts=0, entry_z=-2.0)
    trade = ve.close_pair(price_a=100, price_b=50, ts=1, reason="exit", exit_z=0.0)
    assert trade.fees > 0
    assert trade.net_pnl < trade.gross_pnl  # комиссии уменьшают net


# --- human-in-the-loop ---
def test_entry_requires_approval():
    cfg = AppConfig()
    cfg.auto_approve = False
    eng = Engine(cfg)
    res = eng.step(_row(z=-2.5))
    assert res.awaiting_approval is True
    assert eng.exch.position is None        # без подтверждения позиции нет
    assert eng.pending is not None
    eng.approve()
    assert eng.exch.position is not None     # после approve — открыта


def test_reject_skips_entry():
    cfg = AppConfig()
    cfg.auto_approve = False
    eng = Engine(cfg)
    eng.step(_row(z=-2.5))
    eng.reject()
    assert eng.exch.position is None
    assert eng.pending is None


def test_auto_exit_on_revert():
    import math
    cfg = AppConfig()
    cfg.auto_approve = True
    eng = Engine(cfg)
    # Вход LONG_spread: спред внизу (z=-2.5). Цены подобраны так, чтобы spread по ценам
    # ln(pa)-ln(pb) согласовался со средней mid=ln(100/50)=0.693 (β=1).
    mid = math.log(100.0 / 50.0)
    pa_in, pb_in = 90.0, 50.0          # spread_in = ln(90/50)=0.588 < mid (внизу)
    eng.step(_row(z=-2.5, ts=0, mid=mid, beta=1.0, pa=pa_in, pb=pb_in))
    assert eng.exch.position is not None
    # выход по возврату к СРЕДНЕЙ: спред поднимается до mid (A дорожает до 100)
    res = eng.step(_row(z=-0.1, ts=300_000, mid=mid, beta=1.0, pa=100.0, pb=50.0))
    assert res.trade is not None
    assert eng.exch.position is None
    assert res.trade.reason == "exit"
    assert res.trade.gross_pnl > 0          # дошёл до средней → валовый P&L положителен


def test_summary_keys():
    cfg = AppConfig()
    cfg.auto_approve = True
    eng = Engine(cfg)
    for row in Engine.rows_from_df(generate_synthetic(n=1500), cfg.strategy):
        eng.step(row)
    s = eng.summary()
    for k in ("trades", "win_rate_pct", "net_pnl", "fees_paid", "balance", "equity", "return_pct"):
        assert k in s
