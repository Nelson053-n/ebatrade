"""Юнит-тесты st5 (directional momentum на одиночном инструменте).

Покрывают: индикатор momentum (готовность по lookback, направление сигнала),
объёмный SMA, сигналы вход (по тренду) / выход (holding, стоп), знак P&L, FSM-ветки
движка (вход LONG/SHORT, выход по holding/стопу, flat-at-session-end, дневной
kill-switch), объёмный фильтр, бэктест, обратная совместимость со старым VWAP-конфигом.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from pairsignal.st5 import data_feed as feed
from pairsignal.st5.backtest import run_backtest, vwap_frame_for_chart
from pairsignal.st5.config import St5Config
from pairsignal.st5.engine import TradingEngine, _pnl_rub
from pairsignal.st5.indicators import MomentumIndicator, VolumeAverage
from pairsignal.st5.models import BotState, MomentumReading, Position, PriceBar, Signal
from pairsignal.st5.service import St5Session
from pairsignal.st5.strategy import entry_signal, exit_signal, is_session_end

_MSK = timezone(timedelta(hours=3))


def _spec():
    return feed.synthetic_spec()


def _ts(day: str, hh: int, mm: int) -> int:
    """unix ms для MSK-времени конкретного дня."""
    dt = datetime.fromisoformat(f"{day}T{hh:02d}:{mm:02d}:00").replace(tzinfo=_MSK)
    return int(dt.timestamp() * 1000)


def _bar(ts: int, close: float, vol: float = 100.0, hl: float = 5.0) -> PriceBar:
    return PriceBar(ts, close, close + hl, close - hl, close, vol)


def _mr(price, ref, ready=True) -> MomentumReading:
    sig = 1 if price > ref else (-1 if price < ref else 0)
    ret = (price - ref) / ref if ref else 0.0
    return MomentumReading(ts=0, price=price, ref_price=ref, signal=sig,
                           lookback_return=ret, is_ready=ready)


def _pos(state: BotState, entry: float) -> Position:
    side = "buy" if state == BotState.LONG else "sell"
    return Position(state=state, side=side, lots=1, entry_price=entry,
                    entry_ts=0, entry_vwap=entry, entry_fee_rub=0.0)


# ============================ индикатор momentum ============================

def test_momentum_not_ready_below_lookback():
    """is_ready только когда накоплено > lookback закрытых баров."""
    mom = MomentumIndicator(lookback=3)
    for c in (100, 101, 102):           # 3 бара — пока нет close[-3]
        assert not mom.update(c).is_ready
    assert mom.update(103).is_ready      # 4-й бар: есть с чем сравнивать


def test_momentum_direction_up():
    """close > close[-lookback] → signal +1, корректный lookback_return."""
    mom = MomentumIndicator(lookback=2)
    mom.update(100)
    mom.update(100)
    r = mom.update(110)                  # сравнение с close[-2] = 100
    assert r.is_ready
    assert r.signal == 1
    assert r.ref_price == pytest.approx(100)
    assert r.lookback_return == pytest.approx(0.10)


def test_momentum_direction_down_and_flat():
    mom = MomentumIndicator(lookback=2)
    mom.update(100)
    mom.update(100)
    assert mom.update(90).signal == -1   # вниз
    mom2 = MomentumIndicator(lookback=1)
    mom2.update(100)
    assert mom2.update(100).signal == 0  # равно → флэт


def test_volume_average_resets_daily():
    va = VolumeAverage()
    va.update(_ts("2026-06-10", 10, 0), 100)
    va.update(_ts("2026-06-10", 10, 10), 200)
    assert va.update(_ts("2026-06-10", 10, 20), 300) == pytest.approx(200)  # (100+200+300)/3
    assert va.update(_ts("2026-06-11", 10, 0), 50) == pytest.approx(50)     # новый день


# ============================ сигналы ============================

def test_entry_follows_trend():
    # тренд вверх → BUY (вход в направлении)
    assert entry_signal(_mr(110, 100)) == Signal.BUY
    # тренд вниз → SELL
    assert entry_signal(_mr(90, 100)) == Signal.SELL
    # флэт → нет сигнала
    assert entry_signal(_mr(100, 100)) == Signal.NONE
    # индикатор не готов → нет сигнала
    assert entry_signal(_mr(110, 100, ready=False)) == Signal.NONE


def test_exit_by_holding():
    pos = _pos(BotState.LONG, 100.0)
    # держим < holding → не выходим
    assert exit_signal(pos, 5, 101.0, holding=18, stop_pct=0.02) == (False, "")
    # bars_held >= holding → выход по времени
    ok, reason = exit_signal(pos, 18, 101.0, holding=18, stop_pct=0.02)
    assert ok and reason == "time_stop"


def test_exit_by_stop():
    # LONG: цена упала на >2% от входа → ранний стоп
    pos = _pos(BotState.LONG, 100.0)
    ok, reason = exit_signal(pos, 1, 97.0, holding=18, stop_pct=0.02)
    assert ok and reason == "stop"
    # SHORT: цена выросла на >2% → стоп
    sp = _pos(BotState.SHORT, 100.0)
    ok, reason = exit_signal(sp, 1, 103.0, holding=18, stop_pct=0.02)
    assert ok and reason == "stop"
    # цена в пределах stop_pct и держим < holding → не выходим
    assert exit_signal(pos, 1, 99.0, holding=18, stop_pct=0.02) == (False, "")


def test_session_end_predicate():
    cfg = St5Config().session
    assert is_session_end(_ts("2026-06-10", 23, 45), cfg)       # после 23:40
    assert not is_session_end(_ts("2026-06-10", 15, 0), cfg)


# ============================ P&L ============================

def test_pnl_sign():
    spec = _spec()
    assert _pnl_rub("buy", 100, 110, 1, spec) == pytest.approx(10.0)   # лонг + рост
    assert _pnl_rub("sell", 100, 90, 1, spec) == pytest.approx(10.0)   # шорт + падение
    assert _pnl_rub("sell", 100, 110, 1, spec) == pytest.approx(-10.0)  # шорт + рост


# ============================ движок (FSM) ============================

def _cfg(lookback=3, holding=4, stop_pct=0.02, flat=False):
    cfg = St5Config()
    cfg.strategy.lookback = lookback
    cfg.strategy.holding = holding
    cfg.strategy.stop_pct = stop_pct
    cfg.strategy.flat_at_session_end = flat
    return cfg


def _feed(eng, day, prices, hh0=10):
    """Прогнать список close по барам дня; вернуть (последний ts, шаг)."""
    ts0 = _ts(day, hh0, 0)
    iv = eng.cfg.strategy.candle_interval_minutes * 60_000
    last = None
    for i, c in enumerate(prices):
        last = eng.step(_bar(ts0 + i * iv, c))
    return ts0 + (len(prices) - 1) * iv, iv, last


def test_long_entry_on_uptrend_and_profit():
    """Рост close[i] > close[i-lookback] → вход LONG; рост дальше → P&L > 0."""
    eng = TradingEngine(_cfg(lookback=3, holding=10), _spec())
    # 4 бара прогрева (lookback=3 → готов на 4-м), затем растущий тренд → BUY
    eng.step(_bar(_ts("2026-06-10", 10, 0), 32000))
    eng.step(_bar(_ts("2026-06-10", 10, 10), 32000))
    eng.step(_bar(_ts("2026-06-10", 10, 20), 32000))
    eng.step(_bar(_ts("2026-06-10", 10, 30), 32010))   # 32010 > close[-3]=32000 → LONG
    assert eng.state == BotState.LONG
    assert eng.position.side == "buy"
    entry = eng.position.entry_price
    # цена растёт → закрытие через holding в плюс
    ts = _ts("2026-06-10", 10, 40)
    for i in range(12):
        eng.step(_bar(ts + i * 600_000, 32050 + i * 5))
        if eng.state == BotState.FLAT:
            break
    assert len(eng.trades) == 1
    t = eng.trades[0]
    assert t.exit_price > entry
    assert t.gross_pnl_rub > 0


def test_short_entry_on_downtrend():
    """Падение close[i] < close[i-lookback] → вход SHORT."""
    eng = TradingEngine(_cfg(lookback=3, holding=10), _spec())
    _feed(eng, "2026-06-10", [32000, 32000, 32000, 31990])   # 31990 < close[-3] → SHORT
    assert eng.state == BotState.SHORT
    assert eng.position.side == "sell"


def test_exit_by_holding_closes_position():
    """Позиция держится ровно holding баров, затем выход по времени."""
    eng = TradingEngine(_cfg(lookback=2, holding=3, stop_pct=0.0), _spec())
    _feed(eng, "2026-06-10", [32000, 32000, 32010])   # вход LONG на 3-м баре
    assert eng.state == BotState.LONG
    ts = _ts("2026-06-10", 11, 0)
    iv = 600_000
    # держим: bars_held инкрементится каждый шаг; на 3-м шаге в позиции — выход
    for i in range(5):
        eng.step(_bar(ts + i * iv, 32010))
        if eng.state == BotState.FLAT:
            break
    assert len(eng.trades) == 1
    assert eng.trades[0].reason == "time_stop"
    assert eng.trades[0].bars_held == 3


def test_exit_by_stop_closes_position():
    """Цена против LONG более чем на stop_pct → ранний выход по стопу."""
    eng = TradingEngine(_cfg(lookback=2, holding=100, stop_pct=0.01), _spec())
    _feed(eng, "2026-06-10", [32000, 32000, 32010])   # вход LONG
    assert eng.state == BotState.LONG
    entry = eng.position.entry_price
    # обвал на >1% от входа → стоп
    eng.step(_bar(_ts("2026-06-10", 12, 0), entry * 0.98))
    assert eng.state == BotState.FLAT
    assert eng.trades[-1].reason == "stop"
    assert eng.trades[-1].gross_pnl_rub < 0


def test_session_end_forces_flat():
    eng = TradingEngine(_cfg(lookback=2, holding=100, flat=True), _spec())
    _feed(eng, "2026-06-10", [32000, 32000, 32010])   # вход LONG
    assert eng.state == BotState.LONG
    eng.step(_bar(_ts("2026-06-10", 23, 45), 32010))  # конец сессии
    assert eng.state == BotState.FLAT
    assert eng.trades[-1].reason == "eod"


def test_day_loss_kill_switch():
    cfg = _cfg(lookback=2, holding=100, stop_pct=0.0)
    cfg.risk.max_daily_loss_rub = 50.0
    eng = TradingEngine(cfg, _spec())
    _feed(eng, "2026-06-10", [32000, 32000, 32010])   # вход LONG
    assert eng.state == BotState.LONG
    # цена резко против лонга → unrealized < −50 → HALTED
    eng.step(_bar(_ts("2026-06-10", 12, 0), 32010 - 400))
    assert eng.state == BotState.HALTED
    assert eng.risk.halted


def test_volume_filter_blocks_low():
    cfg = _cfg(lookback=2, holding=10)
    cfg.strategy.volume_filter_mult = 2.0
    eng = TradingEngine(cfg, _spec())
    eng.step(_bar(_ts("2026-06-10", 10, 0), 32000, vol=100))
    eng.step(_bar(_ts("2026-06-10", 10, 10), 32000, vol=100))
    # сигнал на тренде, но НИЗКИЙ объём (10 << 2·~100) → вход заблокирован
    res = eng.step(_bar(_ts("2026-06-10", 10, 20), 32010, vol=10))
    assert eng.state == BotState.FLAT
    assert any("объём" in e.message for e in res.events)


# ============================ бэктест ============================

def test_backtest_synthetic_runs():
    cfg = St5Config()
    cfg.strategy.lookback = 12          # синтетика ~800 баров — короче lookback
    cfg.strategy.holding = 6
    df = feed.generate_synthetic(n=800)
    r = run_backtest(df, cfg, feed.synthetic_spec())
    assert r["bars"] == 800
    assert r["trades"] >= 1
    # net согласован с суммой сделок
    assert r["net_pnl_rub"] == pytest.approx(
        sum(t["net_pnl_rub"] for t in r["trades_detail"]), abs=1)
    assert len(vwap_frame_for_chart(df, cfg)) > 0


# ============================ совместимость ============================

def test_load_session_ignores_old_vwap_config(tmp_path, monkeypatch):
    """Старый session_state_5_*.json с VWAP-параметрами не должен ронять load."""
    s = St5Session("sber")
    old = {
        "session_started": 1_700_000_000.0,
        "balance_rub": 999_000.0,
        "trades": [],
        "history": [],
        # старый VWAP-конфиг + вовсе незнакомый ключ
        "config": {"strategy": {"band_sigma": 1.5, "take_profit_sigma": 0.5,
                                 "entry_trigger": "ReEntry", "totally_unknown": 42}},
        "spec": {"code": "SRU6", "tick_size": 1.0, "tick_value_rub": 1.0,
                 "lot": 1, "expiry": None},
        "position": None, "halted": False,
    }
    s._session_file.write_text(json.dumps(old))
    assert s.load_session() is True
    assert s.engine.balance_rub == pytest.approx(999_000.0)
    # momentum-дефолты на месте
    assert s.cfg.strategy.lookback == 48
    assert s.cfg.strategy.holding == 18
    s._session_file.unlink(missing_ok=True)


def test_snapshot_has_momentum_fields():
    s = St5Session("sber")
    snap = s.snapshot(0.0)
    for key in ("lookback", "holding", "stop_pct", "cur_signal",
                "wait_reason", "summary", "history", "events", "strategy_name"):
        assert key in snap
    assert snap["lookback"] == 48
    assert "Momentum" in snap["strategy_name"]
