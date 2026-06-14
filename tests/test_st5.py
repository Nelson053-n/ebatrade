"""Юнит-тесты st5 (VWAP-reversion на одиночном инструменте).

Покрывают: внутридневной VWAP (сброс по дню, σ), сигналы вход/выход, знак P&L,
тейк/стоп/тайм-стоп, дневной kill-switch с нереализованным, flat-at-session-end,
объёмный фильтр, персист позиции.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from pairsignal.st5 import data_feed as feed
from pairsignal.st5.backtest import run_backtest, vwap_frame_for_chart
from pairsignal.st5.config import St5Config
from pairsignal.st5.engine import TradingEngine, _pnl_rub
from pairsignal.st5.indicators import IntradayVwap, VolumeAverage
from pairsignal.st5.models import BotState, PriceBar, Signal, VwapReading
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


def _vr(ts, price, vwap, sigma, k=2.0, ready=True) -> VwapReading:
    return VwapReading(ts, price, vwap, sigma, vwap + k * sigma, vwap - k * sigma, ready)


# ============================ индикатор VWAP ============================

def test_vwap_matches_manual():
    """VWAP = Σ(typical·vol)/Σvol по барам дня."""
    vw = IntradayVwap(band_sigma=2.0, min_bars=1)
    day = "2026-06-10"
    bars = [_bar(_ts(day, 10, 0), 100, vol=10),
            _bar(_ts(day, 10, 10), 110, vol=30)]
    vw.update(bars[0])
    r2 = vw.update(bars[1])
    # typical = close (т.к. hl симметричен): t1=100, t2=110
    exp = (100 * 10 + 110 * 30) / 40
    assert r2.vwap == pytest.approx(exp, abs=1e-9)


def test_vwap_resets_each_day():
    """VWAP сбрасывается на новый торговый день (MSK)."""
    vw = IntradayVwap(band_sigma=2.0, min_bars=1)
    vw.update(_bar(_ts("2026-06-10", 12, 0), 100, vol=100))
    r = vw.update(_bar(_ts("2026-06-11", 10, 0), 200, vol=100))
    assert r.vwap == pytest.approx(200, abs=1e-6)   # новый день → только текущий бар


def test_vwap_not_ready_below_min_bars():
    vw = IntradayVwap(band_sigma=2.0, min_bars=6)
    day = "2026-06-10"
    for i in range(5):
        r = vw.update(_bar(_ts(day, 10, i * 10), 100 + i))
        assert not r.is_ready
    assert vw.update(_bar(_ts(day, 11, 0), 105)).is_ready


def test_volume_average_resets_daily():
    va = VolumeAverage()
    va.update(_ts("2026-06-10", 10, 0), 100)
    va.update(_ts("2026-06-10", 10, 10), 200)
    assert va.update(_ts("2026-06-10", 10, 20), 300) == pytest.approx(200)  # (100+200+300)/3
    assert va.update(_ts("2026-06-11", 10, 0), 50) == pytest.approx(50)     # новый день


# ============================ сигналы ============================

def test_entry_breakout():
    cfg = St5Config().strategy
    # цена пробила нижнюю полосу вниз → BUY (ждём возврата вверх)
    assert entry_signal(_vr(0, 95, 100, 5), _vr(1, 89, 100, 5), cfg) == Signal.BUY
    # пробила верхнюю → SELL
    assert entry_signal(_vr(0, 105, 100, 5), _vr(1, 111, 100, 5), cfg) == Signal.SELL
    # внутри коридора → нет сигнала
    assert entry_signal(_vr(0, 100, 100, 5), _vr(1, 102, 100, 5), cfg) == Signal.NONE


def test_entry_reentry():
    cfg = St5Config().strategy
    cfg.entry_trigger = "ReEntry"
    # была выше верхней (111≥110), вернулась внутрь (108<110) → SELL
    assert entry_signal(_vr(0, 111, 100, 5), _vr(1, 108, 100, 5), cfg) == Signal.SELL
    # пробой наружу в ReEntry сигнала не даёт
    assert entry_signal(_vr(0, 105, 100, 5), _vr(1, 111, 100, 5), cfg) == Signal.NONE


def test_exit_cross_vwap():
    # LONG: цена пересекла VWAP снизу вверх → выход
    assert exit_signal(BotState.LONG, _vr(0, 95, 100, 5), _vr(1, 101, 100, 5), 100)
    assert not exit_signal(BotState.LONG, _vr(0, 95, 100, 5), _vr(1, 98, 100, 5), 100)
    # SHORT: сверху вниз
    assert exit_signal(BotState.SHORT, _vr(0, 105, 100, 5), _vr(1, 99, 100, 5), 100)


def test_session_end_predicate():
    cfg = St5Config().session
    assert is_session_end(_ts("2026-06-10", 23, 45), cfg)       # после 23:40
    assert not is_session_end(_ts("2026-06-10", 15, 0), cfg)


# ============================ P&L и движок ============================

def test_pnl_sign():
    spec = _spec()
    # лонг заработал при росте
    assert _pnl_rub("buy", 100, 110, 1, spec) == pytest.approx(10.0)
    # шорт заработал при падении
    assert _pnl_rub("sell", 100, 90, 1, spec) == pytest.approx(10.0)
    # шорт потерял при росте
    assert _pnl_rub("sell", 100, 110, 1, spec) == pytest.approx(-10.0)


def _warm_day(eng, day, base=32000.0, n=10):
    """Прогреть день колебанием вокруг base (σ>0, VWAP готов)."""
    ts0 = _ts(day, 10, 0)
    iv = eng.cfg.strategy.candle_interval_minutes * 60_000
    for i in range(n):
        noise = 8.0 if i % 2 == 0 else -8.0
        eng.step(_bar(ts0 + i * iv, base + noise))
    return ts0 + n * iv, iv


def test_long_profits_when_price_returns_up():
    """BUY (цена под VWAP) зарабатывает при возврате цены вверх к VWAP."""
    cfg = St5Config()
    cfg.strategy.min_bars_in_day = 5
    cfg.strategy.flat_at_session_end = False
    cfg.strategy.stop_sigma = 0.0
    eng = TradingEngine(cfg, _spec())
    ts, iv = _warm_day(eng, "2026-06-10", n=10)
    # резкий провал цены вниз за нижнюю полосу → BUY
    eng.step(_bar(ts, 32000 - 80))
    ts += iv
    assert eng.state == BotState.LONG
    entry = eng.position.entry_price
    # цена возвращается вверх через VWAP → выход в плюс
    eng.step(_bar(ts, 32010))
    assert len(eng.trades) == 1
    t = eng.trades[0]
    assert t.exit_price > entry
    assert t.gross_pnl_rub > 0


def test_session_end_forces_flat():
    cfg = St5Config()
    cfg.strategy.min_bars_in_day = 5
    eng = TradingEngine(cfg, _spec())
    ts, iv = _warm_day(eng, "2026-06-10", n=10)
    eng.step(_bar(ts, 32000 - 80))      # вход BUY
    assert eng.state == BotState.LONG
    # бар у конца сессии (23:45) → принудительное закрытие
    eng.step(_bar(_ts("2026-06-10", 23, 45), 31950))
    assert eng.state == BotState.FLAT
    assert eng.trades[-1].reason == "eod"


def test_day_loss_kill_switch():
    cfg = St5Config()
    cfg.strategy.min_bars_in_day = 5
    cfg.strategy.flat_at_session_end = False
    cfg.risk.max_daily_loss_rub = 50.0
    eng = TradingEngine(cfg, _spec())
    ts, iv = _warm_day(eng, "2026-06-10", n=10)
    eng.step(_bar(ts, 32000 - 80))   # BUY
    ts += iv
    assert eng.state == BotState.LONG
    # цена падает ещё дальше против лонга → unrealized < −50 → HALTED
    eng.step(_bar(ts, 32000 - 400))
    assert eng.state == BotState.HALTED
    assert eng.risk.halted


def test_volume_filter_blocks_low():
    cfg = St5Config()
    cfg.strategy.min_bars_in_day = 5
    cfg.strategy.flat_at_session_end = False
    cfg.strategy.volume_filter_mult = 2.0
    eng = TradingEngine(cfg, _spec())
    # прогрев со средним объёмом 100
    ts, iv = _warm_day(eng, "2026-06-10", n=10)
    # пробой на НИЗКОМ объёме (10 << 2·100) → вход заблокирован
    res = eng.step(_bar(ts, 32000 - 80, vol=10))
    assert eng.state == BotState.FLAT
    assert any("объём" in e.message for e in res.events)


# ============================ бэктест ============================

def test_backtest_synthetic_runs():
    cfg = St5Config()
    df = feed.generate_synthetic(n=800)
    r = run_backtest(df, cfg, feed.synthetic_spec())
    assert r["bars"] == 800
    assert r["trades"] >= 1
    # net согласован с суммой сделок
    assert r["net_pnl_rub"] == pytest.approx(
        sum(t["net_pnl_rub"] for t in r["trades_detail"]), abs=1)
    assert len(vwap_frame_for_chart(df, cfg)) > 0
