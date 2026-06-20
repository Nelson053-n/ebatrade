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
    """Базовые фикс-поля snapshot присутствуют (для любого режима)."""
    s = St5Session("sber")
    snap = s.snapshot(0.0)
    for key in ("lookback", "holding", "stop_pct", "cur_signal", "cur_z",
                "wait_reason", "summary", "history", "events", "strategy_name",
                "strategy_mode", "mr_ma_n", "mr_entry_z"):
        assert key in snap
    assert snap["lookback"] == 48


def test_snapshot_has_connector_fields():
    """snapshot отдаёт поля коннектора (без токена — только token_set bool)."""
    s = St5Session("sber")
    snap = s.snapshot(0.0)
    for key in ("connector_mode", "sandbox_active", "sandbox_error", "token_set"):
        assert key in snap
    assert snap["connector_mode"] == "paper"          # дефолт
    assert snap["sandbox_active"] is False
    assert "token" not in snap                          # сам секрет не утекает
    assert isinstance(snap["token_set"], bool)


# ===================== Phase 2: T-Bank sandbox executor =====================

class FakeSB:
    """Мок модуля tbank_sandbox для юнит-тестов St5SandboxExecutor (без сети).

    fail_from: с какого по счёту вызова post_order начинать реджектить (None = все fill).
    futures: список позиций фьючерсов на счёте (для broker_lots/reconciliation).
    """

    def __init__(self, fail_from=None, accounts=None, futures=None, tradable=True):
        self.orders: list[tuple] = []
        self.fail_from = fail_from
        self.payins: list[tuple] = []
        self.opened: list[str] = []
        self._accounts = accounts or []
        self._futures = futures or []
        self._tradable = tradable

    @staticmethod
    def _uid(it):
        return it["uid"]

    @staticmethod
    def _q_to_float(q):
        if not q:
            return 0.0
        return float(q.get("units", 0)) + float(q.get("nano", 0)) / 1e9

    def find_future(self, ticker):
        return {"ticker": ticker, "uid": "uid-" + ticker, "figi": "figi-" + ticker,
                "lot": 1, "apiTradeAvailableFlag": True}

    def is_tradable(self, uid):
        return self._tradable

    def list_accounts(self):
        return self._accounts

    def open_account(self, name="x"):
        self.opened.append(name)
        return "acc-new"

    def pay_in(self, acc, rub):
        self.payins.append((acc, rub))
        return float(rub)

    def positions(self, acc):
        return {"futures": self._futures}

    def post_order(self, acc, uid, lots, direction, order_id,
                   order_type="ORDER_TYPE_MARKET", price=None):
        import uuid as _uuid
        _uuid.UUID(order_id)            # упадёт, если order_id не валидный UUID
        self.orders.append((uid, direction, order_id))
        n = len(self.orders)
        if self.fail_from is not None and n >= self.fail_from:
            return {"executionReportStatus": "EXECUTION_REPORT_STATUS_REJECTED",
                    "executedOrderPrice": None}
        price = 32000
        # T-Bank: executedOrderPrice = СУММА за все лоты + lotsExecuted
        return {"executionReportStatus": "EXECUTION_REPORT_STATUS_FILL",
                "lotsExecuted": lots,
                "executedOrderPrice": {"units": str(price * lots), "nano": 0}}


def _st5_exec(fail_from=None, accounts=None, futures=None, tradable=True):
    from pairsignal.st5.tinkoff_executor import St5SandboxExecutor
    cfg = St5Config()
    spec = feed.synthetic_spec()
    spec.code = "SRU6"
    sb = FakeSB(fail_from=fail_from, accounts=accounts, futures=futures, tradable=tradable)
    ex = St5SandboxExecutor(cfg.execution, cfg.connector, spec, sb=sb)
    return ex, sb


def test_st5_executor_execute_ok():
    """execute() → market-ордер fill, счёт открыт + pay_in, orderId — валидный UUID."""
    ex, sb = _st5_exec()
    f = ex.execute("buy", lots=1, ref_price=32000)
    assert f.side == "buy" and f.lots == 1 and f.avg_price == 32000
    assert len(sb.orders) == 1
    assert sb.orders[0][1] == "ORDER_DIRECTION_BUY"
    assert sb.opened == ["st5-momentum-sandbox"]
    assert sb.payins and sb.payins[0][1] == 200_000


def test_st5_executor_price_per_contract_multi_lot():
    """При lots>1 цена входа = за КОНТРАКТ, не сумма за лоты."""
    ex, sb = _st5_exec()
    f = ex.execute("sell", lots=10, ref_price=32000)
    assert f.avg_price == 32000 and f.lots == 10     # не 320000


def test_st5_executor_execute_fail_raises():
    """Ордер входа не залился за все попытки → UnwindError."""
    from pairsignal.st5.tinkoff_executor import UnwindError
    ex, sb = _st5_exec(fail_from=1)
    with pytest.raises(UnwindError):
        ex.execute("buy", lots=1, ref_price=32000)


def test_st5_executor_close_reverse_order():
    """close() ставит ОБРАТНЫЙ ордер и возвращает цену выхода."""
    ex, sb = _st5_exec()
    pos = Position(state=BotState.LONG, side="buy", lots=1, entry_price=32000,
                   entry_ts=0, entry_vwap=32000, entry_fee_rub=0.0)
    f = ex.close(pos, ref_price=32050)
    assert f.side == "sell"                            # закрытие лонга = sell
    assert f.avg_price == 32000
    assert sb.orders[-1][1] == "ORDER_DIRECTION_SELL"


def test_st5_executor_is_tradable():
    """is_tradable проксирует флаг биржи (гейт неторгового времени)."""
    ex_ok, _ = _st5_exec(tradable=True)
    assert ex_ok.is_tradable() is True
    ex_no, _ = _st5_exec(tradable=False)
    assert ex_no.is_tradable() is False


def test_st5_executor_broker_lots_and_flat():
    """broker_lots читает баланс инструмента на счёте; flat_broker закрывает его."""
    futs = [{"instrumentUid": "uid-SRU6", "balance": 3}]
    ex, sb = _st5_exec(futures=futs)
    assert ex.broker_lots() == 3
    n = len(sb.orders)
    assert ex.flat_broker() is True
    assert sb.orders[-1][1] == "ORDER_DIRECTION_SELL"  # закрытие лонга +3 → sell
    assert len(sb.orders) == n + 1


def test_st5_executor_account_reuse():
    """Существующий OPEN-счёт с нужным именем переиспользуется — open_account не зван."""
    accs = [{"id": "acc-existing", "name": "st5-momentum-sandbox",
             "status": "ACCOUNT_STATUS_OPEN"}]
    ex, sb = _st5_exec(accounts=accs)
    assert ex._account_id == "acc-existing"
    assert sb.opened == []


def test_engine_paper_executor_by_default():
    """mode='paper' (дефолт) → engine.executor is None (встроенный paper-филл)."""
    eng = TradingEngine(St5Config(), _spec())
    assert eng.executor is None


def test_engine_uses_sandbox_when_mode(monkeypatch):
    """mode='tbank_sandbox' → engine.executor = St5SandboxExecutor (с моком sb)."""
    import pairsignal.st5.tinkoff_executor as te
    monkeypatch.setattr(te, "tbank_sandbox", FakeSB())
    cfg = St5Config()
    cfg.connector.mode = "tbank_sandbox"
    spec = _spec()
    spec.code = "SRU6"
    eng = TradingEngine(cfg, spec)
    assert type(eng.executor).__name__ == "St5SandboxExecutor"


# ============================ mean-reversion (z-score) ============================

from pairsignal.st5.indicators import ZScoreIndicator
from pairsignal.st5.strategy import (
    entry_signal_mr, exit_signal_mr, in_session_window,
)


def _mr_cfg(ma_n=5, entry_z=2.0, exit_z=0.5, stop_z=4.0, max_hold=10):
    cfg = St5Config()
    cfg.strategy.strategy_mode = "meanrev"
    cfg.strategy.mr_ma_n = ma_n
    cfg.strategy.mr_entry_z = entry_z
    cfg.strategy.mr_exit_z = exit_z
    cfg.strategy.mr_stop_z = stop_z
    cfg.strategy.mr_max_hold = max_hold
    cfg.strategy.flat_at_session_end = False
    # широкое окно сессии, чтобы тестовые бары (10:00+) попадали
    cfg.strategy.session_lo_min = 0
    cfg.strategy.session_hi_min = 1440
    return cfg


def test_zscore_not_ready_below_ma_n():
    """is_ready только когда накоплено ≥ ma_n баров и std>0."""
    z = ZScoreIndicator(ma_n=4)
    for c in (100, 101, 102):
        assert not z.update(c).is_ready
    assert z.update(103).is_ready          # 4-й бар → готов


def test_zscore_flat_not_ready():
    """Плоский участок (std=0) → не готов (вырожденный z)."""
    z = ZScoreIndicator(ma_n=3)
    z.update(100); z.update(100)
    assert not z.update(100).is_ready


def test_zscore_no_repaint_matches_pandas():
    """z боевого индикатора == pandas rolling(ma_n).std(ddof=0) на тех же закрытых барах."""
    import numpy as np
    import pandas as pd
    closes = [100, 102, 98, 105, 95, 110, 90, 103, 99, 101, 104, 97]
    ma_n = 5
    s = pd.Series(closes, dtype=float)
    ma = s.rolling(ma_n, min_periods=ma_n).mean()
    sd = s.rolling(ma_n, min_periods=ma_n).std(ddof=0)
    zi = ZScoreIndicator(ma_n=ma_n)
    for i, c in enumerate(closes):
        r = zi.update(float(c))
        if i + 1 < ma_n or sd.iloc[i] == 0:
            continue
        expected = (closes[i] - ma.iloc[i]) / sd.iloc[i]
        assert r.is_ready
        assert r.z == pytest.approx(expected, abs=1e-9)


def _zr(z, ready=True):
    from pairsignal.st5.models import ZScoreReading
    return ZScoreReading(ts=0, price=100.0, sma=100.0, std=1.0, z=z, is_ready=ready)


def test_mr_entry_signal_by_z():
    cfg = _mr_cfg(entry_z=2.5, stop_z=4.0).strategy
    assert entry_signal_mr(_zr(-3.0), cfg) == Signal.BUY     # ниже −entry_z
    assert entry_signal_mr(_zr(3.0), cfg) == Signal.SELL     # выше +entry_z
    assert entry_signal_mr(_zr(-1.0), cfg) == Signal.NONE    # внутри полосы
    assert entry_signal_mr(_zr(-5.0), cfg) == Signal.NONE    # за стопом → не входим
    assert entry_signal_mr(_zr(-3.0, ready=False), cfg) == Signal.NONE


def test_mr_exit_tp_stop_time():
    cfg = _mr_cfg(exit_z=0.5, stop_z=4.0, max_hold=10).strategy
    long_pos = _pos(BotState.LONG, 100.0)
    # TP: z вернулся к −exit_z (≥ −0.5)
    assert exit_signal_mr(long_pos, -0.4, 1, cfg) == (True, "take")
    # стоп: z ушёл глубже −stop_z
    assert exit_signal_mr(long_pos, -4.5, 1, cfg) == (True, "stop")
    # держим: z всё ещё в зоне
    assert exit_signal_mr(long_pos, -2.0, 1, cfg) == (False, "")
    # время
    assert exit_signal_mr(long_pos, -2.0, 10, cfg) == (True, "time_stop")
    short_pos = _pos(BotState.SHORT, 100.0)
    assert exit_signal_mr(short_pos, 0.3, 1, cfg) == (True, "take")
    assert exit_signal_mr(short_pos, 4.2, 1, cfg) == (True, "stop")


def test_in_session_window():
    cfg = St5Config()
    cfg.strategy.session_lo_min = 600    # 10:00
    cfg.strategy.session_hi_min = 1125   # 18:45
    s, ses = cfg.strategy, cfg.session
    assert in_session_window(_ts("2026-06-10", 12, 0), s, ses)
    assert not in_session_window(_ts("2026-06-10", 9, 30), s, ses)   # до открытия
    assert not in_session_window(_ts("2026-06-10", 19, 0), s, ses)   # вечерняя — вне окна


def _feed_mr(eng, ts0, closes, iv=600_000):
    last = None
    for i, c in enumerate(closes):
        last = eng.step(_bar(ts0 + i * iv, c))
    return last


def test_mr_engine_long_entry_on_low_z_and_tp_exit():
    """meanrev: глубокое отрицательное z → LONG; возврат к среднему → выход TP."""
    eng = TradingEngine(_mr_cfg(ma_n=5, entry_z=1.5, exit_z=0.5, stop_z=4.0, max_hold=50), _spec())
    ts0 = _ts("2026-06-10", 11, 0)
    # 5 баров около 100 (прогрев), затем резкий провал → z << 0 → LONG
    _feed_mr(eng, ts0, [100, 100, 101, 99, 100])
    eng.step(_bar(ts0 + 5 * 600_000, 88))     # сильное отклонение вниз → LONG
    assert eng.state == BotState.LONG
    assert eng.position.side == "buy"
    # возврат к среднему → z к 0 → выход TP
    for i in range(6, 30):
        eng.step(_bar(ts0 + i * 600_000, 100))
        if eng.state == BotState.FLAT:
            break
    assert len(eng.trades) == 1
    assert eng.trades[0].reason in ("take", "time_stop")


def test_mr_engine_short_entry_on_high_z():
    eng = TradingEngine(_mr_cfg(ma_n=5, entry_z=1.5), _spec())
    ts0 = _ts("2026-06-10", 11, 0)
    _feed_mr(eng, ts0, [100, 100, 101, 99, 100])
    eng.step(_bar(ts0 + 5 * 600_000, 115))    # сильное отклонение вверх → SHORT
    assert eng.state == BotState.SHORT
    assert eng.position.side == "sell"


def test_mr_session_window_blocks_entry():
    """Сигнал вне окна основной сессии → вход не открывается."""
    cfg = _mr_cfg(ma_n=5, entry_z=1.5)
    cfg.strategy.session_lo_min = 600     # 10:00
    cfg.strategy.session_hi_min = 1125    # 18:45
    eng = TradingEngine(cfg, _spec())
    ts0 = _ts("2026-06-10", 20, 0)        # 20:00 — вечерняя, вне окна
    _feed_mr(eng, ts0, [100, 100, 101, 99, 100])
    res = eng.step(_bar(ts0 + 5 * 600_000, 88))
    assert eng.state == BotState.FLAT
    assert any("окна основной сессии" in e.message for e in res.events)


def test_mr_no_lookahead_first_signal_only_after_warmup():
    """Сигнал не появляется раньше, чем накоплено ma_n баров (no look-ahead)."""
    eng = TradingEngine(_mr_cfg(ma_n=8, entry_z=0.5), _spec())
    ts0 = _ts("2026-06-10", 11, 0)
    # первые 7 баров: индикатор не готов → позиция не открывается даже при отклонении
    for i, c in enumerate([100, 130, 70, 120, 80, 110, 90]):
        eng.step(_bar(ts0 + i * 600_000, c))
        assert eng.state == BotState.FLAT


def test_mr_snapshot_fields():
    """snapshot в meanrev отдаёт cur_z и strategy_name='meanrev', прежние поля на месте."""
    s = St5Session("sber")           # дефолт meanrev
    snap = s.snapshot(0.0)
    assert snap["strategy_mode"] == "meanrev"
    assert snap["strategy_name"] == "meanrev"
    assert "cur_z" in snap
    assert snap["mr_ma_n"] == 36 and snap["mr_entry_z"] == 2.5
    # прежний контракт фронта /st5/state не сломан
    for key in ("fsm_state", "position", "pending", "summary", "history",
                "trades", "wait_reason", "cur_signal", "leg"):
        assert key in snap


def test_mr_session_defaults_meanrev_and_backtest_runs():
    """SBER-сессия по умолчанию meanrev; backtest боевого кода в meanrev исполняет сделки."""
    from pairsignal.st5.backtest import run_backtest, vwap_frame_for_chart
    s = St5Session("sber")
    assert s.cfg.strategy.strategy_mode == "meanrev"
    cfg = _mr_cfg(ma_n=12, entry_z=1.0, exit_z=0.5, stop_z=4.0, max_hold=20)
    df = feed.generate_synthetic(n=800)
    r = run_backtest(df, cfg, feed.synthetic_spec())
    assert r["bars"] == 800
    assert r["trades"] >= 1
    frame = vwap_frame_for_chart(df, cfg)
    assert frame and "z" in frame[0]
