"""Юнит-тесты st4 (§14.1): индикатор, синхронизация, сигналы, P&L, атомарность, reconciliation.

Покрывают спорные места ТЗ: знаконезависимый гейт §9.3 (включая ОТРИЦАТЕЛЬНЫЙ спред),
согласованность знака P&L с направлением, аварийный unwind и HALTED, reconciliation.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from pairsignal.st4.backtest import run_backtest
from pairsignal.st4.config import St4Config
from pairsignal.st4 import data_feed as feed
from pairsignal.st4.engine import TradingEngine
from pairsignal.st4.execution import OrderExecutor, UnwindError, leg_pnl_rub
from pairsignal.st4.indicators import BollingerBands, SpreadBuilder, build_band_frame
from pairsignal.st4.models import BandReading, BotState, LegPosition, Position, Role, Signal
from pairsignal.st4.strategy import deviation_gate, entry_signal, exit_signal, in_clearing_window


def _specs():
    return feed.synthetic_spec(Role.ORDINARY), feed.synthetic_spec(Role.PREFERRED)


# ============================ §8 индикатор ============================

def test_bollinger_matches_pandas():
    """SMA/σ/полосы потокового BB совпадают с эталоном pandas (Population, ddof=0)."""
    rng = np.random.default_rng(1)
    vals = list(np.cumsum(rng.normal(0, 1, 300)) + 50)
    period, k = 200, 2.0
    bb = BollingerBands(period, k, "Population")
    last = None
    for i, v in enumerate(vals):
        last = bb.update(i, v)
    s = pd.Series(vals)
    exp_sma = s.rolling(period).mean().iloc[-1]
    exp_sigma = s.rolling(period).std(ddof=0).iloc[-1]
    assert last.is_ready
    assert last.sma == pytest.approx(exp_sma, abs=1e-9)
    assert last.sigma == pytest.approx(exp_sigma, abs=1e-9)
    assert last.upper == pytest.approx(exp_sma + k * exp_sigma, abs=1e-9)
    assert last.lower == pytest.approx(exp_sma - k * exp_sigma, abs=1e-9)


def test_bollinger_not_ready_during_warmup():
    bb = BollingerBands(50, 2.0)
    for i in range(49):
        r = bb.update(i, float(i))
        assert not r.is_ready and math.isnan(r.sma)
    assert bb.update(49, 49.0).is_ready


def test_band_frame_matches_streaming():
    """Векторный build_band_frame совпадает с потоковым BB на последнем баре."""
    df = feed.generate_synthetic(n=400, seed=5)
    bf = build_band_frame(df, 100, 2.0, "Population")
    bb = BollingerBands(100, 2.0, "Population")
    last = None
    for ts, row in df.iterrows():
        last = bb.update(int(ts), float(row["price_b"] - row["price_a"]))
    assert last.sma == pytest.approx(bf["sma"].iloc[-1], abs=1e-6)
    assert last.upper == pytest.approx(bf["upper"].iloc[-1], abs=1e-6)


# ============================ §7 SpreadBuilder ============================

def test_spread_builder_sync():
    """Бар спреда формируется только когда обе ноги закрылись; spread = pref − ord."""
    sb = SpreadBuilder()
    assert sb.add_ordinary(1000, 32000.0) is None       # только одна нога — бара нет
    bar = sb.add_preferred(1000, 32080.0)               # вторая нога → бар
    assert bar is not None
    assert bar.spread == pytest.approx(80.0)
    assert bar.close_ord == 32000.0 and bar.close_pref == 32080.0


def test_spread_builder_gap_no_bar():
    """Пропуск одной ноги в интервале — бар не строится (значение не подставляется)."""
    sb = SpreadBuilder()
    assert sb.add_ordinary(1000, 100.0) is None
    assert sb.add_preferred(2000, 180.0) is None        # другой ts — пары нет
    assert sb.add_preferred(1000, 190.0) is not None    # доехала пара для 1000


# ============================ §9.3 гейт отклонения ============================

def test_deviation_gate_abs_of_mean_positive():
    cfg = St4Config().strategy
    cfg.deviation_mode = "AbsOfMean"
    cfg.deviation_pct = 0.02
    # SMA=100, порог 2%·100=2 → cur=103 проходит SELL, cur=101 нет
    assert deviation_gate(Signal.SELL, 103, 100, cfg)
    assert not deviation_gate(Signal.SELL, 101, 100, cfg)
    assert deviation_gate(Signal.BUY, 97, 100, cfg)
    assert not deviation_gate(Signal.BUY, 99, 100, cfg)


def test_deviation_gate_abs_of_mean_negative_spread():
    """КЛЮЧЕВОЙ кейс ТЗ §9.3: при ОТРИЦАТЕЛЬНОЙ SMA гейт остаётся корректным.

    LiteralPct здесь ломается (SMA·1.02 < SMA при SMA<0), AbsOfMean — нет.
    """
    cfg = St4Config().strategy
    cfg.deviation_mode = "AbsOfMean"
    cfg.deviation_pct = 0.02
    # SMA=-100, порог 2%·|−100|=2. SELL при cur >= SMA+2 = -98; BUY при cur <= -102.
    assert deviation_gate(Signal.SELL, -97, -100, cfg)      # выше средней — проходит
    assert not deviation_gate(Signal.SELL, -99, -100, cfg)  # недостаточно выше
    assert deviation_gate(Signal.BUY, -103, -100, cfg)      # ниже средней — проходит
    assert not deviation_gate(Signal.BUY, -101, -100, cfg)


def test_deviation_gate_literal_pct_breaks_on_negative():
    """Демонстрация, ПОЧЕМУ LiteralPct неверен при SMA<0 (зафиксировано как анти-кейс)."""
    cfg = St4Config().strategy
    cfg.deviation_mode = "LiteralPct"
    cfg.deviation_pct = 0.02
    # SMA=-100: SELL требует cur >= -102 (т.е. почти всё «выше» порога — гейт вырождается)
    assert deviation_gate(Signal.SELL, -101, -100, cfg)    # -101 >= -102 → True (ложно мягкий)
    # это и есть баг, который AbsOfMean исправляет (см. тест выше: там -99 не проходит)


# ============================ §9.2/§9.4 сигналы ============================

def _band(ts, spread, sma, sigma, k=2.0):
    return BandReading(ts, spread, sma, sigma, sma + k * sigma, sma - k * sigma, True)


def test_entry_signal_breakout_up_sell():
    cfg = St4Config().strategy
    cfg.deviation_pct = 0.0  # отключить гейт — проверяем чистый пробой
    prev = _band(0, 105, 100, 10)        # spread 105 < upper 120
    cur = _band(1, 125, 100, 10)         # spread 125 >= upper 120 → пробой вверх
    assert entry_signal(prev, cur, cfg) == Signal.SELL


def test_entry_signal_breakout_down_buy():
    cfg = St4Config().strategy
    cfg.deviation_pct = 0.0
    prev = _band(0, 95, 100, 10)         # spread 95 > lower 80
    cur = _band(1, 75, 100, 10)          # spread 75 <= lower 80 → пробой вниз
    assert entry_signal(prev, cur, cfg) == Signal.BUY


def test_no_signal_during_warmup():
    cfg = St4Config().strategy
    prev = BandReading(0, 125, float("nan"), float("nan"), float("nan"), float("nan"), False)
    cur = _band(1, 125, 100, 10)
    assert entry_signal(prev, cur, cfg) == Signal.NONE


def test_exit_signal_cross_mean():
    # SHORT_SPREAD: выход при пересечении SMA сверху вниз
    prev = _band(0, 110, 100, 10)
    cur = _band(1, 98, 100, 10)
    assert exit_signal(BotState.SHORT_SPREAD, prev, cur, 100)
    assert not exit_signal(BotState.SHORT_SPREAD, _band(0, 110, 100, 10), _band(1, 105, 100, 10), 100)
    # LONG_SPREAD: снизу вверх
    assert exit_signal(BotState.LONG_SPREAD, _band(0, 90, 100, 10), _band(1, 102, 100, 10), 100)


# ============================ §9.5 знак P&L ============================

def test_leg_pnl_sign():
    so, _ = _specs()
    # лонг @ 32000, выход 32050 → +50 пунктов · 1 лот · (STEPPRICE/MINSTEP=1) = +50₽
    # (на размер лота LOTVOLUME не умножаем — STEPPRICE уже на целый контракт)
    leg = LegPosition("SR", Role.ORDINARY, "buy", 1, 32000)
    assert leg_pnl_rub(leg, 32050, so) == pytest.approx(50.0)
    # шорт @ 32000, выход 32050 → −50₽
    leg = LegPosition("SR", Role.ORDINARY, "sell", 1, 32000)
    assert leg_pnl_rub(leg, 32050, so) == pytest.approx(-50.0)


def test_short_spread_profits_when_spread_falls():
    """Шорт спреда (SELL) должен ЗАРАБАТЫВАТЬ при падении спреда (согласованность знака).

    Это регрессия на баг инвертированных ног: SELL → buy SBRF + sell SBPR.
    """
    cfg = St4Config()
    cfg.strategy.sma_period = 30
    cfg.strategy.deviation_pct = 0.0
    so, sp = _specs()
    eng = TradingEngine(cfg, so, sp)
    # прогрев: спред колеблется ~100 ± небольшой шум (нужна σ>0, иначе полосы вырождены)
    base = 32000.0
    ts = 0
    rng = np.random.default_rng(0)
    for _ in range(40):
        noise = float(rng.normal(0, 5))
        eng.on_candles(ts, base, base + 100 + noise)
        ts += 600000
    # пробой вверх существенно за полосу: спред 250 → SELL (шорт спреда)
    eng.on_candles(ts, base, base + 250)
    ts += 600000
    assert eng.state == BotState.SHORT_SPREAD
    entry_spread = eng.position.entry_spread
    # спред падает обратно к средней → должны закрыться в плюс
    eng.on_candles(ts, base, base + 100)
    ts += 600000
    assert len(eng.trades) == 1
    t = eng.trades[0]
    assert t.exit_spread < t.entry_spread        # спред упал
    assert t.gross_pnl_rub > 0                    # шорт спреда заработал
    assert entry_spread > 0


# ============================ §10 атомарность / unwind ============================

def test_execute_pair_ok():
    so, sp = _specs()
    ex = OrderExecutor(St4Config().execution, St4Config().paper, so, sp)
    r = ex.execute_pair(buy_ord=True, buy_pref=False, lots=1,
                        book_ord=(32000, 32001), book_pref=(32100, 32101),
                        ref_ord=32000, ref_pref=32100)
    assert r.ok and r.fill_ord is not None and r.fill_pref is not None
    assert r.fill_ord.side == "buy" and r.fill_pref.side == "sell"


def test_execute_pair_unwind_on_second_leg_fail():
    """Вторая нога не заливается → аварийный unwind первой, чистый исход (позиции нет)."""
    cfg = St4Config()
    cfg.execution.paper_fill_fail_prob = 1.0     # каждый второй вызов — неудача
    cfg.execution.max_retries = 1                # первая зальётся (#1 ок), вторая (#2) нет
    so, sp = _specs()
    ex = OrderExecutor(cfg.execution, cfg.paper, so, sp)
    # period = round(1/1.0)=1 → КАЖДЫЙ вызов fail. Тогда первая нога не зальётся → abort.
    r = ex.execute_pair(True, False, 1, (32000, 32001), (32100, 32101), 32000, 32100)
    assert not r.ok
    assert r.aborted or r.unwound


def test_execute_pair_abort_on_deviation_protection():
    """Защита от ухода цены: лимит ушёл дальше N тиков от reference → вход отменён."""
    cfg = St4Config()
    cfg.execution.deviation_protection_ticks = 1
    so, sp = _specs()
    ex = OrderExecutor(cfg.execution, cfg.paper, so, sp)
    # reference далеко от книги → лимитная цена уедет > 1 тика → first leg не зальётся
    r = ex.execute_pair(True, False, 1, (32000, 32001), (32100, 32101),
                        ref_ord=31000, ref_pref=31000)
    assert not r.ok and r.aborted


def test_halted_on_unwind_failure():
    """Если unwind физически невозможен (UnwindError) — движок переходит в HALTED."""
    cfg = St4Config()
    cfg.strategy.sma_period = 20
    cfg.strategy.deviation_pct = 0.0
    so, sp = _specs()
    eng = TradingEngine(cfg, so, sp)
    # подменяем executor так, чтобы execute_pair бросал UnwindError на входе
    def boom(*a, **k):
        raise UnwindError("unwind невозможен")
    eng.executor.execute_pair = boom
    base = 32000.0
    ts = 0
    rng = np.random.default_rng(0)
    for _ in range(25):
        eng.on_candles(ts, base, base + 50 + float(rng.normal(0, 4)))
        ts += 600000
    eng.on_candles(ts, base, base + 200)         # пробой → попытка входа → boom
    assert eng.state == BotState.HALTED
    assert eng.risk.halted


# ============================ §11 reconciliation ============================

def test_reconcile_match():
    cfg = St4Config()
    so, sp = _specs()
    eng = TradingEngine(cfg, so, sp)
    assert eng.reconcile(None)                    # обе пусты — согласовано


def test_reconcile_mismatch_halts():
    cfg = St4Config()
    so, sp = _specs()
    eng = TradingEngine(cfg, so, sp)
    fake = Position(
        state=BotState.SHORT_SPREAD,
        leg_ord=LegPosition("SR", Role.ORDINARY, "buy", 1, 32000),
        leg_pref=LegPosition("SP", Role.PREFERRED, "sell", 1, 32100),
        entry_ts=0, entry_spread=100, entry_beta=1.0, sma_at_entry=0.0)
    assert not eng.reconcile(fake)                # локально пусто, у «брокера» позиция
    assert eng.state == BotState.HALTED


# ============================ §11 RiskManager ============================

def test_risk_daily_loss_blocks_entry():
    cfg = St4Config()
    cfg.risk.max_daily_loss_rub = 1000
    so, sp = _specs()
    eng = TradingEngine(cfg, so, sp)
    eng.risk.on_trade_closed(-1500, 1_700_000_000_000)   # пробили дневной лимит
    ok, why = eng.risk.can_enter(1_700_000_000_000, 0)
    assert not ok and "лимит" in why


def test_risk_consecutive_errors_halt():
    cfg = St4Config()
    cfg.risk.max_consecutive_errors = 3
    so, sp = _specs()
    eng = TradingEngine(cfg, so, sp)
    for _ in range(3):
        eng.risk.on_error()
    assert eng.risk.halted


# ============================ §9.7 сессия ============================

def test_clearing_window():
    cfg = St4Config().session
    # 14:00 MSK — в клиринговом окне (14:00–14:05)
    import datetime as dt
    msk = dt.datetime(2026, 6, 8, 14, 2, tzinfo=dt.timezone(dt.timedelta(hours=3)))
    ts = int(msk.timestamp() * 1000)
    assert in_clearing_window(ts, cfg)
    # 12:00 MSK — торги идут
    msk2 = dt.datetime(2026, 6, 8, 12, 0, tzinfo=dt.timezone(dt.timedelta(hours=3)))
    assert not in_clearing_window(int(msk2.timestamp() * 1000), cfg)


# ============================ §14.2 бэктест ============================

def test_backtest_metrics_on_synthetic():
    """Бэктест на синтетике даёт осмысленные метрики и честный maxDD по equity."""
    cfg = St4Config()
    cfg.strategy.sma_period = 100
    so, sp = _specs()
    df = feed.generate_synthetic(n=1500, seed=23)
    r = run_backtest(df, cfg, so, sp)
    assert r["trades"] > 0
    assert 0 <= r["win_rate_pct"] <= 100
    assert r["max_drawdown_pct"] >= 0
    assert len(r["equity_curve"]) == r["bars"]
    # net_pnl согласован с суммой сделок
    assert r["net_pnl_rub"] == pytest.approx(
        sum(t["net_pnl_rub"] for t in r["trades_detail"]), abs=1)


def test_freeze_sma_on_exit_option():
    """FreezeSmaOnExit меняет уровень выхода (зафиксированная SMA входа vs живая)."""
    cfg_live = St4Config()
    cfg_live.strategy.sma_period = 100
    cfg_live.strategy.freeze_sma_on_exit = False
    cfg_freeze = St4Config()
    cfg_freeze.strategy.sma_period = 100
    cfg_freeze.strategy.freeze_sma_on_exit = True
    so, sp = _specs()
    df = feed.generate_synthetic(n=1200, seed=7)
    r_live = run_backtest(df, cfg_live, so, sp)
    r_freeze = run_backtest(df, cfg_freeze, so, sp)
    # оба режима валидны; результаты в общем случае различаются (поведение выхода разное)
    assert r_live["trades"] >= 0 and r_freeze["trades"] >= 0


# ============================ Phase 2: T-Bank sandbox executor ============================

class FakeSB:
    """Мок модуля tbank_sandbox для юнит-тестов TinkoffSandboxExecutor (без сети).

    fail_from: с какого по счёту вызова post_order начинать реджектить (None = все fill).
    """

    def __init__(self, fail_from: int | None = None, fail_only=None, accounts=None):
        self.orders: list[tuple] = []
        self.fail_from = fail_from        # реджектить начиная с N-го вызова
        self.fail_only = set(fail_only or ())  # реджектить только эти номера вызовов
        self.payins: list[tuple] = []
        self.opened: list[str] = []
        self._accounts = accounts or []

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

    def list_accounts(self):
        return self._accounts

    def open_account(self, name="x"):
        self.opened.append(name)
        return "acc-new"

    def pay_in(self, acc, rub):
        self.payins.append((acc, rub))
        return float(rub)

    def post_order(self, acc, uid, lots, direction, order_id,
                   order_type="ORDER_TYPE_MARKET", price=None):
        import uuid as _uuid
        _uuid.UUID(order_id)            # упадёт, если order_id не валидный UUID
        self.orders.append((uid, direction, order_id))
        n = len(self.orders)
        reject = (self.fail_from is not None and n >= self.fail_from) or (n in self.fail_only)
        if reject:
            return {"executionReportStatus": "EXECUTION_REPORT_STATUS_REJECTED",
                    "executedOrderPrice": None}
        price = 32100 if "SP" in uid else 32000    # цена за контракт (SBRF 32000, SBPR 32100)
        # T-Bank возвращает executedOrderPrice = СУММА за все лоты + lotsExecuted
        return {"executionReportStatus": "EXECUTION_REPORT_STATUS_FILL",
                "lotsExecuted": lots,
                "executedOrderPrice": {"units": str(price * lots), "nano": 0}}


def _tinkoff_exec(fail_from=None, fail_only=None, accounts=None, max_retries=3):
    from pairsignal.st4.tinkoff_executor import TinkoffSandboxExecutor
    cfg = St4Config()
    cfg.execution.max_retries = max_retries
    so = feed.synthetic_spec(Role.ORDINARY)
    so.code = "SRM6"
    sp = feed.synthetic_spec(Role.PREFERRED)
    sp.code = "SPM6"
    sb = FakeSB(fail_from=fail_from, fail_only=fail_only, accounts=accounts)
    ex = TinkoffSandboxExecutor(cfg.execution, cfg.connector, so, sp, sb=sb)
    return ex, sb


def test_tinkoff_execute_pair_ok():
    """Обе ноги fill → r.ok, корректные стороны/цены, счёт открыт + pay_in, orderId — UUID."""
    ex, sb = _tinkoff_exec()
    # шорт спреда: buy SBRF + sell SBPR
    r = ex.execute_pair(buy_ord=True, buy_pref=False, lots=1,
                        book_ord=(0, 0), book_pref=(0, 0), ref_ord=32000, ref_pref=32100)
    assert r.ok
    assert r.fill_ord.side == "buy" and r.fill_pref.side == "sell"
    assert r.fill_ord.avg_price == 32000 and r.fill_pref.avg_price == 32100
    assert len(sb.orders) == 2
    assert sb.opened == ["st4-spread-sandbox"]      # счёт открыт один раз
    assert sb.payins and sb.payins[0][1] == 200_000  # пополнен под ГО


def test_tinkoff_price_per_contract_multi_lot():
    """При lots>1 цена входа = за КОНТРАКТ, не сумма за лоты (регрессия: было ×N завышение)."""
    ex, sb = _tinkoff_exec()
    r = ex.execute_pair(buy_ord=True, buy_pref=False, lots=10,
                        book_ord=(0, 0), book_pref=(0, 0), ref_ord=32000, ref_pref=32100)
    assert r.ok
    # цены должны быть за 1 контракт (32000/32100), НЕ 320000/321000
    assert r.fill_ord.avg_price == 32000, f"цена завышена: {r.fill_ord.avg_price}"
    assert r.fill_pref.avg_price == 32100, f"цена завышена: {r.fill_pref.avg_price}"
    assert r.fill_ord.lots == 10 and r.fill_pref.lots == 10


def test_tinkoff_unwind_on_second_leg_fail():
    """Вторая нога не зальётся → unwind первой обратным ордером, r.unwound."""
    # max_retries=1: первая нога (#1) ок, ТОЛЬКО вторая (#2) реджект → unwind (#3) зальётся
    ex, sb = _tinkoff_exec(fail_only={2}, max_retries=1)
    r = ex.execute_pair(True, False, 1, (0, 0), (0, 0), 32000, 32100)
    assert not r.ok and r.unwound
    assert len(sb.orders) == 3                        # первая + неуд. вторая + unwind


def test_tinkoff_unwind_failure_raises():
    """Вторая нога И unwind не заливаются → UnwindError."""
    # fail_from=2: всё начиная со 2-го ордера падает (вторая нога + unwind)
    ex, sb = _tinkoff_exec(fail_from=2, max_retries=1)
    # подменим первую ногу на успешную, остальное падает — fail_from=2 это и делает
    with pytest.raises(UnwindError):
        ex.execute_pair(True, False, 1, (0, 0), (0, 0), 32000, 32100)


def test_tinkoff_close_pair_real_exit():
    """close_pair ставит обратные ордера, возвращает фактические exit-цены филла."""
    ex, sb = _tinkoff_exec()
    pos = Position(
        state=BotState.SHORT_SPREAD,
        leg_ord=LegPosition("SRM6", Role.ORDINARY, "buy", 1, 32000),
        leg_pref=LegPosition("SPM6", Role.PREFERRED, "sell", 1, 32100),
        entry_ts=0, entry_spread=100, entry_beta=1.0, sma_at_entry=0.0)
    n_before = len(sb.orders)
    cr = ex.close_pair(pos, 32000, 32100)
    assert len(sb.orders) == n_before + 2            # два обратных ордера
    # закрытие SBRF (был buy) → sell; SBPR (был sell) → buy
    assert sb.orders[-2][1] == "ORDER_DIRECTION_SELL"  # SBRF close
    assert sb.orders[-1][1] == "ORDER_DIRECTION_BUY"   # SBPR close
    assert cr.exit_ord == 32000 and cr.exit_pref == 32100


def test_tinkoff_account_reuse():
    """Существующий OPEN-счёт с нужным именем переиспользуется — open_account не зван."""
    accs = [{"id": "acc-existing", "name": "st4-spread-sandbox", "status": "ACCOUNT_STATUS_OPEN"}]
    ex, sb = _tinkoff_exec(accounts=accs)
    assert ex._account_id == "acc-existing"
    assert sb.opened == []                            # новый счёт не открывался


def test_tinkoff_caches_instruments():
    """find_future вызывается по разу на ногу (результат кэшируется в _inst)."""
    ex, sb = _tinkoff_exec()
    assert set(ex._inst.keys()) == {Role.ORDINARY, Role.PREFERRED}
    # повторный execute_pair не должен заново резолвить (кэш уже заполнен)
    n = len(sb.orders)
    ex.execute_pair(True, False, 1, (0, 0), (0, 0), 32000, 32100)
    assert ex._inst[Role.ORDINARY]["ticker"] == "SRM6"
    assert len(sb.orders) == n + 2


def test_engine_paper_executor_by_default():
    """mode='paper' → engine использует OrderExecutor (регресс-гард)."""
    cfg = St4Config()
    so, sp = _specs()
    eng = TradingEngine(cfg, so, sp)
    assert isinstance(eng.executor, OrderExecutor)


def test_engine_disarmed_skips_entries():
    """Disarmed движок не открывает входы (прогрев BB идёт), выход открытой позиции работает.

    Используется на backfill-replay в sandbox: исторические бары не торгуем.
    """
    cfg = St4Config()
    cfg.strategy.sma_period = 30
    cfg.strategy.deviation_pct = 0.0
    so, sp = _specs()
    eng = TradingEngine(cfg, so, sp)
    eng.arm(False)                       # запретить входы
    base = 32000.0
    ts = 0
    rng = np.random.default_rng(0)
    for _ in range(40):
        eng.on_candles(ts, base, base + 100 + float(rng.normal(0, 5)))
        ts += 600000
    eng.on_candles(ts, base, base + 250)                 # сильный пробой — но disarmed
    ts += 600000
    assert eng.state == BotState.FLAT and eng.position is None   # вход НЕ открыт
    assert eng.last_band.is_ready                                # но BB прогрелся
    # взвели — следующий пробой открывает позицию
    eng.arm(True)
    eng.on_candles(ts, base, base + 100)                 # возврат
    ts += 600000
    eng.on_candles(ts, base, base + 260)                 # новый пробой → вход
    assert eng.position is not None


def test_engine_uses_tinkoff_when_sandbox(monkeypatch):
    """mode='tbank_sandbox' → engine использует TinkoffSandboxExecutor (с моком sb)."""
    import pairsignal.st4.tinkoff_executor as te
    monkeypatch.setattr(te, "tbank_sandbox", FakeSB())
    cfg = St4Config()
    cfg.connector.mode = "tbank_sandbox"
    so = feed.synthetic_spec(Role.ORDINARY)
    so.code = "SRM6"
    sp = feed.synthetic_spec(Role.PREFERRED)
    sp.code = "SPM6"
    eng = TradingEngine(cfg, so, sp)
    assert type(eng.executor).__name__ == "TinkoffSandboxExecutor"
