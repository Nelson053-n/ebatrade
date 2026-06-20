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


def _synthetic_fixed_pair_basket(n=1500, seed=7):
    """Синтетическая коинт-пара под именами активной фикс-пары (TATN/TATNP) +
    шумовой тикер. Фикс-пара торгуется именно по своим тикерам."""
    from pairsignal.st6.config import St6Config
    a, b = make_synthetic_pair(n=n, seed=seed)
    ta, tb = St6Config().pair_tickers()
    rng = np.random.default_rng(seed + 1)
    noise = np.exp(np.cumsum(rng.normal(0, 0.012, n)) + 4)
    return {ta: list(a), tb: list(b), "NOISE": list(noise)}


def test_st6_session_player_trades_on_fixed_pair():
    """Прогон синтетики через FSM сессии на ФИКС-паре: paper-сделки идут, rank_pairs НЕ вызывается."""
    import asyncio
    from unittest import mock
    from pairsignal.st6.service import St6Session
    s = St6Session("test_player")
    s.player_closes = _synthetic_fixed_pair_basket(n=1500, seed=7)
    s.state["player"] = True
    # rank_pairs не должен вызываться — отбор отключён, пара фиксирована
    with mock.patch("pairsignal.st6.core.rank_pairs",
                    side_effect=AssertionError("rank_pairs must not be called")):
        asyncio.run(s.run_player())
    snap = s.snapshot(0.0)
    assert snap["pair"] == list(s.cfg.pair_tickers()), snap["pair"]
    assert snap["fixed_pair"] == s.cfg.fixed_pair
    assert snap["summary"]["trades"] > 0, "синтетика должна дать сделки"


def test_st6_fixed_pair_no_rank_pairs_import_in_service():
    """В боевом сервисе отбор из корзины отключён: select_pair НЕ зовёт rank_pairs."""
    from unittest import mock
    from pairsignal.st6.service import St6Session
    s = St6Session("test")
    ta, tb = s.cfg.pair_tickers()
    a, b = make_synthetic_pair(n=900, seed=4)
    s.closes = {ta: list(a), tb: list(b)}
    with mock.patch("pairsignal.st6.core.rank_pairs",
                    side_effect=AssertionError("rank_pairs must not be called")):
        assert s.select_pair() is True
    assert s.pair == (ta, tb)
    assert s.pair_stat is not None and {s.pair_stat.a, s.pair_stat.b} == {ta, tb}


def test_st6_snapshot_has_honest_maker_note():
    """Snapshot отдаёт честную пометку про maker-эдж и режим издержек."""
    from pairsignal.st6.service import St6Session
    s = St6Session("test")
    snap = s.snapshot(0.0)
    assert "edge_note" in snap and "maker" in snap["edge_note"].lower()
    assert snap["cost_mode"] == "maker"  # дефолтные издержки maker
    # contract: фикс-поля на месте
    for k in ("pair", "pair_stat", "cur_z", "cur_corr", "cur_beta", "params",
              "position", "summary", "history", "trades", "wait_reason"):
        assert k in snap, k


def test_st6_pair_specific_z_exit():
    """z_exit пара-специфичен: TATN/TATNP=1.0, SBER/SBERP=0.5."""
    from pairsignal.st6.config import St6Config
    c = St6Config(fixed_pair="TATN/TATNP")
    assert c.pair_z_exit() == 1.0
    assert c.strategy.to_params(z_exit=c.pair_z_exit()).z_exit == 1.0
    c2 = St6Config(fixed_pair="SBER/SBERP")
    assert c2.pair_z_exit() == 0.5
    assert c2.pair_tickers() == ("SBER", "SBERP")
    assert c2.strategy.to_params(z_exit=c2.pair_z_exit()).z_exit == 0.5


def test_st6_maker_costs_default():
    """Дефолтные издержки — maker (0.0001/0.0001), не taker."""
    from pairsignal.st6.config import St6Config
    p = St6Config().strategy.to_params()
    assert p.fee_rate == 0.0001 and p.slippage_rate == 0.0001
    assert (p.fee_rate + p.slippage_rate) <= 0.0002  # maker-порог


def test_st6_old_session_state_loads(tmp_path, monkeypatch):
    """Старый session_state_6_*.json (без fixed_pair, timeframe=1d) грузится без падения."""
    import json
    from pairsignal.st6 import service as svc
    from pairsignal.st6.service import St6Session
    old = {
        "session_started": 1.0,
        "config": {"basket": ["LKOH", "ROSN"], "strategy": {"beta_window": 240,
                   "z_window": 240, "corr_window": 120, "z_exit": 0.3,
                   "fee_rate": 0.0006, "slippage_rate": 0.0005},
                   "timeframe": "1d", "history_days": 720, "connector": "paper"},
        "equity": 1_000_000.0, "trades": [], "position": None, "open_ctx": None,
        "pair": None, "pair_stat": None, "cur": {"z": None, "corr": None, "beta": None},
        "history": [], "bars_since_select": 0, "last_live_ts": 0, "live": False,
        "data_source": "synthetic", "paused_by_user": False,
    }
    monkeypatch.setattr(svc, "_BASE", tmp_path)
    (tmp_path / "session_state_6_legacy.json").write_text(json.dumps(old))
    s = St6Session("legacy")
    assert s.load_session() is True
    # legacy stored config сохранён (timeframe=1d), fixed_pair дефолтнулся
    assert s.cfg.timeframe == "1d"
    assert s.cfg.fixed_pair == "TATN/TATNP"
    snap = s.snapshot(0.0)
    assert snap["summary"]["trades"] == 0


# ----------------------------------------------------- sandbox-исполнитель (мок)
class _FakeSB:
    """Мок tbank_sandbox: акции с РАЗНЫМИ лотами, локальный учёт штук на счёте.

    NLMK lot=10, CHMF lot=1 — проверяем lot-кратное исполнение и reconciliation.
    """

    LOTS = {"NLMK": 10, "CHMF": 1}
    PRICE = {"NLMK": 200.0, "CHMF": 1200.0}

    def __init__(self, fail_second: bool = False, not_tradable: bool = False):
        self.fail_second = fail_second
        self.not_tradable = not_tradable
        self.accounts = []
        self.shares = {"NLMK": 0, "CHMF": 0}   # штук на счёте (по тикеру)
        self.calls = 0

    # справочник
    def find_share(self, ticker):
        if ticker not in self.LOTS:
            raise RuntimeError(f"акция {ticker} не найдена")
        return {"ticker": ticker, "figi": f"FIGI_{ticker}", "uid": f"UID_{ticker}",
                "lot": self.LOTS[ticker], "minPriceIncrement": {"units": "0", "nano": 10_000_000}}

    def _uid(self, it):
        return it.get("uid") or it.get("figi")

    def _q_to_float(self, q):
        if not q:
            return 0.0
        return int(q.get("units", 0)) + int(q.get("nano", 0)) / 1e9

    def is_tradable(self, uid):
        return not self.not_tradable

    # счёт
    def list_accounts(self):
        return self.accounts

    def open_account(self, name):
        acc = {"id": "ACC1", "name": name, "status": "ACCOUNT_STATUS_OPEN"}
        self.accounts.append(acc)
        return acc["id"]

    def pay_in(self, account_id, rub):
        return float(rub)

    # ордера
    def _ticker_by_uid(self, uid):
        return "NLMK" if uid.endswith("NLMK") else "CHMF"

    def post_order(self, account_id, uid, lots, direction, order_id, order_type="ORDER_TYPE_MARKET"):
        self.calls += 1
        ticker = self._ticker_by_uid(uid)
        # вторая нога всегда падает (для теста unwind): CHMF исполняется второй (меньший лот),
        # реджектим все её заявки — первую (NLMK) исполнитель откатит обратным ордером.
        if self.fail_second and ticker == "CHMF":
            return {"executionReportStatus": "EXECUTION_REPORT_STATUS_REJECTED"}
        shares = lots * self.LOTS[ticker]
        signed = shares if direction == "ORDER_DIRECTION_BUY" else -shares
        self.shares[ticker] += signed
        total = self.PRICE[ticker] * shares    # СУММА за все штуки
        return {"executionReportStatus": "EXECUTION_REPORT_STATUS_FILL",
                "lotsExecuted": str(lots),
                "executedOrderPrice": {"units": str(int(total)), "nano": 0}}

    def positions(self, account_id):
        return {"securities": [
            {"instrumentUid": f"UID_{t}", "balance": n} for t, n in self.shares.items() if n]}


def _make_executor(**kw):
    from pairsignal.st6.config import ConnectorConfig
    from pairsignal.st6.tinkoff_executor import St6SandboxExecutor
    sb = _FakeSB(**kw)
    ex = St6SandboxExecutor(ConnectorConfig(), "NLMK", "CHMF", sb=sb)
    return ex, sb


def test_sandbox_executor_different_lots_entry_and_exit():
    """Вход/выход пары с РАЗНЫМИ лотами: штуки кратны лоту, счёт возвращается к нулю."""
    from pairsignal.st6.core import Side
    ex, sb = _make_executor()
    # 95 штук NLMK (lot 10 → 9 лотов = 90 шт), 7 штук CHMF (lot 1 → 7 шт)
    r = ex.execute_pair(Side.LONG_SPREAD, 95, 7, 200.0, 1200.0)
    assert r.ok, r.reason
    assert r.fill_a.ticker == "NLMK" and r.fill_a.shares == 90   # 95//10*10
    assert r.fill_b.ticker == "CHMF" and r.fill_b.shares == 7
    assert sb.shares == {"NLMK": 90, "CHMF": -7}                 # long A / short B
    # выход — обратные ордера, счёт к нулю
    from pairsignal.st6.core import Position
    pos = Position(side=Side.LONG_SPREAD, qty_a=90, qty_b=7, entry_a=200.0, entry_b=1200.0)
    cr = ex.close_pair(pos)
    assert cr.exit_a == 200.0 and cr.exit_b == 1200.0
    assert sb.shares == {"NLMK": 0, "CHMF": 0}


def test_sandbox_executor_unwind_on_second_leg_fail():
    """Срыв второй ноги → unwind первой, итог чистый (счёт к нулю), ok=False."""
    from pairsignal.st6.core import Side
    ex, sb = _make_executor(fail_second=True)
    r = ex.execute_pair(Side.SHORT_SPREAD, 30, 5, 200.0, 1200.0)
    assert not r.ok and r.unwound
    assert all(v == 0 for v in sb.shares.values()), sb.shares


def test_sandbox_executor_is_tradable_gate():
    """is_tradable=False, когда биржа закрыта (обе ноги проверяются)."""
    ex, _ = _make_executor(not_tradable=True)
    assert ex.is_tradable() is False
    ex2, _ = _make_executor()
    assert ex2.is_tradable() is True


def test_sandbox_executor_flat_broker_reconciliation():
    """flat_broker закрывает осиротевшие ноги по штукам (reconciliation)."""
    ex, sb = _make_executor()
    sb.shares = {"NLMK": 20, "CHMF": -3}     # висят ноги, движок их не знает
    assert ex.broker_lots() == {"NLMK": 20, "CHMF": -3}
    assert ex.flat_broker() is True
    assert sb.shares == {"NLMK": 0, "CHMF": 0}
