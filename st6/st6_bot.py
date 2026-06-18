#!/usr/bin/env python3
"""
st6 · Correlation-Gated Pairs — рабочий бот для T-Bank Invest API (песочница).

Реализует:
  • загрузку закрытых свечей по корзине бумаг (T-Bank get_all_candles);
  • сканер пар: выбор лучшей коррелированной + коинтегрированной пары;
  • торговый цикл по закрытым свечам с корреляционным гейтом (st6_core.decide);
  • атомарное исполнение пары sandbox-ордерами (менее ликвидную ногу первой,
    при срыве второй — аварийный unwind первой);
  • бэктест-режим на истории T-Bank (без сети используйте st6_backtest).

⚠ Только SandboxService. Реальные биржевые ордера не отправляются.

Запуск:
    pip install "tinkoff-investments>=0.2" numpy
    export INVEST_TOKEN=t.xxxxx         # токен T-Bank Invest (sandbox-доступ)
    python st6_bot.py scan              # показать лучшие пары корзины
    python st6_bot.py backtest          # бэктест лучшей пары на истории T-Bank
    python st6_bot.py live              # торговый цикл в песочнице
    python st6_bot.py live --dry        # тот же цикл, но без отправки ордеров

Документация SDK: https://russianinvestments.github.io/invest-python/
"""

from __future__ import annotations

import argparse
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional

import numpy as np

from st6_core import (
    ExitReason, Params, PairStat, Position, Side,
    decide, leg_quantities, rank_pairs, trade_pnl,
)

log = logging.getLogger("st6")

# --------------------------------------------------------------------------
# Корзина: ликвидные коррелированные бумаги MOEX, доступные на T-Bank.
# Сектора подобраны так, чтобы внутри был сильный коинтеграционный эдж.
# Сканер сам выберет лучшую пару — список можно расширять/сужать.
# --------------------------------------------------------------------------
DEFAULT_BASKET = [
    # нефть и газ
    "LKOH", "ROSN", "SIBN", "TATN", "GAZP", "NVTK",
    # металлурги
    "GMKN", "NLMK", "MAGN", "CHMF",
    # банки / голубые фишки
    "SBER",
]

# Таймфрейм по умолчанию — часовой (баланс между шумом и числом сделок).
DEFAULT_DAYS_BACK = 200
DEFAULT_INTERVAL = "1h"

INTERVAL_MAP = {
    "1m": "CANDLE_INTERVAL_1_MIN",
    "5m": "CANDLE_INTERVAL_5_MIN",
    "10m": "CANDLE_INTERVAL_10_MIN",
    "15m": "CANDLE_INTERVAL_15_MIN",
    "1h": "CANDLE_INTERVAL_HOUR",
    "4h": "CANDLE_INTERVAL_4_HOUR",
    "1d": "CANDLE_INTERVAL_DAY",
}

# приблизительная длительность бара в секундах (для частоты опроса live)
INTERVAL_SECONDS = {
    "1m": 60, "5m": 300, "10m": 600, "15m": 900,
    "1h": 3600, "4h": 14400, "1d": 86400,
}


# --------------------------------------------------------------------------
# Утилиты T-Bank (ленивый импорт, чтобы ядро/тесты не требовали пакет)
# --------------------------------------------------------------------------
def _q2f(q) -> float:
    """Quotation/MoneyValue -> float."""
    return q.units + q.nano / 1e9


def _tinkoff():
    import tinkoff.invest as ti  # noqa
    return ti


@dataclass
class Instrument:
    ticker: str
    figi: str
    uid: str
    lot: int
    min_increment: float


@dataclass
class MarketData:
    instruments: dict[str, Instrument] = field(default_factory=dict)
    closes: dict[str, np.ndarray] = field(default_factory=dict)  # выровненные ряды


class TBank:
    """Тонкая обёртка над SandboxClient: справочник, свечи, ордера."""

    def __init__(self, token: str, interval: str = DEFAULT_INTERVAL):
        self.token = token
        self.interval = interval
        ti = _tinkoff()
        self._ti = ti
        from tinkoff.invest.sandbox.client import SandboxClient
        self._client_cm = SandboxClient(token)
        self.client = None
        self.account_id: Optional[str] = None

    # ---- контекст ----
    def __enter__(self):
        self.client = self._client_cm.__enter__()
        return self

    def __exit__(self, *exc):
        return self._client_cm.__exit__(*exc)

    # ---- аккаунт песочницы ----
    def ensure_account(self, initial_rub: float = 1_000_000.0) -> str:
        accs = self.client.users.get_accounts().accounts
        if accs:
            self.account_id = accs[0].id
        else:
            self.account_id = self.client.sandbox.open_sandbox_account().account_id
            self._pay_in(initial_rub)
        return self.account_id

    def _pay_in(self, rub: float):
        ti = self._ti
        from tinkoff.invest.utils import decimal_to_quotation
        from decimal import Decimal
        money = decimal_to_quotation(Decimal(str(rub)))
        self.client.sandbox.sandbox_pay_in(
            account_id=self.account_id,
            amount=ti.MoneyValue(units=money.units, nano=money.nano, currency="rub"),
        )

    def equity_rub(self) -> float:
        pos = self.client.operations.get_positions(account_id=self.account_id)
        rub = 0.0
        for m in pos.money:
            if m.currency == "rub":
                rub += _q2f(m)
        # плюс рыночная стоимость бумаг (упрощённо — по последним ценам)
        return rub

    # ---- справочник инструментов ----
    def resolve(self, tickers: list[str]) -> dict[str, Instrument]:
        out: dict[str, Instrument] = {}
        shares = self.client.instruments.shares().instruments
        by_ticker = {s.ticker: s for s in shares}
        for t in tickers:
            s = by_ticker.get(t)
            if not s:
                log.warning("тикер %s не найден среди акций T-Bank — пропуск", t)
                continue
            if not s.api_trade_available_flag:
                log.warning("по %s недоступна торговля через API — пропуск", t)
                continue
            out[t] = Instrument(
                ticker=t, figi=s.figi, uid=s.uid, lot=s.lot,
                min_increment=_q2f(s.min_price_increment),
            )
        return out

    # ---- свечи ----
    def load_closes(self, inst: Instrument, days_back: int) -> np.ndarray:
        ti = self._ti
        from tinkoff.invest.utils import now
        interval = getattr(ti.CandleInterval, INTERVAL_MAP[self.interval])
        closes = []
        for c in self.client.get_all_candles(
            instrument_id=inst.figi,
            from_=now() - timedelta(days=days_back),
            interval=interval,
        ):
            if c.is_complete:           # только закрытые свечи — no repaint
                closes.append(_q2f(c.close))
        return np.asarray(closes, dtype=float)

    def market_data(self, tickers: list[str], days_back: int) -> MarketData:
        md = MarketData()
        md.instruments = self.resolve(tickers)
        raw = {}
        for t, inst in md.instruments.items():
            arr = self.load_closes(inst, days_back)
            if len(arr) > 10:
                raw[t] = arr
                log.info("%s: %d закрытых свечей", t, len(arr))
        # выравниваем по минимальной длине (грубое выравнивание по хвосту)
        if raw:
            n = min(len(v) for v in raw.values())
            md.closes = {t: v[-n:] for t, v in raw.items()}
        return md

    def last_price(self, inst: Instrument) -> float:
        lp = self.client.market_data.get_last_prices(
            instrument_id=[inst.figi]).last_prices
        return _q2f(lp[0].price) if lp else 0.0

    # ---- ордера (sandbox) ----
    def market_order(self, inst: Instrument, lots: int, buy: bool, dry: bool):
        ti = self._ti
        direction = (ti.OrderDirection.ORDER_DIRECTION_BUY if buy
                     else ti.OrderDirection.ORDER_DIRECTION_SELL)
        if dry:
            log.info("[dry] %s %s %d лот", "BUY" if buy else "SELL",
                     inst.ticker, lots)
            return None
        return self.client.sandbox.post_sandbox_order(
            quantity=lots,
            direction=direction,
            account_id=self.account_id,
            order_type=ti.OrderType.ORDER_TYPE_MARKET,
            order_id=str(uuid.uuid4()),
            instrument_id=inst.figi,
        )


# --------------------------------------------------------------------------
# Атомарное исполнение пары
# --------------------------------------------------------------------------
def open_pair(tb: TBank, ia: Instrument, ib: Instrument,
              qa: int, qb: int, side: Side, dry: bool) -> bool:
    """
    LONG_SPREAD = long A / short B; SHORT_SPREAD = short A / long B.
    Менее ликвидную ногу (B — обычно она дальше в корзине) заливаем первой;
    при срыве второй ноги откатываем первую (нет «голой» ноги).
    """
    a_buy = (side == Side.LONG_SPREAD)
    b_buy = not a_buy
    try:
        tb.market_order(ib, qb, b_buy, dry)       # сначала менее ликвидная нога B
    except Exception as e:
        log.error("нога B не открылась: %s — вход отменён", e)
        return False
    try:
        tb.market_order(ia, qa, a_buy, dry)       # затем нога A
    except Exception as e:
        log.error("нога A не открылась: %s — аварийный unwind B", e)
        try:
            tb.market_order(ib, qb, not b_buy, dry)  # откат B
        except Exception as e2:
            log.critical("UNWIND B ТОЖЕ СОРВАЛСЯ: %s — нужна ручная проверка!", e2)
        return False
    return True


def close_pair(tb: TBank, ia: Instrument, ib: Instrument,
               qa: int, qb: int, side: Side, dry: bool) -> bool:
    """Закрытие — обратные сделки к open_pair."""
    a_was_buy = (side == Side.LONG_SPREAD)
    try:
        tb.market_order(ia, qa, not a_was_buy, dry)
        tb.market_order(ib, qb, a_was_buy, dry)
        return True
    except Exception as e:
        log.error("ошибка закрытия пары: %s — проверьте позицию вручную", e)
        return False


# --------------------------------------------------------------------------
# Команды
# --------------------------------------------------------------------------
def cmd_scan(tb: TBank, basket: list[str], days_back: int, p: Params):
    md = tb.market_data(basket, days_back)
    if len(md.closes) < 2:
        log.error("недостаточно данных для сканирования")
        return None
    ranked = rank_pairs(md.closes, p)
    if not ranked:
        print("\nГодных пар не найдено (ослабьте select_min_corr / "
              "select_max_pvalue).")
        return None
    print("\n  пара            corr    β      p-value  half-life  score")
    print("  " + "-" * 58)
    for s in ranked[:12]:
        print(f"  {s.a:>5}/{s.b:<5}  {s.corr:+.2f}  {s.beta:5.2f}  "
              f"{s.pvalue:6.3f}   {s.halflife:7.1f}   {s.score:.3f}")
    best = ranked[0]
    print(f"\n→ лучшая пара: {best.a}/{best.b}\n")
    return ranked


def cmd_backtest(tb: TBank, basket: list[str], days_back: int, p: Params):
    from st6_backtest import backtest_pair
    ranked = cmd_scan(tb, basket, days_back, p)
    if not ranked:
        return
    best = ranked[0]
    md = tb.market_data([best.a, best.b], days_back)
    ia, ib = md.instruments[best.a], md.instruments[best.b]
    r = backtest_pair(md.closes[best.a], md.closes[best.b], p,
                      start_equity=1_000_000.0, lot_a=ia.lot, lot_b=ib.lot)
    print(f"\nБэктест {best.a}/{best.b} на истории T-Bank ({days_back}д, "
          f"{tb.interval}):")
    print(f"  сделок={r.n_trades}  win={r.win_rate:.0%}  net={r.total_net:,.0f}₽  "
          f"ret={r.return_pct:.1f}%  maxDD={r.max_drawdown_pct:.1f}%  "
          f"sharpe={r.sharpe:.2f}")
    if r.trades:
        print("\n  направление   держал  net ₽     причина")
        for t in r.trades[-12:]:
            print(f"  {t.side:<12}  {t.bars:>5}  {t.net:>9,.0f}  {t.reason}")


def cmd_live(tb: TBank, basket: list[str], days_back: int, p: Params,
             dry: bool, once: bool):
    tb.ensure_account()
    log.info("sandbox account: %s  equity≈%.0f₽", tb.account_id, tb.equity_rub())

    # выбираем пару один раз при старте (можно периодически реселектить)
    ranked = cmd_scan(tb, basket, days_back, p)
    if not ranked:
        return
    best = ranked[0]
    md = tb.market_data([best.a, best.b], days_back)
    ia, ib = md.instruments[best.a], md.instruments[best.b]
    log.info("торгуем пару %s/%s", best.a, best.b)

    pos = Position()
    poll = max(15, INTERVAL_SECONDS.get(tb.interval, 3600) // 4)

    while True:
        try:
            a = tb.load_closes(ia, days_back)
            b = tb.load_closes(ib, days_back)
            n = min(len(a), len(b))
            a, b = a[-n:], b[-n:]
            sig = decide(a, b, pos, p)
            ca, cb = a[-1], b[-1]
            log.info("z=%+.2f corr=%+.2f β=%.2f %s %s", sig.z, sig.corr, sig.beta,
                     sig.action, "" if not pos.is_open else f"({pos.side.name})")

            if pos.is_open:
                pos.bars_held += 1
                if sig.action == "EXIT":
                    if close_pair(tb, ia, ib, pos.qty_a, pos.qty_b, pos.side, dry):
                        units_a, units_b = pos.qty_a * ia.lot, pos.qty_b * ib.lot
                        net = trade_pnl(pos.side, pos.entry_a, ca,
                                        pos.entry_b, cb, units_a, units_b, p)
                        log.info("ВЫХОД %s | %s | net≈%.0f₽",
                                 pos.side.name, sig.reason.value, net)
                        pos = Position()
            else:
                if sig.action in ("ENTER_LONG", "ENTER_SHORT"):
                    eq = tb.equity_rub() or 1_000_000.0
                    qa, qb = leg_quantities(eq, ca, cb, sig.beta, ia.lot, ib.lot, p)
                    side = (Side.LONG_SPREAD if sig.action == "ENTER_LONG"
                            else Side.SHORT_SPREAD)
                    if qa > 0 and qb > 0 and open_pair(tb, ia, ib, qa, qb, side, dry):
                        pos = Position(side=side, entry_z=sig.z, beta=sig.beta,
                                       bars_held=0, qty_a=qa, qty_b=qb,
                                       entry_a=ca, entry_b=cb)
                        log.info("ВХОД %s | qa=%d qb=%d | z=%.2f",
                                 side.name, qa, qb, sig.z)
        except Exception as e:
            log.error("ошибка цикла: %s", e)

        if once:
            break
        time.sleep(poll)


# --------------------------------------------------------------------------
def build_params(args) -> Params:
    p = Params()
    if args.entry is not None:
        p.z_entry = args.entry
    if args.stop is not None:
        p.z_stop = args.stop
    if args.risk is not None:
        p.risk_fraction = args.risk
    return p


def main():
    ap = argparse.ArgumentParser(description="st6 Correlation-Gated Pairs bot (T-Bank sandbox)")
    ap.add_argument("command", choices=["scan", "backtest", "live"])
    ap.add_argument("--dry", action="store_true", help="не отправлять ордера")
    ap.add_argument("--once", action="store_true", help="один проход live-цикла")
    ap.add_argument("--interval", default=DEFAULT_INTERVAL, choices=list(INTERVAL_MAP))
    ap.add_argument("--days", type=int, default=DEFAULT_DAYS_BACK)
    ap.add_argument("--tickers", nargs="*", default=None, help="своя корзина")
    ap.add_argument("--entry", type=float, default=None, help="z_entry")
    ap.add_argument("--stop", type=float, default=None, help="z_stop")
    ap.add_argument("--risk", type=float, default=None, help="доля equity на ногу")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    token = os.environ.get("INVEST_TOKEN")
    if not token:
        raise SystemExit("Не задан INVEST_TOKEN (токен T-Bank Invest, sandbox-доступ).")

    basket = args.tickers or DEFAULT_BASKET
    p = build_params(args)

    with TBank(token, interval=args.interval) as tb:
        if args.command == "scan":
            cmd_scan(tb, basket, args.days, p)
        elif args.command == "backtest":
            cmd_backtest(tb, basket, args.days, p)
        elif args.command == "live":
            cmd_live(tb, basket, args.days, p, dry=args.dry, once=args.once)


if __name__ == "__main__":
    main()
