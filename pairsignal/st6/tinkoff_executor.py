"""Реальный (sandbox) исполнитель пары АКЦИЙ MOEX через T-Bank Invest API (st6).

St6SandboxExecutor реализует ПАРНОЕ исполнение двух акций (нога A / нога B) в ПЕСОЧНИЦЕ
T-Bank по образцу st4.TinkoffSandboxExecutor (execute_pair/close_pair, атомарность с
unwind), но для динамической пары: тикеры приходят из выбранной пары (напр. NLMK/CHMF)
и резолвятся в инструменты через tbank_sandbox.find_share(ticker). Лоты у акций РАЗНЫЕ
(NLMK lot=10, CHMF lot=1) — сайзинг ведём в ШТУКАХ (как ядро st6), а на бирже оперируем
ЛОТАМИ (lots = штуки // lot_size).

БЕЗОПАСНОСТЬ: ходит ИСКЛЮЧИТЕЛЬНО через функции tbank_sandbox (только SandboxService.*).
Боевой OrdersService не импортируется. Токен — из окружения процесса (TBANK_TOKEN),
общий с st4 (st4/tbank_sandbox.save_token), здесь не логируется и не возвращается.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from ..st4 import tbank_sandbox
from .config import ConnectorConfig
from .core import Position, Side

_FILL_OK = "EXECUTION_REPORT_STATUS_FILL"


class UnwindError(RuntimeError):
    """Вторую ногу не залили и аварийный unwind первой тоже не удался → HALT."""


@dataclass
class LegFill:
    """Факт исполнения одной ноги-акции в песочнице."""
    ticker: str
    side: str            # "buy" | "sell"
    lots: int            # фактически исполненные ЛОТЫ (биржевые)
    shares: int          # штук бумаги = lots * lot_size
    avg_price: float     # средняя цена за ОДНУ бумагу


@dataclass
class PairFillResult:
    """Результат входа в пару: обе ноги или аварийный исход (как в st4.execution)."""
    ok: bool
    fill_a: LegFill | None = None
    fill_b: LegFill | None = None
    aborted: bool = False    # первая нога не залилась — позиции нет, чисто
    unwound: bool = False    # вторая не залилась, первую закрыли — чисто
    reason: str = ""


@dataclass
class PairCloseResult:
    """Фактические цены выхода обеих ног (для P&L закрытия)."""
    exit_a: float
    exit_b: float


class St6SandboxExecutor:
    """Исполнитель пары акций MOEX в песочнице T-Bank (market-ордера).

    Пара ДИНАМИЧЕСКАЯ: тикеры (ticker_a, ticker_b) задаются при создании и при смене
    пары (reselect) executor пересоздаётся сервисом на новые тикеры. max_retries —
    число повторов рыночной ноги при реджекте.
    """

    def __init__(self, conn_cfg: ConnectorConfig, ticker_a: str, ticker_b: str,
                 max_retries: int = 3, sb=None) -> None:
        self.conn = conn_cfg
        self.ticker_a = ticker_a
        self.ticker_b = ticker_b
        self.max_retries = max_retries
        # sb — модуль-зависимость (инъекция для моков). По умолчанию — реальный tbank_sandbox.
        self.sb = sb if sb is not None else tbank_sandbox
        self._account_id: str = ""
        self._inst: dict[str, dict] = {}      # тикер → запись справочника
        self._ensure_started()

    # ---------- инициализация ----------
    def _resolve_instruments(self) -> None:
        """Найти обе акции в справочнике T-Bank по тикеру, кэшировать (figi/uid/lot)."""
        if self._inst:
            return
        for t in (self.ticker_a, self.ticker_b):
            self._inst[t] = self.sb.find_share(t)

    def _lot_size(self, ticker: str) -> int:
        """Размер лота акции (штук в одном биржевом лоте). РАЗНЫЙ у бумаг."""
        return int(self._inst[ticker].get("lot") or 1) or 1

    def leg_uids(self) -> tuple[str, str]:
        """uid обеих ног — для запроса real-time данных T-Bank при желании."""
        return (self.sb._uid(self._inst[self.ticker_a]),
                self.sb._uid(self._inst[self.ticker_b]))

    def is_tradable(self) -> bool:
        """Торгуются ли ОБЕ ноги сейчас (гейт неторгового времени). Обе должны быть доступны —
        парный ордер атомарен. Не смогли проверить — не блокируем (как в st4)."""
        try:
            self._resolve_instruments()
            ua = self.sb._uid(self._inst[self.ticker_a])
            ub = self.sb._uid(self._inst[self.ticker_b])
            return self.sb.is_tradable(ua) and self.sb.is_tradable(ub)
        except Exception:  # noqa: BLE001
            return True

    def _account(self) -> str:
        """Переиспользовать sandbox-счёт (conn.account_id / по имени) или открыть новый."""
        accs = self.sb.list_accounts()
        if self.conn.account_id:
            for a in accs:
                if (a.get("id") == self.conn.account_id
                        and a.get("status") == "ACCOUNT_STATUS_OPEN"):
                    return self.conn.account_id
        for a in accs:
            if (a.get("name") == self.conn.account_name
                    and a.get("status") == "ACCOUNT_STATUS_OPEN"):
                return a["id"]
        return self.sb.open_account(self.conn.account_name)

    def _ensure_started(self) -> None:
        """Резолв инструментов + счёт + пополнение. Ошибки T-Bank пробрасываем наверх
        (service ловит и откатывает в paper)."""
        self._resolve_instruments()
        self._account_id = self._account()
        self.conn.account_id = self._account_id
        self.sb.pay_in(self._account_id, self.conn.payin_rub)

    # ---------- одна нога ----------
    def _post_leg(self, ticker: str, side: str, shares: int) -> LegFill | None:
        """Поставить ОДНУ market-ногу. shares — ШТУК бумаги (конвертируем в лоты).

        None — нога не исполнилась (реджект/частичный/нулевой объём). Лоты = штуки // lot;
        если штук меньше лота — заявку не ставим (нечего покупать целым лотом).
        """
        lot = self._lot_size(ticker)
        lots = int(shares) // lot
        if lots <= 0:
            return None
        it = self._inst[ticker]
        direction = "ORDER_DIRECTION_BUY" if side == "buy" else "ORDER_DIRECTION_SELL"
        resp = self.sb.post_order(self._account_id, self.sb._uid(it), lots, direction,
                                  str(uuid.uuid4()), order_type="ORDER_TYPE_MARKET")
        status = resp.get("executionReportStatus")
        # executedOrderPrice — СУММА за все лоты. Делим на исполненные штуки → цена за бумагу.
        executed_lots = int(resp.get("lotsExecuted") or lots) or lots
        executed_shares = executed_lots * lot
        avg = self.sb._q_to_float(resp.get("executedOrderPrice")) / executed_shares
        if status != _FILL_OK or avg <= 0:
            return None
        return LegFill(ticker=ticker, side=side, lots=executed_lots,
                       shares=executed_shares, avg_price=avg)

    def _retry_leg(self, ticker: str, side: str, shares: int) -> LegFill | None:
        for _ in range(self.max_retries):
            f = self._post_leg(ticker, side, shares)
            if f is not None:
                return f
        return None

    def _unwind(self, leg: LegFill) -> bool:
        """Закрыть уже открытую ногу обратным market-ордером (по штукам)."""
        close_side = "sell" if leg.side == "buy" else "buy"
        f = self._post_leg(leg.ticker, close_side, leg.shares)
        return f is not None

    # ---------- вход (атомарность/unwind) ----------
    def execute_pair(self, side: Side, qty_a: int, qty_b: int,
                     ref_a: float, ref_b: float) -> PairFillResult:
        """Атомарный парный вход в песочнице (структура как st4 execute_pair).

        side — направление спреда (LONG_SPREAD = long A / short B; SHORT_SPREAD наоборот).
        qty_a/qty_b — ШТУК каждой ноги (из core.leg_quantities). Менее ликвидную ногу
        исполняем первой — для акций это нога с БОЛЬШИМ лотом (NLMK lot=10 ликвиднее по
        деньгам, но крупный лот = меньше гранулярность → ставим первой), при равных лотах —
        нога A. ref_* пока не используются (market-ордер) — приняты для совместимости сигнатуры.
        """
        if side == Side.LONG_SPREAD:
            side_a, side_b = "buy", "sell"
        elif side == Side.SHORT_SPREAD:
            side_a, side_b = "sell", "buy"
        else:
            return PairFillResult(ok=False, aborted=True, reason="FLAT — нет направления входа")

        # «первой» ставим ногу с бо́льшим размером лота (грубее гранулярность → выше риск
        # частичного несоответствия); при равенстве — A.
        first_is_a = self._lot_size(self.ticker_a) >= self._lot_size(self.ticker_b)
        if first_is_a:
            first = (self.ticker_a, side_a, qty_a)
            second = (self.ticker_b, side_b, qty_b)
        else:
            first = (self.ticker_b, side_b, qty_b)
            second = (self.ticker_a, side_a, qty_a)

        f1 = self._retry_leg(*first)
        if f1 is None:
            return PairFillResult(ok=False, aborted=True,
                                  reason="первая нога (sandbox) не исполнилась — вход отменён")
        f2 = self._retry_leg(*second)
        if f2 is None:
            if not self._unwind(f1):
                raise UnwindError("вторая нога sandbox не залилась, unwind первой не удался")
            return PairFillResult(ok=False, unwound=True,
                                  reason="вторая нога (sandbox) не исполнилась — первая закрыта (unwind)")

        fa = f1 if f1.ticker == self.ticker_a else f2
        fb = f2 if f1.ticker == self.ticker_a else f1
        return PairFillResult(ok=True, fill_a=fa, fill_b=fb)

    # ---------- выход (реальный обратный ордер) ----------
    def close_pair(self, pos: Position) -> PairCloseResult:
        """Реальный выход: обратные market-ордера по обеим ногам.

        Закрываем по фактически открытым штукам (lot-кратным). LONG_SPREAD: A была buy →
        sell, B была sell → buy; SHORT_SPREAD — зеркально. UnwindError, если нога не закрылась.
        """
        if pos.side == Side.LONG_SPREAD:
            close_a, close_b = "sell", "buy"
        else:
            close_a, close_b = "buy", "sell"
        fa = self._retry_leg(self.ticker_a, close_a, pos.qty_a)
        fb = self._retry_leg(self.ticker_b, close_b, pos.qty_b)
        if fa is None or fb is None:
            raise UnwindError("не удалось закрыть ногу акции в sandbox при выходе (голая позиция)")
        return PairCloseResult(exit_a=fa.avg_price, exit_b=fb.avg_price)

    # ---------- reconciliation ----------
    def broker_lots(self) -> dict[str, int]:
        """Фактические ШТУКИ ног на sandbox-счёте (securities по uid). Для сверки на старте.

        Знак: + лонг, − шорт. Ключи — тикеры пары. Акции лежат в positions().securities.
        """
        ua = self.sb._uid(self._inst[self.ticker_a])
        ub = self.sb._uid(self._inst[self.ticker_b])
        out = {self.ticker_a: 0, self.ticker_b: 0}
        pos = self.sb.positions(self._account_id)
        for s in pos.get("securities", []):
            uid = s.get("instrumentUid", "")
            bal = int(s.get("balance", 0))
            if uid == ua:
                out[self.ticker_a] = bal
            elif uid == ub:
                out[self.ticker_b] = bal
        return out

    def flat_broker(self) -> bool:
        """Закрыть ВСЕ реальные позиции пары на sandbox-счёте по рынку (привести к FLAT)."""
        shares = self.broker_lots()
        ok = True
        for ticker, bal in shares.items():
            if bal == 0:
                continue
            side = "sell" if bal > 0 else "buy"
            f = self._retry_leg(ticker, side, abs(bal))
            ok = ok and (f is not None)
        return ok
