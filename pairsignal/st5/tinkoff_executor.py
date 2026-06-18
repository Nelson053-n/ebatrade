"""Реальный (sandbox) исполнитель ОДИНОЧНОГО инструмента через T-Bank Invest API.

St5SandboxExecutor — аналог st4.TinkoffSandboxExecutor, но упрощённый под ОДНУ ногу
(st5 — directional momentum на одиночном фьючерсе FORTS: SRU6 Сбер / GZU6 Газпром).
Ставит market-ордера в ПЕСОЧНИЦЕ T-Bank через st4.tbank_sandbox (общий REST + общий токен).
Engine не знает про figi/uid/счёт — вся реальная идентификация инкапсулирована здесь.

БЕЗОПАСНОСТЬ: ходит ИСКЛЮЧИТЕЛЬНО через функции tbank_sandbox (только SandboxService.*).
Боевой OrdersService не импортируется — реальный ордер отправить нельзя. Токен — из
окружения процесса (env TBANK_TOKEN, общий с st4), не на диске у st5.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from ..st4 import tbank_sandbox            # ПЕРЕИСПОЛЬЗУЕМ низкоуровневый REST st4
from .config import ConnectorConfig, ExecutionConfig
from .models import InstrumentSpec, Position

_FILL_OK = "EXECUTION_REPORT_STATUS_FILL"


@dataclass(slots=True)
class Fill:
    """Факт исполнения одиночного market-ордера в песочнице."""
    side: str                  # "buy" | "sell"
    lots: int                  # фактически исполненные лоты
    avg_price: float           # средняя цена за КОНТРАКТ (не сумма за лоты)
    reference_price: float     # опорная цена (close бара) для расчёта проскальзывания
    slippage_ticks: float


class UnwindError(RuntimeError):
    """Не удалось закрыть реальную позицию в песочнице (голая позиция) → HALTED."""


class St5SandboxExecutor:
    """Исполнитель одиночного фьючерса st5 в песочнице T-Bank (market-ордера)."""

    def __init__(self, exec_cfg: ExecutionConfig, conn_cfg: ConnectorConfig,
                 spec: InstrumentSpec, sb=None) -> None:
        self.cfg = exec_cfg
        self.conn = conn_cfg
        self.spec = spec
        # sb — модуль-зависимость (инъекция для моков). По умолчанию реальный tbank_sandbox.
        self.sb = sb if sb is not None else tbank_sandbox
        self._account_id: str = ""
        self._inst: dict | None = None      # кэш записи инструмента из справочника
        self._ensure_started()

    # ---------- инициализация ----------
    def _resolve_instrument(self) -> None:
        """Найти фьючерс в справочнике T-Bank по тикеру (= SECID FORTS = spec.code)."""
        if self._inst is None:
            self._inst = self.sb.find_future(self.spec.code)

    def leg_uid(self) -> str:
        """uid инструмента для запроса real-time свечей T-Bank."""
        self._resolve_instrument()
        return self.sb._uid(self._inst)

    def is_tradable(self) -> bool:
        """Торгуется ли инструмент сейчас (для гейта неторгового времени в движке)."""
        try:
            return self.sb.is_tradable(self.leg_uid())
        except Exception:  # noqa: BLE001  не смогли проверить — не блокируем
            return True

    def _account(self) -> str:
        """Переиспользовать sandbox-счёт (conn.account_id / по имени) или открыть новый."""
        accs = self.sb.list_accounts()
        if self.conn.account_id:
            for a in accs:
                if a.get("id") == self.conn.account_id and a.get("status") == "ACCOUNT_STATUS_OPEN":
                    return self.conn.account_id
        for a in accs:
            if a.get("name") == self.conn.account_name and a.get("status") == "ACCOUNT_STATUS_OPEN":
                return a["id"]
        return self.sb.open_account(self.conn.account_name)

    def _ensure_started(self) -> None:
        """Резолв инструмента + счёт + пополнение под ГО. Ошибки T-Bank пробрасываем наверх."""
        self._resolve_instrument()
        self._account_id = self._account()
        self.conn.account_id = self._account_id    # запомнить для сериализации/переиспользования
        self.sb.pay_in(self._account_id, self.conn.payin_rub)

    # ---------- одна нога ----------
    def _post(self, side: str, lots: int, ref: float) -> Fill | None:
        """Поставить ОДИН market-ордер в песочнице. None — не исполнился (реджект/частичный)."""
        self._resolve_instrument()
        direction = "ORDER_DIRECTION_BUY" if side == "buy" else "ORDER_DIRECTION_SELL"
        resp = self.sb.post_order(self._account_id, self.sb._uid(self._inst), lots, direction,
                                  str(uuid.uuid4()), order_type="ORDER_TYPE_MARKET")
        status = resp.get("executionReportStatus")
        # executedOrderPrice — СУММА за все лоты (не цена за контракт!). Делим на исполненные
        # лоты, иначе при lots>1 цена входа завышается в N раз → искажённый P&L и стопы.
        executed = int(resp.get("lotsExecuted") or lots) or lots
        avg = self.sb._q_to_float(resp.get("executedOrderPrice")) / executed
        if status != _FILL_OK or avg <= 0:
            return None
        slip = (avg - ref) / self.spec.tick_size * (1 if side == "buy" else -1)
        return Fill(side=side, lots=executed, avg_price=avg, reference_price=ref,
                    slippage_ticks=slip)

    def _retry(self, side: str, lots: int, ref: float) -> Fill | None:
        for _ in range(getattr(self.cfg, "max_retries", 3)):
            f = self._post(side, lots, ref)
            if f is not None:
                return f
        return None

    # ---------- вход ----------
    def execute(self, side: str, lots: int, ref_price: float) -> Fill:
        """Вход: реальный market-ордер в песочнице. UnwindError если не исполнился."""
        f = self._retry(side, lots, ref_price)
        if f is None:
            raise UnwindError("ордер входа (sandbox) не исполнился за все попытки")
        return f

    # ---------- выход ----------
    def close(self, pos: Position, ref_price: float) -> Fill:
        """Выход: обратный market-ордер. UnwindError если не закрылся (голая позиция)."""
        close_side = "sell" if pos.side == "buy" else "buy"
        f = self._retry(close_side, pos.lots, ref_price)
        if f is None:
            raise UnwindError("не удалось закрыть позицию в sandbox при выходе (голая позиция)")
        return f

    # ---------- reconciliation ----------
    def broker_lots(self) -> int:
        """Фактический баланс инструмента на sandbox-счёте T-Bank (знак = направление).
        + длинная (buy), − короткая (sell). Для сверки на старте."""
        uid = self.leg_uid()
        for f in self.sb.positions(self._account_id).get("futures", []):
            if f.get("instrumentUid", "") == uid:
                return int(f.get("balance", 0))
        return 0

    def flat_broker(self) -> bool:
        """Закрыть реальную позицию на sandbox-счёте по рынку (привести к FLAT).

        Для устранения рассинхрона на старте: если на счёте висит нога, а движок её не знает.
        """
        bal = self.broker_lots()
        if bal == 0:
            return True
        side = "sell" if bal > 0 else "buy"   # закрыть в противоположную сторону
        return self._retry(side, abs(bal), 0.0) is not None
