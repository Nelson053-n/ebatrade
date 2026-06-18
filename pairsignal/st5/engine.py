"""Движок st5 — FSM directional momentum на одиночном инструменте + paper-исполнение.

Принимает закрытые OHLCV-бары → гоняет MomentumIndicator (close vs close[-lookback]) →
вход по направлению тренда → исполнение одиночного ордера (paper) → выход по holding /
ранний стоп (stop_pct) / конец сессии. Без атомарных пар (один инструмент → один ордер).
FSM: FLAT → (сигнал) → LONG/SHORT → (holding/стоп/EOD) → FLAT.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

from .config import St5Config
from .indicators import MomentumIndicator, VolumeAverage
from .models import (
    BotState,
    EngineEvent,
    InstrumentSpec,
    MomentumReading,
    Position,
    PriceBar,
    Signal,
    Trade,
)
from .strategy import entry_signal, exit_signal, in_clearing_window, is_session_end


@dataclass
class StepResult:
    state: BotState
    reading: Optional[MomentumReading] = None
    signal: Signal = Signal.NONE
    trade: Optional[Trade] = None
    awaiting_approval: bool = False
    events: list[EngineEvent] = field(default_factory=list)


def _pnl_rub(side: str, entry: float, exit_price: float, lots: int, spec: InstrumentSpec) -> float:
    """P&L одиночной позиции: (exit−entry)·dir·lots·(STEPPRICE/MINSTEP)."""
    direction = 1.0 if side == "buy" else -1.0
    return (exit_price - entry) * direction * lots * (spec.tick_value_rub / spec.tick_size)


class RiskManager:
    """Лимиты, kill-switch, дневной P&L (учитывает нереализованный)."""

    def __init__(self, cfg, session) -> None:
        self.cfg = cfg
        self.session = session
        self.consecutive_errors = 0
        self._day = ""
        self.day_pnl_rub = 0.0
        self.halted = False
        self.halt_reason = ""

    def _day_key(self, ts_ms: int) -> str:
        from zoneinfo import ZoneInfo
        try:
            local = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).astimezone(
                ZoneInfo(self.session.timezone))
        except Exception:  # noqa: BLE001
            local = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        return local.strftime("%Y-%m-%d")

    def on_trade_closed(self, net_pnl_rub: float, ts_ms: int) -> None:
        day = self._day_key(ts_ms)
        if day != self._day:
            self._day = day
            self.day_pnl_rub = 0.0
        self.day_pnl_rub += net_pnl_rub

    def day_loss_breached(self, ts_ms: int, unrealized_rub: float = 0.0) -> bool:
        realized = self.day_pnl_rub if self._day_key(ts_ms) == self._day else 0.0
        return realized + min(0.0, unrealized_rub) <= -self.cfg.max_daily_loss_rub

    def can_enter(self, ts_ms: int) -> tuple[bool, str]:
        if self.halted:
            return False, f"HALTED: {self.halt_reason}"
        if not self.cfg.trading_enabled:
            return False, "торговля выключена оператором"
        if self._day_key(ts_ms) == self._day and self.day_pnl_rub <= -self.cfg.max_daily_loss_rub:
            return False, f"дневной лимит убытка {self.cfg.max_daily_loss_rub:.0f}₽"
        return True, ""

    def on_error(self) -> None:
        self.consecutive_errors += 1
        if self.consecutive_errors >= self.cfg.max_consecutive_errors:
            self.halt(f"серия ошибок ≥ {self.cfg.max_consecutive_errors}")

    def on_success(self) -> None:
        self.consecutive_errors = 0

    def halt(self, reason: str) -> None:
        self.halted = True
        self.halt_reason = reason

    def resume(self) -> None:
        self.halted = False
        self.halt_reason = ""
        self.consecutive_errors = 0


class TradingEngine:
    def __init__(self, cfg: St5Config, spec: InstrumentSpec) -> None:
        self.cfg = cfg
        self.spec = spec
        self.momentum = MomentumIndicator(cfg.strategy.lookback)
        self.volavg = VolumeAverage()
        self.risk = RiskManager(cfg.risk, cfg.session)

        self.state = BotState.FLAT
        self.position: Optional[Position] = None
        self.trades: list[Trade] = []
        self.balance_rub = cfg.paper.start_balance_rub
        self._bars_held = 0
        self._pending: Optional[tuple[Signal, MomentumReading]] = None
        self.last_reading: Optional[MomentumReading] = None
        self._last_bar: Optional[PriceBar] = None
        self._last_vol_avg = float("nan")
        self._armed = True
        self._check_lag = False         # гейт свежести только в live

    def arm(self, on: bool) -> None:
        self._armed = on

    # ---------- подача данных ----------
    def on_bar(self, bar: PriceBar) -> StepResult:
        return self.step(bar)

    def run_df(self, df) -> list[StepResult]:
        """Прогон по DataFrame (open/high/low/close/volume) — бэктест/плеер/смоук."""
        out: list[StepResult] = []
        for ts, row in df.iterrows():
            bar = PriceBar(int(ts), float(row["open"]), float(row["high"]), float(row["low"]),
                           float(row["close"]), float(row.get("volume", 0.0)))
            res = self.step(bar)
            if res.events or res.trade or res.awaiting_approval:
                out.append(res)
        return out

    def warmup(self, bars: list[PriceBar]) -> None:
        """Прогрев momentum-буфера историей (без сигналов): прокручиваем индикатор."""
        for b in bars:
            self.momentum.update(b.close)
            self.volavg.update(b.ts, b.volume)

    def days_to_expiry(self, ts_ms: int) -> Optional[int]:
        if not self.spec.expiry:
            return None
        try:
            exp = date.fromisoformat(self.spec.expiry)
        except ValueError:
            return None
        cur = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).date()
        return (exp - cur).days

    # ---------- основной шаг FSM ----------
    def step(self, bar: PriceBar) -> StepResult:
        self._last_bar = bar
        reading = self.momentum.update(bar.close)
        reading.ts = bar.ts
        self._last_vol_avg = self.volavg.update(bar.ts, bar.volume)
        self.last_reading = reading
        events: list[EngineEvent] = []

        if self.state == BotState.HALTED:
            return StepResult(state=self.state, reading=reading, events=events)

        if self.position is not None:
            self._bars_held += 1

        # ---- принудительное закрытие к концу сессии (овернайт-риск) ----
        if (self.position is not None and self.cfg.strategy.flat_at_session_end
                and is_session_end(bar.ts, self.cfg.session)):
            trade = self._close_position(bar, "eod")
            events.append(EngineEvent(bar.ts, "exit",
                          f"конец сессии — закрытие (net {trade.net_pnl_rub:+.0f}₽)", {}))
            return StepResult(state=self.state, reading=reading, signal=Signal.EXIT,
                              trade=trade, events=events)

        # ---- роллировер: позиция не доживает до экспирации ----
        if self.position is not None:
            d2e = self.days_to_expiry(bar.ts)
            if d2e is not None and d2e < self.cfg.instrument.rollover_days_before_expiry:
                trade = self._close_position(bar, "rollover")
                events.append(EngineEvent(bar.ts, "exit",
                              f"роллировер: до экспирации {d2e} дн (net {trade.net_pnl_rub:+.0f}₽)", {}))
                return StepResult(state=self.state, reading=reading, signal=Signal.EXIT,
                                  trade=trade, events=events)

        # ---- дневной kill-switch (реализованный + нереализованный) ----
        if self.position is not None and self.risk.day_loss_breached(bar.ts, self.unrealized_rub()):
            trade = self._close_position(bar, "stop")
            self.risk.halt(f"дневной лимит убытка {self.cfg.risk.max_daily_loss_rub:.0f}₽")
            self.state = BotState.HALTED
            events.append(EngineEvent(bar.ts, "halt",
                          f"дневной лимит: позиция закрыта (net {trade.net_pnl_rub:+.0f}₽), HALTED", {}))
            return StepResult(state=self.state, reading=reading, signal=Signal.EXIT,
                              trade=trade, events=events)

        # ---- управление открытой позицией (выход — авто): holding / стоп ----
        if self.position is not None:
            should_exit, reason = exit_signal(
                self.position, self._bars_held, bar.close,
                self.cfg.strategy.holding, self.cfg.strategy.stop_pct)
            if should_exit:
                trade = self._close_position(bar, reason)
                label = "стоп" if reason == "stop" else "holding"
                events.append(EngineEvent(bar.ts, "exit",
                              f"выход ({label}): net {trade.net_pnl_rub:+.0f}₽", {}))
                return StepResult(state=self.state, reading=reading, signal=Signal.EXIT,
                                  trade=trade, events=events)

        # ---- поиск входа ----
        if self.position is None and self._armed and reading.is_ready:
            sig = entry_signal(reading)
            if sig != Signal.NONE:
                d2e = self.days_to_expiry(bar.ts)
                no_entry = self.cfg.instrument.rollover_no_new_entry_days_before
                exec_ts = bar.ts + self.cfg.strategy.candle_interval_minutes * 60_000
                if d2e is not None and d2e < no_entry:
                    events.append(EngineEvent(bar.ts, "warn",
                                  f"сигнал пропущен: до экспирации {d2e} дн (< {no_entry})", {}))
                elif in_clearing_window(exec_ts, self.cfg.session):
                    events.append(EngineEvent(bar.ts, "warn", "сигнал в клиринге — пропуск", {}))
                elif self.cfg.strategy.flat_at_session_end and is_session_end(exec_ts, self.cfg.session):
                    events.append(EngineEvent(bar.ts, "warn", "конец сессии — не входим", {}))
                elif not self._volume_ok(bar):
                    events.append(EngineEvent(bar.ts, "warn", "объём бара ниже порога — пропуск", {}))
                elif not self._data_fresh(bar):
                    events.append(EngineEvent(bar.ts, "warn", "данные устарели — пропуск входа", {}))
                else:
                    ok, why = self.risk.can_enter(bar.ts)
                    if not ok:
                        events.append(EngineEvent(bar.ts, "warn", f"вход запрещён: {why}", {}))
                    elif self.cfg.auto_approve:
                        return self._open_position(sig, reading, bar, events)
                    else:
                        self._pending = (sig, reading)
                        self.state = BotState.ENTERING
                        events.append(EngineEvent(bar.ts, "signal",
                                      f"сигнал {sig.value.upper()} — ждёт подтверждения", {}))
                        return StepResult(state=self.state, reading=reading, signal=sig,
                                          awaiting_approval=True, events=events)

        return StepResult(state=self.state, reading=reading, events=events)

    # ---------- фильтры входа ----------
    def _volume_ok(self, bar: PriceBar) -> bool:
        mult = self.cfg.strategy.volume_filter_mult
        if mult <= 0 or bar.volume <= 0:
            return True
        avg = self._last_vol_avg
        if avg is None or math.isnan(avg) or avg <= 0:
            return True
        return bar.volume >= mult * avg

    def _data_fresh(self, bar: PriceBar) -> bool:
        lag = self.cfg.strategy.max_data_lag_min
        if not self._check_lag or lag <= 0:
            return True
        return (time.time() * 1000 - bar.ts) <= lag * 60_000

    # ---------- human-in-the-loop ----------
    def approve(self) -> Optional[StepResult]:
        if self._pending is None or self.position is not None:
            return None
        sig, reading = self._pending
        self._pending = None
        return self._open_position(sig, reading, self._last_bar, [])

    def reject(self) -> None:
        self._pending = None
        if self.state == BotState.ENTERING:
            self.state = BotState.FLAT

    # ---------- исполнение (paper, одиночный ордер) ----------
    def _fill_price(self, side: str, ref: float) -> float:
        """Paper-филл: marketable-limit за полспреда книги + offset (как st4)."""
        cost = (self.cfg.execution.paper_book_halfspread_ticks
                + self.cfg.execution.tick_offset) * self.spec.tick_size
        return ref + cost if side == "buy" else ref - cost

    def _open_position(self, sig: Signal, reading: MomentumReading, bar: PriceBar,
                       events: list[EngineEvent]) -> StepResult:
        self.state = BotState.ENTERING
        lots = self.cfg.execution.quantity_lots
        side = "buy" if sig == Signal.BUY else "sell"
        fill = self._fill_price(side, bar.close)
        slip = abs(fill - bar.close) / self.spec.tick_size
        fee = self.cfg.paper.taker_fee_rub_per_lot * lots
        self.balance_rub -= fee
        self.risk.on_success()
        new_state = BotState.LONG if sig == Signal.BUY else BotState.SHORT
        # entry_vwap (имя поля персиста) = close[-lookback] на входе — база momentum
        ref = reading.ref_price if not math.isnan(reading.ref_price) else fill
        self.position = Position(state=new_state, side=side, lots=lots, entry_price=fill,
                                 entry_ts=bar.ts, entry_vwap=ref, entry_fee_rub=fee)
        self._entry_slip = slip
        self.state = new_state
        self._bars_held = 0
        events.append(EngineEvent(bar.ts, "position",
                      f"вход {new_state.value}: {side} @ {fill:.0f} "
                      f"(momentum {reading.lookback_return * 100:+.2f}%)", {}))
        return StepResult(state=self.state, reading=reading, signal=sig, events=events)

    def _close_position(self, bar: PriceBar, reason: str) -> Trade:
        p = self.position
        close_side = "sell" if p.side == "buy" else "buy"
        fill = self._fill_price(close_side, bar.close)
        gross = _pnl_rub(p.side, p.entry_price, fill, p.lots, self.spec)
        exit_fee = self.cfg.paper.taker_fee_rub_per_lot * p.lots
        net = gross - p.entry_fee_rub - exit_fee
        self.balance_rub += gross - exit_fee
        trade = Trade(
            state=p.state, entry_ts=p.entry_ts, exit_ts=bar.ts,
            entry_price=p.entry_price, exit_price=fill, lots=p.lots,
            gross_pnl_rub=gross, fees_rub=p.entry_fee_rub + exit_fee, net_pnl_rub=net,
            reason=reason, bars_held=self._bars_held, side=p.side,
            slippage_ticks=getattr(self, "_entry_slip", 0.0),
        )
        self.trades.append(trade)
        self.risk.on_trade_closed(net, bar.ts)
        self.position = None
        self.state = BotState.FLAT
        self._bars_held = 0
        return trade

    def flat_all(self, reason: str = "flat_all") -> Optional[Trade]:
        self._pending = None
        if self.position is None or self._last_bar is None:
            if self.state != BotState.HALTED:
                self.state = BotState.FLAT
            return None
        return self._close_position(self._last_bar, reason)

    # ---------- сводка ----------
    def unrealized_rub(self) -> float:
        if self.position is None or self._last_bar is None:
            return 0.0
        return _pnl_rub(self.position.side, self.position.entry_price,
                        self._last_bar.close, self.position.lots, self.spec)

    def summary(self) -> dict:
        wins = [t for t in self.trades if t.net_pnl_rub > 0]
        net = sum(t.net_pnl_rub for t in self.trades)
        eq = self.balance_rub + self.unrealized_rub()
        start = self.cfg.paper.start_balance_rub
        return {
            "trades": len(self.trades),
            "win_rate_pct": round(100 * len(wins) / len(self.trades), 1) if self.trades else 0.0,
            "net_pnl_rub": round(net, 0),
            "fees_rub": round(sum(t.fees_rub for t in self.trades), 0),
            "balance_rub": round(self.balance_rub, 0),
            "equity_rub": round(eq, 0),
            "return_pct": round(100 * (eq - start) / start, 3),
            "day_pnl_rub": round(self.risk.day_pnl_rub, 0),
            "stops": sum(1 for t in self.trades if t.reason in ("stop", "time_stop")),
        }
