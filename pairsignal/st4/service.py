"""Сервисный слой st4: состояние торговой сессии + фоновые задачи + персистентность.

St4Session держит движок, конфиг, историю графика, потоки live/player — по аналогии со
SlotState из api.py, но для FSM-движка SBRF/SBPR. Live тянет 10m-свечи с MOEX ISS и
прогоняет новые бары; player подаёт синтетику. Состояние переживает рестарт сервера
(session_state_4.json): журнал сделок, баланс, время сессии, настройки.
"""
from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

from . import data_feed as feed
from .config import St4Config
from .engine import TradingEngine
from .models import InstrumentSpec, Role
from .tbank_sandbox import has_token as _has_token

_BASE = Path(__file__).resolve().parent.parent.parent
_SESSION_FILE = _BASE / "session_state_4.json"
_OUT_DIR = _BASE / "pairsignal" / "out"
HISTORY_LEN = 300
EVENTS_LEN = 40             # сколько последних действий держим в журнале бота
BT_HISTORY_LEN = 60        # сколько прогонов бэктеста храним


def _bt_history_file(source: str) -> Path:
    """Файл истории прогонов по источнику: tbank | moex (разные периоды/данные)."""
    name = "st4_backtest_history.json" if source == "tbank" else f"st4_backtest_{source}_history.json"
    return _OUT_DIR / name


def bt_history_load(source: str = "tbank") -> list[dict]:
    """История прогонов бэктеста по источнику (для отслеживания результативности во времени)."""
    f = _bt_history_file(source)
    try:
        if f.exists():
            return json.loads(f.read_text())
    except Exception:  # noqa: BLE001
        pass
    return []


def bt_history_append(entry: dict, source: str = "tbank") -> list[dict]:
    """Добавить прогон в историю источника (дедуп по дню+stop_sigma — не плодим за один день)."""
    hist = bt_history_load(source)
    day = entry.get("date", "")[:10]
    ss = entry.get("stop_sigma")
    hist = [h for h in hist if not (h.get("date", "")[:10] == day and h.get("stop_sigma") == ss)]
    hist.append(entry)
    hist = hist[-BT_HISTORY_LEN:]
    f = _bt_history_file(source)
    try:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(hist))
    except Exception:  # noqa: BLE001
        pass
    return hist


def _clean(obj):
    """NaN/inf → None (JSON их не допускает)."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


class St4Session:
    """Полное состояние торговой сессии st4 (один экземпляр на процесс)."""

    def __init__(self) -> None:
        self.cfg = St4Config()
        self.spec_ord: InstrumentSpec = feed.synthetic_spec(Role.ORDINARY)
        self.spec_pref: InstrumentSpec = feed.synthetic_spec(Role.PREFERRED)
        self.engine = TradingEngine(self.cfg, self.spec_ord, self.spec_pref)
        self.state = {"live": False, "player": False, "session_started": None,
                      "paused_by_user": False, "last_event": None,
                      "data_source": "synthetic", "warmup_done": False,
                      "sandbox_active": False, "trade_start_ts": None}
        self.history: list[dict] = []          # бары спреда + полосы для графика
        self.events: list[dict] = []           # кольцевой журнал действий (последние EVENTS_LEN)
        self.player_df = None
        self.player_idx = 0
        self.last_live_ts = 0
        self._lock = asyncio.Lock()            # сериализация шагов движка между потоками

    # ---------- инструменты ----------
    def resolve_real_legs(self) -> None:
        """Определить реальные серии SBRF/SBPR (роллировер) и их спецификации."""
        self.spec_ord, self.spec_pref = feed.resolve_legs(self.cfg)

    def reset_engine(self, real: bool = False) -> None:
        if real:
            try:
                self.resolve_real_legs()
            except Exception:  # noqa: BLE001  оффлайн — остаёмся на синтетических спеках
                self.spec_ord = feed.synthetic_spec(Role.ORDINARY)
                self.spec_pref = feed.synthetic_spec(Role.PREFERRED)
        # cfg.connector.mode — НАМЕРЕНИЕ оператора (не трогаем). sandbox_active — ФАКТ:
        # активен ли реальный исполнитель сейчас. Sandbox активируется только в live (real=True);
        # на синтетике движок строим как paper (рыночные ордера по выдуманным барам бессмысленны).
        want_sandbox = self.cfg.connector.mode == "tbank_sandbox"
        self.state["sandbox_active"] = False
        self.state["sandbox_error"] = None
        if want_sandbox and real:
            try:
                self.engine = TradingEngine(self.cfg, self.spec_ord, self.spec_pref)
                self.state["sandbox_active"] = True
                # RECONCILIATION (§11): движок стартует FLAT, но на sandbox-счёте могут висеть
                # позиции от прошлой сессии (рестарт прервал стратегию). Приводим счёт к FLAT,
                # чтобы не было рассинхрона «движок FLAT / на счёте позиция».
                try:
                    ex = self.engine.executor
                    lots = ex.broker_lots()
                    if any(v != 0 for v in lots.values()):
                        self.log_event("warn", f"reconciliation: на sandbox-счёте висят позиции "
                                       f"{dict((k.value, v) for k, v in lots.items())} — закрываю")
                        if ex.flat_broker():
                            self.log_event("info", "reconciliation: счёт приведён к FLAT")
                        else:
                            self.log_event("warn", "reconciliation: не удалось закрыть все ноги")
                except Exception as e:  # noqa: BLE001  сверка не должна ронять старт
                    self.log_event("warn", f"reconciliation пропущена: {e}")
            except Exception as e:  # noqa: BLE001  sandbox недоступен → откат в paper-движок
                # частый случай — HTTP 401 (токен невалиден/отозван): сообщаем явно
                msg = str(e)
                if "401" in msg:
                    msg = "токен T-Bank невалиден или отозван (HTTP 401) — выпустите новый"
                self.state["sandbox_error"] = msg
                self.log_event("warn", f"sandbox не активирован: {msg} → исполнение paper")
                paper_cfg = self.cfg.model_copy(deep=True)
                paper_cfg.connector.mode = "paper"
                self.engine = TradingEngine(paper_cfg, self.spec_ord, self.spec_pref)
        else:
            # синтетика или paper-намерение: строим paper-движок (не трогая cfg.connector.mode)
            paper_cfg = self.cfg
            if want_sandbox:
                paper_cfg = self.cfg.model_copy(deep=True)
                paper_cfg.connector.mode = "paper"
                self.state["last_event"] = "sandbox активен только в live — на синтетике paper"
            self.engine = TradingEngine(paper_cfg, self.spec_ord, self.spec_pref)
        self.history = []
        self.player_df = None
        self.player_idx = 0
        self.last_live_ts = 0
        self.state["last_event"] = None
        self.state["warmup_done"] = False
        self.state["trade_start_ts"] = None    # новая сессия — линия старта сбросится
        self.state["session_started"] = time.time()
        self.save_session()

    def log_event(self, kind: str, message: str) -> None:
        """Записать действие в кольцевой журнал бота (+ last_event для совместимости)."""
        self.events.append({"ts": time.time(), "kind": kind, "message": message})
        if len(self.events) > EVENTS_LEN:
            del self.events[0]
        self.state["last_event"] = message

    def push_history(self, ts: int) -> None:
        """Добавить срез спреда/полос текущего бара в историю графика."""
        b = self.engine.last_band
        if b is None or not b.is_ready:
            return
        if any(v is None or (isinstance(v, float) and math.isnan(v))
               for v in (b.spread, b.sma, b.upper, b.lower)):
            return
        self.history.append({"ts": ts, "spread": round(b.spread, 1), "sma": round(b.sma, 1),
                             "upper": round(b.upper, 1), "lower": round(b.lower, 1),
                             "sigma": round(b.sigma, 1)})
        if len(self.history) > HISTORY_LEN:
            del self.history[0]

    def warmup_limit(self) -> int:
        """Сколько баров тянуть, чтобы прогреть BB(sma_period) + дать запас на сигналы."""
        return int(self.cfg.strategy.sma_period * 1.5) + 120

    # ---------- персистентность ----------
    def save_session(self) -> None:
        try:
            data = {
                "session_started": self.state["session_started"],
                "balance_rub": self.engine.balance_rub,
                "trades": [self._trade_json(t) for t in self.engine.trades],
                "history": self.history,
                "config": self.cfg.model_dump(),
                "spec_ord": asdict(self.spec_ord), "spec_pref": asdict(self.spec_pref),
                "day_pnl_rub": self.engine.risk.day_pnl_rub,
                "day_key": self.engine.risk._day,
            }
            _SESSION_FILE.write_text(json.dumps(_clean(data)))
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _trade_json(t) -> dict:
        d = asdict(t)
        d["state"] = t.state.value
        return d

    def load_session(self) -> bool:
        if not _SESSION_FILE.exists():
            return False
        try:
            data = json.loads(_SESSION_FILE.read_text())
        except Exception:  # noqa: BLE001
            return False
        try:
            for k, v in (data.get("config") or {}).items():
                if hasattr(self.cfg, k) and isinstance(v, dict):
                    sub = getattr(self.cfg, k)
                    for kk, vv in v.items():
                        if hasattr(sub, kk):
                            setattr(sub, kk, vv)
            so, sp = data.get("spec_ord"), data.get("spec_pref")
            if so:
                self.spec_ord = InstrumentSpec(role=Role(so["role"]),
                                               **{k: v for k, v in so.items() if k != "role"})
            if sp:
                self.spec_pref = InstrumentSpec(role=Role(sp["role"]),
                                                **{k: v for k, v in sp.items() if k != "role"})
            # восстановление — всегда paper-движок: sandbox-исполнитель в конструкторе
            # делает сетевые вызовы (счёт + pay_in). Sandbox активируется только через
            # reset_engine(real=True) при старте live.
            cfg = self.cfg
            if cfg.connector.mode == "tbank_sandbox":
                cfg = self.cfg.model_copy(deep=True)
                cfg.connector.mode = "paper"
            self.engine = TradingEngine(cfg, self.spec_ord, self.spec_pref)
            self.engine.balance_rub = data.get("balance_rub", self.engine.balance_rub)
            self.engine.risk.day_pnl_rub = data.get("day_pnl_rub", 0.0)
            self.engine.risk._day = data.get("day_key", "")
            self.engine.trades = [self._trade_from_json(t) for t in data.get("trades", [])]
            self.history = data.get("history", [])
            self.state["session_started"] = data.get("session_started", time.time())
            return True
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _trade_from_json(d: dict):
        from .models import BotState, Trade
        d = dict(d)
        d["state"] = BotState(d["state"])
        return Trade(**d)

    # ---------- фоновые задачи ----------
    async def run_live(self) -> None:
        """Live на реальных данных MOEX ISS: backfill+replay, затем ждём новые свечи.

        Старт: тянем историю (warmup_limit баров), быстро проигрываем replay'ем (график
        оживает на настоящих котировках), прогрев BB проматываем мгновенно. Догнав до
        последней закрытой свечи — ждём новые (раз в interval). Формирующийся бар не берём.
        """
        replayed = False
        # sandbox исполняет по ТЕКУЩЕЙ цене T-Bank → на backfill-replay (исторические бары)
        # реальные ордера бессмысленны. На replay движок disarmed (только прогрев BB);
        # arm включаем после replay — торгуем с первого живого бара. Для paper всегда armed.
        sandbox = self.state.get("sandbox_active", False)
        # источник свечей: в sandbox — T-Bank (REAL-TIME, без лага ISS), иначе MOEX ISS
        # (публичный, задержка 15-30 мин). uid ног берём из sandbox-исполнителя.
        tbank_uids = None
        if sandbox and hasattr(self.engine.executor, "leg_uids"):
            try:
                tbank_uids = self.engine.executor.leg_uids()
            except Exception:  # noqa: BLE001  не удалось — откатываемся на ISS
                tbank_uids = None
        src_lbl = "T-Bank real-time" if tbank_uids else "MOEX ISS"
        mode_lbl = "T-Bank sandbox" if sandbox else "paper"
        self.log_event("info", f"live запущен ({mode_lbl}, данные {src_lbl}): "
                       f"{self.spec_ord.code}/{self.spec_pref.code}, прогрев BB({self.cfg.strategy.sma_period})…")
        while self.state["live"]:
            try:
                if tbank_uids:
                    df = await asyncio.to_thread(
                        feed.read_ohlcv_tbank, self.cfg, self.warmup_limit(),
                        tbank_uids[0], tbank_uids[1])
                    # T-Bank get_candles уже отдаёт только закрытые бары (isComplete) → не режем хвост
                else:
                    df = await asyncio.to_thread(
                        feed.read_ohlcv_moex, self.cfg, self.warmup_limit(),
                        self.spec_ord.code, self.spec_pref.code)
                    df = df.iloc[:-1]  # ISS: без формирующегося бара
                n = len(df)
                slow_from = n - 60 if not replayed else 0
                self.engine.arm(not (sandbox and not replayed))  # на replay-проходе входы закрыты
                new_bars = 0
                async with self._lock:
                    for i, (ts, row) in enumerate(df.iterrows()):
                        if not self.state["live"]:
                            return
                        ts = int(ts)
                        if ts <= self.last_live_ts:
                            continue
                        if self.engine._pending is not None:   # ждём решения оператора
                            break
                        res = self.engine.on_candles(ts, float(row["price_a"]),
                                                     float(row["price_b"]))
                        self.last_live_ts = ts
                        new_bars += 1
                        if res is not None:
                            self.push_history(ts)
                            for ev in (res.events or []):
                                if ev.kind in ("position", "exit", "halt", "warn"):
                                    self.log_event(ev.kind, ev.message)
                            if res.trade is not None:
                                self.save_session()
                        if not replayed and self.history and i >= slow_from:
                            await asyncio.sleep(0.05)
                # диагностика данных: возраст последнего бара (лаг ISS)
                lag_min = (time.time() - self.last_live_ts / 1000) / 60 if self.last_live_ts else 0
                if not replayed:
                    self.log_event("info", f"прогрев завершён: {len(self.history)} баров готово, "
                                   f"вход {'взведён' if self.engine._armed else 'ждёт живой бар'}")
                    # момент, с которого бот реально торгует по стратегии (конец backfill).
                    # для sandbox вход откроется со СЛЕДУЮЩЕГО живого бара (disarm на replay),
                    # для paper/ISS — уже armed; в обоих случаях линия = граница backfill/live.
                    self.state["trade_start_ts"] = self.last_live_ts
                if replayed and new_bars > 0:
                    self.log_event("info", f"новых баров: {new_bars} (последний {lag_min:.0f} мин назад)")
                elif replayed and lag_min > 25:
                    self.log_event("warn", f"нет свежих свечей ISS {lag_min:.0f} мин — ждём "
                                   "(ISS строит 10m-свечи с задержкой)")
            except Exception as e:  # noqa: BLE001
                self.log_event("warn", f"ошибка ISS: {e}")
            replayed = True
            self.save_session()
            await asyncio.sleep(self.cfg.poll_seconds)

    async def run_player(self) -> None:
        """Synthetic-player: подаём офлайн-бары по одному через реальный движок."""
        while self.state["player"] and self.player_df is not None \
                and self.player_idx < len(self.player_df):
            if self.engine._pending is not None:
                await asyncio.sleep(0.4)
                continue
            async with self._lock:
                ts = int(self.player_df.index[self.player_idx])
                row = self.player_df.iloc[self.player_idx]
                self.player_idx += 1
                res = self.engine.on_candles(ts, float(row["price_a"]), float(row["price_b"]))
                if res is not None:
                    self.push_history(ts)
                    if res.events:
                        self.state["last_event"] = res.events[-1].message
                    if res.trade is not None:
                        self.save_session()
            warming = not self.history
            await asyncio.sleep(0 if warming else 0.6)
        self.state["player"] = False
        self.save_session()

    # ---------- снимок состояния для API ----------
    def snapshot(self, server_started: float) -> dict:
        eng = self.engine
        pos = None
        if eng.position is not None:
            p = eng.position
            pos = {"state": p.state.value, "entry_ts": p.entry_ts,
                   "entry_spread": round(p.entry_spread, 1), "entry_beta": p.entry_beta,
                   "lots": p.leg_ord.lots, "unrealized_rub": round(eng.unrealized_rub(), 0),
                   "bars_held": eng._bars_held,
                   "legs": [
                       {"code": p.leg_ord.code, "role": "SBRF", "side": p.leg_ord.side,
                        "lots": p.leg_ord.lots, "entry": round(p.leg_ord.entry_price, 0)},
                       {"code": p.leg_pref.code, "role": "SBPR", "side": p.leg_pref.side,
                        "lots": p.leg_pref.lots, "entry": round(p.leg_pref.entry_price, 0)},
                   ]}
        pending = None
        if eng._pending is not None:
            sig, band = eng._pending
            pending = {"signal": sig.value, "spread": round(band.spread, 1),
                       "sma": round(band.sma, 1)}

        # --- диагностика: возраст данных и что бот ждёт (для понятного UI) ---
        last_bar_ts = self.last_live_ts or (self.history[-1]["ts"] if self.history else 0)
        lag_min = round((time.time() - last_bar_ts / 1000) / 60) if last_bar_ts else None
        b = eng.last_band
        cur_z = ((b.spread - b.sma) / b.sigma) if (b and b.is_ready and b.sigma > 0) else None
        # человекочитаемая причина простоя
        if eng.risk.halted:
            wait = "HALTED — нужен ручной разбор"
        elif eng.position is not None:
            wait = "в позиции — ждём возврата спреда к средней"
        elif not (self.state["live"] or self.state["player"]):
            wait = "остановлено"
        elif b is None or not b.is_ready:
            need = self.cfg.strategy.sma_period
            have = len(self.history)
            wait = f"прогрев индикатора BB({need}): {have}/{need} баров"
        elif self.state["data_source"] == "live" and lag_min is not None and lag_min > 25 \
                and not self.state.get("sandbox_active"):
            # только для ISS (задержка свечей). T-Bank real-time — этот гейт не применяем.
            wait = f"ждём свежий бар — MOEX ISS не отдаёт 10m-свечи ({lag_min} мин лаг)"
        elif cur_z is not None and abs(cur_z) < self.cfg.strategy.sigma_multiplier:
            wait = (f"ждём сигнал: спред внутри канала "
                    f"(z={cur_z:+.2f}, вход при |z|≥{self.cfg.strategy.sigma_multiplier:g})")
        else:
            wait = "ждём закрытие следующего бара"

        # честный провайдер котировок: в sandbox-live свечи из T-Bank (real-time),
        # в обычном live — MOEX ISS (с задержкой), иначе синтетика
        if self.state["data_source"] == "live":
            data_provider = "T-Bank" if self.state.get("sandbox_active") else "MOEX ISS"
        else:
            data_provider = "синтетика"
        return _clean({
            "live": self.state["live"], "player": self.state["player"],
            "data_source": self.state["data_source"],
            "data_provider": data_provider,        # откуда реально берутся свечи
            "auto_approve": self.cfg.auto_approve,
            # коннектор: mode — намерение оператора, sandbox_active — реально ли активен sandbox
            # сейчас (только в live). Сам токен никогда не отдаём, только token_set.
            "connector_mode": self.cfg.connector.mode,
            "sandbox_active": self.state.get("sandbox_active", False),
            "sandbox_error": self.state.get("sandbox_error"),   # почему sandbox не активен
            "token_set": _has_token(),
            "connector_account": self.cfg.connector.account_id or None,
            "fsm_state": eng.state.value,
            "halted": eng.risk.halted, "halt_reason": eng.risk.halt_reason,
            "trading_enabled": self.cfg.risk.trading_enabled,
            "session_started": self.state["session_started"],
            "server_started": server_started, "now": time.time(),
            "paused_by_user": self.state["paused_by_user"],
            "legs": {"ord": self.spec_ord.code, "pref": self.spec_pref.code,
                     "ord_expiry": self.spec_ord.expiry, "pref_expiry": self.spec_pref.expiry},
            "interval_min": self.cfg.strategy.candle_interval_minutes,
            "sma_period": self.cfg.strategy.sma_period,
            "sigma_mult": self.cfg.strategy.sigma_multiplier,
            "deviation_mode": self.cfg.strategy.deviation_mode,
            "deviation_pct": self.cfg.strategy.deviation_pct,
            "stop_sigma": self.cfg.strategy.stop_sigma,
            "freeze_sma_on_exit": self.cfg.strategy.freeze_sma_on_exit,
            "pending": pending, "position": pos,
            "summary": eng.summary(),
            "history": self.history,
            "trades": [self._trade_json(t) for t in eng.trades],
            "last_event": self.state["last_event"],
            "events": self.events[-20:],           # журнал последних действий
            "wait_reason": wait,                   # человекочитаемо: что бот делает/ждёт
            "data_lag_min": lag_min,               # возраст последнего бара, мин
            "last_bar_ts": last_bar_ts or None,
            "warmup_done": bool(b and b.is_ready), # прогрет ли индикатор
            "trade_start_ts": self.state.get("trade_start_ts"),  # граница backfill→live торговля
            "strategy_name": "Спред SBRF/SBPR · Bollinger(%d, %gσ)" % (
                self.cfg.strategy.sma_period, self.cfg.strategy.sigma_multiplier),
        })
