"""Сервисный слой st5: состояние сессии directional momentum + фоновые задачи + персист.

St5Session держит движок, конфиг, историю графика, потоки live/player — по образцу
St4Session, но для одиночного инструмента (без пары, без sandbox: paper-only Phase 1).
Live тянет OHLCV с MOEX ISS. Состояние переживает рестарт (session_state_5_<id>.json).
"""
from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

from . import data_feed as feed
from ..st4.tbank_sandbox import has_token as _has_token  # токен общий с st4
from .config import St5Config
from .engine import TradingEngine
from .models import BotState, InstrumentSpec, Position, PriceBar

_BASE = Path(__file__).resolve().parent.parent.parent
_OUT_DIR = _BASE / "pairsignal" / "out"
HISTORY_LEN = 300
EVENTS_LEN = 40
BT_HISTORY_LEN = 60

# инструменты, доступные как независимые сессии st5 (?inst=).
# ключ — id в API, значение — (ASSETCODE, ярлык, strategy_mode). Новый инструмент = одна строка.
# meanrev по умолчанию для SBER (устойчивый OOS-эдж, Sharpe ~1.5) и GAZP (paper-наблюдение,
# OOS breakeven под издержками — держим в meanrev для накопления статистики).
ST5_INSTRUMENTS: dict[str, tuple[str, str, str]] = {
    "sber": ("SBRF", "Сбербанк", "meanrev"),
    "gazp": ("GAZR", "Газпром", "meanrev"),
}


def _bt_history_file(inst: str) -> Path:
    return _OUT_DIR / f"st5_backtest_{inst}_history.json"


def bt_history_load(inst: str = "sber") -> list[dict]:
    f = _bt_history_file(inst)
    try:
        if f.exists():
            return json.loads(f.read_text())
    except Exception:  # noqa: BLE001
        pass
    return []


def bt_history_append(entry: dict, inst: str = "sber") -> list[dict]:
    hist = bt_history_load(inst)
    day = entry.get("date", "")[:10]
    bs = entry.get("band_sigma")
    hist = [h for h in hist if not (h.get("date", "")[:10] == day and h.get("band_sigma") == bs)]
    hist.append(entry)
    hist = hist[-BT_HISTORY_LEN:]
    f = _bt_history_file(inst)
    try:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(hist))
    except Exception:  # noqa: BLE001
        pass
    return hist


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


class St5Session:
    """Полное состояние сессии st5 (один экземпляр на инструмент)."""

    def __init__(self, inst: str = "sber") -> None:
        if inst not in ST5_INSTRUMENTS:
            raise ValueError(f"неизвестный инструмент st5: {inst}")
        self.inst = inst
        asset, self.inst_label, mode = ST5_INSTRUMENTS[inst]
        self._session_file = _BASE / f"session_state_5_{inst}.json"
        self.cfg = St5Config()
        self.cfg.instrument.asset = asset
        self.cfg.strategy.strategy_mode = mode
        self.spec: InstrumentSpec = feed.synthetic_spec()
        self.engine = TradingEngine(self.cfg, self.spec)
        self.state = {"live": False, "player": False, "session_started": None,
                      "paused_by_user": False, "last_event": None,
                      "data_source": "synthetic", "warmup_done": False,
                      "sandbox_active": False, "trade_start_ts": None}
        self.history: list[dict] = []
        self.events: list[dict] = []
        self.player_df = None
        self.player_idx = 0
        self.last_live_ts = 0
        self._lock = asyncio.Lock()

    # ---------- инструмент ----------
    def resolve_real_leg(self) -> None:
        self.spec = feed.resolve_leg(self.cfg)

    def _restore_engine_position(self) -> None:
        """Восстановить состояние движка из session-файла в (только что пересозданный в
        reset_engine) движок: ЖУРНАЛ СДЕЛОК, баланс, дневной P&L и открытую позицию.

        Нужно для sandbox: позиция требуется для reconciliation (если на счёте та же
        позиция, что вёл движок до рестарта — она ЛЕГИТИМНА, не закрываем). Журнал/баланс
        переносим, чтобы форвард-тест продолжался, а не обнулялся при активации sandbox."""
        try:
            if not self._session_file.exists():
                return
            data = json.loads(self._session_file.read_text())
            if "balance_rub" in data:
                self.engine.balance_rub = data["balance_rub"]
            self.engine.trades = [self._trade_from_json(t) for t in data.get("trades", [])]
            self.engine.risk.day_pnl_rub = data.get("day_pnl_rub", 0.0)
            self.engine.risk._day = data.get("day_key", "")
            pos = data.get("position")
            if pos:
                self.engine.position = self._position_from_json(pos)
                self.engine.state = self.engine.position.state
                self.engine._bars_held = int(data.get("bars_held", 0))
        except Exception:  # noqa: BLE001  битый файл не должен ронять старт
            pass

    def _position_matches_lots(self, lots: int) -> bool:
        """Совпадает ли позиция движка с фактическим балансом инструмента на sandbox-счёте.
        LONG (side=buy) = +lots, SHORT (side=sell) = −lots."""
        p = self.engine.position
        if p is None:
            return False
        want = p.lots * (1 if p.side == "buy" else -1)
        return lots == want

    def reset_engine(self, real: bool = False) -> None:
        if real:
            try:
                self.resolve_real_leg()
            except Exception:  # noqa: BLE001  оффлайн — синтетическая спека
                self.spec = feed.synthetic_spec()
        # cfg.connector.mode — НАМЕРЕНИЕ оператора (не трогаем). sandbox_active — ФАКТ:
        # активен ли реальный исполнитель сейчас. Sandbox активируется только в live (real=True);
        # на синтетике движок строим как paper (рыночные ордера по выдуманным барам бессмысленны).
        want_sandbox = self.cfg.connector.mode == "tbank_sandbox"
        self.state["sandbox_active"] = False
        self.state["sandbox_error"] = None
        if want_sandbox and real:
            try:
                self.engine = TradingEngine(self.cfg, self.spec)
                # восстановить позицию из сессии ДО reconciliation: если на счёте та же
                # позиция, что вёл движок до рестарта — она ЛЕГИТИМНА, не закрываем.
                self._restore_engine_position()
                self.state["sandbox_active"] = True
                # RECONCILIATION: сверяем баланс инструмента на sandbox-счёте с позицией движка.
                # Совпали → принимаем (движок продолжает вести сделку). Разошлись → закрываем
                # «осиротевшую» ногу (счёт привести к состоянию движка).
                try:
                    ex = self.engine.executor
                    lots = ex.broker_lots()
                    if lots != 0:
                        if self._position_matches_lots(lots):
                            self.log_event("info", f"reconciliation: позиция на счёте "
                                           f"{lots:+d} совпала с движком — продолжаем сделку")
                        else:
                            self.log_event("warn", f"reconciliation: на sandbox-счёте висит "
                                           f"{lots:+d} (не совпало с движком) — закрываю")
                            if ex.flat_broker():
                                self.log_event("info", "reconciliation: счёт приведён к FLAT")
                                self.engine.position = None
                                self.engine.state = BotState.FLAT
                            else:
                                self.log_event("warn", "reconciliation: не удалось закрыть ногу")
                except Exception as e:  # noqa: BLE001  сверка не должна ронять старт
                    self.log_event("warn", f"reconciliation пропущена: {e}")
            except Exception as e:  # noqa: BLE001  sandbox недоступен → откат в paper-движок
                msg = str(e)
                if "401" in msg:
                    msg = "токен T-Bank невалиден или отозван (HTTP 401) — выпустите новый"
                self.state["sandbox_error"] = msg
                self.log_event("warn", f"sandbox не активирован: {msg} → исполнение paper")
                paper_cfg = self.cfg.model_copy(deep=True)
                paper_cfg.connector.mode = "paper"
                self.engine = TradingEngine(paper_cfg, self.spec)
                self._restore_engine_position()   # live-фолбэк: сохранить журнал/баланс/позицию
        else:
            # синтетика или paper-намерение: строим paper-движок (не трогая cfg.connector.mode)
            paper_cfg = self.cfg
            if want_sandbox:
                paper_cfg = self.cfg.model_copy(deep=True)
                paper_cfg.connector.mode = "paper"
                self.state["last_event"] = "sandbox активен только в live — на синтетике paper"
            self.engine = TradingEngine(paper_cfg, self.spec)
        self.history = []
        self.player_df = None
        self.player_idx = 0
        self.last_live_ts = 0
        self.state["last_event"] = None
        self.state["warmup_done"] = False
        self.state["trade_start_ts"] = None
        self.state["session_started"] = time.time()
        self.save_session()

    def log_event(self, kind: str, message: str) -> None:
        self.events.append({"ts": time.time(), "kind": kind, "message": message})
        if len(self.events) > EVENTS_LEN:
            del self.events[0]
        self.state["last_event"] = message

    def push_history(self, ts: int) -> None:
        if self.engine.mode == "meanrev":
            zr = self.engine.last_z
            if zr is None or not zr.is_ready:
                return
            if any(v is None or (isinstance(v, float) and math.isnan(v))
                   for v in (zr.price, zr.sma, zr.z)):
                return
            # для графика meanrev: close + SMA-база + z (метка направления — знак z)
            self.history.append({"ts": ts, "price": round(zr.price, 1),
                                 "ref_price": round(zr.sma, 1),
                                 "signal": (-1 if zr.z > 0 else (1 if zr.z < 0 else 0)),
                                 "z": round(zr.z, 3)})
            if len(self.history) > HISTORY_LEN:
                del self.history[0]
            return
        r = self.engine.last_reading
        if r is None or not r.is_ready:
            return
        if any(v is None or (isinstance(v, float) and math.isnan(v))
               for v in (r.price, r.ref_price)):
            return
        # для графика: close + база сравнения + метка направления momentum
        self.history.append({"ts": ts, "price": round(r.price, 1),
                             "ref_price": round(r.ref_price, 1), "signal": r.signal,
                             "lookback_return": round(r.lookback_return * 100, 3)})
        if len(self.history) > HISTORY_LEN:
            del self.history[0]

    def warmup_limit(self) -> int:
        # индикатор смотрит на N баров назад — берём с запасом (N + 3 дня)
        bpd = int(14 * 60 / self.cfg.strategy.candle_interval_minutes)
        n = (self.cfg.strategy.mr_ma_n if self.cfg.strategy.strategy_mode == "meanrev"
             else self.cfg.strategy.lookback)
        return n + bpd * 3 + 20

    # ---------- персист ----------
    def save_session(self) -> None:
        try:
            eng = self.engine
            data = {
                "session_started": self.state["session_started"],
                "balance_rub": eng.balance_rub,
                "trades": [self._trade_json(t) for t in eng.trades],
                "history": self.history,
                "config": self.cfg.model_dump(),
                "spec": asdict(self.spec),
                "day_pnl_rub": eng.risk.day_pnl_rub,
                "day_key": eng.risk._day,
                "position": asdict(eng.position) if eng.position else None,
                "bars_held": eng._bars_held,
                "halted": eng.risk.halted,
                "halt_reason": eng.risk.halt_reason,
                "live": self.state["live"],
                "data_source": self.state["data_source"],
                "paused_by_user": self.state["paused_by_user"],
                "last_live_ts": self.last_live_ts,
            }
            self._session_file.write_text(json.dumps(_clean(data)))
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _trade_json(t) -> dict:
        d = asdict(t)
        d["state"] = t.state.value
        return d

    @staticmethod
    def _trade_from_json(d: dict):
        from .models import Trade
        d = dict(d)
        d["state"] = BotState(d["state"])
        return Trade(**d)

    @staticmethod
    def _position_from_json(d: dict) -> Position:
        d = dict(d)
        d["state"] = BotState(d["state"])
        return Position(**d)

    def load_session(self) -> bool:
        if not self._session_file.exists():
            return False
        try:
            data = json.loads(self._session_file.read_text())
        except Exception:  # noqa: BLE001
            return False
        try:
            for k, v in (data.get("config") or {}).items():
                if hasattr(self.cfg, k) and isinstance(v, dict):
                    sub = getattr(self.cfg, k)
                    for kk, vv in v.items():
                        if hasattr(sub, kk):
                            setattr(sub, kk, vv)
            sp = data.get("spec")
            if sp:
                self.spec = InstrumentSpec(**sp)
            # восстановление — всегда paper-движок: sandbox-исполнитель в конструкторе ходит
            # в сеть (счёт/инструмент), а при загрузке сети может не быть. Sandbox активируется
            # отдельно в reset_engine при старте live. cfg.connector.mode (намерение) не трогаем.
            cfg = self.cfg
            if cfg.connector.mode == "tbank_sandbox":
                cfg = self.cfg.model_copy(deep=True)
                cfg.connector.mode = "paper"
            self.engine = TradingEngine(cfg, self.spec)
            self.engine.balance_rub = data.get("balance_rub", self.engine.balance_rub)
            self.engine.risk.day_pnl_rub = data.get("day_pnl_rub", 0.0)
            self.engine.risk._day = data.get("day_key", "")
            self.engine.trades = [self._trade_from_json(t) for t in data.get("trades", [])]
            pos = data.get("position")
            if pos:
                self.engine.position = self._position_from_json(pos)
                self.engine.state = self.engine.position.state
                self.engine._bars_held = int(data.get("bars_held", 0))
            if data.get("halted"):
                self.engine.risk.halt(data.get("halt_reason") or "восстановлено из сессии")
                self.engine.state = BotState.HALTED
            self.history = data.get("history", [])
            self.state["session_started"] = data.get("session_started", time.time())
            self.last_live_ts = int(data.get("last_live_ts") or 0)
            self.state["paused_by_user"] = bool(data.get("paused_by_user", False))
            self.state["resume_live"] = (bool(data.get("live"))
                                         and data.get("data_source") == "live"
                                         and not self.state["paused_by_user"])
            return True
        except Exception:  # noqa: BLE001
            return False

    # ---------- фоновые задачи ----------
    async def run_live(self) -> None:
        """Live: backfill+replay, затем ждём новые свечи (как st4, но одна нога).

        Источник свечей: в sandbox — T-Bank (REAL-TIME, без лага ISS), иначе MOEX ISS
        (публичный, задержка 15-30 мин). uid инструмента берём из sandbox-исполнителя.
        Sandbox исполняет по ТЕКУЩЕЙ цене T-Bank → на backfill-replay (исторические бары)
        реальные ордера бессмысленны: на replay движок disarmed (только прогрев momentum),
        arm включаем после replay — торгуем с первого живого бара. Для paper всегда armed.
        """
        replayed = False
        self.engine._check_lag = True
        sandbox = self.state.get("sandbox_active", False)
        tbank_uid = None
        if sandbox and self.engine.executor is not None:
            try:
                tbank_uid = self.engine.executor.leg_uid()
            except Exception:  # noqa: BLE001  не удалось — откатываемся на ISS
                tbank_uid = None
        src_lbl = "T-Bank real-time" if tbank_uid else "MOEX ISS"
        mode_lbl = "T-Bank sandbox" if sandbox else "paper"
        self.log_event("info", f"live запущен ({mode_lbl}, данные {src_lbl}): {self.spec.code}, "
                       f"прогрев momentum…")
        while self.state["live"]:
            try:
                d2e = self.engine.days_to_expiry(int(time.time() * 1000))
                if (d2e is not None and self.engine.position is None
                        and d2e < self.cfg.instrument.rollover_days_before_expiry
                        and await asyncio.to_thread(self._rollover)):
                    self.log_event("info", f"роллировер: торгуем {self.spec.code}")
                    if sandbox and self.engine.executor is not None:
                        try:
                            tbank_uid = self.engine.executor.leg_uid()
                        except Exception:  # noqa: BLE001
                            tbank_uid = None
                if tbank_uid:
                    df = await asyncio.to_thread(
                        feed.read_ohlcv_tbank, self.cfg, self.warmup_limit(), tbank_uid)
                    # T-Bank get_candles уже отдаёт только закрытые бары → не режем хвост
                else:
                    df = await asyncio.to_thread(
                        feed.read_ohlcv_moex, self.cfg, self.warmup_limit(), self.spec.code)
                    df = df.iloc[:-1]   # ISS: без формирующегося бара
                self.engine.arm(not (sandbox and not replayed))  # на replay входы закрыты
                async with self._lock:
                    new_bars = 0
                    for ts, row in df.iterrows():
                        if not self.state["live"]:
                            return
                        ts = int(ts)
                        if ts <= self.last_live_ts:
                            continue
                        if self.engine._pending is not None:
                            ttl_ms = (self.cfg.strategy.pending_ttl_bars
                                      * self.cfg.strategy.candle_interval_minutes * 60_000)
                            if ts - self.engine._pending[1].ts > ttl_ms:
                                self.engine.reject()
                                self.log_event("warn", "рекомендация не подтверждена — TTL")
                            else:
                                break
                        bar = PriceBar(ts, float(row["open"]), float(row["high"]),
                                       float(row["low"]), float(row["close"]),
                                       float(row.get("volume", 0.0)))
                        res = self.engine.step(bar)
                        self.last_live_ts = ts
                        new_bars += 1
                        self.push_history(ts)
                        for ev in (res.events or []):
                            if ev.kind in ("position", "exit", "halt", "warn"):
                                self.log_event(ev.kind, ev.message)
                        if res.trade is not None:
                            self.save_session()
                        if not replayed and self.history:
                            await asyncio.sleep(0.02)
                lag_min = (time.time() - self.last_live_ts / 1000) / 60 if self.last_live_ts else 0
                if not replayed:
                    self.state["warmup_done"] = True
                    self.state["trade_start_ts"] = self.last_live_ts
                    self.log_event("info", f"прогрев завершён: {len(self.history)} баров")
                if replayed and new_bars > 0:
                    self.log_event("info", f"новых баров: {new_bars} (последний {lag_min:.0f} мин назад)")
                elif replayed and lag_min > 25:
                    self.log_event("warn", f"нет свежих свечей ISS {lag_min:.0f} мин — ждём")
            except Exception as e:  # noqa: BLE001
                self.log_event("warn", f"ошибка ISS: {e}")
            replayed = True
            self.save_session()
            await asyncio.sleep(self.cfg.poll_seconds)

    def _rollover(self) -> bool:
        """Пере-резолв серии с переносом журнала/баланса. True — серия сменилась.

        В sandbox новый движок поднимает sandbox-исполнитель на новую серию (сетевой вызов —
        ОК, вызывается из to_thread). Если sandbox не активен (paper/фолбэк), строим paper-движок,
        не трогая cfg.connector.mode."""
        old = self.spec.code
        self.resolve_real_leg()
        if self.spec.code == old:
            return False
        prev = self.engine
        cfg = self.cfg
        if not self.state.get("sandbox_active"):
            cfg = self.cfg.model_copy(deep=True)
            cfg.connector.mode = "paper"
        eng = TradingEngine(cfg, self.spec)
        eng.balance_rub = prev.balance_rub
        eng.trades = prev.trades
        eng.risk.day_pnl_rub = prev.risk.day_pnl_rub
        eng.risk._day = prev.risk._day
        self.engine = eng
        self.history = []
        self.save_session()
        return True

    async def run_player(self) -> None:
        """Synthetic-player: подаём офлайн-бары по одному через движок."""
        while self.state["player"] and self.player_df is not None \
                and self.player_idx < len(self.player_df):
            if self.engine._pending is not None:
                await asyncio.sleep(0.4)
                continue
            async with self._lock:
                ts = int(self.player_df.index[self.player_idx])
                row = self.player_df.iloc[self.player_idx]
                self.player_idx += 1
                bar = PriceBar(ts, float(row["open"]), float(row["high"]), float(row["low"]),
                               float(row["close"]), float(row.get("volume", 0.0)))
                res = self.engine.step(bar)
                self.push_history(ts)
                if res.events:
                    self.state["last_event"] = res.events[-1].message
                if res.trade is not None:
                    self.save_session()
            warming = not self.history
            await asyncio.sleep(0 if warming else 0.4)
        self.state["player"] = False
        self.save_session()

    # ---------- снимок для API ----------
    def snapshot(self, server_started: float) -> dict:
        eng = self.engine
        s = self.cfg.strategy
        meanrev = s.strategy_mode == "meanrev"
        hold_max = s.mr_max_hold if meanrev else s.holding
        pos = None
        if eng.position is not None:
            p = eng.position
            pos = {"state": p.state.value, "entry_ts": p.entry_ts,
                   "side": p.side, "lots": p.lots,
                   "entry_price": round(p.entry_price, 0),
                   "entry_ref": round(p.entry_vwap, 0),   # momentum: close[-lookback]; meanrev: SMA
                   "unrealized_rub": round(eng.unrealized_rub(), 0),
                   "bars_held": eng._bars_held, "holding": hold_max}
        pending = None
        if eng._pending is not None:
            sig, r = eng._pending
            pending = {"signal": sig.value}
            if r is not None:   # momentum-режим несёт MomentumReading; meanrev — None
                pending.update({"price": round(r.price, 1),
                                "ref_price": round(r.ref_price, 1)})
            elif eng.last_z is not None and eng.last_z.is_ready:
                pending.update({"price": round(eng.last_z.price, 1),
                                "ref_price": round(eng.last_z.sma, 1)})

        last_bar_ts = self.last_live_ts or (self.history[-1]["ts"] if self.history else 0)
        lag_min = round((time.time() - last_bar_ts / 1000) / 60) if last_bar_ts else None

        if meanrev:
            zr = eng.last_z
            z_ready = bool(zr and zr.is_ready)
            cur_z = round(zr.z, 3) if z_ready else None
            # cur_signal: +1 если z в зоне лонга (ниже −entry_z), −1 если зоне шорта, иначе 0
            if z_ready and zr.z <= -s.mr_entry_z:
                cur_signal = 1
            elif z_ready and zr.z >= s.mr_entry_z:
                cur_signal = -1
            else:
                cur_signal = 0
            lookback_return = None
            ind_ready = z_ready
        else:
            r = eng.last_reading
            ind_ready = bool(r and r.is_ready)
            cur_signal = r.signal if ind_ready else 0
            lookback_return = round(r.lookback_return * 100, 3) if ind_ready else None
            cur_z = None
        _dir = {1: "вверх (LONG)", -1: "вниз (SHORT)", 0: "флэт"}.get(cur_signal, "—")

        if eng.risk.halted:
            wait = "HALTED — нужен ручной разбор"
        elif eng.position is not None:
            if meanrev:
                wait = (f"в позиции — держим {eng._bars_held}/{hold_max} баров "
                        f"(TP |z|≤{s.mr_exit_z:g}, стоп |z|≥{s.mr_stop_z:g})")
            else:
                wait = f"в позиции — держим {eng._bars_held}/{hold_max} баров (стоп {s.stop_pct * 100:g}%)"
        elif not (self.state["live"] or self.state["player"]):
            wait = "остановлено"
        elif not ind_ready:
            need = s.mr_ma_n if meanrev else s.lookback
            label = "z-score" if meanrev else "momentum"
            wait = f"прогрев {label} (нужно ≥{need} баров)"
        elif self.state["data_source"] == "live" and lag_min is not None and lag_min > 25:
            wait = f"ждём свежий бар — MOEX ISS ({lag_min} мин лаг)"
        elif meanrev:
            wait = f"ждём отклонение |z|≥{s.mr_entry_z:g}: z={cur_z:+.2f} → {_dir}"
        else:
            wait = f"ждём вход по тренду: momentum {lookback_return:+.2f}% → {_dir}"

        # честный провайдер котировок: в sandbox-live свечи из T-Bank (real-time)
        if self.state["data_source"] == "live":
            data_provider = "T-Bank" if self.state.get("sandbox_active") else "MOEX ISS"
        else:
            data_provider = "синтетика"
        return _clean({
            "live": self.state["live"], "player": self.state["player"],
            "data_source": self.state["data_source"],
            "data_provider": data_provider,
            # коннектор: mode — намерение оператора, sandbox_active — реально ли активен sandbox
            # сейчас (только в live). Сам токен НИКОГДА не отдаём, только token_set (bool).
            "connector_mode": self.cfg.connector.mode,
            "sandbox_active": self.state.get("sandbox_active", False),
            "sandbox_error": self.state.get("sandbox_error"),
            "token_set": _has_token(),
            "connector_account": self.cfg.connector.account_id or None,
            "auto_approve": self.cfg.auto_approve,
            "fsm_state": eng.state.value,
            "halted": eng.risk.halted, "halt_reason": eng.risk.halt_reason,
            "trading_enabled": self.cfg.risk.trading_enabled,
            "session_started": self.state["session_started"],
            "server_started": server_started, "now": time.time(),
            "paused_by_user": self.state["paused_by_user"],
            "inst": self.inst, "inst_label": self.inst_label,
            "asset": self.cfg.instrument.asset,
            "leg": {"code": self.spec.code, "expiry": self.spec.expiry},
            "interval_min": self.cfg.strategy.candle_interval_minutes,
            "strategy_mode": s.strategy_mode,
            "lookback": s.lookback,
            "holding": s.holding,
            "stop_pct": s.stop_pct,
            # параметры meanrev (фронт/вкладка читает при strategy_mode=="meanrev")
            "mr_ma_n": s.mr_ma_n,
            "mr_entry_z": s.mr_entry_z,
            "mr_exit_z": s.mr_exit_z,
            "mr_stop_z": s.mr_stop_z,
            "mr_max_hold": s.mr_max_hold,
            "session_lo_min": s.session_lo_min,
            "session_hi_min": s.session_hi_min,
            "flat_at_session_end": s.flat_at_session_end,
            "pending": pending, "position": pos,
            "summary": eng.summary(),
            "history": self.history,
            "trades": [self._trade_json(t) for t in eng.trades],
            "last_event": self.state["last_event"],
            "events": self.events[-20:],
            "wait_reason": wait,
            "data_lag_min": lag_min,
            "last_bar_ts": last_bar_ts or None,
            "warmup_done": ind_ready,
            "trade_start_ts": self.state.get("trade_start_ts"),
            "cur_signal": cur_signal,
            "lookback_return": lookback_return,
            "cur_z": cur_z,
            "strategy_name": ("meanrev" if meanrev else
                              "Momentum %s · lb%d/h%d" % (
                                  self.cfg.instrument.asset, s.lookback, s.holding)),
        })
