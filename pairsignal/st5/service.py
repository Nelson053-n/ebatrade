"""Сервисный слой st5: состояние сессии VWAP-reversion + фоновые задачи + персист.

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
from .config import St5Config
from .engine import TradingEngine
from .models import BotState, InstrumentSpec, Position, PriceBar

_BASE = Path(__file__).resolve().parent.parent.parent
_OUT_DIR = _BASE / "pairsignal" / "out"
HISTORY_LEN = 300
EVENTS_LEN = 40
BT_HISTORY_LEN = 60

# инструменты, доступные как независимые сессии st5 (?inst=).
# ключ — id в API, значение — (ASSETCODE, ярлык). Новый инструмент = одна строка.
ST5_INSTRUMENTS: dict[str, tuple[str, str]] = {
    "sber": ("SBRF", "Сбербанк"),
    "gazp": ("GAZR", "Газпром"),
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
        asset, self.inst_label = ST5_INSTRUMENTS[inst]
        self._session_file = _BASE / f"session_state_5_{inst}.json"
        self.cfg = St5Config()
        self.cfg.instrument.asset = asset
        self.spec: InstrumentSpec = feed.synthetic_spec()
        self.engine = TradingEngine(self.cfg, self.spec)
        self.state = {"live": False, "player": False, "session_started": None,
                      "paused_by_user": False, "last_event": None,
                      "data_source": "synthetic", "warmup_done": False,
                      "trade_start_ts": None}
        self.history: list[dict] = []
        self.events: list[dict] = []
        self.player_df = None
        self.player_idx = 0
        self.last_live_ts = 0
        self._lock = asyncio.Lock()

    # ---------- инструмент ----------
    def resolve_real_leg(self) -> None:
        self.spec = feed.resolve_leg(self.cfg)

    def reset_engine(self, real: bool = False) -> None:
        if real:
            try:
                self.resolve_real_leg()
            except Exception:  # noqa: BLE001  оффлайн — синтетическая спека
                self.spec = feed.synthetic_spec()
        self.engine = TradingEngine(self.cfg, self.spec)
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
        r = self.engine.last_reading
        if r is None or not r.is_ready:
            return
        if any(v is None or (isinstance(v, float) and math.isnan(v))
               for v in (r.price, r.vwap, r.upper, r.lower)):
            return
        self.history.append({"ts": ts, "price": round(r.price, 1), "vwap": round(r.vwap, 1),
                             "upper": round(r.upper, 1), "lower": round(r.lower, 1),
                             "sigma": round(r.sigma, 1)})
        if len(self.history) > HISTORY_LEN:
            del self.history[0]

    def warmup_limit(self) -> int:
        # VWAP внутридневной — баров дня хватает; берём 3 дня запасом
        bpd = int(14 * 60 / self.cfg.strategy.candle_interval_minutes)
        return bpd * 3 + 20

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
            self.engine = TradingEngine(self.cfg, self.spec)
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
        """Live на MOEX ISS: backfill+replay, затем ждём новые свечи (как st4, но одна нога)."""
        replayed = False
        self.engine._check_lag = True
        self.log_event("info", f"live запущен (paper, MOEX ISS): {self.spec.code}, "
                       f"прогрев VWAP…")
        while self.state["live"]:
            try:
                d2e = self.engine.days_to_expiry(int(time.time() * 1000))
                if (d2e is not None and self.engine.position is None
                        and d2e < self.cfg.instrument.rollover_days_before_expiry
                        and await asyncio.to_thread(self._rollover)):
                    self.log_event("info", f"роллировер: торгуем {self.spec.code}")
                df = await asyncio.to_thread(
                    feed.read_ohlcv_moex, self.cfg, self.warmup_limit(), self.spec.code)
                df = df.iloc[:-1]   # ISS: без формирующегося бара
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
        """Пере-резолв серии с переносом журнала/баланса. True — серия сменилась."""
        old = self.spec.code
        self.resolve_real_leg()
        if self.spec.code == old:
            return False
        prev = self.engine
        eng = TradingEngine(self.cfg, self.spec)
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
        pos = None
        if eng.position is not None:
            p = eng.position
            pos = {"state": p.state.value, "entry_ts": p.entry_ts,
                   "side": p.side, "lots": p.lots,
                   "entry_price": round(p.entry_price, 0),
                   "entry_vwap": round(p.entry_vwap, 0),
                   "unrealized_rub": round(eng.unrealized_rub(), 0),
                   "bars_held": eng._bars_held}
        pending = None
        if eng._pending is not None:
            sig, r = eng._pending
            pending = {"signal": sig.value, "price": round(r.price, 1), "vwap": round(r.vwap, 1)}

        last_bar_ts = self.last_live_ts or (self.history[-1]["ts"] if self.history else 0)
        lag_min = round((time.time() - last_bar_ts / 1000) / 60) if last_bar_ts else None
        r = eng.last_reading
        cur_z = ((r.price - r.vwap) / r.sigma) if (r and r.is_ready and r.sigma > 0) else None

        if eng.risk.halted:
            wait = "HALTED — нужен ручной разбор"
        elif eng.position is not None:
            wait = "в позиции — ждём возврата к VWAP"
        elif not (self.state["live"] or self.state["player"]):
            wait = "остановлено"
        elif r is None or not r.is_ready:
            wait = f"прогрев VWAP (нужно ≥{self.cfg.strategy.min_bars_in_day} баров дня)"
        elif self.state["data_source"] == "live" and lag_min is not None and lag_min > 25:
            wait = f"ждём свежий бар — MOEX ISS ({lag_min} мин лаг)"
        elif cur_z is not None and abs(cur_z) < self.cfg.strategy.band_sigma:
            wait = (f"ждём сигнал: цена в коридоре "
                    f"(z={cur_z:+.2f}, вход при |z|≥{self.cfg.strategy.band_sigma:g})")
        else:
            wait = "ждём закрытие следующего бара"

        return _clean({
            "live": self.state["live"], "player": self.state["player"],
            "data_source": self.state["data_source"],
            "data_provider": "MOEX ISS" if self.state["data_source"] == "live" else "синтетика",
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
            "band_sigma": self.cfg.strategy.band_sigma,
            "stop_sigma": self.cfg.strategy.stop_sigma,
            "take_profit_sigma": self.cfg.strategy.take_profit_sigma,
            "entry_trigger": self.cfg.strategy.entry_trigger,
            "flat_at_session_end": self.cfg.strategy.flat_at_session_end,
            "pending": pending, "position": pos,
            "summary": eng.summary(),
            "history": self.history,
            "trades": [self._trade_json(t) for t in eng.trades],
            "last_event": self.state["last_event"],
            "events": self.events[-20:],
            "wait_reason": wait,
            "data_lag_min": lag_min,
            "last_bar_ts": last_bar_ts or None,
            "warmup_done": bool(r and r.is_ready),
            "trade_start_ts": self.state.get("trade_start_ts"),
            "cur_z": round(cur_z, 2) if cur_z is not None else None,
            "strategy_name": "VWAP-reversion %s · коридор %gσ" % (
                self.cfg.instrument.asset, self.cfg.strategy.band_sigma),
        })
