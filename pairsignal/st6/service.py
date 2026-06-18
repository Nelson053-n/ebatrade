"""Сервисный слой st6: состояние сессии Correlation-Gated Pairs + фон + персист.

St6Session по образцу St5Session, но для парной стратегии с динамическим отбором
пары из корзины акций MOEX:
  • держит корзину закрытий, периодически через rank_pairs выбирает лучшую пару;
  • ведёт FSM core.decide() по закрытым свечам выбранной пары;
  • paper-исполнение (PaperPortfolio: equity, открытая позиция, журнал сделок);
  • persist в session_state_6.json (переживает рестарт);
  • snapshot() для API; run_live (поллинг MOEX ISS) / run_player (синтетика).

Phase 1 — PAPER ONLY. Реальные ордера (T-Bank) не подключаются.
"""
from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

from . import data_feed as feed
from .config import St6Config
from .core import (
    PairStat, Position, Side,
    decide, leg_quantities, rank_pairs, trade_pnl,
)
from .models import EngineEvent, PairTrade

_BASE = Path(__file__).resolve().parent.parent.parent
HISTORY_LEN = 400
EVENTS_LEN = 40

# Реестр сессий st6: корзина одна → одиночная сессия "main". Оставлено словарём,
# чтобы при желании завести второй профиль (другую корзину) без смены контракта API.
ST6_SESSIONS: dict[str, "St6Session"] = {}


def get_session(sid: str = "main") -> "St6Session":
    """Получить (или создать) сессию по id. Восстанавливает состояние с диска."""
    s = ST6_SESSIONS.get(sid)
    if s is None:
        s = St6Session(sid)
        s.load_session()
        ST6_SESSIONS[sid] = s
    return s


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


class PaperPortfolio:
    """Бумажный учёт пары: equity, открытая позиция (core.Position), журнал сделок.

    Лот = 1 шт (акции MOEX торгуются лотами, но для paper-учёта берём штучный лот,
    как в бэктесте st6; стоимость ноги считается в рублях по close). Реальные ордера
    НЕ отправляются.
    """

    def __init__(self, start_equity: float) -> None:
        self.start_equity = start_equity
        self.equity = start_equity
        self.position = Position()
        self.trades: list[PairTrade] = []

    # --- открытие/закрытие пары (paper) ---
    def open_pair(self, side: Side, ticker_a: str, ticker_b: str, ts: int,
                  ca: float, cb: float, qa: int, qb: int, beta: float, z: float) -> None:
        self.position = Position(side=side, entry_z=z, beta=beta, bars_held=0,
                                 qty_a=qa, qty_b=qb, entry_a=ca, entry_b=cb)
        self._open_ctx = {"ticker_a": ticker_a, "ticker_b": ticker_b, "entry_ts": ts}

    def close_pair(self, ts: int, ca: float, cb: float, z: float, reason: str, p) -> PairTrade:
        pos = self.position
        net = trade_pnl(pos.side, pos.entry_a, ca, pos.entry_b, cb,
                        pos.qty_a, pos.qty_b, p)
        self.equity += net
        ctx = getattr(self, "_open_ctx", {})
        tr = PairTrade(
            side=pos.side.name,
            ticker_a=ctx.get("ticker_a", "A"), ticker_b=ctx.get("ticker_b", "B"),
            entry_ts=ctx.get("entry_ts", ts), exit_ts=ts,
            entry_a=pos.entry_a, exit_a=ca, entry_b=pos.entry_b, exit_b=cb,
            qty_a=pos.qty_a, qty_b=pos.qty_b, beta=pos.beta,
            entry_z=pos.entry_z, exit_z=z, bars_held=pos.bars_held,
            net_pnl_rub=net, reason=reason,
        )
        self.trades.append(tr)
        self.position = Position()
        return tr

    # --- сводка ---
    def summary(self) -> dict:
        wins = [t for t in self.trades if t.net_pnl_rub > 0]
        net = sum(t.net_pnl_rub for t in self.trades)
        return {
            "trades": len(self.trades),
            "wins": len(wins),
            "win_rate_pct": round(100 * len(wins) / len(self.trades), 1) if self.trades else 0.0,
            "net_pnl_rub": round(net, 0),
            "equity_rub": round(self.equity, 0),
            "return_pct": round(100 * net / self.start_equity, 3) if self.start_equity else 0.0,
            "avg_bars_held": round(sum(t.bars_held for t in self.trades) / len(self.trades), 1)
            if self.trades else 0.0,
            "open_position": self.position.side.name if self.position.is_open else None,
        }


class St6Session:
    """Полное состояние сессии st6 (одна корзина, одна активная пара)."""

    def __init__(self, sid: str = "main") -> None:
        self.sid = sid
        self._session_file = _BASE / f"session_state_6_{sid}.json"
        self.cfg = St6Config()
        self.port = PaperPortfolio(self.cfg.start_equity_rub)
        # выбранная пара и её статистика отбора
        self.pair: tuple[str, str] | None = None
        self.pair_stat: PairStat | None = None
        # текущий срез индикаторов (для snapshot)
        self.cur = {"z": None, "corr": None, "beta": None}
        # выровненные ряды закрытий корзины (ticker -> list[float])
        self.closes: dict[str, list[float]] = {}
        self.bars_since_select = 0
        self.state = {"live": False, "player": False, "session_started": None,
                      "paused_by_user": False, "last_event": None,
                      "data_source": "synthetic", "warmup_done": False}
        self.history: list[dict] = []     # ряд z/spread выбранной пары для графика
        self.events: list[EngineEvent] = []
        self.player_closes: dict[str, list[float]] | None = None
        self.player_idx = 0
        self.last_live_ts = 0
        self._lock = asyncio.Lock()

    @property
    def params(self):
        return self.cfg.strategy.to_params()

    def warmup_limit(self) -> int:
        p = self.params
        return max(p.beta_window, p.z_window, p.corr_window) + 5

    # ---------- журнал/история ----------
    def log_event(self, kind: str, message: str) -> None:
        self.events.append(EngineEvent(ts=time.time(), kind=kind, message=message))
        if len(self.events) > EVENTS_LEN:
            del self.events[0]
        self.state["last_event"] = message

    def push_history(self, ts: int) -> None:
        z = self.cur.get("z")
        if z is None or (isinstance(z, float) and not math.isfinite(z)):
            return
        self.history.append({"ts": ts, "z": round(float(z), 3),
                             "corr": round(float(self.cur.get("corr") or 0.0), 3),
                             "beta": round(float(self.cur.get("beta") or 0.0), 4)})
        if len(self.history) > HISTORY_LEN:
            del self.history[0]

    # ---------- отбор пары ----------
    def select_pair(self) -> bool:
        """Через rank_pairs выбрать лучшую пару корзины. True — пара выбрана/сменилась."""
        ranked = rank_pairs(self.closes, self.params)
        if not ranked:
            return False
        best = ranked[0]
        new = (best.a, best.b)
        changed = new != self.pair
        self.pair, self.pair_stat = new, best
        self.bars_since_select = 0
        if changed:
            self.log_event("select", f"выбрана пара {best.a}/{best.b} "
                           f"(corr {best.corr:+.2f}, p={best.pvalue:.3f}, "
                           f"hl={best.halflife:.0f})")
        return True

    def _maybe_reselect(self) -> None:
        """Пере-выбор пары вне позиции по таймеру reselect_every_bars."""
        if self.port.position.is_open:
            return
        if self.pair is None or self.bars_since_select >= self.cfg.reselect_every_bars:
            self.select_pair()

    # ---------- шаг по последнему закрытому бару ----------
    def step(self, ts: int) -> None:
        """Один шаг FSM по текущему хвосту закрытий выбранной пары. No repaint."""
        self.bars_since_select += 1
        self._maybe_reselect()
        if self.pair is None:
            return
        ta, tb = self.pair
        a = self.closes.get(ta)
        b = self.closes.get(tb)
        if not a or not b:
            self.pair = None
            return
        p = self.params
        pos = self.port.position
        sig = decide(a, b, pos, p)
        self.cur = {"z": sig.z, "corr": sig.corr, "beta": sig.beta}
        ca, cb = float(a[-1]), float(b[-1])

        if pos.is_open:
            pos.bars_held += 1
            if sig.action == "EXIT":
                tr = self.port.close_pair(ts, ca, cb, sig.z, sig.reason.value, p)
                self.log_event("exit", f"выход {tr.side} {ta}/{tb} | {tr.reason} | "
                               f"net {tr.net_pnl_rub:+.0f}₽ (z={sig.z:+.2f})")
        else:
            if sig.action in ("ENTER_LONG", "ENTER_SHORT"):
                qa, qb = leg_quantities(self.port.equity, ca, cb, sig.beta, 1, 1, p)
                if qa > 0 and qb > 0:
                    side = (Side.LONG_SPREAD if sig.action == "ENTER_LONG"
                            else Side.SHORT_SPREAD)
                    self.port.open_pair(side, ta, tb, ts, ca, cb, qa, qb, sig.beta, sig.z)
                    self.log_event("position", f"вход {side.name} {ta}/{tb} | "
                                   f"qa={qa} qb={qb} | z={sig.z:+.2f} corr={sig.corr:+.2f}")
        self.push_history(ts)

    def reset_engine(self) -> None:
        self.port = PaperPortfolio(self.cfg.start_equity_rub)
        self.pair = None
        self.pair_stat = None
        self.cur = {"z": None, "corr": None, "beta": None}
        self.closes = {}
        self.history = []
        self.bars_since_select = 0
        self.player_closes = None
        self.player_idx = 0
        self.last_live_ts = 0
        self.state["last_event"] = None
        self.state["warmup_done"] = False
        self.state["session_started"] = time.time()
        self.save_session()

    # ---------- персист ----------
    def save_session(self) -> None:
        try:
            data = {
                "session_started": self.state["session_started"],
                "config": self.cfg.model_dump(),
                "equity": self.port.equity,
                "trades": [asdict(t) for t in self.port.trades],
                "position": self._position_json(self.port.position),
                "open_ctx": getattr(self.port, "_open_ctx", None)
                if self.port.position.is_open else None,
                "pair": list(self.pair) if self.pair else None,
                "pair_stat": asdict(self.pair_stat) if self.pair_stat else None,
                "cur": self.cur,
                "history": self.history,
                "bars_since_select": self.bars_since_select,
                "last_live_ts": self.last_live_ts,
                "live": self.state["live"],
                "data_source": self.state["data_source"],
                "paused_by_user": self.state["paused_by_user"],
            }
            self._session_file.write_text(json.dumps(_clean(data), ensure_ascii=False))
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _position_json(pos: Position) -> dict | None:
        if not pos.is_open:
            return None
        return {"side": pos.side.value, "entry_z": pos.entry_z, "beta": pos.beta,
                "bars_held": pos.bars_held, "qty_a": pos.qty_a, "qty_b": pos.qty_b,
                "entry_a": pos.entry_a, "entry_b": pos.entry_b}

    @staticmethod
    def _position_from_json(d: dict) -> Position:
        return Position(side=Side(d["side"]), entry_z=d.get("entry_z", 0.0),
                        beta=d.get("beta", 0.0), bars_held=int(d.get("bars_held", 0)),
                        qty_a=int(d.get("qty_a", 0)), qty_b=int(d.get("qty_b", 0)),
                        entry_a=d.get("entry_a", 0.0), entry_b=d.get("entry_b", 0.0))

    def load_session(self) -> bool:
        if not self._session_file.exists():
            return False
        try:
            data = json.loads(self._session_file.read_text())
        except Exception:  # noqa: BLE001
            return False
        try:
            cfg = data.get("config")
            if cfg:
                self.cfg = St6Config(**cfg)
            self.port = PaperPortfolio(self.cfg.start_equity_rub)
            self.port.equity = data.get("equity", self.port.equity)
            self.port.trades = [PairTrade(**t) for t in data.get("trades", [])]
            pos = data.get("position")
            if pos:
                self.port.position = self._position_from_json(pos)
                octx = data.get("open_ctx")
                if octx:
                    self.port._open_ctx = octx
            pr = data.get("pair")
            self.pair = tuple(pr) if pr else None
            ps = data.get("pair_stat")
            if ps:
                self.pair_stat = PairStat(**ps)
            self.cur = data.get("cur") or {"z": None, "corr": None, "beta": None}
            self.history = data.get("history", [])
            self.bars_since_select = int(data.get("bars_since_select", 0))
            self.last_live_ts = int(data.get("last_live_ts") or 0)
            self.state["session_started"] = data.get("session_started", time.time())
            self.state["paused_by_user"] = bool(data.get("paused_by_user", False))
            self.state["resume_live"] = (bool(data.get("live"))
                                         and data.get("data_source") == "live"
                                         and not self.state["paused_by_user"])
            return True
        except Exception:  # noqa: BLE001
            return False

    # ---------- фоновые задачи ----------
    async def run_live(self) -> None:
        """Live на MOEX ISS (акции): backfill корзины, шаг FSM, ждём новые свечи.

        Дневной ТФ: новый бар раз в день — опрос редкий (poll_seconds). Последний
        формирующийся бар отбрасываем (только закрытые свечи, no repaint).
        """
        self.state["data_source"] = "live"
        self.log_event("info", f"live запущен (paper, MOEX ISS): корзина "
                       f"{len(self.cfg.basket)} тикеров, прогрев…")
        first = True
        while self.state["live"]:
            try:
                closes = await asyncio.to_thread(
                    feed.load_basket, self.cfg.basket, self.cfg.history_days,
                    self.cfg.timeframe)
                if len(closes) < 2:
                    self.log_event("warn", "ISS: недостаточно тикеров с данными — ждём")
                else:
                    async with self._lock:
                        # последний бар может быть формирующимся — отбрасываем хвост
                        self.closes = {t: v[:-1] for t, v in closes.items()}
                        n = min(len(v) for v in self.closes.values())
                        ts = int(time.time() * 1000)
                        if n > self.warmup_limit() and ts != self.last_live_ts:
                            self.step(ts)
                            self.last_live_ts = ts
                            if first:
                                self.state["warmup_done"] = True
                                self.log_event("info", f"прогрев завершён: {n} баров, "
                                               f"пара {self.pair}")
                            self.save_session()
                first = False
            except Exception as e:  # noqa: BLE001
                self.log_event("warn", f"ошибка ISS: {e}")
            await asyncio.sleep(self.cfg.poll_seconds)

    async def run_player(self) -> None:
        """Synthetic-player: подаём офлайн-корзину бар за баром через FSM."""
        self.state["data_source"] = "synthetic"
        if self.player_closes is None:
            self.player_closes = feed.synthetic_basket()
        full = self.player_closes
        n = min(len(v) for v in full.values())
        warm = self.warmup_limit()
        if self.player_idx < warm:
            self.player_idx = warm
        while self.state["player"] and self.player_idx < n:
            async with self._lock:
                i = self.player_idx
                self.closes = {t: v[:i + 1] for t, v in full.items()}
                ts = i  # бар-индекс как ts для синтетики
                self.step(ts)
                self.player_idx += 1
                if self.player_idx % 25 == 0:
                    self.save_session()
            await asyncio.sleep(0.05)
        self.state["player"] = False
        self.save_session()

    # ---------- снимок для API ----------
    def snapshot(self, server_started: float) -> dict:
        p = self.params
        pos = self.port.position
        position = None
        if pos.is_open:
            ctx = getattr(self.port, "_open_ctx", {})
            position = {"side": pos.side.name, "ticker_a": ctx.get("ticker_a"),
                        "ticker_b": ctx.get("ticker_b"), "entry_ts": ctx.get("entry_ts"),
                        "qty_a": pos.qty_a, "qty_b": pos.qty_b,
                        "entry_a": round(pos.entry_a, 4), "entry_b": round(pos.entry_b, 4),
                        "beta": round(pos.beta, 4), "entry_z": round(pos.entry_z, 3),
                        "bars_held": pos.bars_held}

        z = self.cur.get("z")
        corr = self.cur.get("corr")
        running = self.state["live"] or self.state["player"]
        n_have = min((len(v) for v in self.closes.values()), default=0)
        if not running:
            wait = "остановлено"
        elif self.pair is None:
            wait = "годной пары нет (ослабьте corr/pvalue/half-life в конфиге)"
        elif n_have <= self.warmup_limit():
            wait = f"прогрев: {n_have}/{self.warmup_limit()} баров"
        elif pos.is_open:
            wait = "в позиции — ждём возврата спреда к среднему"
        elif corr is not None and abs(corr) < p.corr_enter:
            wait = (f"гейт закрыт: corr={corr:+.2f} < {p.corr_enter:g} "
                    f"(пара рассогласована)")
        elif z is not None and abs(z) < p.z_entry:
            wait = f"ждём сигнал: z={z:+.2f}, вход при |z|≥{p.z_entry:g}"
        else:
            wait = "ждём закрытие следующего бара"

        stat = self.pair_stat
        pair_info = None
        if self.pair and stat:
            pair_info = {"a": stat.a, "b": stat.b, "corr": round(stat.corr, 3),
                         "beta": round(stat.beta, 4), "pvalue": round(stat.pvalue, 4),
                         "halflife": round(stat.halflife, 1), "score": round(stat.score, 4)}

        return _clean({
            "sid": self.sid,
            "live": self.state["live"], "player": self.state["player"],
            "data_source": self.state["data_source"],
            "data_provider": "MOEX ISS" if self.state["data_source"] == "live" else "синтетика",
            "auto_approve": self.cfg.auto_approve,
            "connector": self.cfg.connector,
            "session_started": self.state["session_started"],
            "server_started": server_started, "now": time.time(),
            "paused_by_user": self.state["paused_by_user"],
            "basket": self.cfg.basket,
            "timeframe": self.cfg.timeframe,
            "pair": list(self.pair) if self.pair else None,
            "pair_stat": pair_info,
            "cur_z": round(z, 3) if isinstance(z, (int, float)) and math.isfinite(z) else None,
            "cur_corr": round(corr, 3) if isinstance(corr, (int, float)) and math.isfinite(corr) else None,
            "cur_beta": round(self.cur.get("beta"), 4)
            if isinstance(self.cur.get("beta"), (int, float)) and math.isfinite(self.cur.get("beta")) else None,
            "params": {"z_entry": p.z_entry, "z_exit": p.z_exit, "z_stop": p.z_stop,
                       "corr_enter": p.corr_enter, "corr_break": p.corr_break,
                       "beta_window": p.beta_window, "z_window": p.z_window,
                       "corr_window": p.corr_window, "risk_fraction": p.risk_fraction},
            "position": position,
            "summary": self.port.summary(),
            "history": self.history,
            "trades": [asdict(t) for t in self.port.trades],
            "events": [asdict(e) for e in self.events[-20:]],
            "last_event": self.state["last_event"],
            "wait_reason": wait,
            "warmup_done": n_have > self.warmup_limit(),
            "bars_have": n_have,
            "strategy_name": "Correlation-Gated Pairs · корзина %d · ТФ %s" % (
                len(self.cfg.basket), self.cfg.timeframe),
        })
