"""Минимальный backend под панель (phase 1).

Оборачивает Engine: отдаёт состояние/рекомендацию/историю, принимает approve/reject.
Два источника баров:
  • live-режим — фоновая задача тянет котировки через CCXT и прогоняет последний бар;
  • synthetic-player — по таймеру подаёт офлайн-бары в реальный Engine (для демо панели
    без сети, но через настоящую логику strategy.py/virtual_exchange.py).

Две независимые стратегии («слоты»): SLOTS[0]=balanced, SLOTS[1]=conservative. Каждый
слот держит свой Engine/конфиг/историю/поток; эндпоинты выбирают слот через ?slot=1|2
(по умолчанию 1 — обратная совместимость). Оба потока работают одновременно в фоне.

Панель (dashboard.html) отдаётся на «/».

Запуск:  uvicorn pairsignal.api:app --reload
"""
from __future__ import annotations

import asyncio
import json
import math
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from .config import AppConfig
from .data_feed import (
    generate_synthetic,
    generate_synthetic_cross,
    read_ohlcv_ccxt,
    read_ohlcv_cross,
)
from .engine import Engine
from .models import IndicatorRow, SpreadDirection, Trade

DASHBOARD = Path(__file__).resolve().parent.parent / "dashboard.html"
HISTORY_LEN = 120  # сколько последних закрытых баров держим для графика
_BASE = Path(__file__).resolve().parent.parent
_LEGACY_SESSION_FILE = _BASE / "session_state.json"  # старый одно-слотовый файл (миграция)


def _session_file(idx: int) -> Path:
    """Файл персистентности слота: session_state_1.json / session_state_2.json."""
    return _BASE / f"session_state_{idx + 1}.json"

# популярные ликвидные перпы MEXC для парного спреда (формат CCXT)
PAIRS = [
    ("BTC/USDT:USDT", "ETH/USDT:USDT"),
    ("BTC/USDT:USDT", "SOL/USDT:USDT"),
    ("ETH/USDT:USDT", "SOL/USDT:USDT"),
    ("BTC/USDT:USDT", "BNB/USDT:USDT"),
    ("ETH/USDT:USDT", "BNB/USDT:USDT"),
    ("SOL/USDT:USDT", "AVAX/USDT:USDT"),
    ("BNB/USDT:USDT", "SOL/USDT:USDT"),
    ("LTC/USDT:USDT", "BCH/USDT:USDT"),
    ("XRP/USDT:USDT", "ADA/USDT:USDT"),
]

# поддерживаемые таймфреймы MEXC → длительность в минутах
TIMEFRAMES = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240}

# пресеты параметров (подобраны grid-search + multi-agent ресёрч, out-of-sample train/test)
PRESETS = {
    "balanced": {
        "label": "Сбалансированный",
        "desc": "win 73%, net +91 (6 мес, 9 пар, OOS-подтверждён): ранний тейк + широкий стоп",
        "entry_z": 2.5, "stop_z": 8.0, "profit_target_fees": 6.0,
        "bb_period": 240, "min_width_pct": 0.5,
    },
    "conservative": {
        "label": "Консервативный",
        "desc": "win 65%, net +68: реже фиксирует прибыль, самый стабильный net по периодам",
        "entry_z": 2.5, "stop_z": 8.0, "profit_target_fees": 8.0,
        "bb_period": 240, "min_width_pct": 0.5,
    },
    "cross_pct": {
        "label": "Кросс-биржа SUI (gateio/mexc)",
        "desc": "линейный спред gateio−mexc, адаптивные полосы bb_k·σ(спреда), SMA(200), выход к SMA",
        "entry_z": 1.0, "stop_z": 8.0, "profit_target_fees": 6.0,
        "bb_period": 240, "min_width_pct": 0.0,
        # cross-специфика (применяется в SlotState._apply_preset):
        "spread_mode": "cross_pct", "band_mode": "vol", "band_pct": 0.03, "sma_period": 200,
    },
}


def _sym_short(symbol: str) -> str:
    """BTC/USDT:USDT → BTC (короткая подпись ноги для UI)."""
    return symbol.split("/")[0]


def _day_key(ts_ms: int) -> str:
    """unix ms → 'YYYY-MM-DD' (UTC) для группировки доходности по дням."""
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def _clean(obj):
    """Рекурсивно заменяет NaN/inf на None — JSON их не допускает."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


class SlotState:
    """Полное состояние одной торговой сессии (слота): конфиг, движок, поток, история.

    Два экземпляра живут параллельно в SLOTS. Engine уже per-instance, поэтому слоты
    не делят мутируемого состояния — общее только read-only константы и _server_started.
    """

    def __init__(self, idx: int, preset_key: str | None = None,
                 pair: tuple[str, str] | None = None) -> None:
        self.idx = idx                         # 0 | 1 — адрес файла персистентности
        self.cfg = AppConfig()
        if preset_key is not None:
            self._apply_preset(preset_key)     # стартовый пресет слота
        if pair is not None:                   # стартовая пара слота (чтобы вкладки отличались)
            self.cfg.strategy.symbol_a, self.cfg.strategy.symbol_b = pair
        self.engine = Engine(self.cfg)
        self.state: dict = {"live": False, "player": False, "last_rec": None,
                            "session_started": None,   # старт текущей торговой сессии
                            "paused_by_user": False}   # пауза нажата оператором
        self.history: list[dict] = []          # срез индикаторов по барам (для графика)
        self.player_rows: list[IndicatorRow] = []
        self.player_idx = 0
        self.last_live_ts = 0                  # ts последнего обработанного live-бара (дедуп)

    def _apply_preset(self, key: str) -> None:
        """Записать параметры пресета в cfg.strategy (без сброса движка)."""
        p = PRESETS[key]
        s = self.cfg.strategy
        s.entry_z = p["entry_z"]
        s.stop_z = p["stop_z"]
        s.profit_target_fees = p["profit_target_fees"]
        s.bb_period = p["bb_period"]
        s.min_width_pct = p["min_width_pct"]
        # cross-специфика: применяем, если пресет её задаёт; иначе ВОЗВРАЩАЕМ в log,
        # чтобы переключение cross_pct → balanced/conservative не залипло в кросс-режиме
        s.spread_mode = p.get("spread_mode", "log")
        if "band_mode" in p:
            s.band_mode = p["band_mode"]
        if "band_pct" in p:
            s.band_pct = p["band_pct"]
        if "sma_period" in p:
            s.sma_period = p["sma_period"]

    def reset_engine(self) -> None:
        self.engine = Engine(self.cfg)
        self.history = []
        self.player_rows = []
        self.player_idx = 0
        self.last_live_ts = 0
        self.state["last_rec"] = None
        self.state["session_started"] = time.time()   # старт новой торговой сессии
        self.save_session()

    def save_session(self) -> None:
        """Сохранить результаты сессии слота на диск (переживают перезапуск сервера).

        Пишем журнал сделок, баланс, время сессии, историю графика и ключевые настройки.
        Точное место плеера/открытую позицию не персистим — после рестарта поток
        стартует заново (свежая синтетика), но РЕЗУЛЬТАТЫ прошлой сессии сохраняются.
        """
        try:
            data = {
                "session_started": self.state["session_started"],
                "balance": self.engine.exch.balance,
                "trades": [asdict(t) for t in self.engine.exch.trades],
                "history": self.history,
                "strategy": self.cfg.strategy.model_dump(),
                "paper": self.cfg.paper.model_dump(),
                "auto_approve": self.cfg.auto_approve,
            }
            _session_file(self.idx).write_text(json.dumps(_clean(data)))
        except Exception:  # noqa: BLE001  персистентность не должна ронять рантайм
            pass

    def load_session(self) -> bool:
        """Восстановить результаты сессии слота при старте сервера. True — если восстановили."""
        path = _session_file(self.idx)
        if not path.exists() and self.idx == 0 and _LEGACY_SESSION_FILE.exists():
            path = _LEGACY_SESSION_FILE          # одноразовая миграция со старого файла
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            return False
        try:
            # настройки
            for k, v in (data.get("strategy") or {}).items():
                if hasattr(self.cfg.strategy, k):
                    setattr(self.cfg.strategy, k, v)
            for k, v in (data.get("paper") or {}).items():
                if hasattr(self.cfg.paper, k):
                    setattr(self.cfg.paper, k, v)
            self.cfg.auto_approve = data.get("auto_approve", self.cfg.auto_approve)
            # результаты в чистый движок
            self.engine = Engine(self.cfg)
            self.engine.exch.balance = data.get("balance", self.engine.exch.balance)
            self.engine.exch.trades = [
                Trade(**{**t, "direction": SpreadDirection(t["direction"])})
                for t in data.get("trades", [])
            ]
            self.history = data.get("history", [])
            self.state["session_started"] = data.get("session_started", time.time())
            return True
        except Exception:  # noqa: BLE001
            return False

    def warmup_limit(self) -> int:
        """Сколько баров тянуть, чтобы индикаторы прогрелись.

        В log-режиме сначала нужна β (окно beta_window), потом BB поверх спреда
        (окно bb_period) — то есть beta_window + bb_period; плюс запас на валидные бары.
        """
        s = self.cfg.strategy
        if s.spread_mode == "cross_pct":
            # запас ×1.5: после inner-join свечей двух бирж общих баров меньше, чем тянули
            return int(s.sma_period * 1.5) + 80
        elif s.spread_mode == "log":
            base = s.bb_period + s.beta_window
        else:
            base = s.bb_period
        return base + 80

    def push_history(self, row: IndicatorRow) -> None:
        # на прогреве индикаторов mid/z = NaN — такие бары в историю не кладём
        # (иначе JSON-сериализация /state падает на nan)
        if any(math.isnan(v) for v in (row.spread, row.mid, row.z, row.width_pct, row.beta)):
            return
        self.history.append({
            "ts": row.ts, "spread": row.spread, "mid": row.mid,
            "upper": row.upper, "lower": row.lower, "z": row.z,
            "width_pct": row.width_pct, "beta": row.beta,
        })
        if len(self.history) > HISTORY_LEN:
            del self.history[0]

    def current_preset(self) -> str | None:
        """Определить, какому пресету соответствует текущий конфиг слота (или None)."""
        s = self.cfg.strategy
        for key, p in PRESETS.items():
            if (s.entry_z == p["entry_z"] and s.stop_z == p["stop_z"]
                    and s.profit_target_fees == p["profit_target_fees"]
                    and s.bb_period == p["bb_period"]):
                return key
        return None


_server_started = time.time()       # момент запуска процесса (uptime сервера, общий)
SLOTS = [
    SlotState(0, "balanced", ("XRP/USDT:USDT", "ADA/USDT:USDT")),
    SlotState(1, "cross_pct", ("SUI/USDT:USDT", "SUI/USDT:USDT")),
]


def _slot(slot: int = 1) -> SlotState:
    """Выбрать слот по 1|2 (дефолт 1). Невалидный → 400."""
    if slot not in (1, 2):
        raise HTTPException(400, "slot: 1 или 2")
    return SLOTS[slot - 1]


def _reader_for(cfg):
    """Источник реальных котировок по режиму: cross_pct → две биржи, иначе одна биржа."""
    return read_ohlcv_cross if cfg.strategy.spread_mode == "cross_pct" else read_ohlcv_ccxt


async def _poller(st: SlotState):
    """Live-режим на РЕАЛЬНЫХ данных MEXC (no repaint).

    Старт: тянем реальную историю и БЫСТРО проигрываем её бар за баром (replay) —
    график «оживает» за секунды, но на настоящих котировках. Догнав до последней
    закрытой свечи, переходим в живой режим: ждём появления новых свечей (раз в
    timeframe). Последний формирующийся бар всегда пропускаем.
    """
    replayed = False
    while st.state["live"]:
        try:
            # CCXT блокирующий — в пул потоков, чтобы не вешать event loop
            df = await asyncio.to_thread(_reader_for(st.cfg), st.cfg.strategy, st.warmup_limit())
            rows = Engine.rows_from_df(df, st.cfg.strategy)[:-1]   # без формирующегося бара
            new = [r for r in rows if r.ts > st.last_live_ts]
            for r in new:
                if not st.state["live"]:
                    return
                if st.engine.pending is not None:    # ждём решения оператора (ручной режим)
                    break
                res = st.engine.step(r)
                st.push_history(r)
                st.last_live_ts = r.ts
                st.state["last_rec"] = _clean(asdict(res.rec))
                if res.trade is not None:
                    st.save_session()
                # на стартовом replay — быстрый темп между ВАЛИДНЫМИ барами (прогрев,
                # где mid/z=NaN и история пуста, проматываем мгновенно)
                if not replayed and st.history:
                    await asyncio.sleep(0.12)
        except Exception as e:  # noqa: BLE001
            st.state["last_rec"] = {"error": str(e)}
        replayed = True
        # история проиграна — дальше ждём новые закрытые свечи (живой режим)
        await asyncio.sleep(st.cfg.poll_seconds)


async def _player(st: SlotState):
    """Synthetic-player: подаём офлайн-бары по одному через реальный Engine.

    Останавливаемся на баре, ждущем решения оператора (human-in-the-loop), пока
    не будет вызван approve/reject. Так панель повторяет настоящий поток phase 1.
    """
    while st.state["player"] and st.player_idx < len(st.player_rows):
        if st.engine.pending is not None:           # ждём решения оператора
            await asyncio.sleep(0.4)
            continue
        row = st.player_rows[st.player_idx]
        st.player_idx += 1
        res = st.engine.step(row)
        st.push_history(row)
        st.state["last_rec"] = _clean(asdict(res.rec))
        if res.trade is not None:
            st.save_session()         # сделка закрыта — фиксируем результаты на диск
        # прогрев индикаторов (mid/z=NaN) проматываем мгновенно — иначе панель
        # минуты висит пустой; живой темп держим только по валидным барам
        warming = not st.history
        await asyncio.sleep(0 if warming else 0.7)
    st.state["player"] = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    for st in SLOTS:
        st.load_session()             # восстановить результаты прошлой сессии (если есть)
    yield
    for st in SLOTS:
        st.save_session()             # сохранить при остановке сервера
        st.state["live"] = False
        st.state["player"] = False


app = FastAPI(title="pairsignal", version="0.1", lifespan=lifespan)


@app.get("/")
def dashboard():
    if not DASHBOARD.exists():
        raise HTTPException(404, "dashboard.html не найден")
    return FileResponse(DASHBOARD)


@app.get("/health")
def health():
    return {"ok": True, "slots": [{"live": s.state["live"], "player": s.state["player"]}
                                  for s in SLOTS]}


@app.get("/config")
def get_config(slot: int = 1):
    return _slot(slot).cfg.model_dump()


@app.get("/pairs")
def pairs(slot: int = 1):
    """Список доступных пар для выбора на панели."""
    st = _slot(slot)
    return {
        "pairs": [
            {"a": a, "b": b, "label": f"{_sym_short(a)} / {_sym_short(b)}"}
            for a, b in PAIRS
        ],
        "current": {"a": st.cfg.strategy.symbol_a, "b": st.cfg.strategy.symbol_b},
    }


@app.get("/timeframes")
def timeframes(slot: int = 1):
    """Список доступных таймфреймов для графика."""
    st = _slot(slot)
    return {"timeframes": list(TIMEFRAMES), "current": st.cfg.strategy.timeframe}


@app.get("/presets")
def presets(slot: int = 1):
    """Доступные пресеты параметров для панели."""
    st = _slot(slot)
    return {
        "presets": [{"key": k, "label": p["label"], "desc": p["desc"]} for k, p in PRESETS.items()],
        "current": st.current_preset(),
    }


@app.post("/preset")
def set_preset(key: str, slot: int = 1):
    """Применить пресет параметров (со сбросом сессии)."""
    if key not in PRESETS:
        raise HTTPException(400, f"неизвестный пресет: {key}")
    st = _slot(slot)
    st._apply_preset(key)
    was_live, was_player = st.state["live"], st.state["player"]
    st.state["live"] = st.state["player"] = False
    st.reset_engine()
    return {"ok": True, "preset": key, "config": st.cfg.model_dump(),
            "was_live": was_live, "was_player": was_player}


@app.get("/analyze")
async def analyze(period: str = "day", slot: int = 1):
    """Моделирование стратегии на РЕАЛЬНОЙ истории MEXC за период.

    Прогоняет текущие параметры (auto-вход) по историческим барам и считает,
    сколько было бы входов и какая прибыль. Не трогает живую сессию панели —
    работает на отдельном Engine.
    """
    if period not in ("day", "week"):
        raise HTTPException(400, "period: day или week")
    st = _slot(slot)
    # баров в окне периода по таймфрейму
    tf_min = TIMEFRAMES.get(st.cfg.strategy.timeframe, 5)
    per_day = 24 * 60 // tf_min
    window = per_day if period == "day" else per_day * 7
    limit = st.warmup_limit() + window          # прогрев + окно анализа

    try:
        df = await asyncio.to_thread(_reader_for(st.cfg), st.cfg.strategy, limit)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"не удалось получить историю: {e}")

    sim_cfg = AppConfig(**st.cfg.model_dump())
    sim_cfg.auto_approve = True                 # считаем все входы
    sim = Engine(sim_cfg)
    rows = Engine.rows_from_df(df, sim_cfg.strategy)
    # окно анализа — последние `window` баров (прогрев остаётся за кадром)
    start_ts = rows[-window].ts if len(rows) > window else rows[0].ts
    for r in rows:
        sim.step(r)

    trades = [t for t in sim.exch.trades if t.exit_ts >= start_ts]
    wins = [t for t in trades if t.net_pnl > 0]
    net = sum(t.net_pnl for t in trades)
    gross = sum(t.gross_pnl for t in trades)
    fees = sum(t.fees for t in trades)
    stops = sum(1 for t in trades if t.reason == "stop")

    # история периода для графика: спред, полосы BB и уровни стопа (mid ± stop_z·std)
    sz = sim_cfg.strategy.stop_z
    hist = [
        {
            "ts": r.ts, "spread": r.spread, "mid": r.mid,
            "upper": r.upper, "lower": r.lower,
            "stop_up": r.mid + sz * r.std, "stop_down": r.mid - sz * r.std,
        }
        for r in rows
        if r.ts >= start_ts and not math.isnan(r.mid) and not math.isnan(r.spread)
    ]

    # кривая доходности по дням: накопленный net P&L, сгруппированный по дате выхода
    by_day: dict[str, float] = {}
    for t in trades:
        day = _day_key(t.exit_ts)
        by_day[day] = by_day.get(day, 0.0) + t.net_pnl
    cum = 0.0
    equity_curve = []
    for day in sorted(by_day):
        cum += by_day[day]
        equity_curve.append({"day": day, "pnl": round(by_day[day], 2), "cum": round(cum, 2)})

    return _clean({
        "period": period,
        "pair": {"a": _sym_short(st.cfg.strategy.symbol_a), "b": _sym_short(st.cfg.strategy.symbol_b)},
        "timeframe": st.cfg.strategy.timeframe,
        "bars_analyzed": window,
        "entries": len(trades),
        "wins": len(wins),
        "win_rate_pct": round(100 * len(wins) / len(trades), 1) if trades else 0.0,
        "net_pnl": round(net, 2),
        "gross_pnl": round(gross, 2),
        "fees": round(fees, 2),
        "stops": stops,
        "return_pct": round(100 * net / sim_cfg.paper.start_balance, 3),
        "start_balance": sim_cfg.paper.start_balance,
        "history": hist,                         # для графика спреда периода
        "equity_curve": equity_curve,            # доходность по дням
        "trades": [asdict(t) for t in trades],
    })


@app.post("/config")
def set_config(payload: dict, slot: int = 1):
    """Обновить пороги/сайзинг/таймфрейм в рантайме и сбросить сессию.

    Принимает любое подмножество: entry_z, stop_z, exit_z, risk_pct, timeframe.
    Возвращает применённый конфиг. Параметры применяются с чистого листа.
    """
    st = _slot(slot)
    s, p = st.cfg.strategy, st.cfg.paper

    def _num(key, lo, hi):
        if key not in payload or payload[key] is None:
            return None
        try:
            v = float(payload[key])
        except (TypeError, ValueError):
            raise HTTPException(400, f"{key}: не число")
        if not (lo <= v <= hi):
            raise HTTPException(400, f"{key}: вне диапазона [{lo}, {hi}]")
        return v

    entry = _num("entry_z", 0.1, 10)
    stop = _num("stop_z", 0.1, 20)
    exit_ = _num("exit_z", 0.0, 10)
    risk = _num("risk_pct", 0.1, 100)
    tf = payload.get("timeframe")

    # согласованность порогов: выход < вход < стоп
    e = entry if entry is not None else s.entry_z
    sp = stop if stop is not None else s.stop_z
    ex = exit_ if exit_ is not None else s.exit_z
    if not (ex < e < sp):
        raise HTTPException(400, f"нужно выход({ex}) < вход({e}) < стоп({sp})")
    if tf is not None and tf not in TIMEFRAMES:
        raise HTTPException(400, f"timeframe: один из {list(TIMEFRAMES)}")

    # пара бумаг — только из списка PAIRS (надёжные символы MEXC)
    sa, sb = payload.get("symbol_a"), payload.get("symbol_b")
    if sa is not None or sb is not None:
        sa = sa or s.symbol_a
        sb = sb or s.symbol_b
        if (sa, sb) not in PAIRS:
            raise HTTPException(400, f"пара {sa}/{sb} не из списка доступных")

    s.entry_z, s.stop_z, s.exit_z = e, sp, ex
    if risk is not None:
        p.risk_pct = risk
    if tf is not None:
        s.timeframe = tf
    if sa is not None:
        s.symbol_a, s.symbol_b = sa, sb
    if "auto_approve" in payload:
        st.cfg.auto_approve = bool(payload["auto_approve"])

    was_live, was_player = st.state["live"], st.state["player"]
    st.state["live"] = st.state["player"] = False
    st.reset_engine()
    return {"ok": True, "config": st.cfg.model_dump(), "was_live": was_live, "was_player": was_player}


@app.get("/state")
def state(slot: int = 1):
    st = _slot(slot)
    pos = asdict(st.engine.exch.position) if st.engine.exch.position else None
    pending = asdict(st.engine.pending) if st.engine.pending else None
    return _clean({
        "live": st.state["live"],
        "player": st.state["player"],
        "auto_approve": st.cfg.auto_approve,     # режим авто-входа
        "preset": st.current_preset(),           # активный пресет параметров
        "session_started": st.state["session_started"],  # unix-сек старта торговой сессии
        "server_started": _server_started,    # unix-сек запуска сервера (uptime)
        "now": time.time(),                   # серверное «сейчас» (для расчёта длительности на клиенте)
        "pair": {                             # текущая пара (короткие подписи ног)
            "a": _sym_short(st.cfg.strategy.symbol_a),
            "b": _sym_short(st.cfg.strategy.symbol_b),
        },
        "timeframe": st.cfg.strategy.timeframe,  # текущий таймфрейм
        "spread_mode": st.cfg.strategy.spread_mode,  # log | ratio | cross_pct (для подписи графика)
        "band_mode": st.cfg.strategy.band_mode,      # pct | vol (для подписи cross)
        "band_pct": st.cfg.strategy.band_pct,        # ширина коридора cross-режима (pct)
        "bb_k": st.cfg.strategy.bb_k,                # множитель σ для полос (vol)
        "sma_period": st.cfg.strategy.sma_period,    # окно SMA/σ cross-режима
        "exchange_a": st.cfg.strategy.exchange_a,    # биржа ноги A (cross)
        "exchange_b": st.cfg.strategy.exchange_b,    # биржа ноги B (cross)
        "symbol_cross": st.cfg.strategy.symbol_cross,
        "paused_by_user": st.state["paused_by_user"],  # пауза нажата оператором
        "pending_recommendation": pending,   # ждёт решения оператора
        "position": pos,
        "summary": st.engine.summary(),
        "history": st.history,                # последние бары для графика
        "trades": [asdict(t) for t in st.engine.exch.trades],
        "last": st.state["last_rec"],
    })


@app.post("/auto")
def set_auto(on: bool = True, slot: int = 1):
    """Переключить авто-вход без сброса сессии. При включении сразу подтверждает
    висящую рекомендацию (если оператор уже ждал решения)."""
    st = _slot(slot)
    st.cfg.auto_approve = on
    if on and st.engine.pending is not None:
        st.engine.approve()
    return {"ok": True, "auto_approve": st.cfg.auto_approve}


@app.post("/approve")
def approve(slot: int = 1):
    st = _slot(slot)
    if not st.engine.pending:
        raise HTTPException(400, "нет ожидающей рекомендации")
    st.engine.approve()
    return {"ok": True, "position": asdict(st.engine.exch.position) if st.engine.exch.position else None}


@app.post("/reject")
def reject(slot: int = 1):
    st = _slot(slot)
    st.engine.reject()
    return {"ok": True}


@app.post("/reset")
def reset(slot: int = 1):
    _slot(slot).reset_engine()
    return {"ok": True}


@app.post("/live/start")
async def live_start(slot: int = 1):
    st = _slot(slot)
    if st.state["live"]:
        return {"ok": True, "already": True}
    st.state["player"] = False        # live и плеер взаимоисключаются
    st.reset_engine()                 # чистый старт live с бэкфиллом
    st.state["live"] = True
    asyncio.create_task(_poller(st))
    return {"ok": True, "mode": "live"}


@app.post("/live/stop")
def live_stop(slot: int = 1):
    _slot(slot).state["live"] = False
    return {"ok": True}


@app.post("/player/start")
async def player_start(limit: int = 2000, slot: int = 1):
    """Запустить/ВОЗОБНОВИТЬ синтетический плеер.

    Если плеер был на паузе и не доигран — ВОЗОБНОВЛЯЕМ с того же места, БЕЗ сброса
    сессии (журнал/сделки/время сохраняются). Свежий старт со сбросом — только когда
    плеера ещё не было или он доигран до конца. Сброс сессии — отдельная кнопка/эндпоинт.
    """
    st = _slot(slot)
    if st.state["player"]:
        return {"ok": True, "already": True}
    st.state["live"] = False          # плеер и live взаимоисключаются
    st.state["paused_by_user"] = False  # старт/возобновление снимает флаг паузы
    paused = bool(st.player_rows) and st.player_idx < len(st.player_rows)
    if paused:
        # возобновление: движок и позиция сохранены, продолжаем фоновую подачу баров
        st.state["player"] = True
        asyncio.create_task(_player(st))
        return {"ok": True, "resumed": True}
    # генерируем новые бары для графика (кросс-режим — своя синтетика двух бирж)
    gen = generate_synthetic_cross if st.cfg.strategy.spread_mode == "cross_pct" else generate_synthetic
    df = gen(n=limit)
    rows = Engine.rows_from_df(df, st.cfg.strategy)
    if st.state["session_started"] is None:
        # сессии ещё не было — свежий старт с чистого листа
        st.reset_engine()
    else:
        # сессия есть (восстановлена с диска / доиграл плеер): НЕ сбрасываем результаты,
        # просто продолжаем график новыми барами — журнал/баланс/время сохраняются
        st.history.clear()
    st.player_rows = rows
    st.player_idx = 0
    st.state["player"] = True
    asyncio.create_task(_player(st))
    return {"ok": True, "bars": len(st.player_rows)}


@app.post("/player/stop")
def player_stop(slot: int = 1):
    st = _slot(slot)
    st.state["player"] = False
    st.state["paused_by_user"] = True   # намеренная пауза — фронт не возобновляет авто
    return {"ok": True}


@app.post("/replay/synthetic")
def replay_synthetic(limit: int = 2000, slot: int = 1):
    """Прогнать синтетику целиком разом (для смоук-теста) — авто-подтверждение входов."""
    st = _slot(slot)
    st.reset_engine()
    st.cfg.auto_approve = True
    gen = generate_synthetic_cross if st.cfg.strategy.spread_mode == "cross_pct" else generate_synthetic
    try:
        for row in Engine.rows_from_df(gen(n=limit), st.cfg.strategy):
            st.engine.step(row)
            st.push_history(row)
    finally:
        st.cfg.auto_approve = False
    return st.engine.summary()
