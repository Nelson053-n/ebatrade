"""Минимальный backend под панель (phase 1).

Оборачивает Engine: отдаёт состояние/рекомендацию/историю, принимает approve/reject.
Два источника баров:
  • live-режим — фоновая задача тянет котировки через CCXT и прогоняет последний бар;
  • synthetic-player — по таймеру подаёт офлайн-бары в реальный Engine (для демо панели
    без сети, но через настоящую логику strategy.py/virtual_exchange.py).

Панель (dashboard.html) отдаётся на «/».

Запуск:  uvicorn pairsignal.api:app --reload
"""
from __future__ import annotations

import asyncio
import math
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from .config import AppConfig
from .data_feed import generate_synthetic, read_ohlcv_ccxt
from .engine import Engine
from .models import IndicatorRow

DASHBOARD = Path(__file__).resolve().parent.parent / "dashboard.html"
HISTORY_LEN = 120  # сколько последних закрытых баров держим для графика

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
}


def _sym_short(symbol: str) -> str:
    """BTC/USDT:USDT → BTC (короткая подпись ноги для UI)."""
    return symbol.split("/")[0]


def _day_key(ts_ms: int) -> str:
    """unix ms → 'YYYY-MM-DD' (UTC) для группировки доходности по дням."""
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

cfg = AppConfig()
engine = Engine(cfg)
_server_started = time.time()       # момент запуска процесса (uptime сервера)
_state: dict = {"live": False, "player": False, "last_rec": None,
                "session_started": None}   # старт текущей торговой сессии (обнуляется сбросом)
_history: list[dict] = []           # срез индикаторов по обработанным барам (для графика)
_player_rows: list[IndicatorRow] = []
_player_idx = 0
_last_live_ts = 0                    # ts последнего обработанного live-бара (дедуп)


def _reset_engine() -> None:
    global engine, _history, _player_rows, _player_idx, _last_live_ts
    engine = Engine(cfg)
    _history = []
    _player_rows = []
    _player_idx = 0
    _last_live_ts = 0
    _state["last_rec"] = None
    _state["session_started"] = time.time()   # старт новой торговой сессии


def _warmup_limit() -> int:
    """Сколько баров тянуть, чтобы индикаторы прогрелись.

    В log-режиме сначала нужна β (окно beta_window), потом BB поверх спреда
    (окно bb_period) — то есть beta_window + bb_period; плюс запас на валидные бары.
    """
    s = cfg.strategy
    base = s.bb_period
    if s.spread_mode == "log":
        base += s.beta_window
    return base + 80


def _clean(obj):
    """Рекурсивно заменяет NaN/inf на None — JSON их не допускает."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _push_history(row: IndicatorRow) -> None:
    # на прогреве индикаторов mid/z = NaN — такие бары в историю не кладём
    # (иначе JSON-сериализация /state падает на nan)
    if any(math.isnan(v) for v in (row.spread, row.mid, row.z, row.width_pct, row.beta)):
        return
    _history.append({
        "ts": row.ts, "spread": row.spread, "mid": row.mid,
        "upper": row.upper, "lower": row.lower, "z": row.z,
        "width_pct": row.width_pct, "beta": row.beta,
    })
    if len(_history) > HISTORY_LEN:
        del _history[0]


async def _poller():
    """Live-режим на РЕАЛЬНЫХ данных MEXC (no repaint).

    Старт: тянем реальную историю и БЫСТРО проигрываем её бар за баром (replay) —
    график «оживает» за секунды, но на настоящих котировках. Догнав до последней
    закрытой свечи, переходим в живой режим: ждём появления новых свечей (раз в
    timeframe). Последний формирующийся бар всегда пропускаем.
    """
    global _last_live_ts
    replayed = False
    while _state["live"]:
        try:
            # CCXT блокирующий — в пул потоков, чтобы не вешать event loop
            df = await asyncio.to_thread(read_ohlcv_ccxt, cfg.strategy, _warmup_limit())
            rows = Engine.rows_from_df(df, cfg.strategy)[:-1]   # без формирующегося бара
            new = [r for r in rows if r.ts > _last_live_ts]
            for r in new:
                if not _state["live"]:
                    return
                if engine.pending is not None:    # ждём решения оператора (ручной режим)
                    break
                res = engine.step(r)
                _push_history(r)
                _last_live_ts = r.ts
                _state["last_rec"] = _clean(asdict(res.rec))
                # на стартовом replay — быстрый темп между ВАЛИДНЫМИ барами (прогрев,
                # где mid/z=NaN и история пуста, проматываем мгновенно)
                if not replayed and _history:
                    await asyncio.sleep(0.12)
        except Exception as e:  # noqa: BLE001
            _state["last_rec"] = {"error": str(e)}
        replayed = True
        # история проиграна — дальше ждём новые закрытые свечи (живой режим)
        await asyncio.sleep(cfg.poll_seconds)


async def _player():
    """Synthetic-player: подаём офлайн-бары по одному через реальный Engine.

    Останавливаемся на баре, ждущем решения оператора (human-in-the-loop), пока
    не будет вызван approve/reject. Так панель повторяет настоящий поток phase 1.
    """
    global _player_idx
    while _state["player"] and _player_idx < len(_player_rows):
        if engine.pending is not None:           # ждём решения оператора
            await asyncio.sleep(0.4)
            continue
        row = _player_rows[_player_idx]
        _player_idx += 1
        res = engine.step(row)
        _push_history(row)
        _state["last_rec"] = _clean(asdict(res.rec))
        # прогрев индикаторов (mid/z=NaN) проматываем мгновенно — иначе панель
        # минуты висит пустой; живой темп держим только по валидным барам
        warming = not _history
        await asyncio.sleep(0 if warming else 0.7)
    _state["player"] = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    _state["live"] = False
    _state["player"] = False


app = FastAPI(title="pairsignal", version="0.1", lifespan=lifespan)


@app.get("/")
def dashboard():
    if not DASHBOARD.exists():
        raise HTTPException(404, "dashboard.html не найден")
    return FileResponse(DASHBOARD)


@app.get("/health")
def health():
    return {"ok": True, "live": _state["live"], "player": _state["player"]}


@app.get("/config")
def get_config():
    return cfg.model_dump()


@app.get("/pairs")
def pairs():
    """Список доступных пар для выбора на панели."""
    return {
        "pairs": [
            {"a": a, "b": b, "label": f"{_sym_short(a)} / {_sym_short(b)}"}
            for a, b in PAIRS
        ],
        "current": {"a": cfg.strategy.symbol_a, "b": cfg.strategy.symbol_b},
    }


def _current_preset() -> str | None:
    """Определить, какому пресету соответствует текущий конфиг (или None)."""
    s = cfg.strategy
    for key, p in PRESETS.items():
        if (s.entry_z == p["entry_z"] and s.stop_z == p["stop_z"]
                and s.profit_target_fees == p["profit_target_fees"]
                and s.bb_period == p["bb_period"]):
            return key
    return None


@app.get("/presets")
def presets():
    """Доступные пресеты параметров для панели."""
    return {
        "presets": [{"key": k, "label": p["label"], "desc": p["desc"]} for k, p in PRESETS.items()],
        "current": _current_preset(),
    }


@app.post("/preset")
def set_preset(key: str):
    """Применить пресет параметров (со сбросом сессии)."""
    if key not in PRESETS:
        raise HTTPException(400, f"неизвестный пресет: {key}")
    p = PRESETS[key]
    s = cfg.strategy
    s.entry_z = p["entry_z"]
    s.stop_z = p["stop_z"]
    s.profit_target_fees = p["profit_target_fees"]
    s.bb_period = p["bb_period"]
    s.min_width_pct = p["min_width_pct"]
    was_live, was_player = _state["live"], _state["player"]
    _state["live"] = _state["player"] = False
    _reset_engine()
    return {"ok": True, "preset": key, "config": cfg.model_dump(),
            "was_live": was_live, "was_player": was_player}


@app.get("/analyze")
async def analyze(period: str = "day"):
    """Моделирование стратегии на РЕАЛЬНОЙ истории MEXC за период.

    Прогоняет текущие параметры (auto-вход) по историческим барам и считает,
    сколько было бы входов и какая прибыль. Не трогает живую сессию панели —
    работает на отдельном Engine.
    """
    if period not in ("day", "week"):
        raise HTTPException(400, "period: day или week")
    # баров в окне периода по таймфрейму
    tf_min = 15 if cfg.strategy.timeframe == "15m" else 5
    per_day = 24 * 60 // tf_min
    window = per_day if period == "day" else per_day * 7
    limit = _warmup_limit() + window            # прогрев + окно анализа

    try:
        df = await asyncio.to_thread(read_ohlcv_ccxt, cfg.strategy, limit)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"не удалось получить историю MEXC: {e}")

    sim_cfg = AppConfig(**cfg.model_dump())
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
        "pair": {"a": _sym_short(cfg.strategy.symbol_a), "b": _sym_short(cfg.strategy.symbol_b)},
        "timeframe": cfg.strategy.timeframe,
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
def set_config(payload: dict):
    """Обновить пороги/сайзинг/таймфрейм в рантайме и сбросить сессию.

    Принимает любое подмножество: entry_z, stop_z, exit_z, risk_pct, timeframe.
    Возвращает применённый конфиг. Параметры применяются с чистого листа.
    """
    s, p = cfg.strategy, cfg.paper

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
    st = stop if stop is not None else s.stop_z
    ex = exit_ if exit_ is not None else s.exit_z
    if not (ex < e < st):
        raise HTTPException(400, f"нужно выход({ex}) < вход({e}) < стоп({st})")
    if tf is not None and tf not in ("5m", "15m"):
        raise HTTPException(400, "timeframe: только 5m или 15m")

    # пара бумаг — только из списка PAIRS (надёжные символы MEXC)
    sa, sb = payload.get("symbol_a"), payload.get("symbol_b")
    if sa is not None or sb is not None:
        sa = sa or s.symbol_a
        sb = sb or s.symbol_b
        if (sa, sb) not in PAIRS:
            raise HTTPException(400, f"пара {sa}/{sb} не из списка доступных")

    s.entry_z, s.stop_z, s.exit_z = e, st, ex
    if risk is not None:
        p.risk_pct = risk
    if tf is not None:
        s.timeframe = tf
    if sa is not None:
        s.symbol_a, s.symbol_b = sa, sb
    if "auto_approve" in payload:
        cfg.auto_approve = bool(payload["auto_approve"])

    was_live, was_player = _state["live"], _state["player"]
    _state["live"] = _state["player"] = False
    _reset_engine()
    return {"ok": True, "config": cfg.model_dump(), "was_live": was_live, "was_player": was_player}


@app.get("/state")
def state():
    pos = asdict(engine.exch.position) if engine.exch.position else None
    pending = asdict(engine.pending) if engine.pending else None
    return _clean({
        "live": _state["live"],
        "player": _state["player"],
        "auto_approve": cfg.auto_approve,     # режим авто-входа
        "preset": _current_preset(),          # активный пресет параметров
        "session_started": _state["session_started"],  # unix-сек старта торговой сессии
        "server_started": _server_started,    # unix-сек запуска сервера (uptime)
        "now": time.time(),                   # серверное «сейчас» (для расчёта длительности на клиенте)
        "pair": {                             # текущая пара (короткие подписи ног)
            "a": _sym_short(cfg.strategy.symbol_a),
            "b": _sym_short(cfg.strategy.symbol_b),
        },
        "pending_recommendation": pending,   # ждёт решения оператора
        "position": pos,
        "summary": engine.summary(),
        "history": _history,                  # последние бары для графика
        "trades": [asdict(t) for t in engine.exch.trades],
        "last": _state["last_rec"],
    })


@app.post("/auto")
def set_auto(on: bool = True):
    """Переключить авто-вход без сброса сессии. При включении сразу подтверждает
    висящую рекомендацию (если оператор уже ждал решения)."""
    cfg.auto_approve = on
    if on and engine.pending is not None:
        engine.approve()
    return {"ok": True, "auto_approve": cfg.auto_approve}


@app.post("/approve")
def approve():
    if not engine.pending:
        raise HTTPException(400, "нет ожидающей рекомендации")
    engine.approve()
    return {"ok": True, "position": asdict(engine.exch.position) if engine.exch.position else None}


@app.post("/reject")
def reject():
    engine.reject()
    return {"ok": True}


@app.post("/reset")
def reset():
    _reset_engine()
    return {"ok": True}


@app.post("/live/start")
async def live_start():
    if _state["live"]:
        return {"ok": True, "already": True}
    _state["player"] = False          # live и плеер взаимоисключаются
    _reset_engine()                   # чистый старт live с бэкфиллом
    _state["live"] = True
    asyncio.create_task(_poller())
    return {"ok": True, "mode": "live"}


@app.post("/live/stop")
def live_stop():
    _state["live"] = False
    return {"ok": True}


@app.post("/player/start")
async def player_start(limit: int = 2000):
    """Запустить синтетический плеер с чистого листа (human-in-the-loop)."""
    global _player_rows
    if _state["player"]:
        return {"ok": True, "already": True}
    _state["live"] = False            # плеер и live взаимоисключаются
    _reset_engine()
    df = generate_synthetic(n=limit)
    _player_rows = Engine.rows_from_df(df, cfg.strategy)
    _state["player"] = True
    asyncio.create_task(_player())
    return {"ok": True, "bars": len(_player_rows)}


@app.post("/player/stop")
def player_stop():
    _state["player"] = False
    return {"ok": True}


@app.post("/replay/synthetic")
def replay_synthetic(limit: int = 2000):
    """Прогнать синтетику целиком разом (для смоук-теста) — авто-подтверждение входов."""
    _reset_engine()
    cfg.auto_approve = True
    try:
        for row in Engine.rows_from_df(generate_synthetic(n=limit), cfg.strategy):
            engine.step(row)
            _push_history(row)
    finally:
        cfg.auto_approve = False
    return engine.summary()
