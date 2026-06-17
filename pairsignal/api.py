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
    # отобраны по коинтеграции (pairs_coint.py, OOS 2026-03..06): лучшие 2 из 8 пар,
    # прибыльные out-of-sample. ENA/WLD — win 86%/99 сделок; BTC/ETH (уже выше) — p<0.001.
    ("ENA/USDT:USDT", "WLD/USDT:USDT"),
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
                # для автостарта live после рестарта сервера (как у st4/st5)
                "live": self.state["live"],
                "paused_by_user": self.state["paused_by_user"],
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
            self.state["paused_by_user"] = bool(data.get("paused_by_user", False))
            # автостарт live: шёл на момент сохранения и не остановлен оператором
            self.state["resume_live"] = bool(data.get("live")) and not self.state["paused_by_user"]
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


from .st4 import data_feed as feed   # noqa: E402  st4 — FSM-арбитраж SBRF/SBPR
from .st4.backtest import band_frame_for_chart, run_backtest  # noqa: E402
from .st4.service import ST4_PAIRS, St4Session   # noqa: E402

# независимая форвард-тест сессия на каждую пару обычка/преф (?pair=, см. ST4_PAIRS)
ST4S: dict[str, St4Session] = {p: St4Session(p) for p in ST4_PAIRS}
ST4 = ST4S["sber"]                    # пара по умолчанию (совместимость)


def _st4(pair: str = "sber") -> St4Session:
    if pair not in ST4S:
        raise HTTPException(400, "pair: " + " | ".join(ST4S))
    return ST4S[pair]


from .st5 import data_feed as feed5   # noqa: E402  st5 — VWAP-reversion одиночного инструмента
from .st5.backtest import run_backtest as run_backtest5  # noqa: E402
from .st5.backtest import vwap_frame_for_chart  # noqa: E402
from .st5.service import ST5_INSTRUMENTS, St5Session   # noqa: E402

# независимая сессия на каждый инструмент st5 (?inst=, см. ST5_INSTRUMENTS)
ST5S: dict[str, St5Session] = {i: St5Session(i) for i in ST5_INSTRUMENTS}


def _st5(inst: str = "sber") -> St5Session:
    if inst not in ST5S:
        raise HTTPException(400, "inst: " + " | ".join(ST5S))
    return ST5S[inst]


@asynccontextmanager
async def lifespan(app: FastAPI):
    for st in SLOTS:
        st.load_session()             # восстановить результаты прошлой сессии (если есть)
        if st.state.pop("resume_live", False):
            # автостарт live после рестарта: продолжаем сессию без сброса журнала
            # (_poller сам догонит бары по last_live_ts), как у st4/st5
            st.state["live"] = True
            asyncio.create_task(_poller(st))
    from .st4 import tbank_sandbox as _sb
    _sb.load_token()                  # подтянуть сохранённый токен T-Bank (переживает рестарт)
    for s4 in ST4S.values():
        s4.load_session()
        if s4.state.pop("resume_live", False):
            asyncio.create_task(_st4_autoresume(s4))   # автостарт: live шёл до рестарта
    for s5 in ST5S.values():
        s5.load_session()
        if s5.state.pop("resume_live", False):
            asyncio.create_task(_st5_autoresume(s5))
    _auto_bt_task = asyncio.create_task(_auto_backtest_loop())   # авто-бэктест раз в ~2.5 дня
    yield
    _auto_bt_task.cancel()
    for st in SLOTS:
        st.save_session()             # сохранить при остановке сервера
        st.state["live"] = False
        st.state["player"] = False
    for s4 in ST4S.values():
        s4.save_session()
        s4.state["live"] = False
        s4.state["player"] = False
    for s5 in ST5S.values():
        s5.save_session()
        s5.state["live"] = False
        s5.state["player"] = False


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
    if "spread_mode" in payload:
        sm = payload["spread_mode"]
        if sm not in ("log", "ratio", "cross_pct"):
            raise HTTPException(400, "spread_mode: log | ratio | cross_pct")
        s.spread_mode = sm
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
    st.state["paused_by_user"] = False  # старт снимает намеренную остановку
    st.reset_engine()                 # чистый старт live с бэкфиллом
    st.state["live"] = True
    st.save_session()
    asyncio.create_task(_poller(st))
    return {"ok": True, "mode": "live"}


@app.post("/live/stop")
def live_stop(slot: int = 1):
    st = _slot(slot)
    st.state["live"] = False
    st.state["paused_by_user"] = True   # намеренная остановка — автостарт не возобновляет
    st.save_session()
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


# ============================================================================
# st3 — momentum (cross-sectional). Отдельная стратегия, НЕ через SlotState:
# портфельная (2k ног), векторная. Эндпоинты read-only + on-demand бэктест.
# ============================================================================

_MOM_STATE = _BASE / "pairsignal" / "out" / "momentum_fwd.json"
_MOM_CACHE: dict[str, dict] = {}  # кэш тяжёлых бэктестов/сравнений по ключу параметров


@app.get("/momentum/state")
def momentum_state():
    """Состояние live forward-test портфеля (читает momentum_fwd.json). Не падает без файла."""
    from .momentum import START_EQUITY, load_state

    loaded = load_state(_MOM_STATE)
    if loaded is None:
        return {"running": False}
    port, meta = loaded
    longs = sorted(s for s, w in port.weights.items() if w > 0)
    shorts = sorted(s for s, w in port.weights.items() if w < 0)
    # «жив» — если последняя отметка свежее 2 типичных интервалов бара (грубо, по last_ts)
    fresh = bool(port.last_ts) and (time.time() * 1000 - port.last_ts) < 3 * 3600_000
    return _clean({
        "running": fresh,
        "equity": round(port.equity, 2),
        "return_pct": round((port.equity / START_EQUITY - 1.0) * 100, 2),
        "start_equity": START_EQUITY,
        "meta": meta,
        "longs": [s.split("/")[0] for s in longs],
        "shorts": [s.split("/")[0] for s in shorts],
        "last_ts": port.last_ts,
        "rebalances": port.rebalances[-50:],  # последние 50 для графика/журнала
    })


@app.get("/momentum/daily")
def momentum_daily():
    """Ежедневная сводка: текущий статус + агрегация ребалансов по дням (для вкладки st3)."""
    from datetime import datetime, timezone

    from .momentum import START_EQUITY, load_state

    loaded = load_state(_MOM_STATE)
    if loaded is None:
        return {"running": False, "text": "Нет состояния — forward-test ещё не запускался.",
                "days": []}
    port, meta = loaded
    active = (time.time() * 1000 - port.last_ts) < 3 * 3600_000 if port.last_ts else False
    ret = (port.equity / START_EQUITY - 1.0) * 100
    longs = ", ".join(s.split("/")[0] for s, w in port.weights.items() if w > 0) or "—"
    shorts = ", ".join(s.split("/")[0] for s, w in port.weights.items() if w < 0) or "—"

    def _d(ms: int) -> str:
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

    last_reb = _d(port.rebalances[-1]["ts"]) + " " + datetime.fromtimestamp(
        port.rebalances[-1]["ts"] / 1000, tz=timezone.utc).strftime("%H:%M") if port.rebalances else "—"
    text = (
        f"{'🟢 активен' if active else '🔴 остановлен'} · "
        f"equity ${port.equity:.2f} ({'+' if ret >= 0 else ''}{ret:.2f}%) · "
        f"ребалансов {len(port.rebalances)} · lookback={meta.get('lookback')} "
        f"holding={meta.get('holding')}\nЛОНГ: {longs} | ШОРТ: {shorts} · "
        f"последний ребаланс: {last_reb}"
    )

    # агрегация ребалансов по дням: последний equity дня + число ребалансов в дне
    by_day: dict[str, dict] = {}
    for r in port.rebalances:
        d = _d(r["ts"])
        e = by_day.setdefault(d, {"date": d, "equity": r["equity"], "rebals": 0})
        e["equity"] = r["equity"]  # последний за день
        e["rebals"] += 1
    days = sorted(by_day.values(), key=lambda x: x["date"])
    prev = START_EQUITY
    for d in days:
        d["change_pct"] = round((d["equity"] / prev - 1.0) * 100, 2) if prev else 0.0
        d["equity"] = round(d["equity"], 2)
        prev = d["equity"]
    return _clean({"running": active, "text": text, "days": days})


def _run_backtest_mom(top: int, days: int, timeframe: str, k: int,
                      fee: float, slippage: float) -> dict:
    from .momentum import _BARS_PER_YEAR, walk_forward
    from .pairs_coint import load_universe
    from .scan_year import top_symbols

    until = int(time.time() * 1000)
    since = until - days * 24 * 3600 * 1000
    prices = load_universe(top_symbols(top), timeframe, since, until, 5, min_coverage=0.9)
    prices.attrs["timeframe"] = timeframe
    if prices.shape[1] < 2 * k or len(prices) < 500:
        return {"error": f"мало данных: {prices.shape[1]} монет, {len(prices)} баров"}
    bph = _BARS_PER_YEAR.get(timeframe, 365 * 24) / 365
    rows = walk_forward(prices, int(60 * bph), int(30 * bph),
                        [24, 48, 96, 240], [12, 24, 48], k, True, fee, slippage)
    # склеенная OOS equity
    eq, cum = [], 1.0
    for r in rows:
        seg = r["_oos_eq"] / r["_oos_eq"].iloc[0] * cum
        eq.extend([{"ts": int(t), "v": round(float(v), 4)} for t, v in seg.items()])
        cum = float(seg.iloc[-1])
    windows = [{kk: vv for kk, vv in r.items() if kk != "_oos_eq"} for r in rows]
    for w in windows:
        w["test_start"], w["test_end"] = int(w["test_start"]), int(w["test_end"])
    return {"windows": windows, "equity": eq,
            "total_return_pct": round((cum - 1.0) * 100, 2),
            "wins": sum(1 for r in rows if r["oos_return_pct"] > 0), "n": len(rows)}


@app.get("/momentum/backtest")
async def momentum_backtest(top: int = 30, days: int = 365, timeframe: str = "1h",
                            k: int = 3, fee: float = 0.0006, slippage: float = 0.0002):
    """Walk-forward бэктест на лету. Тяжёлый (загрузка) → to_thread + кэш по параметрам."""
    key = f"bt:{top}:{days}:{timeframe}:{k}:{fee}:{slippage}"
    if key not in _MOM_CACHE:
        _MOM_CACHE[key] = await asyncio.to_thread(
            _run_backtest_mom, top, days, timeframe, k, fee, slippage)
    return _clean(_MOM_CACHE[key])


def _run_compare_mom(top: int, days: int, timeframe: str) -> dict:
    from .compare_strategies import eq_buy_hold, eq_momentum, eq_pairs
    from .pairs_coint import load_universe
    from .scan_year import top_symbols

    until = int(time.time() * 1000)
    since = until - days * 24 * 3600 * 1000
    prices = load_universe(top_symbols(top), timeframe, since, until, 5, min_coverage=0.9)
    prices.attrs["timeframe"] = timeframe
    out: dict[str, list] = {}
    series = {"BuyHold": eq_buy_hold(prices), "Momentum": eq_momentum(prices, 0.0006, 0.0002)}
    pairs = eq_pairs(prices, 0.0006, 0.0002, timeframe)
    if pairs is not None:
        series["Pairs"] = pairs
    res = {}
    for name, s in series.items():
        s = s.dropna()
        s = s / s.iloc[0]
        out[name] = [{"ts": int(t), "v": round(float(v), 4)} for t, v in s.items()]
        res[name] = round((float(s.iloc[-1]) - 1.0) * 100, 2)
    return {"curves": out, "returns": res}


@app.get("/momentum/compare")
async def momentum_compare(top: int = 30, days: int = 365, timeframe: str = "1h"):
    """Сравнение equity Momentum/Pairs/BuyHold на одном периоде. to_thread + кэш."""
    key = f"cmp:{top}:{days}:{timeframe}"
    if key not in _MOM_CACHE:
        _MOM_CACHE[key] = await asyncio.to_thread(_run_compare_mom, top, days, timeframe)
    return _clean(_MOM_CACHE[key])


@app.get("/momentum/tests")
async def momentum_tests():
    """Статус юнит-тестов momentum: запускает pytest по test_momentum*.py, парсит итог."""
    import re
    import subprocess
    import sys

    def _run() -> dict:
        try:
            # без -q, чтобы pytest напечатал итоговую строку "N passed[, M failed]"
            p = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/test_momentum.py",
                 "tests/test_momentum_live.py", "tests/test_momentum_api.py",
                 "--no-header", "-p", "no:cacheprovider"],
                cwd=str(_BASE), capture_output=True, text=True, timeout=180)
            out = p.stdout + p.stderr
            passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", out)) else 0
            failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", out)) else 0
            return {"passed": passed, "failed": failed, "ok": failed == 0 and passed > 0,
                    "tail": out.strip().splitlines()[-1:] if out else []}
        except Exception as e:  # noqa: BLE001
            return {"passed": 0, "failed": 0, "ok": False, "error": str(e)}

    return await asyncio.to_thread(_run)


# ============================================================================
# st4 — арбитраж спреда фьючерсов SBRF/SBPR (FORTS, MOEX ISS). FSM-движок,
# paper-исполнение с атомарностью пар. Отдельная сессия (St4Session), не SlotState.
# ============================================================================
from datetime import datetime as _dt  # noqa: E402
from datetime import timedelta as _td  # noqa: E402
from datetime import timezone as _tz  # noqa: E402

_MSK = _tz(_td(hours=3))  # московское время для меток st4


@app.get("/st4/pairs")
def st4_pairs():
    """Список доступных пар обычка/преф — фронт строит переключатель динамически
    (новая пара добавляется ТОЛЬКО в ST4_PAIRS, UI подхватывает сам)."""
    return {"pairs": [{"id": pid, "ord": spec[0], "pref": spec[1], "label": spec[2]}
                      for pid, spec in ST4_PAIRS.items()]}


@app.get("/st4/state")
def st4_state(pair: str = "sber"):
    return _st4(pair).snapshot(_server_started)


@app.get("/st4/config")
def st4_config(pair: str = "sber"):
    return _st4(pair).cfg.model_dump()


@app.post("/st4/config")
async def st4_set_config(payload: dict, pair: str = "sber"):
    """Обновить параметры стратегии/риска/исполнения (валидация) и сбросить сессию.

    Если до применения был активен live/демо — автоматически перезапускаем его с новыми
    параметрами (чтобы не нажимать «live» вручную после каждого изменения)."""
    ST4 = _st4(pair)
    s = ST4.cfg.strategy
    r = ST4.cfg.risk
    e = ST4.cfg.execution

    def _num(key, lo, hi, cur):
        if key not in payload or payload[key] is None:
            return cur
        try:
            v = float(payload[key])
        except (TypeError, ValueError):
            raise HTTPException(400, f"{key}: не число")
        if not (lo <= v <= hi):
            raise HTTPException(400, f"{key}: вне диапазона [{lo}, {hi}]")
        return v

    s.sma_period = int(_num("sma_period", 20, 1000, s.sma_period))
    s.sigma_multiplier = _num("sigma_multiplier", 0.5, 5.0, s.sigma_multiplier)
    s.deviation_pct = _num("deviation_pct", 0.0, 0.2, s.deviation_pct)
    s.stop_sigma = _num("stop_sigma", 0.0, 10.0, s.stop_sigma)
    s.max_bars_in_trade = int(_num("max_bars_in_trade", 0, 100000, s.max_bars_in_trade))
    s.deviation_sigma = _num("deviation_sigma", 0.0, 10.0, s.deviation_sigma)
    s.pending_ttl_bars = int(_num("pending_ttl_bars", 1, 100, s.pending_ttl_bars))
    if "deviation_mode" in payload:
        if payload["deviation_mode"] not in ("AbsOfMean", "LiteralPct", "Sigma"):
            raise HTTPException(400, "deviation_mode: AbsOfMean | LiteralPct | Sigma")
        s.deviation_mode = payload["deviation_mode"]
    if "entry_trigger" in payload:
        if payload["entry_trigger"] not in ("Breakout", "ReEntry"):
            raise HTTPException(400, "entry_trigger: Breakout | ReEntry")
        s.entry_trigger = payload["entry_trigger"]
    if "freeze_sma_on_exit" in payload:
        s.freeze_sma_on_exit = bool(payload["freeze_sma_on_exit"])
    if "interval_min" in payload:
        iv = int(payload["interval_min"])
        if iv not in (1, 10, 60):
            raise HTTPException(400, "interval_min: 1 | 10 | 60 (ISS не отдаёт 5m)")
        s.candle_interval_minutes = iv
    r.max_daily_loss_rub = _num("max_daily_loss_rub", 0, 1e9, r.max_daily_loss_rub)
    e.quantity_lots = int(_num("quantity_lots", 1, 1000, e.quantity_lots))
    if "auto_approve" in payload:
        ST4.cfg.auto_approve = bool(payload["auto_approve"])

    was_live, was_player = ST4.state["live"], ST4.state["player"]
    ST4.state["live"] = ST4.state["player"] = False
    # авто-перезапуск: если был live — поднять заново в фоне (резолв+прогрев), если демо — плеер
    if was_live:
        ST4.state["live"] = True
        ST4.log_event("info", "параметры применены — перезапуск live в фоне")

        async def _boot():
            await asyncio.to_thread(ST4.reset_engine, True)
            if ST4.state["live"]:
                await ST4.run_live()

        asyncio.create_task(_boot())
    elif was_player:
        ST4.reset_engine(real=False)
        ST4.state["player"] = True
        ST4.player_df = feed.generate_synthetic(
            n=1500, interval_min=ST4.cfg.strategy.candle_interval_minutes)
        ST4.player_idx = 0
        asyncio.create_task(ST4.run_player())
    else:
        ST4.reset_engine(real=(ST4.state["data_source"] == "live"))
    return {"ok": True, "config": ST4.cfg.model_dump(), "was_live": was_live,
            "was_player": was_player, "restarted": was_live or was_player}


async def _st4_autoresume(ST4: St4Session):
    """Автостарт live после рестарта сервера: ПРОДОЛЖАЕМ сессию без сброса журнала.

    Вызывается из lifespan для каждой пары, чья сессия была в live и не остановлена
    оператором. Paper: восстановленный движок продолжает (BB прогревается в run_live
    по last_live_ts). Sandbox: исполнителю нужен счёт — полный старт через reset_engine.
    """
    ST4.state["player"] = False
    ST4.state["data_source"] = "live"
    ST4.state["live"] = True
    ST4.log_event("info", "автовозобновление live после рестарта сервера")
    if ST4.cfg.connector.mode == "tbank_sandbox":
        prev = ST4.engine                       # восстановленная сессия (журнал/баланс)
        started = ST4.state["session_started"]
        await asyncio.to_thread(ST4.reset_engine, True)
        # продолжение форвард-теста: reset нужен только ради sandbox-счёта — журнал,
        # баланс и день риска переносим. Позицию не переносим: reconciliation в
        # reset_engine уже привела счёт к FLAT.
        eng = ST4.engine
        eng.trades = prev.trades
        eng.balance_rub = prev.balance_rub
        eng.risk.day_pnl_rub = prev.risk.day_pnl_rub
        eng.risk._day = prev.risk._day
        ST4.state["session_started"] = started
        ST4.save_session()
    if ST4.state["live"]:
        await ST4.run_live()


@app.post("/st4/control/start")
async def st4_start(pair: str = "sber"):
    """Запустить live. Тяжёлый старт (резолв серий + sandbox-счёт + pay_in) — в фоне,
    чтобы HTTP-ответ вернулся сразу, а не висел десятки секунд (sandbox делает сетевые вызовы)."""
    ST4 = _st4(pair)
    if ST4.state["live"]:
        return {"ok": True, "already": True}
    ST4.state["player"] = False
    ST4.state["data_source"] = "live"
    ST4.state["live"] = True
    ST4.state["paused_by_user"] = False  # старт снимает намеренную остановку
    ST4.log_event("info", "запуск live… (резолв инструментов и счёта в фоне)")

    async def _boot():
        await asyncio.to_thread(ST4.reset_engine, True)   # сеть — в пуле потоков
        if ST4.state["live"]:                              # не отменили за время старта
            await ST4.run_live()

    asyncio.create_task(_boot())
    return {"ok": True, "mode": "live", "starting": True}


@app.post("/st4/control/stop")
def st4_stop(pair: str = "sber"):
    ST4 = _st4(pair)
    ST4.state["live"] = False
    ST4.state["paused_by_user"] = True   # намеренная остановка — автостарт не возобновляет
    ST4.save_session()   # зафиксировать флаги: иначе крэш до следующего save возобновит live
    return {"ok": True}


@app.post("/st4/player/start")
async def st4_player_start(limit: int = 1500, pair: str = "sber"):
    """Запустить/возобновить синтетический плеер (офлайн-демо FSM)."""
    ST4 = _st4(pair)
    if ST4.state["player"]:
        return {"ok": True, "already": True}
    ST4.state["live"] = False
    ST4.state["paused_by_user"] = False
    ST4.state["data_source"] = "synthetic"
    resuming = ST4.player_df is not None and ST4.player_idx < len(ST4.player_df)
    if not resuming:
        ST4.reset_engine(real=False)
        iv = ST4.cfg.strategy.candle_interval_minutes
        ST4.player_df = feed.generate_synthetic(n=limit, interval_min=iv)
        ST4.player_idx = 0
    ST4.state["player"] = True
    ST4.save_session()   # зафиксировать data_source=synthetic (автостарт не должен поднять live)
    asyncio.create_task(ST4.run_player())
    return {"ok": True, "resumed": resuming}


@app.post("/st4/player/stop")
def st4_player_stop(pair: str = "sber"):
    ST4 = _st4(pair)
    ST4.state["player"] = False
    ST4.state["paused_by_user"] = True
    return {"ok": True}


@app.post("/st4/control/flat-all")
def st4_flat_all(payload: dict | None = None, pair: str = "sber"):
    """Паник-закрытие позиции по рынку (требует confirm=true в теле, §12.1)."""
    ST4 = _st4(pair)
    if not payload or not payload.get("confirm"):
        raise HTTPException(400, "нужно подтверждение: {\"confirm\": true}")
    trade = ST4.engine.flat_all("flat_all")
    ST4.save_session()
    return {"ok": True, "closed": trade is not None,
            "net_pnl_rub": round(trade.net_pnl_rub, 0) if trade else None}


@app.post("/st4/control/trading")
def st4_trading(on: bool = True, pair: str = "sber"):
    """Глобальный флаг новых входов (TradingEnabled, §11)."""
    ST4 = _st4(pair)
    ST4.cfg.risk.trading_enabled = on
    return {"ok": True, "trading_enabled": on}


@app.post("/st4/control/resume")
def st4_resume(pair: str = "sber"):
    """Снять HALTED после ручного разбора (§11)."""
    ST4 = _st4(pair)
    ST4.engine.risk.resume()
    from .st4.models import BotState
    if ST4.engine.state == BotState.HALTED:
        ST4.engine.state = BotState.FLAT
    return {"ok": True, "halted": ST4.engine.risk.halted}


@app.post("/st4/approve")
def st4_approve(pair: str = "sber"):
    ST4 = _st4(pair)
    if ST4.engine._pending is None:
        raise HTTPException(400, "нет ожидающей рекомендации")
    ST4.engine.approve()
    ST4.save_session()
    return {"ok": True}


@app.post("/st4/reject")
def st4_reject(pair: str = "sber"):
    ST4 = _st4(pair)
    ST4.engine.reject()
    return {"ok": True}


@app.post("/st4/auto")
def st4_auto(on: bool = True, pair: str = "sber"):
    ST4 = _st4(pair)
    ST4.cfg.auto_approve = on
    if on and ST4.engine._pending is not None:
        ST4.engine.approve()
        ST4.save_session()
    return {"ok": True, "auto_approve": on}


@app.post("/st4/reset")
def st4_reset(pair: str = "sber"):
    ST4 = _st4(pair)
    ST4.reset_engine(real=(ST4.state["data_source"] == "live"))
    return {"ok": True}


@app.post("/st4/restore-position")
def st4_restore_position(payload: dict, pair: str = "sber"):
    """Восстановить открытую позицию в ЖИВОМ движке (без файловой гонки).

    Для случая, когда позиция «осиротела»: висит на sandbox-счёте, а движок её потерял
    (напр. reconciliation не смог закрыть в неторговое время). Ставим позицию прямо в
    объект движка в памяти и сохраняем сессию. payload: state(long_spread|short_spread),
    ord_code, pref_code, lots, ord_entry, pref_entry [, sma_at_entry, bars_held].
    """
    from .st4.models import BotState, LegPosition, Position, Role
    ST4 = _st4(pair)
    try:
        state = BotState(payload["state"])
        if state not in (BotState.LONG_SPREAD, BotState.SHORT_SPREAD):
            raise HTTPException(400, "state: long_spread | short_spread")
        lots = int(payload["lots"])
        # long_spread = обычка buy / преф sell; short_spread — наоборот
        ord_side = "buy" if state == BotState.LONG_SPREAD else "sell"
        pref_side = "sell" if state == BotState.LONG_SPREAD else "buy"
        ord_entry = float(payload["ord_entry"])
        pref_entry = float(payload["pref_entry"])
        leg_ord = LegPosition(code=payload["ord_code"], role=Role.ORDINARY, side=ord_side,
                              lots=lots, entry_price=ord_entry)
        leg_pref = LegPosition(code=payload["pref_code"], role=Role.PREFERRED, side=pref_side,
                               lots=lots, entry_price=pref_entry)
        ST4.engine.position = Position(
            state=state, leg_ord=leg_ord, leg_pref=leg_pref,
            entry_ts=int(payload.get("entry_ts", 0)),
            entry_spread=pref_entry - ord_entry, entry_beta=1.0,
            sma_at_entry=float(payload.get("sma_at_entry", pref_entry - ord_entry)),
            entry_fee_rub=float(payload.get("entry_fee_rub", 0.0)))
        ST4.engine.state = state
        ST4.engine._bars_held = int(payload.get("bars_held", 0))
        ST4.save_session()
        return {"ok": True, "position": ST4.engine.state.value, "lots": lots}
    except KeyError as e:
        raise HTTPException(400, f"нет поля: {e}")


@app.post("/st4/connector")
def st4_connector(payload: dict, pair: str = "sber"):
    """Установить режим исполнителя (paper|tbank_sandbox) и (опц.) API-токен T-Bank.

    Токен сохраняется в файл .tbank_token (0600, в .gitignore — не в git) и в env процесса,
    чтобы переживать рестарт. В ответе НЕ возвращается. Sandbox активен только в live;
    при недоступности sandbox reset откатывает в paper.
    """
    from .st4 import tbank_sandbox as _sb

    ST4 = _st4(pair)
    mode = payload.get("mode")
    if mode not in ("paper", "tbank_sandbox"):
        raise HTTPException(400, "mode: paper | tbank_sandbox")
    token = (payload.get("token") or "").strip()
    if token:
        _sb.save_token(token)                        # в файл (0600) + env
    if mode == "tbank_sandbox" and not _sb.has_token():
        raise HTTPException(400, "для sandbox нужен токен (вставьте в поле API-токен)")
    if "payin_rub" in payload:
        try:
            ST4.cfg.connector.payin_rub = int(payload["payin_rub"])
        except (TypeError, ValueError):
            raise HTTPException(400, "payin_rub: не число")
    ST4.cfg.connector.mode = mode
    ST4.state["live"] = ST4.state["player"] = False
    ST4.reset_engine(real=(ST4.state["data_source"] == "live"))
    return {"ok": True, "connector_mode": ST4.cfg.connector.mode,
            "token_set": _sb.has_token(),
            "fell_back": ST4.cfg.connector.mode != mode}


@app.post("/st4/connector/forget-token")
def st4_forget_token():
    """Удалить сохранённый токен (из файла и env). Откатить в paper ВСЕ пары (токен общий)."""
    from .st4 import tbank_sandbox as _sb
    _sb.save_token("")                               # пусто → удаляет файл + env
    for s4 in ST4S.values():
        if s4.cfg.connector.mode != "paper":
            s4.cfg.connector.mode = "paper"
            s4.reset_engine(real=(s4.state["data_source"] == "live"))
    return {"ok": True, "token_set": False, "connector_mode": "paper"}


@app.get("/st4/trades")
def st4_trades(pair: str = "sber"):
    ST4 = _st4(pair)
    return {"trades": [ST4._trade_json(t) for t in ST4.engine.trades]}


@app.get("/st4/daily")
def st4_daily(pair: str = "sber"):
    """Доходность по дням: дата (МСК), net P&L за день (₽) и число сделок в день.

    Группировка по дню ВЫХОДА сделки (exit_ts) в московском времени — день торгов FORTS.
    """
    ST4 = _st4(pair)
    by_day: dict[str, dict] = {}
    for t in ST4.engine.trades:
        d = _dt.fromtimestamp(t.exit_ts / 1000, tz=_MSK).strftime("%Y-%m-%d")
        e = by_day.setdefault(d, {"date": d, "net_pnl_rub": 0.0, "trades": 0, "wins": 0})
        e["net_pnl_rub"] += t.net_pnl_rub
        e["trades"] += 1
        if t.net_pnl_rub > 0:
            e["wins"] += 1
    days = sorted(by_day.values(), key=lambda x: x["date"])
    cum = 0.0
    for d in days:
        cum += d["net_pnl_rub"]
        d["net_pnl_rub"] = round(d["net_pnl_rub"], 0)
        d["cum_pnl_rub"] = round(cum, 0)             # накопленный P&L для контекста
    return _clean({"pair": pair, "days": days})


@app.get("/st4/backtest")
async def st4_backtest(days: int = 90, stop_sigma: float | None = None, pair: str = "sber"):
    """Бэктест на реальной истории MOEX ISS за period (honest: maxDD по equity)."""
    ST4 = _st4(pair)

    def _run() -> dict:
        # спецификации ног резолвим ЛОКАЛЬНО: бэктест не должен трогать живую сессию
        # (resolve_real_legs перезаписывает ST4.spec_ord/spec_pref)
        try:
            spec_ord, spec_pref = feed.resolve_legs(ST4.cfg)
        except Exception as e:  # noqa: BLE001
            return {"error": f"не удалось определить серии: {e}"}
        since = _dt.now(_tz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        since = since.fromtimestamp(since.timestamp() - days * 86400, tz=_tz.utc)
        try:
            df = feed.read_ohlcv_moex_range(ST4.cfg, since, spec_ord.code, spec_pref.code)
        except Exception as e:  # noqa: BLE001
            return {"error": f"не удалось получить историю: {e}"}
        if len(df) < ST4.cfg.strategy.sma_period + 20:
            return {"error": f"мало данных: {len(df)} баров (нужно > {ST4.cfg.strategy.sma_period})"}
        from .st4.config import St4Config as _Cfg
        bt_cfg = _Cfg(**ST4.cfg.model_dump())
        if stop_sigma is not None:
            bt_cfg.strategy.stop_sigma = stop_sigma
        res = run_backtest(df, bt_cfg, spec_ord, spec_pref)
        res["bands"] = band_frame_for_chart(df, bt_cfg)[-400:]
        res["legs"] = {"ord": spec_ord.code, "pref": spec_pref.code}
        # запись в историю прогонов ISS (отдельно от T-Bank)
        from .st4.service import bt_history_append
        entry = {
            "date": _dt.now(_MSK).strftime("%Y-%m-%d %H:%M"),
            "days": days,
            "stop_sigma": stop_sigma if stop_sigma is not None else ST4.cfg.strategy.stop_sigma,
            "bars": res["bars"], "trades": res["trades"], "win_rate_pct": res["win_rate_pct"],
            "net_pnl_rub": res["net_pnl_rub"], "return_pct": res["return_pct"],
            "max_drawdown_pct": res["max_drawdown_pct"], "stops": res["stops"],
        }
        res["history"] = bt_history_append(entry, source="moex", pair=ST4.pair)
        return res

    return _clean(await asyncio.to_thread(_run))


def _run_backtest_tbank(stop_sigma: float | None, ST4: St4Session) -> dict:
    """Прогон бэктеста на T-Bank-свечах за неделю + запись в историю. Блокирующий (to_thread)."""
    from .st4 import tbank_sandbox as _sb
    if not _sb.has_token():
        return {"error": "нужен токен T-Bank (вставьте в блоке «Коннектор»)"}
    # спецификации ног — локально, не трогая ST4.spec_ord/spec_pref живой сессии
    try:
        spec_ord, spec_pref = feed.resolve_legs(ST4.cfg)
        it_o = _sb.find_future(spec_ord.code)
        it_p = _sb.find_future(spec_pref.code)
    except Exception as e:  # noqa: BLE001
        return {"error": f"T-Bank: {e}"}
    try:
        df = feed.read_ohlcv_tbank(ST4.cfg, 1000, _sb._uid(it_o), _sb._uid(it_p))
    except Exception as e:  # noqa: BLE001
        return {"error": f"не удалось получить свечи T-Bank: {e}"}
    if len(df) < ST4.cfg.strategy.sma_period + 20:
        return {"error": f"мало данных: {len(df)} баров (нужно > {ST4.cfg.strategy.sma_period})."}
    from .st4.config import St4Config as _Cfg
    from .st4.service import bt_history_append
    bt_cfg = _Cfg(**ST4.cfg.model_dump())
    if stop_sigma is not None:
        bt_cfg.strategy.stop_sigma = stop_sigma
    res = run_backtest(df, bt_cfg, spec_ord, spec_pref)
    res["legs"] = {"ord": spec_ord.code, "pref": spec_pref.code}
    res["source"] = "T-Bank real-time"
    entry = {
        "date": _dt.now(_MSK).strftime("%Y-%m-%d %H:%M"),
        "stop_sigma": stop_sigma if stop_sigma is not None else ST4.cfg.strategy.stop_sigma,
        "bars": res["bars"], "trades": res["trades"], "win_rate_pct": res["win_rate_pct"],
        "net_pnl_rub": res["net_pnl_rub"], "return_pct": res["return_pct"],
        "max_drawdown_pct": res["max_drawdown_pct"], "stops": res["stops"],
    }
    res["history"] = bt_history_append(entry, pair=ST4.pair)
    return res


@app.get("/st4/backtest_tbank")
async def st4_backtest_tbank(stop_sigma: float | None = None, pair: str = "sber"):
    """Бэктест на РЕАЛЬНЫХ котировках T-Bank за неделю (тот же источник, что sandbox-ордера)."""
    return _clean(await asyncio.to_thread(_run_backtest_tbank, stop_sigma, _st4(pair)))


async def _auto_backtest_loop():
    """Авто-бэктест раз в ~2.5 дня: копит историю результативности на свежих данных T-Bank.

    Срабатывает только если есть токен. Тихо логирует в события st4.
    """
    import asyncio as _aio
    await _aio.sleep(120)                 # первый прогон — через 2 мин после старта
    while True:
        try:
            from .st4 import tbank_sandbox as _sb
            if _sb.has_token():
                for s4 in ST4S.values():
                    res = await _aio.to_thread(_run_backtest_tbank, None, s4)
                    if "error" not in res:
                        s4.log_event("info", f"авто-бэктест T-Bank: сделок {res['trades']}, "
                                     f"net {res['net_pnl_rub']:+.0f}₽, win {res['win_rate_pct']}%")
        except Exception:  # noqa: BLE001
            pass
        await _aio.sleep(2.5 * 24 * 3600)  # раз в 2.5 дня


@app.get("/st4/backtest_history")
def st4_backtest_history(source: str = "tbank", pair: str = "sber"):
    """История прогонов бэктеста (source: tbank | moex) — результативность во времени."""
    from .st4.service import bt_history_load
    if source not in ("tbank", "moex"):
        raise HTTPException(400, "source: tbank | moex")
    return {"history": bt_history_load(source, _st4(pair).pair)}


# --- скан пар обычка/преф FORTS (отдельная страница-отчёт) ---
ST4_REPORT_HTML = Path(__file__).resolve().parent.parent / "st4_report.html"
_ST4_SCAN = {"running": False, "error": None}


@app.get("/st4/report")
def st4_report_page():
    """Страница отчёта скана пар (ссылка с вкладки st4)."""
    if not ST4_REPORT_HTML.exists():
        raise HTTPException(404, "st4_report.html не найден")
    return FileResponse(ST4_REPORT_HTML)


@app.get("/st4/scan/report")
def st4_scan_report():
    """Последний результат скана пар + статус текущего прогона."""
    from .st4.scan_pairs import OUT_JSON
    rep = None
    if OUT_JSON.exists():
        try:
            rep = json.loads(OUT_JSON.read_text())
        except Exception:  # noqa: BLE001
            rep = None
    return {"report": rep, "running": _ST4_SCAN["running"], "error": _ST4_SCAN["error"]}


@app.post("/st4/scan/run")
async def st4_scan_run(days: int = 60, stop_sigma: float | None = None, pair: str = "sber"):
    """Запустить скан в фоне (ISS медленный — минуты). Параметры стратегии — текущие st4."""
    ST4 = _st4(pair)
    if _ST4_SCAN["running"]:
        return {"ok": True, "already": True}
    _ST4_SCAN["running"] = True
    _ST4_SCAN["error"] = None

    async def _job():
        try:
            from .st4.scan_pairs import run_scan
            rep = await asyncio.to_thread(run_scan, days, stop_sigma, None, ST4.cfg)
            ok = sum(1 for r in rep["rows"] if "error" not in r)
            ST4.log_event("info", f"скан пар FORTS завершён: {ok}/{len(rep['rows'])} пар, {days}д")
        except Exception as e:  # noqa: BLE001
            _ST4_SCAN["error"] = str(e)
        finally:
            _ST4_SCAN["running"] = False

    asyncio.create_task(_job())
    return {"ok": True, "started": True}


@app.get("/st4/margin")
async def st4_margin(pair: str = "sber"):
    """Гарантийное обеспечение пары SRM6+SPM6 и расчёт для 1/5/10/100 контрактов.

    ГО — INITIALMARGIN из ISS (актуальное, меняется ежедневно). Для парной позиции биржа
    обычно даёт скидку (ноги коррелированы) — показываем ГО без скидки (верхняя граница).
    """
    ST4 = _st4(pair)

    def _run() -> dict:
        try:
            ST4.resolve_real_legs()
            m_ord = feed.leg_margin(ST4.spec_ord.code)
            m_pref = feed.leg_margin(ST4.spec_pref.code)
        except Exception as e:  # noqa: BLE001
            return {"error": f"не удалось получить ГО: {e}"}
        pair = m_ord + m_pref
        # баланс sandbox (если есть токен) — для расчёта % использования
        balance = None
        try:
            from .st4 import tbank_sandbox as _sb
            if _sb.has_token() and ST4.cfg.connector.account_id:
                pf = _sb.portfolio(ST4.cfg.connector.account_id)
                q = pf.get("totalAmountPortfolio")
                balance = (int(q.get("units", 0)) + int(q.get("nano", 0)) / 1e9) if q else None
        except Exception:  # noqa: BLE001
            pass
        rows = []
        for n in (1, 5, 10, 100):
            margin = round(pair * n)
            rows.append({
                "lots": n, "margin_rub": margin,
                "pct_of_balance": round(100 * margin / balance, 1) if balance else None,
            })
        return {
            "legs": {"ord": ST4.spec_ord.code, "pref": ST4.spec_pref.code},
            "margin_ord": round(m_ord), "margin_pref": round(m_pref),
            "margin_pair": round(pair),
            "balance_rub": round(balance) if balance else None,
            "rows": rows,
        }

    return _clean(await asyncio.to_thread(_run))


@app.get("/st4/tests")
async def st4_tests():
    """Статус юнит-тестов st4."""
    import re
    import subprocess
    import sys

    def _run() -> dict:
        try:
            p = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/test_st4.py",
                 "--no-header", "-p", "no:cacheprovider"],
                cwd=str(_BASE), capture_output=True, text=True, timeout=180)
            out = p.stdout + p.stderr
            passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", out)) else 0
            failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", out)) else 0
            return {"passed": passed, "failed": failed, "ok": failed == 0 and passed > 0,
                    "tail": out.strip().splitlines()[-1:] if out else []}
        except Exception as e:  # noqa: BLE001
            return {"passed": 0, "failed": 0, "ok": False, "error": str(e)}

    return await asyncio.to_thread(_run)


# ============================================================================
# st5 — VWAP-reversion на одиночном инструменте (FORTS, MOEX ISS). Один фьючерс,
# отклонение цены от внутридневного VWAP. Paper-only (Phase 1). Сессия на инструмент.
# ============================================================================


@app.get("/st5/instruments")
def st5_instruments():
    """Список инструментов — фронт строит переключатель динамически."""
    return {"instruments": [{"id": iid, "asset": a, "label": lbl}
                            for iid, (a, lbl) in ST5_INSTRUMENTS.items()]}


@app.get("/st5/state")
def st5_state(inst: str = "sber"):
    return _st5(inst).snapshot(_server_started)


@app.get("/st5/config")
def st5_config(inst: str = "sber"):
    return _st5(inst).cfg.model_dump()


@app.post("/st5/config")
async def st5_set_config(payload: dict, inst: str = "sber"):
    """Обновить параметры стратегии/риска и перезапустить активный поток."""
    ST5 = _st5(inst)
    s = ST5.cfg.strategy
    r = ST5.cfg.risk
    e = ST5.cfg.execution

    def _num(key, lo, hi, cur):
        if key not in payload or payload[key] is None:
            return cur
        try:
            v = float(payload[key])
        except (TypeError, ValueError):
            raise HTTPException(400, f"{key}: не число")
        if not (lo <= v <= hi):
            raise HTTPException(400, f"{key}: вне диапазона [{lo}, {hi}]")
        return v

    s.band_sigma = _num("band_sigma", 0.5, 5.0, s.band_sigma)
    s.stop_sigma = _num("stop_sigma", 0.0, 10.0, s.stop_sigma)
    s.take_profit_sigma = _num("take_profit_sigma", 0.0, 5.0, s.take_profit_sigma)
    s.min_bars_in_day = int(_num("min_bars_in_day", 2, 100, s.min_bars_in_day))
    s.max_bars_in_trade = int(_num("max_bars_in_trade", 0, 100000, s.max_bars_in_trade))
    s.pending_ttl_bars = int(_num("pending_ttl_bars", 1, 100, s.pending_ttl_bars))
    s.volume_filter_mult = _num("volume_filter_mult", 0.0, 10.0, s.volume_filter_mult)
    s.max_data_lag_min = _num("max_data_lag_min", 0.0, 1440.0, s.max_data_lag_min)
    if "entry_trigger" in payload:
        if payload["entry_trigger"] not in ("Breakout", "ReEntry"):
            raise HTTPException(400, "entry_trigger: Breakout | ReEntry")
        s.entry_trigger = payload["entry_trigger"]
    if "flat_at_session_end" in payload:
        s.flat_at_session_end = bool(payload["flat_at_session_end"])
    if "interval_min" in payload:
        iv = int(payload["interval_min"])
        if iv not in (1, 10, 60):
            raise HTTPException(400, "interval_min: 1 | 10 | 60 (ISS не отдаёт 5m)")
        s.candle_interval_minutes = iv
    r.max_daily_loss_rub = _num("max_daily_loss_rub", 0, 1e9, r.max_daily_loss_rub)
    e.quantity_lots = int(_num("quantity_lots", 1, 1000, e.quantity_lots))
    if "auto_approve" in payload:
        ST5.cfg.auto_approve = bool(payload["auto_approve"])

    was_live, was_player = ST5.state["live"], ST5.state["player"]
    ST5.state["live"] = ST5.state["player"] = False
    if was_live:
        ST5.state["live"] = True
        ST5.log_event("info", "параметры применены — перезапуск live в фоне")

        async def _boot():
            await asyncio.to_thread(ST5.reset_engine, True)
            if ST5.state["live"]:
                await ST5.run_live()

        asyncio.create_task(_boot())
    elif was_player:
        ST5.reset_engine(real=False)
        ST5.state["player"] = True
        ST5.player_df = feed5.generate_synthetic(
            n=1500, interval_min=ST5.cfg.strategy.candle_interval_minutes)
        ST5.player_idx = 0
        asyncio.create_task(ST5.run_player())
    else:
        ST5.reset_engine(real=(ST5.state["data_source"] == "live"))
    return {"ok": True, "config": ST5.cfg.model_dump(), "was_live": was_live,
            "was_player": was_player, "restarted": was_live or was_player}


async def _st5_autoresume(ST5: St5Session):
    """Автостарт live после рестарта сервера (paper — продолжение сессии без сброса)."""
    ST5.state["player"] = False
    ST5.state["data_source"] = "live"
    ST5.state["live"] = True
    ST5.log_event("info", "автовозобновление live после рестарта сервера")
    await ST5.run_live()


@app.post("/st5/control/start")
async def st5_start(inst: str = "sber"):
    ST5 = _st5(inst)
    if ST5.state["live"]:
        return {"ok": True, "already": True}
    ST5.state["player"] = False
    ST5.state["data_source"] = "live"
    ST5.state["live"] = True
    ST5.state["paused_by_user"] = False
    ST5.log_event("info", "запуск live… (резолв серии в фоне)")

    async def _boot():
        await asyncio.to_thread(ST5.reset_engine, True)
        if ST5.state["live"]:
            await ST5.run_live()

    asyncio.create_task(_boot())
    return {"ok": True, "mode": "live", "starting": True}


@app.post("/st5/control/stop")
def st5_stop(inst: str = "sber"):
    ST5 = _st5(inst)
    ST5.state["live"] = False
    ST5.state["paused_by_user"] = True
    ST5.save_session()
    return {"ok": True}


@app.post("/st5/player/start")
async def st5_player_start(limit: int = 1500, inst: str = "sber"):
    ST5 = _st5(inst)
    if ST5.state["player"]:
        return {"ok": True, "already": True}
    ST5.state["live"] = False
    ST5.state["paused_by_user"] = False
    ST5.state["data_source"] = "synthetic"
    resuming = ST5.player_df is not None and ST5.player_idx < len(ST5.player_df)
    if not resuming:
        ST5.reset_engine(real=False)
        iv = ST5.cfg.strategy.candle_interval_minutes
        ST5.player_df = feed5.generate_synthetic(n=limit, interval_min=iv)
        ST5.player_idx = 0
    ST5.state["player"] = True
    ST5.save_session()
    asyncio.create_task(ST5.run_player())
    return {"ok": True, "resumed": resuming}


@app.post("/st5/player/stop")
def st5_player_stop(inst: str = "sber"):
    ST5 = _st5(inst)
    ST5.state["player"] = False
    ST5.state["paused_by_user"] = True
    return {"ok": True}


@app.post("/st5/control/flat-all")
def st5_flat_all(payload: dict | None = None, inst: str = "sber"):
    ST5 = _st5(inst)
    if not payload or not payload.get("confirm"):
        raise HTTPException(400, "нужно подтверждение: {\"confirm\": true}")
    trade = ST5.engine.flat_all("flat_all")
    ST5.save_session()
    return {"ok": True, "closed": trade is not None,
            "net_pnl_rub": round(trade.net_pnl_rub, 0) if trade else None}


@app.post("/st5/control/trading")
def st5_trading(on: bool = True, inst: str = "sber"):
    ST5 = _st5(inst)
    ST5.cfg.risk.trading_enabled = on
    return {"ok": True, "trading_enabled": on}


@app.post("/st5/control/resume")
def st5_resume(inst: str = "sber"):
    ST5 = _st5(inst)
    ST5.engine.risk.resume()
    from .st5.models import BotState as _BS
    if ST5.engine.state == _BS.HALTED:
        ST5.engine.state = _BS.FLAT
    return {"ok": True, "halted": ST5.engine.risk.halted}


@app.post("/st5/approve")
def st5_approve(inst: str = "sber"):
    ST5 = _st5(inst)
    if ST5.engine._pending is None:
        raise HTTPException(400, "нет ожидающей рекомендации")
    ST5.engine.approve()
    ST5.save_session()
    return {"ok": True}


@app.post("/st5/reject")
def st5_reject(inst: str = "sber"):
    ST5 = _st5(inst)
    ST5.engine.reject()
    return {"ok": True}


@app.post("/st5/auto")
def st5_auto(on: bool = True, inst: str = "sber"):
    ST5 = _st5(inst)
    ST5.cfg.auto_approve = on
    if on and ST5.engine._pending is not None:
        ST5.engine.approve()
        ST5.save_session()
    return {"ok": True, "auto_approve": on}


@app.post("/st5/reset")
def st5_reset(inst: str = "sber"):
    ST5 = _st5(inst)
    ST5.reset_engine(real=(ST5.state["data_source"] == "live"))
    return {"ok": True}


@app.get("/st5/trades")
def st5_trades(inst: str = "sber"):
    ST5 = _st5(inst)
    return {"trades": [ST5._trade_json(t) for t in ST5.engine.trades]}


@app.get("/st5/backtest")
async def st5_backtest(days: int = 30, band_sigma: float | None = None, inst: str = "sber"):
    """Бэктест VWAP-reversion на истории MOEX ISS за период."""
    ST5 = _st5(inst)

    def _run() -> dict:
        try:
            spec = feed5.resolve_leg(ST5.cfg)
        except Exception as ex:  # noqa: BLE001
            return {"error": f"не удалось определить серию: {ex}"}
        since = _dt.now(_tz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        since = since.fromtimestamp(since.timestamp() - days * 86400, tz=_tz.utc)
        try:
            df = feed5.read_ohlcv_moex_range(ST5.cfg, since, spec.code)
        except Exception as ex:  # noqa: BLE001
            return {"error": f"не удалось получить историю: {ex}"}
        if len(df) < 50:
            return {"error": f"мало данных: {len(df)} баров"}
        from .st5.config import St5Config as _Cfg
        bt_cfg = _Cfg(**ST5.cfg.model_dump())
        if band_sigma is not None:
            bt_cfg.strategy.band_sigma = band_sigma
        res = run_backtest5(df, bt_cfg, spec)
        res["bands"] = vwap_frame_for_chart(df, bt_cfg)[-400:]
        res["leg"] = {"code": spec.code, "expiry": spec.expiry}
        from .st5.service import bt_history_append as _app
        entry = {
            "date": _dt.now(_MSK).strftime("%Y-%m-%d %H:%M"),
            "days": days,
            "band_sigma": band_sigma if band_sigma is not None else ST5.cfg.strategy.band_sigma,
            "bars": res["bars"], "trades": res["trades"], "win_rate_pct": res["win_rate_pct"],
            "net_pnl_rub": res["net_pnl_rub"], "return_pct": res["return_pct"],
            "max_drawdown_pct": res["max_drawdown_pct"], "stops": res["stops"],
        }
        res["history"] = _app(entry, inst=ST5.inst)
        return res

    return _clean(await asyncio.to_thread(_run))


@app.get("/st5/backtest_history")
def st5_backtest_history(inst: str = "sber"):
    from .st5.service import bt_history_load as _load
    return {"history": _load(_st5(inst).inst)}


@app.get("/st5/tests")
async def st5_tests():
    """Статус юнит-тестов st5."""
    import re
    import subprocess
    import sys

    def _run() -> dict:
        try:
            p = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/test_st5.py",
                 "--no-header", "-p", "no:cacheprovider"],
                cwd=str(_BASE), capture_output=True, text=True, timeout=180)
            out = p.stdout + p.stderr
            passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", out)) else 0
            failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", out)) else 0
            return {"passed": passed, "failed": failed, "ok": failed == 0 and passed > 0,
                    "tail": out.strip().splitlines()[-1:] if out else []}
        except Exception as e:  # noqa: BLE001
            return {"passed": 0, "failed": 0, "ok": False, "error": str(e)}

    return await asyncio.to_thread(_run)
