"""Конфигурация st6 (часовой парный mean-reversion обычка/преф) — pydantic v2.

Часовой парный mean-reversion для ДВУХ фиксированных пар обычка/преф одного
эмитента: TATN/TATNP и SBER/SBERP — единственный доказанный OOS-эдж на MOEX
(research/moex/pairs_h1_RESULTS.txt). Динамический отбор пары из корзины (rank_pairs)
ОТКЛЮЧЁН: честный walk-forward на корзине давал t≤1.15 (data snooping). Пара
ФИКСИРУЕТСЯ.

⚠ ЧЕСТНОСТЬ: эдж подтверждён ТОЛЬКО под maker/limit-исполнением (~0.01% fee + 0.01%
slip на ногу, ~6.4 bps round-trip). Под taker (0.05%+0.03%) эдж съедается (t~1.0).
Часовой ТФ (interval=60), НЕ дневной. Это форвард-тест гипотезы, не гарантия прибыли.

Данные — часовые свечи MOEX ISS (рынок акций), paper-only (Phase 1). Params ядра
(st6.core.Params) продублирован здесь как StrategyConfig для сериализации в
session_state_6.json; переносится в ядро через to_params().
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .core import Params

# Корзина оставлена для совместимости (старые session-файлы, snapshot.basket), но
# для ОТБОРА не используется — пара фиксирована. Содержит обе ноги фикс-пар.
DEFAULT_BASKET: list[str] = ["TATN", "TATNP", "SBER", "SBERP"]

# Фиксированные пары обычка/преф с пара-специфичным z_exit (доказано research):
# TATN/TATNP лучший на zx=1.0 (maker t=4.37, Sh=1.85); SBER/SBERP на zx=0.5 (t=2.35).
# Активная пара по умолчанию — TATN/TATNP (сильнейший эдж). Вторая доступна сменой
# fixed_pair через /st6/config. Отбора (rank_pairs) больше нет.
FIXED_PAIRS: dict[str, dict] = {
    "TATN/TATNP": {"a": "TATN", "b": "TATNP", "z_exit": 1.0},
    "SBER/SBERP": {"a": "SBER", "b": "SBERP", "z_exit": 0.5},
}
DEFAULT_PAIR = "TATN/TATNP"


class StrategyConfig(BaseModel):
    """Параметры ядра st6 (зеркало core.Params для сериализации/редактирования).

    Дефолты = конфиг research для часового парного MR обычка/преф под MAKER:
    окна β/z=480ч, corr=240ч, z_entry=2.0, z_exit=1.0 (TATN; SBER переопределяет на
    0.5 через FIXED_PAIRS), издержки maker 0.0001/0.0001.
    """
    # окна (часовые бары)
    beta_window: int = 480            # бар для оценки β (OLS) и средней спреда
    z_window: int = 480               # окно z-score
    corr_window: int = 240            # окно скользящей корреляции лог-доходностей
    # пороги входа/выхода по z
    z_entry: float = 2.0
    z_exit: float = 1.0               # пара-специфичен (FIXED_PAIRS), это дефолт TATN
    z_stop: float = 3.5
    # корреляционный гейт (мягкий — на фикс-парах corr ~0.90, не мешает; страховка от слома)
    corr_enter: float = 0.58          # вход разрешён только если |corr| >= этого
    corr_break: float = 0.45          # если |corr| падает ниже — аварийный выход
    # отбор пары (legacy-поля, НЕ используются — пара фиксирована; оставлены для совместимости)
    select_min_corr: float = 0.60
    select_max_pvalue: float = 0.20   # ADF p-value для коинтеграции спреда
    select_max_halflife: float = 720.0
    # риск/сайзинг
    risk_fraction: float = 0.02       # доля equity на нотионал каждой ноги
    time_stop_bars: int = 0           # 0 = выкл; иначе принудительный выход по барам
    # издержки maker/limit (на оборот, доля от нотионала ноги). ⚠ эдж жив только под этими;
    # taker (0.0005/0.0003) убивает эдж — см. модуль-docstring.
    fee_rate: float = 0.0001
    slippage_rate: float = 0.0001

    def to_params(self, z_exit: float | None = None) -> Params:
        """Перенести конфиг в датакласс ядра. z_exit можно переопределить пара-специфично."""
        return Params(
            beta_window=self.beta_window,
            z_window=self.z_window,
            corr_window=self.corr_window,
            z_entry=self.z_entry,
            z_exit=self.z_exit if z_exit is None else z_exit,
            z_stop=self.z_stop,
            corr_enter=self.corr_enter,
            corr_break=self.corr_break,
            select_min_corr=self.select_min_corr,
            select_max_pvalue=self.select_max_pvalue,
            select_max_halflife=self.select_max_halflife,
            risk_fraction=self.risk_fraction,
            time_stop_bars=self.time_stop_bars,
            fee_rate=self.fee_rate,
            slippage_rate=self.slippage_rate,
        )


class ConnectorConfig(BaseModel):
    """Выбор исполнителя ордеров st6: paper (дефолт) или T-Bank sandbox.

    Боевой контур запрещён — только SandboxService. Токена здесь НЕТ: секрет не
    сериализуется в session_state_6.json, живёт только в окружении процесса (env
    TBANK_TOKEN, общий с st4 через st4/tbank_sandbox.save_token). account_id/payin_rub
    несекретны и переживают рестарт. Sandbox активен только в live (на синтетике — paper).
    """
    mode: Literal["paper", "tbank_sandbox"] = "paper"
    account_id: str = ""              # переиспользуемый sandbox-счёт; пусто → открыть новый
    payin_rub: int = 500_000          # пополнение sandbox-счёта при старте (акции — рубли)
    account_name: str = "st6-pairs-sandbox"


class St6Config(BaseModel):
    """Полный конфиг сессии st6.

    Все НОВЫЕ поля строго С ДЕФОЛТАМИ: St6Config(**cfg) грузит старые
    session_state_6_*.json целиком (service.load_session ~стр 461) — без дефолта
    загрузка упадёт.
    """
    basket: list[str] = Field(default_factory=lambda: list(DEFAULT_BASKET))
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    # активная ФИКСИРОВАННАЯ пара (ключ FIXED_PAIRS). Отбора из корзины больше нет.
    fixed_pair: str = DEFAULT_PAIR
    # таймфрейм свечей. '1h' — часовые (доказанный эдж); '1d' — дневные (legacy).
    timeframe: str = "1h"
    # часовых баров нужно: warmup max(480,480,240)+5 ≈ 485 + торговля. ~15 баров/торг.день
    # → 365 дней ≈ 5500 часовых баров (с запасом на выходные/прогрев).
    history_days: int = 365           # сколько дней истории тянуть с MOEX ISS
    reselect_every_bars: int = 20     # legacy (отбора нет) — оставлено для совместимости
    start_equity_rub: float = 1_000_000.0
    poll_seconds: int = 600           # период опроса ISS в live (часовой ТФ — раз в ~час)
    # коннектор: paper (дефолт) или tbank_sandbox. Структурный (ранее был строкой "paper" —
    # legacy-строка из старых session-файлов конвертируется в service.load_session).
    connector: ConnectorConfig = Field(default_factory=ConnectorConfig)
    auto_approve: bool = True         # вход без ручного подтверждения (как st5)

    def pair_tickers(self) -> tuple[str, str]:
        """Тикеры активной фикс-пары (a, b). Падает обратно на DEFAULT_PAIR."""
        spec = FIXED_PAIRS.get(self.fixed_pair) or FIXED_PAIRS[DEFAULT_PAIR]
        return spec["a"], spec["b"]

    def pair_z_exit(self) -> float:
        """Пара-специфичный z_exit активной фикс-пары (TATN=1.0, SBER=0.5)."""
        spec = FIXED_PAIRS.get(self.fixed_pair) or FIXED_PAIRS[DEFAULT_PAIR]
        return float(spec["z_exit"])
