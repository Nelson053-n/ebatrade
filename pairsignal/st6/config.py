"""Конфигурация st6 (Correlation-Gated Pairs) — pydantic v2, без хардкода.

Парный mean-reversion с корреляционным гейтом и динамическим отбором пары из
корзины акций MOEX. Данные — дневные свечи MOEX ISS (рынок акций), paper-only
(Phase 1, реальные ордера не отправляются). Params ядра (st6.core.Params)
продублирован здесь как pydantic-модель StrategyConfig, чтобы конфиг сериализовался
в session_state_6.json и редактировался из API; в ядро он переносится через
to_params().
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from .core import Params

# Ликвидные коррелированные акции MOEX (нефтегаз / металлурги / SBER).
# Сканер сам выберет лучшую пару — корзину можно расширять/сужать.
DEFAULT_BASKET: list[str] = [
    # нефть и газ
    "LKOH", "ROSN", "SIBN", "TATN", "GAZP", "NVTK",
    # металлурги
    "GMKN", "NLMK", "MAGN", "CHMF",
    # банки / голубые фишки
    "SBER",
]


class StrategyConfig(BaseModel):
    """Параметры ядра st6 (зеркало core.Params для сериализации/редактирования)."""
    # окна
    beta_window: int = 240            # бар для оценки β (OLS) и средней спреда
    z_window: int = 240               # окно z-score
    corr_window: int = 120            # окно скользящей корреляции лог-доходностей
    # пороги входа/выхода по z
    z_entry: float = 2.0
    z_exit: float = 0.3
    z_stop: float = 3.5
    # корреляционный гейт. Калибровано под реальные дневные данные акций MOEX: на горизонте
    # ~2 года |corr| родственных бумаг ~0.55-0.65 (не 0.8 как на синтетике/внутри дня) —
    # иначе годной пары нет вовсе (бэктест 600 баров: при 0.6 находится SIBN/NVTK corr0.62 p0.06).
    corr_enter: float = 0.58          # вход разрешён только если |corr| >= этого
    corr_break: float = 0.45          # если |corr| падает ниже — аварийный выход
    # отбор пары (скан корзины)
    select_min_corr: float = 0.60
    select_max_pvalue: float = 0.20   # ADF p-value для коинтеграции спреда
    select_max_halflife: float = 400.0
    # риск/сайзинг
    risk_fraction: float = 0.02       # доля equity на нотионал каждой ноги
    time_stop_bars: int = 0           # 0 = выкл; иначе принудительный выход по барам
    # издержки (на оборот, доля от нотионала ноги)
    fee_rate: float = 0.0006
    slippage_rate: float = 0.0005

    def to_params(self) -> Params:
        """Перенести конфиг в датакласс ядра."""
        return Params(
            beta_window=self.beta_window,
            z_window=self.z_window,
            corr_window=self.corr_window,
            z_entry=self.z_entry,
            z_exit=self.z_exit,
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


class St6Config(BaseModel):
    """Полный конфиг сессии st6."""
    basket: list[str] = Field(default_factory=lambda: list(DEFAULT_BASKET))
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    # таймфрейм свечей. '1d' — дневные (глубокая история, дефолт), '1h' — часовые.
    timeframe: str = "1d"
    history_days: int = 720           # сколько дней истории тянуть с MOEX ISS
    reselect_every_bars: int = 20     # как часто пере-выбирать лучшую пару (вне позиции)
    start_equity_rub: float = 1_000_000.0
    poll_seconds: int = 1800          # период опроса ISS в live (дневной ТФ — редко)
    connector: str = "paper"          # paper-only (Phase 1); T-Bank НЕ подключаем
    auto_approve: bool = True         # вход без ручного подтверждения (как st5)
