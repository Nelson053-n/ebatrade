"""Бэктест-движок FORTS-фьючерсов с реальными издержками + walk-forward.

Контракт совпадает с боевым st5: P&L одной ноги = (exit-entry)*dir*lots*(STEPPRICE/MINSTEP).
Для всех серий STEPPRICE/MINSTEP = 1.0 → 1 пункт цены = 1₽/лот.

Издержки (round-trip на лот):
  fee     — комиссия за лот за сделку (вход+выход) = 2*fee_per_lot
  slippage — проскальзывание в тиках на каждую сторону (вход+выход) = 2*slip_ticks*tick

Все сигналы — по ЗАКРЫТЫМ барам. Вход исполняется по close бара сигнала + проскальзывание
(консервативно: реально вошли бы по открытию следующего бара; разница — часть проскальзывания).
Выход — по close бара выхода + проскальзывание против нас.

Семья стратегий задаётся callable signal_fn(df_slice)->pd.Series(+1/-1/0 desired position),
а движок применяет позиционную модель с холдингом/стопом/TP внутри конкретной стратегии —
поэтому стратегии возвращают уже готовый ряд целевой позиции по барам (state machine внутри).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TICK = 1.0
TICK_VALUE = 1.0  # STEPPRICE/MINSTEP для SR*/GZ*/LK*


@dataclass
class Costs:
    fee_per_lot: float = 1.0      # ₽ за лот за одну сделку (вход ИЛИ выход)
    slip_ticks: float = 1.0       # тиков проскальзывания на одну сторону


@dataclass
class TradeRec:
    entry_i: int
    exit_i: int
    side: int           # +1 long, -1 short
    entry_px: float
    exit_px: float
    gross: float        # ₽/лот
    cost: float         # ₽/лот
    net: float          # ₽/лот
    bars: int
    reason: str


def run_position_series(close: np.ndarray, pos: np.ndarray, costs: Costs,
                        seam: np.ndarray | None = None,
                        reasons: np.ndarray | None = None) -> list[TradeRec]:
    """Прогон по ряду целевой позиции pos[i] in {-1,0,+1} (на закрытии бара i).

    Вход/выход исполняется по close[i] с проскальзыванием. На стыке контрактов (seam
    меняется) форсируем выход по close предыдущего бара (без переноса позиции через
    разрыв базиса). reasons[i] — метка причины выхода для статистики (опционально).
    """
    n = len(close)
    trades: list[TradeRec] = []
    cur = 0          # текущая позиция
    entry_px = 0.0
    entry_i = 0
    slip = costs.slip_ticks * TICK
    fee = costs.fee_per_lot

    def _close(i: int, px: float, reason: str):
        nonlocal cur
        # выход: против нас проскальзывание
        exit_px = px - slip if cur > 0 else px + slip
        gross = (exit_px - entry_px) * cur * TICK_VALUE
        cost = fee  # выходная комиссия (входная уже учтена при открытии)
        net = gross - cost
        trades.append(TradeRec(entry_i, i, cur, entry_px, exit_px,
                               round(gross, 2), 0.0, 0.0, i - entry_i, reason))
        cur = 0

    for i in range(n):
        # принудительный выход на стыке контракта
        if seam is not None and cur != 0 and i > 0 and seam[i] != seam[i - 1]:
            _close(i - 1, close[i - 1], "seam")
        target = pos[i]
        if cur != 0 and target != cur:
            r = reasons[i] if reasons is not None else "signal"
            _close(i, close[i], r)
        if cur == 0 and target != 0:
            # вход: против нас проскальзывание
            entry_px = close[i] + slip if target > 0 else close[i] - slip
            entry_i = i
            cur = target
    if cur != 0:
        _close(n - 1, close[-1], "end")

    # дораспределяем издержки: каждая сделка = входная + выходная комиссия
    out: list[TradeRec] = []
    for t in trades:
        cost = 2 * fee  # вход + выход
        net = t.gross - cost
        out.append(TradeRec(t.entry_i, t.exit_i, t.side, t.entry_px, t.exit_px,
                            t.gross, round(cost, 2), round(net, 2), t.bars, t.reason))
    return out


def metrics(trades: list[TradeRec], close: np.ndarray, bars_per_day: float = 84.0) -> dict:
    """Сводка по сделкам: net, sharpe (по дневной агрегации), winrate, DD, BuyHold."""
    if not trades:
        return {"trades": 0, "net": 0.0, "sharpe": 0.0, "winrate": 0.0,
                "max_dd": 0.0, "avg_bars": 0.0, "gross": 0.0, "cost": 0.0}
    net = sum(t.net for t in trades)
    gross = sum(t.gross for t in trades)
    cost = sum(t.cost for t in trades)
    wins = sum(1 for t in trades if t.net > 0)

    # equity по индексам выхода → дневной P&L для Sharpe
    n = len(close)
    pnl_by_bar = np.zeros(n)
    for t in trades:
        pnl_by_bar[t.exit_i] += t.net
    day = (np.arange(n) // bars_per_day).astype(int)
    daily = pd.Series(pnl_by_bar).groupby(day).sum()
    mu, sd = daily.mean(), daily.std(ddof=1)
    sharpe = (mu / sd * np.sqrt(252)) if sd > 0 else 0.0

    # max drawdown по кумулятивной equity (на закрытиях сделок)
    eq = np.cumsum([t.net for t in trades])
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq)
    max_dd = float(dd.max()) if len(dd) else 0.0

    return {
        "trades": len(trades),
        "net": round(net, 0),
        "gross": round(gross, 0),
        "cost": round(cost, 0),
        "sharpe": round(float(sharpe), 2),
        "winrate": round(100 * wins / len(trades), 1),
        "max_dd": round(max_dd, 0),
        "avg_bars": round(np.mean([t.bars for t in trades]), 1),
    }


def buy_hold(close: np.ndarray, seam: np.ndarray, costs: Costs) -> float:
    """P&L пассивного лонга 1 лот (роллируемого на стыках), ₽/лот, с издержками ролла."""
    pnl = 0.0
    fee = costs.fee_per_lot
    slip = costs.slip_ticks * TICK
    n_rolls = 0
    for i in range(1, len(close)):
        if seam[i] == seam[i - 1]:
            pnl += close[i] - close[i - 1]
        else:
            n_rolls += 1
    pnl -= n_rolls * (2 * fee + 2 * slip)  # каждый ролл = выход+вход
    pnl -= (2 * fee + 2 * slip)            # начальный вход + финальный выход
    return round(pnl, 0)
