"""st6 · бэктест парной стратегии по историческим закрытиям (walk-forward,
no look-ahead). Работает поверх st6.core — без сети.

Перенос st6_backtest.py в пакет: backtest_pair, метрики, синтетика. Плюс
run_backtest (обёртка под dict-отчёт в стиле st5) и spread_frame для графика.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .core import (
    Params, Position, Side,
    decide, hedge_ratio, leg_quantities, spread_series, trade_pnl, zscore,
)


@dataclass
class Trade:
    side: str
    entry_i: int
    exit_i: int
    entry_a: float
    exit_a: float
    entry_b: float
    exit_b: float
    bars: int
    net: float
    reason: str


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    start_equity: float = 0.0

    @property
    def end_equity(self) -> float:
        return self.equity_curve[-1] if self.equity_curve else self.start_equity

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return float("nan")
        wins = sum(1 for t in self.trades if t.net > 0)
        return wins / len(self.trades)

    @property
    def total_net(self) -> float:
        return sum(t.net for t in self.trades)

    @property
    def return_pct(self) -> float:
        if self.start_equity == 0:
            return float("nan")
        return 100.0 * (self.end_equity - self.start_equity) / self.start_equity

    @property
    def max_drawdown_pct(self) -> float:
        if not self.equity_curve:
            return float("nan")
        eq = np.asarray(self.equity_curve, dtype=float)
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak
        return float(-dd.min() * 100.0)

    @property
    def sharpe(self) -> float:
        """Sharpe по доходностям сделок (грубо, не аннуализирован)."""
        nets = np.array([t.net for t in self.trades], dtype=float)
        if len(nets) < 2 or nets.std() == 0:
            return float("nan")
        return float(nets.mean() / nets.std() * np.sqrt(len(nets)))


def backtest_pair(prices_a: Sequence[float], prices_b: Sequence[float],
                  p: Params, start_equity: float = 100_000.0,
                  lot_a: int = 1, lot_b: int = 1) -> BacktestResult:
    """
    Бар-за-баром прогон одной пары. Сигнал на close[i] исполняется по close[i]
    (консервативно: следующий шаг ничего не подсматривает вперёд).
    """
    a = np.asarray(prices_a, dtype=float)
    b = np.asarray(prices_b, dtype=float)
    n = min(len(a), len(b))
    a, b = a[-n:], b[-n:]

    res = BacktestResult(start_equity=start_equity)
    equity = start_equity
    pos = Position()
    warmup = max(p.beta_window, p.z_window, p.corr_window) + 1

    for i in range(warmup, n):
        wa = a[:i + 1]
        wb = b[:i + 1]
        sig = decide(wa, wb, pos, p)
        ca, cb = a[i], b[i]

        if pos.is_open:
            pos.bars_held += 1
            if sig.action == "EXIT":
                units_a = pos.qty_a * lot_a
                units_b = pos.qty_b * lot_b
                net = trade_pnl(pos.side, pos.entry_a, ca, pos.entry_b, cb,
                                units_a, units_b, p)
                equity += net
                res.trades.append(Trade(
                    side=pos.side.name, entry_i=i - pos.bars_held, exit_i=i,
                    entry_a=pos.entry_a, exit_a=ca, entry_b=pos.entry_b, exit_b=cb,
                    bars=pos.bars_held, net=net, reason=sig.reason.value,
                ))
                pos = Position()
        else:
            if sig.action in ("ENTER_LONG", "ENTER_SHORT"):
                qa, qb = leg_quantities(equity, ca, cb, sig.beta, lot_a, lot_b, p)
                if qa > 0 and qb > 0:
                    pos = Position(
                        side=Side.LONG_SPREAD if sig.action == "ENTER_LONG"
                        else Side.SHORT_SPREAD,
                        entry_z=sig.z, beta=sig.beta, bars_held=0,
                        qty_a=qa, qty_b=qb, entry_a=ca, entry_b=cb,
                    )
        res.equity_curve.append(equity)

    return res


def run_backtest(prices_a: Sequence[float], prices_b: Sequence[float], p: Params,
                 ticker_a: str = "A", ticker_b: str = "B",
                 start_equity: float = 1_000_000.0) -> dict:
    """Обёртка под компактный dict-отчёт (стиль st5.run_backtest) + кривая капитала."""
    r = backtest_pair(prices_a, prices_b, p, start_equity=start_equity)
    eq_curve = [{"i": i, "equity": round(e, 0)} for i, e in enumerate(r.equity_curve)]
    return {
        "pair": f"{ticker_a}/{ticker_b}",
        "bars": min(len(prices_a), len(prices_b)),
        "trades": r.n_trades,
        "wins": sum(1 for t in r.trades if t.net > 0),
        "win_rate_pct": round(r.win_rate * 100, 1) if r.trades else 0.0,
        "net_pnl_rub": round(r.total_net, 0),
        "return_pct": round(r.return_pct, 3),
        "max_drawdown_pct": round(r.max_drawdown_pct, 2) if r.trades else 0.0,
        "sharpe": round(r.sharpe, 2) if r.n_trades >= 2 else None,
        "avg_bars_held": round(sum(t.bars for t in r.trades) / r.n_trades, 1) if r.trades else 0,
        "corr_breaks": sum(1 for t in r.trades if t.reason == "corr_break"),
        "stops": sum(1 for t in r.trades if t.reason == "stop_z"),
        "equity_curve": eq_curve,
        "trades_detail": [
            {"side": t.side, "bars": t.bars, "net": round(t.net, 0), "reason": t.reason}
            for t in r.trades
        ],
    }


def spread_frame(prices_a: Sequence[float], prices_b: Sequence[float],
                 p: Params) -> list[dict]:
    """Ряд спреда и z по rolling-β для графика вкладки (прогрев → отброшен)."""
    a = np.asarray(prices_a, dtype=float)
    b = np.asarray(prices_b, dtype=float)
    n = min(len(a), len(b))
    a, b = a[-n:], b[-n:]
    warmup = max(p.beta_window, p.z_window) + 1
    out: list[dict] = []
    for i in range(warmup, n):
        beta, _ = hedge_ratio(np.log(a[i - p.beta_window:i + 1]),
                              np.log(b[i - p.beta_window:i + 1]))
        spr = spread_series(a[:i + 1], b[:i + 1], beta)
        z = zscore(spr, p.z_window)
        if not np.isfinite(z):
            continue
        out.append({"i": i, "spread": round(float(spr[-1]), 5), "z": round(z, 3),
                    "beta": round(beta, 4)})
    return out


# --------------------------------------------------------------------------
# Синтетический генератор коинтегрированной пары (для демо/тестов)
# --------------------------------------------------------------------------
def make_synthetic_pair(n: int = 2000, beta: float = 1.0, seed: int = 7,
                        spread_kappa: float = 0.05, spread_sigma: float = 0.01,
                        drift_sigma: float = 0.012):
    """
    B блуждает случайно; A = beta*B + стационарный OU-спред. Получаем пару
    с высокой корреляцией и возвратом спреда к среднему — идеально для
    проверки, что логика реально ловит и закрывает сделки.
    """
    rng = np.random.default_rng(seed)
    log_b = np.cumsum(rng.normal(0, drift_sigma, n)) + np.log(100.0)
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = spread[t - 1] * (1 - spread_kappa) + rng.normal(0, spread_sigma)
    log_a = beta * log_b + spread + np.log(1.5)
    return np.exp(log_a), np.exp(log_b)
