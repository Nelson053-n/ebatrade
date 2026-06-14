"""Бэктест st5: офлайн-прогон VWAP-reversion на исторических OHLCV-свечах.

Честный отчёт: число сделок, win-rate, P&L, max-DD по equity (с нереализованным),
средние держание/проскальзывание, кривая капитала.
"""
from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from .config import St5Config
from .engine import TradingEngine
from .indicators import IntradayVwap
from .models import InstrumentSpec, PriceBar


def run_backtest(df: pd.DataFrame, cfg: St5Config, spec: InstrumentSpec) -> dict:
    """Прогон по df (open/high/low/close/volume), метрики + equity-кривая."""
    cfg = St5Config(**cfg.model_dump())
    eng = TradingEngine(cfg, spec)

    start = cfg.paper.start_balance_rub
    equity_curve: list[dict] = []
    peak = start
    max_dd = 0.0
    for ts, row in df.iterrows():
        bar = PriceBar(int(ts), float(row["open"]), float(row["high"]), float(row["low"]),
                       float(row["close"]), float(row.get("volume", 0.0)))
        eng.step(bar)
        eq = eng.balance_rub + eng.unrealized_rub()
        peak = max(peak, eq)
        dd = (peak - eq) / peak * 100 if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
        equity_curve.append({"ts": int(ts), "equity": round(eq, 0)})

    trades = eng.trades
    wins = [t for t in trades if t.net_pnl_rub > 0]
    net = sum(t.net_pnl_rub for t in trades)
    return {
        "bars": len(df),
        "trades": len(trades),
        "wins": len(wins),
        "win_rate_pct": round(100 * len(wins) / len(trades), 1) if trades else 0.0,
        "net_pnl_rub": round(net, 0),
        "gross_pnl_rub": round(sum(t.gross_pnl_rub for t in trades), 0),
        "fees_rub": round(sum(t.fees_rub for t in trades), 0),
        "avg_pnl_rub": round(net / len(trades), 0) if trades else 0.0,
        "avg_slippage_ticks": round(sum(t.slippage_ticks for t in trades) / len(trades), 2) if trades else 0.0,
        "max_drawdown_pct": round(max_dd, 2),
        "return_pct": round(100 * net / start, 3),
        "stops": sum(1 for t in trades if t.reason in ("stop", "time_stop")),
        "eod_exits": sum(1 for t in trades if t.reason == "eod"),
        "avg_bars_held": round(sum(t.bars_held for t in trades) / len(trades), 1) if trades else 0,
        "max_bars_held": max((t.bars_held for t in trades), default=0),
        "open_position": eng.state.value if eng.position else None,
        "open_unrealized_rub": round(eng.unrealized_rub(), 0),
        "equity_curve": equity_curve,
        "trades_detail": [_trade_dict(t) for t in trades],
    }


def _trade_dict(t) -> dict:
    d = asdict(t)
    d["state"] = t.state.value
    return d


def vwap_frame_for_chart(df: pd.DataFrame, cfg: St5Config) -> list[dict]:
    """Цена + VWAP + коридор по всему df для графика вкладки (прогрев → отброшен)."""
    vw = IntradayVwap(cfg.strategy.band_sigma, cfg.strategy.min_bars_in_day,
                      cfg.strategy.std_mode)
    out = []
    for ts, row in df.iterrows():
        bar = PriceBar(int(ts), float(row["open"]), float(row["high"]), float(row["low"]),
                       float(row["close"]), float(row.get("volume", 0.0)))
        r = vw.update(bar)
        if not r.is_ready:
            continue
        out.append({"ts": int(ts), "price": round(r.price, 1), "vwap": round(r.vwap, 1),
                    "upper": round(r.upper, 1), "lower": round(r.lower, 1),
                    "sigma": round(r.sigma, 1)})
    return out
