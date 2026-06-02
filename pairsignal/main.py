"""CLI-раннер.

Примеры:
  python -m pairsignal.main --synthetic --auto        # быстрый прогон без сети
  python -m pairsignal.main --synthetic               # с ручным подтверждением входов
  python -m pairsignal.main --live                    # реальные котировки (нужен ccxt + сеть)
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

from .config import AppConfig
from .data_feed import generate_synthetic, read_ohlcv_ccxt
from .engine import Engine
from .models import Action


def _fmt_ts(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _log(rec, extra: str = "") -> None:
    print(
        f"{_fmt_ts(rec.ts)} | z={rec.z:+.2f} | spread={rec.spread:.4f} "
        f"| BB[{rec.lower:.3f}/{rec.mid:.3f}/{rec.upper:.3f}] | width={rec.width_pct:.2f}% "
        f"| {rec.action.value.upper():5} | {rec.reason}{extra}"
    )


def run_backtest(cfg: AppConfig, df, verbose: bool = True) -> Engine:
    """Прогон по историческому/синтетическому DataFrame как поток баров."""
    eng = Engine(cfg)
    rows = Engine.rows_from_df(df, cfg.strategy)
    for row in rows:
        res = eng.step(row)
        if res.awaiting_approval:
            # human-in-the-loop: спрашиваем оператора
            _log(res.rec, extra="  ← ТРЕБУЕТСЯ РЕШЕНИЕ")
            ans = input("    Входить в виртуальную сделку? [y/N]: ").strip().lower()
            if ans == "y":
                eng.approve()
                print("    → вход подтверждён")
            else:
                eng.reject()
                print("    → пропуск")
        elif verbose and res.rec.action != Action.NONE:
            extra = ""
            if res.trade:
                extra = f"  net={res.trade.net_pnl:+.2f} fees={res.trade.fees:.2f}"
            _log(res.rec, extra=extra)
    return eng


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="офлайн-демо на сгенерированной паре")
    ap.add_argument("--live", action="store_true", help="реальные котировки через CCXT")
    ap.add_argument("--auto", action="store_true", help="авто-подтверждение входов (без оператора)")
    ap.add_argument("--limit", type=int, default=3000, help="число баров")
    args = ap.parse_args()

    cfg = AppConfig()
    cfg.auto_approve = args.auto

    if args.live:
        df = read_ohlcv_ccxt(cfg.strategy, limit=args.limit)
    else:
        df = generate_synthetic(n=args.limit)

    eng = run_backtest(cfg, df, verbose=True)

    print("\n=== СВОДКА ===")
    for k, v in eng.summary().items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
