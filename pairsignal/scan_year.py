"""Годовой скан st2 (cross_pct) по топ-монетам, доступным на gateio И mexc.

Берёт топ-N перпетуалов по 24h-обороту на пересечении бирж, для каждого прогоняет
кросс-биржевую стратегию cross_pct за год и печатает сводную таблицу (валюта, средний
спред, винрейт, число сделок, P&L при старте $1000), плюс сохраняет CSV.

Прогон — через существующие run_backtest + Engine.summary (движок не меняется).

  python -m pairsignal.scan_year                                   # топ-20, год
  python -m pairsignal.scan_year --symbols "BTC/USDT:USDT" --days 14
"""
from __future__ import annotations

import argparse
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from .config import AppConfig, PaperConfig, StrategyConfig
from .data_feed import read_ohlcv_cross_range
from .main import run_backtest

EXCHANGE_A = "gateio"
EXCHANGE_B = "mexc"
START_BALANCE = 1000.0


def top_symbols(n: int = 20, quote: str = "USDT") -> list[str]:
    """Топ-N перпетуалов по обороту на пересечении EXCHANGE_A и EXCHANGE_B.

    Ликвидность пары = min(оборот_A, оборот_B): пара торгуема настолько, насколько
    торгуема её слабейшая нога.
    """
    import ccxt

    def _markets(name: str) -> set[str]:
        ex = getattr(ccxt, name)({"enableRateLimit": True})
        ex.options["defaultType"] = "swap"
        ex.load_markets()
        return {
            s for s, m in ex.markets.items()
            if m.get("swap") and m.get("quote") == quote and m.get("active", True)
        }

    def _volumes(name: str) -> dict[str, float]:
        ex = getattr(ccxt, name)({"enableRateLimit": True})
        ex.options["defaultType"] = "swap"
        out: dict[str, float] = {}
        for s, t in ex.fetch_tickers().items():
            qv = t.get("quoteVolume")
            if qv is None:
                bv, last = t.get("baseVolume"), t.get("last")
                qv = (bv or 0.0) * (last or 0.0)
            out[s] = float(qv or 0.0)
        return out

    common = _markets(EXCHANGE_A) & _markets(EXCHANGE_B)
    vol_a, vol_b = _volumes(EXCHANGE_A), _volumes(EXCHANGE_B)
    ranked = sorted(
        common,
        key=lambda s: min(vol_a.get(s, 0.0), vol_b.get(s, 0.0)),
        reverse=True,
    )
    return ranked[:n]


def top_symbols_by_spread(n: int = 20, quote: str = "USDT", min_vol: float = 1_000_000.0) -> list[str]:
    """Топ-N перпетуалов по МГНОВЕННОМУ кросс-биржевому спреду |last_a−last_b|/mid.

    Спред берётся снимком из fetch_tickers (2 запроса). min_vol отсеивает неликвид/мёртвые
    рынки, иначе наверх вылезают пустышки с аномальным расхождением цен и нулём истории.
    """
    import ccxt

    def _markets(name: str) -> set[str]:
        ex = getattr(ccxt, name)({"enableRateLimit": True})
        ex.options["defaultType"] = "swap"
        ex.load_markets()
        return {
            s for s, m in ex.markets.items()
            if m.get("swap") and m.get("quote") == quote and m.get("active", True)
        }

    def _tickers(name: str) -> dict[str, dict]:
        ex = getattr(ccxt, name)({"enableRateLimit": True})
        ex.options["defaultType"] = "swap"
        return ex.fetch_tickers()

    def _vol(t: dict) -> float:
        qv = t.get("quoteVolume")
        if qv is None:
            qv = (t.get("baseVolume") or 0.0) * (t.get("last") or 0.0)
        return float(qv or 0.0)

    common = _markets(EXCHANGE_A) & _markets(EXCHANGE_B)
    ta, tb = _tickers(EXCHANGE_A), _tickers(EXCHANGE_B)

    scored: list[tuple[str, float]] = []
    for s in common:
        ta_s, tb_s = ta.get(s), tb.get(s)
        if not ta_s or not tb_s:
            continue
        la, lb = ta_s.get("last"), tb_s.get("last")
        if not la or not lb:
            continue
        if min(_vol(ta_s), _vol(tb_s)) < min_vol:  # отсев неликвида
            continue
        mid = (la + lb) / 2.0
        spread_pct = abs(la - lb) / mid * 100.0
        scored.append((s, spread_pct))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in scored[:n]]


def make_cfg(symbol: str, timeframe: str, slippage_pct: float = 0.0002,
             fee: float = 0.0006) -> AppConfig:
    """Конфиг cross_pct под одну монету (консервативный st2-пресет, баланс $1000)."""
    s = StrategyConfig(
        symbol_cross=symbol, exchange_a=EXCHANGE_A, exchange_b=EXCHANGE_B,
        spread_mode="cross_pct", band_mode="vol", sma_period=200, bb_k=2.0,
        entry_z=1.0, stop_z=8.0, min_width_pct=0.0, timeframe=timeframe,
        symbol_a=symbol, symbol_b=symbol,  # для cross_pct — лишь ярлыки ног
    )
    p = PaperConfig(
        start_balance=START_BALANCE, risk_pct=2.0, taker_fee=fee, slippage_pct=slippage_pct,
    )
    return AppConfig(strategy=s, paper=p, auto_approve=True)


def _err_row(symbol: str, status: str) -> dict:
    return {"symbol": symbol, "status": status,
            "avg_spread_pct": 0.0, "win_rate_pct": 0.0, "trades": 0,
            "net_pnl_$": 0.0, "final_equity_$": START_BALANCE,
            "return_pct": 0.0, "bars": 0}


def _book_halfspread(ex, symbol: str, notional_usd: float) -> float | None:
    """Эффективный half-spread (доля) при исполнении notional_usd рыночным ордером СЕЙЧАС.

    Идёт по уровням стакана (walk the book), считает VWAP-цену набора notional на обе
    стороны (ask для покупки, bid для продажи) и возвращает отклонение от mid:
        max(vwap_ask/mid − 1, mid/vwap_bid − 1).
    Это реальная цена «дёрнуть рынок» на наш размер — прокси slippage на ногу.
    None, если стакан пуст или глубины не хватает на notional.
    """
    ob = ex.fetch_order_book(symbol, limit=50)
    bids, asks = ob.get("bids") or [], ob.get("asks") or []
    if not bids or not asks:
        return None
    mid = (bids[0][0] + asks[0][0]) / 2.0
    if mid <= 0:
        return None

    def _vwap(levels: list) -> float | None:
        need, cost, got = notional_usd, 0.0, 0.0
        for lvl in levels:
            price, qty = lvl[0], lvl[1]  # уровень может быть [price, qty, count]
            lvl_usd = price * qty
            take = min(need, lvl_usd)
            cost += take
            got += take / price
            need -= take
            if need <= 0:
                break
        if need > 0:  # глубины не хватило на notional
            return None
        return cost / got  # средняя цена исполнения

    vwap_ask, vwap_bid = _vwap(asks), _vwap(bids)
    if vwap_ask is None or vwap_bid is None:
        return None
    return max(vwap_ask / mid - 1.0, mid / vwap_bid - 1.0)


_OB_CLIENTS: dict[str, object] = {}
_OB_LOCK = Lock()


def _ob_client(name: str):
    """Кэш ccxt-клиента для стакана: один load_markets на биржу, общий таймаут 30с.

    fetch_order_book требует загруженные markets — иначе CCXT неверно резолвит символ
    (лезет в options вместо futures). Клиент потокобезопасно создаётся один раз.
    """
    import ccxt

    with _OB_LOCK:
        ex = _OB_CLIENTS.get(name)
        if ex is None:
            ex = getattr(ccxt, name)({"enableRateLimit": True, "timeout": 30000})
            ex.options["defaultType"] = "swap"
            ex.load_markets()
            _OB_CLIENTS[name] = ex
        return ex


def measure_slippage(symbol: str, notional_leg_usd: float) -> tuple[float | None, str]:
    """Замеряет slippage по живому стакану обеих бирж под размер ноги notional_leg_usd.

    Берёт max half-spread из gateio и mexc (исполнение лимитирует худшая нога).
    Возвращает (slippage_доля | None, причина-если-None).
    """
    worst = 0.0
    for name in (EXCHANGE_A, EXCHANGE_B):
        ex = _ob_client(name)
        hs = _book_halfspread(ex, symbol, notional_leg_usd)
        if hs is None:
            return None, f"нет стакана/глубины на {name}"
        worst = max(worst, hs)
    return worst, "ok"


def scan_one(symbol: str, since_ms: int, until_ms: int, timeframe: str,
             slippage_pct: float = 0.0002, use_orderbook: bool = False,
             fee: float = 0.0006) -> dict:
    """Прогон одной монеты за период. Возвращает строку результата (со статусом).

    Исключения не пробрасывает — оборачивает в строку-статус, чтобы падение одной
    монеты (делистинг, лимит глубины биржи, сеть) не валило весь скан.
    """
    try:
        return _scan_one_inner(symbol, since_ms, until_ms, timeframe,
                               slippage_pct, use_orderbook, fee)
    except Exception as e:  # noqa: BLE001 — изоляция отдельной монеты
        return _err_row(symbol, f"ошибка: {type(e).__name__}: {e}")


def _scan_one_inner(symbol: str, since_ms: int, until_ms: int, timeframe: str,
                    slippage_pct: float, use_orderbook: bool, fee: float) -> dict:
    measured = None
    if use_orderbook:
        # notional на ногу ≈ balance·risk_pct% / 2 (β=1 в cross_pct → ноги равны)
        leg_usd = START_BALANCE * 0.02 / 2.0
        measured, why = measure_slippage(symbol, leg_usd)
        if measured is None:
            return _err_row(symbol, f"стакан: {why}")
        slippage_pct = measured  # живой замер замещает фиксированный %

    cfg = make_cfg(symbol, timeframe, slippage_pct, fee)
    warmup_min = int(cfg.strategy.sma_period * 1.5) + 80  # cross_pct прогрев (~380)

    df = read_ohlcv_cross_range(cfg.strategy, since_ms, until_ms)
    if len(df) < warmup_min:
        r = _err_row(symbol, f"мало данных ({len(df)} баров)")
        r["bars"] = len(df)
        return r

    avg_spread_pct = float(((df["price_a"] - df["price_b"]).abs() / df["price_a"]).mean() * 100)
    eng = run_backtest(cfg, df, verbose=False)
    summ = eng.summary()
    return {
        "symbol": symbol,
        "status": "ok",
        "avg_spread_pct": round(avg_spread_pct, 4),
        "slip_meas_pct": round(measured * 100, 4) if measured is not None else "",
        "win_rate_pct": summ["win_rate_pct"],
        "trades": summ["trades"],
        "net_pnl_$": summ["net_pnl"],
        "final_equity_$": summ["equity"],
        "return_pct": summ["return_pct"],
        "bars": len(df),
    }


def _print_table(rows: list[dict]) -> None:
    hdr = ["symbol", "avg_spread_pct", "slip_meas_pct", "win_rate_pct", "trades",
           "net_pnl_$", "final_equity_$", "return_pct", "status"]
    widths = {h: len(h) for h in hdr}
    for r in rows:
        for h in hdr:
            widths[h] = max(widths[h], len(str(r.get(h, ""))))
    line = "  ".join(h.ljust(widths[h]) for h in hdr)
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(r.get(h, "")).ljust(widths[h]) for h in hdr))


def main() -> None:
    ap = argparse.ArgumentParser(description="Скан cross_pct по топ-монетам gateio×mexc")
    ap.add_argument("--top", type=int, default=20, help="число монет (по обороту)")
    ap.add_argument("--timeframe", default="5m", help="таймфрейм свечей")
    # gateio отдаёт максимум 10000 свечей: на 5m это ~34 дня. Дефолт — месяц.
    ap.add_argument("--days", type=int, default=30, help="глубина истории, дней")
    ap.add_argument("--workers", type=int, default=5, help="параллельных потоков (сетевой I/O)")
    ap.add_argument("--slippage", type=float, default=0.0002,
                    help="проскальзывание на ногу (доля; round-trip ≈ 4× от этого)")
    ap.add_argument("--fee", type=float, default=0.0006,
                    help="комиссия на ногу (доля). Мейкер-модель: --fee 0.0002 --slippage 0")
    ap.add_argument("--orderbook", action="store_true",
                    help="замерить slippage по живому стакану обеих бирж (вместо --slippage)")
    ap.add_argument("--rank", choices=("volume", "spread"), default="volume",
                    help="отбор топ-N: по обороту или по величине спреда")
    ap.add_argument("--symbols", default="", help="явный список через запятую вместо авто-топа")
    ap.add_argument("--out", default="", help="путь CSV (дефолт: pairsignal/out/scan_<date>.csv)")
    args = ap.parse_args()

    until_ms = int(time.time() * 1000)
    since_ms = until_ms - args.days * 24 * 3600 * 1000

    if args.symbols.strip():
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    elif args.rank == "spread":
        print(f"Отбор топ-{args.top} монет по величине спреда на {EXCHANGE_A}×{EXCHANGE_B}…")
        symbols = top_symbols_by_spread(args.top)
    else:
        print(f"Отбор топ-{args.top} монет по обороту на {EXCHANGE_A}×{EXCHANGE_B}…")
        symbols = top_symbols(args.top)

    # Параллелим по монетам: узкое место — сетевой fetch_ohlcv, на нём GIL отпускается.
    # Воркеров немного (дефолт 5), чтобы не упереться в rate-limit бирж.
    rows: list[dict] = []
    done = 0
    total = len(symbols)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = {pool.submit(scan_one, s, since_ms, until_ms, args.timeframe,
                            args.slippage, args.orderbook, args.fee): s
                for s in symbols}
        for fut in as_completed(futs):
            done += 1
            row = fut.result()  # scan_one не бросает — статус внутри строки
            print(f"[{done}/{total}] {row['symbol']} — {row['status']}", flush=True)
            rows.append(row)

    rows.sort(key=lambda r: r.get("return_pct", 0.0), reverse=True)

    print("\n=== СВОДКА (старт $1000 на монету) ===")
    _print_table(rows)

    if args.out:
        out_path = Path(args.out)
    else:
        date = datetime.fromtimestamp(until_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        out_path = Path(__file__).parent / "out" / f"scan_{date}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["symbol", "avg_spread_pct", "slip_meas_pct", "win_rate_pct", "trades",
              "net_pnl_$", "final_equity_$", "return_pct", "bars", "status"]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\nCSV сохранён: {out_path}")


if __name__ == "__main__":
    main()
