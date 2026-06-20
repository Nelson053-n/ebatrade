"""Загрузка реальных 1h-перпов в кэш (parquet) — отдельно от боевых файлов.

Тянет топ-N ликвидных USDT-перпетуалов с биржи (gate/mexc) за N дней с пагинацией
(как fetch_ohlcv_paged в data_feed). Кэширует close-матрицу в parquet, чтобы не
дёргать биржу повторно при каждом бэктесте.

SURVIVORSHIP BIAS: юниверс = монеты, ЖИВЫЕ И ЛИКВИДНЫЕ СЕГОДНЯ. Делистнутые/умершие
за период монеты не попадают в выборку — это завышает результат любой стратегии,
особенно long-only momentum (мёртвые монеты обычно падали перед делистингом). Эффект
отмечаем явно; нивелировать частично можно market-neutral (long/short) тестами.

  python -m pairsignal.research.crypto.load_data --exchange gate --top 40 --days 400
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

CACHE = Path(__file__).parent / "cache"


def top_symbols(exchange: str, n: int, quote: str = "USDT") -> list[str]:
    """Топ-N перпетуалов по 24h-обороту на одной бирже."""
    import ccxt
    ex = getattr(ccxt, exchange)({"enableRateLimit": True, "timeout": 30000})
    ex.options["defaultType"] = "swap"
    ex.load_markets()
    valid = {s for s, m in ex.markets.items()
             if m.get("swap") and m.get("quote") == quote and m.get("active", True)
             and m.get("settle") == quote}
    tickers = ex.fetch_tickers()
    vols = {}
    for s in valid:
        t = tickers.get(s)
        if not t:
            continue
        qv = t.get("quoteVolume")
        if qv is None:
            qv = (t.get("baseVolume") or 0.0) * (t.get("last") or 0.0)
        vols[s] = float(qv or 0.0)
    ranked = sorted(vols, key=lambda s: vols[s], reverse=True)
    return ranked[:n]


def fetch_paged(ex, symbol: str, timeframe: str, since_ms: int, until_ms: int) -> pd.Series:
    """Постранично close [since;until) (копия логики data_feed.fetch_ohlcv_paged)."""
    step = ex.parse_timeframe(timeframe) * 1000
    out: dict[int, float] = {}
    since = since_ms
    while since < until_ms:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
        if not batch:
            break
        for c in batch:
            ts = int(c[0])
            if ts >= until_ms:
                break
            out[ts] = float(c[4])
        nxt = int(batch[-1][0]) + step
        if nxt <= since:
            break
        since = nxt
    s = pd.Series(out, dtype="float64").sort_index()
    s.index.name = "ts"
    return s


def load(exchange: str, symbols: list[str], timeframe: str, since_ms: int,
         until_ms: int, workers: int, min_coverage: float = 0.85) -> pd.DataFrame:
    import ccxt

    def _one(sym: str):
        ex = getattr(ccxt, exchange)({"enableRateLimit": True, "timeout": 30000})
        ex.options["defaultType"] = "swap"
        return sym, fetch_paged(ex, sym, timeframe, since_ms, until_ms)

    cols: dict[str, pd.Series] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = [pool.submit(_one, s) for s in symbols]
        for fut in as_completed(futs):
            try:
                sym, s = fut.result()
                if len(s) > 0:
                    cols[sym] = s
            except Exception as e:  # noqa: BLE001
                print("  skip", e)
                continue
    if min_coverage > 0.0 and cols:
        longest = max(len(s) for s in cols.values())
        cols = {k: s for k, s in cols.items() if len(s) >= min_coverage * longest}
    df = pd.DataFrame(cols).dropna().sort_index()
    return df


def cache_path(exchange: str, timeframe: str, days: int, top: int) -> Path:
    return CACHE / f"{exchange}_{timeframe}_{days}d_top{top}.pkl"


def get_prices(exchange: str, timeframe: str, days: int, top: int,
               workers: int = 6, force: bool = False) -> pd.DataFrame:
    """Закэшированная матрица close. Грузит из сети только если кэша нет/force."""
    path = cache_path(exchange, timeframe, days, top)
    if path.exists() and not force:
        df = pd.read_pickle(path)
        df.index = df.index.astype("int64")
        return df
    CACHE.mkdir(parents=True, exist_ok=True)
    until_ms = int(time.time() * 1000)
    since_ms = until_ms - days * 24 * 3600 * 1000
    print(f"Отбор топ-{top} перпов на {exchange}…")
    symbols = top_symbols(exchange, top)
    print(f"Загрузка {len(symbols)} монет ({days}д, {timeframe})…")
    df = load(exchange, symbols, timeframe, since_ms, until_ms, workers)
    df.to_pickle(path)
    print(f"Сохранено: {path}  ({df.shape[1]} монет, {len(df)} баров)")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchange", default="gate")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--days", type=int, default=400)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    df = get_prices(args.exchange, args.timeframe, args.days, args.top,
                    args.workers, args.force)
    import datetime as dt
    def _d(ts): return dt.datetime.fromtimestamp(ts/1000, tz=dt.timezone.utc).strftime("%Y-%m-%d")
    print(f"\nИтог: {df.shape[1]} монет, {len(df)} баров "
          f"({_d(df.index[0])} … {_d(df.index[-1])})")
    print("Монеты:", ", ".join(s.split('/')[0] for s in df.columns))


if __name__ == "__main__":
    main()
