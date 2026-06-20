"""Загрузка OHLC дневных + часовых баров MOEX ISS для календарных/микроструктурных тестов.

Существующий cache хранит только close. Для overnight (close→open) и
intraday-декомпозиции нужны open/high/low/close. Тянем OHLC через ISS (без ключей),
кэшируем в cache/ohlc/{TICKER}_{interval}.csv. Только чтение.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pairsignal.research.moex.load_data import ALL_TICKERS  # noqa: E402

ISS = "https://iss.moex.com/iss"
_MSK = timezone(timedelta(hours=3))
_INTERVAL = {"1d": 24, "1h": 60}
CACHE = Path(__file__).resolve().parent / "cache" / "ohlc"
CACHE.mkdir(parents=True, exist_ok=True)


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "pairsignal-cal/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310
        return json.loads(r.read().decode("utf-8"))


def _table(doc: dict, name: str) -> list[dict]:
    blk = doc[name]
    cols = blk["columns"]
    return [dict(zip(cols, row)) for row in blk["data"]]


def read_ohlc_moex(secid: str, since: datetime, interval: str = "1d") -> pd.DataFrame:
    """OHLC свечи акции → DataFrame[ts_ms -> open,high,low,close,volume], сортировано."""
    icode = _INTERVAL.get(interval, 24)
    frm = since.strftime("%Y-%m-%d")
    rows_all: dict[int, dict] = {}
    start = 0
    while True:
        url = (f"{ISS}/engines/stock/markets/shares/securities/{secid}/candles.json"
               f"?iss.meta=off&interval={icode}&from={frm}&start={start}"
               "&candles.columns=open,high,low,close,volume,begin")
        rows = _table(_get(url), "candles")
        if not rows:
            break
        for r in rows:
            ts = int(datetime.strptime(r["begin"], "%Y-%m-%d %H:%M:%S")
                     .replace(tzinfo=_MSK).timestamp() * 1000)
            rows_all[ts] = {
                "open": float(r["open"]), "high": float(r["high"]),
                "low": float(r["low"]), "close": float(r["close"]),
                "volume": float(r["volume"]), "begin": r["begin"],
            }
        start += len(rows)
        if len(rows) < 500:
            break
    df = pd.DataFrame.from_dict(rows_all, orient="index").sort_index()
    df.index.name = "ts"
    return df


def load_ohlc(interval: str = "1d", days: int = 2200, refresh: bool = False) -> dict[str, pd.DataFrame]:
    since = datetime.now(timezone.utc) - timedelta(days=days)
    out: dict[str, pd.DataFrame] = {}
    for t in ALL_TICKERS:
        fp = CACHE / f"{t}_{interval}.csv"
        if fp.exists() and not refresh:
            df = pd.read_csv(fp, index_col=0)
            df.index = df.index.astype("int64")
        else:
            try:
                df = read_ohlc_moex(t, since, interval)
            except Exception as e:  # noqa: BLE001
                print(f"  {t}: SKIP ({e})")
                continue
            if len(df) > 10:
                df.to_csv(fp)
        if len(df) > 10:
            out[t] = df
            d0 = datetime.fromtimestamp(df.index[0] / 1000, tz=timezone.utc).date()
            d1 = datetime.fromtimestamp(df.index[-1] / 1000, tz=timezone.utc).date()
            print(f"  {t}: {len(df)} bars [{d0} .. {d1}]")
    return out


if __name__ == "__main__":
    refresh = "--refresh" in sys.argv
    for iv in ("1d", "1h"):
        print(f"=== {iv} ===")
        d = load_ohlc(iv, refresh=refresh)
        print(f"{iv}: {len(d)} tickers loaded\n")
