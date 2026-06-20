"""Загрузка непрерывной внутридневной истории FORTS-фьючерсов (MOEX ISS, 10m).

Каждая квартальная серия (SRU6, SRM6, ...) живёт ~3 мес. Чтобы получить длинную
непрерывную ценовую серию по активу, сшиваем последовательные контракты: каждый
контракт используется только в окне его наибольшей ликвидности — последние
`active_days` дней до экспирации. На стыке (rollover) переключаемся на следующий.

ISS хранит 10m-свечи и по ИСТЁКШИМ сериям (доступ по SECID) — глубина до ~2025-01.
Кэш — pairsignal/research/forts/cache/{ASSET}_10m.parquet.

Только чтение публичных свечей. Индикаторы считаются вызывающим по закрытым барам.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pairsignal.st4 import data_feed as st4

CACHE = Path(__file__).resolve().parent / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

# Явная цепочка квартальных контрактов по активу (старые → новые), с датой экспирации.
# Коды FORTS: <2 буквы актива><месяц-буква><год-цифра>. Месяцы: H=март, M=июнь, U=сен, Z=дек.
# Берём всё, что отдаёт ISS (включая истёкшие). Экспирации — 3-я пятница месяца контракта.
CHAINS: dict[str, list[tuple[str, str]]] = {
    "SBRF": [
        ("SRH6", "2026-03-19"),
        ("SRM6", "2026-06-18"),
        ("SRU6", "2026-09-17"),
    ],
    "GAZR": [
        ("GZH6", "2026-03-19"),
        ("GZM6", "2026-06-18"),
        ("GZU6", "2026-09-17"),
    ],
    "LKOH": [
        ("LKH6", "2026-03-19"),
        ("LKM6", "2026-06-18"),
        ("LKU6", "2026-09-17"),
    ],
}


def _fetch_series(code: str, since: datetime) -> pd.DataFrame:
    """Сырые 10m OHLCV одной серии с ISS, ts (UTC ms) индекс (полный OHLCV)."""
    frm = f"&from={since.strftime('%Y-%m-%d')}"
    out: dict[int, tuple] = {}
    start = 0
    while True:
        url = (f"{st4.ISS}/engines/futures/markets/forts/securities/{code}/candles.json"
               f"?iss.meta=off&interval=10{frm}&start={start}"
               "&candles.columns=open,high,low,close,volume,begin")
        rows = st4._table(st4._get(url), "candles")
        if not rows:
            break
        for r in rows:
            ts = int(datetime.strptime(r["begin"], "%Y-%m-%d %H:%M:%S")
                     .replace(tzinfo=st4._MSK).timestamp() * 1000)
            out[ts] = (float(r["open"]), float(r["high"]), float(r["low"]),
                       float(r["close"]), float(r.get("volume") or 0.0))
        start += len(rows)
        if len(rows) < 500:
            break
    df = pd.DataFrame.from_dict(out, orient="index",
                                columns=["open", "high", "low", "close", "volume"]).sort_index()
    df.index.name = "ts"
    return df


def build_continuous(asset: str, active_days: int = 90,
                     since: datetime | None = None, refresh: bool = False) -> pd.DataFrame:
    """Непрерывная стыкованная серия: каждый контракт используется в [exp-active_days; exp].

    Возвращает OHLCV-DataFrame с дополнительной колонкой `code` (какой контракт дал бар).
    Кэшируется в parquet.
    """
    cache_file = CACHE / f"{asset}_10m_{active_days}d.pkl"
    if cache_file.exists() and not refresh:
        return pd.read_pickle(cache_file)

    if since is None:
        since = datetime(2025, 1, 1, tzinfo=timezone.utc)
    chain = CHAINS[asset]
    parts: list[pd.DataFrame] = []
    prev_exp_ms = 0
    for code, exp_str in chain:
        exp = datetime.strptime(exp_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        exp_ms = int(exp.timestamp() * 1000)
        start_ms = int((exp.timestamp() - active_days * 86400) * 1000)
        df = _fetch_series(code, since)
        if df.empty:
            continue
        # окно ликвидности этого контракта: [exp-active_days; exp), но не раньше предыдущего стыка
        lo = max(start_ms, prev_exp_ms)
        win = df[(df.index >= lo) & (df.index < exp_ms)].copy()
        if win.empty:
            continue
        win["code"] = code
        parts.append(win)
        prev_exp_ms = exp_ms

    out = pd.concat(parts).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out.to_pickle(cache_file)
    return out


if __name__ == "__main__":
    for asset in ["SBRF", "GAZR"]:
        df = build_continuous(asset, active_days=90, refresh=True)
        first = pd.Timestamp(df.index[0], unit="ms", tz="UTC")
        last = pd.Timestamp(df.index[-1], unit="ms", tz="UTC")
        # сколько баров от каждого контракта
        by_code = df.groupby("code").size().to_dict()
        print(f"{asset}: {len(df)} bars  {first:%Y-%m-%d} -> {last:%Y-%m-%d}  {by_code}")
