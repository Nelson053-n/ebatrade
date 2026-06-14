"""Источник данных st5 — MOEX ISS (один инструмент, OHLCV).

Переиспользует ISS-плумбинг st4 (list_series, nearest_series, _candles_ohlcv, _get) —
без дублирования HTTP-кода. Отличие от st4: одна нога вместо пары, нужен полный OHLCV
(VWAP считается по typical price и объёму, не только по close).
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..st4 import data_feed as _st4feed
from .config import St5Config
from .models import InstrumentSpec

# реэкспорт общих ISS-функций (один источник истины — st4)
list_series = _st4feed.list_series
nearest_series = _st4feed.nearest_series
leg_margin = _st4feed.leg_margin


def instrument_spec(secid: str) -> InstrumentSpec:
    """Спецификация серии (MINSTEP/STEPPRICE/LOTVOLUME/LASTTRADEDATE) — модель st5."""
    s = _st4feed.instrument_spec(secid, _st4feed.Role.ORDINARY)  # role игнорируем
    return InstrumentSpec(code=s.code, tick_size=s.tick_size,
                          tick_value_rub=s.tick_value_rub, lot=s.lot, expiry=s.expiry)


def resolve_leg(cfg: St5Config) -> InstrumentSpec:
    """Определить код и спецификацию инструмента (с учётом роллировера)."""
    inst = cfg.instrument
    if not inst.auto_rollover and inst.leg_code:
        code = inst.leg_code
    else:
        code = nearest_series(inst.asset, inst.rollover_days_before_expiry)["SECID"]
    return instrument_spec(code)


def _ohlcv(secid: str, interval: int, since: datetime | None = None,
           count: int | None = None) -> pd.DataFrame:
    """OHLCV-свечи серии по begin-времени (UTC ms) — обёртка над _candles_ohlcv st4.

    Возвращает DataFrame с колонками open, high, low, close, volume (индекс ts).
    """
    icode = _st4feed._INTERVAL.get(interval, 10)
    if since is None and count is not None:
        from datetime import timedelta
        span_min = count * interval * 1.6 + 3 * 24 * 60
        since = datetime.now(timezone.utc) - timedelta(minutes=span_min)
    frm = f"&from={since.strftime('%Y-%m-%d')}" if since else ""
    out: dict[int, tuple] = {}
    start = 0
    while True:
        url = (f"{_st4feed.ISS}/engines/futures/markets/forts/securities/{secid}/candles.json"
               f"?iss.meta=off&interval={icode}{frm}&start={start}"
               "&candles.columns=open,high,low,close,volume,begin")
        rows = _st4feed._table(_st4feed._get(url), "candles")
        if not rows:
            break
        for r in rows:
            ts = int(datetime.strptime(r["begin"], "%Y-%m-%d %H:%M:%S")
                     .replace(tzinfo=_st4feed._MSK).timestamp() * 1000)
            out[ts] = (float(r["open"]), float(r["high"]), float(r["low"]),
                       float(r["close"]), float(r.get("volume") or 0.0))
        start += len(rows)
        if len(rows) < 500:
            break
    df = pd.DataFrame.from_dict(out, orient="index",
                                columns=["open", "high", "low", "close", "volume"])
    df = df.sort_index()
    df.index.name = "ts"
    if count is not None:
        df = df.iloc[-count:]
    return df


def read_ohlcv_moex(cfg: St5Config, limit: int = 600, code: str | None = None) -> pd.DataFrame:
    """OHLCV инструмента за последние ~limit баров."""
    if code is None:
        code = resolve_leg(cfg).code
    return _ohlcv(code, cfg.strategy.candle_interval_minutes, count=limit)


def read_ohlcv_moex_range(cfg: St5Config, since: datetime, code: str | None = None) -> pd.DataFrame:
    """OHLCV инструмента за период [since; now) — для бэктеста."""
    if code is None:
        code = resolve_leg(cfg).code
    return _ohlcv(code, cfg.strategy.candle_interval_minutes, since=since)


def generate_synthetic(n: int = 1500, seed: int = 17, interval_min: int = 10) -> pd.DataFrame:
    """Синтетика OHLCV одиночного инструмента: цена вокруг внутридневного якоря (OU),
    с дневным сбросом — чтобы VWAP-reversion имел что ловить (цена отходит от VWAP и
    возвращается). Объём — пуассон вокруг среднего (для объёмного фильтра).
    """
    rng = np.random.default_rng(seed)
    step = interval_min * 60_000
    base_ts = 1_700_000_000_000
    bars_per_day = max(1, int(14 * 60 / interval_min))   # ~14 торговых часов FORTS

    close = np.zeros(n)
    anchor = 32000.0
    price = anchor
    for i in range(n):
        if i % bars_per_day == 0:          # новый день: якорь дрейфует
            anchor *= float(np.exp(rng.normal(0, 0.004)))
            price = anchor
        # OU-возврат к якорю дня + шум
        price += 0.05 * (anchor - price) + rng.normal(0, 18.0)
        close[i] = price
    close = np.round(close)
    spread_hl = np.abs(rng.normal(0, 8.0, n)) + 2.0
    high = np.round(close + spread_hl)
    low = np.round(close - spread_hl)
    open_ = np.round(close + rng.normal(0, 5.0, n))
    volume = np.maximum(1, rng.poisson(120, n)).astype(float)
    ts = (np.arange(n) * step + base_ts).astype("int64")
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                       "volume": volume}, index=ts)
    df.index.name = "ts"
    return df


def synthetic_spec() -> InstrumentSpec:
    """Спецификация-заглушка для офлайн-режима (как реальная серия SR*)."""
    return InstrumentSpec(code="STSYN", tick_size=1.0, tick_value_rub=1.0, lot=1, expiry=None)
