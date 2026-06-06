"""Источник данных.

read_ohlcv_ccxt — реальные котировки (только чтение, ключи не нужны для публичных свечей).
generate_synthetic — коинтегрированная пара для офлайн-демо/тестов без сети.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StrategyConfig


def read_ohlcv_ccxt(cfg: StrategyConfig, limit: int = 1000) -> pd.DataFrame:
    """Тянем close обеих ног через CCXT и выравниваем по времени.

    Публичные OHLCV не требуют API-ключей. Сетевой доступ к бирже нужен в рантайме.
    """
    import ccxt  # импорт внутри, чтобы офлайн-демо работало без установленного ccxt

    ex = getattr(ccxt, cfg.data_exchange)({"enableRateLimit": True})
    ex.options["defaultType"] = "swap"

    def _close(symbol: str) -> pd.Series:
        raw = ex.fetch_ohlcv(symbol, timeframe=cfg.timeframe, limit=limit)
        s = pd.Series(
            [c[4] for c in raw], index=[int(c[0]) for c in raw], dtype="float64"
        )
        s.index.name = "ts"
        return s

    a = _close(cfg.symbol_a)
    b = _close(cfg.symbol_b)
    df = pd.DataFrame({"price_a": a, "price_b": b}).dropna()
    df = df.sort_index()
    return df


def read_ohlcv_cross(cfg: StrategyConfig, limit: int = 1000) -> pd.DataFrame:
    """Кросс-биржевой режим: ОДИН символ (symbol_cross) с ДВУХ бирж.

    price_a — цена на cfg.exchange_a (BitMEX), price_b — на cfg.exchange_b (OKX).
    Выравниваем по ts (inner-join) — спред считается только по общим свечам.
    Реальные CCXT-id символа SUI на разных биржах могут отличаться; уточнять при включении.
    """
    import ccxt  # импорт внутри, чтобы офлайн-демо работало без установленного ccxt

    def _ex(name: str):
        ex = getattr(ccxt, name)({"enableRateLimit": True})
        ex.options["defaultType"] = "swap"
        return ex

    def _close(ex, symbol: str) -> pd.Series:
        raw = ex.fetch_ohlcv(symbol, timeframe=cfg.timeframe, limit=limit)
        s = pd.Series([c[4] for c in raw], index=[int(c[0]) for c in raw], dtype="float64")
        s.index.name = "ts"
        return s

    a = _close(_ex(cfg.exchange_a), cfg.symbol_cross)
    b = _close(_ex(cfg.exchange_b), cfg.symbol_cross)
    df = pd.DataFrame({"price_a": a, "price_b": b}).dropna().sort_index()
    return df


def fetch_ohlcv_paged(ex, symbol: str, timeframe: str, since_ms: int, until_ms: int) -> pd.Series:
    """Постранично тянет close [since_ms; until_ms) через ccxt.fetch_ohlcv(since=...).

    Пачками по 1000 свечей: сдвигаем since на ts последней + шаг ТФ. Стоп, когда биржа
    вернула < limit, ts вышли за until_ms, либо since не продвинулся (защита от зацикливания).
    Возвращает Series close с индексом ts(ms).
    """
    step = ex.parse_timeframe(timeframe) * 1000  # длина свечи в мс
    out: dict[int, float] = {}
    since = since_ms
    # Биржи отдают переменный размер страницы (gateio ~999, mexc 1000) и часто
    # покрывают лишь несколько дней за вызов — крутим, пока не дойдём до until_ms.
    # Остановка только по: пустой ответ / выход за until_ms / непродвижение since.
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
        if nxt <= since:  # биржа не продвинулась — прерываем
            break
        since = nxt
    s = pd.Series(out, dtype="float64").sort_index()
    s.index.name = "ts"
    return s


def read_ohlcv_cross_range(cfg: StrategyConfig, since_ms: int, until_ms: int) -> pd.DataFrame:
    """Как read_ohlcv_cross, но за диапазон [since_ms; until_ms) с пагинацией обеих бирж.

    price_a — cfg.exchange_a, price_b — cfg.exchange_b, символ — cfg.symbol_cross.
    inner-join по ts (dropna): спред считается только по общим свечам.
    """
    import ccxt

    def _ex(name: str):
        ex = getattr(ccxt, name)({"enableRateLimit": True})
        ex.options["defaultType"] = "swap"
        return ex

    a = fetch_ohlcv_paged(_ex(cfg.exchange_a), cfg.symbol_cross, cfg.timeframe, since_ms, until_ms)
    b = fetch_ohlcv_paged(_ex(cfg.exchange_b), cfg.symbol_cross, cfg.timeframe, since_ms, until_ms)
    df = pd.DataFrame({"price_a": a, "price_b": b}).dropna().sort_index()
    return df


def generate_synthetic_cross(n: int = 3000, seed: int = 11) -> pd.DataFrame:
    """Кросс-биржевая синтетика: одна монета (SUI) на двух биржах с малым спредом.

    Обе биржи двигает общий рыночный фактор (цены почти равны), поверх — mean-reverting
    рассогласование s (OU вокруг 0). Размах s сделан ЗАВЕДОМО > 2·band_pct (~±6% против
    коридора ±3%), иначе границы не пробиваются и сделок нет (для обкатки логики).
    """
    rng = np.random.default_rng(seed)
    base = 2.0 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))   # общий уровень SUI (~$2)

    # рассогласование бирж: OU вокруг 0, амплитуда ~±6% (> 2·band_pct=6%)
    s = np.zeros(n)
    theta, sigma = 0.05, 0.018
    for i in range(1, n):
        s[i] = s[i - 1] + theta * (0.0 - s[i - 1]) + rng.normal(0, sigma)
    s = np.clip(s, -0.06, 0.06)

    price_a = base * (1 + s / 2)     # биржа A (BitMEX)
    price_b = base * (1 - s / 2)     # биржа B (OKX)
    ts = (np.arange(n) * 300_000 + 1_700_000_000_000).astype("int64")  # шаг 5m
    df = pd.DataFrame({"price_a": price_a, "price_b": price_b}, index=ts)
    df.index.name = "ts"
    return df


def generate_synthetic(n: int = 3000, seed: int = 7) -> pd.DataFrame:
    """Пара с возвращающимся к среднему лог-спредом — чтобы демо реально давало сигналы."""
    rng = np.random.default_rng(seed)
    # общий рыночный фактор (случайное блуждание)
    factor = np.cumsum(rng.normal(0, 0.004, n))
    p_eth = 3000 * np.exp(factor)

    # спред = ln(BTC) - ln(ETH) колеблется вокруг ~2.4 (BTC/ETH ≈ 11) по OU-процессу
    spread = np.zeros(n)
    spread[0] = 2.4
    theta, mu, sigma = 0.02, 2.4, 0.01
    for i in range(1, n):
        spread[i] = spread[i - 1] + theta * (mu - spread[i - 1]) + rng.normal(0, sigma)

    p_btc = np.exp(np.log(p_eth) + spread)
    ts = (np.arange(n) * 300_000 + 1_700_000_000_000).astype("int64")  # шаг 5m
    df = pd.DataFrame({"price_a": p_btc, "price_b": p_eth}, index=ts)
    df.index.name = "ts"
    return df
