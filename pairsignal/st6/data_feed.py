"""Источник данных st6 — MOEX ISS (рынок акций, engine=stock/market=shares).

ISS отдаёт дневные/часовые свечи акций без авторизации и без ключей. Тянем
закрытия по тикерам корзины и выравниваем их по общему набору дат (inner-join),
чтобы ряды были синхронны для корреляции/спреда. Плюс синтетический генератор
коинтегрированной корзины для офлайн-теста.

ВНИМАНИЕ: интервалы ISS — 1/10/60/24/7/31 (минуты, кроме 24=дни, 7=неделя,
31=месяц). st6 по умолчанию работает на дневных (interval=24) ради глубины истории.
Индикаторы — только по закрытым свечам (последний формирующийся бар отбрасывает
вызывающий код, как в st4/st5).
"""
from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

ISS = "https://iss.moex.com/iss"
_MSK = timezone(timedelta(hours=3))
_HTTP_TIMEOUT = 30
# таймфрейм ТЗ → код интервала свечей ISS
_INTERVAL = {"1d": 24, "1h": 60, "10m": 10, "1m": 1}


def _get(url: str) -> dict:
    """GET JSON с ISS (UA — ISS режет дефолтный python-urllib)."""
    req = urllib.request.Request(url, headers={"User-Agent": "pairsignal-st6/1.0"})
    with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as r:  # noqa: S310 (доверенный хост)
        return json.loads(r.read().decode("utf-8"))


def _table(doc: dict, name: str) -> list[dict]:
    """ISS-блок {columns,data} → список dict'ов по именам колонок."""
    blk = doc[name]
    cols = blk["columns"]
    return [dict(zip(cols, row)) for row in blk["data"]]


def read_closes_moex(secid: str, since: datetime, interval: str = "1d") -> pd.Series:
    """Закрытия акции `secid` с даты `since` → Series[ts_ms -> close], сортировано.

    ISS отдаёт свечи страницами по ~500 от старых к новым через start=; листаем до
    конца. begin метим как московское время сессии → корректный UTC unix ms.
    """
    icode = _INTERVAL.get(interval, 24)
    frm = since.strftime("%Y-%m-%d")
    closes: dict[int, float] = {}
    start = 0
    while True:
        url = (f"{ISS}/engines/stock/markets/shares/securities/{secid}/candles.json"
               f"?iss.meta=off&interval={icode}&from={frm}&start={start}"
               "&candles.columns=close,begin")
        rows = _table(_get(url), "candles")
        if not rows:
            break
        for r in rows:
            ts = int(datetime.strptime(r["begin"], "%Y-%m-%d %H:%M:%S")
                     .replace(tzinfo=_MSK).timestamp() * 1000)
            closes[ts] = float(r["close"])
        start += len(rows)
        if len(rows) < 500:  # последняя страница
            break
    s = pd.Series(closes, dtype="float64").sort_index()
    s.index.name = "ts"
    return s


def load_basket(tickers: list[str], days: int = 720,
                interval: str = "1d") -> dict[str, list[float]]:
    """Закрытия корзины, выровненные по общим датам (inner-join по ts).

    Возвращает {ticker -> list[float]} одинаковой длины. Тикеры без данных
    пропускаются. Используется и в live, и в бэктесте.
    """
    since = datetime.now(timezone.utc) - timedelta(days=days)
    series: dict[str, pd.Series] = {}
    for t in tickers:
        try:
            s = read_closes_moex(t, since, interval)
        except Exception:  # noqa: BLE001  тикер недоступен/сетевой сбой — пропуск
            continue
        if len(s) > 10:
            series[t] = s
    if not series:
        return {}
    df = pd.DataFrame(series).dropna().sort_index()
    return {t: df[t].tolist() for t in df.columns}


def load_basket_df(tickers: list[str], days: int = 720,
                   interval: str = "1d") -> pd.DataFrame:
    """То же, что load_basket, но как DataFrame[ts -> close по тикерам] (для бэктеста)."""
    since = datetime.now(timezone.utc) - timedelta(days=days)
    series: dict[str, pd.Series] = {}
    for t in tickers:
        try:
            s = read_closes_moex(t, since, interval)
        except Exception:  # noqa: BLE001
            continue
        if len(s) > 10:
            series[t] = s
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series).dropna().sort_index()


# --------------------------------------------------------------------------
# Синтетика: корзина с одной коинтегрированной парой (для офлайн-теста)
# --------------------------------------------------------------------------
def synthetic_basket(n: int = 1500, seed: int = 7, n_noise: int = 2,
                     beta: float = 1.0) -> dict[str, list[float]]:
    """Корзина из коинтегрированной пары AA/BB + независимых «шумовых» тикеров.

    AA = beta*BB + стационарный OU-спред — высокая корреляция и возврат к среднему.
    Шумовые ряды — независимые случайные блуждания (сканер их отсеет). Метрики на
    синтетике ничего не говорят о реальной доходности — это проверка логики.
    """
    rng = np.random.default_rng(seed)
    # дрейф общего фактора сильнее шума спреда → высокая корреляция (~0.92),
    # спред — стационарный OU вокруг нуля (быстрый возврат к среднему).
    log_b = np.cumsum(rng.normal(0, 0.02, n)) + np.log(100.0)
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = spread[t - 1] * (1 - 0.05) + rng.normal(0, 0.008)
    log_a = beta * log_b + spread + np.log(1.5)
    out: dict[str, list[float]] = {
        "AA": np.exp(log_a).tolist(),
        "BB": np.exp(log_b).tolist(),
    }
    for k in range(n_noise):
        noise = np.exp(np.cumsum(rng.normal(0, 0.012, n)) + np.log(50.0 + 10 * k))
        out[f"NN{k}"] = noise.tolist()
    return out
