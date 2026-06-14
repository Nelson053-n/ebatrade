"""VWAP-индикатор st5: внутридневной VWAP + коридор ±k·σ отклонений цены от VWAP.

VWAP сбрасывается каждый ТОРГОВЫЙ ДЕНЬ (день сессии FORTS в TZ MSK): копит Σ(typical·vol)
и Σvol с начала дня. Коридор строится на σ отклонений (price − vwap) текущего дня —
аналог BB, но якорь дрейфует к концу дня (нормально: VWAP — внутридневной ориентир).

Все расчёты — по закрытым свечам (no repaint).
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from .models import PriceBar, VwapReading

_MSK = timezone(timedelta(hours=3))   # сессия FORTS в МСК (день VWAP считаем по ней)


def _day_key(ts_ms: int) -> str:
    """Торговый день (YYYY-MM-DD) в TZ MSK — ключ сброса VWAP."""
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).astimezone(_MSK).strftime("%Y-%m-%d")


class IntradayVwap:
    """Потоковый внутридневной VWAP + σ отклонений. Сбрасывается на новый день.

    band_sigma — полуширина коридора. min_bars — сколько баров дня нужно, чтобы σ была
    осмысленной (утренний шум на 1-2 барах даёт вырожденный коридор → is_ready=False).
    std_mode: Population (/N) | Sample (/(N−1)).
    """

    def __init__(self, band_sigma: float = 2.0, min_bars: int = 6,
                 std_mode: str = "Population") -> None:
        self.k = band_sigma
        self.min_bars = min_bars
        self.std_mode = std_mode
        self._day = ""
        self._sum_pv = 0.0      # Σ(typical · volume)
        self._sum_v = 0.0       # Σ volume
        self._dev2: list[float] = []   # отклонения (price − vwap) для σ дня
        self._n = 0

    def _reset(self, day: str) -> None:
        self._day = day
        self._sum_pv = 0.0
        self._sum_v = 0.0
        self._dev2 = []
        self._n = 0

    @property
    def is_ready(self) -> bool:
        return self._n >= self.min_bars

    def update(self, bar: PriceBar) -> VwapReading:
        """Добавить закрытый бар, вернуть срез VWAP (после добавления)."""
        day = _day_key(bar.ts)
        if day != self._day:
            self._reset(day)

        # объём может быть 0 (тонкий бар/нет данных) — тогда вес даём по typical с весом 1,
        # иначе VWAP «застынет». Это редкий край; на ликвидных сериях volume > 0.
        w = bar.volume if bar.volume > 0 else 1.0
        self._sum_pv += bar.typical * w
        self._sum_v += w
        self._n += 1
        vwap = self._sum_pv / self._sum_v if self._sum_v > 0 else bar.close

        self._dev2.append(bar.close - vwap)
        if not self.is_ready:
            return VwapReading(ts=bar.ts, price=bar.close, vwap=vwap, sigma=float("nan"),
                               upper=float("nan"), lower=float("nan"), is_ready=False)

        n = len(self._dev2)
        mean = math.fsum(self._dev2) / n
        ddof = 0 if self.std_mode == "Population" else 1
        denom = n - ddof
        var = math.fsum((d - mean) ** 2 for d in self._dev2) / denom if denom > 0 else 0.0
        sigma = math.sqrt(var)
        return VwapReading(ts=bar.ts, price=bar.close, vwap=vwap, sigma=sigma,
                           upper=vwap + self.k * sigma, lower=vwap - self.k * sigma,
                           is_ready=True)


class VolumeAverage:
    """SMA объёма за день (для объёмного фильтра входа). Сброс на новый день."""

    def __init__(self) -> None:
        self._day = ""
        self._sum = 0.0
        self._n = 0

    def update(self, ts_ms: int, volume: float) -> float:
        day = _day_key(ts_ms)
        if day != self._day:
            self._day = day
            self._sum = 0.0
            self._n = 0
        self._sum += volume
        self._n += 1
        return self._sum / self._n if self._n else float("nan")
