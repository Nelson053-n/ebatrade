"""Индикаторы st5.

Основной — MomentumIndicator: буфер закрытий, на каждом баре сравнивает текущий close
с close[-lookback] и выдаёт направление сигнала. Все расчёты — по ЗАКРЫТЫМ свечам (no
repaint): сигнал считается только после полного бара, сравнение — с уже закрытым баром.

IntradayVwap/VolumeAverage оставлены DEPRECATED для совместимости (логика их не вызывает).
"""
from __future__ import annotations

import math
from collections import deque
from datetime import datetime, timedelta, timezone

from .models import MomentumReading, PriceBar, VwapReading, ZScoreReading

_MSK = timezone(timedelta(hours=3))   # сессия FORTS в МСК (день VWAP считаем по ней)


class MomentumIndicator:
    """Потоковый directional momentum: close[i] vs close[i−lookback].

    Держит кольцевой буфер последних (lookback+1) закрытий. На баре i:
      signal = +1, если close[i] > close[i−lookback] (тренд вверх → LONG);
      signal = −1, если close[i] < close[i−lookback] (тренд вниз → SHORT);
      signal =  0, если равны.
    is_ready — когда накоплено > lookback баров (есть close[i−lookback] для сравнения).
    Без дневного сброса: momentum внутридневной по баровому окну, а не по якорю дня.
    """

    def __init__(self, lookback: int = 48) -> None:
        self.lookback = max(1, int(lookback))
        # храним lookback+1 закрытий: текущий + тот, что lookback баров назад
        self._closes: deque[float] = deque(maxlen=self.lookback + 1)

    @property
    def is_ready(self) -> bool:
        return len(self._closes) > self.lookback

    def update(self, close: float) -> MomentumReading:
        """Добавить close закрытого бара, вернуть срез momentum (после добавления)."""
        self._closes.append(close)
        if not self.is_ready:
            return MomentumReading(ts=0, price=close, ref_price=float("nan"),
                                   signal=0, lookback_return=float("nan"), is_ready=False)
        ref = self._closes[0]          # close[-lookback] (буфер длиной lookback+1)
        signal = 1 if close > ref else (-1 if close < ref else 0)
        ret = (close - ref) / ref if ref else 0.0
        return MomentumReading(ts=0, price=close, ref_price=ref, signal=signal,
                               lookback_return=ret, is_ready=True)


class ZScoreIndicator:
    """Потоковый mean-reversion z-score: (close[i] − SMA)/std по ma_n закрытым барам.

    Держит кольцевой буфер последних ma_n закрытий. На баре i (когда накоплено ma_n):
      sma = mean(closes), std = population-std (ddof=0) — как rolling(ma_n).std(ddof=0)
      в research; z = (close − sma)/std при std > 0.
    is_ready — буфер заполнен (≥ ma_n баров) И std > 0 (вырожденный плоский участок → не готов).
    Без дневного сброса: окно скользящее по ma_n барам (no repaint — только закрытые бары).
    """

    def __init__(self, ma_n: int = 36) -> None:
        self.ma_n = max(2, int(ma_n))
        self._closes: deque[float] = deque(maxlen=self.ma_n)

    @property
    def is_ready(self) -> bool:
        return len(self._closes) >= self.ma_n

    def update(self, close: float) -> ZScoreReading:
        """Добавить close закрытого бара, вернуть z-срез (после добавления)."""
        self._closes.append(close)
        if len(self._closes) < self.ma_n:
            return ZScoreReading(ts=0, price=close, sma=float("nan"), std=float("nan"),
                                 z=float("nan"), is_ready=False)
        n = len(self._closes)
        mean = math.fsum(self._closes) / n
        var = math.fsum((c - mean) ** 2 for c in self._closes) / n   # ddof=0 (population)
        std = math.sqrt(var)
        if std <= 0:
            return ZScoreReading(ts=0, price=close, sma=mean, std=0.0,
                                 z=float("nan"), is_ready=False)
        z = (close - mean) / std
        return ZScoreReading(ts=0, price=close, sma=mean, std=std, z=z, is_ready=True)


def _day_key(ts_ms: int) -> str:
    """Торговый день (YYYY-MM-DD) в TZ MSK — ключ сброса VWAP."""
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).astimezone(_MSK).strftime("%Y-%m-%d")


class IntradayVwap:
    """DEPRECATED (VWAP-reversion). Потоковый внутридневной VWAP + σ отклонений.

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
