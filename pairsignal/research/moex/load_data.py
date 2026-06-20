"""Загрузка дневных закрытий MOEX ISS для walk-forward исследования st6.

Тянем широкую корзину ликвидных акций (нефтегаз, металлурги, банки, прочее),
глубокая история (по умолчанию ~6 лет). Кэшируем сырьё по тикеру в cache/, плюс
выровненный inner-join DataFrame. ISS — без ключей, дневные свечи (interval=24).
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

# переиспользуем боевой ридер ISS (только чтение)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pairsignal.st6.data_feed import read_closes_moex  # noqa: E402

CACHE = Path(__file__).resolve().parent / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

# Широкая корзина ликвидных акций MOEX, сгруппированная по секторам.
# Внутрисекторные пары — экономически обоснованные кандидаты (один драйвер).
BASKET: dict[str, list[str]] = {
    "oil_gas": ["LKOH", "ROSN", "SIBN", "TATN", "GAZP", "NVTK", "TATNP", "BANE", "RNFT"],
    "metals": ["GMKN", "NLMK", "MAGN", "CHMF", "PLZL", "RUAL", "ALRS"],
    "banks": ["SBER", "SBERP", "VTBR", "CBOM"],
    "power": ["HYDR", "IRAO", "FEES", "UPRO"],
    "telecom_it": ["MTSS", "RTKM", "YDEX"],
    "consumer": ["MGNT", "FIVE", "FIXP"],
}
ALL_TICKERS = [t for sub in BASKET.values() for t in sub]


def load_all(days: int = 2200, interval: str = "1d", refresh: bool = False) -> pd.DataFrame:
    """Закрытия всех тикеров корзины, кэш по тикеру. Возвращает DataFrame[ts -> close]."""
    since = datetime.now(timezone.utc) - timedelta(days=days)
    series: dict[str, pd.Series] = {}
    for t in ALL_TICKERS:
        fp = CACHE / f"{t}_{interval}.csv"
        if fp.exists() and not refresh:
            s = pd.read_csv(fp, index_col=0)["close"]
            s.index = s.index.astype("int64")
        else:
            try:
                s = read_closes_moex(t, since, interval)
            except Exception as e:  # noqa: BLE001
                print(f"  {t}: SKIP ({e})")
                continue
            if len(s) > 10:
                s.to_csv(fp, header=["close"])
        if len(s) > 10:
            series[t] = s
            print(f"  {t}: {len(s)} bars  [{datetime.fromtimestamp(s.index[0]/1000, tz=timezone.utc).date()} .. {datetime.fromtimestamp(s.index[-1]/1000, tz=timezone.utc).date()}]")
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series).sort_index()


if __name__ == "__main__":
    refresh = "--refresh" in sys.argv
    df = load_all(refresh=refresh)
    print(f"\nRaw union: {df.shape[0]} dates x {df.shape[1]} tickers")
    aligned = df.dropna()
    print(f"Inner-join (all present): {aligned.shape[0]} dates x {aligned.shape[1]} tickers")
    if len(aligned):
        print(f"Range: {datetime.fromtimestamp(aligned.index[0]/1000, tz=timezone.utc).date()} .. {datetime.fromtimestamp(aligned.index[-1]/1000, tz=timezone.utc).date()}")
    df.to_csv(CACHE / "union_1d.csv")
    print(f"Saved union -> {CACHE/'union_1d.csv'}")
