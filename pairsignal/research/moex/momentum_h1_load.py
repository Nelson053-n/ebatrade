"""Загрузка ЧАСОВЫХ закрытий MOEX ISS для momentum walk-forward (27 тикеров, ~6 лет).

Дневки дали всего 9 walk-forward окон — недостаточно, чтобы честно судить об эдже.
Часовой interval=60 ISS отдаёт ~24k баров/тикер (×15) → десятки-сотни OOS окон.
Тянем те же тикеры что load_data.py, кэшируем сырьё по тикеру (cache/{T}_1h.csv) и
выровненный inner-join (cache/union_1h.csv). Только чтение ISS, без ключей.

  python -m pairsignal.research.moex.momentum_h1_load            # из кэша / докачать
  python -m pairsignal.research.moex.momentum_h1_load --refresh  # перекачать всё
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pairsignal.research.moex.load_data import ALL_TICKERS  # noqa: E402
from pairsignal.st6.data_feed import read_closes_moex  # noqa: E402

CACHE = Path(__file__).resolve().parent / "cache"
CACHE.mkdir(parents=True, exist_ok=True)


def load_all_h1(days: int = 2200, refresh: bool = False) -> pd.DataFrame:
    """Часовые закрытия всех тикеров корзины, кэш по тикеру. DataFrame[ts_ms -> close]."""
    since = datetime.now(timezone.utc) - timedelta(days=days)
    series: dict[str, pd.Series] = {}
    for t in ALL_TICKERS:
        fp = CACHE / f"{t}_1h.csv"
        if fp.exists() and not refresh:
            s = pd.read_csv(fp, index_col=0)["close"]
            s.index = s.index.astype("int64")
        else:
            try:
                s = read_closes_moex(t, since, "1h")
            except Exception as e:  # noqa: BLE001
                print(f"  {t}: SKIP ({e})")
                continue
            if len(s) > 100:
                s.to_csv(fp, header=["close"])
        if len(s) > 100:
            series[t] = s
            d0 = datetime.fromtimestamp(s.index[0] / 1000, tz=timezone.utc).date()
            d1 = datetime.fromtimestamp(s.index[-1] / 1000, tz=timezone.utc).date()
            print(f"  {t}: {len(s)} bars  [{d0} .. {d1}]")
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series).sort_index()


if __name__ == "__main__":
    refresh = "--refresh" in sys.argv
    df = load_all_h1(refresh=refresh)
    print(f"\nRaw union: {df.shape[0]} hours x {df.shape[1]} tickers")
    aligned = df.dropna()
    print(f"Inner-join (all present): {aligned.shape[0]} hours x {aligned.shape[1]} tickers")
    if len(aligned):
        d0 = datetime.fromtimestamp(aligned.index[0] / 1000, tz=timezone.utc)
        d1 = datetime.fromtimestamp(aligned.index[-1] / 1000, tz=timezone.utc)
        print(f"Range: {d0} .. {d1}")
    df.to_csv(CACHE / "union_1h.csv")
    print(f"Saved union -> {CACHE/'union_1h.csv'}")
