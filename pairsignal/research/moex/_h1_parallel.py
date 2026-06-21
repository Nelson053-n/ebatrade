"""Параллельная дозагрузка недостающих часовых тикеров (ISS, только чтение).

Однократный помощник: тянет в N потоков те тикеры из ALL_TICKERS, для которых ещё
нет cache/{T}_1h.csv. Каждый готовый ряд сразу пишет на диск. Идемпотентно.
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pairsignal.research.moex.load_data import ALL_TICKERS  # noqa: E402
from pairsignal.st6.data_feed import read_closes_moex  # noqa: E402

CACHE = Path(__file__).resolve().parent / "cache"
SINCE = datetime.now(timezone.utc) - timedelta(days=2200)


def fetch(t: str) -> tuple[str, int, str]:
    fp = CACHE / f"{t}_1h.csv"
    if fp.exists():
        return t, -1, "cached"
    try:
        s = read_closes_moex(t, SINCE, "1h")
    except Exception as e:  # noqa: BLE001
        return t, 0, f"ERR {e}"
    if len(s) > 100:
        s.to_csv(fp, header=["close"])
        return t, len(s), "saved"
    return t, len(s), "too-short"


if __name__ == "__main__":
    missing = [t for t in ALL_TICKERS if not (CACHE / f"{t}_1h.csv").exists()]
    print(f"missing {len(missing)}: {missing}", flush=True)
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(fetch, t): t for t in missing}
        for f in as_completed(futs):
            t, n, st = f.result()
            print(f"  {t}: {n} {st}", flush=True)
    print("DONE", flush=True)
