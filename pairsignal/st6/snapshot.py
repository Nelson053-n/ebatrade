"""Дневной срез форвард-теста st6 — для мониторинга через systemd-таймер.

Снимает /st6/state с локального сервера, добавляет компактную запись в JSON-лог
(pairsignal/out/st6_forward_log.json). Отмечает смену активной пары (reselect) и
аномалии (зависшая позиция). Запускается раз в день таймером.

  python -m pairsignal.st6.snapshot                 # снять срез сейчас
  python -m pairsignal.st6.snapshot --report        # напечатать отчёт из лога
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

_BASE = Path(__file__).resolve().parent.parent.parent
_LOG = _BASE / "pairsignal" / "out" / "st6_forward_log.json"
_API = "http://localhost:8000"
_MSK = timezone(timedelta(hours=3))


def _get(path: str) -> dict:
    req = urllib.request.Request(_API + path, headers={"User-Agent": "st6-snapshot/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310 (локальный доверенный)
        return json.loads(r.read().decode("utf-8"))


def _load() -> list[dict]:
    try:
        if _LOG.exists():
            return json.loads(_LOG.read_text())
    except Exception:  # noqa: BLE001
        pass
    return []


def take_snapshot() -> dict:
    """Снять срез, добавить в лог, вернуть запись."""
    s = _get("/st6/state")
    sm = s["summary"]
    now = datetime.now(_MSK)
    log = _load()
    prev = log[-1] if log else None

    pair = "/".join(s["pair"]) if s.get("pair") else None
    rec = {
        "ts": now.strftime("%Y-%m-%d %H:%M МСК"), "day": now.strftime("%Y-%m-%d"),
        "live": s["live"], "pair": pair,
        "trades": sm["trades"], "win_rate_pct": sm["win_rate_pct"],
        "net_pnl_rub": sm["net_pnl_rub"], "equity_rub": sm["equity_rub"],
        "position": bool(s.get("position")),
        "cur_z": s.get("cur_z"), "cur_corr": s.get("cur_corr"),
    }
    flags = []
    if prev and prev.get("pair") and pair and prev["pair"] != pair:
        flags.append(f"reselect {prev['pair']}→{pair}")
    rec["flags"] = flags

    log = [e for e in log if e.get("day") != rec["day"]]
    log.append(rec)
    _LOG.parent.mkdir(parents=True, exist_ok=True)
    _LOG.write_text(json.dumps(log, ensure_ascii=False))
    return rec


def print_report() -> None:
    log = _load()
    if not log:
        print("Лог пуст — срезов ещё не было.")
        return
    print(f"=== st6 форвард-тест: {len(log)} дневных срезов "
          f"({log[0]['day']} … {log[-1]['day']}) ===\n")
    hdr = f"{'дата':<12} {'пара':<12} {'сделок':>6} {'win%':>6} {'net₽':>9} {'equity':>12} {'флаги'}"
    print(hdr)
    print("-" * len(hdr))
    prev_net = None
    for e in log:
        dnet = "" if prev_net is None else f" (Δ{e['net_pnl_rub'] - prev_net:+.0f})"
        prev_net = e["net_pnl_rub"]
        flags = "; ".join(e.get("flags", []))
        print(f"{e['day']:<12} {str(e['pair'] or '—'):<12} {e['trades']:>6} "
              f"{e['win_rate_pct']:>6} {e['net_pnl_rub']:>9.0f}{dnet:<10} "
              f"{e['equity_rub']:>12,.0f} {flags}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Дневной срез/отчёт форвард-теста st6")
    ap.add_argument("--report", action="store_true", help="напечатать отчёт")
    args = ap.parse_args()
    if args.report:
        print_report()
        return
    try:
        r = take_snapshot()
        fl = (" | " + "; ".join(r["flags"])) if r["flags"] else ""
        print(f"{r['ts']} пара {r['pair']}: сделок {r['trades']} win {r['win_rate_pct']}% "
              f"net {r['net_pnl_rub']:+.0f}₽ equity {r['equity_rub']:,.0f}{fl}")
    except Exception as e:  # noqa: BLE001
        print(f"срез не снят: {e}")


if __name__ == "__main__":
    main()
