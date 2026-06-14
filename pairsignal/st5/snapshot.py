"""Дневной срез форвард-теста st5 — для недельного мониторинга через systemd-таймер.

Снимает /st5/state по всем инструментам с локального сервера, добавляет компактную
запись в JSON-лог (pairsignal/out/st5_forward_log.json). Отмечает роллировер серии
(смена leg.code) и аномалии (HALTED, зависшая позиция). Запускается раз в день таймером.

  python -m pairsignal.st5.snapshot                 # снять срез сейчас
  python -m pairsignal.st5.snapshot --report        # напечатать недельный отчёт из лога
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

_BASE = Path(__file__).resolve().parent.parent.parent
_LOG = _BASE / "pairsignal" / "out" / "st5_forward_log.json"
_API = "http://localhost:8000"
_MSK = timezone(timedelta(hours=3))


def _get(path: str) -> dict:
    req = urllib.request.Request(_API + path, headers={"User-Agent": "st5-snapshot/1.0"})
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
    """Снять срез по всем инструментам, добавить в лог, вернуть запись."""
    insts = _get("/st5/instruments")["instruments"]
    now = datetime.now(_MSK)
    log = _load()
    prev = log[-1] if log else None

    rows = {}
    for it in insts:
        iid = it["id"]
        s = _get(f"/st5/state?inst={iid}")
        sm = s["summary"]
        leg = s.get("leg") or {}
        rec = {
            "live": s["live"], "leg": leg.get("code"), "leg_expiry": leg.get("expiry"),
            "trades": sm["trades"], "win_rate_pct": sm["win_rate_pct"],
            "net_pnl_rub": sm["net_pnl_rub"], "equity_rub": sm["equity_rub"],
            "halted": s["halted"], "halt_reason": s.get("halt_reason") or None,
            "position": bool(s.get("position")),
            "cur_z": s.get("cur_z"),
        }
        # детект роллировера и аномалий относительно прошлого среза
        flags = []
        if prev and iid in prev.get("inst", {}):
            p = prev["inst"][iid]
            if p.get("leg") and rec["leg"] and p["leg"] != rec["leg"]:
                flags.append(f"роллировер {p['leg']}→{rec['leg']}")
        if rec["halted"]:
            flags.append(f"HALTED: {rec['halt_reason']}")
        rec["flags"] = flags
        rows[iid] = rec

    entry = {"ts": now.strftime("%Y-%m-%d %H:%M МСК"), "day": now.strftime("%Y-%m-%d"),
             "inst": rows}
    # дедуп по дню — один срез на день (последний за день побеждает)
    log = [e for e in log if e.get("day") != entry["day"]]
    log.append(entry)
    _LOG.parent.mkdir(parents=True, exist_ok=True)
    _LOG.write_text(json.dumps(log, ensure_ascii=False))
    return entry


def print_report() -> None:
    """Сводный отчёт по накопленным дневным срезам (динамика день-к-дню)."""
    log = _load()
    if not log:
        print("Лог пуст — срезов ещё не было.")
        return
    insts = sorted({i for e in log for i in e.get("inst", {})})
    print(f"=== st5 форвард-тест: {len(log)} дневных срезов "
          f"({log[0]['day']} … {log[-1]['day']}) ===\n")
    for iid in insts:
        print(f"--- {iid} ---")
        hdr = f"{'дата':<12} {'leg':<7} {'сделок':>6} {'win%':>6} {'net₽':>8} {'equity':>11} {'флаги'}"
        print(hdr)
        print("-" * len(hdr))
        prev_net = None
        for e in log:
            r = e.get("inst", {}).get(iid)
            if not r:
                continue
            dnet = "" if prev_net is None else f" (Δ{r['net_pnl_rub'] - prev_net:+.0f})"
            prev_net = r["net_pnl_rub"]
            flags = "; ".join(r.get("flags", []))
            print(f"{e['day']:<12} {str(r['leg'] or '—'):<7} {r['trades']:>6} "
                  f"{r['win_rate_pct']:>6} {r['net_pnl_rub']:>8.0f}{dnet:<10} "
                  f"{r['equity_rub']:>11,.0f} {flags}")
        print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Дневной срез/отчёт форвард-теста st5")
    ap.add_argument("--report", action="store_true", help="напечатать недельный отчёт")
    args = ap.parse_args()
    if args.report:
        print_report()
        return
    try:
        entry = take_snapshot()
        for iid, r in entry["inst"].items():
            fl = (" | " + "; ".join(r["flags"])) if r["flags"] else ""
            print(f"{entry['ts']} {iid}: сделок {r['trades']} win {r['win_rate_pct']}% "
                  f"net {r['net_pnl_rub']:+.0f}₽ equity {r['equity_rub']:,.0f}{fl}")
    except Exception as e:  # noqa: BLE001
        print(f"срез не снят: {e}")


if __name__ == "__main__":
    main()
