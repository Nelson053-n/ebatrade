"""Ежедневная сводка forward-test momentum в Telegram.

Читает состояние paper-портфеля (momentum_fwd.json) + статус systemd-сервиса и шлёт
компактное сообщение в Telegram через Bot API. Секреты — из переменных окружения
(TG_BOT_TOKEN, TG_CHAT_ID), не хардкодятся. Запускается systemd-таймером раз в день.

  TG_BOT_TOKEN=... TG_CHAT_ID=... python -m pairsignal.tg_summary
"""
from __future__ import annotations

import os
import subprocess
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .momentum import START_EQUITY, load_state

STATE = Path(__file__).resolve().parent / "out" / "momentum_fwd.json"
SERVICE = "momentum-fwd.service"


def _service_active() -> bool:
    try:
        r = subprocess.run(["systemctl", "--user", "is-active", SERVICE],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() == "active"
    except Exception:  # noqa: BLE001
        return False


def build_summary() -> str:
    """Собрать текст сводки (Markdown для Telegram)."""
    active = _service_active()
    head = "📊 *Momentum forward-test*"
    loaded = load_state(STATE)
    if loaded is None:
        return f"{head}\n⚠️ Нет состояния (файл не найден). Сервис active={active}."

    port, meta = loaded
    ret = (port.equity / START_EQUITY - 1.0) * 100
    longs = ", ".join(s.split("/")[0] for s, w in port.weights.items() if w > 0) or "—"
    shorts = ", ".join(s.split("/")[0] for s, w in port.weights.items() if w < 0) or "—"
    last = ""
    if port.rebalances:
        ts = port.rebalances[-1]["ts"]
        last = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    status_line = "🟢 сервис active" if active else "🔴 *СЕРВИС НЕ АКТИВЕН* — `systemctl --user restart momentum-fwd`"
    sign = "+" if ret >= 0 else ""
    return (
        f"{head}\n"
        f"{status_line}\n"
        f"💰 equity *${port.equity:.2f}* ({sign}{ret:.2f}%)\n"
        f"🔁 ребалансов: {len(port.rebalances)} · lookback={meta.get('lookback')} holding={meta.get('holding')}\n"
        f"🟩 ЛОНГ: {longs}\n"
        f"🟥 ШОРТ: {shorts}\n"
        f"🕒 последний ребаланс: {last or '—'}"
    )


def send_telegram(text: str) -> None:
    token, chat = os.environ.get("TG_BOT_TOKEN"), os.environ.get("TG_CHAT_ID")
    if not token or not chat:
        raise SystemExit("TG_BOT_TOKEN / TG_CHAT_ID не заданы в окружении")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": chat, "text": text, "parse_mode": "Markdown",
        "disable_web_page_preview": "true",
    }).encode()
    with urllib.request.urlopen(url, data=data, timeout=20) as resp:  # noqa: S310
        if resp.status != 200:
            raise SystemExit(f"Telegram API вернул {resp.status}")


def main() -> None:
    send_telegram(build_summary())


if __name__ == "__main__":
    main()
