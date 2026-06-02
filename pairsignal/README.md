# pairsignal — phase 1: аналитик-советник + виртуальная биржа

Система **читает** котировки пары перпетуалов, считает спред и индикаторы, выдаёт
**рекомендации** на вход, ждёт решения оператора (human-in-the-loop) и ведёт сделки
на **виртуальной бирже** (paper). С реальной биржей по торгам пока не работает —
только чтение данных и симуляция исполнения.

## Что считается

- Спред: `ln(A) − β·ln(B)` (динамическая β по rolling-OLS) или `A/B` (режим `ratio`).
- Полосы Боллинджера поверх спреда + **z-score** + ширина канала (анти-флэт фильтр).
- Сигнал входа: спред у нижней полосы (`z ≤ −entry_z`) → лонг A / шорт B; симметрично сверху.
- Выход (авто): возврат к средней (`|z| ≤ exit_z`).
- Стоп (авто): `|z| ≥ stop_z` или тайм-стоп по числу баров.
- Сайзинг: доллар-нейтрально, нотационал = `risk_pct` % от баланса, распределение ног по β.
- Комиссии и проскальзывание учитываются в P&L.

Все расчёты — по **закрытым** свечам (no repaint).

## Структура

```
pairsignal/
  config.py            # pydantic-конфиг (пары, пороги, комиссии, сайзинг)
  models.py            # dataclass'ы и enum'ы
  indicators.py        # β, спред, BB, z, ширина
  data_feed.py         # CCXT (чтение) + синтетический генератор
  strategy.py          # SignalEngine — чистый конечный автомат сигналов
  virtual_exchange.py  # paper-исполнение: филлы, комиссии, P&L, позиции
  engine.py            # оркестратор + human-in-the-loop (approve/reject)
  main.py              # CLI-раннер (синтетика / live)
  api.py               # FastAPI backend под панель
```

## Запуск

```bash
pip install -r requirements.txt

# офлайн-демо без сети (авто-подтверждение входов)
python -m pairsignal.main --synthetic --auto

# то же, но с ручным подтверждением каждого входа
python -m pairsignal.main --synthetic

# реальные котировки (нужны сеть и доступ к бирже из config.data_exchange)
python -m pairsignal.main --live

# backend для панели
uvicorn pairsignal.api:app --reload
#  GET  /state            — рекомендация, позиция, сводка
#  POST /approve|/reject  — решение оператора по входу
#  POST /live/start|/stop — фоновый опрос котировок
#  POST /replay/synthetic — прогнать синтетику целиком
```

## Дальше (вне phase 1)

- Бэктест-отчёт (Sharpe, max DD, кривая капитала) и подбор параметров.
- Алерты в Telegram на новую рекомендацию.
- Фронт (Next.js + Lightweight Charts) поверх `/state` и WebSocket.
- Только после устойчивого paper-результата — реальное исполнение на бирже
  с открытым фьючерсным API (Bybit/OKX/Bitget/Binance; MEXC для розницы закрыта).
```
