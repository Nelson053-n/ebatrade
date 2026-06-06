# План: вкладка st3 (momentum) в веб-панели

## Context

Momentum — наш лучший результат (единственный положительный OOS). Сейчас он живёт только в
CLI (`momentum.py`: backtest, walk-forward, live paper-портфель, compare_strategies). Оператор
хочет видеть его в веб-панели рядом со st1/st2: live forward-test, walk-forward бэктест,
сравнение стратегий и статус тестов — с графиками.

Панель (`dashboard.html` + `api.py`) построена на вкладках `?slot=1|2` поверх SlotState/Engine
с одним canvas-графиком спреда. Momentum — **портфельная** стратегия, не парная: в SlotState её
пихать нельзя (это ломает модель). Поэтому st3 — **отдельная панель-блок** (read-only виджет),
переключаемая третьей вкладкой, со своими эндпоинтами, читающими файлы/функции `momentum.py`.

**Решения оператора:** на st3 показать всё четыре блока — live forward-test, walk-forward бэктест,
сравнение стратегий, статус тестов; графики — тот же canvas-подход (без внешних библиотек).

## Что переиспользуем
- `momentum.load_state` (`pairsignal/momentum.py`) — чтение live-портфеля из momentum_fwd.json.
- `momentum.walk_forward`, `fit_params`, `_BARS_PER_YEAR` — для бэктеста по запросу.
- `pairs_coint.load_universe`, `scan_year.top_symbols` — загрузка данных для бэктеста.
- `compare_strategies.eq_momentum/eq_pairs/eq_buy_hold` — кривые сравнения.
- canvas-рендер из `dashboard.html` (draw, аггрегация) — паттерн для новой equity-функции.
- `_clean` (api.py) — NaN→None в JSON-ответах.

## Изменения

### 1. API — новые эндпоинты `pairsignal/api.py` (момент НЕ через `_slot`/SlotState)
Отдельный роутер momentum, read-only + запуск бэктеста:
- `GET /momentum/state` — читает `out/momentum_fwd.json` через `load_state`. Отдаёт: equity,
  return_pct, веса (longs/shorts), журнал ребалансов (ts, equity, turnover), meta (lookback/holding/k),
  «жив ли live-процесс» (проверка по существованию/свежести файла last_ts). Если файла нет — `{running:false}`.
- `GET /momentum/backtest?top&days&timeframe&k&fee&slippage` — на лету: `top_symbols`→`load_universe`
  →`walk_forward`; возвращает строки окон (return/Sharpe/maxDD/turnover) + точки склеенной OOS
  equity-кривой. Тяжёлый (загрузка) — кэшировать результат в памяти по ключу параметров.
- `GET /momentum/compare?top&days&timeframe` — кривые Momentum/Pairs/BuyHold (точки equity на общей
  оси) через функции compare_strategies. Тоже кэш.
- `GET /momentum/tests` — статус юнит-тестов momentum: запускает `pytest -q tests/test_momentum*.py`
  через subprocess, парсит passed/failed, возвращает счётчики + имена тестов. (Read-only по сути.)

Тяжёлые backtest/compare гонять в `asyncio.to_thread` (CCXT блокирующий, как в `_poller`).

### 2. Frontend — третья вкладка `dashboard.html`
- **Кнопка вкладки**: добавить `<button class="strat-btn" data-slot="3">st3</button>` в `#stratSwitch`
  (строка ~243). Расширить валидацию SLOT на 3 (строка ~464) и `switchSlot`.
- **Переключение макета**: st1/st2 показывают существующий парный layout; при SLOT===3 —
  **прятать** парный layout и **показывать** новый контейнер `#st3panel` (display toggle в `markSlot`/
  `switchSlot`). Запросы st3 идут на `/momentum/*` (без `?slot=`), а не на `/state?slot=`.
- **Блоки `#st3panel`** (4 секции, в стиле существующих карточек):
  1. **Live forward-test**: сводка (equity, return%, lookback/holding), таблица текущих ЛОНГ/ШОРТ,
     canvas equity-кривая по `rebalances[].equity`, журнал последних ребалансов. Поллинг `/momentum/state`.
  2. **Walk-forward бэктест**: форма (top/days/k/fee), кнопка «Прогнать» → `/momentum/backtest`,
     таблица окон + canvas склеенной OOS-кривой. Спиннер на время загрузки.
  3. **Сравнение стратегий**: кнопка → `/momentum/compare`, canvas с 3 линиями (Momentum/Pairs/
     BuyHold), легенда + итоговые return%.
  4. **Тесты**: кнопка → `/momentum/tests`, бейдж «N passed / M failed» + список.
- **Canvas equity-рендер**: новая JS-функция `drawEquity(canvas, series)` — простая линия(и) с
  автошкалой, в стиле `draw()` (те же CSS-переменные цветов). Без библиотек.

### 3. Тесты `tests/test_momentum_api.py`
FastAPI `TestClient` (паттерн из отчёта разведки):
- `/momentum/state` без файла → `{running:false}`, со свежесозданным файлом (фикстура tmp) → корректный парс.
- `/momentum/tests` → 200, содержит счётчики.
- `/momentum/backtest` с маленьким top/days на коротком окне (или мок load_universe) → структура ответа.
  (Если сеть недоступна в CI — пометить как skip/мок; не вешать тест на биржу.)

## Подводные камни
- **Не ломать st1/st2.** st3 — отдельная ветка в UI и отдельные эндпоинты; парный layout не трогать,
  только прятать при SLOT===3. `_slot()` остаётся 1|2 — momentum не использует его.
- **momentum_fwd.json может отсутствовать/устареть.** `/momentum/state` должен отдавать `running:false`
  без падения; «жив» определять по свежести last_ts (напр. < 2·poll). Не врать, что процесс жив.
- **Тяжёлые эндпоинты блокируют.** backtest/compare грузят историю (секунды-минуты) — `to_thread`
  + in-memory кэш по параметрам, иначе каждый клик = новая загрузка и риск таймаута.
- **pytest из subprocess**: фиксировать рабочую директорию и `.venv` python; парсить хвост вывода;
  таймаут. Не запускать весь набор — только `tests/test_momentum*.py`.
- **CSV прошлых прогонов** (out/momentum_*.csv) — опционально показать как «история прогонов», но
  не обязательно для MVP; live-state и on-demand backtest важнее.
- Путь файлов momentum: `Path(__file__).parent / "out" / "momentum_fwd.json"` (рядом с momentum.py).

## Verification
1. `uvicorn pairsignal.api:app --reload`, открыть панель, переключиться на вкладку **st3**.
2. Live-блок: forward-test уже пишет `out/momentum_fwd.json` — увидеть equity-кривую, позиции,
   ребалансы; если процесс остановлен — корректный статус «не запущен».
3. Walk-forward: нажать «Прогнать» (top=12, days=90) → таблица окон + кривая за ~минуту.
4. Сравнение: кнопка → 3 линии, легенда, return% (Momentum выше Pairs/BuyHold).
5. Тесты: кнопка → «N passed» (сверить с `pytest -q tests/test_momentum*.py`).
6. Проверить st1/st2 не сломаны: переключение вкладок, графики спреда, approve/reject работают.
7. `pytest -q` (прежние 32 + новые api-тесты зелёные), `ruff check pairsignal`.
