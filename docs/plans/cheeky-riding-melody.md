# План: коннектор T-Bank sandbox в st4 + ввод токена через UI

## Context

st4 (арбитраж спреда фьючерсов SBRF/SBPR) сейчас исполняет ордера только «на бумаге»
(`OrderExecutor` в `execution.py`). Уже написан и проверён REST-клиент песочницы T-Bank
(`pairsignal/st4/tbank_sandbox.py`) — полный sandbox-прогон входа/выхода пары прошёл
(SRM6/SPM6, реальные ордера на виртуальном счёте). Цель этого шага (начало Phase 2 из ТЗ,
§14.3): дать движку st4 ставить ордера в песочницу T-Bank **по своим сигналам** автоматически,
и добавить **ввод API-токена через UI**, чтобы режим переключался со страницы.

**Решения оператора (зафиксированы):**
1. Токен — **только в памяти процесса** (`os.environ["TBANK_TOKEN"]`), не на диске.
2. Выход из позиции в sandbox — **реальный обратный ордер** (счёт и движок синхронны; никакого
   рассинхрона «вход реальный / выход бумажный»).
3. Sandbox-исполнение разрешено **только в live-режиме** (MOEX ISS). На синтетике — запрет
   (бессмысленно слать рыночные ордера по выдуманным барам).

**Жёсткое ограничение (CLAUDE.md):** только `SandboxService.*`, никаких боевых `OrdersService`.
`tbank_sandbox.py` физически не содержит боевых методов — отправить реальный ордер нельзя.

## Принцип минимального вмешательства

Engine работает через узкий интерфейс `executor.execute_pair(...) -> PairFillResult` и читает
только `r.ok / r.fill_ord / r.fill_pref` + поля `Fill`. Новый sandbox-исполнитель реализует ту
же сигнатуру + добавляет `close_pair(...)` для реального выхода. Сигнатуры `execute_pair`,
`PairFillResult`, `Fill` — НЕ меняем.

---

## Изменения по файлам

### 1. `pairsignal/st4/config.py` — раздел ConnectorConfig
Новый класс + поле в `St4Config` (рядом с другими разделами):
```python
class ConnectorConfig(BaseModel):
    mode: Literal["paper", "tbank_sandbox"] = "paper"
    account_id: str = ""          # переиспользуемый sandbox-счёт; пусто → открыть новый
    payin_rub: int = 200_000      # пополнение под ГО при старте
    account_name: str = "st4-spread-sandbox"
```
`connector: ConnectorConfig = Field(default_factory=ConnectorConfig)` в St4Config.
**Токена в конфиге НЕТ** (секрет не сериализуется в session_state_4.json). `account_id`/`payin_rub`
несекретны — переживают рестарт через паттерн hasattr/setattr в `load_session`.

### 2. `pairsignal/st4/execution.py` — протокол исполнителя
`OrderExecutor` (paper) оставить как есть. Добавить `Protocol PairExecutor` для типизации с
методами `execute_pair(...)` и `close_pair(...)`. Добавить `close_pair` в `OrderExecutor`
(paper-вариант: возвращает цены бара — те, что передали; ниже в engine это и есть текущее
поведение). `UnwindError`, `PairFillResult`, `Fill`, `leg_pnl_rub`, `pair_fee_rub` — без изменений.

Новый dataclass для результата закрытия (минимальный):
```python
@dataclass
class PairCloseResult:
    exit_ord: float      # фактическая цена выхода ноги SBRF (пункты)
    exit_pref: float     # фактическая цена выхода ноги SBPR
    slippage_ticks: float = 0.0
```
`OrderExecutor.close_pair(pos, ref_ord, ref_pref) -> PairCloseResult` — paper: вернуть
`PairCloseResult(ref_ord, ref_pref)` (= цены бара, текущее поведение).

### 3. НОВЫЙ `pairsignal/st4/tinkoff_executor.py` — `TinkoffSandboxExecutor`
Реализует `execute_pair` (та же сигнатура/типы) + `close_pair`. Ходит ТОЛЬКО через `tbank_sandbox`.

Конструктор (инъекция `sb`-модуля для моков):
```python
def __init__(self, exec_cfg, conn_cfg, spec_ord, spec_pref, sb=tbank_sandbox):
    ...
    self._ensure_started()   # резолв инструментов + счёт + pay_in
```
Методы:
- `_resolve_instruments()` — `sb.find_future(self.spec[role].code)` по обеим ногам (тикер FORTS =
  тикер T-Bank), кэш в `self._inst`. uid для ордера = `sb._uid(it)`.
- `_ensure_started()` — переиспользовать `conn.account_id`/счёт по `account_name` из
  `list_accounts()` (OPEN), иначе `open_account`; записать id обратно в `conn.account_id`;
  `pay_in(payin_rub)`. `TBankError` → пробросить (reset_engine обернёт в try → откат в paper).
- `_post_leg(role, side, lots, ref) -> Fill | None` — `post_order(..., str(uuid.uuid4()),
  "ORDER_TYPE_MARKET")`; orderId ОБЯЗАН быть UUID. Проверять `executionReportStatus ==
  EXECUTION_REPORT_STATUS_FILL` и `avg = sb._q_to_float(executedOrderPrice) > 0`, иначе None.
  `slippage_ticks = (avg-ref)/tick·знак`.
- `execute_pair(...)` — структура как `OrderExecutor.execute_pair` (порядок first_leg_to_fill,
  unwind обратным ордером, `UnwindError` если unwind не удался, раскладка fill_ord/fill_pref).
- `close_pair(pos, ref_ord, ref_pref) -> PairCloseResult` — **реальный выход**: обратные
  market-ордера по обеим ногам (SBRF: противоположно `pos.leg_ord.side`; SBPR аналогично),
  вернуть фактические avg как `exit_ord/exit_pref`. Если нога не закрылась — ретраи, при провале
  → `UnwindError` (engine → HALTED; голую ногу в sandbox не оставляем).

### 4. `pairsignal/st4/engine.py` — фабрика исполнителя + реальный выход
**Правка 1 (конструктор, ~стр. 52):** выбор исполнителя по `cfg.connector.mode`:
```python
if cfg.connector.mode == "tbank_sandbox":
    from .tinkoff_executor import TinkoffSandboxExecutor   # локальный импорт
    self.executor = TinkoffSandboxExecutor(cfg.execution, cfg.connector, spec_ord, spec_pref)
else:
    self.executor = OrderExecutor(cfg.execution, cfg.paper, spec_ord, spec_pref)
```
**Правка 2 (`_close_position`, стр. 236-264):** вместо жёстких `exit_ord=bar.close_ord` —
спросить исполнителя:
```python
cr = self.executor.close_pair(p, bar.close_ord, bar.close_pref)
exit_ord, exit_pref = cr.exit_ord, cr.exit_pref
```
Paper-executor вернёт те же цены бара (поведение не меняется → нет регрессии). Sandbox-executor
вернёт реальный филл обратного ордера. P&L по тикам считается дальше как сейчас.

### 5. `pairsignal/st4/service.py` — гейт live + snapshot + откат
- `reset_engine` обернуть создание `TradingEngine` в try: при ошибке sandbox (нет токена/сети) →
  `cfg.connector.mode="paper"`, событие в last_event, пересоздать paper-движок.
- **Гейт live (решение 3):** при старте плеера/синтетики, если `connector.mode=="tbank_sandbox"`
  — НЕ слать в sandbox. Проще: в `run_player` режим коннектора не активен (sandbox только в
  `run_live`). Реализация: при `mode==tbank_sandbox` и запуске player — предупредить и временно
  трактовать как paper для синтетики (или запретить старт player). Минимально: в `/st4/player/start`
  не активировать sandbox; sandbox-движок создаётся только в live-пути.
- `snapshot()` добавить: `connector_mode`, `token_set` (bool по env, без самого токена),
  `connector_account`. `import os` в service.py.
- `load_session`/`save_session` — без изменений (hasattr/setattr восстановит `connector`).

### 6. `pairsignal/api.py` — POST /st4/connector
Рядом с другими /st4/* (~стр. 1060). `import os` (проверить наличие):
```python
@app.post("/st4/connector")
def st4_connector(payload: dict):
    mode = payload.get("mode")
    if mode not in ("paper", "tbank_sandbox"):
        raise HTTPException(400, "mode: paper | tbank_sandbox")
    token = (payload.get("token") or "").strip()
    if token:
        os.environ["TBANK_TOKEN"] = token            # секрет только в env процесса
    if mode == "tbank_sandbox" and not os.environ.get("TBANK_TOKEN", "").strip():
        raise HTTPException(400, "для sandbox нужен токен")
    if "payin_rub" in payload:
        ST4.cfg.connector.payin_rub = int(payload["payin_rub"])
    ST4.cfg.connector.mode = mode
    was_live, was_player = ST4.state["live"], ST4.state["player"]
    ST4.state["live"] = ST4.state["player"] = False
    ST4.reset_engine(real=(ST4.state["data_source"] == "live"))
    return {"ok": True, "connector_mode": ST4.cfg.connector.mode,
            "token_set": bool(os.environ.get("TBANK_TOKEN","").strip()),
            "fell_back": ST4.cfg.connector.mode != mode}
```
Ответ НЕ содержит токен.

### 7. `dashboard.html` — UI-блок «Коннектор» в правой колонке #st4panel
Новый `.panel` после блока «Управление» (~стр. 414):
- `<select id="s4-conn-mode">` (paper / T-Bank sandbox)
- `<input id="s4-conn-token" type="password" autocomplete="off">` (для вставки токена)
- кнопка `#s4-btn-conn` «применить», статус `#s4-conn-msg`, бейдж `#s4-conn-badge`
- подпись: «токен только в памяти процесса; sandbox активен только в live; реальные биржевые
  ордера не отправляются».

JS в `s4Init()` (блок `if(!_s4Started)`): обработчик `#s4-btn-conn` → `s4api('/connector','POST',
{mode, token?})`, очистить поле токена после применения, показать `fell_back`/`token_set`.
В `s4Render(s)` — выставить `s4-conn-mode.value`, бейдж с `token_set`, placeholder токена.
Токен после применения в DOM не держим (`value=''`).

---

## Порядок реализации
1. config.py → 2. execution.py (Protocol + close_pair + PairCloseResult) → 3. tinkoff_executor.py
→ 4. engine.py (фабрика + close_pair в _close_position) → 5. service.py → 6. api.py → 7. dashboard.html
→ 8. tests. Каждый шаг: `pytest tests/test_st4.py -q`; paper-путь не должен регрессировать.

## Верификация
**Юнит (tests/test_st4.py, мок `sb`, без сети):**
- `execute_pair` ok → r.ok, корректные side/avg_price, два post_order, orderId — валидные UUID,
  счёт открыт + pay_in.
- unwind при отказе второй ноги → r.unwound, обратный ордер поставлен.
- unwind не удался → `UnwindError`.
- `close_pair` → два обратных ордера, exit-цены из филла.
- переиспользование счёта (list_accounts с OPEN → open_account не зван).
- кэш инструментов (find_future по разу на ногу).
- engine mode="tbank_sandbox" → executor — TinkoffSandboxExecutor; mode="paper" → OrderExecutor
  (регресс-гард). Полный вход+выход через engine.on_candles в sandbox-режиме (с моком) даёт
  Trade с exit-ценами из филла.

**API:** POST /st4/connector mode=paper → ok; mode=sandbox без токена → 400; sandbox + фейк-токен
при недоступной сети → откат в paper, snapshot token_set=true но токен НЕ в ответе/файле.
GET /st4/state содержит connector_mode/token_set.

**Интеграция (ручная, реальный токен, live):** uvicorn → вкладка st4 → блок «Коннектор»:
T-Bank sandbox + токен → «применить» → бейдж «токен ✓». Запустить **live** → на сигнале входа
парный market-ордер в sandbox (позиция в snapshot, avg из филла), на выходе — реальный обратный
ордер (sandbox-счёт синхронен). `grep` по session_state_4.json — токена нет. ruff чист.

## Известные упрощения (отметить в README/комментарии)
- Reconciliation (`engine.reconcile`) в sandbox мог бы сверяться с `positions(account_id)`;
  сейчас остаётся paper-сверкой со снимком. Возможное расширение, вне scope.
- На синтетике sandbox не активен (по решению 3) — это by design, не баг.
