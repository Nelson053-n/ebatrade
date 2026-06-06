# План: реальные данные (gateio/mexc) + адаптивный vol-band для cross_pct

## Context

Режим cross_pct (кросс-биржевой арбитраж SUI) реализован на синтетике. Теперь подключаем
**реальные данные**. Две находки при проверке сети с боевого сервера:

1. **BitMEX prod и OKX недоступны** (firewall/гео, curl→000). Доступны и отдают
   `SUI/USDT:USDT` перп: bybit, binance, gateio, bitget, mexc. Пользователь выбрал
   **gateio + mexc** (самый широкий реальный спред сейчас ~0.085%).
2. **Реальный спред 0.01–0.09%** — фиксированные ±3% (band_pct=0.03) НИКОГДА не пробьются.
   Пользователь выбрал **адаптивный band от волатильности**: `band = bb_k·σ(спреда)`. Это
   превращает cross в классические Боллинджеры на линейном спреде — полосы сами подстраиваются.

### Развилка нормировки — Вариант A (принят)
`band = bb_k·rolling_std(spread)`, `entry_z=1.0` сохраняется → **|z|=1 на полосе**,
`std=band/bb_k=σ(спреда)`. Главное преимущество: **strategy.py не меняется вообще**
(выход z→0, стоп |z|≥8, вход entry_z=1.0 работают как есть). Меняется только источник
полуширины band: `band_pct·price` → `bb_k·σ`. `std0`/стоп при этом осмысленны (настоящая σ).

## Изменения по файлам

### 1. `config.py` (StrategyConfig)
- Дефолты бирж: `exchange_a="gateio"`, `exchange_b="mexc"`. `symbol_cross="SUI/USDT:USDT"` (есть на обеих).
- Новое поле `band_mode: Literal["pct","vol"] = "vol"` рядом с `band_pct`.
- `vol_window` — **переиспользуем `sma_period`** (одно окно сглаживания, прогрев совпадает).
- Docstring cross-блока обновить под vol.

### 2. `indicators.py` ветка cross_pct (стр.37-51)
Dispatch по `cfg.band_mode`:
```
spread = price_a − price_b
mid = rolling_mean(spread, sma_period)
if band_mode == "vol":
    sd = rolling_std(spread, sma_period, ddof=0)
    band = (bb_k * sd).clip(lower=1e-9); std = sd
else:  # pct (fallback, как сейчас)
    band = (band_pct * price_a).clip(lower=1e-9); std = band / bb_k
upper/lower = mid ± band
z = (spread − mid) / band          # |z|=1 на полосе (оба режима)
width_pct = 100.0; return          # ранний return
```
`warmup_limit` не меняем (vol тоже валиден после sma_period баров; rolling std → NaN на прогреве).
Деление на ноль в штиль гасит `clip(1e-9)` + `_valid` (std>0 отсекает).

### 3. `strategy.py` — без изменений
Вариант A сохраняет |z|=1 на полосе и entry_z=1.0 → вход/выход/стоп cross работают как есть.

### 4. `api.py`
- **Импорт** `read_ohlcv_cross` (data_feed уже содержит её: 2 биржи, inner-join по ts).
- **Хелпер** `_reader_for(cfg)` → `read_ohlcv_cross if spread_mode=="cross_pct" else read_ohlcv_ccxt`.
  Применить в **`_poller`** (стр.278) **И в `/analyze`** (стр.421 — иначе аналитика cross-слота
  читает мусор по symbol_a/b). to_thread-обёртка над всей read_ohlcv_cross — loop не блокируется.
- **PRESETS["cross_pct"]**: добавить `"band_mode":"vol"`; `entry_z=1.0` оставить; label
  «Кросс-биржа SUI (gateio/mexc)», desc «адаптивные полосы bb_k·σ(спреда), SMA(200), выход к SMA».
- **`_apply_preset`**: добавить `s.band_mode = p.get("band_mode","pct")`.
- **`/state`**: добавить `band_mode`, `bb_k`, `sma_period`, `exchange_a`, `exchange_b`, `symbol_cross`.
- **`warmup_limit`** для cross: запас на рассинхрон бирж — `int(sma_period*1.5)+80` (см. РИСК 2).

### 5. `dashboard.html` (подпись графика, стр.740-744)
Имена бирж из `s.exchange_a/b` (fallback A/B). При vol: «P(gateio) − P(mexc) · SMA(200) · полосы N·σ»;
при pct: старая «±band_pct%». Опционально (стр.732-734) — при cross показывать `exA+'+'+exB`
вместо «MEXC».

### НЕ трогаем
- `data_feed.read_ohlcv_cross` — готова. `read_ohlcv_ccxt` (st1) — не трогаем.
- `virtual_exchange.py`, `models.py`, `engine.py` — без изменений (Вариант A не требует).

## РИСКИ
1. **Одна нога упала** → `read_ohlcv_cross` бросает целиком (нет частичного результата), feed
   встаёт. `_poller` ловит Exception → `last_rec.error`, не падает. Per-leg try/except — отдельно.
2. **Рассинхрон ts двух бирж** → мало общих баров после inner-join, прогрев sma_period(200) не
   набирается. Митигация: `warmup_limit` cross тянет `sma_period*1.5+80`. После первого live —
   проверить `len(df)` после join ≥ sma_period.
3. **Спред уезжает** (расхождение фандинга) → mean-reversion слабее синтетики, затяжные позиции
   до тайм-стопа/стопа. Профит не гарантирован.
4. **Per-exchange fees/funding не учтены** — paper-комиссия единая 0.06%, реальный кросс =
   комиссии gateio+mexc + funding обеих ног. Paper оптимистичен — задокументировать.
5. **Полоса схлопывается в штиль** (sd→0 → узкая полоса → ложные входы). Гасит clip+_valid.

## Verification

1. **Тесты** (`tests/`): `_cross_cfg` новый дефолт `band_mode="vol"`.
   - `test_cross_pct_indicators` РАЗВЕСТИ: vol-кейс (`upper−mid==bb_k·rolling_std`, `std==σ`,
     |z|≈1) + pct-кейс (явный `band_mode="pct"`, старая проверка).
   - `test_cross_pct_synthetic_has_trades`: прогнать в vol-дефолте → `trades>0`, все `reason=="exit"`
     (если 0 — поднять амплитуду синтетики; страж согласованности band/генератора).
   - вход short/long, выход к SMA — не зависят от band_mode (|z|=1 на полосе), проходят.
   - `pytest -q` зелёный, `ruff check pairsignal` чист.
2. **Локально:** `uvicorn ... :8099`; `POST /live/start?slot=2` → дождаться прогрева, `GET /state?slot=2`
   показывает реальный спред gateio−mexc, полосы bb_k·σ, сделки появляются. Проверить
   `len(df)` после join ≥ sma_period (нет вечного прогрева). st1 live (MEXC, log) не затронут.
3. **Изоляция:** `/analyze?slot=2` использует read_ohlcv_cross (реальные две биржи), не MEXC-пару.
4. **Браузер:** st2 подпись «P(gateio) − P(mexc) · SMA(200) · N·σ», график реального спреда.
5. **Деплой:** `systemctl --user restart ebatrade`; на проде включить live для st2
   (`/live/start?slot=2`); проверить https://trade.bananagen.ru. `git grep` на 192.168.* перед push.
