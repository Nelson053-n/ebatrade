# FORTS intraday research (st5 successor)

Исследование алгоритма для одиночных FORTS-фьючерсов (SBER=SR*, GAZP=GZ*, LKOH=LK*),
10m, MOEX ISS. Цель: положительный честный out-of-sample на ОБОИХ инструментах после
реальных издержек. **Боевые файлы st5 не тронуты** — только скрипты в этой папке.

## Главный вывод
Внутридневной **momentum НЕ работает** на этих фьючерсах (Sharpe −2…−7 на обоих) —
подтверждает минус в paper. Рабочий режим — **mean-reversion к скользящей средней**
(FORTS внутри дня mean-reverting). Эдж на SBER устойчив OOS (Sharpe ~1.5–2.5); на GAZP
эдж есть, но **тоньше и cost-sensitive** — на трендовом квартале (GZM6, апр–май 2026)
уходит в минус. LKOH внутри дня НЕ mean-reverting → режим специфичен для SBER/GAZP,
а не общее свойство FORTS.

## Алгоритм-победитель (mean-reversion z-score)
Индикаторы по ЗАКРЫТЫМ барам (no look-ahead; проверено next-bar-open исполнением).
```
z[i]   = (close[i] − SMA(close, ma_n)[i]) / std(close, ma_n)[i]   # по барам ≤ i
вход   (только окно основной сессии 10:00–18:45 MSK):
        z ≤ −entry_z  → LONG  (ждём возврата вверх)
        z ≥ +entry_z  → SHORT
выход  (приоритет): |z| вернулся к exit_z → TP; |z| ушёл к stop_z → стоп; max_hold баров → time
сайзинг: 1 лот (фикс). P&L/лот = (exit−entry)·dir·(STEPPRICE/MINSTEP=1.0)
```
Параметры (выбраны joint-объективом min-Sharpe обоих инструментов на train):
`ma_n=36, entry_z=2.5, exit_z=0.5, stop_z=4.0, max_hold=36, сессия [600,1125] мин MSK`.

## OOS-метрики (реальные, fee=1₽/лот/сторона, slip=1 тик/сторона)
Честный train(1-я половина)→test(2-я половина), параметры выбраны на train, не тронуты:
```
SBRF OOS: net +835₽/лот  Sharpe +1.50  trades 104  winrate 65%  (BuyHold −922)
GAZR OOS: net   +5₽/лот  Sharpe +0.01  trades 113  winrate 67%  (BuyHold −3563)
```
Полная 6-мес история на том же фикс-конфиге (in-sample):
```
SBRF: net +2744₽  Sharpe 2.46  wr 66.5%  — плюс в 6/7 месяцев
GAZR: net +1053₽  Sharpe 1.41  wr 70.4%  — минус в апр/мае (трендовый GZM6)
```

## Стресс
- Издержки: SBRF держит плюс до slip 2 тика (Sharpe 1.1); GAZR уходит в минус уже при slip≥1.5т.
- По кварталам: SBRF плюс на SRH6 и SRM6; GAZR плюс на GZH6, минус на GZM6.
- Трендовый гейт (не фейдить сильный тренд) GAZR НЕ спасает — шум, не выбирается на train.
- Сессионный фильтр (только основная 10:00–18:45) — реальное улучшение для GAZR
  (Sharpe +0.42 → +1.49 in-sample), исключает вечернюю/ночную сессию.

## Данные
`load_data.build_continuous(asset)` — непрерывная 10m-серия, сшитая из квартальных
контрактов (каждый в окне ≤90 дн до экспирации), кэш `cache/{ASSET}_10m_90d.pkl`.
Глубина: 2025-12-19 → 2026-06-19 (~13k баров) для SBRF/GAZR/LKOH. ISS отдаёт 10m и по
истёкшим сериям (доступ по SECID); 5m в ISS нет.

## Скрипты
- `load_data.py` — загрузка/сшивка/кэш истории.
- `bt.py` — бэктест-движок (позиционный ряд → сделки) + издержки + метрики + BuyHold.
- `strategies.py` — momentum_atr, **meanrev_z** (победитель), donchian_breakout.
- `screen.py` — широкий in-sample скрин семейств (momentum/MR/breakout).
- `walk_forward.py` — скользящий WF; joint-объектив (min-Sharpe обоих) против оверфита.
- `final_oos.py` — честный train-half→test-half OOS + cost/fee-стресс (главный отчёт).
- `fixed_test.py` — фикс-конфиг на всей истории + по кварталам + cost-sweep.
- `validate3.py` — проверка на 3-м инструменте (LKOH) + сессионный фильтр.

Запуск: `python -m pairsignal.research.forts.final_oos` (и т.п.).

## Рецепт переноса в боевой st5 (НЕ ломая контракты)
st5 сейчас — momentum (`indicators.MomentumIndicator`, `strategy.entry_signal/exit_signal`,
`config.StrategyConfig.lookback/holding/stop_pct`). Mean-reversion — это новый РЕЖИМ,
а не правка параметров. Минимальный безопасный путь:

1. **config.py** — добавить в `StrategyConfig` поля с дефолтами (extra="ignore" уже стоит,
   старые сессии не сломаются): `strategy_mode: Literal["momentum","meanrev"]="meanrev"`,
   `mr_ma_n=36, mr_entry_z=2.5, mr_exit_z=0.5, mr_stop_z=4.0, mr_max_hold=36`,
   `session_lo_min=600, session_hi_min=1125`. Существующие momentum-поля не трогать.
2. **indicators.py** — добавить `ZScoreIndicator` (rolling SMA+std по close, выдаёт z),
   рядом с `MomentumIndicator`. По закрытым барам, как существующий код.
3. **strategy.py** — добавить `entry_signal_mr(z, cfg)` и `exit_signal_mr(pos, z, bars_held, cfg)`;
   существующие momentum-функции оставить. Добавить сессионный фильтр входа по минутам MSK
   (есть `in_clearing_window`/`is_session_end` — аналогичная функция `in_session_window`).
4. **engine.py** — в `TradingEngine.__init__` по `cfg.strategy.strategy_mode` выбрать
   индикатор; в `step()` ветвление вызова entry/exit (momentum vs meanrev). FSM, исполнение,
   paper-P&L (`_pnl_rub`), реконсиляция, sandbox — БЕЗ изменений (контракт InstrumentSpec/
   Position/Trade тот же).
5. **api.py `st5_set_config` (стр. 1622)** — ВНИМАНИЕ: сейчас принимает только DEPRECATED
   VWAP-ключи (band_sigma…), momentum/mr-параметры не пробрасывает. Для UI-управления MR —
   добавить приём `mr_*`/`strategy_mode` через тот же `_num`. Через `hasattr`/`extra="ignore"`
   безопасно; st6 грузит конфиг целиком, но st5 set_config точечный — не затронет.
6. **service.py snapshot** — поля `lookback/holding/stop_pct` в snapshot заменить/дополнить
   на `mr_*` при `strategy_mode=="meanrev"` (вкладка/чарт читает их; см. строки 527–547).
7. session_state_5_*.json: новые поля с дефолтами → `load_session` не падает; reset-инвариант
   и `paused_by_user` не затрагиваются.

Рекомендация по запуску: включить MR на **SBER** (устойчивый OOS-эдж), GAZP держать на
наблюдении/paper до накопления данных следующего квартала — его эдж пока не доказан OOS
под издержками. Sharpe>1 достигнут на SBER честно; на GAZP — только in-sample.
```
