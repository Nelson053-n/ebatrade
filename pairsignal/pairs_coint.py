"""PairsCoint — pairs-trading с отбором пар по коинтеграции (out-of-sample).

Тот же принцип mean-reversion, что и в cross_pct, но между ДВУМЯ коррелированными
монетами на ОДНОЙ бирже (mexc) — исполнимо в Phase 1, спред крупный, издержки не
съедают. Главное отличие от scan_year: пары не берутся вслепую, а отбираются научно —
тест Энгла-Грейнджера на ПЕРВОЙ половине истории (train), торговля на ВТОРОЙ (test),
чтобы не переобучаться.

Поток: топ-монеты → загрузка цен (mexc) → split 50/50 →
        отбор коинтегрированных пар на train → торговля на test → сводка + CSV.

  python -m pairsignal.pairs_coint                       # топ-30, 180д, тейкер
  python -m pairsignal.pairs_coint --fee 0.0002 --slippage 0   # мейкер-модель
"""
from __future__ import annotations

import argparse
import csv
import itertools
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from .config import AppConfig, PaperConfig, StrategyConfig
from .data_feed import fetch_ohlcv_paged
from .main import run_backtest
from .scan_year import top_symbols

EXCHANGE = "mexc"  # одна биржа: исполнимо, узкий bid/ask у ликвидных монет
START_BALANCE = 1000.0


# --- загрузка юниверса ---------------------------------------------------------

def load_universe(symbols: list[str], timeframe: str, since_ms: int, until_ms: int,
                  workers: int = 5, min_coverage: float = 0.0) -> pd.DataFrame:
    """close каждой монеты с одной биржи (EXCHANGE), выровненные по ts (inner-join).

    Возвращает DataFrame: индекс ts(ms), по колонке на монету. Загрузку параллелим —
    узкое место сетевое.

    min_coverage>0: монеты, чья история короче min_coverage·(макс длина среди всех),
    отбрасываются ДО inner-join — иначе одна недавно листнувшаяся монета обрезает весь
    юниверс под свою короткую историю. Дефолт 0.0 — прежнее поведение (строгий join).
    """
    import ccxt

    def _one(sym: str) -> tuple[str, pd.Series]:
        ex = getattr(ccxt, EXCHANGE)({"enableRateLimit": True, "timeout": 30000})
        ex.options["defaultType"] = "swap"
        s = fetch_ohlcv_paged(ex, sym, timeframe, since_ms, until_ms)
        return sym, s

    cols: dict[str, pd.Series] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = [pool.submit(_one, s) for s in symbols]
        for fut in as_completed(futs):
            try:
                sym, s = fut.result()
                if len(s) > 0:
                    cols[sym] = s
            except Exception:  # noqa: BLE001 — пропускаем монету, что не загрузилась
                continue

    if min_coverage > 0.0 and cols:
        longest = max(len(s) for s in cols.values())
        cols = {k: s for k, s in cols.items() if len(s) >= min_coverage * longest}

    df = pd.DataFrame(cols).dropna().sort_index()
    return df


# --- отбор пар по коинтеграции -------------------------------------------------

def _half_life(spread: np.ndarray) -> float:
    """Half-life возврата спреда к среднему: OLS Δs_t ~ s_{t-1}, hl = -ln2 / λ.

    λ — коэффициент при лаге (для mean-reverting ряда λ<0). Возвращает inf, если
    ряд не возвращается (λ≥0) — такую пару отбрасываем.
    """
    s = spread[~np.isnan(spread)]
    if len(s) < 10:
        return float("inf")
    s_lag = s[:-1]
    ds = np.diff(s)
    # OLS ds = a + λ·s_lag
    x = np.column_stack([np.ones_like(s_lag), s_lag])
    coef, *_ = np.linalg.lstsq(x, ds, rcond=None)
    lam = coef[1]
    if lam >= 0:
        return float("inf")
    return float(-np.log(2.0) / lam)


def select_pairs(prices_train: pd.DataFrame, pvalue_max: float = 0.05,
                 corr_min: float = 0.7, max_half_life: float = 600.0,
                 top_pairs: int = 15, workers: int = 5, max_points: int = 2000) -> list[dict]:
    """Отбор коинтегрированных пар на TRAIN-данных.

    Для каждой пары C(n,2):
      1) предфильтр по corr лог-доходностей (отсечь некоррелированный мусор быстро);
      2) тест Энгла-Грейнджера coint(log_a, log_b) → p-value;
      3) half-life возврата спреда ln(a)−β·ln(b) (β — OLS-наклон на train).
    Порядок монет в паре фиксируем по алфавиту (coint асимметричен → воспроизводимость).

    coint() на длинных рядах дорог по CPU (435 пар × 8640 баров → десятки минут). Train
    прорежается до ~max_points точек: тест на стационарность от равномерного прореживания
    не страдает, а скорость растёт на порядок. half-life пересчитываем в исходные бары.
    Возвращает список dict, отсортированный по p-value возр., топ-N.
    """
    syms = sorted(prices_train.columns)
    step = max(1, len(prices_train) // max_points)  # коэффициент прореживания
    ds = prices_train.iloc[::step]
    log = {s: np.log(ds[s].to_numpy()) for s in syms}
    rets = {s: np.diff(log[s]) for s in syms}

    def _test(a: str, b: str) -> dict | None:
        corr = float(np.corrcoef(rets[a], rets[b])[0, 1])
        if not np.isfinite(corr) or abs(corr) < corr_min:
            return None
        la, lb = log[a], log[b]
        try:
            _, pval, _ = coint(la, lb)
        except Exception:  # noqa: BLE001
            return None
        if not np.isfinite(pval) or pval > pvalue_max:
            return None
        # β как OLS-наклон ln(a) ~ ln(b), спред = ln(a) − β·ln(b)
        x = np.column_stack([np.ones_like(lb), lb])
        coef, *_ = np.linalg.lstsq(x, la, rcond=None)
        beta = float(coef[1])
        spread = la - beta * lb
        hl = _half_life(spread) * step  # прорежённые бары → исходные
        if hl > max_half_life:
            return None
        return {"sym_a": a, "sym_b": b, "coint_pvalue": round(pval, 5),
                "half_life_bars": round(hl, 1), "corr": round(corr, 3),
                "beta_ols": round(beta, 4)}

    results: list[dict] = []
    combos = list(itertools.combinations(syms, 2))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = [pool.submit(_test, a, b) for a, b in combos]
        for fut in as_completed(futs):
            r = fut.result()
            if r is not None:
                results.append(r)

    results.sort(key=lambda r: r["coint_pvalue"])
    return results[:top_pairs]


# --- торговля пары на TEST ------------------------------------------------------

def make_pair_cfg(sym_a: str, sym_b: str, timeframe: str,
                  slippage_pct: float = 0.0002, fee: float = 0.0006) -> AppConfig:
    """Конфиг log-режима под одну пару (классический pairs-trading, баланс $1000)."""
    s = StrategyConfig(
        symbol_a=sym_a, symbol_b=sym_b, data_exchange=EXCHANGE,
        spread_mode="log", beta_window=240, bb_period=240, bb_k=2.0,
        entry_z=2.5, stop_z=8.0, exit_z=0.2, min_width_pct=0.5,
        profit_target_fees=6.0, max_bars_in_trade=576, timeframe=timeframe,
    )
    p = PaperConfig(start_balance=START_BALANCE, risk_pct=2.0,
                    taker_fee=fee, slippage_pct=slippage_pct)
    return AppConfig(strategy=s, paper=p, auto_approve=True)


def trade_pair(pair: dict, prices_test: pd.DataFrame, timeframe: str,
               slippage_pct: float, fee: float) -> dict:
    """Прогон одной отобранной пары на TEST-данных. Возвращает строку результата."""
    a, b = pair["sym_a"], pair["sym_b"]
    label = f"{a.split('/')[0]}/{b.split('/')[0]}"
    base = {"pair": label, "coint_pvalue": pair["coint_pvalue"],
            "half_life_bars": pair["half_life_bars"], "corr": pair["corr"],
            "win_rate_pct": 0.0, "trades": 0, "net_pnl_$": 0.0,
            "final_equity_$": START_BALANCE, "return_pct": 0.0, "status": "ok"}
    try:
        df = (prices_test[[a, b]]
              .rename(columns={a: "price_a", b: "price_b"})
              .dropna().sort_index())
        warmup_min = 240 + 240 + 80  # bb_period + beta_window + запас (log-режим)
        if len(df) < warmup_min:
            base["status"] = f"мало test-данных ({len(df)} баров)"
            return base
        cfg = make_pair_cfg(a, b, timeframe, slippage_pct, fee)
        eng = run_backtest(cfg, df, verbose=False)
        summ = eng.summary()
        base.update({"win_rate_pct": summ["win_rate_pct"], "trades": summ["trades"],
                     "net_pnl_$": summ["net_pnl"], "final_equity_$": summ["equity"],
                     "return_pct": summ["return_pct"]})
        return base
    except Exception as e:  # noqa: BLE001 — изоляция пары
        base["status"] = f"ошибка: {type(e).__name__}: {e}"
        return base


# --- вывод ---------------------------------------------------------------------

def _print_pairs_table(rows: list[dict]) -> None:
    hdr = ["pair", "coint_pvalue", "half_life_bars", "corr", "win_rate_pct",
           "trades", "net_pnl_$", "final_equity_$", "return_pct", "status"]
    widths = {h: len(h) for h in hdr}
    for r in rows:
        for h in hdr:
            widths[h] = max(widths[h], len(str(r.get(h, ""))))
    line = "  ".join(h.ljust(widths[h]) for h in hdr)
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(r.get(h, "")).ljust(widths[h]) for h in hdr))


def main() -> None:
    ap = argparse.ArgumentParser(description="PairsCoint: pairs-trading с отбором по коинтеграции")
    ap.add_argument("--top", type=int, default=30, help="монет в юниверс (по обороту)")
    ap.add_argument("--timeframe", default="5m", help="таймфрейм свечей")
    ap.add_argument("--days", type=int, default=180, help="глубина истории, дней")
    ap.add_argument("--pvalue", type=float, default=0.05, help="макс p-value коинтеграции")
    ap.add_argument("--corr-min", type=float, default=0.7, help="мин |corr| лог-доходностей")
    # half-life возврата спреда между монетами обычно ~1-2 дня (сотни 5m-баров);
    # 600 ≈ тайм-стоп max_bars_in_trade, чтобы сделка успевала закрыться по цели.
    ap.add_argument("--max-half-life", type=float, default=600.0, help="макс half-life, бары")
    ap.add_argument("--top-pairs", type=int, default=15, help="сколько лучших пар торговать")
    ap.add_argument("--fee", type=float, default=0.0006, help="комиссия на ногу (мейкер: 0.0002)")
    ap.add_argument("--slippage", type=float, default=0.0002, help="проскальзывание на ногу")
    ap.add_argument("--workers", type=int, default=5, help="параллельных потоков")
    ap.add_argument("--symbols", default="", help="явный юниверс через запятую (вместо топа)")
    ap.add_argument("--out", default="", help="путь CSV")
    args = ap.parse_args()

    until_ms = int(time.time() * 1000)
    since_ms = until_ms - args.days * 24 * 3600 * 1000

    if args.symbols.strip():
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        print(f"Отбор топ-{args.top} ликвидных монет на {EXCHANGE}…")
        symbols = top_symbols(args.top)

    print(f"Загрузка {len(symbols)} монет ({args.days}д, {args.timeframe})…")
    prices = load_universe(symbols, args.timeframe, since_ms, until_ms, args.workers)
    if prices.shape[1] < 2 or len(prices) < 1000:
        print(f"Недостаточно данных: {prices.shape[1]} монет, {len(prices)} баров.")
        return

    # split 50/50: отбор на train, торговля на test (out-of-sample)
    mid = len(prices) // 2
    train, test = prices.iloc[:mid], prices.iloc[mid:]

    def _d(ts: int) -> str:
        return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    print(f"Юниверс: {prices.shape[1]} монет, {len(prices)} общих баров.")
    print(f"TRAIN (отбор): {_d(train.index[0])} … {_d(train.index[-1])} ({len(train)} баров)")
    print(f"TEST  (торг):  {_d(test.index[0])} … {_d(test.index[-1])} ({len(test)} баров)")

    selected = select_pairs(train, args.pvalue, args.corr_min,
                            args.max_half_life, args.top_pairs, args.workers)
    print(f"\nОтобрано коинтегрированных пар (p<{args.pvalue}): {len(selected)}")
    if not selected:
        print("Коинтегрированных пар не найдено — стратегия не применима на этом юниверсе.")
        return

    rows: list[dict] = []
    for i, pair in enumerate(selected, 1):
        row = trade_pair(pair, test, args.timeframe, args.slippage, args.fee)
        print(f"[{i}/{len(selected)}] {row['pair']} — {row['status']}", flush=True)
        rows.append(row)

    rows.sort(key=lambda r: r.get("return_pct", 0.0), reverse=True)

    print("\n=== PAIRSCOINT: OUT-OF-SAMPLE (старт $1000 на пару) ===")
    _print_pairs_table(rows)

    # агрегат портфеля: равный вес $1000/пара
    ok = [r for r in rows if r["status"] == "ok"]
    if ok:
        total_pnl = sum(r["final_equity_$"] - START_BALANCE for r in ok)
        invested = START_BALANCE * len(ok)
        wins = sum(1 for r in ok if r["return_pct"] > 0)
        print(f"\nПортфель из {len(ok)} пар: вложено ${invested:.0f}, "
              f"P&L ${total_pnl:+.2f} ({100*total_pnl/invested:+.2f}%), "
              f"прибыльных пар {wins}/{len(ok)}")

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path(__file__).parent / "out" / f"pairs_{_d(until_ms)}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["pair", "coint_pvalue", "half_life_bars", "corr", "win_rate_pct",
              "trades", "net_pnl_$", "final_equity_$", "return_pct", "status"]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\nCSV сохранён: {out_path}")


if __name__ == "__main__":
    main()
