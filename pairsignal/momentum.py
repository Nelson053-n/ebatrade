"""CSMomentum — cross-sectional momentum с walk-forward валидацией.

Контрапункт к mean-reversion (cross_pct/PairsCoint ставили на ВОЗВРАТ спреда — дали ~0
out-of-sample). Momentum ставит на ПРОДОЛЖЕНИЕ относительного тренда: периодически
ранжируем монеты по прошлой доходности, лонг лидеров / шорт аутсайдеров (доллар-нейтраль),
держим до следующего ребаланса.

Портфельная стратегия (2K ног), не парная → НЕ через VirtualExchange, а векторный бэктест
на pandas (матрица доходностей → веса по рангам → P&L минус издержки оборота). Честность —
walk-forward: параметры подбираются на in-sample, прогон на невиданном out-of-sample.

  python -m pairsignal.momentum --top 30 --days 365 --timeframe 1h
  python -m pairsignal.momentum --fee 0.0002 --slippage 0   # мейкер-модель
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .pairs_coint import load_universe
from .scan_year import top_symbols

EXCHANGE = "mexc"
START_EQUITY = 1000.0  # стартовый бумажный капитал live-портфеля
# баров в году по таймфрейму — для annualization Sharpe
_BARS_PER_YEAR = {"5m": 365 * 288, "15m": 365 * 96, "1h": 365 * 24, "4h": 365 * 6, "1d": 365}


# --- ядро: веса портфеля и векторный P&L --------------------------------------

def momentum_weights(prices: pd.DataFrame, lookback: int, holding: int, k: int,
                     long_short: bool) -> pd.DataFrame:
    """Веса портфеля на каждый бар (forward-fill между ребалансами).

    Ранг = доходность за lookback ЗАКРЫТЫХ баров: prices/prices.shift(lookback) − 1.
    На барах ребаланса (каждые holding) топ-k → +1/k, дно-k → −1/k (long_short=True),
    иначе только лонг (вес +1/k). Доллар-нейтраль: Σ лонг = +1, Σ шорт = −1.
    No repaint: на баре t используется только momentum по данным ≤ t.
    """
    mom = prices / prices.shift(lookback) - 1.0
    # NaN-строка = «нет ребаланса» (держим прошлый портфель), числовая строка = новый портфель.
    weights = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)

    for i in range(lookback, len(prices), holding):
        row = mom.iloc[i].dropna()
        need = 2 * k if long_short else k
        if len(row) < need:
            continue
        ranked = row.sort_values(ascending=False)
        wrow = pd.Series(0.0, index=prices.columns)  # выпавшие из топ/дно → 0 (закрытие)
        wrow[ranked.index[:k]] = 1.0 / k
        if long_short:
            wrow[ranked.index[-k:]] = -1.0 / k
        weights.iloc[i] = wrow.to_numpy()

    # forward-fill ЦЕЛЫМИ строками: портфель держится до следующего ребаланса.
    weights = weights.ffill().fillna(0.0)
    return weights


def backtest_momentum(prices: pd.DataFrame, lookback: int, holding: int, k: int,
                      long_short: bool, fee: float, slippage: float) -> dict:
    """Векторный P&L портфеля с издержками оборота. Возвращает метрики + equity-кривую."""
    rets = prices.pct_change().fillna(0.0)
    w = momentum_weights(prices, lookback, holding, k, long_short)

    # доходность портфеля на баре t = Σ вес_{t−1} · доходность_t (вход исполнен прошлым баром)
    port_ret = (w.shift(1) * rets).sum(axis=1)

    # оборот = Σ|Δвес| при ребалансе; издержки = оборот·(fee+slippage)
    turnover = w.diff().abs().sum(axis=1).fillna(0.0)
    costs = turnover * (fee + slippage)

    net = port_ret - costs
    equity = (1.0 + net).cumprod()

    bpy = _BARS_PER_YEAR.get(prices.attrs.get("timeframe", "1h"), 365 * 24)
    mean, std = net.mean(), net.std(ddof=0)
    sharpe = float(mean / std * np.sqrt(bpy)) if std > 0 else 0.0
    dd = float((equity / equity.cummax() - 1.0).min() * 100)
    n_reb = int((turnover > 0).sum())

    return {
        "equity": equity,
        "return_pct": round(float(equity.iloc[-1] - 1.0) * 100, 2),
        "sharpe": round(sharpe, 2),
        "max_dd_pct": round(dd, 2),
        "n_rebalances": n_reb,
        "avg_turnover": round(float(turnover[turnover > 0].mean()) if n_reb else 0.0, 3),
        "costs_pct": round(float(costs.sum()) * 100, 2),
    }


# --- walk-forward валидация ---------------------------------------------------

def walk_forward(prices: pd.DataFrame, train_bars: int, test_bars: int,
                 lookbacks: list[int], holdings: list[int], k: int,
                 long_short: bool, fee: float, slippage: float) -> list[dict]:
    """Скользящие окна: на train подбираем (lookback, holding) по Sharpe, прогоняем на
    следующем test, сдвигаем на test_bars. Параметры выбираются ТОЛЬКО на train → честный OOS.
    """
    out: list[dict] = []
    grid = list(itertools.product(lookbacks, holdings))
    start = 0
    while start + train_bars + test_bars <= len(prices):
        train = prices.iloc[start:start + train_bars]
        test = prices.iloc[start + train_bars:start + train_bars + test_bars]

        # подбор параметров на train по Sharpe
        best, best_sharpe = None, -np.inf
        for lb, hd in grid:
            if lb >= len(train):
                continue
            m = backtest_momentum(train, lb, hd, k, long_short, fee, slippage)
            if m["sharpe"] > best_sharpe:
                best_sharpe, best = m["sharpe"], (lb, hd)
        if best is None:
            start += test_bars
            continue

        lb, hd = best
        # OOS: lookback нужен прогрев — берём хвост train + test, метрики считаем на test-части
        warm = prices.iloc[start + train_bars - lb:start + train_bars + test_bars]
        oos = backtest_momentum(warm, lb, hd, k, long_short, fee, slippage)
        # equity OOS обрезаем на test-период
        oos_eq = oos["equity"].iloc[-len(test):]
        oos_ret = float(oos_eq.iloc[-1] / oos_eq.iloc[0] - 1.0) * 100

        out.append({
            "test_start": test.index[0], "test_end": test.index[-1],
            "lookback": lb, "holding": hd,
            "oos_return_pct": round(oos_ret, 2),
            "oos_sharpe": oos["sharpe"], "oos_max_dd_pct": oos["max_dd_pct"],
            "avg_turnover": oos["avg_turnover"], "costs_pct": oos["costs_pct"],
            "_oos_eq": oos_eq,
        })
        start += test_bars
    return out


# --- вывод --------------------------------------------------------------------

def _d(ts: int) -> str:
    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def _print_table(rows: list[dict]) -> None:
    hdr = ["window", "lookback", "holding", "oos_return_pct", "oos_sharpe",
           "oos_max_dd_pct", "avg_turnover", "costs_pct"]
    disp = [{
        "window": f"{_d(r['test_start'])}…{_d(r['test_end'])}",
        "lookback": r["lookback"], "holding": r["holding"],
        "oos_return_pct": r["oos_return_pct"], "oos_sharpe": r["oos_sharpe"],
        "oos_max_dd_pct": r["oos_max_dd_pct"], "avg_turnover": r["avg_turnover"],
        "costs_pct": r["costs_pct"],
    } for r in rows]
    widths = {h: len(h) for h in hdr}
    for r in disp:
        for h in hdr:
            widths[h] = max(widths[h], len(str(r.get(h, ""))))
    line = "  ".join(h.ljust(widths[h]) for h in hdr)
    print(line)
    print("-" * len(line))
    for r in disp:
        print("  ".join(str(r.get(h, "")).ljust(widths[h]) for h in hdr))


# --- live-режим: paper-портфель с ребалансом (БЕЗ реальных ордеров, Phase 1) ----

def _clean(obj):
    """NaN/inf → None (JSON их не допускает). Тот же приём, что в api.save_session."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def fit_params(prices: pd.DataFrame, lookbacks: list[int], holdings: list[int],
               k: int, long_short: bool, fee: float, slippage: float) -> tuple[int, int]:
    """Подбор (lookback, holding) по Sharpe на переданной истории (последнее train-окно)."""
    best, best_sharpe = (lookbacks[0], holdings[0]), -np.inf
    for lb, hd in itertools.product(lookbacks, holdings):
        if lb >= len(prices):
            continue
        m = backtest_momentum(prices, lb, hd, k, long_short, fee, slippage)
        if m["sharpe"] > best_sharpe:
            best_sharpe, best = m["sharpe"], (lb, hd)
    return best


def target_weights_now(prices: pd.DataFrame, lookback: int, k: int,
                       long_short: bool) -> dict[str, float]:
    """Целевые веса портфеля по ПОСЛЕДНЕМУ закрытому бару (no repaint).

    Ранг = доходность за lookback закрытых баров. Топ-k → +1/k, дно-k → −1/k.
    """
    mom = (prices.iloc[-1] / prices.iloc[-1 - lookback] - 1.0).dropna()
    need = 2 * k if long_short else k
    if len(mom) < need:
        return {}
    ranked = mom.sort_values(ascending=False)
    w = {s: 1.0 / k for s in ranked.index[:k]}
    if long_short:
        w.update({s: -1.0 / k for s in ranked.index[-k:]})
    return w


class PaperPortfolio:
    """Бумажный momentum-портфель: держит веса, считает equity и оборот при ребалансе.

    Реальных ордеров нет — только учёт. Состояние сериализуемо в JSON (forward-test
    переживает перезапуск процесса).
    """

    def __init__(self, start_equity: float, fee: float, slippage: float):
        self.equity = start_equity
        self.fee = fee
        self.slippage = slippage
        self.weights: dict[str, float] = {}       # текущие веса (доля equity на ногу)
        self.last_prices: dict[str, float] = {}   # цены последней отметки (для mark-to-market)
        self.last_ts: int = 0
        self.rebalances: list[dict] = []          # журнал ребалансов

    def mark(self, prices: dict[str, float], ts: int) -> None:
        """Mark-to-market: обновить equity по движению цен с прошлой отметки."""
        if self.last_prices and self.weights:
            ret = 0.0
            for s, w in self.weights.items():
                p0, p1 = self.last_prices.get(s), prices.get(s)
                if p0 and p1:
                    ret += w * (p1 / p0 - 1.0)
            self.equity *= (1.0 + ret)
        self.last_prices = dict(prices)
        self.last_ts = ts

    def rebalance(self, target: dict[str, float], prices: dict[str, float], ts: int) -> dict:
        """Перейти к целевым весам, списать издержки оборота. Вернуть запись журнала."""
        self.mark(prices, ts)
        syms = set(self.weights) | set(target)
        turnover = sum(abs(target.get(s, 0.0) - self.weights.get(s, 0.0)) for s in syms)
        cost = turnover * (self.fee + self.slippage)
        self.equity *= (1.0 - cost)
        self.weights = {s: w for s, w in target.items() if w != 0.0}
        rec = {"ts": ts, "equity": round(self.equity, 2), "turnover": round(turnover, 3),
               "cost_pct": round(cost * 100, 4),
               "longs": sorted(s for s, w in self.weights.items() if w > 0),
               "shorts": sorted(s for s, w in self.weights.items() if w < 0)}
        self.rebalances.append(rec)
        return rec

    def to_dict(self) -> dict:
        return {"equity": self.equity, "fee": self.fee, "slippage": self.slippage,
                "weights": self.weights, "last_prices": self.last_prices,
                "last_ts": self.last_ts, "rebalances": self.rebalances}

    @classmethod
    def from_dict(cls, d: dict) -> "PaperPortfolio":
        p = cls(d["equity"], d["fee"], d["slippage"])
        p.weights = d.get("weights", {})
        p.last_prices = d.get("last_prices", {})
        p.last_ts = d.get("last_ts", 0)
        p.rebalances = d.get("rebalances", [])
        return p


def save_state(path: Path, port: PaperPortfolio, meta: dict) -> None:
    """Атомарная запись состояния (temp+rename) — переживает kill процесса в любой момент."""
    data = _clean({"meta": meta, "portfolio": port.to_dict()})
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)


def load_state(path: Path) -> tuple[PaperPortfolio, dict] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return PaperPortfolio.from_dict(data["portfolio"]), data.get("meta", {})
    except Exception:  # noqa: BLE001
        return None


def _fetch_closes(symbols: list[str], timeframe: str, limit: int, workers: int) -> pd.DataFrame:
    """Свежие закрытые close по юниверсу (последний формирующийся бар отбрасываем)."""
    import ccxt

    def _one(sym: str):
        ex = getattr(ccxt, EXCHANGE)({"enableRateLimit": True, "timeout": 30000})
        ex.options["defaultType"] = "swap"
        raw = ex.fetch_ohlcv(sym, timeframe=timeframe, limit=limit)
        if len(raw) > 1:
            raw = raw[:-1]  # no repaint: убрать формирующийся бар
        return sym, pd.Series([c[4] for c in raw], index=[int(c[0]) for c in raw], dtype="float64")

    cols = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for fut in as_completed([pool.submit(_one, s) for s in symbols]):
            try:
                s, ser = fut.result()
                if len(ser) > 0:
                    cols[s] = ser
            except Exception:  # noqa: BLE001
                continue
    return pd.DataFrame(cols).dropna().sort_index()


def run_live(symbols: list[str], timeframe: str, k: int, long_short: bool,
             fee: float, slippage: float, lookback: int, holding: int,
             poll_seconds: int, state_path: Path, workers: int) -> None:
    """Live forward-test: цикл поллинга, ребаланс каждые holding закрытых баров.

    БЕЗ реальных ордеров — ведёт бумажный портфель. Ctrl-C — graceful shutdown (сохраняет
    состояние). holding и lookback заданы в барах; ребаланс при появлении новых баров.
    """
    loaded = load_state(state_path)
    if loaded:
        port, meta = loaded
        bars_since = meta.get("bars_since_reb", 0)
        print(f"Восстановлено состояние: equity ${port.equity:.2f}, "
              f"{len(port.rebalances)} ребалансов, файл {state_path}")
    else:
        port = PaperPortfolio(START_EQUITY, fee, slippage)
        bars_since = holding  # форсируем ребаланс на первом баре
        print(f"Новый paper-портфель: ${START_EQUITY:.0f}")

    print(f"LIVE momentum (PAPER, без реальных ордеров): {len(symbols)} монет, {timeframe}, "
          f"lookback={lookback} holding={holding} k={k} "
          f"{'long/short' if long_short else 'long-only'}. Ctrl-C для остановки.")

    last_bar_ts = port.last_ts
    try:
        while True:
            prices = _fetch_closes(symbols, timeframe, lookback + 5, workers)
            if len(prices) >= lookback + 1:
                bar_ts = int(prices.index[-1])
                cur = {s: float(prices[s].iloc[-1]) for s in prices.columns}
                if bar_ts > last_bar_ts:  # появился новый закрытый бар
                    bars_since += 1 if last_bar_ts else holding
                    port.mark(cur, bar_ts)
                    if bars_since >= holding:
                        tgt = target_weights_now(prices, lookback, k, long_short)
                        if tgt:
                            rec = port.rebalance(tgt, cur, bar_ts)
                            bars_since = 0
                            ts_s = datetime.fromtimestamp(bar_ts / 1000, tz=timezone.utc
                                                          ).strftime("%Y-%m-%d %H:%M")
                            longs = ", ".join(s.split("/")[0] for s in rec["longs"])
                            shorts = ", ".join(s.split("/")[0] for s in rec["shorts"])
                            print(f"{ts_s} | РЕБАЛАНС | equity ${rec['equity']:.2f} "
                                  f"| оборот {rec['turnover']:.2f} | ЛОНГ: {longs} "
                                  f"| ШОРТ: {shorts or '—'}")
                    last_bar_ts = bar_ts
                    save_state(state_path, port, {"bars_since_reb": bars_since,
                                                  "lookback": lookback, "holding": holding,
                                                  "k": k, "long_short": long_short})
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        save_state(state_path, port, {"bars_since_reb": bars_since, "lookback": lookback,
                                      "holding": holding, "k": k, "long_short": long_short})
        ret = (port.equity / START_EQUITY - 1.0) * 100
        print(f"\nОстановлено. Equity ${port.equity:.2f} ({ret:+.2f}%), "
              f"{len(port.rebalances)} ребалансов. Состояние: {state_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="CSMomentum: cross-sectional momentum, walk-forward")
    ap.add_argument("--top", type=int, default=30, help="монет в юниверс (по обороту)")
    ap.add_argument("--timeframe", default="1h", help="таймфрейм (момент шумен на 5m)")
    ap.add_argument("--days", type=int, default=365, help="глубина истории, дней")
    ap.add_argument("--k", type=int, default=3, help="ног в каждую сторону (лонг/шорт)")
    ap.add_argument("--long-only", action="store_true", help="только лонг (без шортов)")
    ap.add_argument("--train-days", type=int, default=60, help="окно подбора параметров, дней")
    ap.add_argument("--test-days", type=int, default=30, help="окно OOS-прогона, дней")
    ap.add_argument("--fee", type=float, default=0.0006, help="комиссия на ногу (мейкер: 0.0002)")
    ap.add_argument("--slippage", type=float, default=0.0002, help="проскальзывание на ногу")
    ap.add_argument("--workers", type=int, default=5, help="параллельных потоков загрузки")
    ap.add_argument("--symbols", default="", help="явный юниверс через запятую")
    ap.add_argument("--out", default="", help="путь CSV")
    # --- live (paper forward-test, БЕЗ реальных ордеров) ---
    ap.add_argument("--live", action="store_true", help="live paper-портфель (поллинг+ребаланс)")
    ap.add_argument("--poll", type=int, default=60, help="период опроса в live, сек")
    ap.add_argument("--lookback", type=int, default=0, help="live: lookback в барах (0=автоподбор)")
    ap.add_argument("--holding", type=int, default=0, help="live: holding в барах (0=автоподбор)")
    ap.add_argument("--state", default="", help="live: JSON-файл состояния портфеля")
    args = ap.parse_args()

    long_short = not args.long_only
    until_ms = int(time.time() * 1000)
    since_ms = until_ms - args.days * 24 * 3600 * 1000

    if args.symbols.strip():
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        print(f"Отбор топ-{args.top} ликвидных монет на {EXCHANGE}…")
        symbols = top_symbols(args.top)

    print(f"Загрузка {len(symbols)} монет ({args.days}д, {args.timeframe})…")
    # min_coverage: отбрасываем недавно листнувшиеся монеты, иначе inner-join обрежет
    # весь юниверс под самую короткую историю (напр. XAU даёт ~10% баров).
    prices = load_universe(symbols, args.timeframe, since_ms, until_ms, args.workers,
                           min_coverage=0.9)
    prices.attrs["timeframe"] = args.timeframe
    if prices.shape[1] < 2 * args.k or len(prices) < 500:
        print(f"Недостаточно данных: {prices.shape[1]} монет, {len(prices)} баров.")
        return

    lookbacks = [24, 48, 96, 240]
    holdings = [12, 24, 48]

    if args.live:
        # параметры: из флагов либо автоподбор на последнем train-окне (как в walk-forward)
        bph_d = _BARS_PER_YEAR.get(args.timeframe, 365 * 24) / 365
        if args.lookback and args.holding:
            lb, hd = args.lookback, args.holding
            print(f"Параметры из флагов: lookback={lb}, holding={hd}")
        else:
            tail = prices.iloc[-int(args.train_days * bph_d):]
            lb, hd = fit_params(tail, lookbacks, holdings, args.k, long_short,
                                args.fee, args.slippage)
            print(f"Автоподбор на последних {args.train_days}д: lookback={lb}, holding={hd}")
        state_path = Path(args.state) if args.state else (
            Path(__file__).parent / "out" / "momentum_live_state.json")
        state_path.parent.mkdir(parents=True, exist_ok=True)
        run_live(list(prices.columns), args.timeframe, args.k, long_short, args.fee,
                 args.slippage, lb, hd, args.poll, state_path, args.workers)
        return

    bph = _BARS_PER_YEAR.get(args.timeframe, 365 * 24) / 365  # баров в дне
    train_bars = int(args.train_days * bph)
    test_bars = int(args.test_days * bph)
    print(f"Юниверс: {prices.shape[1]} монет, {len(prices)} баров "
          f"({_d(prices.index[0])} … {_d(prices.index[-1])}).")
    print(f"Walk-forward: train={train_bars} / test={test_bars} баров "
          f"({'long/short' if long_short else 'long-only'}, k={args.k}).")

    rows = walk_forward(prices, train_bars, test_bars, lookbacks, holdings,
                        args.k, long_short, args.fee, args.slippage)
    if not rows:
        print("Недостаточно истории для walk-forward окон. Увеличь --days или уменьши окна.")
        return

    print(f"\n=== CSMOMENTUM: OUT-OF-SAMPLE по {len(rows)} окнам ===")
    _print_table(rows)

    # склейка OOS equity всех окон в одну кривую портфеля
    eqs = []
    cum = 1.0
    for r in rows:
        seg = r["_oos_eq"] / r["_oos_eq"].iloc[0] * cum
        eqs.append(seg)
        cum = float(seg.iloc[-1])
    full_eq = pd.concat(eqs)
    total_ret = (cum - 1.0) * 100
    bpy = _BARS_PER_YEAR.get(args.timeframe, 365 * 24)
    fr = full_eq.pct_change().dropna()
    sharpe = float(fr.mean() / fr.std(ddof=0) * np.sqrt(bpy)) if fr.std(ddof=0) > 0 else 0.0
    maxdd = float((full_eq / full_eq.cummax() - 1.0).min() * 100)
    wins = sum(1 for r in rows if r["oos_return_pct"] > 0)
    print(f"\nСклеенный OOS-портфель: return {total_ret:+.2f}%, Sharpe {sharpe:.2f}, "
          f"maxDD {maxdd:.2f}%, прибыльных окон {wins}/{len(rows)}")

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path(__file__).parent / "out" / f"momentum_{_d(until_ms)}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["window", "lookback", "holding", "oos_return_pct", "oos_sharpe",
              "oos_max_dd_pct", "avg_turnover", "costs_pct"]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({
                "window": f"{_d(r['test_start'])}…{_d(r['test_end'])}",
                "lookback": r["lookback"], "holding": r["holding"],
                "oos_return_pct": r["oos_return_pct"], "oos_sharpe": r["oos_sharpe"],
                "oos_max_dd_pct": r["oos_max_dd_pct"], "avg_turnover": r["avg_turnover"],
                "costs_pct": r["costs_pct"],
            })
    print(f"\nCSV сохранён: {out_path}")


if __name__ == "__main__":
    main()
