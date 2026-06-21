"""Календарные/микроструктурные аномалии на дневных OHLC MOEX.

Тестируем:
  A) Overnight vs Intraday декомпозиция: close[t]->open[t+1] (overnight)
     против open[t]->close[t] (intraday). Классика: overnight > 0, intraday ~ 0.
  B) Short-term reversal (cross-sectional): вчерашние лузеры -> лонг сегодня.
  C) Day-of-week сезонность дневной доходности.

Метод: split-sample (первая половина / вторая половина) + t-stat + издержки.
no look-ahead: сигнал на баре t использует только данные <= t, исполнение на t (+open/close).
"""
from __future__ import annotations

import glob
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

CACHE = os.path.join(os.path.dirname(__file__), "cache", "ohlc")


def load_panel() -> dict[str, pd.DataFrame]:
    out = {}
    for f in sorted(glob.glob(os.path.join(CACHE, "*_1d.csv"))):
        t = os.path.basename(f).replace("_1d.csv", "")
        df = pd.read_csv(f, index_col=0)
        df.index = df.index.astype("int64")
        df["date"] = pd.to_datetime(df["begin"]).dt.normalize()
        df = df[df[["open", "high", "low", "close"]].gt(0).all(axis=1)]
        out[t] = df
    return out


def tstat(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return 0.0
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def ann_sharpe(daily: np.ndarray, ppy: float = 250) -> float:
    daily = daily[~np.isnan(daily)]
    if len(daily) < 3 or daily.std(ddof=1) == 0:
        return 0.0
    return float(daily.mean() / daily.std(ddof=1) * np.sqrt(ppy))


# ---------------------------------------------------------------------------
# A) Overnight vs Intraday decomposition
# ---------------------------------------------------------------------------
def overnight_intraday(panel: dict[str, pd.DataFrame], cost: float = 0.0005):
    """Per-ticker: средние overnight (close[t-1]->open[t]) и intraday (open[t]->close[t]).

    cost — round-trip издержки (комиссия+слип) на сделку. Overnight-стратегия:
    купить на close[t-1], продать на open[t] -> один round-trip за ночь.
    """
    rows = []
    pooled_on, pooled_id = [], []
    pooled_on_h1, pooled_on_h2 = [], []
    for t, df in panel.items():
        o = df["open"].values
        c = df["close"].values
        # overnight return бар t: open[t]/close[t-1]-1
        on = o[1:] / c[:-1] - 1.0
        # intraday бар t: close[t]/open[t]-1
        idr = (c / o - 1.0)[1:]
        dates = df["date"].values[1:]
        n = len(on)
        half = n // 2
        rows.append({
            "ticker": t, "n": n,
            "on_mean": on.mean(), "on_t": tstat(on),
            "on_mean_net": on.mean() - cost,
            "id_mean": idr.mean(), "id_t": tstat(idr),
            "on_h1": on[:half].mean(), "on_h2": on[half:].mean(),
        })
        pooled_on.append(on)
        pooled_id.append(idr)
        pooled_on_h1.append(on[:half])
        pooled_on_h2.append(on[half:])
    res = pd.DataFrame(rows).set_index("ticker")
    return res


# ---------------------------------------------------------------------------
# B) Short-term reversal (cross-sectional, daily)
# ---------------------------------------------------------------------------
def build_close_matrix(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    s = {t: pd.Series(df["close"].values, index=df["date"].values) for t, df in panel.items()}
    m = pd.DataFrame(s).sort_index()
    return m


def build_oc_matrices(panel: dict[str, pd.DataFrame]):
    op = {t: pd.Series(df["open"].values, index=df["date"].values) for t, df in panel.items()}
    cl = {t: pd.Series(df["close"].values, index=df["date"].values) for t, df in panel.items()}
    return pd.DataFrame(op).sort_index(), pd.DataFrame(cl).sort_index()


def reversal_xs(panel, lookback=1, frac=0.2, cost=0.0005, hold_overnight_only=False):
    """Cross-sectional reversal.

    Сигнал на close[t]: ранжируем по доходности за lookback дней (close).
    Лонг нижний frac (лузеры), шорт верхний frac (победители).
    Исполнение:
      hold_overnight_only=False: вход close[t] -> выход close[t+1] (дневной holding).
      hold_overnight_only=True:  вход close[t] -> выход open[t+1] (только ночь).
    Доллар-нейтрально, equal-weight внутри ног. cost — на ногу round-trip.
    """
    opn, cls = build_oc_matrices(panel)
    cls = cls.dropna(how="all")
    ret_lb = cls.pct_change(lookback)
    fwd_close = cls.shift(-1) / cls - 1.0          # close[t]->close[t+1]
    fwd_on = opn.shift(-1) / cls - 1.0             # close[t]->open[t+1]
    fwd = fwd_on if hold_overnight_only else fwd_close

    dates = cls.index
    pnl = []
    pnl_dates = []
    for i in range(lookback, len(dates) - 1):
        r = ret_lb.iloc[i].dropna()
        f = fwd.iloc[i]
        valid = r.index.intersection(f.dropna().index)
        r = r.loc[valid]
        if len(r) < 6:
            continue
        k = max(1, int(len(r) * frac))
        losers = r.nsmallest(k).index
        winners = r.nlargest(k).index
        long_ret = f.loc[losers].mean()
        short_ret = f.loc[winners].mean()
        gross = 0.5 * (long_ret - short_ret)        # market-neutral, half capital each side
        # turnover: полностью переоткрываем обе ноги каждый день -> 2 ноги round-trip
        net = gross - cost
        pnl.append(net)
        pnl_dates.append(dates[i])
    return pd.Series(pnl, index=pd.DatetimeIndex(pnl_dates))


# ---------------------------------------------------------------------------
# C) Day-of-week (pooled across tickers, daily close-to-close)
# ---------------------------------------------------------------------------
def day_of_week(panel: dict[str, pd.DataFrame]):
    rows = []
    for t, df in panel.items():
        c = df["close"].values
        r = c[1:] / c[:-1] - 1.0
        d = pd.to_datetime(df["date"].values[1:])
        for dow in range(5):
            mask = d.dayofweek == dow
            x = r[mask]
            if len(x) > 20:
                rows.append({"dow": dow, "ret": x})
    out = {}
    for dow in range(5):
        allr = np.concatenate([row["ret"] for row in rows if row["dow"] == dow])
        out[dow] = (allr.mean(), tstat(allr), len(allr))
    return out


def report():
    print("Loading daily OHLC panel...")
    panel = load_panel()
    d0 = min(df["date"].min() for df in panel.values())
    d1 = max(df["date"].max() for df in panel.values())
    print(f"{len(panel)} tickers, {d0.date()} .. {d1.date()}\n")

    cost = 0.0005  # 0.05% round-trip (комиссия+слип), середина диапазона 0.04-0.06

    print("=" * 70)
    print("A) OVERNIGHT vs INTRADAY (per-ticker means, daily)")
    print("=" * 70)
    res = overnight_intraday(panel, cost=cost)
    res_sorted = res.sort_values("on_mean", ascending=False)
    print(f"{'tkr':6} {'on_mean%':>9} {'on_t':>6} {'on_net%':>8} {'id_mean%':>9} {'id_t':>6} {'on_h1%':>7} {'on_h2%':>7}")
    for t, r in res_sorted.iterrows():
        print(f"{t:6} {r.on_mean*100:9.4f} {r.on_t:6.2f} {r.on_mean_net*100:8.4f} "
              f"{r.id_mean*100:9.4f} {r.id_t:6.2f} {r.on_h1*100:7.4f} {r.on_h2*100:7.4f}")
    print(f"\nPOOLED overnight mean: {res.on_mean.mean()*100:.4f}% | intraday mean: {res.id_mean.mean()*100:.4f}%")
    print(f"# tickers overnight>0: {(res.on_mean>0).sum()}/{len(res)} | net>0: {(res.on_mean_net>0).sum()}/{len(res)}")
    print(f"# overnight t>2: {(res.on_t>2).sum()} | overnight>0 BOTH halves: {((res.on_h1>0)&(res.on_h2>0)).sum()}/{len(res)}")

    print("\n" + "=" * 70)
    print("B) SHORT-TERM REVERSAL (cross-sectional, daily close->close)")
    print("=" * 70)
    for lb in (1, 2, 3, 5):
        for frac in (0.1, 0.2, 0.3):
            pnl = reversal_xs(panel, lookback=lb, frac=frac, cost=cost)
            if len(pnl) < 50:
                continue
            half = len(pnl) // 2
            h1, h2 = pnl.iloc[:half], pnl.iloc[half:]
            print(f"lb={lb} frac={frac:.1f}: mean/day={pnl.mean()*100:7.4f}% t={tstat(pnl.values):5.2f} "
                  f"Sharpe={ann_sharpe(pnl.values):5.2f} | H1 mean={h1.mean()*100:7.4f}(t={tstat(h1.values):4.1f}) "
                  f"H2 mean={h2.mean()*100:7.4f}(t={tstat(h2.values):4.1f}) n={len(pnl)}")
    print("  -- gross (no cost) for reference --")
    for lb in (1,):
        for frac in (0.2,):
            pnl = reversal_xs(panel, lookback=lb, frac=frac, cost=0.0)
            print(f"  lb={lb} frac={frac:.1f} GROSS: mean/day={pnl.mean()*100:.4f}% t={tstat(pnl.values):.2f} Sharpe={ann_sharpe(pnl.values):.2f}")

    print("\n" + "=" * 70)
    print("C) DAY-OF-WEEK (pooled close-to-close)")
    print("=" * 70)
    dow = day_of_week(panel)
    names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    for d in range(5):
        m, ts, n = dow[d]
        print(f"{names[d]}: mean={m*100:7.4f}% t={ts:6.2f} n={n}")


if __name__ == "__main__":
    report()
