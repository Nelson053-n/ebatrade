"""Overnight-эффект на MOEX: торгуемая стратегия + честный стресс-тест.

Сигнал/правило (no look-ahead, безпараметрический по сути):
  Каждый торговый день: купить корзину на close[t-1], продать на open[t].
  Доходность ночи = open[t]/close[t-1] - 1. Издержки — round-trip за ночь.

Варианты корзины:
  - EW_ALL: equal-weight все тикеры (где есть и close[t-1], и open[t]).
  - TOP_K: only те тикеры, у кого overnight-эффект устойчив В ОБУЧЕНИИ (in-sample),
    отбор на первой половине, торговля на второй (honest OOS).

Стресс:
  - split-sample H1/H2,
  - jackknife по тикерам (выкинуть по одному — устойчиво ли),
  - jackknife по годам (не артефакт ли одного года),
  - издержки 0.02 / 0.04 / 0.06 / 0.10%,
  - сравнение с BuyHold (держать корзину 24/7).
"""
from __future__ import annotations

import glob
import os

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


def matrices(panel):
    op = pd.DataFrame({t: pd.Series(df["open"].values, index=df["date"].values) for t, df in panel.items()}).sort_index()
    cl = pd.DataFrame({t: pd.Series(df["close"].values, index=df["date"].values) for t, df in panel.items()}).sort_index()
    return op, cl


def tstat(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 3 or x.std(ddof=1) == 0: return 0.0
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def ann_sharpe(x, ppy=250):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 3 or x.std(ddof=1) == 0: return 0.0
    return float(x.mean() / x.std(ddof=1) * np.sqrt(ppy))


def overnight_matrix(op, cl):
    """Матрица overnight-доходностей: ON[t] = open[t]/close[t-1]-1 (по каждому тикеру)."""
    return op / cl.shift(1) - 1.0


def intraday_matrix(op, cl):
    return cl / op - 1.0


def portfolio_overnight(op, cl, tickers, cost):
    """EW overnight-портфель по списку tickers. Возврат: Series доходностей по дате (net)."""
    on = overnight_matrix(op, cl)[tickers]
    daily = on.mean(axis=1, skipna=True)          # EW по доступным тикерам в день
    daily = daily.dropna()
    net = daily - cost                            # round-trip каждую ночь
    return net


def buyhold(op, cl, tickers):
    """EW BuyHold: close-to-close доходность корзины (держим 24/7)."""
    c = cl[tickers]
    r = (c / c.shift(1) - 1.0).mean(axis=1, skipna=True).dropna()
    return r


def split_report(net: pd.Series, label: str):
    n = len(net)
    half = n // 2
    h1, h2 = net.iloc[:half], net.iloc[half:]
    print(f"{label}: mean/day={net.mean()*100:.4f}% t={tstat(net.values):.2f} "
          f"Sharpe={ann_sharpe(net.values):.2f} ann={((1+net).prod()**(250/n)-1)*100:.1f}% "
          f"| H1={h1.mean()*100:.4f}(t={tstat(h1.values):.1f}) H2={h2.mean()*100:.4f}(t={tstat(h2.values):.1f}) n={n}")
    return h1, h2


def main():
    panel = load_panel()
    op, cl = matrices(panel)
    all_t = list(panel.keys())
    cost = 0.0005
    print(f"{len(all_t)} tickers, {op.index[0].date()} .. {op.index[-1].date()}\n")

    print("=" * 78)
    print("1) EW_ALL overnight portfolio (buy all @close[t-1], sell @open[t])")
    print("=" * 78)
    net = portfolio_overnight(op, cl, all_t, cost)
    split_report(net, f"EW_ALL net(cost={cost*100:.2f}%)")
    bh = buyhold(op, cl, all_t)
    bh = bh.loc[net.index.intersection(bh.index)]
    print(f"BuyHold EW (c2c): mean/day={bh.mean()*100:.4f}% Sharpe={ann_sharpe(bh.values):.2f} "
          f"ann={((1+bh).prod()**(250/len(bh))-1)*100:.1f}% n={len(bh)}")
    # intraday-only портфель (для контраста)
    idm = intraday_matrix(op, cl)[all_t].mean(axis=1).dropna() - cost
    print(f"Intraday-only EW net: mean/day={idm.mean()*100:.4f}% t={tstat(idm.values):.2f} Sharpe={ann_sharpe(idm.values):.2f}")

    print("\n" + "=" * 78)
    print("2) COST SENSITIVITY (EW_ALL overnight)")
    print("=" * 78)
    for c in (0.0, 0.0002, 0.0004, 0.0005, 0.0006, 0.0008, 0.0010):
        x = portfolio_overnight(op, cl, all_t, c)
        print(f"  cost={c*100:.2f}%: mean/day={x.mean()*100:7.4f}% t={tstat(x.values):5.2f} "
              f"Sharpe={ann_sharpe(x.values):5.2f} ann={((1+x).prod()**(250/len(x))-1)*100:6.1f}%")
    # breakeven cost
    gross = portfolio_overnight(op, cl, all_t, 0.0)
    print(f"  -> breakeven round-trip cost = {gross.mean()*100:.4f}% (gross mean/day)")

    print("\n" + "=" * 78)
    print("3) JACKKNIFE по тикерам (выкинуть по одному, cost=0.05%)")
    print("=" * 78)
    base = portfolio_overnight(op, cl, all_t, cost)
    means = []
    for t in all_t:
        sub = [x for x in all_t if x != t]
        x = portfolio_overnight(op, cl, sub, cost)
        means.append(x.mean())
    means = np.array(means)
    print(f"base mean/day={base.mean()*100:.4f}%  | leave-one-out range "
          f"[{means.min()*100:.4f}%, {means.max()*100:.4f}%]  all>0: {(means>0).all()}")

    print("\n" + "=" * 78)
    print("4) JACKKNIFE по годам (выкинуть по году, cost=0.05%)")
    print("=" * 78)
    net = portfolio_overnight(op, cl, all_t, cost)
    yrs = net.index.year
    print("  per-year:")
    for y in sorted(set(yrs)):
        x = net[yrs == y]
        print(f"    {y}: mean/day={x.mean()*100:7.4f}% t={tstat(x.values):5.2f} Sharpe={ann_sharpe(x.values):5.2f} n={len(x)}")
    print("  leave-one-year-out:")
    loo = []
    for y in sorted(set(yrs)):
        x = net[yrs != y]
        loo.append(x.mean())
        print(f"    drop {y}: mean/day={x.mean()*100:.4f}% t={tstat(x.values):.2f}")
    print(f"  all leave-one-year-out >0: {(np.array(loo)>0).all()}")

    print("\n" + "=" * 78)
    print("5) HONEST OOS: отбор тикеров на H1, торговля на H2 (TOP_K по in-sample on_t)")
    print("=" * 78)
    on = overnight_matrix(op, cl)
    n = len(net)
    # граница по индексу overnight-серии
    dates = on.dropna(how="all").index
    half_date = dates[len(dates)//2]
    on_h1 = on.loc[on.index < half_date]
    # отбор: тикеры с положительным и значимым overnight в H1
    sel = [t for t in all_t if tstat(on_h1[t].dropna().values) > 2 and on_h1[t].mean() > cost]
    print(f"  selected on H1 (on_t>2 & mean>{cost*100:.2f}%): {len(sel)} -> {sorted(sel)}")
    # торгуем выбранную корзину только на H2 (OOS)
    on_h2 = on.loc[on.index >= half_date]
    oos = on_h2[sel].mean(axis=1).dropna() - cost
    print(f"  OOS (H2) selected basket: mean/day={oos.mean()*100:.4f}% t={tstat(oos.values):.2f} "
          f"Sharpe={ann_sharpe(oos.values):.2f} ann={((1+oos).prod()**(250/len(oos))-1)*100:.1f}% n={len(oos)}")
    # для сравнения — всю корзину на H2
    oos_all = on_h2[all_t].mean(axis=1).dropna() - cost
    print(f"  OOS (H2) full basket:     mean/day={oos_all.mean()*100:.4f}% t={tstat(oos_all.values):.2f} "
          f"Sharpe={ann_sharpe(oos_all.values):.2f} ann={((1+oos_all).prod()**(250/len(oos_all))-1)*100:.1f}%")


if __name__ == "__main__":
    main()
