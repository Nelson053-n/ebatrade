"""Внутридневные микроструктурные эффекты на часовых барах MOEX.

Сессия акций MOEX (MSK): основная 10:00..18:39 + вечерняя 19:00..23:49.
Часовой бар с begin=10:00 — первый полный час; 18:00 — последний час основной сессии.

Тестируем:
  - First-hour reversal/continuation: знак первого часа (10:00 open->close) предсказывает
    остаток основной сессии (11:00 open -> 18:00 close)? Разворот или продолжение.
  - Overnight через часовые: close вчерашней сессии -> open первого часа (контроль к дневному).

no look-ahead: сигнал по первому часу t используется для входа на 11:00 open того же дня.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

CACHE = os.path.join(os.path.dirname(__file__), "cache", "ohlc")


def tstat(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 3 or x.std(ddof=1) == 0: return 0.0
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def ann_sharpe(x, ppy=250):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 3 or x.std(ddof=1) == 0: return 0.0
    return float(x.mean() / x.std(ddof=1) * np.sqrt(ppy))


def load_h1(t):
    df = pd.read_csv(os.path.join(CACHE, f"{t}_1h.csv"), index_col=0)
    df["begin"] = pd.to_datetime(df["begin"])
    df = df[df[["open", "high", "low", "close"]].gt(0).all(axis=1)]
    df["date"] = df["begin"].dt.normalize()
    df["h"] = df["begin"].dt.hour
    return df


def day_blocks(df):
    """Для каждого дня: open первого часа (10:00), close первого часа,
    open 11:00 (вход в остаток), close 18:00 (последний час основной сессии)."""
    rows = []
    for d, g in df.groupby("date"):
        g = g.sort_values("h")
        main = g[(g["h"] >= 10) & (g["h"] <= 18)]
        if len(main) < 4:
            continue
        h10 = main[main["h"] == 10]
        h11 = main[main["h"] == 11]
        h18 = main[main["h"] == 18]
        if len(h10) == 0 or len(h11) == 0 or len(h18) == 0:
            continue
        rows.append({
            "date": d,
            "fh_open": h10["open"].iloc[0],
            "fh_close": h10["close"].iloc[0],
            "rest_open": h11["open"].iloc[0],
            "rest_close": h18["close"].iloc[0],
        })
    b = pd.DataFrame(rows).set_index("date").sort_index()
    b["fh_ret"] = b["fh_close"] / b["fh_open"] - 1.0      # первый час
    b["rest_ret"] = b["rest_close"] / b["rest_open"] - 1.0  # остаток основной сессии
    return b


def main():
    tickers = [os.path.basename(f).replace("_1h.csv", "")
               for f in sorted(glob.glob(os.path.join(CACHE, "*_1h.csv")))]
    print(f"{len(tickers)} tickers with 1h data\n")

    cost = 0.0005
    pooled_fh, pooled_rest = [], []
    cont_pnl, rev_pnl = [], []  # pooled per-day per-ticker
    rows = []
    for t in tickers:
        try:
            b = day_blocks(load_h1(t))
        except Exception as e:  # noqa: BLE001
            print(f"  {t}: skip ({e})"); continue
        if len(b) < 100:
            continue
        fh = b["fh_ret"].values
        rest = b["rest_ret"].values
        # corr знака первого часа с остатком
        corr = np.corrcoef(fh, rest)[0, 1]
        # continuation: позиция = sign(fh) на остаток; reversal: -sign(fh)
        cont = np.sign(fh) * rest
        rows.append({"ticker": t, "n": len(b),
                     "fh_mean": fh.mean(), "rest_mean": rest.mean(),
                     "corr": corr,
                     "cont_mean": cont.mean(), "cont_t": tstat(cont)})
        pooled_fh.append(fh); pooled_rest.append(rest)
        cont_pnl.append(cont)
    res = pd.DataFrame(rows).set_index("ticker")
    print("Per-ticker: first-hour vs rest-of-day")
    print(f"{'tkr':6} {'fh_mean%':>9} {'rest_mean%':>10} {'corr':>7} {'cont_mean%':>11} {'cont_t':>7}")
    for t, r in res.sort_values("corr").iterrows():
        print(f"{t:6} {r['fh_mean']*100:9.4f} {r['rest_mean']*100:10.4f} {r['corr']:7.3f} {r['cont_mean']*100:11.4f} {r['cont_t']:7.2f}")

    allcont = np.concatenate(cont_pnl)
    allrev = -allcont
    avg_corr = res["corr"].mean()
    print(f"\nPOOLED avg corr(first-hour, rest) = {avg_corr:.3f}")
    print(f"  (corr>0 => continuation, corr<0 => intraday reversal)")
    print(f"CONTINUATION pooled (sign(fh)*rest, gross): mean={allcont.mean()*100:.4f}% t={tstat(allcont):.2f}")
    print(f"  net(cost={cost*100:.2f}%): mean={(allcont.mean()-cost)*100:.4f}%  Sharpe(gross)={ann_sharpe(allcont):.2f}")
    print(f"REVERSAL pooled (-sign(fh)*rest, gross): mean={allrev.mean()*100:.4f}% t={tstat(allrev):.2f}")
    print(f"  net(cost={cost*100:.2f}%): mean={(allrev.mean()-cost)*100:.4f}%")
    # split-sample на лучшем направлении
    best = allcont if abs(allcont.mean()) >= abs(allrev.mean()) else allrev
    lbl = "CONT" if best is allcont else "REV"
    half = len(best)//2
    print(f"\nBest direction = {lbl}; split-sample: H1 mean={best[:half].mean()*100:.4f}%(t={tstat(best[:half]):.1f}) "
          f"H2 mean={best[half:].mean()*100:.4f}%(t={tstat(best[half:]):.1f})")


if __name__ == "__main__":
    main()
