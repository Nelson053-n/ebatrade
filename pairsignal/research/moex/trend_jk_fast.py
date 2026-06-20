"""Быстрый jackknife по тикерам на ФИКСИРОВАННЫХ параметрах (без per-window refit).

Отвечает на вопрос «не тащат ли результат 1-2 тикера»: дроп-один на полной выборке
и по половинам. Параметры — победитель IS (передаются аргументом)."""
from __future__ import annotations
import sys
import numpy as np
from trend_core import load_closes, sig_donchian, sig_ma_cross, basket_metrics, d

interval = sys.argv[1] if len(sys.argv) > 1 else "1h"
# params: 'donchian:en,ex' or 'ma:f,s'
spec = sys.argv[2] if len(sys.argv) > 2 else "donchian:200,100"
fee, slip = 0.0005, 0.0003
kind, ps = spec.split(":")
nums = [int(x) for x in ps.split(",")]
if kind == "donchian":
    sig = lambda c: sig_donchian(c, nums[0], nums[1], False)
else:
    sig = lambda c: sig_ma_cross(c, nums[1], nums[0], False)

closes = load_closes(interval)
def bsh(df): return basket_metrics(df, sig, fee, slip, interval)["sharpe"]
def bret(df): return basket_metrics(df, sig, fee, slip, interval)["ret_pct"]

mid = len(closes) // 2
h1, h2 = closes.iloc[:mid], closes.iloc[mid:]
print(f"=== FAST JACKKNIFE {interval} {spec} (fixed params, no refit) ===")
print(f"FULL Sharpe {bsh(closes):.2f} ret {bret(closes):+.1f}%  | "
      f"H1[{d(int(closes.index[0]))}..{d(int(h1.index[-1]))}] Sharpe {bsh(h1):.2f} ret {bret(h1):+.1f}%  | "
      f"H2[{d(int(h2.index[0]))}..{d(int(closes.index[-1]))}] Sharpe {bsh(h2):.2f} ret {bret(h2):+.1f}%")
res = sorted(((t, bsh(closes.drop(columns=[t]))) for t in closes.columns), key=lambda x: x[1])
arr = np.array([s for _, s in res])
print(f"drop-one FULL Sharpe: min {arr.min():.2f} ({res[0][0]}), max {arr.max():.2f} ({res[-1][0]}), median {np.median(arr):.2f}")
print("most influential (drop lowers Sharpe most):", ", ".join(f"{t}->{s:.2f}" for t, s in res[:6]))
# H2-only drop-one: is even the weak H2 carried by one name?
res2 = sorted(((t, bsh(h2.drop(columns=[t]))) for t in h2.columns), key=lambda x: x[1])
arr2 = np.array([s for _, s in res2])
print(f"drop-one H2 Sharpe: min {arr2.min():.2f} ({res2[0][0]}), max {arr2.max():.2f} ({res2[-1][0]}), median {np.median(arr2):.2f}")
