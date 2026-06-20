"""Honest walk-forward для st6-подобной парной стратегии на реальных данных MOEX.

Главное отличие от боевого st6: отбор пары (и при желании тюнинг порогов)
делается ТОЛЬКО на train-окне; торговля — на следующем test-окне теми же
зафиксированными парой/порогами. Скользящий walk-forward → агрегированный OOS.

Переиспользует чистое ядро st6.core (decide/rank_pairs/leg_quantities/trade_pnl),
ничего в боевых файлах не меняет.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pairsignal.st6.core import (  # noqa: E402
    Params, Position, Side, decide, leg_quantities, rank_pairs, trade_pnl,
)


@dataclass
class WFTrade:
    pair: str
    side: str
    entry_i: int
    exit_i: int
    bars: int
    net: float
    reason: str
    ret_on_notional: float  # net / notional ноги A (для Sharpe по сделкам)


@dataclass
class WFResult:
    trades: list[WFTrade] = field(default_factory=list)
    fold_log: list[dict] = field(default_factory=list)
    start_equity: float = 1_000_000.0
    end_equity: float = 1_000_000.0
    equity_path: list[float] = field(default_factory=list)
    test_bars: int = 0  # сколько test-баров суммарно прошли (для аннуализации)

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def total_net(self) -> float:
        return sum(t.net for t in self.trades)

    @property
    def return_pct(self) -> float:
        return 100.0 * (self.end_equity - self.start_equity) / self.start_equity

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return float("nan")
        return sum(1 for t in self.trades if t.net > 0) / len(self.trades)

    @property
    def max_drawdown_pct(self) -> float:
        if not self.equity_path:
            return 0.0
        eq = np.asarray(self.equity_path, dtype=float)
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak
        return float(-dd.min() * 100.0)

    @property
    def sharpe_pertrade(self) -> float:
        """Sharpe по доходностям сделок (на нотионал), не аннуализированный."""
        r = np.array([t.ret_on_notional for t in self.trades], dtype=float)
        if len(r) < 2 or r.std() == 0:
            return float("nan")
        return float(r.mean() / r.std() * np.sqrt(len(r)))

    def sharpe_annual(self, bars_per_year: float = 252.0) -> float:
        """Аннуализированный Sharpe по дневной equity-кривой OOS."""
        if len(self.equity_path) < 3:
            return float("nan")
        eq = np.asarray(self.equity_path, dtype=float)
        rets = np.diff(eq) / eq[:-1]
        if rets.std() == 0:
            return float("nan")
        return float(rets.mean() / rets.std() * np.sqrt(bars_per_year))


def _select_pair_on_train(train_prices: dict[str, np.ndarray], p: Params,
                          fixed_pair: tuple[str, str] | None,
                          bonferroni_n: int | None) -> tuple[str, str] | None:
    """Отбор пары на train. fixed_pair → её и возвращаем (без перебора).
    Иначе rank_pairs по train; bonferroni_n>0 → ужесточаем p-value порог."""
    if fixed_pair is not None:
        a, b = fixed_pair
        if a in train_prices and b in train_prices:
            return fixed_pair
        return None
    pp = p
    if bonferroni_n:
        # поправка Бонферрони: эффективный порог p делим на число тестов
        pp = Params(**{**p.__dict__})
        pp.select_max_pvalue = p.select_max_pvalue / bonferroni_n
    ranked = rank_pairs({t: s for t, s in train_prices.items()}, pp)
    if not ranked:
        return None
    best = ranked[0]
    return (best.a, best.b)


def walk_forward(prices: dict[str, list[float]], p: Params,
                 train_bars: int, test_bars: int, step: int,
                 start_equity: float = 1_000_000.0,
                 fixed_pair: tuple[str, str] | None = None,
                 bonferroni_n: int | None = None,
                 lots: dict[str, int] | None = None) -> WFResult:
    """
    Скользящий walk-forward. На каждом фолде:
      train = prices[t0 : t0+train_bars]  → отбираем пару (OOS-отбор)
      test  = prices[t0+train_bars : +test_bars] → торгуем зафиксированной парой,
              FSM получает полную историю до текущего бара (warmup из train),
              но открываем/считаем только сделки, чей вход попал в test-зону.
    Позиция, открытая в test-зоне, доводится до закрытия (может выйти за край фолда —
    допускаем 1 'хвост' внутри того же test-сегмента; следующий фолд стартует чисто).
    """
    lots = lots or {}
    arr = {t: np.asarray(v, dtype=float) for t, v in prices.items()}
    n = min(len(v) for v in arr.values())
    arr = {t: v[-n:] for t, v in arr.items()}

    res = WFResult(start_equity=start_equity, end_equity=start_equity)
    equity = start_equity
    warmup = max(p.beta_window, p.z_window, p.corr_window) + 1

    # equity снимаем на каждом test-баре (для DD/Sharpe по кривой)
    t0 = 0
    while t0 + train_bars + test_bars <= n:
        train_lo, train_hi = t0, t0 + train_bars
        test_lo, test_hi = train_hi, train_hi + test_bars

        train_prices = {t: v[train_lo:train_hi] for t, v in arr.items()}
        pair = _select_pair_on_train(train_prices, p, fixed_pair, bonferroni_n)
        if pair is None:
            res.fold_log.append({"test_lo": test_lo, "test_hi": test_hi,
                                 "pair": None, "trades": 0, "net": 0.0})
            res.equity_path.extend([equity] * (test_hi - test_lo))
            t0 += step
            continue

        ta, tb = pair
        a_full, b_full = arr[ta], arr[tb]
        lot_a, lot_b = lots.get(ta, 1), lots.get(tb, 1)

        pos = Position()
        fold_net = 0.0
        fold_trades = 0
        # идём по test-барам; FSM видит всю историю [0..i]
        for i in range(test_lo, test_hi):
            if i < warmup:
                res.equity_path.append(equity)
                continue
            wa, wb = a_full[:i + 1], b_full[:i + 1]
            sig = decide(wa, wb, pos, p)
            ca, cb = a_full[i], b_full[i]

            if pos.is_open:
                pos.bars_held += 1
                if sig.action == "EXIT":
                    units_a, units_b = pos.qty_a * lot_a, pos.qty_b * lot_b
                    net = trade_pnl(pos.side, pos.entry_a, ca, pos.entry_b, cb,
                                    units_a, units_b, p)
                    notional = pos.entry_a * units_a
                    equity += net
                    fold_net += net
                    fold_trades += 1
                    res.trades.append(WFTrade(
                        pair=f"{ta}/{tb}", side=pos.side.name,
                        entry_i=i - pos.bars_held, exit_i=i, bars=pos.bars_held,
                        net=net, reason=sig.reason.value,
                        ret_on_notional=net / notional if notional else 0.0,
                    ))
                    pos = Position()
            else:
                if sig.action in ("ENTER_LONG", "ENTER_SHORT"):
                    qa, qb = leg_quantities(equity, ca, cb, sig.beta, lot_a, lot_b, p)
                    if qa > 0 and qb > 0:
                        pos = Position(
                            side=Side.LONG_SPREAD if sig.action == "ENTER_LONG"
                            else Side.SHORT_SPREAD,
                            entry_z=sig.z, beta=sig.beta, bars_held=0,
                            qty_a=qa, qty_b=qb, entry_a=ca, entry_b=cb,
                        )
            res.equity_path.append(equity)

        # принудительно закрываем хвост по последней цене test-сегмента
        if pos.is_open:
            i = test_hi - 1
            ca, cb = a_full[i], b_full[i]
            units_a, units_b = pos.qty_a * lot_a, pos.qty_b * lot_b
            net = trade_pnl(pos.side, pos.entry_a, ca, pos.entry_b, cb,
                            units_a, units_b, p)
            notional = pos.entry_a * units_a
            equity += net
            fold_net += net
            fold_trades += 1
            res.trades.append(WFTrade(
                pair=f"{ta}/{tb}", side=pos.side.name,
                entry_i=i - pos.bars_held, exit_i=i, bars=pos.bars_held,
                net=net, reason="eof", ret_on_notional=net / notional if notional else 0.0,
            ))

        res.test_bars += (test_hi - test_lo)
        res.fold_log.append({"test_lo": test_lo, "test_hi": test_hi,
                             "pair": f"{ta}/{tb}", "trades": fold_trades,
                             "net": round(fold_net, 0)})
        t0 += step

    res.end_equity = equity
    return res
