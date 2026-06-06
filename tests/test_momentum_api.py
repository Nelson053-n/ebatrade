"""Тесты API-эндпоинтов momentum (st3). Без сети — только state/tests/роутинг."""
from __future__ import annotations

import json

from fastapi.testclient import TestClient

from pairsignal import api
from pairsignal.api import app

client = TestClient(app)


def test_state_no_file(monkeypatch, tmp_path):
    """Нет файла состояния → running:false, без падения."""
    monkeypatch.setattr(api, "_MOM_STATE", tmp_path / "nope.json")
    r = client.get("/momentum/state")
    assert r.status_code == 200
    assert r.json() == {"running": False}


def test_state_reads_portfolio(monkeypatch, tmp_path):
    """Со свежим файлом состояния — корректный парс equity/позиций."""
    from pairsignal.momentum import PaperPortfolio, save_state

    p = PaperPortfolio(start_equity=1000.0, fee=0.0006, slippage=0.0002)
    p.rebalance({"BTC/USDT:USDT": 0.5, "ETH/USDT:USDT": -0.5},
                {"BTC/USDT:USDT": 60000.0, "ETH/USDT:USDT": 3000.0}, ts=1_700_000_000_000)
    path = tmp_path / "fwd.json"
    save_state(path, p, {"lookback": 48, "holding": 24, "k": 3, "long_short": True})
    monkeypatch.setattr(api, "_MOM_STATE", path)

    r = client.get("/momentum/state")
    assert r.status_code == 200
    d = r.json()
    assert "equity" in d and d["equity"] > 0
    assert d["longs"] == ["BTC"]
    assert d["shorts"] == ["ETH"]
    assert d["meta"]["lookback"] == 48
    assert len(d["rebalances"]) == 1


def test_state_handles_corrupt_file(monkeypatch, tmp_path):
    """Битый JSON → running:false (load_state ловит исключение)."""
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json")
    monkeypatch.setattr(api, "_MOM_STATE", bad)
    r = client.get("/momentum/state")
    assert r.status_code == 200
    assert r.json() == {"running": False}


def test_st1_st2_not_broken():
    """Регрессия: парные слоты st1/st2 продолжают работать."""
    assert client.get("/state?slot=1").status_code == 200
    assert client.get("/state?slot=2").status_code == 200
    # дашборд содержит вкладку st3
    assert 'data-slot="3"' in client.get("/").text


def test_momentum_endpoints_registered():
    """Все 4 momentum-эндпоинта зарегистрированы."""
    paths = {r.path for r in app.routes}
    for p in ("/momentum/state", "/momentum/backtest", "/momentum/compare", "/momentum/tests"):
        assert p in paths


def test_backtest_cache_used(monkeypatch):
    """backtest кэширует результат по ключу параметров (тяжёлую функцию не дёргаем дважды)."""
    calls = {"n": 0}

    def fake(top, days, timeframe, k, fee, slippage):
        calls["n"] += 1
        return {"windows": [], "equity": [], "total_return_pct": 0.0, "wins": 0, "n": 0}

    monkeypatch.setattr(api, "_run_backtest_mom", fake)
    api._MOM_CACHE.clear()
    q = "top=6&days=30&timeframe=1h&k=2&fee=0.0006&slippage=0.0002"
    client.get("/momentum/backtest?" + q)
    client.get("/momentum/backtest?" + q)  # второй раз — из кэша
    assert calls["n"] == 1


def test_daily_no_file(monkeypatch, tmp_path):
    """Нет состояния → running:false, пустые дни, без падения."""
    monkeypatch.setattr(api, "_MOM_STATE", tmp_path / "nope.json")
    d = client.get("/momentum/daily").json()
    assert d["running"] is False
    assert d["days"] == []


def test_daily_aggregates_by_day(monkeypatch, tmp_path):
    """Ребалансы агрегируются по дням: дата, equity конца дня, изменение, число."""
    from pairsignal.momentum import PaperPortfolio, save_state

    p = PaperPortfolio(1000.0, 0.0006, 0.0002)
    # два ребаланса в один день UTC (2023-11-15 09:00 и 12:00)
    p.rebalance({"BTC/USDT:USDT": 1.0}, {"BTC/USDT:USDT": 100.0}, ts=1_700_038_800_000)
    p.rebalance({"ETH/USDT:USDT": 1.0}, {"ETH/USDT:USDT": 50.0}, ts=1_700_049_600_000)
    path = tmp_path / "fwd.json"
    save_state(path, p, {"lookback": 48, "holding": 24})
    monkeypatch.setattr(api, "_MOM_STATE", path)

    d = client.get("/momentum/daily").json()
    assert len(d["days"]) == 1            # оба ребаланса в одну дату
    assert d["days"][0]["rebals"] == 2
    assert "change_pct" in d["days"][0]
    assert "equity" in d["text"]


def test_save_state_produces_valid_json(tmp_path):
    """Доп. санити: save_state даёт валидный JSON со структурой meta+portfolio."""
    from pairsignal.momentum import PaperPortfolio, save_state

    p = PaperPortfolio(1000.0, 0.0006, 0.0002)
    path = tmp_path / "s.json"
    save_state(path, p, {"k": 3})
    d = json.loads(path.read_text())
    assert "meta" in d and "portfolio" in d
