from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

from jev_trader.features import BAR_MS
from jev_trader.jev import JevClient, JevConfigError, JevSettings, build_questions
from jev_trader import markets
from jev_trader.grid import sl_key, tp_key
from jev_trader.markets import fetch_closed_bars
from jev_trader.paper import PaperEngine
from jev_trader.shadow import _connect, run_once, shadow_run_id
from tests.test_causality import synthetic

ROOT = Path(__file__).resolve().parents[1]


def _okx_public_feed(server_ms: int):
    latest = server_ms // BAR_MS * BAR_MS - BAR_MS
    frame = synthetic(n=1500).drop(columns="taker_buy_volume")
    frame["ts"] += latest - int(frame.iloc[-1]["ts"])
    rows = frame.to_dict("records")

    def fetch(path, params):
        if path == "/api/v5/public/time":
            return {"code": "0", "data": [{"ts": str(server_ms)}]}
        assert path == "/api/v5/market/history-candles"
        assert params["instId"] == "BTC-USDT-SWAP"
        cursor = int(params.get("after", latest + BAR_MS))
        prior = [row for row in reversed(rows) if row["ts"] < cursor]
        page = [[
            str(int(row["ts"])), str(row["open"]), str(row["high"]),
            str(row["low"]), str(row["close"]), str(row["volume"]),
            str(row["volume"]), str(row["volume"] * row["close"]), "1",
        ] for row in prior[:300]]
        return {"code": "0", "data": page}

    return fetch


def test_okx_market_only_uses_separate_exchange_database_and_never_infers(tmp_path):
    cfg = {
        "data_dir": str(tmp_path),
        "symbols": ["BTC/USDT:USDT"],
        "grid": {"horizon_bars": 16, "tp": [0.01], "sl": [0.01]},
        "costs": {},
    }
    fetch = _okx_public_feed(1_800_000_000_000)
    output = run_once(["BTC/USDT:USDT"], cfg, exchange="okx", fetch_json=fetch)
    assert output[0]["status"] == "market_ok", output
    assert output[0]["feature_version"] == "ohlcv-features-v1"
    assert output[0]["state_version"] == "jev-state-v4"
    assert (tmp_path / "shadow_okx.sqlite").is_file()
    assert not (tmp_path / "shadow.sqlite").exists()

    def forbidden(*_args):
        raise AssertionError("paid path must not be reached")

    with pytest.raises(JevConfigError, match="requires prompt_version jev-ohlcv-v4"):
        run_once(["BTCUSDT"], cfg, exchange="okx", live_jev=True,
                 max_requests=1, max_usd=0.01, fetch_json=forbidden)


@pytest.mark.parametrize("exchange", ["binance", "okx"])
def test_run_once_default_transport_routes_only_to_selected_exchange(
    tmp_path, monkeypatch, exchange,
):
    server_ms = 1_800_000_000_000
    latest = server_ms // BAR_MS * BAR_MS - BAR_MS
    called = []
    okx_feed = _okx_public_feed(server_ms)

    def binance(path, params):
        called.append(("binance", path))
        if path == "/fapi/v1/time":
            return {"serverTime": server_ms}
        klines = []
        for i, ts in enumerate(range(latest - 1499 * BAR_MS, latest + 1, BAR_MS)):
            price = 100 + i / 1000
            klines.append([
                ts, str(price), str(price + 0.2), str(price - 0.2),
                str(price + 0.01), str(10 + i % 19), ts + BAR_MS - 1,
                "1000", 1, str(4 + i % 5), "400", "0",
            ])
        return klines

    def okx(path, params):
        called.append(("okx", path))
        return okx_feed(path, params)

    monkeypatch.setattr(markets, "_binance_json", binance)
    monkeypatch.setattr(markets, "_okx_json", okx)
    monkeypatch.setattr(markets, "_OKX_LAST_REQUEST", 0.0)
    cfg = {
        "data_dir": str(tmp_path / exchange),
        "symbols": ["BTC/USDT:USDT"],
        "grid": {"horizon_bars": 16, "tp": [0.01], "sl": [0.01]},
        "costs": {},
    }

    result = run_once(["BTC/USDT:USDT"], cfg, exchange=exchange)
    assert result[0]["status"] == "market_ok"
    assert called
    assert {venue for venue, _ in called} == {exchange}
    expected_prefix = "/api/v5/" if exchange == "okx" else "/fapi/v1/"
    assert all(path.startswith(expected_prefix) for _, path in called)


def test_dryrun_configs_are_exchange_and_state_isolated():
    binance = json.loads((ROOT / "config/freqtrade.binance.example.json").read_text())
    okx = json.loads((ROOT / "config/freqtrade.okx.example.json").read_text())
    assert binance["dry_run"] is okx["dry_run"] is True
    assert binance["trading_mode"] == okx["trading_mode"] == "futures"
    assert binance["margin_mode"] == okx["margin_mode"] == "isolated"
    assert binance["exchange"]["name"] == "binance"
    assert okx["exchange"]["name"] == "okx"
    assert okx["exchange"]["password"] == ""
    assert all(okx["exchange"][key] == "" for key in ("key", "secret"))
    assert binance["db_url"] != okx["db_url"]
    assert binance["logfile"] != okx["logfile"]
    assert binance["exchange"]["pair_whitelist"]
    assert okx["exchange"]["pair_whitelist"]
    assert not binance["api_server"]["enabled"]
    assert not okx["api_server"]["enabled"]


def test_market_database_refuses_cross_exchange_reuse(tmp_path):
    path = tmp_path / "venue.sqlite"
    db = _connect(path, "binance")
    db.close()
    with pytest.raises(RuntimeError, match="belongs to 'binance'"):
        _connect(path, "okx")


def test_v4_prompt_is_not_reused_with_binance_v3_state(tmp_path):
    cfg = {
        "data_dir": str(tmp_path),
        "grid": {"horizon_bars": 16, "tp": [0.01], "sl": [0.01]},
        "costs": {},
    }

    def forbidden(*_args):
        raise AssertionError("candidate protocol must stop before transport")

    client = JevClient(JevSettings(
        "unused", "https://fixture.invalid/decisions", "fixture/model",
        prompt_version="jev-ohlcv-v4",
    ), transport=forbidden)
    with pytest.raises(JevConfigError, match="registered v3 feature/state contract"):
        run_once(["BTCUSDT"], cfg, live_jev=True, max_requests=1,
                 max_usd=0.01, client=client, fetch_json=forbidden)
    assert not (tmp_path / "shadow.sqlite").exists()


def test_okx_v4_smoke_uses_approved_contract_and_bounded_request(tmp_path, monkeypatch):
    from jev_trader.jev import PROMPT_VERSION_OHLCV_V4

    grid = {"horizon_bars": 16, "tp": [0.01], "sl": [0.01]}
    cfg = {
        "data_dir": str(tmp_path), "grid": grid,
        "costs": {"taker_fee": 0.0005, "half_spread": 0.00005,
                  "slippage": 0.0001, "funding_per_8h": 0.0001},
    }
    server_ms = (int(time.time() * 1000) // BAR_MS) * BAR_MS + 1_000
    market_feed = _okx_public_feed(server_ms)
    calls = []

    def transport(_url, _headers, body):
        payload = json.loads(body)
        calls.append(payload)
        answers = {}
        for key, question in payload["questions"].items():
            if question["type"] == "choice":
                choices = question["criteria"]
                answers[key] = {"type": "choice", "probabilities": {
                    name: 1 / len(choices) for name in choices
                }}
            else:
                probability = {
                    tp_key("long", 0.01, 0.01): 0.7,
                    sl_key("long", 0.01, 0.01): 0.2,
                    tp_key("short", 0.01, 0.01): 0.2,
                    sl_key("short", 0.01, 0.01): 0.6,
                }[key]
                answers[key] = {"type": "noul", "noul": probability}
        return 200, json.dumps({
            "model": "fixture/jev-v4", "answers": answers,
            "usage": {"input_tokens": 100, "output_tokens": 100, "cost": 0.001},
        }).encode()

    client = JevClient(JevSettings(
        "fixture-key", "https://fixture.invalid/decisions", "~typesafe/jev-latest",
        prompt_version=PROMPT_VERSION_OHLCV_V4,
        cache_dir=tmp_path / "cache",
    ), transport=transport)
    committed = []
    record_signal = PaperEngine.record_signal

    def observe_committed_signal(self, symbol, ts, state_hash, answers, **kwargs):
        result = record_signal(self, symbol, ts, state_hash, answers, **kwargs)
        db_path = self.db.execute("PRAGMA database_list").fetchone()[2]
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as observer:
            allowed = observer.execute(
                "SELECT entry_allowed FROM diagnostic_signals "
                "WHERE run_id=? AND symbol=? AND ts=?",
                (self.run_id, symbol, ts),
            ).fetchone()[0]
            pending = observer.execute(
                "SELECT COUNT(*) FROM dry_orders WHERE run_id=? AND status='pending'",
                (self.run_id,),
            ).fetchone()[0]
        committed.append((allowed, pending))
        return result

    monkeypatch.setattr(PaperEngine, "record_signal", observe_committed_signal)
    result = run_once(
        ["BTC/USDT:USDT"], cfg, exchange="okx", live_jev=True,
        dry_run_trades=True, max_requests=1, max_usd=1.0,
        fetch_json=market_feed, client=client,
        data_provenance="synthetic_fixture",
    )

    assert result[0]["status"] == "valid", result
    assert result[0]["signal"]["status"] == "candidate"
    assert result[0]["signal"]["entry_allowed"] is False
    assert result[0]["signal"]["acceptance_reason"] == "synthetic_fixture"
    assert committed == [(0, 0)]
    assert len(calls) == 1
    assert calls[0]["model"] == "~typesafe/jev-latest"
    assert calls[0]["state"]["_jev_protocol"]["prompt_version"] == PROMPT_VERSION_OHLCV_V4
    assert calls[0]["state"]["market"]["exchange"] == "okx"
    assert calls[0]["state"]["market"]["state_version"] == "jev-state-v4"
    assert len(calls[0]["questions"]) == len(build_questions(grid)) == 6
    assert (tmp_path / "shadow_okx_synthetic.sqlite").is_file()
    assert not (tmp_path / "shadow_okx.sqlite").exists()
    assert not (tmp_path / "shadow.sqlite").exists()
    with sqlite3.connect(tmp_path / "shadow_okx_synthetic.sqlite") as db:
        run_id = db.execute("SELECT DISTINCT run_id FROM observations").fetchone()[0]
        assert run_id == shadow_run_id(
            client.settings, grid, "okx", ["BTC/USDT:USDT"],
            data_provenance="synthetic_fixture",
        )
        assert run_id != shadow_run_id(
            client.settings, grid, "okx", ["BTC/USDT:USDT"],
        )
        assert db.execute(
            "SELECT provenance,excluded_from_research FROM signal_run_provenance "
            "WHERE run_id=?", (run_id,),
        ).fetchone() == ("synthetic_fixture", 1)
        assert db.execute(
            "SELECT run_id,symbol,ts FROM research_sample_exclusions"
        ).fetchone() == (run_id, "BTC/USDT:USDT", result[0]["ts"])


def test_okx_contract_uses_volccy_not_contract_count():
    server_ms = 1_800_000_000_000
    latest = server_ms // BAR_MS * BAR_MS - BAR_MS

    def fetch(_path, params):
        cursor = int(params.get("after", latest + BAR_MS))
        count = 300 if cursor == latest + BAR_MS else 100
        return {"code": "0", "data": [
            [str(cursor - i * BAR_MS), "100", "101", "99", "100.5",
             "999", "0.25", "25.125", "1"]
            for i in range(1, count + 1)
        ]}

    bars = fetch_closed_bars("okx", "BTC-USDT-SWAP", server_ms, fetch)
    assert bars.iloc[-1]["volume"] == pytest.approx(0.25)
    assert "taker_buy_volume" not in bars


@pytest.mark.parametrize(("outcome", "expected_reason"), [
    ("tp", "take_profit"),
    ("sl", "stop_loss"),
    ("both", "ambiguous_15m_stop_first"),
    ("timeout", "timeout_16_bars"),
])
def test_okx_v4_local_paper_signal_fills_and_exits(tmp_path, outcome, expected_reason):
    pair = "BTC/USDT:USDT"
    grid = {"horizon_bars": 16, "tp": [0.01], "sl": [0.01]}
    costs = {
        "taker_fee": 0.0005, "half_spread": 0.00005,
        "slippage": 0.0001, "funding_per_8h": 0.0001,
    }
    signal_ts = 1_800_000_000_000
    db = sqlite3.connect(tmp_path / f"{outcome}.sqlite")
    db.execute("CREATE TABLE market_identity (identity_key TEXT, identity_value TEXT)")
    db.execute("INSERT INTO market_identity VALUES ('exchange', 'okx')")
    db.execute("""CREATE TABLE signal_runs (
        run_id TEXT, pair TEXT, exchange TEXT, market_type TEXT,
        feature_version TEXT, state_version TEXT, model_id TEXT,
        prompt_version TEXT, PRIMARY KEY (run_id, pair))""")
    db.execute("INSERT INTO signal_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (
        "fixture-okx-v4", pair, "okx", "usdt-perpetual",
        "ohlcv-features-v1", "jev-state-v4", "fixture/model", "jev-ohlcv-v4",
    ))
    engine = PaperEngine(db, "fixture-okx-v4", grid, costs)
    answers = {
        tp_key("long", 0.01, 0.01): 0.7,
        sl_key("long", 0.01, 0.01): 0.2,
        tp_key("short", 0.01, 0.01): 0.2,
        sl_key("short", 0.01, 0.01): 0.6,
    }
    signal = engine.record_signal(
        pair, signal_ts, "offline-okx-state", answers, dry_run=True,
    )
    assert signal["entry_allowed"] is True

    if outcome == "timeout":
        bars = [{"ts": signal_ts + i * BAR_MS, "open": 100, "high": 100.2,
                 "low": 99.8, "close": 100.1} for i in range(1, 17)]
    else:
        high, low = {
            "tp": (101.2, 99.8),
            "sl": (100.2, 98.8),
            "both": (101.2, 98.8),
        }[outcome]
        bars = [{"ts": signal_ts + BAR_MS, "open": 100, "high": high,
                 "low": low, "close": 100}]
    events = engine.advance(pair, bars)

    assert [event["type"] for event in events] == ["fill", "exit", "feedback"]
    assert events[1]["reason"] == expected_reason
    assert db.execute("SELECT status FROM dry_positions").fetchone() == ("closed",)
    assert db.execute("SELECT reason FROM dry_feedback").fetchone() == (
        f"paper_exit:{expected_reason}",
    )
    assert db.execute("SELECT cost_usd FROM dry_exits").fetchone()[0] > 0
