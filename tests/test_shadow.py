from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

from jev_trader.features import BAR_MS
from jev_trader.grid import sl_key, tp_key
from jev_trader.jev import JevClient, JevSettings, build_questions
import jev_trader.shadow as shadow_module
from jev_trader.shadow import closed_klines, normalize_symbol, run_once
from tests.test_causality import synthetic


GRID = {
    "horizon_bars": 16,
    "tp": [0.005, 0.010, 0.015, 0.020, 0.030],
    "sl": [0.005, 0.0075, 0.010, 0.015],
}


def market(n=1500):
    frame = synthetic(n=n)
    klines = [
        [int(row.ts), str(row.open), str(row.high), str(row.low),
         str(row.close), str(row.volume), int(row.ts) + BAR_MS - 1,
         "0", 1, str(row.taker_buy_volume), "0", "0"]
        for row in frame.itertuples(index=False)
    ]
    server_ms = int(frame.iloc[-1]["ts"]) + BAR_MS + 2_000

    def fetch(path, params):
        if path == "/fapi/v1/time":
            return {"serverTime": server_ms}
        assert path == "/fapi/v1/klines"
        assert params["interval"] == "15m"
        return klines

    return fetch, klines, server_ms


def config(tmp_path):
    return {"data_dir": str(tmp_path), "symbols": ["BTC/USDT:USDT"],
            "grid": GRID, "costs": {
                "taker_fee": 0.0005, "half_spread": 0.00005,
                "slippage": 0.0001, "funding_per_8h": 0.0001,
            }}


def valid_response(questions):
    answers = {}
    for key, question in questions.items():
        if question["type"] == "choice":
            keys = list(question["criteria"])
            answers[key] = {"type": "choice", "probabilities": {
                choice: 1 / len(keys) for choice in keys
            }}
        else:
            answers[key] = {"type": "noul", "noul": 0.2}
    return {"model": "jev-test", "answers": answers,
            "usage": {"input_tokens": 10, "output_tokens": 20, "cost": 0.001}}


def test_shadow_accepts_any_usdt_pair_and_ignores_partial_candle(tmp_path):
    assert normalize_symbol("ethusdt") == ("ETH/USDT:USDT", "ETHUSDT")
    assert normalize_symbol("1000SHIB/USDT:USDT") == (
        "1000SHIB/USDT:USDT", "1000SHIBUSDT"
    )
    with pytest.raises(ValueError):
        normalize_symbol("BTC/USDC:USDC")
    fetch, klines, server_ms = market()
    partial = klines[-1].copy()
    partial[0] += BAR_MS
    partial[6] += BAR_MS
    frame = closed_klines(
        "ETHUSDT", server_ms,
        lambda path, params: [*fetch(path, params), partial],
    )
    assert len(frame) == 1500
    assert int(frame.iloc[-1].ts) == server_ms - 2_000 - BAR_MS


def test_v4_run_identity_binds_exchange_pair_model_and_prompt(tmp_path):
    from jev_trader.jev import JevSettings
    from jev_trader.shadow import shadow_run_id

    questions = build_questions(config(tmp_path)["grid"])
    grid = config(tmp_path)["grid"]
    v4 = JevSettings(
        "not-used", "https://example.test/decisions", "fixture/model-a",
        prompt_version="jev-ohlcv-v4",
    )
    first = shadow_run_id(v4, grid, "okx", ["BTC/USDT:USDT"])
    assert first == shadow_run_id(v4, grid, "okx", ["BTC/USDT:USDT"])
    assert first != shadow_run_id(v4, grid, "binance", ["BTC/USDT:USDT"])
    assert first != shadow_run_id(v4, grid, "okx", ["ETH/USDT:USDT"])

    other_model = JevSettings(
        "not-used", "https://example.test/decisions", "fixture/model-b",
        prompt_version="jev-ohlcv-v4",
    )
    assert first != shadow_run_id(other_model, grid, "okx", ["BTC/USDT:USDT"])
    assert len(questions) == 82


def test_v4_run_identity_binds_prompt_contract_content(tmp_path, monkeypatch):
    grid = config(tmp_path)["grid"]
    settings = JevSettings(
        "not-used", "https://example.test/decisions", "fixture/model",
        prompt_version="jev-ohlcv-v4",
    )
    monkeypatch.setattr(
        shadow_module, "prompt_contract_v4", lambda _horizon: "contract A",
        raising=False,
    )
    first = shadow_module.shadow_run_id(settings, grid, "okx", ["BTC/USDT:USDT"])
    monkeypatch.setattr(
        shadow_module, "prompt_contract_v4", lambda _horizon: "contract B",
    )
    second = shadow_module.shadow_run_id(settings, grid, "okx", ["BTC/USDT:USDT"])
    assert first != second


def test_binance_v3_run_identity_keeps_existing_spend_ledger_key():
    settings = JevSettings(
        "not-used", "https://example.test/decisions", "jev-test",
        prompt_version="v3",
    )
    assert shadow_module.shadow_run_id(settings, GRID, "binance", ["BTCUSDT"]) == (
        "9b0999e01115d01e"
    )


def test_shadow_real_data_shape_budget_and_symbol_dedup(tmp_path):
    fetch, _, _ = market()
    calls = []

    def transport(_url, _headers, body):
        payload = json.loads(body)
        calls.append(payload)
        return 200, json.dumps(valid_response(payload["questions"])).encode()

    client = JevClient(
        JevSettings("test", "https://example.test/decisions", "jev-test",
                    cache_dir=tmp_path / "cache"), transport=transport,
    )
    cfg = config(tmp_path)
    rows = run_once(["ETHUSDT", "SOL/USDT:USDT"], cfg, live_jev=True,
                    max_requests=2, max_usd=0.003, fetch_json=fetch, client=client)
    assert [row["status"] for row in rows] == ["valid", "valid"]
    assert [call["state"]["market"]["symbol"] for call in calls] == [
        "ETH/USDT:USDT", "SOL/USDT:USDT"
    ]
    again = run_once(["SOLUSDT", "ETHUSDT"], cfg, live_jev=True,
                     max_requests=2, max_usd=0.003, fetch_json=fetch, client=client)
    assert [row["status"] for row in again] == ["inference_blocked", "inference_blocked"]
    assert len(calls) == 2
    stopped = run_once(["ADAUSDT"], cfg, live_jev=True, max_requests=2,
                       max_usd=0.003, fetch_json=fetch, client=client)
    assert len(calls) == 2
    assert stopped[0]["status"] == "inference_blocked"
    assert stopped[0]["reason"] == "budget_exhausted"
    with sqlite3.connect(tmp_path / "shadow.sqlite") as db:
        assert db.execute("SELECT count(*) FROM observations WHERE status='valid'").fetchone()[0] == 2


def test_shadow_market_only_requires_no_key_or_jev_call(tmp_path):
    fetch, _, _ = market()
    rows = run_once(["ADAUSDT"], config(tmp_path), fetch_json=fetch)
    assert rows[0]["status"] == "market_ok"


def test_shadow_bad_pair_does_not_block_other_pairs(tmp_path):
    fetch, klines, _ = market()

    def mixed_fetch(path, params):
        if path == "/fapi/v1/time":
            return fetch(path, params)
        return klines[:20] if params["symbol"] == "BADUSDT" else klines

    rows = run_once(["BADUSDT", "ETHUSDT"], config(tmp_path), fetch_json=mixed_fetch)
    assert [row["status"] for row in rows] == ["market_error", "market_ok"]


def test_shadow_replays_paid_response_after_crash_without_resending(tmp_path, monkeypatch):
    fetch, _, _ = market()
    calls = []

    def transport(_url, _headers, body):
        calls.append(1)
        if len(calls) == 1:
            return 429, b'{"error":{"message":"try again"}}'
        payload = json.loads(body)
        return 200, json.dumps(valid_response(payload["questions"])).encode()

    client = JevClient(
        JevSettings("test", "https://example.test/decisions", "jev-test",
                    cache_dir=tmp_path / "cache", retry_delay_s=0),
        transport=transport,
    )
    real_record = shadow_module._record

    def crash_before_record(*_args, **_kwargs):
        raise RuntimeError("simulated crash after paid response")

    monkeypatch.setattr(shadow_module, "_record", crash_before_record)
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_once(["ETHUSDT"], config(tmp_path), live_jev=True,
                 max_requests=2, max_usd=0.003, fetch_json=fetch, client=client)
    assert calls == [1, 1]
    monkeypatch.setattr(shadow_module, "_record", real_record)
    rows = run_once(["ETHUSDT"], config(tmp_path), live_jev=True,
                    max_requests=2, max_usd=0.003, fetch_json=fetch, client=client)
    assert rows[0]["status"] == "inference_blocked"
    assert rows[0]["reason"] == "budget_exhausted"
    assert calls == [1, 1]
    with sqlite3.connect(tmp_path / "shadow.sqlite") as db:
        assert db.execute("SELECT status FROM observations").fetchone()[0] == "valid"


def test_shadow_stale_candle_never_calls_jev(tmp_path):
    fetch, klines, server_ms = market()
    calls = []

    def stale_fetch(path, params):
        if path == "/fapi/v1/time":
            return {"serverTime": server_ms + 70_000}
        return klines

    client = JevClient(
        JevSettings("test", "https://example.test/decisions", "jev-test",
                    cache_dir=tmp_path / "cache"),
        transport=lambda *_: calls.append(1),
    )
    rows = run_once(["ETHUSDT"], config(tmp_path), live_jev=True,
                    max_requests=1, max_usd=0.001, fetch_json=stale_fetch, client=client)
    assert rows[0]["status"] == "stale"
    assert calls == []


def test_shadow_health_stop_blocks_before_transport(tmp_path):
    fetch, _, _ = market()
    calls = []

    def transport(_url, _headers, body):
        calls.append(1)
        payload = json.loads(body)
        return 200, json.dumps(valid_response(payload["questions"])).encode()

    client = JevClient(
        JevSettings("test", "https://example.test/decisions", "jev-test",
                    cache_dir=tmp_path / "cache"), transport=transport,
    )
    cfg = config(tmp_path)
    run_once(["ETHUSDT"], cfg, live_jev=True, max_requests=20,
             max_usd=0.02, fetch_json=fetch, client=client)
    with sqlite3.connect(tmp_path / "shadow.sqlite") as db:
        run_id = db.execute("SELECT run_id FROM observations LIMIT 1").fetchone()[0]
        db.executemany(
            "INSERT INTO observations VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL, NULL)",
            [(run_id, "FAKE/USDT:USDT", i, i, "schema_error") for i in range(10)],
        )
    rows = run_once(["SOLUSDT"], cfg, live_jev=True, max_requests=20,
                    max_usd=0.02, fetch_json=fetch, client=client)
    assert calls == [1]
    assert rows[0]["status"] == "inference_blocked"
    assert rows[0]["reason"] == "health_stop"


def test_shadow_health_stop_cancels_remaining_symbols_in_same_cycle(tmp_path):
    fetch, _, _ = market()
    calls = []

    def bad_transport(_url, _headers, body):
        calls.append(1)
        payload = json.loads(body)
        answer = valid_response(payload["questions"])
        for key, value in answer["answers"].items():
            if key.startswith("p_"):
                value["noul"] = 0.6
        return 200, json.dumps(answer).encode()

    client = JevClient(
        JevSettings("test", "https://example.test/decisions", "jev-test",
                    cache_dir=tmp_path / "cache"), transport=bad_transport,
    )
    symbols = [f"COIN{i}USDT" for i in range(11)]
    rows = run_once(symbols, config(tmp_path), live_jev=True,
                    max_requests=11, max_usd=0.02, fetch_json=fetch, client=client)
    assert len(calls) == 10
    assert rows[-1]["status"] == "inference_blocked"
    assert rows[-1]["reason"] == "health_stop"


def test_same_cycle_health_stop_revokes_prior_symbol_before_freqtrade_confirmation(
    tmp_path, monkeypatch,
):
    pytest.importorskip("freqtrade.strategy")
    import pandas as pd
    from jev_trader.paper import PaperEngine

    fetch, klines, _ = market()
    calls = []

    def transport(_url, _headers, body):
        calls.append(json.loads(body))
        answers = valid_response(calls[-1]["questions"])
        if len(calls) in {1, 2}:
            answers["answers"][tp_key("long", 0.01, 0.01)]["noul"] = 0.7
            answers["answers"][sl_key("long", 0.01, 0.01)]["noul"] = 0.2
            answers["answers"][tp_key("short", 0.01, 0.01)]["noul"] = 0.2
            answers["answers"][sl_key("short", 0.01, 0.01)]["noul"] = 0.6
        elif len(calls) == 3:
            for key, value in answers["answers"].items():
                if key.startswith("p_"):
                    value["noul"] = 0.6
        return 200, json.dumps(answers).encode()

    client = JevClient(
        JevSettings("test", "https://example.test/decisions", "jev-test",
                    cache_dir=tmp_path / "cache"), transport=transport,
    )
    cfg = config(tmp_path)
    run_once(["SOLUSDT"], cfg, live_jev=True, dry_run_trades=True,
             max_requests=20, max_usd=0.05, fetch_json=fetch, client=client)
    t0 = int(klines[-1][0])
    t1 = t0 + BAR_MS
    klines.append([t1, "100", "102", "99.5", "101", "12",
                   t1 + BAR_MS - 1, "0", 1, "6", "0", "0"])
    second_fetch = lambda path, params: (
        {"serverTime": t1 + BAR_MS + 2_000} if path == "/fapi/v1/time" else klines
    )
    with sqlite3.connect(tmp_path / "shadow.sqlite") as db:
        run_id = db.execute("SELECT run_id FROM observations LIMIT 1").fetchone()[0]
        identity = db.execute(
            "SELECT exchange, market_type, feature_version, state_version, model_id, "
            "prompt_version FROM signal_runs WHERE run_id=? AND pair=?",
            (run_id, "SOL/USDT:USDT"),
        ).fetchone()
        assert identity is not None
        db.executemany(
            "INSERT INTO observations VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL, NULL)",
            [(run_id, f"BAD{i}/USDT:USDT", i, i, "schema_error") for i in range(7)],
        )

    monkeypatch.setenv("JEV_SIGNAL_DB", str(tmp_path / "shadow.sqlite"))
    monkeypatch.setenv("JEV_SIGNAL_RUN_ID", run_id)
    for name, value in zip((
        "JEV_SIGNAL_EXCHANGE", "JEV_SIGNAL_MARKET_TYPE",
        "JEV_SIGNAL_FEATURE_VERSION", "JEV_SIGNAL_STATE_VERSION",
        "JEV_SIGNAL_MODEL_ID", "JEV_SIGNAL_PROMPT_VERSION",
    ), identity):
        monkeypatch.setenv(name, value)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "freqtrade/strategies"))
    from JevSignalStore import JevSignalStore

    strategy = JevSignalStore({"exchange": {"name": "binance"}})
    authorized = {}
    record_signal = PaperEngine.record_signal

    def capture_authorized(self, symbol, ts, state_hash, answers, **kwargs):
        result = record_signal(self, symbol, ts, state_hash, answers, **kwargs)
        if symbol == "SOL/USDT:USDT":
            frame = pd.DataFrame({"date": [pd.to_datetime(ts, unit="ms", utc=True)]})
            signal = strategy.populate_entry_trend(frame, {"pair": symbol})
            assert signal.loc[0, "enter_long"] == 1
            authorized["ts"] = ts
            authorized["tag"] = signal.loc[0, "enter_tag"]
        return result

    monkeypatch.setattr(PaperEngine, "record_signal", capture_authorized)
    rows = run_once(["SOLUSDT", "ETHUSDT"], cfg, live_jev=True,
                    dry_run_trades=True, max_requests=20, max_usd=0.05,
                    fetch_json=second_fetch, client=client)
    assert [row["status"] for row in rows] == ["valid", "schema_error"]
    with sqlite3.connect(tmp_path / "shadow.sqlite") as db:
        assert db.execute(
            "SELECT entry_allowed, acceptance_reason FROM diagnostic_signals "
            "WHERE run_id=? AND symbol='SOL/USDT:USDT' AND ts=?",
            (run_id, t0),
        ).fetchone() == (1, "accepted")
        assert db.execute(
            "SELECT status FROM dry_orders WHERE run_id=? AND symbol='SOL/USDT:USDT' "
            "AND signal_ts=?", (run_id, t0),
        ).fetchone() == ("filled",)
        assert db.execute(
            "SELECT count(*) FROM dry_exits WHERE order_id IN "
            "(SELECT order_id FROM dry_orders WHERE run_id=? AND signal_ts=?)",
            (run_id, t0),
        ).fetchone() == (1,)
        assert db.execute(
            "SELECT entry_allowed, acceptance_reason FROM diagnostic_signals "
            "WHERE run_id=? AND symbol='SOL/USDT:USDT' AND ts=?",
            (run_id, authorized["ts"]),
        ).fetchone() == (0, "health_stop")
        assert db.execute(
            "SELECT status, reason FROM dry_orders WHERE run_id=? "
            "AND symbol='SOL/USDT:USDT' AND signal_ts=?", (run_id, t1),
        ).fetchone() == ("cancelled", "health_stop")
    now = pd.to_datetime(authorized["ts"] + BAR_MS, unit="ms", utc=True).to_pydatetime()
    assert strategy.confirm_trade_entry(
        pair="SOL/USDT:USDT", order_type="limit", amount=1, rate=100,
        time_in_force="GTC", current_time=now, entry_tag=authorized["tag"],
        side="long",
    ) is False


def test_shadow_local_paper_path_is_non_btc_and_restart_idempotent(tmp_path):
    fetch, klines, server_ms = market()
    calls = []

    def transport(_url, _headers, body):
        payload = json.loads(body)
        calls.append(payload)
        answer = valid_response(payload["questions"])
        answer["answers"][tp_key("long", 0.01, 0.01)]["noul"] = 0.7
        answer["answers"][sl_key("long", 0.01, 0.01)]["noul"] = 0.2
        answer["answers"][tp_key("short", 0.01, 0.01)]["noul"] = 0.2
        answer["answers"][sl_key("short", 0.01, 0.01)]["noul"] = 0.6
        return 200, json.dumps(answer).encode()

    client = JevClient(
        JevSettings("test", "https://example.test/decisions", "jev-test",
                    cache_dir=tmp_path / "cache"), transport=transport,
    )
    rows = run_once(["SOLUSDT"], config(tmp_path), live_jev=True,
                    dry_run_trades=True, max_requests=3, max_usd=0.004,
                    fetch_json=fetch, client=client)
    assert rows[0]["signal"]["side"] == "long"
    with sqlite3.connect(tmp_path / "shadow.sqlite") as db:
        order = db.execute("SELECT status, reason FROM dry_orders").fetchone()
        assert order == ("pending", "next_bar_open")

    latest_ts = int(klines[-1][0]) + BAR_MS
    klines.append([latest_ts, "100", "102", "99.5", "101", "12",
                   latest_ts + BAR_MS - 1, "0", 1, "6", "0", "0"])
    second_fetch = lambda path, params: (
        {"serverTime": latest_ts + BAR_MS + 2_000} if path == "/fapi/v1/time" else klines
    )
    rows = run_once(["SOLUSDT"], config(tmp_path), live_jev=True,
                    dry_run_trades=True, max_requests=3, max_usd=0.004,
                    fetch_json=second_fetch, client=client)
    with sqlite3.connect(tmp_path / "shadow.sqlite") as db:
        assert db.execute("SELECT status FROM dry_orders").fetchone()[0] == "filled"
        assert db.execute("SELECT reason FROM dry_exits").fetchone()[0] == "take_profit"
        assert db.execute("SELECT status FROM dry_feedback").fetchone()[0] == "closed"
        order_count = db.execute("SELECT count(*) FROM dry_orders").fetchone()[0]
    duplicate = run_once(["SOLUSDT"], config(tmp_path), live_jev=True,
                         dry_run_trades=True, max_requests=3, max_usd=0.004,
                         fetch_json=second_fetch, client=client)
    assert duplicate[0]["status"] == "already_recorded"
    with sqlite3.connect(tmp_path / "shadow.sqlite") as db:
        assert db.execute("SELECT count(*) FROM dry_orders").fetchone()[0] == order_count
        assert db.execute("SELECT count(*) FROM dry_fills").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM dry_exits").fetchone()[0] == 1


def test_final_successful_request_exhausting_budget_revokes_candidate_and_order(tmp_path):
    fetch, _, _ = market()

    def transport(_url, _headers, body):
        payload = json.loads(body)
        answer = valid_response(payload["questions"])
        answer["answers"][tp_key("long", 0.01, 0.01)]["noul"] = 0.7
        answer["answers"][sl_key("long", 0.01, 0.01)]["noul"] = 0.2
        answer["answers"][tp_key("short", 0.01, 0.01)]["noul"] = 0.2
        answer["answers"][sl_key("short", 0.01, 0.01)]["noul"] = 0.6
        return 200, json.dumps(answer).encode()

    client = JevClient(
        JevSettings("test", "https://example.test/decisions", "jev-test",
                    cache_dir=tmp_path / "cache"), transport=transport,
    )
    rows = run_once(["SOLUSDT"], config(tmp_path), live_jev=True,
                    dry_run_trades=True, max_requests=1, max_usd=0.002,
                    fetch_json=fetch, client=client)
    assert rows[0]["status"] == "valid"
    with sqlite3.connect(tmp_path / "shadow.sqlite") as db:
        assert db.execute(
            "SELECT entry_allowed, acceptance_reason FROM diagnostic_signals"
        ).fetchone() == (0, "budget_exhausted")
        assert db.execute("SELECT COUNT(*) FROM dry_orders").fetchone()[0] == 0


def test_startup_health_stop_preserves_historical_signal_permission(tmp_path):
    fetch, _, _ = market()

    def transport(_url, _headers, body):
        payload = json.loads(body)
        answer = valid_response(payload["questions"])
        answer["answers"][tp_key("long", 0.01, 0.01)]["noul"] = 0.7
        answer["answers"][sl_key("long", 0.01, 0.01)]["noul"] = 0.2
        answer["answers"][tp_key("short", 0.01, 0.01)]["noul"] = 0.2
        answer["answers"][sl_key("short", 0.01, 0.01)]["noul"] = 0.6
        return 200, json.dumps(answer).encode()

    client = JevClient(
        JevSettings("test", "https://example.test/decisions", "jev-test",
                    cache_dir=tmp_path / "cache"), transport=transport,
    )
    cfg = config(tmp_path)
    initial = run_once(["SOLUSDT"], cfg, live_jev=True, dry_run_trades=True,
                       max_requests=20, max_usd=0.05,
                       fetch_json=fetch, client=client)
    ts = initial[0]["ts"]
    with sqlite3.connect(tmp_path / "shadow.sqlite") as db:
        run_id = db.execute("SELECT run_id FROM observations LIMIT 1").fetchone()[0]
        current = db.execute(
            "SELECT * FROM diagnostic_signals WHERE run_id=? AND ts=?", (run_id, ts),
        ).fetchone()
        history = list(current)
        history[2] -= 2 * BAR_MS
        db.execute("INSERT INTO diagnostic_signals VALUES (" + ",".join("?" * 15) + ")", history)
        db.executemany(
            "INSERT INTO observations VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL, NULL)",
            [(run_id, f"BAD{i}/USDT:USDT", i, i, "schema_error") for i in range(10)],
        )

    rows = run_once(["SOLUSDT"], cfg, live_jev=True, dry_run_trades=True,
                    max_requests=20, max_usd=0.05,
                    fetch_json=fetch, client=client)
    assert rows[0]["reason"] == "health_stop"
    with sqlite3.connect(tmp_path / "shadow.sqlite") as db:
        assert db.execute(
            "SELECT entry_allowed, acceptance_reason FROM diagnostic_signals "
            "WHERE run_id=? AND symbol='SOL/USDT:USDT' AND ts=?", (run_id, ts),
        ).fetchone() == (0, "health_stop")
        assert db.execute(
            "SELECT entry_allowed, acceptance_reason FROM diagnostic_signals "
            "WHERE run_id=? AND symbol='SOL/USDT:USDT' AND ts=?",
            (run_id, ts - 2 * BAR_MS),
        ).fetchone() == (1, "accepted")
        assert db.execute("SELECT status, reason FROM dry_orders").fetchone() == (
            "cancelled", "health_stop"
        )


def test_shadow_invalid_probabilities_fail_closed_without_paper_order(tmp_path):
    fetch, _, _ = market()

    def transport(_url, _headers, body):
        payload = json.loads(body)
        answer = valid_response(payload["questions"])
        answer["answers"][tp_key("long", 0.01, 0.01)]["noul"] = 0.8
        answer["answers"][sl_key("long", 0.01, 0.01)]["noul"] = 0.4
        return 200, json.dumps(answer).encode()

    client = JevClient(
        JevSettings("test", "https://example.test/decisions", "jev-test",
                    cache_dir=tmp_path / "cache"), transport=transport,
    )
    rows = run_once(["ADAUSDT"], config(tmp_path), live_jev=True,
                    dry_run_trades=True, max_requests=1, max_usd=0.002,
                    fetch_json=fetch, client=client)
    assert rows[0]["status"] == "schema_error"
    with sqlite3.connect(tmp_path / "shadow.sqlite") as db:
        assert db.execute("SELECT status, reason FROM diagnostic_signals").fetchone() == (
            "no_signal", "schema_error"
        )
        assert db.execute("SELECT count(*) FROM dry_orders").fetchone()[0] == 0


@pytest.mark.parametrize("block", ["health", "budget", "stale"])
def test_existing_valid_observation_loses_entry_permission_when_blocked(tmp_path, block):
    fetch, _, server_ms = market()
    calls = []

    def transport(_url, _headers, body):
        calls.append(1)
        payload = json.loads(body)
        answer = valid_response(payload["questions"])
        answer["answers"][tp_key("long", 0.01, 0.01)]["noul"] = 0.7
        answer["answers"][sl_key("long", 0.01, 0.01)]["noul"] = 0.2
        answer["answers"][tp_key("short", 0.01, 0.01)]["noul"] = 0.2
        answer["answers"][sl_key("short", 0.01, 0.01)]["noul"] = 0.6
        return 200, json.dumps(answer).encode()

    client = JevClient(
        JevSettings("test", "https://example.test/decisions", "jev-test",
                    cache_dir=tmp_path / "cache"), transport=transport,
    )
    cfg = config(tmp_path)
    cap = 1 if block == "budget" else 3
    initial = run_once(["SOLUSDT"], cfg, live_jev=True, dry_run_trades=True,
                       max_requests=cap, max_usd=0.004,
                       fetch_json=fetch, client=client)
    ts = initial[0]["ts"]
    with sqlite3.connect(tmp_path / "shadow.sqlite") as db:
        run_id = db.execute("SELECT run_id FROM observations LIMIT 1").fetchone()[0]
        db.executemany(
            "INSERT INTO observations VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL, NULL)",
            [(run_id, f"BAD{i}/USDT:USDT", i, i, "schema_error") for i in range(9)],
    ) if block == "health" else None

    second_fetch = fetch
    if block == "stale":
        second_fetch = lambda path, params: (
            {"serverTime": server_ms + 70_000} if path == "/fapi/v1/time" else fetch(path, params)
        )
    rows = run_once(["SOLUSDT"], cfg, live_jev=True, dry_run_trades=True,
                    max_requests=cap, max_usd=0.004,
                    fetch_json=second_fetch, client=client)
    assert calls == [1]
    reason = {"health": "health_stop", "budget": "budget_exhausted",
              "stale": "stale_candle"}[block]
    assert rows[0]["reason"] == reason
    with sqlite3.connect(tmp_path / "shadow.sqlite") as db:
        signal = db.execute(
            "SELECT status, entry_allowed, acceptance_reason FROM diagnostic_signals "
            "WHERE run_id=? AND symbol='SOL/USDT:USDT' AND ts=?", (run_id, ts),
        ).fetchone()
        assert signal == ("candidate", 0, reason)
        expected_orders = 0 if block == "budget" else 1
        assert db.execute("SELECT COUNT(*) FROM dry_orders").fetchone()[0] == expected_orders
        assert db.execute("SELECT count(*) FROM dry_orders WHERE status='pending'").fetchone()[0] == 0


@pytest.mark.parametrize("block", ["health", "budget"])
def test_existing_paper_position_advances_when_inference_is_blocked(tmp_path, block):
    fetch, klines, _ = market()
    calls = []

    def transport(_url, _headers, body):
        calls.append(1)
        payload = json.loads(body)
        answer = valid_response(payload["questions"])
        answer["answers"][tp_key("long", 0.01, 0.01)]["noul"] = 0.7
        answer["answers"][sl_key("long", 0.01, 0.01)]["noul"] = 0.2
        return 200, json.dumps(answer).encode()

    client = JevClient(
        JevSettings("test", "https://example.test/decisions", "jev-test",
                    cache_dir=tmp_path / "cache"), transport=transport,
    )
    cfg = config(tmp_path)
    cap = 2 if block == "budget" else 4
    run_once(["SOLUSDT"], cfg, live_jev=True, dry_run_trades=True,
             max_requests=cap, max_usd=0.004, fetch_json=fetch, client=client)
    latest_ts = int(klines[-1][0]) + BAR_MS
    klines.append([latest_ts, "100", "100.5", "99.5", "100", "12",
                   latest_ts + BAR_MS - 1, "0", 1, "6", "0", "0"])
    second_fetch = lambda path, params: ({"serverTime": latest_ts + BAR_MS + 2_000}
                                         if path == "/fapi/v1/time" else klines)
    healthy = run_once(["SOLUSDT"], cfg, live_jev=True, dry_run_trades=True,
                       max_requests=cap, max_usd=0.004,
                       fetch_json=second_fetch, client=client)
    assert healthy[0]["status"] == "valid"
    latest_ts += BAR_MS
    klines.append([latest_ts, "100", "102", "99.5", "101", "12",
                   latest_ts + BAR_MS - 1, "0", 1, "6", "0", "0"])
    blocked_fetch = lambda path, params: ({"serverTime": latest_ts + BAR_MS + 2_000}
                                         if path == "/fapi/v1/time" else klines)
    if block == "health":
        with sqlite3.connect(tmp_path / "shadow.sqlite") as db:
            run_id = db.execute("SELECT run_id FROM observations LIMIT 1").fetchone()[0]
            db.executemany(
                "INSERT INTO observations VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL, NULL)",
                [(run_id, f"BAD{i}/USDT:USDT", i, i, "schema_error") for i in range(9)],
            )

    rows = run_once(["SOLUSDT"], cfg, live_jev=True, dry_run_trades=True,
                    max_requests=cap, max_usd=0.004,
                    fetch_json=blocked_fetch, client=client)
    assert calls == [1, 1]
    assert rows[0]["status"] in {"inference_blocked", "budget_stopped"}
    assert rows[0]["reason"] in {"health_stop", "budget_exhausted"}
    with sqlite3.connect(tmp_path / "shadow.sqlite") as db:
        assert db.execute("SELECT status FROM dry_orders WHERE signal_ts=(SELECT min(signal_ts) FROM dry_orders)").fetchone()[0] == "filled"
        assert db.execute("SELECT reason FROM dry_exits").fetchone()[0] == "take_profit"
        assert db.execute("SELECT count(*) FROM dry_orders WHERE status='pending'").fetchone()[0] == 0
        expected_orders = 1 if block == "budget" else 2
        assert db.execute("SELECT COUNT(*) FROM dry_orders").fetchone()[0] == expected_orders
