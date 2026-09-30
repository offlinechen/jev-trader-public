from __future__ import annotations

import sqlite3
from pytest import approx

import pandas as pd

from jev_trader.features import BAR_MS
from jev_trader.grid import sl_key, tp_key
from jev_trader.paper import PaperEngine, diagnostic_signal


GRID = {"horizon_bars": 16, "tp": [0.01], "sl": [0.01]}
COSTS = {
    "taker_fee": 0.0005, "half_spread": 0.00005,
    "slippage": 0.0001, "funding_per_8h": 0.0001,
}


def answers(long=(0.7, 0.2), short=(0.2, 0.6)):
    result = {}
    for side, (p_tp, p_sl) in (("long", long), ("short", short)):
        result[tp_key(side, 0.01, 0.01)] = p_tp
        result[sl_key(side, 0.01, 0.01)] = p_sl
    return result


def bar(ts, *, open_=100.0, high=100.2, low=99.8, close=100.0):
    return {"ts": ts, "open": open_, "high": high, "low": low, "close": close}


def test_fixed_diagnostic_signal_is_auditable_and_fails_closed():
    signal = diagnostic_signal(answers(), GRID)
    assert signal["status"] == "candidate"
    assert signal["side"] == "long"
    assert signal["policy_id"] == "fixed_tp1_sl1_raw_direction_v1"
    assert signal["score_long"] == approx(0.5)

    bad = answers(long=(0.8, 0.4))
    rejected = diagnostic_signal(bad, GRID)
    assert rejected["status"] == "no_signal"
    assert rejected["reason"] == "invalid_fixed_cell_probability"


def test_paper_order_fills_exits_and_feedback_are_idempotent(tmp_path):
    db = sqlite3.connect(tmp_path / "paper.sqlite")
    engine = PaperEngine(db, "run-1", GRID, COSTS)
    signal_ts = 1_700_000_000_000
    signal = engine.record_signal("SOL/USDT:USDT", signal_ts, "state-hash", answers(), dry_run=True)
    assert signal["side"] == "long"

    events = engine.advance("SOL/USDT:USDT", [bar(signal_ts + BAR_MS, high=101.2)])
    assert [event["type"] for event in events] == ["fill", "exit", "feedback"]
    assert events[0]["price"] == 100.0
    assert events[1]["reason"] == "take_profit"
    assert events[2]["status"] == "closed"

    restarted = PaperEngine(sqlite3.connect(tmp_path / "paper.sqlite"), "run-1", GRID, COSTS)
    restarted.record_signal("SOL/USDT:USDT", signal_ts, "state-hash", answers(), dry_run=True)
    restarted.advance("SOL/USDT:USDT", [bar(signal_ts + BAR_MS, high=101.2)])
    with sqlite3.connect(tmp_path / "paper.sqlite") as check:
        assert check.execute("SELECT count(*) FROM diagnostic_signals").fetchone()[0] == 1
        assert check.execute("SELECT count(*) FROM dry_orders").fetchone()[0] == 1
        assert check.execute("SELECT count(*) FROM dry_fills").fetchone()[0] == 1
        assert check.execute("SELECT count(*) FROM dry_exits").fetchone()[0] == 1
        assert check.execute("SELECT count(*) FROM dry_feedback").fetchone()[0] == 1


def test_paper_engine_enforces_global_pending_position_and_exposure_caps():
    db = sqlite3.connect(":memory:")
    engine = PaperEngine(db, "run-1", GRID, COSTS)
    first_ts = 1_700_000_000_000
    engine.record_signal("SOL/USDT:USDT", first_ts, "a", answers(), dry_run=True)
    blocked = engine.record_signal(
        "ADA/USDT:USDT", first_ts, "b", answers(), dry_run=True
    )
    assert blocked["order_reason"] == "single_position_limit"

    engine.advance("SOL/USDT:USDT", [bar(first_ts + BAR_MS)])
    position = db.execute("SELECT quantity, notional_usd, status FROM dry_positions").fetchone()
    assert position[2] == "open"
    assert position[1] <= 100.0
    assert position[0] * 100.0 * 0.01 <= 1.0
    blocked_open = engine.record_signal(
        "ADA/USDT:USDT", first_ts + BAR_MS, "c", answers(), dry_run=True
    )
    assert blocked_open["order_reason"] == "single_position_limit"


def test_paper_timeout_is_at_fixed_horizon():
    db = sqlite3.connect(":memory:")
    engine = PaperEngine(db, "run-1", GRID, COSTS)
    signal_ts = 1_700_000_000_000
    engine.record_signal("SOL/USDT:USDT", signal_ts, "h", answers(), dry_run=True)
    bars = [bar(signal_ts + i * BAR_MS) for i in range(1, 17)]
    events = engine.advance("SOL/USDT:USDT", bars)
    exit_event = next(event for event in events if event["type"] == "exit")
    assert exit_event["reason"] == "timeout_16_bars"
    assert exit_event["ts"] == signal_ts + 16 * BAR_MS


def test_legacy_raw_candidate_is_not_executable_after_schema_migration():
    db = sqlite3.connect(":memory:")
    db.execute("""CREATE TABLE diagnostic_signals (
        run_id TEXT NOT NULL, symbol TEXT NOT NULL, ts INTEGER NOT NULL,
        status TEXT NOT NULL, side TEXT, tp REAL, sl REAL,
        score_long REAL, score_short REAL, policy_id TEXT,
        state_hash TEXT NOT NULL, reason TEXT NOT NULL,
        created_at_ms INTEGER NOT NULL, PRIMARY KEY(run_id, symbol, ts))""")
    ts = 1_700_000_000_000
    db.execute("INSERT INTO diagnostic_signals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (
        "run-1", "SOL/USDT:USDT", ts, "candidate", "long", 0.01, 0.01,
        0.5, -0.4, "fixed_tp1_sl1_raw_direction_v1", "old-hash",
        "fixed_cell_raw_tp_vs_sl", 0,
    ))
    engine = PaperEngine(db, "run-1", GRID, COSTS)
    signal = engine.record_signal(
        "SOL/USDT:USDT", ts, "new-hash", answers(), dry_run=True
    )

    assert signal["status"] == "candidate"
    assert signal["entry_allowed"] is False
    assert db.execute("SELECT count(*) FROM dry_orders").fetchone()[0] == 0
