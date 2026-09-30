"""使用已安装 Freqtrade 引擎的可选端到端验证。 / Optional end-to-end Freqtrade proof."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("freqtrade.strategy")

ROOT = Path(__file__).resolve().parents[1]
PAIR = "SOL/USDT:USDT"
RUN_ID = "offline-smoke-run"


def test_freqtrade_reads_signal_store_and_backtests_next_open_and_exits(tmp_path, monkeypatch):
    userdir, datadir = tmp_path / "user_data", tmp_path / "candles"
    result_dir = tmp_path / "results"
    userdir.mkdir()
    candle_dir = datadir / "futures"
    candle_dir.mkdir(parents=True)
    dates = pd.date_range("2025-01-01", periods=80, freq="15min", tz="UTC")
    frame = pd.DataFrame({
        "date": dates, "open": 100.0, "high": 100.5,
        "low": 99.5, "close": 100.0, "volume": 10.0,
    })
    frame.loc[22, ["open", "high", "low", "close"]] = [100.0, 102.0, 99.5, 101.5]
    frame.loc[62, ["open", "high", "low", "close"]] = [100.0, 100.5, 98.5, 99.0]
    frame.to_feather(candle_dir / "SOL_USDT_USDT-15m-futures.feather")

    signal_db = tmp_path / "signals.sqlite"
    with sqlite3.connect(signal_db) as db:
        db.execute("""CREATE TABLE diagnostic_signals (
            run_id TEXT, symbol TEXT, ts INTEGER, status TEXT, side TEXT,
            tp REAL, sl REAL, score_long REAL, score_short REAL, policy_id TEXT,
            state_hash TEXT, reason TEXT, created_at_ms INTEGER,
            entry_allowed INTEGER NOT NULL DEFAULT 0, acceptance_reason TEXT NOT NULL DEFAULT '',
            PRIMARY KEY(run_id, symbol, ts))""")
        db.execute("CREATE TABLE market_identity (identity_key TEXT, identity_value TEXT)")
        db.execute("INSERT INTO market_identity VALUES ('exchange', 'binance')")
        db.execute("""CREATE TABLE signal_runs (
            run_id TEXT, pair TEXT, exchange TEXT, market_type TEXT,
            feature_version TEXT, state_version TEXT, model_id TEXT,
            prompt_version TEXT, PRIMARY KEY(run_id, pair))""")
        db.execute("INSERT INTO signal_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (
            RUN_ID, PAIR, "binance", "usdt-perpetual", "binance-features-v3",
            "jev-state-v3", "fixture/binance-v3", "v3",
        ))
        db.execute("INSERT INTO diagnostic_signals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (
            RUN_ID, PAIR, int(dates[20].timestamp() * 1000), "candidate", "long",
            0.01, 0.01, 0.5, -0.4, "fixed_tp1_sl1_raw_direction_v1",
            "offline-hash", "fixed_cell_raw_tp_vs_sl", 0, 1, "accepted",
        ))
        db.execute("INSERT INTO diagnostic_signals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (
            RUN_ID, PAIR, int(dates[30].timestamp() * 1000), "candidate", "long",
            0.01, 0.01, 0.5, -0.4, "fixed_tp1_sl1_raw_direction_v1",
            "offline-revoked-hash", "fixed_cell_raw_tp_vs_sl", 0, 0, "health_stop",
        ))
        db.execute("INSERT INTO diagnostic_signals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (
            RUN_ID, PAIR, int(dates[40].timestamp() * 1000), "candidate", "long",
            0.01, 0.01, 0.5, -0.4, "fixed_tp1_sl1_raw_direction_v1",
            "offline-timeout-hash", "fixed_cell_raw_tp_vs_sl", 0, 1, "accepted",
        ))
        db.execute("INSERT INTO diagnostic_signals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (
            RUN_ID, PAIR, int(dates[60].timestamp() * 1000), "candidate", "long",
            0.01, 0.01, 0.5, -0.4, "fixed_tp1_sl1_raw_direction_v1",
            "offline-stop-hash", "fixed_cell_raw_tp_vs_sl", 0, 1, "accepted",
        ))

    env = os.environ.copy()
    env.update(JEV_SIGNAL_DB=str(signal_db), JEV_SIGNAL_RUN_ID=RUN_ID,
               JEV_SIGNAL_EXCHANGE="binance", JEV_SIGNAL_MARKET_TYPE="usdt-perpetual",
               JEV_SIGNAL_FEATURE_VERSION="binance-features-v3",
               JEV_SIGNAL_STATE_VERSION="jev-state-v3",
               JEV_SIGNAL_MODEL_ID="fixture/binance-v3",
               JEV_SIGNAL_PROMPT_VERSION="v3")
    monkeypatch.setenv("JEV_SIGNAL_DB", str(signal_db))
    monkeypatch.setenv("JEV_SIGNAL_RUN_ID", RUN_ID)
    monkeypatch.setenv("JEV_SIGNAL_EXCHANGE", "binance")
    monkeypatch.setenv("JEV_SIGNAL_MARKET_TYPE", "usdt-perpetual")
    monkeypatch.setenv("JEV_SIGNAL_FEATURE_VERSION", "binance-features-v3")
    monkeypatch.setenv("JEV_SIGNAL_STATE_VERSION", "jev-state-v3")
    monkeypatch.setenv("JEV_SIGNAL_MODEL_ID", "fixture/binance-v3")
    monkeypatch.setenv("JEV_SIGNAL_PROMPT_VERSION", "v3")
    sys.path.insert(0, str(ROOT / "freqtrade/strategies"))
    from JevSignalStore import JevSignalStore

    strategy = JevSignalStore({"exchange": {"name": "binance"}})
    populated = strategy.populate_entry_trend(frame.copy(), {"pair": PAIR})
    assert populated.loc[20, "enter_long"] == 1
    assert populated.loc[30, "enter_long"] == 0
    result_dir.mkdir()
    freqtrade = shutil.which("freqtrade")
    command = ([freqtrade] if freqtrade else [sys.executable, "-m", "freqtrade"])
    command = [
        *command, "backtesting",
        "--config", str(ROOT / "config/freqtrade.binance.example.json"),
        "--userdir", str(userdir), "--datadir", str(datadir),
        "--strategy", "JevSignalStore",
        "--strategy-path", str(ROOT / "freqtrade/strategies"),
        "--timeframe", "15m", "--data-format-ohlcv", "feather",
        "--pairs", PAIR, "--fee", "0.0005", "--export", "trades",
        "--backtest-directory", str(result_dir),
    ]
    result = subprocess.run(command, cwd=ROOT, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=90)
    assert result.returncode == 0, result.stdout
    artifacts = list(result_dir.glob("*.zip"))
    assert len(artifacts) == 1, result.stdout
    with zipfile.ZipFile(artifacts[0]) as artifact:
        payload = json.loads(artifact.read(next(name for name in artifact.namelist()
                                                if name.endswith(".json")
                                                and "config" not in name)))
    trades = payload["strategy"]["JevSignalStore"]["trades"]
    assert len(trades) == 3
    roi_trade, timeout_trade, stop_trade = trades
    assert roi_trade["open_rate"] == pytest.approx(100.0)
    assert pd.Timestamp(roi_trade["open_date"]) == dates[21]
    assert roi_trade["exit_reason"] == "roi"
    # Freqtrade ROI 扣双边费，净 1% 需价格涨幅超过 1%。 / Net 1% ROI needs >1% move after two fees.
    assert roi_trade["close_rate"] == pytest.approx(101.1)
    assert timeout_trade["open_rate"] == pytest.approx(100.0)
    assert timeout_trade["exit_reason"] == "timeout_16_bars"
    assert stop_trade["open_rate"] == pytest.approx(100.0)
    assert stop_trade["exit_reason"] == "stop_loss"
    assert stop_trade["initial_stop_loss_abs"] == pytest.approx(99.0)
    assert stop_trade["profit_ratio"] < -0.01


def test_freqtrade_rechecks_revoked_entry_permission_at_confirmation(tmp_path, monkeypatch):
    signal_db = tmp_path / "signals.sqlite"
    signal_ts = pd.Timestamp("2025-01-01T00:00:00Z")
    with sqlite3.connect(signal_db) as db:
        db.execute("""CREATE TABLE diagnostic_signals (
            run_id TEXT, symbol TEXT, ts INTEGER, status TEXT, side TEXT,
            policy_id TEXT, entry_allowed INTEGER,
            PRIMARY KEY(run_id, symbol, ts))""")
        db.execute("CREATE TABLE market_identity (identity_key TEXT, identity_value TEXT)")
        db.execute("INSERT INTO market_identity VALUES ('exchange', 'binance')")
        db.execute("""CREATE TABLE signal_runs (
            run_id TEXT, pair TEXT, exchange TEXT, market_type TEXT,
            feature_version TEXT, state_version TEXT, model_id TEXT,
            prompt_version TEXT, PRIMARY KEY(run_id, pair))""")
        db.execute("INSERT INTO signal_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (
            RUN_ID, PAIR, "binance", "usdt-perpetual", "binance-features-v3",
            "jev-state-v3", "fixture/binance-v3", "v3",
        ))
        db.execute(
            "INSERT INTO diagnostic_signals VALUES (?, ?, ?, ?, ?, ?, ?)",
            (RUN_ID, PAIR, int(signal_ts.timestamp() * 1000), "candidate",
             "long", "fixed_tp1_sl1_raw_direction_v1", 1),
        )

    monkeypatch.setenv("JEV_SIGNAL_DB", str(signal_db))
    monkeypatch.setenv("JEV_SIGNAL_RUN_ID", RUN_ID)
    monkeypatch.setenv("JEV_SIGNAL_EXCHANGE", "binance")
    monkeypatch.setenv("JEV_SIGNAL_MARKET_TYPE", "usdt-perpetual")
    monkeypatch.setenv("JEV_SIGNAL_FEATURE_VERSION", "binance-features-v3")
    monkeypatch.setenv("JEV_SIGNAL_STATE_VERSION", "jev-state-v3")
    monkeypatch.setenv("JEV_SIGNAL_MODEL_ID", "fixture/binance-v3")
    monkeypatch.setenv("JEV_SIGNAL_PROMPT_VERSION", "v3")
    sys.path.insert(0, str(ROOT / "freqtrade/strategies"))
    from JevSignalStore import JevSignalStore

    strategy = JevSignalStore({"exchange": {"name": "binance"}})
    frame = pd.DataFrame({"date": [signal_ts], "close": [100.0]})
    populated = strategy.populate_entry_trend(frame, {"pair": PAIR})
    assert populated.loc[0, "enter_long"] == 1
    tag = populated.loc[0, "enter_tag"]
    assert tag == f"jev_fixed_1pct:{int(signal_ts.timestamp() * 1000)}"

    confirmation = dict(
        pair=PAIR, order_type="limit", amount=1.0, rate=100.0,
        time_in_force="GTC", current_time=signal_ts + pd.Timedelta(minutes=15),
        entry_tag=tag, side="long",
    )
    assert strategy.confirm_trade_entry(**confirmation) is True
    monkeypatch.setenv("JEV_SIGNAL_PROMPT_VERSION", "jev-ohlcv-v4")
    assert strategy.confirm_trade_entry(**confirmation) is False
    monkeypatch.setenv("JEV_SIGNAL_PROMPT_VERSION", "v3")
    assert strategy.leverage(
        PAIR, signal_ts.to_pydatetime(), 100, 5, 10, tag, "long"
    ) == 1.0
    strategy.config["exchange"] = {"name": "okx"}
    assert strategy.confirm_trade_entry(**confirmation) is False
    assert strategy.populate_entry_trend(frame.copy(), {"pair": PAIR}).loc[0, "enter_long"] == 0
    strategy.config["exchange"] = {"name": "binance"}
    with sqlite3.connect(signal_db) as db:
        db.execute("UPDATE market_identity SET identity_value='okx' WHERE identity_key='exchange'")
    assert strategy.confirm_trade_entry(**confirmation) is False
    assert strategy.populate_entry_trend(frame.copy(), {"pair": PAIR}).loc[0, "enter_long"] == 0
    with sqlite3.connect(signal_db) as db:
        db.execute("UPDATE market_identity SET identity_value='binance' WHERE identity_key='exchange'")
    assert strategy.confirm_trade_entry(**{
        **confirmation, "side": "short",
    }) is False
    assert strategy.confirm_trade_entry(**{
        **confirmation, "current_time": signal_ts + pd.Timedelta(minutes=17),
    }) is False

    with sqlite3.connect(signal_db) as db:
        db.execute("UPDATE diagnostic_signals SET entry_allowed=0")
    assert strategy.confirm_trade_entry(**confirmation) is False


def test_okx_v4_fixture_flows_through_freqtrade_signal_boundary(tmp_path, monkeypatch):
    from jev_trader.grid import sl_key, tp_key
    from jev_trader.jev import JevSettings
    from jev_trader.paper import PaperEngine
    from jev_trader.shadow import shadow_run_id

    pair = "BTC/USDT:USDT"
    grid = {"horizon_bars": 16, "tp": [0.01], "sl": [0.01]}
    fixture_settings = JevSettings(
        "unused-fixture-key", "https://fixture.invalid/decisions",
        "fixture/jev-response-only", prompt_version="jev-ohlcv-v4",
    )
    run_id = shadow_run_id(fixture_settings, grid, "okx", [pair])
    dates = pd.date_range("2025-02-01", periods=80, freq="15min", tz="UTC")

    signal_db = tmp_path / "signals_okx.sqlite"
    metadata = (
        "okx", "usdt-perpetual", "ohlcv-features-v1", "jev-state-v4",
        "fixture/jev-response-only", "jev-ohlcv-v4",
    )
    observations = [(20, "long", 1), (30, "long", 0),
                    (40, "long", 1), (60, "long", 1)]
    with sqlite3.connect(signal_db) as db:
        db.execute("CREATE TABLE market_identity (identity_key TEXT, identity_value TEXT)")
        db.executemany("INSERT INTO market_identity VALUES (?, ?)", [
            ("exchange", "okx"), ("market_type", "usdt-perpetual"),
        ])
        db.execute("""CREATE TABLE signal_runs (
            run_id TEXT, pair TEXT, exchange TEXT, market_type TEXT,
            feature_version TEXT, state_version TEXT, model_id TEXT,
            prompt_version TEXT, PRIMARY KEY(run_id, pair))""")
        db.execute("INSERT INTO signal_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                   (run_id, pair, *metadata))
        db.execute("""CREATE TABLE signal_run_provenance (
            run_id TEXT PRIMARY KEY, provenance TEXT NOT NULL,
            excluded_from_research INTEGER NOT NULL)""")
        db.execute("INSERT INTO signal_run_provenance VALUES (?, 'live_market', 0)",
                   (run_id,))
        engine = PaperEngine(db, run_id, grid, {
            "taker_fee": 0.0005, "half_spread": 0.00005,
            "slippage": 0.0001, "funding_per_8h": 0.0001,
        })
        fixed_answers = {
            tp_key("long", 0.01, 0.01): 0.7,
            sl_key("long", 0.01, 0.01): 0.2,
            tp_key("short", 0.01, 0.01): 0.2,
            sl_key("short", 0.01, 0.01): 0.6,
        }
        for index, _side, allowed in observations:
            engine.record_signal(
                pair, int(dates[index].timestamp() * 1000),
                f"fixture-state-{index}", fixed_answers,
                allow_entry=bool(allowed),
                acceptance_reason="local_fixture" if allowed else "health_stop",
            )

    expected_env = {
        "JEV_SIGNAL_DB": str(signal_db), "JEV_SIGNAL_RUN_ID": run_id,
        "JEV_SIGNAL_EXCHANGE": metadata[0], "JEV_SIGNAL_MARKET_TYPE": metadata[1],
        "JEV_SIGNAL_FEATURE_VERSION": metadata[2], "JEV_SIGNAL_STATE_VERSION": metadata[3],
        "JEV_SIGNAL_MODEL_ID": metadata[4], "JEV_SIGNAL_PROMPT_VERSION": metadata[5],
    }
    for key, value in expected_env.items():
        monkeypatch.setenv(key, value)
    sys.path.insert(0, str(ROOT / "freqtrade/strategies"))
    from JevSignalStore import JevSignalStore

    strategy = JevSignalStore({"exchange": {"name": "okx"}})
    populated = strategy.populate_entry_trend(
        pd.DataFrame({"date": dates, "close": 100.0}), {"pair": pair}
    )
    assert [i for i, value in enumerate(populated["enter_long"]) if value] == [20, 40, 60]
    assert populated.loc[30, "enter_long"] == 0
    for index in (20, 40, 60):
        tag = populated.loc[index, "enter_tag"]
        confirm = dict(
            pair=pair, order_type="limit", amount=1.0, rate=100.0,
            time_in_force="GTC", current_time=dates[index + 1].to_pydatetime(),
            entry_tag=tag, side="long",
        )
        assert strategy.confirm_trade_entry(**confirm)
        assert not strategy.confirm_trade_entry(**{
            **confirm, "current_time": dates[index].to_pydatetime(),
        })
    with sqlite3.connect(signal_db) as db:
        db.execute(
            "UPDATE signal_run_provenance SET provenance='synthetic_fixture', "
            "excluded_from_research=1 WHERE run_id=?", (run_id,),
        )
    revoked = strategy.populate_entry_trend(
        pd.DataFrame({"date": dates, "close": 100.0}), {"pair": pair}
    )
    assert revoked["enter_long"].sum() == 0
    assert not strategy.confirm_trade_entry(**{
        **confirm, "current_time": dates[61].to_pydatetime(),
    })
    with sqlite3.connect(signal_db) as db:
        db.execute("DELETE FROM signal_run_provenance WHERE run_id=?", (run_id,))
    unmarked = strategy.populate_entry_trend(
        pd.DataFrame({"date": dates, "close": 100.0}), {"pair": pair}
    )
    assert unmarked["enter_long"].sum() == 0
    monkeypatch.setenv("JEV_SIGNAL_MODEL_ID", "fixture/other-model")
    assert not strategy.confirm_trade_entry(**{
        **confirm, "current_time": dates[61].to_pydatetime(),
    })
    monkeypatch.setenv("JEV_SIGNAL_MODEL_ID", metadata[4])
    trade = type("TradeFixture", (), {"open_date_utc": dates[41].to_pydatetime()})()
    assert strategy.custom_exit(
        pair, trade, dates[57].to_pydatetime(), 100.0, 0.0
    ) == "timeout_16_bars"
