from __future__ import annotations

from jev_trader.state import (
    LIVE_FEATURE_WINDOW, market_state_from_bars, market_state_from_bars_v4,
)
from jev_trader.jev import canonical_hash
from tests.test_causality import synthetic


GRID = {"horizon_bars": 16, "tp": [0.01], "sl": [0.01]}


def test_market_state_uses_identical_bounded_raw_bar_window():
    history = synthetic(n=LIVE_FEATURE_WINDOW + 80)
    bounded = history.tail(LIVE_FEATURE_WINDOW).reset_index(drop=True)
    from_window = market_state_from_bars("SOL/USDT:USDT", bounded, GRID)
    from_history = market_state_from_bars("SOL/USDT:USDT", history, GRID)

    assert from_window == from_history
    assert from_history["ts"] == int(history.iloc[-1]["ts"])


def test_ohlcv_v4_is_common_versioned_and_does_not_require_taker_buy():
    bars = synthetic(n=LIVE_FEATURE_WINDOW)
    common = bars.drop(columns="taker_buy_volume")
    state = market_state_from_bars_v4(
        "okx", "BTC/USDT:USDT", common, GRID
    )

    assert state["state_version"] == "jev-state-v4"
    assert state["feature_version"] == "ohlcv-features-v1"
    assert state["exchange"] == "okx"
    assert state["market_type"] == "usdt-perpetual"
    assert state["pair"] == "BTC/USDT:USDT"
    assert "f_buy_sell_ratio" not in state["features"]
    assert len(state["features"]) == 29
    assert all(value == value and abs(value) < float("inf")
               for value in state["features"].values())


def test_binance_v3_state_hash_remains_historically_unchanged():
    bars = synthetic(n=LIVE_FEATURE_WINDOW)
    state = market_state_from_bars(
        "BTC/USDT:USDT", bars, GRID
    )
    assert canonical_hash(state) == (
        "321a3a8adaa73860c82748df3d791820a47628a9a246af0ccb154066723e42de"
    )
