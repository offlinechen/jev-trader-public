"""The identical Jev market state for historical and live candles."""

from __future__ import annotations

import math
from collections.abc import Mapping

import pandas as pd

from .features import OHLCV_FEATURES_V1, build_features, build_ohlcv_features_v1

LIVE_FEATURE_WINDOW = 1500
FEATURE_VERSION_V4 = "ohlcv-features-v1"
STATE_VERSION_V4 = "jev-state-v4"


def market_state(symbol: str, feature_row: Mapping, recent: pd.DataFrame,
                 grid: Mapping, feature_cols: list[str]) -> dict:
    ts = int(feature_row["ts"])
    if len(recent) != 20 or int(recent.iloc[-1]["ts"]) != ts:
        raise ValueError(f"bar {ts} does not have 20 contiguous recent candles")
    if recent["ts"].diff().iloc[1:].ne(900_000).any():
        raise ValueError(f"bar {ts} has a gap in recent 15m candles")
    price = float(recent.iloc[-1]["close"])
    features = {col: float(feature_row[col]) for col in feature_cols}
    if price <= 0 or not math.isfinite(price) or not all(
        math.isfinite(value) for value in features.values()
    ):
        raise ValueError(f"bar {ts} has incomplete market features")
    return {
        "symbol": symbol,
        "ts": ts,
        "price": price,
        "features": features,
        "recent_bars": [
            [
                float(bar.open / price), float(bar.high / price),
                float(bar.low / price), float(bar.close / price),
                float(bar.volume),
            ]
            for bar in recent.itertuples(index=False)
        ],
        "grid": {
            "horizon_bars": grid["horizon_bars"],
            "tp": grid["tp"],
            "sl": grid["sl"],
        },
    }


def market_state_from_bars(symbol: str, bars: pd.DataFrame, grid: Mapping) -> dict:
    """Build inference state from the same bounded raw-bar window in every caller."""
    window = bars.tail(LIVE_FEATURE_WINDOW).reset_index(drop=True)
    features = build_features(window)
    feature_row = features.iloc[-1]
    feature_cols = [column for column in features if column.startswith("f_")]
    return market_state(
        symbol, feature_row, window.tail(20), grid, feature_cols
    )


def market_state_from_bars_v4(exchange: str, pair: str, bars: pd.DataFrame,
                              grid: Mapping) -> dict:
    """Build the cross-venue OHLCV-only state, separate from the Jev v3 state."""
    window = bars.tail(LIVE_FEATURE_WINDOW).reset_index(drop=True)
    features = build_ohlcv_features_v1(window)
    state = market_state(
        pair, features.iloc[-1], window.tail(20), grid, list(OHLCV_FEATURES_V1)
    )
    state.update(
        exchange=exchange,
        market_type="usdt-perpetual",
        pair=pair,
        feature_version=FEATURE_VERSION_V4,
        state_version=STATE_VERSION_V4,
    )
    return state
