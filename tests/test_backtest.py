from __future__ import annotations

import numpy as np
import pandas as pd

from jev_trader import labels
from jev_trader.backtest import _execution_outcome, _round_trip_cost, run_backtest


CFG = {
    "grid": {"horizon_bars": 2, "tp": [0.01], "sl": [0.01]},
    "costs": {"taker_fee": 0.0005, "half_spread": 0.00005,
              "slippage": 0.0001, "funding_per_8h": 0.0001,
              "expected_hold_hours": 2.0},
    "thresholds": {"min_ev_net": 0.0, "max_regime_entropy": 1.10,
                    "max_vol_extreme": 0.25},
    "risk": {"risk_frac": 0.005, "leverage": 1, "max_open_trades": 1},
}


def label_row(outcome, ret=0.0):
    return pd.Series({
        "ts": 1_000_000, "entry_ts": 1_900_000, "entry_price": 100.0,
        "up_100": 0 if outcome == labels.Outcome.TP_FIRST else -1,
        "dn_100": 0 if outcome == labels.Outcome.SL_FIRST else -1,
        "ret_at_horizon": ret,
    })


def test_execution_uses_first_touch_and_horizon():
    row = label_row(labels.Outcome.TP_FIRST)
    assert _execution_outcome(row, "long", 0.01, 0.01, 2) == (labels.Outcome.TP_FIRST, 1)
    row = label_row(labels.Outcome.TIMEOUT, ret=-0.002)
    assert _execution_outcome(row, "long", 0.01, 0.01, 2) == (labels.Outcome.TIMEOUT, 30)


def test_costs_include_both_sides_and_funding():
    assert abs(_round_trip_cost(CFG["costs"], 2) - 0.001225) < 1e-12


def test_backtest_never_opens_overlapping_position():
    timestamps = [1_000_000, 1_900_000]
    second = label_row(labels.Outcome.TP_FIRST)
    second["ts"], second["entry_ts"] = timestamps[1], 2_800_000
    labels_df = pd.DataFrame([label_row(labels.Outcome.TP_FIRST), second])
    predictions = pd.DataFrame([
        {"ts": timestamps[0], "regime_up": 1.0, "regime_down": 0.0,
         "regime_range": 0.0, "regime_transition": 0.0, "vol_extreme": 0.0,
         "p_long_tp100_sl100": 0.9, "p_long_sl100_tp100": 0.05,
         "p_short_tp100_sl100": 0.1, "p_short_sl100_tp100": 0.8},
        {"ts": timestamps[1], "regime_up": 1.0, "regime_down": 0.0,
         "regime_range": 0.0, "regime_transition": 0.0, "vol_extreme": 0.0,
         "p_long_tp100_sl100": 0.9, "p_long_sl100_tp100": 0.05,
         "p_short_tp100_sl100": 0.1, "p_short_sl100_tp100": 0.8},
    ])
    trades, decisions, metrics = run_backtest(predictions, labels_df, CFG)
    assert len(trades) == 1
    assert "blocked_open_position" in decisions.action.tolist()
    assert metrics["trades"] == 1
