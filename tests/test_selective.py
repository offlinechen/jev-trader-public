"""Guards for the selective-trading evaluator.

The failure mode that matters is a result that looks tradeable only because
some step saw the future: calibration fitted on the fold it is applied to, a
cut-off taken from the evaluation period's own score distribution, or a
confidence interval that treats 40 views of one price path as 40 trades.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import jev_trader.selective as selective_module
from jev_trader.cli import _default_compare
from jev_trader.labels import Outcome, level_col
from jev_trader.selective import (
    BAR_MS, DAY_MS, MATCH_KEYS, _observation_hash, calibrate, matched_oof,
    non_overlapping, per_bar, select, summarise, universe,
)

H = 16
COSTS = {"taker_fee": 0.0005, "half_spread": 0.00005, "slippage": 0.0001,
         "funding_per_8h": 0.0001, "expected_hold_hours": 2.0}
MONTH_BARS = 30 * 96


def oof_frame(n_folds=8, per_fold=400, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_folds):
        ts = (k * MONTH_BARS + np.arange(per_fold) * 7) * BAR_MS
        p = rng.dirichlet([2, 1, 3], per_fold)
        y = np.array([rng.choice(3, p=pi) for pi in p])
        rows.append(pd.DataFrame({
            "ts": ts, "side": "long", "tp": 0.01, "sl": 0.005, "fold": k, "outcome": y,
            "p_sl_first": p[:, 0], "p_tp_first": p[:, 1], "p_timeout": p[:, 2]}))
    return pd.concat(rows, ignore_index=True)


def test_calibration_never_sees_the_fold_it_calibrates():
    df = oof_frame()
    base = calibrate(df, H)
    shuffled = df.copy()
    k = 6
    m = shuffled["fold"] == k
    shuffled.loc[m, "outcome"] = np.random.default_rng(9).permutation(shuffled.loc[m, "outcome"].to_numpy())
    again = calibrate(shuffled, H)
    a = base.loc[base["fold"] == k, ["c_sl", "c_tp", "c_to"]].to_numpy()
    b = again.loc[again["fold"] == k, ["c_sl", "c_tp", "c_to"]].to_numpy()
    assert np.array_equal(a, b), "fold k's own outcomes leaked into its calibration"


def test_calibration_skips_folds_without_enough_history():
    out = calibrate(oof_frame(), H, min_prior_folds=3)
    assert out["fold"].min() == 3
    assert np.allclose(out[["c_sl", "c_tp", "c_to"]].sum(1), 1.0)


def bars_frame(n_folds=10, per_fold=200, seed=1):
    rng = np.random.default_rng(seed)
    ts = np.concatenate([(k * MONTH_BARS + np.arange(per_fold) * 5) * BAR_MS for k in range(n_folds)])
    fold = np.repeat(np.arange(n_folds), per_fold)
    return pd.DataFrame({"ts": ts, "fold": fold, "ev": rng.normal(size=len(ts)),
                         "day": ts // DAY_MS, "net": rng.normal(0, 0.01, len(ts)),
                         "win": rng.random(len(ts)) < 0.4, "outcome": 0,
                         "entry_ts": ts + BAR_MS, "exit_ts": ts + BAR_MS + 240 * 60_000})


def test_walk_forward_cutoffs_ignore_the_future():
    bars = bars_frame()
    before = select(bars, "ev", 0.1, "walk_forward", H)
    future = bars.copy()
    future.loc[future["fold"] >= 7, "ev"] += 100.0      # wildly change later folds
    after = select(future, "ev", 0.1, "walk_forward", H)
    early = lambda s: set(s.loc[s["fold"] < 7, "ts"])
    assert early(before) == early(after)


def test_diagnostic_and_walk_forward_share_one_universe():
    bars = bars_frame()
    assert len(select(bars, "ev", 1.0, "diagnostic", H)) == len(select(bars, "ev", 1.0, "walk_forward", H))


def test_explicit_eligible_universe_is_not_recomputed():
    bars = bars_frame()
    eligible = universe(bars, H)
    selected = select(bars, "ev", 1.0, "walk_forward", H, eligible=eligible)
    assert len(selected) == len(eligible)
    assert set(selected.ts) == set(eligible.ts)


def test_observation_hash_is_order_independent():
    frame = pd.DataFrame({
        "ts": [2, 1], "side": ["short", "long"], "tp": [0.01, 0.005],
        "sl": [0.005, 0.01], "fold": [2, 1], "outcome": [0, 1],
    })
    assert _observation_hash(frame) == _observation_hash(frame.iloc[::-1])
    changed = frame.copy()
    changed.loc[0, "outcome"] = 99
    assert _observation_hash(frame) != _observation_hash(changed)


def test_matched_oof_returns_one_strict_key_set_for_all_models(tmp_path, monkeypatch):
    rows = []
    for fold in range(5):
        for i in range(2):
            ts = fold * 100_000_000 + i * 1_000_000
            outcome = (fold + i) % 2
            rows.append({"ts": ts, "side": "long", "tp": 0.01, "sl": 0.005,
                         "fold": fold, "outcome": outcome,
                         "p_sl_first": 0.25 if outcome else 0.65,
                         "p_tp_first": 0.65 if outcome else 0.25,
                         "p_timeout": 0.10})
    lgbm = pd.DataFrame(rows)
    jev = lgbm.iloc[:-1].copy()
    (tmp_path / "g3a_long.parquet").write_bytes(b"fixture")
    monkeypatch.setattr(selective_module, "load_oof", lambda *_args: lgbm.copy())
    monkeypatch.setattr(selective_module, "from_jev_long", lambda *_args: jev.copy())

    matched, meta = matched_oof(tmp_path, H)
    keys = [set(frame[MATCH_KEYS].itertuples(index=False, name=None))
            for frame in matched.values()]
    assert keys[0] == keys[1] == keys[2]
    assert len(keys[0]) == meta["matched_n"]
    assert len({
        _observation_hash(frame) for frame in matched.values()
    }) == 1
    assert _observation_hash(matched["lgbm"]) == meta["observation_hash"]


def test_selective_default_comparison_never_self_compares():
    assert _default_compare("lgbm") == "jev"
    assert _default_compare("jev") == "lgbm"
    assert _default_compare("stack") == "jev"
    assert all(_default_compare(model) != model for model in ("lgbm", "jev", "stack"))


def test_realised_returns_follow_first_touch_and_actual_hold():
    tp, sl = 0.01, 0.005
    long = pd.DataFrame({
        "ts": [0, BAR_MS, 2 * BAR_MS], "side": "long", "tp": tp, "sl": sl, "fold": 0,
        "outcome": [Outcome.TP_FIRST, Outcome.SL_FIRST, Outcome.TIMEOUT],
        "c_tp": .4, "c_sl": .3, "c_to": .3, "e_to": 0.0, "ev": 0.001, "conf": .57,
        "ret_at_horizon": [0.0, 0.0, 0.003],
    })
    labels = pd.DataFrame({"ts": long["ts"],
                           level_col(tp, "up"): [29, -1, -1], level_col(sl, "dn"): [-1, 59, -1]})
    out = per_bar(long, labels, COSTS, H)
    friction = 2 * .0005 + 2 * .00005 + .0001
    fund = lambda minutes: .0001 * minutes / 60 / 8
    assert out["gross"].tolist() == pytest.approx([tp, -sl, 0.003])
    assert out["hold_min"].tolist() == [30, 60, 240]
    assert out["net"].tolist() == pytest.approx(
        [tp - friction - fund(30), -sl - friction - fund(60), 0.003 - friction - fund(240)])


def test_short_timeout_return_is_sign_flipped():
    df = pd.DataFrame({"ts": [0], "side": "short", "tp": 0.01, "sl": 0.005, "fold": 0,
                       "outcome": [Outcome.TIMEOUT], "c_tp": .3, "c_sl": .3, "c_to": .4,
                       "e_to": 0.0, "ev": 0.0, "conf": .5, "ret_at_horizon": [0.004]})
    labels = pd.DataFrame({"ts": [0], level_col(0.01, "dn"): [-1], level_col(0.005, "up"): [-1]})
    assert per_bar(df, labels, COSTS, H)["gross"].iloc[0] == pytest.approx(-0.004)


def test_ess_is_n_when_independent_and_days_when_duplicated():
    rng = np.random.default_rng(3)
    iid = pd.DataFrame({"net": rng.normal(size=4000), "day": np.arange(4000),
                        "win": False, "outcome": 0, "ev": 0.0})
    assert summarise(iid)["ess"] == pytest.approx(4000, rel=0.1)
    one = rng.normal(size=200)
    dup = pd.DataFrame({"net": np.repeat(one, 40), "day": np.repeat(np.arange(200), 40),
                        "win": False, "outcome": 0, "ev": 0.0})
    assert summarise(dup)["ess"] == pytest.approx(200, rel=0.15)   # 40 copies != 40 trades


def test_non_overlapping_keeps_one_position_at_a_time():
    b = pd.DataFrame({"ts": [0, 1, 2, 10], "entry_ts": [0, 1, 2, 10],
                      "exit_ts": [5, 3, 4, 12], "net": [1, 2, 3, 4]})
    assert non_overlapping(b)["ts"].tolist() == [0, 10]
