from __future__ import annotations

import json

import pandas as pd

from jev_trader.labels import level_col
from jev_trader.meter import _g3b_candidate_timestamps, _historic_attempts
from jev_trader.sample import composition, stratified_sample


def _oof_rows(ts, side, tp, sl, outcome):
    return [{
        "ts": ts, "side": side, "tp": tp, "sl": sl, "fold": 1,
        "horizon": 16,
        "model": model, "outcome": outcome,
        "p_sl_first": 0.2, "p_tp_first": 0.3, "p_timeout": 0.5,
    } for model in ("climatology", "logit", "lgbm")]


def test_g3b_candidates_require_full_matched_oof_coverage(tmp_path):
    data_dir = tmp_path / "data"
    oof_dir = data_dir / "oof"
    oof_dir.mkdir(parents=True)
    grid = {"tp": [0.01], "sl": [0.005], "horizon_bars": 16}
    labels = pd.DataFrame({
        "ts": [100, 200, 300],
        level_col(0.01, "up"): [-1, 0, -1],
        level_col(0.01, "dn"): [-1, -1, -1],
        level_col(0.005, "up"): [-1, -1, -1],
        level_col(0.005, "dn"): [-1, 0, -1],
    })
    labels.attrs["horizon_min"] = 16 * 15
    features = pd.DataFrame({
        "ts": [100, 200, 300], "f_atr_pct": [0.1, 0.2, 0.3],
        "f_4h_trend": [-1.0, 0.0, 1.0],
    })
    rows = []
    # Bar 100: both cells have complete predictions for all baselines.
    for side in ("long", "short"):
        rows += _oof_rows(100, side, 0.01, 0.005, 2)
    # Bar 200 has one ambiguous cell; its other cell is still fully covered.
    rows += _oof_rows(200, "short", 0.01, 0.005, 2)
    # Bar 300's non-ambiguous long cell is missing one baseline model.
    rows += _oof_rows(300, "long", 0.01, 0.005, 2)[:-1]
    pd.DataFrame(rows).to_parquet(oof_dir / "fold_001.parquet", index=False)

    timestamps, audit = _g3b_candidate_timestamps(data_dir, features, labels, grid)
    assert timestamps.tolist() == [100, 200]
    assert audit["eligible_bars"] == 2
    assert audit["matched_oof_cell_rows"] == 3
    assert audit["coverage"] == 1.0


def test_historic_attempts_include_successes_and_timestamped_failures(tmp_path):
    success = pd.DataFrame({"ts": [10, 20], "sample_index": [1, 2]})
    success.to_parquet(tmp_path / "metering_10000.parquet", index=False)
    (tmp_path / "metering_10000_failures.json").write_text(
        json.dumps([{"sample_index": 3, "ts": 30, "error": "402"}]),
        encoding="utf-8",
    )
    old_g3a = pd.DataFrame({"ts": [20, 40], "sample_index": [1, 2]})
    old_g3a.to_parquet(tmp_path / "metering_2000.parquet", index=False)
    (tmp_path / "metering_2000_failures.json").write_text(
        json.dumps([{"sample_index": 3, "error": "legacy failure"}]),
        encoding="utf-8",
    )

    attempted, unattributed = _historic_attempts(tmp_path)
    assert attempted == {10, 20, 30, 40}
    assert unattributed == 1


def test_formal_sample_keeps_quarter_dimension_and_exact_size():
    ts = pd.date_range("2021-01-01", "2024-12-31", periods=4800, tz="UTC").as_unit("ns").asi8 // 1_000_000
    features = pd.DataFrame({
        "ts": ts,
        "f_atr_pct": [i % 97 for i in range(len(ts))],
        "f_4h_trend": [((i * 31) % 101) - 50 for i in range(len(ts))],
    })
    picked = stratified_sample(features, ts, n=1000, seed=42, floor=1)
    if len(picked) < 1000:
        rest = features[~features.ts.isin(picked.ts)].sample(
            n=1000 - len(picked), random_state=42
        )
        picked = pd.concat([picked, rest], ignore_index=True)
    elif len(picked) > 1000:
        picked = picked.sample(n=1000, random_state=42)
    check = composition(picked, features)
    assert len(picked) == 1000
    assert picked.ts.is_unique
    assert check.loc[check.dim.eq("quarter"), "level"].nunique() == 16
    assert check.ok.all()
