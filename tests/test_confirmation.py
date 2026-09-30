import json

import numpy as np
import pandas as pd
import pytest

from jev_trader import config
from jev_trader import labels as lab
from jev_trader.confirmation import _has_credit_exhaustion, evaluate_frames, run
from jev_trader.grid import cell_key, sl_key, tp_key


def _frames():
    rows = []
    base = []
    start = pd.Timestamp("2022-01-01", tz="UTC").value // 1_000_000
    for fold in range(1, 5):
        for i in range(80):
            ts = start + ((fold - 1) * 90 + i) * 86_400_000
            y = int((i + fold) % 2)
            p = 0.7 if y else 0.3
            for side in ("long", "short"):
                rows.append({"ts": ts, "side": side, "tp": 0.01, "sl": 0.005,
                             "outcome": y, "p_tp": p, "p_sl": 1 - p,
                             "p_timeout": 0.0, "prob_invalid": False,
                             "response_valid": True, "year": 2022 + (fold > 2),
                             "quarter": "Q1", "vol_q": "v1", "dir_t": "up"})
                base.append({"ts": ts, "side": side, "tp": 0.01, "sl": 0.005,
                             "fold": fold, "outcome": y,
                             "p_tp_first": 0.65 if y else 0.35,
                             "p_sl_first": 0.35 if y else 0.65,
                             "p_timeout": 0.0})
    return pd.DataFrame(rows), pd.DataFrame(base)


def test_confirmation_uses_same_matched_rows_and_prior_folds_only():
    jev, baseline = _frames()
    result = evaluate_frames(jev, baseline, horizon_bars=16, n_boot=20, min_train=20)
    assert result["matched_observations"] == 480
    assert result["stack_train_folds"] == [2, 3, 4]
    assert result["stacker_train_max_ts_by_fold"]["2"] < result["stacker_test_min_ts_by_fold"]["2"]
    assert result["observation_hash"]
    assert set(result["metrics"]) == {"jev", "lgbm", "stack"}


def test_invalid_jev_bar_is_removed_from_all_matched_models():
    jev, baseline = _frames()
    bad_ts = int(jev.ts.iloc[160])
    jev.loc[jev.ts.eq(bad_ts), "prob_invalid"] = True
    baseline = baseline[~baseline.ts.eq(bad_ts)]
    result = evaluate_frames(jev, baseline, horizon_bars=16, n_boot=10, min_train=20)
    assert result["invalid_bars"] == 1
    assert result["matched_observations"] == 478


def test_missing_nonambiguous_baseline_observation_blocks_confirmation():
    jev, baseline = _frames()
    baseline = baseline.drop(index=baseline.index[160])
    with pytest.raises(ValueError, match="missing LightGBM OOF"):
        evaluate_frames(jev, baseline, horizon_bars=16, n_boot=10, min_train=20)


def test_changed_nonambiguous_baseline_label_blocks_confirmation():
    jev, baseline = _frames()
    baseline.loc[baseline.index[160], "outcome"] = 1 - baseline.loc[baseline.index[160], "outcome"]
    with pytest.raises(ValueError, match="label mismatch"):
        evaluate_frames(jev, baseline, horizon_bars=16, n_boot=10, min_train=20)


def test_timeout_and_ambiguous_rows_are_excluded_from_ranking_pairing():
    jev, baseline = _frames()
    ts = int(jev.ts.iloc[160])
    jev.loc[jev.ts.eq(ts) & jev.side.eq("long"), "outcome"] = int(lab.Outcome.AMBIGUOUS)
    jev.loc[jev.ts.eq(ts) & jev.side.eq("short"), "outcome"] = int(lab.Outcome.TIMEOUT)
    baseline = baseline[~baseline.ts.eq(ts)]
    result = evaluate_frames(jev, baseline, horizon_bars=16, n_boot=10, min_train=20)
    assert result["matched_observations"] == 478


def test_unestimable_cell_support_is_incomplete_not_a_ranking_failure():
    jev, baseline = _frames()
    jev.loc[jev.side.eq("long"), "outcome"] = 0
    baseline.loc[baseline.side.eq("long"), "outcome"] = 0
    result = evaluate_frames(jev, baseline, horizon_bars=16, n_boot=10, min_train=20)
    assert result["gate"].startswith("INCOMPLETE")
    assert result["readiness"]["cells_with_min_class_support"] < result["readiness"]["expected_cells"]


def test_failed_and_invalid_bars_are_counted_without_double_subtraction():
    jev, baseline = _frames()
    failed_ts = int(jev.ts.iloc[320])
    invalid_ts = int(jev.ts.iloc[322])
    jev.loc[jev.ts.eq(failed_ts), "response_valid"] = False
    jev.loc[jev.ts.eq(invalid_ts), "prob_invalid"] = True
    result = evaluate_frames(jev, baseline, horizon_bars=16, n_boot=10, min_train=20)
    assert result["response_failed_bars"] == 1
    assert result["probability_invalid_bars"] == 1
    assert result["all_cell_valid_bars"] == 318
    assert result["invalid_bars"] == 2


@pytest.mark.parametrize("credit_failure", [False, True])
def test_complete_manifest_with_small_failures_gates_unless_402(tmp_path, monkeypatch, credit_failure):
    n = 10_000
    grid = {"tp": [0.01], "sl": [0.005], "horizon_bars": 16}
    start = pd.Timestamp("2022-01-01", tz="UTC").value // 1_000_000
    rows, oof = [], []
    for i in range(n):
        fold = i // 2_500 + 1
        y = i % 2
        ts = start + (i + (fold - 1) * 16) * 900_000
        failed = 3000 <= i < 3010
        invalid = 3010 <= i < 3020
        row = {"ts": ts, "year": 2022 + (i > 5000), "quarter": "Q1",
               "vol_q": "v1", "dir_t": "up", "response_valid": not failed}
        for side in ("long", "short"):
            cell = cell_key(side, 0.01, 0.005)
            row[tp_key(side, 0.01, 0.005)] = np.nan if failed else (0.8 if y else 0.8 if invalid else 0.2)
            row[sl_key(side, 0.01, 0.005)] = np.nan if failed else (0.4 if invalid else (0.2 if y else 0.8))
            row[f"outcome_{cell}"] = y
            if not (failed or invalid):
                oof.append({"ts": ts, "side": side, "tp": 0.01, "sl": 0.005,
                            "fold": fold, "model": "lgbm", "outcome": y,
                            "p_sl_first": 0.45 if y else 0.55,
                            "p_tp_first": 0.55 if y else 0.45, "p_timeout": 0.0})
        rows.append(row)
    data_dir = tmp_path / "data"
    (data_dir / "oof").mkdir(parents=True)
    artifact = data_dir / "metering_10000_0123456789abcdef.parquet"
    pd.DataFrame(rows).to_parquet(artifact, index=False)
    pd.DataFrame(oof).to_parquet(data_dir / "oof" / "fold_all.parquet", index=False)
    failures = ({"3000": {"error": "Jev API request failed (402): insufficient credits"}}
                if credit_failure else {})
    manifest = {"complete": True, "identity": {"requested": n},
                "sampling_audit": {"coverage": 1.0, "composition_ok": True},
                "failure_details": failures,
                "budget": {"by_status": {"402": int(credit_failure)}}}
    artifact.with_name(f"{artifact.stem}_manifest.json").write_text(json.dumps(manifest))
    cfg = {"jev": {"prompt_version": "v3", "model_id": "test"},
           "grid": grid, "symbols": ["BTC/USDT:USDT"]}
    monkeypatch.setattr(config, "load", lambda: cfg)
    from jev_trader import g3a
    monkeypatch.setattr(g3a, "_validate_prediction_artifact", lambda *args, **kwargs: None)

    report_path = run(str(artifact), str(data_dir), str(tmp_path / "report.md"), n_boot=400)
    report = report_path.read_text()
    assert ("Gate: **INCOMPLETE — HTTP 402/credit exhaustion**" in report
            if credit_failure else "Gate: **PASS G3b**" in report)
    assert "9990 / 10" in report
    assert "9980" in report
    assert "valid bootstrap replicates | 400 / 360" in report


def test_402_is_credit_exhaustion_but_ordinary_503_is_not():
    assert _has_credit_exhaustion({"budget": {"by_status": {"402": 1}}}, pd.DataFrame())
    assert _has_credit_exhaustion({"failure_details": {"1": {"error": "HTTP 402"}}}, pd.DataFrame())
    assert not _has_credit_exhaustion({"budget": {"by_status": {"503": 4}}}, pd.DataFrame())


def test_confirmation_cli_rejects_fewer_than_400_bootstrap_replicates(capsys):
    from jev_trader.cli import main

    with pytest.raises(SystemExit) as error:
        main(["confirmation", "--predictions", "not-read.parquet", "--bootstrap", "399"])
    assert error.value.code == 2
    assert "at least 400" in capsys.readouterr().err


def test_input_order_does_not_change_matched_observation_hash_or_metrics():
    jev, baseline = _frames()
    forward = evaluate_frames(jev, baseline, horizon_bars=16, n_boot=10, min_train=20)
    reverse = evaluate_frames(jev.sample(frac=1, random_state=7),
                              baseline.sample(frac=1, random_state=9),
                              horizon_bars=16, n_boot=10, min_train=20)
    assert forward["observation_hash"] == reverse["observation_hash"]
    assert forward["metrics"] == reverse["metrics"]
