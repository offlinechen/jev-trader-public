import json

import pandas as pd
import pytest
import jev_trader.g3a as g3a_module

from jev_trader.g3a import _validate_prediction_artifact, gate_decision
from jev_trader.jev import build_questions, canonical_hash, prompt_contract
from jev_trader.meter import METER_IDENTITY_VERSION


MINI_GRID = {"horizon_bars": 16, "tp": [0.01], "sl": [0.005]}


def test_g3a_gate_uses_relative_brier_improvement_direction():
    assert gate_decision((0.48, 0.50), 0.0, 0.0, "ok").startswith("STOP")
    assert gate_decision((0.51, 0.57), 0.0001, -0.001, "ok").startswith("PASS")
    assert gate_decision((0.4854, 0.5712), 0.00035, 0.0048, "ok").startswith("INCONCLUSIVE")


def test_g3a_rejects_incomplete_meter_manifest(tmp_path):
    path, manifest = _meter_fixture(tmp_path, complete=False)
    path.with_name(f"{path.stem}_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="not complete"):
        _validate_prediction_artifact(path, pd.read_parquet(path))


def test_g3a_rejects_meter_manifest_identity_mismatch(tmp_path):
    path, manifest = _meter_fixture(tmp_path, complete=True)
    manifest["identity"]["questions_sha256"] = "changed"
    path.with_name(f"{path.stem}_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="run_id/identity mismatch"):
        _validate_prediction_artifact(path, pd.read_parquet(path))


def test_g3a_accepts_complete_meter_manifest_and_explicit_legacy_path(tmp_path):
    path, manifest = _meter_fixture(tmp_path, complete=True)
    path.with_name(f"{path.stem}_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    assert _validate_prediction_artifact(path, pd.read_parquet(path), expected_prompt_version="v3", grid=MINI_GRID) == manifest["run_id"]
    legacy_dir = tmp_path / "data"
    legacy_dir.mkdir()
    legacy = legacy_dir / "metering_2000.parquet"
    legacy_frame = pd.DataFrame({
        "sample_index": [i for i in range(1, 1954)], "ts": list(range(1, 1954)),
        "prompt_version": ["v3"] * 1953, "year": [2025] * 1953,
        "quarter": ["Q1"] * 1953, "vol_q": ["v1"] * 1953, "dir_t": ["up"] * 1953,
    })
    legacy_frame.to_parquet(legacy, index=False)
    (legacy_dir / "metering_2000_failures.json").write_text(
        json.dumps([{"sample_index": i, "error": "legacy"} for i in range(1954, 2001)]),
        encoding="utf-8",
    )
    assert _validate_prediction_artifact(
        legacy, legacy_frame, legacy_dir, "v3"
    ) == "legacy-2000"


def test_g3a_rejects_legacy_nan_prompt_version(tmp_path):
    legacy_dir, legacy_frame = _legacy_fixture(tmp_path)
    legacy_frame.loc[0, "prompt_version"] = None
    legacy = legacy_dir / "metering_2000.parquet"
    legacy_frame.to_parquet(legacy, index=False)
    with pytest.raises(ValueError, match="prompt_version"):
        _validate_prediction_artifact(legacy, legacy_frame, legacy_dir, "v3")


@pytest.mark.parametrize("mutation", ["missing", "null"])
def test_g3a_requires_prompt_version_even_without_expected_value(tmp_path, mutation):
    path, manifest = _meter_fixture(tmp_path, complete=True)
    frame = pd.read_parquet(path)
    if mutation == "missing":
        frame = frame.drop(columns=["prompt_version"])
    else:
        frame.loc[0, "prompt_version"] = None
    frame.to_parquet(path, index=False)
    path.with_name(f"{path.stem}_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="prompt_version"):
        _validate_prediction_artifact(path, frame)


@pytest.mark.parametrize("failure_indices", [
    list(range(1954, 2000)),
    [1] + list(range(1955, 2001)),
    list(range(1954, 2000)) + [2001],
])
def test_g3a_rejects_invalid_legacy_success_failure_partition(tmp_path, failure_indices):
    legacy_dir = tmp_path / "data"
    legacy_dir.mkdir()
    legacy = legacy_dir / "metering_2000.parquet"
    frame = pd.DataFrame({
        "sample_index": list(range(1, 1954)), "ts": list(range(1, 1954)),
        "prompt_version": ["v3"] * 1953, "year": [2025] * 1953,
        "quarter": ["Q1"] * 1953, "vol_q": ["v1"] * 1953, "dir_t": ["up"] * 1953,
    })
    frame.to_parquet(legacy, index=False)
    (legacy_dir / "metering_2000_failures.json").write_text(
        json.dumps([{"sample_index": i} for i in failure_indices]), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        _validate_prediction_artifact(legacy, frame, legacy_dir, "v3")


@pytest.mark.parametrize("field,value,message", [
    ("prompt_version", "v2", "prompt_version"),
    ("horizon_bars", 17, "horizon"),
    ("tp", [0.02], "tp"),
    ("sl", [0.01], "sl"),
    ("questions_sha256", "changed", "questions"),
    ("model_id", "other-model", "model_id"),
    ("symbol", "ETH/USDT:USDT", "symbol"),
])
def test_g3a_rejects_meter_manifest_for_different_current_config(tmp_path, field, value, message):
    path, manifest = _meter_fixture(tmp_path, complete=True)
    manifest["identity"][field] = value
    path = _rewrite_meter_identity(path, manifest)
    path.with_name(f"{path.stem}_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match=message):
        _validate_prediction_artifact(
            path, pd.read_parquet(path), expected_prompt_version="v3", grid=MINI_GRID,
            expected_model_id="~typesafe/jev-latest", expected_symbol="BTC/USDT:USDT",
        )


def test_g3a_rejects_self_consistent_stale_contract(tmp_path):
    path, manifest = _meter_fixture(tmp_path, complete=True)
    manifest["identity"]["contract_sha256"] = "stale-contract"
    path = _rewrite_meter_identity(path, manifest)
    path.with_name(f"{path.stem}_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="contract"):
        _validate_prediction_artifact(path, pd.read_parquet(path), expected_prompt_version="v3", grid=MINI_GRID)


def test_g3a_rejects_self_consistent_old_identity_version(tmp_path):
    path, manifest = _meter_fixture(tmp_path, complete=True)
    manifest["identity"]["identity_version"] = METER_IDENTITY_VERSION - 1
    path = _rewrite_meter_identity(path, manifest)
    path.with_name(f"{path.stem}_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="identity_version"):
        _validate_prediction_artifact(path, pd.read_parquet(path))


@pytest.mark.parametrize("column", ["sample_index", "ts"])
def test_g3a_rejects_fractional_new_artifact_indices(tmp_path, column):
    path, manifest = _meter_fixture(tmp_path, complete=True)
    frame = pd.read_parquet(path)
    frame[column] = frame[column].astype(float)
    frame.loc[0, column] = 1.5
    frame.to_parquet(path, index=False)
    path.with_name(f"{path.stem}_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match=column):
        _validate_prediction_artifact(path, frame)


def test_g3a_rejects_complete_run_with_no_valid_responses(tmp_path, monkeypatch):
    frame = pd.DataFrame({
        "sample_index": [1], "ts": [100], "request_state_hash": ["state-1"],
        "prompt_version": ["v3"], "response_valid": [False],
    })
    identity = {
        "identity_version": METER_IDENTITY_VERSION,
        "requested": 1,
        "picked_ts_sha256": canonical_hash([{"sample_index": 1, "ts": 100}]),
        "request_state_hashes": ["state-1"],
        "questions_sha256": canonical_hash(build_questions(MINI_GRID)),
        "contract_sha256": canonical_hash(prompt_contract(16)),
        "prompt_version": "v3", "model_id": "~typesafe/jev-latest",
        "symbol": "BTC/USDT:USDT", "horizon_bars": 16,
        "tp": [0.01], "sl": [0.005],
    }
    run_id = canonical_hash(identity)[:16]
    path = tmp_path / f"metering_1_{run_id}.parquet"
    frame.to_parquet(path, index=False)
    path.with_name(f"{path.stem}_manifest.json").write_text(json.dumps({
        "complete": True, "run_id": run_id, "identity": identity,
        "completed": {"1": "failure"},
    }), encoding="utf-8")
    monkeypatch.setattr(g3a_module.config, "load", lambda: {
        "data_dir": str(tmp_path), "symbols": ["BTC/USDT:USDT"],
        "jev": {"prompt_version": "v3", "model_id": "~typesafe/jev-latest"},
        "grid": MINI_GRID,
    })
    with pytest.raises(ValueError, match="no valid responses"):
        g3a_module.run(path, tmp_path)


def test_g3a_rejects_nan_prompt_version_in_new_artifact(tmp_path):
    path, manifest = _meter_fixture(tmp_path, complete=True)
    frame = pd.read_parquet(path)
    frame.loc[0, "prompt_version"] = None
    frame.to_parquet(path, index=False)
    path.with_name(f"{path.stem}_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="prompt_version"):
        _validate_prediction_artifact(path, frame, expected_prompt_version="v3", grid=MINI_GRID)


def test_g3a_does_not_accept_arbitrary_legacy_basename(tmp_path):
    path = tmp_path / "metering_2000.parquet"
    pd.DataFrame({"sample_index": [1]}).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="restricted"):
        _validate_prediction_artifact(path, pd.read_parquet(path))


def _meter_fixture(tmp_path, complete):
    rows = pd.DataFrame({
        "sample_index": [1, 2], "ts": [100, 200],
        "request_state_hash": ["state-1", "state-2"],
        "prompt_version": ["v3", "v3"],
    })
    picked = [{"sample_index": 1, "ts": 100}, {"sample_index": 2, "ts": 200}]
    manifest = {
        "complete": complete,
        "identity": {
            "identity_version": METER_IDENTITY_VERSION,
            "model_id": "~typesafe/jev-latest",
            "symbol": "BTC/USDT:USDT",
            "requested": 2,
            "picked_ts_sha256": canonical_hash(picked),
            "request_state_hashes": ["state-1", "state-2"],
            "questions_sha256": canonical_hash(build_questions(MINI_GRID)),
            "prompt_version": "v3",
            "horizon_bars": 16,
            "tp": [0.01],
            "sl": [0.005],
            "contract_sha256": canonical_hash(prompt_contract(16)),
        },
        "completed": {"1": "success", "2": "success"},
    }
    run_id = canonical_hash(manifest["identity"])[:16]
    manifest["run_id"] = run_id
    path = tmp_path / f"metering_2_{run_id}.parquet"
    rows.to_parquet(path, index=False)
    return path, manifest


def _rewrite_meter_identity(path, manifest):
    run_id = canonical_hash(manifest["identity"])[:16]
    new_path = path.with_name(f"metering_2_{run_id}.parquet")
    path.rename(new_path)
    manifest["run_id"] = run_id
    return new_path


def _legacy_fixture(tmp_path):
    legacy_dir = tmp_path / "data"
    legacy_dir.mkdir()
    frame = pd.DataFrame({
        "sample_index": list(range(1, 1954)), "ts": list(range(1, 1954)),
        "prompt_version": ["v3"] * 1953, "year": [2025] * 1953,
        "quarter": ["Q1"] * 1953, "vol_q": ["v1"] * 1953, "dir_t": ["up"] * 1953,
    })
    (legacy_dir / "metering_2000_failures.json").write_text(
        json.dumps([{"sample_index": i, "error": "legacy"} for i in range(1954, 2001)]),
        encoding="utf-8",
    )
    return legacy_dir, frame
