"""G3a evaluator: raw Jev ranking and incremental value, before calibration."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from . import config, labels as lab
from .grid import cell_key, project_barrier_surfaces, sl_key, tp_key
from .jev import build_questions, canonical_hash, prompt_contract
from .metrics import ece, multiclass_brier, reliability
from .meter import METER_IDENTITY_VERSION


RESOLVED = (int(lab.Outcome.SL_FIRST), int(lab.Outcome.TP_FIRST))


def _require_frozen_prompt_version(frame: pd.DataFrame, expected: str | None, scope: str) -> None:
    if "prompt_version" not in frame or frame["prompt_version"].isna().any():
        raise ValueError(f"{scope} prompt_version is not frozen")
    if expected is not None and not frame["prompt_version"].eq(expected).all():
        raise ValueError(f"{scope} prompt_version is not frozen")


def _require_integer_column(frame: pd.DataFrame, column: str, scope: str) -> pd.Series:
    if column not in frame:
        raise ValueError(f"{scope} is missing {column}")
    values = pd.to_numeric(frame[column], errors="coerce")
    numeric = values.to_numpy(float)
    if values.isna().any() or not np.isfinite(numeric).all() or not np.equal(numeric, numeric.astype(np.int64)).all():
        raise ValueError(f"{scope} {column} must be finite integer-valued")
    return values.astype(np.int64)


def _validate_prediction_artifact(predictions: str | Path, frame: pd.DataFrame,
                                  data_dir: str | Path = "data",
                                  expected_prompt_version: str | None = None,
                                  grid: dict | None = None,
                                  expected_model_id: str | None = None,
                                  expected_symbol: str | None = None) -> str:
    """Reject incomplete or mismatched new meter artifacts before any output."""
    path = Path(predictions)
    if path.name == "metering_2000.parquet":
        if path.resolve() != (Path(data_dir) / "metering_2000.parquet").resolve():
            raise ValueError("legacy G3a compatibility is restricted to data/metering_2000.parquet")
        required = {"sample_index", "ts", "prompt_version", "year", "quarter", "vol_q", "dir_t"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"legacy G3a artifact is missing columns: {missing}")
        if frame.empty:
            raise ValueError("legacy G3a artifact is empty")
        sample_indices = pd.to_numeric(frame["sample_index"], errors="coerce")
        if sample_indices.isna().any() or not np.equal(sample_indices, sample_indices.astype(np.int64)).all():
            raise ValueError("legacy G3a sample_index must be integer-valued")
        success_indices = set(sample_indices.astype(np.int64))
        if len(success_indices) != len(frame) or not success_indices <= set(range(1, 2001)):
            raise ValueError("legacy G3a success sample_index is duplicated or out of range")
        failures_path = Path(data_dir) / "metering_2000_failures.json"
        try:
            failures = json.loads(failures_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"legacy G3a failure sidecar is unreadable: {failures_path}") from exc
        if not isinstance(failures, list) or len(failures) != 47:
            raise ValueError("legacy G3a failure sidecar must contain exactly 47 samples")
        failure_values = [item.get("sample_index") if isinstance(item, dict) else None for item in failures]
        failure_indices = pd.to_numeric(pd.Series(failure_values), errors="coerce")
        if failure_indices.isna().any() or not np.equal(failure_indices, failure_indices.astype(np.int64)).all():
            raise ValueError("legacy G3a failure sample_index must be integer-valued")
        failure_indices = set(failure_indices.astype(np.int64))
        expected_indices = set(range(1, 2001))
        if len(failure_indices) != 47 or not failure_indices <= expected_indices:
            raise ValueError("legacy G3a failure sample_index is duplicated or out of range")
        if success_indices & failure_indices or success_indices | failure_indices != expected_indices:
            raise ValueError("legacy G3a success/failure sample indices do not cover 1..2000")
        _require_frozen_prompt_version(frame, expected_prompt_version, "legacy G3a")
        if grid is not None:
            cell_columns = []
            for side in ("long", "short"):
                for tp in grid["tp"]:
                    for sl in grid["sl"]:
                        cell = cell_key(side, tp, sl)
                        cell_columns.extend([
                            tp_key(side, tp, sl), sl_key(side, tp, sl), f"outcome_{cell}",
                        ])
            missing_cells = sorted(set(cell_columns) - set(frame.columns))
            if missing_cells:
                raise ValueError("legacy G3a artifact is missing required grid columns")
        return "legacy-2000"
    match = re.fullmatch(r"metering_(\d+)_([0-9a-f]{16})\.parquet", path.name)
    if not match:
        raise ValueError(
            "G3a requires a complete run-identified meter artifact or the explicit "
            "legacy data/metering_2000.parquet compatibility path"
        )
    n = int(match.group(1))
    run_id = match.group(2)
    manifest_path = path.with_name(f"{path.stem}_manifest.json")
    if not manifest_path.is_file():
        raise ValueError(f"G3a meter manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"G3a meter manifest is unreadable: {manifest_path}") from exc
    identity = manifest.get("identity")
    completed = manifest.get("completed")
    if not isinstance(identity, dict):
        raise ValueError("G3a meter manifest run_id/identity mismatch")
    derived_run_id = canonical_hash(identity)[:16]
    if manifest.get("run_id") != run_id or derived_run_id != run_id:
        raise ValueError("G3a meter manifest run_id/identity mismatch")
    if manifest.get("complete") is not True:
        raise ValueError("G3a refuses a meter artifact whose manifest is not complete")
    if identity.get("identity_version") != METER_IDENTITY_VERSION:
        raise ValueError("G3a meter identity_version is unsupported")
    if expected_model_id is not None and identity.get("model_id") != expected_model_id:
        raise ValueError("G3a meter model_id does not match current config")
    if expected_symbol is not None and identity.get("symbol") != expected_symbol:
        raise ValueError("G3a meter symbol does not match current config")
    if expected_prompt_version and identity.get("prompt_version") != expected_prompt_version:
        raise ValueError("G3a meter prompt_version does not match current config")
    if grid is not None:
        if identity.get("horizon_bars") != int(grid["horizon_bars"]):
            raise ValueError("G3a meter horizon does not match current config")
        for field in ("tp", "sl"):
            if list(identity.get(field, [])) != [float(value) for value in grid[field]]:
                raise ValueError(f"G3a meter {field} grid does not match current config")
        if identity.get("questions_sha256") != canonical_hash(build_questions(grid)):
            raise ValueError("G3a meter questions do not match current config")
        if identity.get("contract_sha256") != canonical_hash(prompt_contract(int(grid["horizon_bars"]))):
            raise ValueError("G3a meter contract does not match current config")
    if identity.get("requested") != n or not isinstance(completed, dict):
        raise ValueError("G3a meter manifest sample count is inconsistent")
    sample_indices = _require_integer_column(frame, "sample_index", "G3a meter")
    _require_integer_column(frame, "ts", "G3a meter")
    expected_indices = {str(index) for index in range(1, n + 1)}
    if set(completed) != expected_indices:
        raise ValueError("G3a meter manifest does not contain every completed sample")
    if len(frame) != n or "sample_index" not in frame:
        raise ValueError("G3a meter parquet sample count is inconsistent with its manifest")
    _require_frozen_prompt_version(frame, expected_prompt_version, "G3a meter")
    ordered = frame.sort_values("sample_index")
    if sample_indices.loc[ordered.index].tolist() != list(range(1, n + 1)):
        raise ValueError("G3a meter parquet sample indices are incomplete or duplicated")
    picked = [
        {"sample_index": int(index), "ts": int(ts)}
        for index, ts in zip(ordered["sample_index"], ordered["ts"])
    ]
    if identity.get("picked_ts_sha256") != canonical_hash(picked):
        raise ValueError("G3a meter picked-sample identity mismatch")
    state_hashes = identity.get("request_state_hashes")
    if not isinstance(state_hashes, list) or "request_state_hash" not in ordered:
        raise ValueError("G3a meter request-state identity is missing")
    if ordered["request_state_hash"].astype(str).tolist() != state_hashes:
        raise ValueError("G3a meter request-state identity mismatch")
    if not identity.get("questions_sha256"):
        raise ValueError("G3a meter questions/schema identity is missing")
    return run_id


def _raw_probabilities(p_tp, p_sl):
    p_tp = np.asarray(p_tp, dtype=float)
    p_sl = np.asarray(p_sl, dtype=float)
    p_sum = p_tp + p_sl
    invalid = (
        ~np.isfinite(p_tp) | ~np.isfinite(p_sl)
        | (p_tp < 0) | (p_tp > 1) | (p_sl < 0) | (p_sl > 1)
        | (p_sum > 1 + 1e-9)
    )
    return p_tp, p_sl, 1.0 - p_sum, p_sum, invalid


def to_long(frame: pd.DataFrame, grid: dict) -> pd.DataFrame:
    """Expand successful responses into raw, labelled barrier-cell rows."""
    rows = []
    for side in ("long", "short"):
        for tp in grid["tp"]:
            for sl in grid["sl"]:
                cell = cell_key(side, tp, sl)
                p_tp, p_sl, p_timeout, p_sum, invalid = _raw_probabilities(
                    frame[tp_key(side, tp, sl)], frame[sl_key(side, tp, sl)]
                )
                rows.append(pd.DataFrame({
                    "ts": frame.ts.to_numpy(np.int64),
                    "year": frame.year.to_numpy(np.int16),
                    "quarter": frame.quarter.astype(str).to_numpy(),
                    "vol_q": frame.vol_q.astype(str).to_numpy(),
                    "dir_t": frame.dir_t.astype(str).to_numpy(),
                    "side": side, "tp": float(tp), "sl": float(sl), "cell": cell,
                    "outcome": frame[f"outcome_{cell}"].to_numpy(np.int8),
                    "p_sl": p_sl, "p_tp": p_tp, "p_timeout": p_timeout,
                    "p_sum": p_sum, "prob_invalid": invalid,
                }))
    return pd.concat(rows, ignore_index=True)


def _resolved(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.outcome.isin(RESOLVED)]


def _mean_score_auc(df: pd.DataFrame, score_col: str) -> float:
    values = []
    for _, group in df.groupby("cell", observed=True):
        group = _resolved(group)
        y = (group.outcome.to_numpy() == int(lab.Outcome.TP_FIRST)).astype(int)
        if len(y) > 1 and np.unique(y).size == 2:
            values.append(roc_auc_score(y, group[score_col].to_numpy(float)))
    return float(np.mean(values)) if values else math.nan


def _mean_cell_auc(df: pd.DataFrame, tp_col: str = "p_tp", sl_col: str = "p_sl") -> float:
    df = _resolved(df).copy()
    df["_score"] = df[tp_col] / np.maximum(df[tp_col] + df[sl_col], 1e-12)
    return _mean_score_auc(df, "_score")


def _mean_binary_brier(df: pd.DataFrame, score_col: str) -> float:
    values = []
    for _, group in df.groupby("cell", observed=True):
        group = _resolved(group)
        if len(group) == 0:
            continue
        y = (group.outcome.to_numpy() == int(lab.Outcome.TP_FIRST)).astype(float)
        values.append(float(np.mean((group[score_col].to_numpy(float) - y) ** 2)))
    return float(np.mean(values)) if values else math.nan


def _metric_table(df: pd.DataFrame) -> dict:
    all_invalid = df.prob_invalid.to_numpy(bool)
    all_sum_violations = df.p_sum.to_numpy(float) > 1 + 1e-9
    df = df[df.outcome != int(lab.Outcome.AMBIGUOUS)]
    y = df.outcome.to_numpy(np.int8)
    p = df[["p_sl", "p_tp", "p_timeout"]].to_numpy(float)
    invalid = df.prob_invalid.to_numpy(bool)
    out = {
        "n": int(len(df)),
        "invalid_probability_cells": int(invalid.sum()),
        "sum_violation_cells": int((df.p_sum.to_numpy(float) > 1 + 1e-9).sum()),
        "invalid_probability_cells_all": int(all_invalid.sum()),
        "sum_violation_cells_all": int(all_sum_violations.sum()),
    }
    for k, name in enumerate(("sl_first", "tp_first", "timeout")):
        y_bin = (y == k).astype(int)
        out[f"base_{name}"] = float(y_bin.mean())
        out[f"auc_{name}"] = (
            float(roc_auc_score(y_bin, p[:, k]))
            if 0 < y_bin.sum() < len(y_bin) else math.nan
        )
        out[f"ece_{name}"] = math.nan if invalid.any() else ece(y_bin, p[:, k])
    out["brier_mc"] = math.nan if invalid.any() else multiclass_brier(y, p)
    out["logloss"] = math.nan if invalid.any() else log_loss(y, p, labels=[0, 1, 2])
    out["auc_tp_vs_sl"] = _mean_pooled_auc(df)
    out["within_cell_auc"] = _mean_cell_auc(df)
    return out


def _mean_pooled_auc(df: pd.DataFrame, score_col: str | None = None) -> float:
    df = _resolved(df)
    y = (df.outcome.to_numpy() == int(lab.Outcome.TP_FIRST)).astype(int)
    if len(y) < 2 or np.unique(y).size < 2:
        return math.nan
    if score_col:
        score = df[score_col].to_numpy(float)
    else:
        score = df.p_tp.to_numpy(float) / np.maximum(df.p_tp.to_numpy(float) + df.p_sl.to_numpy(float), 1e-12)
    return float(roc_auc_score(y, score))


def _day_bootstrap(df: pd.DataFrame, statistic, n: int = 400, seed: int = 20260921):
    days = df.ts.to_numpy(np.int64) // 86_400_000
    unique = np.unique(days)
    by_day = {d: np.flatnonzero(days == d) for d in unique}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        picked = np.concatenate([by_day[d] for d in rng.choice(unique, len(unique), replace=True)])
        value = statistic(df.iloc[picked])
        if np.isfinite(value):
            values.append(value)
    return (
        float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))
    ) if values else (math.nan, math.nan)


def _load_oof_sample(data_dir: str | Path, timestamps: np.ndarray) -> pd.DataFrame:
    wanted = set(int(x) for x in timestamps)
    columns = ["ts", "side", "tp", "sl", "model", "outcome",
               "p_sl_first", "p_tp_first", "p_timeout"]
    frames = []
    for path in sorted((Path(data_dir) / "oof").glob("fold_*.parquet")):
        part = pd.read_parquet(path, columns=columns)
        part = part[part.ts.isin(wanted)]
        if len(part):
            frames.append(part)
    if not frames:
        raise RuntimeError("no OOF predictions overlap the Jev sample")
    return pd.concat(frames, ignore_index=True)


def _merge_baseline(jev: pd.DataFrame, oof: pd.DataFrame, model: str) -> pd.DataFrame:
    base = oof[oof.model.astype(str) == model].copy()
    base["cell"] = [cell_key(side, tp, sl) for side, tp, sl in zip(base.side, base.tp, base.sl)]
    base = base.rename(columns={"p_sl_first": "base_p_sl", "p_tp_first": "base_p_tp",
                                "p_timeout": "base_p_timeout"})
    cols = ["ts", "cell", "outcome", "base_p_sl", "base_p_tp", "base_p_timeout"]
    return jev.merge(base[cols], on=["ts", "cell", "outcome"], how="inner", validate="one_to_one")


def _stacker(jev: pd.DataFrame, oof: pd.DataFrame, horizon_bars: int) -> dict:
    """Chronological stacker with within-cell AUC and day-block deltas."""
    g = _merge_baseline(jev, oof, "lgbm").sort_values("ts")
    resolved = _resolved(g)
    if len(resolved) < 100:
        return {"status": "insufficient_overlap", "n": int(len(resolved))}

    split = int(resolved.ts.quantile(0.70))
    purge_ms = horizon_bars * 15 * 60_000
    train = resolved[resolved.ts < split - purge_ms].copy()
    test = g[g.ts >= split].copy()
    if len(train) < 50 or len(test) < 50:
        return {"status": "insufficient_time_split", "n_train": len(train), "n_test": len(test)}

    def conditional(frame, prefix):
        tp = frame[f"{prefix}_p_tp"].to_numpy(float)
        sl = frame[f"{prefix}_p_sl"].to_numpy(float)
        return tp / np.maximum(tp + sl, 1e-12)

    def logit(probability):
        probability = np.clip(probability, 1e-6, 1 - 1e-6)
        return np.log(probability / (1 - probability))

    train_base = conditional(train, "base")
    train_jev = train.p_tp.to_numpy(float) / np.maximum(train.p_tp.to_numpy(float) + train.p_sl.to_numpy(float), 1e-12)
    test_base = conditional(test, "base")
    test_jev = test.p_tp.to_numpy(float) / np.maximum(test.p_tp.to_numpy(float) + test.p_sl.to_numpy(float), 1e-12)
    x_train = np.column_stack([logit(train_base), logit(train_jev)])
    x_test = np.column_stack([logit(test_base), logit(test_jev)])
    y_train = (train.outcome.to_numpy() == int(lab.Outcome.TP_FIRST)).astype(int)
    model = LogisticRegression(max_iter=1000).fit(x_train, y_train)
    test = test.copy()
    test["score_base"] = test_base
    test["score_jev"] = test_jev
    test["score_stack"] = model.predict_proba(x_test)[:, 1]
    test_resolved = _resolved(test)
    y_test = (test_resolved.outcome.to_numpy() == int(lab.Outcome.TP_FIRST)).astype(int)

    def deltas(sample):
        return (
            _mean_score_auc(sample, "score_stack") - _mean_score_auc(sample, "score_base"),
            _mean_binary_brier(sample, "score_stack") - _mean_binary_brier(sample, "score_base"),
        )

    days = test.ts.to_numpy(np.int64) // 86_400_000
    unique = np.unique(days)
    by_day = {d: np.flatnonzero(days == d) for d in unique}
    rng = np.random.default_rng(20260921)
    boot = []
    for _ in range(400):
        idx = np.concatenate([by_day[d] for d in rng.choice(unique, len(unique), replace=True)])
        boot.append(deltas(test.iloc[idx]))
    boot = np.asarray(boot)
    delta_auc, _ = deltas(test)
    baseline_brier = _mean_binary_brier(test, "score_base")
    stacker_brier = _mean_binary_brier(test, "score_stack")
    orthogonality = float(np.corrcoef(logit(test_resolved.score_jev), y_test - test_resolved.score_base)[0, 1])
    return {
        "status": "ok", "n_train": len(train), "n_test": len(test), "split_ts": split,
        "baseline_within_cell_auc": _mean_score_auc(test, "score_base"),
        "jev_within_cell_auc": _mean_score_auc(test, "score_jev"),
        "stacker_within_cell_auc": _mean_score_auc(test, "score_stack"),
        "delta_auc": float(delta_auc),
        "delta_auc_lo": float(np.percentile(boot[:, 0], 2.5)),
        "delta_auc_hi": float(np.percentile(boot[:, 0], 97.5)),
        "baseline_brier": baseline_brier,
        "stacker_brier": stacker_brier,
        "delta_brier": float((baseline_brier - stacker_brier) / baseline_brier)
        if baseline_brier else math.nan,
        "baseline_weight": float(model.coef_[0, 0]), "jev_weight": float(model.coef_[0, 1]),
        "intercept": float(model.intercept_[0]),
        "orthogonality_corr_logit_jev_vs_baseline_residual": orthogonality,
    }


def project_monotone(jev: pd.DataFrame, grid: dict) -> tuple[pd.DataFrame, dict]:
    projected = jev.copy()
    for _, indices in projected.groupby("ts", sort=False).groups.items():
        bar = projected.loc[indices]
        for side in ("long", "short"):
            sub = bar[bar.side == side]
            tp_matrix = sub.pivot(index="tp", columns="sl", values="p_tp").reindex(index=grid["tp"], columns=grid["sl"]).to_numpy()
            sl_matrix = sub.pivot(index="tp", columns="sl", values="p_sl").reindex(index=grid["tp"], columns=grid["sl"]).to_numpy()
            tp_matrix, sl_matrix, timeout_matrix, sum_matrix = project_barrier_surfaces(
                tp_matrix, sl_matrix
            )
            for i, tp in enumerate(grid["tp"]):
                for j, sl in enumerate(grid["sl"]):
                    match = sub[(sub.tp == tp) & (sub.sl == sl)].index[0]
                    projected.loc[match, "p_tp"] = tp_matrix[i, j]
                    projected.loc[match, "p_sl"] = sl_matrix[i, j]
                    projected.loc[match, "p_timeout"] = timeout_matrix[i, j]
                    projected.loc[match, "p_sum"] = sum_matrix[i, j]
                    projected.loc[match, "prob_invalid"] = (
                        not np.isfinite(sum_matrix[i, j]) or sum_matrix[i, j] > 1 + 1e-9
                    )
    delta = {
        "mean_abs_p_tp": float(np.mean(np.abs(projected.p_tp - jev.p_tp))),
        "mean_abs_p_sl": float(np.mean(np.abs(projected.p_sl - jev.p_sl))),
        "mean_abs_p_timeout": float(np.mean(np.abs(projected.p_timeout - jev.p_timeout))),
        "max_abs_any": float(np.max(np.abs(projected[["p_tp", "p_sl", "p_timeout"]].to_numpy() - jev[["p_tp", "p_sl", "p_timeout"]].to_numpy()))),
    }
    return projected, delta


def run(predictions: str | Path = "data/metering_2000.parquet", data_dir: str | Path = "data") -> Path:
    cfg = config.load()
    frame = pd.read_parquet(predictions)
    _validate_prediction_artifact(
        predictions, frame, data_dir, cfg["jev"]["prompt_version"], cfg["grid"],
        cfg["jev"]["model_id"], cfg["symbols"][0]
    )
    if "response_valid" in frame:
        frame = frame[frame.response_valid.fillna(False)].copy()
    if frame.empty:
        raise ValueError("G3a artifact has no valid responses")
    expected_prompt_version = cfg["jev"]["prompt_version"]
    _require_frozen_prompt_version(frame, expected_prompt_version, "G3a")
    jev = to_long(frame, cfg["grid"])
    out_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jev.to_parquet(out_dir / "g3a_long.parquet", index=False)

    invalid_ts = set(jev.loc[jev.prob_invalid, "ts"].astype(np.int64))
    valid_jev = jev[~jev.ts.isin(invalid_ts)].copy()
    raw_all = _metric_table(jev)
    raw = _metric_table(valid_jev)
    raw["invalid_probability_cells_all"] = raw_all["invalid_probability_cells_all"]
    raw["sum_violation_cells_all"] = raw_all["sum_violation_cells_all"]
    ci = _day_bootstrap(valid_jev, _mean_cell_auc)
    projected, projection_delta = project_monotone(valid_jev, cfg["grid"])
    projected.to_parquet(out_dir / "g3a_projected_long.parquet", index=False)
    projected_metrics = _metric_table(projected)
    oof = _load_oof_sample(data_dir, frame.ts.unique())
    base = _merge_baseline(valid_jev, oof, "climatology")
    lgbm = _merge_baseline(valid_jev, oof, "lgbm")
    base_metrics = _baseline_metrics(base)
    lgbm_metrics = _baseline_metrics(lgbm)
    stack = _stacker(valid_jev, oof, cfg["grid"]["horizon_bars"])

    valid_raw = valid_jev[~valid_jev.prob_invalid]
    reliability((valid_raw.outcome.to_numpy() == int(lab.Outcome.TP_FIRST)).astype(int), valid_raw.p_tp.to_numpy()).to_csv(out_dir / "g3a_reliability_tp.csv", index=False)
    bucket = (valid_jev.groupby("tp", observed=True)
              .agg(n=("outcome", "size"), p_tp_mean=("p_tp", "mean"),
                   observed_tp=("outcome", lambda x: (x == int(lab.Outcome.TP_FIRST)).mean())))
    bucket["calibration_bias"] = bucket.p_tp_mean - bucket.observed_tp
    bucket.to_csv(out_dir / "g3a_tp_bucket_calibration.csv")

    strata = []
    for field in ("year", "quarter", "vol_q", "dir_t"):
        for value, group in valid_jev.groupby(field, observed=True):
            strata.append({"stratum": field, "value": str(value), **_metric_table(group)})
    strata_df = pd.DataFrame(strata)
    strata_df.to_csv(out_dir / "g3a_strata.csv", index=False)

    report = _render_report(frame, jev, valid_jev, raw, ci, base_metrics, lgbm_metrics, stack,
                            bucket, strata_df, predictions, projected_metrics,
                            projection_delta, cfg["grid"], cfg["grid"]["horizon_bars"])
    Path("docs/g3a_report.md").write_text(report, encoding="utf-8")
    (out_dir / "g3a_metrics.json").write_text(json.dumps({
        "raw": raw, "within_cell_auc_ci": ci, "projected": projected_metrics,
        "projection_delta": projection_delta, "climatology": base_metrics,
        "lgbm": lgbm_metrics, "stacker": stack,
    }, indent=2, allow_nan=True), encoding="utf-8")
    return Path("docs/g3a_report.md")


def _baseline_metrics(df: pd.DataFrame) -> dict:
    p = df[["base_p_sl", "base_p_tp", "base_p_timeout"]].to_numpy(float)
    p = p / p.sum(axis=1, keepdims=True)
    y = df.outcome.to_numpy(np.int8)
    out = {"n": len(df), "brier_mc": multiclass_brier(y, p), "logloss": log_loss(y, p, labels=[0, 1, 2])}
    scores = df[["cell", "outcome", "base_p_tp", "base_p_sl"]].rename(
        columns={"base_p_tp": "p_tp", "base_p_sl": "p_sl"}
    )
    out["within_cell_auc"] = _mean_cell_auc(scores)
    out["auc_tp_vs_sl"] = _mean_pooled_auc(scores)
    return out


def _fmt(value):
    return "n/a" if value is None or not np.isfinite(value) else f"{value:.4f}"


def gate_decision(ci: tuple[float, float], delta_auc: float, delta_brier: float,
                  stack_status: str) -> str:
    """Apply the frozen G3a gate to validation-clean, within-cell metrics."""
    auc_lo, auc_hi = ci
    if stack_status != "ok":
        return "INCONCLUSIVE — stacker unavailable"
    if auc_hi < 0.51 and delta_brier <= 0:
        return "STOP — no evidence of ranking signal"
    if auc_lo > 0.50 and delta_auc > 0:
        return "PASS G4 — ranking gate passed"
    return "INCONCLUSIVE — proceed to G3b"


def _render_report(frame, jev, valid_jev, raw, ci, clim, lgbm, stack, bucket, strata,
                   predictions, projected, projection_delta, grid, horizon):
    auc_lo, auc_hi = ci
    delta_brier = stack.get("delta_brier", math.nan)
    delta_auc = stack.get("delta_auc", math.nan)
    gate = gate_decision(ci, delta_auc, delta_brier, stack.get("status", "missing"))
    cell_count = len(grid["tp"]) * len(grid["sl"]) * 2
    strata_text = strata[["stratum", "value", "n", "within_cell_auc", "brier_mc", "ece_tp_first"]].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    bucket_text = bucket.reset_index().to_string(index=False, float_format=lambda x: f"{x:.4f}")
    raw_prob_status = (
        "valid on validation-clean bars; invalid responses excluded"
        if raw["invalid_probability_cells_all"] else "valid"
    )
    return f"""# G3a raw Jev ranking and incremental-value report

Input: `{predictions}`  
Prompt contract: `{frame.prompt_version.iloc[0]}` (frozen)  

## Gate

**{gate}**

Gate is applied exactly as specified: stop when `auc_hi < 0.51 && delta_brier <= 0`,
pass when `auc_lo > 0.50 && delta_auc > 0`, otherwise continue to G3b. No PnL,
Sharpe, EV threshold, or TP/SL optimization is used.

## Sample and raw probabilities

- Successful Jev prediction bars: **{len(frame):,}**
- Bars excluded from G3a because at least one cell failed validation: **{len(frame) - valid_jev.ts.nunique():,}**
- Valid bars used for G3a metrics and gate: **{valid_jev.ts.nunique():,}**
- Prediction cells: **{len(jev):,}** ({cell_count} cells/bar)
- Horizon: **{horizon}** 15m bars
- Resolved cells: **{int(jev.outcome.isin(RESOLVED).sum()):,}**
- Raw Jev within-cell `auc_tp_vs_sl`: **{_fmt(raw['within_cell_auc'])}**
- Day-block 95% CI: **[{_fmt(auc_lo)}, {_fmt(auc_hi)}]**
- Probability status: **{raw_prob_status}**
- `p_tp + p_sl > 1` violations: **{raw['sum_violation_cells_all']:,}**
- Invalid probability cells: **{raw['invalid_probability_cells_all']:,}**
- Raw multiclass Brier/ECE: **reported only on validation-clean bars**

Direction AUC remains reportable because it uses the raw TP-vs-SL ordering;
probability-quality metrics are never silently normalized.

## Same-test-set within-cell baseline comparison

| model | within-cell AUC | pooled AUC | multiclass Brier |
|---|---:|---:|---:|
| Jev raw | {_fmt(raw['within_cell_auc'])} | {_fmt(raw['auc_tp_vs_sl'])} | {_fmt(raw['brier_mc'])} |
| climatology | {_fmt(clim['within_cell_auc'])} | {_fmt(clim['auc_tp_vs_sl'])} | {_fmt(clim['brier_mc'])} |
| LightGBM OOF | {_fmt(lgbm['within_cell_auc'])} | {_fmt(lgbm['auc_tp_vs_sl'])} | {_fmt(lgbm['brier_mc'])} |

## Chronological two-input stacker

The model uses `logit(P_baseline)` and `logit(P_jev)`, with time separation and
a {horizon}-bar purge. All AUC values and ΔAUC are within-cell means. Bootstrap
resamples complete days, including all available cell rows for each day.

```text
{json.dumps(stack, indent=2, sort_keys=True)}
```

Orthogonality is `corr(logit(p_jev), y - p_baseline)`, not correlation between
the two residual series.

## Monotonic projection and TP-bucket diagnostics

| metric | raw | monotonic projection |
|---|---:|---:|
| within-cell AUC | {_fmt(raw['within_cell_auc'])} | {_fmt(projected['within_cell_auc'])} |
| mean abs TP adjustment | 0.0000 | {_fmt(projection_delta['mean_abs_p_tp'])} |
| mean abs SL adjustment | 0.0000 | {_fmt(projection_delta['mean_abs_p_sl'])} |
| max adjustment | 0.0000 | {_fmt(projection_delta['max_abs_any'])} |

```text
{bucket_text}
```

## Strata

```text
{strata_text}
```

Artifacts: `data/g3a_long.parquet`, `data/g3a_projected_long.parquet`,
`data/g3a_metrics.json`, `data/g3a_reliability_tp.csv`,
`data/g3a_tp_bucket_calibration.csv`, `data/g3a_strata.csv`.

G3a has not passed the ranking gate unless the gate above explicitly says
`PASS G4`. EV optimization and formal backtesting remain blocked until G4 passes.
"""


if __name__ == "__main__":
    print(run())
