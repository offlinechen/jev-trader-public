"""Matched, fold-aware G3b ranking confirmation; no strategy or PnL tuning."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from . import config, labels as lab
from .grid import cell_key


KEYS = ["ts", "side", "tp", "sl", "fold", "outcome"]
RESOLVED = (int(lab.Outcome.SL_FIRST), int(lab.Outcome.TP_FIRST))
MIN_CLASS_ROWS_PER_CELL = 20
MIN_BOOTSTRAP_DAYS = 30
MIN_VALID_BOOTSTRAP_FRACTION = 0.90
MIN_BOOTSTRAP_REPLICATES = 400


def _credit_exhaustion_count(manifest: dict, frame: pd.DataFrame) -> int:
    statuses = manifest.get("budget", {}).get("by_status", {})
    status_count = sum(int(count) for status, count in statuses.items() if str(status) == "402")
    errors = []
    details = manifest.get("failure_details", {})
    entries = details.values() if isinstance(details, dict) else details if isinstance(details, list) else []
    for entry in entries:
        if isinstance(entry, dict):
            errors.append(str(entry.get("error", "")))
        else:
            errors.append(str(entry))
    if "request_error" in frame:
        errors.extend(frame.request_error.dropna().astype(str))
    marker = re.compile(
        r"\b402\b|insufficient\s+(?:credits?|balance)|"
        r"(?:credit|balance)\s+(?:exhausted|depleted)|payment\s+required",
        re.IGNORECASE,
    )
    error_count = sum(bool(marker.search(error)) for error in errors)
    return max(status_count, error_count)


def _has_credit_exhaustion(manifest: dict, frame: pd.DataFrame) -> bool:
    return _credit_exhaustion_count(manifest, frame) > 0


def _conditional(tp, sl):
    total = np.asarray(tp, dtype=float) + np.asarray(sl, dtype=float)
    return np.asarray(tp, dtype=float) / np.maximum(total, 1e-12)


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _within_cell_auc(frame: pd.DataFrame, score: str, weight=None) -> float:
    aucs = []
    for _, group in frame.groupby("cell", observed=True, sort=True):
        y = group.y.to_numpy(np.int8)
        if np.unique(y).size != 2:
            continue
        w = None if weight is None else np.asarray(weight)[group.index]
        if w is not None and w.sum() == 0:
            continue
        aucs.append(roc_auc_score(y, group[score].to_numpy(float), sample_weight=w))
    return float(np.mean(aucs)) if aucs else math.nan


def _within_cell_brier(frame: pd.DataFrame, score: str, weight=None) -> float:
    values = []
    for _, group in frame.groupby("cell", observed=True, sort=True):
        w = np.ones(len(group)) if weight is None else np.asarray(weight)[group.index]
        if w.sum() > 0:
            values.append(float(np.average((group[score].to_numpy(float) - group.y.to_numpy(float)) ** 2, weights=w)))
    return float(np.mean(values)) if values else math.nan


def _day_bootstrap(frame: pd.DataFrame, n: int, seed: int) -> dict[str, tuple[float, float]]:
    days = frame.ts.to_numpy(np.int64) // 86_400_000
    unique = np.unique(days)
    day_index = {day: i for i, day in enumerate(unique)}
    row_day = np.array([day_index[d] for d in days], dtype=int)
    rng = np.random.default_rng(seed)
    names = ("jev_auc", "lgbm_auc", "stack_auc", "delta_auc", "relative_delta_brier")
    samples = {name: [] for name in names}
    cells = frame.cell.to_numpy()
    y = frame.y.to_numpy(np.int8)
    scores = {name: frame[name].to_numpy(float) for name in ("jev", "lgbm", "stack")}
    cell_indexes = [np.flatnonzero(cells == cell) for cell in np.unique(cells)]
    for _ in range(n):
        multiplicity = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        weights = multiplicity[row_day]
        def auc(score):
            vals = []
            for idx in cell_indexes:
                active = weights[idx] > 0
                yy, ss, ww = y[idx][active], score[idx][active], weights[idx][active]
                if len(yy) == 0 or np.unique(yy).size != 2:
                    return math.nan
                vals.append(roc_auc_score(yy, ss, sample_weight=ww))
            return float(np.mean(vals)) if vals else math.nan
        def brier(score):
            vals = []
            for idx in cell_indexes:
                if weights[idx].sum():
                    vals.append(float(np.average((score[idx] - y[idx]) ** 2, weights=weights[idx])))
            return float(np.mean(vals)) if vals else math.nan
        ja, la, sa = (auc(scores[k]) for k in ("jev", "lgbm", "stack"))
        lb, sb = brier(scores["lgbm"]), brier(scores["stack"])
        vals = (ja, la, sa, sa - la, (lb - sb) / lb if lb > 0 else math.nan)
        if np.isfinite(vals).all():
            for name, value in zip(names, vals):
                samples[name].append(value)
    result = {
        name: (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))
        if values else (math.nan, math.nan) for name, values in samples.items()
    }
    result["valid_replicates"] = min((len(values) for values in samples.values()), default=0)
    return result


def evaluate_frames(jev: pd.DataFrame, baseline: pd.DataFrame, *, horizon_bars: int,
                   n_boot: int = 400, seed: int = 20260921,
                   min_train: int = 50) -> dict:
    """Return G3b metrics on the exact Jev/LGBM/stack triple intersection."""
    required_j = set(KEYS) - {"fold"} | {"p_tp", "p_sl", "prob_invalid"}
    required_b = set(KEYS) | {"p_tp_first", "p_sl_first", "p_timeout"}
    if missing := required_j - set(jev):
        raise ValueError(f"Jev rows missing columns: {sorted(missing)}")
    if missing := required_b - set(baseline):
        raise ValueError(f"baseline rows missing columns: {sorted(missing)}")
    j = jev.copy()
    b = baseline.copy()
    j["ts"] = j.ts.astype(np.int64)
    b["ts"] = b.ts.astype(np.int64)
    j[["tp", "sl"]] = j[["tp", "sl"]].astype(float).round(6)
    b[["tp", "sl"]] = b[["tp", "sl"]].astype(float).round(6)
    j["side"] = j.side.astype(str)
    b["side"] = b.side.astype(str)
    if j.duplicated([c for c in KEYS if c != "fold"]).any() or b.duplicated(KEYS).any():
        raise ValueError("duplicate Jev or baseline cell observations")
    prob = j[["p_tp", "p_sl"]].to_numpy(float)
    invalid_probability_cell = (
        ~np.isfinite(prob).all(axis=1) | (prob < 0).any(axis=1)
        | (prob > 1).any(axis=1) | (prob.sum(axis=1) > 1 + 1e-9)
        | j.prob_invalid.fillna(True).to_numpy(bool)
    )
    if "response_valid" not in j:
        j["response_valid"] = True
    attempted_bars = j.ts.nunique()
    response_failed_ts = set(j.loc[~j.response_valid.fillna(False), "ts"].astype(int))
    successful_rows = j.response_valid.fillna(False).to_numpy(bool)
    invalid_probability_ts = set(j.loc[invalid_probability_cell & successful_rows, "ts"].astype(int))
    invalid_ts = response_failed_ts | invalid_probability_ts
    invalid_bars = len(invalid_ts)
    successful_bars = attempted_bars - len(response_failed_ts)
    all_cell_valid_bars = successful_bars - len(invalid_probability_ts)
    invalid_probability_cells = int((invalid_probability_cell & successful_rows).sum())
    j = j[~j.ts.isin(invalid_ts)].copy()
    b = b[b.ts.isin(j.ts.unique())].copy()
    targets = j[j.outcome.isin(RESOLVED)].copy()
    b = b.rename(columns={"outcome": "base_outcome", "p_tp_first": "base_tp",
                          "p_sl_first": "base_sl"})
    match_keys = ["ts", "side", "tp", "sl"]
    paired = targets.merge(
        b[match_keys + ["fold", "base_outcome", "base_tp", "base_sl", "p_timeout"]],
        on=match_keys, how="left", validate="one_to_one", indicator=True,
    )
    missing = paired._merge.ne("both")
    mismatch = paired.base_outcome.notna() & paired.base_outcome.ne(paired.outcome)
    if missing.any() or mismatch.any():
        raise ValueError(
            "missing LightGBM OOF rows or label mismatch for valid resolved Jev targets "
            f"(missing={int(missing.sum())}, label_mismatch={int(mismatch.sum())})"
        )
    merged = paired.drop(columns="_merge")
    merged = merged.sort_values(["ts", "side", "tp", "sl", "fold"], kind="stable")
    merged["y"] = (merged.outcome == int(lab.Outcome.TP_FIRST)).astype(np.int8)
    merged["cell"] = [cell_key(s, tp, sl) for s, tp, sl in zip(merged.side, merged.tp, merged.sl)]
    merged["jev"] = _conditional(merged.p_tp, merged.p_sl)
    merged["lgbm"] = _conditional(merged.base_tp, merged.base_sl)
    purge_ms = int(horizon_bars) * 15 * 60_000
    train_counts, train_max, test_min = {}, {}, {}
    stack_parts = []
    for fold in sorted(merged.fold.unique()):
        test = merged[merged.fold == fold].copy()
        test_start = int(test.ts.min())
        train = merged[(merged.fold < fold) & (merged.ts < test_start - purge_ms)].copy()
        if len(train) < min_train or train.y.nunique() < 2:
            continue
        model = LogisticRegression(solver="lbfgs", max_iter=1000)
        model.fit(np.column_stack([_logit(train.lgbm), _logit(train.jev)]), train.y)
        test["stack"] = model.predict_proba(
            np.column_stack([_logit(test.lgbm), _logit(test.jev)])
        )[:, 1]
        stack_parts.append(test)
        train_counts[str(int(fold))] = len(train)
        train_max[str(int(fold))] = int(train.ts.max())
        test_min[str(int(fold))] = test_start
    if not stack_parts:
        raise ValueError("no fold has enough strictly prior resolved rows to fit stacker")
    common = pd.concat(stack_parts, ignore_index=True)
    common["day"] = common.ts // 86_400_000
    metrics = {}
    for name in ("jev", "lgbm", "stack"):
        metrics[name] = {
            "within_cell_auc": _within_cell_auc(common, name),
            "within_cell_brier": _within_cell_brier(common, name),
            "auc_ci": None,
        }
    metrics["stack"]["delta_auc_vs_lgbm"] = metrics["stack"]["within_cell_auc"] - metrics["lgbm"]["within_cell_auc"]
    lb, sb = metrics["lgbm"]["within_cell_brier"], metrics["stack"]["within_cell_brier"]
    metrics["stack"]["relative_delta_brier"] = (lb - sb) / lb if lb > 0 else math.nan
    ci = _day_bootstrap(common.reset_index(drop=True), n_boot, seed)
    for name, key in (("jev", "jev_auc"), ("lgbm", "lgbm_auc"), ("stack", "stack_auc")):
        metrics[name]["auc_ci"] = ci[key]
    metrics["stack"]["delta_auc_ci"] = ci["delta_auc"]
    metrics["stack"]["relative_delta_brier_ci"] = ci["relative_delta_brier"]
    class_counts = common.groupby(["cell", "y"], observed=True).size()
    cells = sorted({cell_key(side, tp, sl) for side, tp, sl in
                    j[["side", "tp", "sl"]].drop_duplicates().itertuples(index=False, name=None)})
    minimum_class_count = min(
        (int(class_counts.get((cell, label), 0)) for cell in cells for label in (0, 1)),
        default=0,
    )
    resolved_days = int(common.day.nunique())
    valid_bootstrap = int(ci["valid_replicates"])
    readiness = {
        "expected_cells": len(cells),
        "cells_with_min_class_support": sum(
            all(int(class_counts.get((cell, label), 0)) >= MIN_CLASS_ROWS_PER_CELL for label in (0, 1))
            for cell in cells
        ),
        "minimum_class_rows_per_cell": minimum_class_count,
        "resolved_days": resolved_days,
        "valid_bootstrap_replicates": valid_bootstrap,
        "required_bootstrap_replicates": math.ceil(
            MIN_VALID_BOOTSTRAP_FRACTION * max(MIN_BOOTSTRAP_REPLICATES, n_boot)
        ),
    }
    estimable = (
        readiness["cells_with_min_class_support"] == readiness["expected_cells"]
        and resolved_days >= MIN_BOOTSTRAP_DAYS
        and valid_bootstrap >= readiness["required_bootstrap_replicates"]
    )
    standalone = metrics["jev"]["auc_ci"][0] > 0.50
    incremental = (
        metrics["stack"]["delta_auc_vs_lgbm"] >= 0.005 and ci["delta_auc"][0] > 0
    ) or (
        metrics["stack"]["relative_delta_brier"] >= 0.01 and ci["relative_delta_brier"][0] > 0
    )
    gate = ("INCOMPLETE — insufficient per-cell/day-block support" if not estimable
            else "PASS G3b" if standalone or incremental else "FAIL G3b")
    obs = common[KEYS].sort_values(KEYS).to_csv(index=False, header=True).encode()
    strata = {}
    for col in ("year", "quarter", "vol_q", "dir_t"):
        if col in common:
            strata[col] = {}
            for value, group in common.groupby(col, observed=True, dropna=False):
                strata[col][str(value)] = {
                    "n": len(group),
                    "jev_auc": _within_cell_auc(group, "jev"),
                    "lgbm_auc": _within_cell_auc(group, "lgbm"),
                    "stack_auc": _within_cell_auc(group, "stack"),
                }
    return {
        "gate": gate, "metrics": metrics, "matched_observations": len(common),
        "attempted_bars": attempted_bars,
        "readiness": readiness, "estimable": estimable,
        "raw_jev_cell_rows": len(jev), "invalid_bars": invalid_bars,
        "response_failed_bars": len(response_failed_ts),
        "successful_response_bars": successful_bars,
        "probability_invalid_bars": len(invalid_probability_ts),
        "probability_invalid_cells": invalid_probability_cells,
        "all_cell_valid_bars": all_cell_valid_bars,
        "unmatched_after_join": 0,
        "resolved_matched_rows_before_stacker": len(merged),
        "stack_excluded_rows_insufficient_history": len(merged) - len(common),
        "stack_train_folds": sorted(int(k) for k in train_counts),
        "stacker_train_counts_by_fold": train_counts,
        "stacker_train_max_ts_by_fold": train_max,
        "stacker_test_min_ts_by_fold": test_min,
        "observation_hash": hashlib.sha256(obs).hexdigest(), "strata": strata,
        "bootstrap_days": int(common.day.nunique()), "bootstrap_replicates": int(n_boot),
    }


def run(predictions: str, data_dir: str = "data", out: str = "docs/confirmation_report.md",
        n_boot: int = 400) -> Path:
    """Evaluate only a complete, manifest-verified 10k+ G3b run."""
    cfg = config.load()
    path = Path(predictions)
    frame = pd.read_parquet(path)
    from .g3a import _validate_prediction_artifact, to_long
    _validate_prediction_artifact(path, frame, data_dir, cfg["jev"]["prompt_version"],
                                  cfg["grid"], cfg["jev"]["model_id"], cfg["symbols"][0])
    manifest = json.loads(path.with_name(f"{path.stem}_manifest.json").read_text())
    sampling = manifest.get("sampling_audit", {})
    if int(manifest["identity"]["requested"]) < 10_000:
        raise ValueError("G3b confirmation requires at least 10,000 requested bars")
    if sampling.get("coverage") != 1.0 or not sampling.get("composition_ok"):
        raise ValueError("G3b sample lacks verified OOF coverage or composition")
    if "response_valid" not in frame or frame.response_valid.isna().any():
        raise ValueError("G3b artifact must explicitly identify valid and failed responses")
    jev = to_long(frame, cfg["grid"])
    if "response_valid" in frame:
        valid = frame[["ts", "response_valid"]]
        jev = jev.merge(valid, on="ts", how="left", validate="many_to_one")
    wanted = set(frame.ts.astype(np.int64))
    pieces = []
    cols = ["ts", "side", "tp", "sl", "fold", "model", "outcome",
            "p_sl_first", "p_tp_first", "p_timeout"]
    for oof_path in sorted((Path(data_dir) / "oof").glob("fold_*.parquet")):
        part = pd.read_parquet(oof_path, columns=cols)
        part = part[(part.model.astype(str) == "lgbm") & part.ts.isin(wanted)]
        if len(part):
            pieces.append(part)
    if not pieces:
        raise ValueError("no LightGBM OOF rows match this Jev run")
    result = evaluate_frames(jev, pd.concat(pieces, ignore_index=True),
                             horizon_bars=cfg["grid"]["horizon_bars"], n_boot=n_boot)
    result["requested_bars"] = int(manifest["identity"]["requested"])
    result["attempted_bars"] = int(frame.ts.nunique())
    result["successful_responses"] = result["successful_response_bars"]
    result["failed_responses"] = result["response_failed_bars"]
    result["valid_bars"] = result["all_cell_valid_bars"]
    result["credit_exhaustion_attempts"] = _credit_exhaustion_count(manifest, frame)
    result["credit_exhaustion"] = result["credit_exhaustion_attempts"] > 0
    if result["credit_exhaustion"]:
        result["gate"] = "INCOMPLETE — HTTP 402/credit exhaustion"
    result["sampling_audit"] = sampling
    result["horizon_bars"] = int(cfg["grid"]["horizon_bars"])
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render(result, path), encoding="utf-8")
    return out_path


def render(result: dict, predictions: str | Path) -> str:
    m = result["metrics"]
    def fmt(x):
        return "n/a" if x is None or not np.isfinite(x) else f"{x:.4f}"
    table = ["| model | within-cell AUC | day-block 95% CI | within-cell Brier |",
             "|---|---:|---:|---:|"]
    for name in ("jev", "lgbm", "stack"):
        ci = m[name]["auc_ci"]
        table.append(f"| {name} | {fmt(m[name]['within_cell_auc'])} | [{fmt(ci[0])}, {fmt(ci[1])}] | {fmt(m[name]['within_cell_brier'])} |")
    stack = m["stack"]
    details = json.dumps({"delta_auc": stack["delta_auc_vs_lgbm"],
                          "delta_auc_ci": stack["delta_auc_ci"],
                          "relative_delta_brier": stack["relative_delta_brier"],
                          "relative_delta_brier_ci": stack["relative_delta_brier_ci"]}, indent=2)
    return f"""# G3b ranking confirmation

Gate: **{result['gate']}**

Input: `{predictions}`. This report uses only matched resolved cell observations;
invalid Jev responses are removed as whole bars. No EV, PnL, Sharpe, threshold,
or TP/SL optimization is performed.

| audit | value |
|---|---:|
| requested / attempted bars | {result.get('requested_bars', 'offline evaluator')} / {result.get('attempted_bars', 'n/a')} |
| successful / failed responses | {result.get('successful_responses', 'n/a')} / {result.get('failed_responses', 'n/a')} |
| HTTP 402 / credit-exhaustion attempts | {result.get('credit_exhaustion_attempts', 0)} |
| successful bars with invalid probabilities | {result.get('probability_invalid_bars', 'n/a')} |
| all-cell-valid bars | {result.get('valid_bars', 'offline evaluator')} |
| resolved days / minimum cell class count | {result['readiness']['resolved_days']} / {result['readiness']['minimum_class_rows_per_cell']} |
| cells meeting class support | {result['readiness']['cells_with_min_class_support']} / {result['readiness']['expected_cells']} |
| valid bootstrap replicates | {result['readiness']['valid_bootstrap_replicates']} / {result['readiness']['required_bootstrap_replicates']} |
| excluded failed / probability-invalid bars | {result.get('response_failed_bars', 'n/a')} / {result.get('probability_invalid_bars', 'n/a')} |
| resolved Jev targets / exact LightGBM matches | {result['resolved_matched_rows_before_stacker']} / {result['resolved_matched_rows_before_stacker']} |
| rows omitted for insufficient stacker history | {result['stack_excluded_rows_insufficient_history']} |
| matched triple-model observations | {result['matched_observations']} |
| observation SHA-256 | `{result['observation_hash']}` |
| day-block bootstrap | {result['bootstrap_replicates']} replicates, {result['bootstrap_days']} days |

{chr(10).join(table)}

Stacker Δ metrics vs LightGBM:

```json
{details}
```

Fold-aware stacker fits use only earlier folds and purge {result.get('horizon_bars', 'configured')} bars before each test interval.
Train counts: `{json.dumps(result['stacker_train_counts_by_fold'], sort_keys=True)}`.

Regime slices (year / quarter / volatility quartile / direction):

```json
{json.dumps(result['strata'], indent=2, sort_keys=True, allow_nan=True)}
```

G3b failure blocks full-history inference, G4 calibration and strategy optimization.
"""
