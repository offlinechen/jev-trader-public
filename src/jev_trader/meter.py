"""Run a reproducible real-request Jev cost meter over labelled 15m bars."""

from __future__ import annotations

import math
import json
import os
import statistics
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, data, labels as lab, sample
from .grid import tp_key
from .state import market_state_from_bars
from .jev import (
    Budget, JevApiError, JevBudgetError, JevClient, JevHealthError, JevJournalError,
    JevSchemaError, JevSettings,
    _result, build_questions,
    canonical_hash, canonical_request_identity,
)

# Measured on 5,916 successful v3 requests (G3a + partial G3b): $0.000284/bar.
# Only the default for pre-flight projection; the ledger charges real cost.
DEFAULT_EST_COST_PER_REQUEST = 0.000284
METER_IDENTITY_VERSION = 3


def _health_stop_reason(outcomes: list[bool], *, minimum: int = 10,
                        window: int = 20, max_invalid_rate: float = 0.25) -> str | None:
    """Stop a metering tranche when recent billed responses fail schema validation."""
    recent = outcomes[-window:]
    if len(recent) < minimum:
        return None
    invalid_rate = sum(recent) / len(recent)
    if invalid_rate > max_invalid_rate:
        return (
            f"schema health stop: {sum(recent)}/{len(recent)} recent bars invalid "
            f"({invalid_rate:.1%} > {max_invalid_rate:.1%})"
        )
    return None


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _append_checkpoint(path: Path, entry: dict) -> None:
    if "record" in entry and "sample_index" in entry:
        entry.setdefault("event_time_ns", time.time_ns())
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, allow_nan=True, separators=(",", ":")) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _append_response_event(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _read_response_events(path: Path, run_id: str) -> list[dict]:
    if not path.is_file():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("context", {}).get("run_id") != run_id:
            raise RuntimeError(f"response journal identity mismatch: {path}")
        if event.get("status", 0) >= 200 and event.get("status", 0) < 300:
            events.append(event)
    return events


def _resume_health_outcomes(manifest: dict, checkpoint_entries: list[dict],
                            response_events: list[dict], questions: dict, *,
                            allow_override: bool = False,
                            minimum: int = 10, window: int = 20,
                            max_invalid_rate: float = 0.25) -> list[bool]:
    if allow_override:
        return []
    previous_health = manifest.get("health") or {}
    previous_stop = previous_health.get("stopped_reason")
    if manifest.get("status") == "health_stopped" or previous_stop:
        raise JevHealthError(
            f"resume blocked: run was previously health-stopped ({previous_stop or 'schema health'})"
        )

    timeline = []
    all_events_timed = all(
        isinstance(event.get("event_time_ns"), int)
        for event in [*checkpoint_entries, *response_events]
    )
    sequence = 0
    for entry in checkpoint_entries:
        index = int(entry["sample_index"])
        record = entry.get("record") or {}
        failure = entry.get("failure") or {}
        if entry.get("status") == "success" or record.get("response_valid") is True:
            outcome = False
        else:
            message = str(failure.get("error") or record.get("request_error") or "")
            if failure.get("kind") == "schema" or message.startswith((
                "Jev response", "Jev returned non-JSON success response",
                "unsupported Jev question type", "Jev usage",
            )):
                outcome = True
            else:
                continue
        event_time = entry.get("event_time_ns") if all_events_timed else sequence
        timeline.append((event_time, sequence, index, outcome))
        sequence += 1

    for event in response_events:
        index = (event.get("context") or {}).get("sample_index")
        if index is None:
            continue
        result, error = _replay_response(event, questions, "resume-health-check")
        event_time = event.get("event_time_ns") if all_events_timed else sequence
        timeline.append((event_time, sequence, int(index), result is None and error is not None))
        sequence += 1

    latest_by_bar = {}
    for event_time, sequence, index, outcome in sorted(timeline):
        latest_by_bar[index] = (event_time, sequence, outcome)
    history = [entry[2] for entry in sorted(latest_by_bar.values())]
    reason = _health_stop_reason(
        history, minimum=minimum, window=window, max_invalid_rate=max_invalid_rate
    )
    if reason:
        raise JevHealthError(f"resume blocked by historical schema health: {reason}")
    return history


def _result_record(index: int, row, result, symbol: str, prompt_version: str,
                   request_state_hash: str) -> dict:
    record = {
        "sample_index": index,
        "ts": int(row.ts),
        "symbol": symbol,
        "model_id": result.model,
        "prompt_version": prompt_version,
        "state_hash": result.state_hash,
        "request_state_hash": request_state_hash,
        "latency_ms": result.latency_ms,
        "cached": result.cached,
        "input_tokens": result.usage["input_tokens"],
        "output_tokens": result.usage["output_tokens"],
        "cache_read_input_tokens": result.usage.get("cache_read_input_tokens", 0),
        "cache_creation_input_tokens": result.usage.get("cache_creation_input_tokens", 0),
        "cost": result.usage.get("cost", math.nan),
        "year": int(row.year),
        "quarter": str(row.quarter),
        "vol_q": str(row.vol_q),
        "dir_t": str(row.dir_t),
        "response_valid": True,
        "request_error": "",
    }
    record.update(result.answers)
    return record


def _replay_response(event: dict, questions: dict, request_state_hash: str):
    try:
        raw = json.loads(event["body"])
        return _result(raw, questions, request_state_hash, cached=False), None
    except (KeyError, TypeError, ValueError, JevApiError) as exc:
        return None, str(exc)[:500]


def _cache_response_for_attempt(path: Path, sample_index: int,
                                attempt_events: list[dict]) -> dict | None:
    """Use a cache entry only when its mtime proves it came from this attempt."""
    if not path.is_file():
        return None
    reservations = {
        int(event["attempt"]): event for event in attempt_events
        if event.get("event") == "reserve"
        and (event.get("context") or {}).get("sample_index") == sample_index
    }
    settled_200 = [
        event for event in attempt_events
        if event.get("event") == "settle" and event.get("status") == 200
        and int(event.get("attempt", -1)) in reservations
    ]
    if not settled_200:
        return None
    try:
        cache_time = path.stat().st_mtime
        reserve_times = [
            datetime.fromisoformat(event["at"].replace("Z", "+00:00")).timestamp()
            for event in reservations.values()
        ]
        settle_times = [
            datetime.fromisoformat(event["at"].replace("Z", "+00:00")).timestamp()
            for event in settled_200
        ]
        if not (min(reserve_times) <= cache_time <= max(settle_times) + 5):
            return None
        response = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return response if isinstance(response, dict) else None


def _append_attempt_event(path: Path, out_dir: Path, n: int, seed: int,
                          client: JevClient, run_id: str, event: dict) -> None:
    entry = {
        **event, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_id": run_id,
    }
    _append_checkpoint(path, entry)
    ledger_entry = {
        "at": entry["at"], "n": n, "seed": seed, "status": "running",
        "run_id": run_id, "model_id": client.settings.model_id,
        "prompt_version": client.settings.prompt_version,
        "event": event["event"], "event_seq": event.get("event_seq"),
        "budget": event["budget"],
    }
    with (out_dir / "jev_spend_ledger.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(ledger_entry, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _read_checkpoint(path: Path, run_id: str | None = None) -> list[dict]:
    if not path.is_file():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict) and "sample_index" in entry and "record" in entry:
            if run_id is not None and entry.get("run_id") != run_id:
                raise RuntimeError(f"checkpoint journal identity mismatch: {path}")
            entries.append(entry)
    return entries


def _read_attempt_wal(path: Path, run_id: str) -> list[dict]:
    if not path.is_file():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue  # a torn final append was never fsynced and never sent
        if not isinstance(event, dict):
            continue
        if event.get("run_id") != run_id:
            raise RuntimeError(f"attempt WAL identity mismatch: {path}")
        if event.get("event") in {"reserve", "settle"}:
            events.append(event)
    return events


def _attempt_wal_state(events: list[dict], max_requests: int, max_usd: float,
                       base: dict | None = None) -> tuple[dict, set[int]]:
    reservations = {}
    settlements = {}
    attempted_indices = set()
    caps_requests, caps_usd = max_requests, max_usd
    base = base or {}
    base_attempts = int(base.get("attempts", 0))
    for event in events:
        attempt = int(event["attempt"])
        if event["event"] == "reserve":
            reservations[attempt] = event
            sample_index = (event.get("context") or {}).get("sample_index")
            if sample_index is not None:
                attempted_indices.add(int(sample_index))
            budget = event.get("budget", {})
            caps_requests = max(caps_requests, int(budget.get("max_requests", caps_requests)))
            caps_usd = max(caps_usd, float(budget.get("max_usd", caps_usd)))
        elif event["event"] == "settle":
            settlements[attempt] = event
    actual = float(base.get("actual_usd", 0.0) or 0.0)
    estimated = float(base.get("estimated_usd", 0.0) or 0.0)
    committed = max(
        actual + estimated,
        float(base.get("committed_usd", actual + estimated) or 0.0),
    )
    billed = int(base.get("billed", 0) or 0)
    by_status = dict(base.get("by_status", {}))
    base_event_seq = base.get("event_seq")
    has_sequence = all("event_seq" in event for event in events) and (
        base_event_seq is not None or not base
    )
    if has_sequence:
        cutoff = int(base_event_seq or 0)
        ordered = sorted(events, key=lambda event: int(event["event_seq"]))
        open_attempts = {
            attempt for attempt, reserve in reservations.items()
            if int(reserve["event_seq"]) <= cutoff
            and (attempt not in settlements or int(settlements[attempt]["event_seq"]) > cutoff)
        }
        for event in ordered:
            if int(event["event_seq"]) <= cutoff:
                continue
            attempt = int(event["attempt"])
            if event["event"] == "reserve":
                amount = float(event.get("reserve_usd", 0.0))
                committed += amount
                open_attempts.add(attempt)
                base_attempts = max(base_attempts, attempt)
            else:
                reserve = reservations.get(attempt)
                if reserve is not None and attempt in open_attempts:
                    committed = max(0.0, committed - float(reserve.get("reserve_usd", 0.0)))
                    open_attempts.remove(attempt)
                cost = float(event.get("actual_usd", 0.0) or 0.0)
                estimate = float(event.get("estimated_usd", 0.0) or 0.0)
                actual += cost
                estimated += estimate
                committed += cost + estimate
                billed += int(bool(event.get("billed")))
                status = "transport" if event.get("status") is None else str(event["status"])
                by_status[status] = by_status.get(status, 0) + 1
        event_seq = max((int(event["event_seq"]) for event in ordered), default=cutoff)
    else:
        # Legacy WALs have no common checkpoint/event sequence. Keep the checkpoint
        # as the baseline, repairing its identifiable in-flight attempts only.
        base_inflight = max(0, base_attempts - sum(int(v) for v in by_status.values()))
        residual = max(0.0, committed - actual - estimated)
        settled_before_base = sorted(
            (attempt for attempt in settlements if attempt <= base_attempts), reverse=True
        )[:base_inflight]
        for attempt, reserve in reservations.items():
            if attempt > base_attempts:
                settled = settlements.get(attempt)
                if settled is None:
                    estimated += float(reserve.get("reserve_usd", 0.0))
                    committed += float(reserve.get("reserve_usd", 0.0))
                    continue
                released = 0.0
            elif attempt not in settled_before_base:
                continue
            else:
                released = min(residual, float(reserve.get("reserve_usd", 0.0)))
                residual -= released
            settled = settlements.get(attempt)
            if settled is None:
                continue
            cost = float(settled.get("actual_usd", 0.0) or 0.0)
            estimate = float(settled.get("estimated_usd", 0.0) or 0.0)
            actual += cost
            estimated += estimate
            committed += cost + estimate - released
            billed += int(bool(settled.get("billed")))
            status = "transport" if settled.get("status") is None else str(settled["status"])
            by_status[status] = by_status.get(status, 0) + 1
        event_seq = max(
            int(base_event_seq or 0),
            max((int(event.get("event_seq", 0)) for event in events), default=0),
        )
    snapshot = {
        "attempts": max(base_attempts, max(reservations, default=0)),
        "event_seq": event_seq, "actual_usd": actual,
        "estimated_usd": estimated, "committed_usd": committed,
        "max_requests": caps_requests, "max_usd": caps_usd,
        "billed": billed, "billed_rejected": int(base.get("billed_rejected", 0) or 0),
        "by_status": by_status,
    }
    return snapshot, attempted_indices


def _recover_missing_settlements(events: list[dict], response_events: list[dict],
                                max_requests: int, max_usd: float,
                                est_cost_per_request: float,
                                base: dict | None = None) -> list[dict]:
    """Build durable settle events for journaled 2xx responses absent from the WAL."""
    working = list(events)
    reserves = {int(event["attempt"]): event for event in working if event["event"] == "reserve"}
    settled = {int(event["attempt"]) for event in working if event["event"] == "settle"}
    next_seq = max(
        [int(event.get("event_seq", 0)) for event in working]
        + [int((base or {}).get("event_seq", 0))]
    )
    recovered = []
    for response in response_events:
        attempt = response.get("attempt")
        if attempt is None:
            continue
        attempt = int(attempt)
        status = int(response.get("status", 0))
        reserve = reserves.get(attempt)
        if reserve is None or attempt in settled or not 200 <= status < 300:
            continue

        cost = None
        try:
            raw = json.loads(response["body"])
            value = raw.get("usage", {}).get("cost") if isinstance(raw, dict) else None
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                value = float(value)
                if math.isfinite(value) and value >= 0:
                    cost = value
        except (KeyError, TypeError, ValueError):
            pass
        if cost is None:
            state, _ = _attempt_wal_state(working, max_requests, max_usd, base)
            observed = state["actual_usd"] / state["billed"] if state["billed"] else 0.0
            estimate = max(
                est_cost_per_request, float(reserve.get("reserve_usd", 0.0)), observed
            )
        else:
            estimate = 0.0

        next_seq += 1
        settlement = {
            "event": "settle", "attempt": attempt, "event_seq": next_seq,
            "status": status, "actual_usd": cost or 0.0,
            "estimated_usd": estimate, "billed": cost is not None,
        }
        state, _ = _attempt_wal_state(
            [*working, settlement], max_requests, max_usd, base
        )
        settlement["budget"] = state
        working.append(settlement)
        settled.add(attempt)
        recovered.append(settlement)
    return recovered


def _attach_current_outcomes(frame: pd.DataFrame, labels_by_ts: pd.DataFrame,
                             grid: dict) -> pd.DataFrame:
    """Derive labels from the current artifact; inference checkpoints stay label-free."""
    out = frame.copy()
    for side in ("long", "short"):
        for tp in grid["tp"]:
            for sl in grid["sl"]:
                key = tp_key(side, tp, sl)
                values = []
                for ts in out["ts"].astype(np.int64):
                    label_row = labels_by_ts.loc[int(ts)]
                    values.append(int(lab.outcomes(
                        pd.DataFrame([label_row]), side, tp, sl, grid["horizon_bars"]
                    )[0]))
                out[f"outcome_{key[2:]}"] = values
    return out


def _run_identity(n: int, seed: int, symbol: str, client: JevClient, workers: int,
                  use_cache: bool, grid: dict, candidate_ts: np.ndarray,
                  picked_ts: np.ndarray, request_identities: list[dict]) -> tuple[dict, str]:
    del workers, use_cache  # execution choices do not change request semantics
    picked = [
        {"sample_index": index, "ts": int(ts)}
        for index, ts in enumerate(np.asarray(picked_ts, dtype=np.int64), start=1)
    ]
    request_hashes = [canonical_hash(identity) for identity in request_identities]
    questions = request_identities[0]["payload"]["questions"] if request_identities else {}
    contract = (
        request_identities[0]["payload"]["state"]["_jev_protocol"]["contract"]
        if request_identities else ""
    )
    identity = {
        "identity_version": METER_IDENTITY_VERSION,
        "requested": n,
        "seed": seed,
        "candidate_ts_sha256": canonical_hash(
            [int(ts) for ts in np.sort(np.unique(np.asarray(candidate_ts, dtype=np.int64)))]
        ),
        "picked_ts_sha256": canonical_hash(picked),
        "request_state_hashes": request_hashes,
        "request_states_sha256": canonical_hash(request_hashes),
        "questions_sha256": canonical_hash(questions),
        "contract_sha256": canonical_hash(contract),
        "symbol": symbol,
        "base_url": client.settings.base_url,
        "model_id": client.settings.model_id,
        "prompt_version": client.settings.prompt_version,
        "horizon_bars": grid["horizon_bars"],
        "tp": list(grid["tp"]),
        "sl": list(grid["sl"]),
    }
    run_id = canonical_hash(identity)[:16]
    return identity, run_id


def _g3b_candidate_timestamps(data_dir: Path, features: pd.DataFrame,
                             labels: pd.DataFrame, grid: dict) -> tuple[np.ndarray, dict]:
    """Bars with features, labels, and complete matched OOF rows for all baselines."""
    models = {"climatology", "logit", "lgbm"}
    cell_keys = {
        (side, round(float(tp), 6), round(float(sl), 6))
        for side in ("long", "short") for tp in grid["tp"] for sl in grid["sl"]
    }
    labels = labels.copy()
    labels.attrs["horizon_min"] = int(grid["horizon_bars"]) * 15
    expected = pd.Series(0, index=labels.ts.astype(np.int64).to_numpy())
    for side, tp, sl in cell_keys:
        outcome = lab.outcomes(labels, side, tp, sl, int(grid["horizon_bars"]))
        expected += (outcome != int(lab.Outcome.AMBIGUOUS)).astype(np.int8)

    feature_cols = [
        col for col in features.columns
        if col != "ts" and pd.api.types.is_numeric_dtype(features[col])
    ]
    feature_ts = set(features.dropna(subset=feature_cols).ts.astype(np.int64))
    label_ts = set(labels.ts.astype(np.int64))
    candidate_universe = feature_ts & label_ts
    seen_ts = set()
    matched_counts = {}
    columns = ["ts", "side", "tp", "sl", "fold", "model", "outcome",
               "p_sl_first", "p_tp_first", "p_timeout", "horizon"]
    for path in sorted((Path(data_dir) / "oof").glob("fold_*.parquet")):
        frame = pd.read_parquet(path, columns=columns)
        frame = frame[frame.model.astype(str).isin(models)].copy()
        if frame.empty:
            continue
        file_timestamps = set(frame.ts.astype(np.int64))
        overlap = seen_ts & file_timestamps
        if overlap:
            raise ValueError(f"baseline OOF timestamps repeat across files: {path}")
        seen_ts.update(file_timestamps)
        if "horizon" in frame and not frame.horizon.eq(int(grid["horizon_bars"])).all():
            raise ValueError(f"baseline OOF horizon mismatch: {path}")
        frame["model"] = frame.model.astype(str)
        frame["side"] = frame.side.astype(str)
        frame[["tp", "sl"]] = frame[["tp", "sl"]].astype(float).round(6)
        observed_cells = set(map(tuple, frame[["side", "tp", "sl"]].drop_duplicates().itertuples(index=False, name=None)))
        if not observed_cells <= cell_keys:
            raise ValueError(f"baseline OOF contains cells outside the configured grid: {path}")
        row_keys = ["ts", "side", "tp", "sl", "model"]
        if frame.duplicated(row_keys).any():
            raise ValueError(f"duplicate baseline OOF observations: {path}")
        probs = frame[["p_sl_first", "p_tp_first", "p_timeout"]].to_numpy(float)
        valid_probs = (
            np.isfinite(probs).all(axis=1) & (probs >= 0).all(axis=1)
            & (probs <= 1).all(axis=1)
            & (np.abs(probs.sum(axis=1) - 1) <= 1e-5)
        )
        frame = frame.loc[valid_probs]
        grouped = frame.groupby(["ts", "side", "tp", "sl"], sort=False, observed=True)
        cells = grouped.agg(
            n=("model", "size"), models=("model", "nunique"),
            outcomes=("outcome", "nunique"), folds=("fold", "nunique"),
            outcome=("outcome", "first"), fold=("fold", "first"),
        ).reset_index()
        cells = cells[(cells.n == len(models)) & (cells.models == len(models))
                      & (cells.outcomes == 1) & (cells.folds == 1)]
        if cells.empty:
            continue
        if cells.groupby("ts").fold.nunique().gt(1).any():
            raise ValueError(f"baseline OOF timestamp has multiple folds: {path}")
        labels_by_ts = labels.set_index("ts")
        for (side, tp, sl), group in cells.groupby(["side", "tp", "sl"], observed=True):
            rows = labels_by_ts.loc[group.ts.astype(np.int64).to_numpy()]
            expected_outcome = lab.outcomes(rows, side, tp, sl, int(grid["horizon_bars"]))
            if not np.array_equal(expected_outcome, group.outcome.to_numpy(np.int8)):
                raise ValueError(f"baseline OOF labels mismatch: {path}")
        counts = cells.groupby("ts").size()
        matched_counts.update({int(ts): int(count) for ts, count in counts.items()})

    observed = pd.Series(matched_counts, dtype=np.int64)
    target = expected.reindex(sorted(candidate_universe)).astype(np.int64)
    covered = observed.reindex(target.index, fill_value=0)
    eligible = target.gt(0) & covered.eq(target)
    timestamps = target.index[eligible.to_numpy()].to_numpy(np.int64)
    expected_cells = int(target.loc[eligible].sum())
    audit = {
        "features_and_labels": len(candidate_universe),
        "eligible_bars": len(timestamps),
        "expected_oof_cell_rows": expected_cells,
        "matched_oof_cell_rows": int(covered.loc[eligible].sum()),
        "coverage": 1.0 if expected_cells else 0.0,
        "models": sorted(models),
    }
    return timestamps, audit


def _historic_attempts(data_dir: Path) -> tuple[set[int], int]:
    """Known prior timestamps; legacy failures without timestamps are counted separately."""
    timestamps: set[int] = set()
    unattributed = 0
    for n in (100, 2000, 10000):
        artifact = Path(data_dir) / f"metering_{n}.parquet"
        if artifact.is_file():
            frame = pd.read_parquet(artifact, columns=["ts"])
            timestamps.update(frame.ts.dropna().astype(np.int64))
        failures_path = Path(data_dir) / f"metering_{n}_failures.json"
        if failures_path.is_file():
            try:
                failures = json.loads(failures_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            for failure in failures if isinstance(failures, list) else []:
                if isinstance(failure, dict) and failure.get("ts") is not None:
                    timestamps.add(int(failure["ts"]))
                else:
                    unattributed += 1
    return timestamps, unattributed


def _load_or_create_manifest(path: Path, identity: dict, run_id: str) -> dict:
    if path.is_file():
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest.get("run_id") == run_id and manifest.get("identity") == identity:
                return manifest
            raise RuntimeError(f"checkpoint identity mismatch: {path}")
        except (OSError, ValueError, TypeError):
            pass
    manifest = {
        "version": 1,
        "run_id": run_id,
        "identity": identity,
        "completed": {},
        "parts": [],
        "complete": False,
    }
    _write_json_atomic(path, manifest)
    return manifest


def _append_ledger(out_dir: Path, n: int, seed: int, client: JevClient, status: str,
                   run_id: str) -> None:
    """Append-only spend record. Reports and artifacts are overwritten when a
    run is repeated at the same n, which erases the earlier run's cost; this
    file never is, so every billed run stays on the books."""
    entry = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n": n, "seed": seed, "status": status,
        "run_id": run_id,
        "model_id": client.settings.model_id,
        "prompt_version": client.settings.prompt_version,
        "budget": client.budget.snapshot() if client.budget else None,
    }
    with open(out_dir / "jev_spend_ledger.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")


def preflight(pending: int, max_requests: int, max_usd: float, est: float) -> dict:
    """Plan this budget tranche; the hard Budget guard stops and resumes at the cap."""
    if pending < 0 or max_requests < 0 or max_usd < 0 or est <= 0:
        raise JevBudgetError("pending work and remaining caps must be non-negative")
    allowance = min(pending, max_requests, int((max_usd + 1e-12) / est))
    if pending and allowance == 0:
        raise JevBudgetError("no remaining budget for another estimated request")
    plan = {
        "pending_bars": pending,
        "request_allowance": allowance,
        "expected_usd": allowance * est,
        "worst_usd": min(max_usd, allowance * est),
        "max_usd": max_usd,
        "max_requests": max_requests,
        "partial": allowance < pending,
    }
    return plan


def run(n: int = 100, seed: int = 20260921, eligible_ts: np.ndarray | None = None,
        workers: int = 1, use_cache: bool = True, *,
        max_requests: int | None = None, max_usd: float | None = None,
        est_cost_per_request: float = DEFAULT_EST_COST_PER_REQUEST,
        log_every: int = 100, health_min_samples: int = 10,
        health_window: int = 20, max_invalid_rate: float = 0.25,
        allow_health_override: bool = False) -> Path:
    if max_requests is None or max_usd is None:
        raise JevBudgetError(
            "a live metering run needs explicit max_requests and max_usd -- "
            "provider credit exhaustion is not a stop mechanism"
        )
    if health_min_samples <= 0 or health_window < health_min_samples:
        raise ValueError("health window must be >= a positive minimum sample count")
    if not 0 <= max_invalid_rate <= 1:
        raise ValueError("max_invalid_rate must be between 0 and 1")
    cfg = config.load()
    symbol = config.to_binance(cfg["symbols"][0])
    features = pd.read_parquet(Path(cfg["data_dir"]) / "features.parquet")
    labels = pd.read_parquet(Path(cfg["data_dir"]) / "labels.parquet")
    ohlcv = data.load(symbol, cfg["timeframe"], cfg["data_dir"])

    feature_cols = [
        col for col in features.columns
        if col != "ts" and pd.api.types.is_numeric_dtype(features[col])
    ]
    eligible = features.dropna(subset=feature_cols)
    all_eligible_ts, oof_audit = _g3b_candidate_timestamps(
        Path(cfg["data_dir"]), eligible, labels, cfg["grid"]
    )
    if eligible_ts is not None:
        all_eligible_ts = np.intersect1d(all_eligible_ts, np.asarray(eligible_ts, dtype=np.int64))
    target_population = eligible[eligible["ts"].isin(all_eligible_ts)]
    historical_attempts, unattributed_failures = _historic_attempts(Path(cfg["data_dir"]))
    excluded_historic_ts = (
        set(all_eligible_ts) & historical_attempts if n >= 10_000 else set()
    )
    draw_ts = np.asarray(sorted(set(all_eligible_ts) - excluded_historic_ts), dtype=np.int64)
    draw_pool = eligible[eligible["ts"].isin(draw_ts)]
    picked = sample.stratified_sample(draw_pool, draw_ts, n=n, seed=seed, floor=1)
    if len(picked) < n:
        remaining = draw_pool[~draw_pool["ts"].isin(picked["ts"])]
        extra = remaining.sample(n=n - len(picked), random_state=seed)
        picked = pd.concat([picked, extra], ignore_index=True)
    if len(picked) > n:
        picked = picked.sample(n=n, random_state=seed).sort_values("ts")
    assert len(picked) == n and picked["ts"].is_unique
    composition = sample.composition(picked, target_population)
    if n >= 10_000 and not composition["ok"].all():
        bad = composition.loc[~composition["ok"], ["dim", "level", "ratio"]]
        raise ValueError(f"G3b sample composition failed before inference:\n{bad.to_string(index=False)}")

    out_dir = Path(cfg["data_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    questions = build_questions(cfg["grid"])
    client = JevClient(JevSettings.from_config(cfg))
    labels_by_ts = labels.set_index("ts")
    started = time.perf_counter()
    def build_state(row) -> dict:
        ts = int(row.ts)
        window = ohlcv[ohlcv["ts"] <= ts].tail(1500)
        if window.empty or int(window.iloc[-1]["ts"]) != ts:
            raise ValueError(f"bar {ts} is absent from raw OHLCV")
        return market_state_from_bars(cfg["symbols"][0], window, cfg["grid"])

    picked_rows = list(picked.itertuples(index=False))
    request_states = [build_state(row) for row in picked_rows]
    state_by_index = {
        index: state for index, state in enumerate(request_states, start=1)
    }
    request_identities = [
        canonical_request_identity(client.settings, state, questions)
        for state in request_states
    ]
    identity, run_id = _run_identity(
        n, seed, cfg["symbols"][0], client, workers, use_cache, cfg["grid"],
        draw_ts, picked["ts"].to_numpy(np.int64), request_identities,
    )
    output = out_dir / f"metering_{n}_{run_id}.parquet"
    manifest_path = out_dir / f"metering_{n}_{run_id}_manifest.json"
    parts_dir = out_dir / f"metering_{n}_{run_id}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = parts_dir / "checkpoint.jsonl"
    attempt_wal_path = parts_dir / "attempts.wal.jsonl"
    response_journal_path = parts_dir / "responses.wal.jsonl"
    client.on_response = lambda event: _append_response_event(response_journal_path, event)
    manifest = _load_or_create_manifest(manifest_path, identity, run_id)
    legacy_g3a = out_dir / "metering_2000.parquet"
    g3a_rows = 0
    if n >= 10_000 and legacy_g3a.is_file():
        g3a_rows = len(pd.read_parquet(legacy_g3a, columns=["ts"]))
    cache_present = client.settings.cache_dir.is_dir() and any(client.settings.cache_dir.glob("*.json"))
    sampling_audit = {
        **oof_audit,
        "candidate_bars_before_history_exclusion": len(all_eligible_ts),
        "known_prior_timestamps_excluded": len(excluded_historic_ts),
        "unattributed_legacy_failures": unattributed_failures,
        "g3a_legacy_rows": g3a_rows,
        "g3a_safe_reusable_rows": 0,
        "g3a_reuse_reason": (
            "legacy G3a has no request_state_hash or run manifest with question/schema identity"
            if g3a_rows else "no G3a artifact found"
        ),
        "local_cache_present": cache_present,
        "composition_ok": bool(composition["ok"].all()),
    }
    manifest["sampling_audit"] = sampling_audit
    _write_json_atomic(manifest_path, manifest)
    attempted_path = out_dir / f"metering_{n}_{run_id}_attempted.csv"
    composition_path = out_dir / f"metering_{n}_{run_id}_composition.csv"
    picked[["ts", "year", "quarter", "vol_q", "dir_t"]].assign(
        sample_index=np.arange(1, len(picked) + 1)
    ).to_csv(attempted_path, index=False)
    composition.to_csv(composition_path, index=False)
    checkpoint_entries = _read_checkpoint(checkpoint_path, run_id)
    resume_budget = manifest.get("budget")
    for entry in checkpoint_entries:
        candidate = entry.get("budget")
        if candidate is not None and int(candidate.get("event_seq", -1)) >= int(
            (resume_budget or {}).get("event_seq", -1)
        ):
            resume_budget = candidate
    attempt_events = _read_attempt_wal(attempt_wal_path, run_id)
    response_events = _read_response_events(response_journal_path, run_id)
    settlements = _recover_missing_settlements(
        attempt_events, response_events, max_requests, max_usd,
        est_cost_per_request, resume_budget,
    )
    for event in settlements:
        _append_attempt_event(
            attempt_wal_path, out_dir, n, seed, client, run_id, event
        )
    if settlements:
        attempt_events = _read_attempt_wal(attempt_wal_path, run_id)
    wal_budget, attempted_indices = _attempt_wal_state(
        attempt_events, max_requests, max_usd, resume_budget
    )
    budget = Budget(
        max_requests, max_usd, est_cost_per_request,
        on_attempt_event=lambda event: _append_attempt_event(
            attempt_wal_path, out_dir, n, seed, client, run_id, event
        ),
    )
    budget.restore(wal_budget if attempt_events else resume_budget)
    client.budget = budget
    completed_indices = {int(index) for index in manifest["completed"]}
    manifest.setdefault("failure_details", {})
    for entry in checkpoint_entries:
        index = str(entry["sample_index"])
        completed_indices.add(int(index))
        manifest["completed"][index] = entry.get("status", "success")
        if entry.get("failure"):
            manifest["failure_details"][index] = entry["failure"]

    try:
        health_history = _resume_health_outcomes(
            manifest, checkpoint_entries, response_events, questions,
            allow_override=allow_health_override,
            minimum=health_min_samples, window=health_window,
            max_invalid_rate=max_invalid_rate,
        )
    except JevHealthError as exc:
        manifest["status"] = "health_stopped"
        manifest["health"] = {
            "minimum_samples": health_min_samples,
            "window": health_window,
            "max_invalid_rate": max_invalid_rate,
            "historical_evaluated_bars": len(checkpoint_entries),
            "stopped_reason": str(exc),
        }
        _write_json_atomic(manifest_path, manifest)
        raise
    if allow_health_override:
        manifest["health_override"] = {
            "authorized": True,
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    checkpoint_indices = set(completed_indices)
    response_by_index: dict[int, list[dict]] = {}
    for event in response_events:
        sample_index = event.get("context", {}).get("sample_index")
        if sample_index is not None:
            response_by_index.setdefault(int(sample_index), []).append(event)
    recovered_failures = []
    recovered_records = []
    for index in sorted(attempted_indices - checkpoint_indices):
        row = picked_rows[index - 1]
        ts = int(row.ts)
        request_state_hash = canonical_hash(request_identities[index - 1])
        latest_response = response_by_index.get(index, [])
        result = None
        error = "request outcome is uncertain after interruption; not retried"
        if latest_response:
            result, replay_error = _replay_response(
                latest_response[-1], questions, request_state_hash
            )
            if replay_error:
                error = replay_error
        else:
            cached_response = _cache_response_for_attempt(
                client.settings.cache_dir / f"{request_state_hash}.json",
                index, attempt_events,
            )
            if cached_response is not None:
                result, replay_error = _replay_response(
                    {"body": json.dumps(cached_response)}, questions, request_state_hash
                )
                if replay_error:
                    error = replay_error
        if result is not None:
            record = _result_record(
                index, row, result, cfg["symbols"][0], client.settings.prompt_version,
                request_state_hash,
            )
            _append_checkpoint(checkpoint_path, {
                "sample_index": index, "run_id": run_id, "status": "success",
                "record": record, "failure": None, "budget": budget.snapshot(),
            })
            recovered_records.append(record)
            manifest["completed"][str(index)] = "success"
            completed_indices.add(index)
            continue
        null_record = {
            "sample_index": index, "ts": ts, "symbol": cfg["symbols"][0],
            "model_id": "", "prompt_version": client.settings.prompt_version,
            "state_hash": "", "request_state_hash": request_state_hash,
            "latency_ms": math.nan, "cached": False, "input_tokens": 0,
            "output_tokens": 0, "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0, "cost": math.nan,
            "year": int(row.year), "quarter": str(row.quarter),
            "vol_q": str(row.vol_q), "dir_t": str(row.dir_t),
            "response_valid": False, "request_error": error,
        }
        for key, question in questions.items():
            if question["type"] == "noul":
                null_record[key] = math.nan
            else:
                prefix = "vol" if key == "volatility" else key
                for option in question.get("criteria", {}):
                    null_record[f"{prefix}_{option}"] = math.nan
        failure = {"sample_index": index, "ts": ts, "error": error}
        _append_checkpoint(checkpoint_path, {
            "sample_index": index, "run_id": run_id, "status": "failure",
            "record": null_record, "failure": failure,
            "budget": budget.snapshot(),
        })
        manifest["completed"][str(index)] = "failure"
        manifest["failure_details"][str(index)] = failure
        recovered_failures.append(null_record)
        completed_indices.add(index)
    if recovered_failures:
        pd.DataFrame(recovered_failures).to_parquet(
            parts_dir / f"part_{len(manifest['parts']):05d}.parquet", index=False
        )
        manifest["parts"].append(f"part_{len(manifest['parts']):05d}.parquet")
        manifest["budget"] = budget.snapshot()
        _write_json_atomic(manifest_path, manifest)

    def request_one(index, row):
        ts = int(row.ts)
        state = state_by_index[index]
        request_state_hash = canonical_hash(request_identities[index - 1])
        try:
            result = client.decide(
                state, questions, use_cache=use_cache,
                request_context={"run_id": run_id, "sample_index": index},
            )
        except (JevBudgetError, JevJournalError):
            raise            # not a bar failure: the run stops, the bar stays pending
        except Exception as exc:  # preserve the failed bar in the audit trail
            failure_kind = "schema" if isinstance(exc, JevSchemaError) else "request"
            null_record = {
                "sample_index": index,
                "ts": ts,
                "symbol": cfg["symbols"][0],
                "model_id": "",
                "prompt_version": client.settings.prompt_version,
                "state_hash": "",
                "request_state_hash": request_state_hash,
                "latency_ms": math.nan,
                "cached": False,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cost": math.nan,
                "year": int(row.year),
                "quarter": str(row.quarter),
                "vol_q": str(row.vol_q),
                "dir_t": str(row.dir_t),
                "response_valid": False,
                "request_error": str(exc)[:500],
            }
            for key, question in questions.items():
                if question["type"] == "noul":
                    null_record[key] = math.nan
                else:
                    prefix = "vol" if key == "volatility" else key
                    for option in question.get("criteria", {}):
                        null_record[f"{prefix}_{option}"] = math.nan
            return index, null_record, {
                "sample_index": index, "ts": ts, "error": str(exc)[:500],
                "kind": failure_kind,
            }
        return index, _result_record(
            index, row, result, cfg["symbols"][0], client.settings.prompt_version,
            request_state_hash,
        ), None

    rows = [
        (index, row)
        for index, row in enumerate(picked.itertuples(index=False), start=1)
        if index not in completed_indices
    ]
    pending_count = len(rows)
    budget_state = budget.snapshot()
    remaining_requests = max_requests - budget_state["attempts"]
    remaining_usd = max_usd - budget_state["committed_usd"]
    plan = preflight(pending_count, remaining_requests, remaining_usd, est_cost_per_request)
    print(
        f"preflight: {pending_count} pending bars, up to {plan['request_allowance']} "
        f"attempts this tranche, expected ${plan['expected_usd']:.4f} "
        f"vs remaining cap "
        f"${remaining_usd:.4f} / {remaining_requests} requests",
        flush=True,
    )
    buffer = []
    failure_buffer = []
    invalid_schema_outcomes = list(health_history)

    def flush_batch():
        if not buffer and not failure_buffer:
            return
        if buffer:
            part_name = f"part_{len(manifest['parts']):05d}.parquet"
            pd.DataFrame(buffer).to_parquet(parts_dir / part_name, index=False)
            manifest["parts"].append(part_name)
            for record in buffer:
                manifest["completed"][str(record["sample_index"])] = "success"
            buffer.clear()
        for failure in failure_buffer:
            index = str(failure["sample_index"])
            manifest["completed"][index] = "failure"
            manifest["failure_details"][index] = failure
        failure_buffer.clear()
        manifest["budget"] = budget.snapshot()
        _write_json_atomic(manifest_path, manifest)

    def log_spend(done: int) -> None:
        b = budget.snapshot()
        print(
            f"spend after {done}/{pending_count}: actual ${b['actual_usd']:.6f} "
            f"+ estimated ${b['estimated_usd']:.6f} = committed ${b['committed_usd']:.6f} "
            f"of ${b['max_usd']:.4f} | attempts {b['attempts']}/{b['max_requests']} "
            f"| rejected-but-billed {b['billed_rejected']} | status {b['by_status']}",
            flush=True,
        )

    stopped: JevBudgetError | None = None
    journal_stop: JevJournalError | None = None
    health_stop: str | None = None
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = [pool.submit(request_one, index, row) for index, row in rows]
        for completed, future in enumerate(as_completed(futures), start=1):
            if future.cancelled():
                continue
            try:
                _, record, failure = future.result()
            except (JevBudgetError, JevJournalError) as exc:
                if isinstance(exc, JevBudgetError) and stopped is None:
                    stopped = exc
                elif isinstance(exc, JevJournalError) and journal_stop is None:
                    journal_stop = exc
                if stopped is not None or journal_stop is not None:
                    for other in futures:
                        other.cancel()           # queued bars never send
                continue                         # in-flight ones still land below
            if record is not None:
                _append_checkpoint(checkpoint_path, {
                    "sample_index": record["sample_index"],
                    "run_id": run_id,
                    "status": "success" if failure is None else "failure",
                    "record": record,
                    "failure": failure,
                    "budget": budget.snapshot(),
                })
                buffer.append(record)
            if failure is not None:
                failure_buffer.append(failure)
                invalid_schema_outcomes.append(failure.get("kind") == "schema")
            else:
                invalid_schema_outcomes.append(False)
            if health_stop is None:
                health_stop = _health_stop_reason(
                    invalid_schema_outcomes,
                    minimum=health_min_samples,
                    window=health_window,
                    max_invalid_rate=max_invalid_rate,
                )
                if health_stop:
                    for other in futures:
                        other.cancel()
            if len(buffer) >= 250 or completed == pending_count:
                flush_batch()
            if completed % log_every == 0 or completed == pending_count:
                log_spend(completed)

    flush_batch()
    log_spend(len(manifest["completed"]))
    status = (
        "journal_stopped" if journal_stop else "health_stopped" if health_stop
        else "stopped" if stopped else "complete"
    )
    manifest["status"] = status
    _append_ledger(out_dir, n, seed, client, status, run_id)
    manifest["health"] = {
        "minimum_samples": health_min_samples,
        "window": health_window,
        "max_invalid_rate": max_invalid_rate,
        "schema_invalid_bars": sum(invalid_schema_outcomes),
        "evaluated_bars": len(invalid_schema_outcomes),
        "stopped_reason": health_stop,
    }
    if stopped is not None:
        # Paid-for results are checkpointed; unsent bars stay pending, so the
        # same command resumes once the caps are raised deliberately.
        _write_json_atomic(manifest_path, manifest)
        raise JevBudgetError(f"run stopped locally by budget guard: {stopped}")
    if journal_stop is not None:
        _write_json_atomic(manifest_path, manifest)
        raise journal_stop
    if health_stop is not None:
        _write_json_atomic(manifest_path, manifest)
        raise JevHealthError(health_stop)
    manifest["complete"] = True
    _write_json_atomic(manifest_path, manifest)
    journal_records = [entry["record"] for entry in _read_checkpoint(checkpoint_path, run_id)]
    if journal_records:
        frame = pd.DataFrame(journal_records)
    else:
        part_paths = [parts_dir / name for name in manifest["parts"]]
        frames = [pd.read_parquet(path) for path in part_paths if path.is_file()]
        if not frames:
            raise RuntimeError("metering completed without a checkpoint part")
        frame = pd.concat(frames, ignore_index=True)
    if frame.empty:
        raise RuntimeError("metering completed without a checkpoint part")
    frame = frame.sort_values("sample_index").drop_duplicates("sample_index").reset_index(drop=True)
    frame = _attach_current_outcomes(frame, labels_by_ts, cfg["grid"])
    frame.to_parquet(output, index=False)
    failure_path = out_dir / f"metering_{n}_{run_id}_failures.json"
    failures = [
        manifest["failure_details"][index]
        for index in sorted(manifest["failure_details"], key=int)
    ]
    failure_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
    report = _report(
        frame, picked, failures, client, n, seed, workers, use_cache, cfg["grid"]["horizon_bars"],
        len(cfg["grid"]["tp"]) * len(cfg["grid"]["sl"]) * 2,
        time.perf_counter() - started,
        output, failure_path,
        composition,
    )
    report_name = f"diagnostic_replay_{n}_report.md" if n == 100 else f"inference_cost_{n}_report.md"
    report_path = Path("docs") / report_name
    report_path.write_text(report, encoding="utf-8")
    return report_path


def _report(frame, picked, failures, client, n, seed, workers, use_cache, horizon_bars, cell_count, elapsed, output, failure_path, composition) -> str:
    valid = frame[frame.get("response_valid", pd.Series(True, index=frame.index)).fillna(True)]
    probability_cols = [
        col for col in valid.columns
        if col.startswith("p_long_tp") or col.startswith("p_short_tp")
    ]
    outcome_cols = [f"outcome_{col[2:]}" for col in probability_cols]
    y = valid[outcome_cols].to_numpy().ravel()
    p = valid[probability_cols].to_numpy().ravel()
    nonambiguous = y != int(lab.Outcome.AMBIGUOUS)
    resolved = np.isin(y, [int(lab.Outcome.SL_FIRST), int(lab.Outcome.TP_FIRST)])
    predicted_tp = p >= 0.5
    actual_tp = y == int(lab.Outcome.TP_FIRST)
    usage = {key: int(valid[key].sum()) for key in (
        "input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"
    )}
    costs = valid["cost"].dropna()
    cost_sum = float(costs.sum()) if len(costs) else math.nan
    latencies = valid["latency_ms"].tolist()
    comp_bad = int((~composition["ok"]).sum())
    actual_counts = pd.Series(y[nonambiguous]).value_counts().to_dict()
    accuracy = float((predicted_tp[nonambiguous] == actual_tp[nonambiguous]).mean())
    resolved_accuracy = float((predicted_tp[resolved] == actual_tp[resolved]).mean()) if resolved.any() else math.nan
    brier = float(np.mean((p[nonambiguous] - actual_tp[nonambiguous]) ** 2))
    title = f"Jev {n}-bar diagnostic replay / inference cost report" if n == 100 else f"Jev {n}-request inference cost report"
    manifest_path = output.with_name(f"{output.stem}_manifest.json")
    parts_path = output.parent / f"{output.stem}_parts/part_*.parquet"
    journal_path = output.parent / f"{output.stem}_parts/checkpoint.jsonl"
    return f"""# {title}

Generated: 2026-09-21  
Input artifact: `{output}`  
Sampling seed: `{seed}`

## Executive result

- **Logical Jev requests:** {n}
- **Inference API requests:** {n} (one per sampled 15m bar)
- **Exchange/order requests:** 0 (inference-only run; no execution adapter)
- **Successful responses:** {len(valid)} ({len(valid) / n:.1%})
- **Failed responses:** {len(failures)}
- **Actual HTTP attempts:** {client.request_attempts}
- **Budget ledger (all billed attempts, incl. rejected responses):** {client.budget.snapshot() if client.budget else 'no budget guard'}
- **Request workers:** {workers}
- **Local response cache enabled:** {use_cache}
- **Checkpoint manifest:** `{manifest_path}` (atomic, resumable)
- **Checkpoint journal:** `{journal_path}` (one durable entry per settled bar)
- **Checkpoint parts:** `{parts_path}`
- **Unique 15m bars attempted:** {picked.ts.nunique()}
- **Successful prediction bars:** {len(valid)}
- **Coverage window:** {pd.to_datetime(picked.ts.min(), unit="ms", utc=True)} → {pd.to_datetime(picked.ts.max(), unit="ms", utc=True)}
- **Prediction cells evaluated:** {len(valid) * cell_count:,} ({cell_count} barrier cells per bar)
- **Candidate trade requests:** not emitted; this run measures inference only and does not contain an EV/signal/execution layer.

## Cost and latency

| Metric | Value |
|---|---:|
| Input tokens | {usage['input_tokens']:,} |
| Output tokens | {usage['output_tokens']:,} |
| Cache-read tokens | {usage['cache_read_input_tokens']:,} |
| Cache-creation tokens | {usage['cache_creation_input_tokens']:,} |
| Mean input tokens/request | {usage['input_tokens'] / max(len(valid), 1):,.1f} |
| Mean output tokens/request | {usage['output_tokens'] / max(len(valid), 1):,.1f} |
| Mean latency | {statistics.mean(latencies):,.1f} ms |
| p50 latency | {np.percentile(latencies, 50):,.1f} ms |
| p95 latency | {np.percentile(latencies, 95):,.1f} ms |
| Max latency | {max(latencies):,.1f} ms |
| Total wall-clock time | {elapsed:,.1f} s |
| Reported API cost | {_money(cost_sum)} |
| Cost/request | {_money(cost_sum / len(valid) if len(valid) and not math.isnan(cost_sum) else math.nan)} |
| Extrapolated cost / 2,000 bars | {_money(cost_sum * 2000 / len(valid) if len(valid) and not math.isnan(cost_sum) else math.nan)} |
| Extrapolated cost / 20,000 bars | {_money(cost_sum * 20000 / len(valid) if len(valid) and not math.isnan(cost_sum) else math.nan)} |
| Extrapolated cost / 105,000 bars | {_money(cost_sum * 105000 / len(valid) if len(valid) and not math.isnan(cost_sum) else math.nan)} |

## Predictive correctness

The API returns a probability that TP occurs before SL within {horizon_bars} bars. For this
report, `p >= 0.5` is treated as TP; ambiguous 1m ties are excluded. Timeout
is counted as a non-TP in the all-outcome metric.

| Metric | Value |
|---|---:|
| Schema success rate | {len(valid) / n:.1%} |
| Accuracy, TP vs. non-TP incl. timeout | {accuracy:.1%} |
| Accuracy, resolved TP vs. SL only | {resolved_accuracy:.1%} |
| Brier score, TP vs. non-TP | {brier:.4f} |
| Ambiguous cells excluded | {int((y == int(lab.Outcome.AMBIGUOUS)).sum()):,} |
| TP_FIRST cells | {actual_counts.get(int(lab.Outcome.TP_FIRST), 0):,} |
| SL_FIRST cells | {actual_counts.get(int(lab.Outcome.SL_FIRST), 0):,} |
| TIMEOUT cells | {actual_counts.get(int(lab.Outcome.TIMEOUT), 0):,} |

The accuracy figures are descriptive only; this diagnostic sample is not a
model-quality gate and no trading signal was executed.

## Sample composition

- Years: {", ".join(f"{k}={v}" for k, v in picked.year.value_counts().sort_index().items())}
- Quarters: {", ".join(f"{k}={v}" for k, v in picked.quarter.value_counts().sort_index().items())}
- Valid-response quarters: {", ".join(f"{k}={v}" for k, v in valid.quarter.value_counts().sort_index().items())}
- Volatility quartiles: {", ".join(f"{k}={v}" for k, v in picked.vol_q.value_counts().sort_index().items())}
- Direction terciles: {", ".join(f"{k}={v}" for k, v in picked.dir_t.value_counts().sort_index().items())}
- Composition checks outside the 0.5–2.0 population ratio: {comp_bad}

## Reproduction

```bash
PYTHONPATH=src .venv/bin/python -m jev_trader.meter
```

Raw rows: `{output}`  
Failed requests: `{failure_path}`  
Sample composition: `{composition_path}`
Attempted sample: `{output.with_name(f'{output.stem}_attempted.csv')}`
Resume state: `{manifest_path}`
"""


def _money(value: float) -> str:
    return "unreported" if math.isnan(value) else f"${value:.6f}"


if __name__ == "__main__":
    import sys

    from .cli import main

    raise SystemExit(main(["meter", *sys.argv[1:]]))
