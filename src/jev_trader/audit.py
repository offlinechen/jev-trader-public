"""Jev spend audit: reconcile local run records against OpenRouter's ledger.

Four sources, from least to most authoritative:

1. local run reports (docs/*report.md) -- optional and not published;
2. run artifacts (data/metering_*.parquet) -- present only where the run was
   executed, since data/ is gitignored;
3. the response cache (data/jev_cache) -- every *accepted* live response that
   was cached, including development calls that never became a report;
4. OpenRouter itself -- GET /api/v1/key (per-key usage, works with the normal
   key) and GET /api/v1/credits (account-wide, management key only).

Why local records alone cannot be trusted as the total: before the budget
guard, a 2xx response that failed local validation was billed but recorded
with no cost, a validation failure was silently retried (billed twice), and
`--no-cache` smoke calls and the live test were never cached. The provider's
per-key usage is the only complete number; the audit reports the gap
explicitly instead of guessing at it.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd

RUN_LABELS = {100: "metering (100 diagnostic)", 2000: "G3a (2,000)", 10000: "G3b attempt (10,000)"}

_REPORT_FIELDS = {
    "successes": r"\*\*Successful responses:\*\*\s*([\d,]+)",
    "failures": r"\*\*Failed responses:\*\*\s*([\d,]+)",
    "attempts": r"\*\*Actual HTTP attempts:\*\*\s*([\d,]+)",
    "input_tokens": r"\|\s*Input tokens\s*\|\s*([\d,]+)",
    "output_tokens": r"\|\s*Output tokens\s*\|\s*([\d,]+)",
    "cost_usd": r"\|\s*Reported API cost\s*\|\s*\$([\d.]+)",
    "logical": r"\*\*Logical Jev requests:\*\*\s*([\d,]+)",
}


def classify_error(text: str) -> str:
    """Bucket a recorded request error by whether it can have been billed."""
    if not text:
        return "ok"
    m = re.search(r"failed \((\d{3})\)", text)
    if m:
        return f"http_{m.group(1)}"          # rejected upstream: not billed
    if "transport failed" in text:
        return "transport"                   # may or may not have been billed
    return "rejected_2xx"                    # answered, billed, rejected locally


def from_reports(docs_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(docs_dir.glob("*report.md")):
        text = path.read_text(encoding="utf-8")
        if "Reported API cost" not in text:
            continue
        row = {"source": f"report:{path.name}"}
        for key, pattern in _REPORT_FIELDS.items():
            m = re.search(pattern, text)
            row[key] = float(m.group(1).replace(",", "")) if m else float("nan")
        run_match = re.search(r"metering_(\d+)_([0-9a-f]{16})\.parquet", text)
        row["run_id"] = run_match.group(2) if run_match else "legacy"
        row["run"] = RUN_LABELS.get(int(row["logical"]), f"run n={int(row['logical'])}")
        rows.append(row)
    return pd.DataFrame(rows)


def _ledger_by_identity(data_dir: Path) -> dict[tuple[int, str], dict]:
    """Return one cumulative ledger snapshot per logical run identity."""
    out = {}
    path = data_dir / "jev_spend_ledger.jsonl"
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        budget = entry.get("budget") or {}
        key = (int(entry.get("n", 0)), entry.get("run_id", "legacy"))
        snapshot = {
            "attempts": int(budget.get("attempts", 0) or 0),
            "cost_usd": max(
                float(budget.get("actual_usd", 0.0) or 0.0)
                + float(budget.get("estimated_usd", 0.0) or 0.0),
                float(budget.get("committed_usd", 0.0) or 0.0),
            ),
            "budget": budget,
            "status": entry.get("status", "unknown"),
        }
        out[key] = snapshot  # append order makes the last event authoritative
    return out


def from_artifacts(data_dir: Path, reports: pd.DataFrame | None = None) -> pd.DataFrame:
    report_by_identity = {}
    if reports is not None and len(reports):
        report_by_identity = {
            (int(row.logical), getattr(row, "run_id", "legacy")): row
            for row in reports.itertuples(index=False)
        }
    ledger_by_identity = _ledger_by_identity(data_dir)
    rows = []
    for path in sorted(data_dir.glob("metering_*.parquet")):
        m = re.fullmatch(r"metering_(\d+)(?:_([0-9a-f]{16}))?\.parquet", path.name)
        if not m:
            continue
        n = int(m.group(1))
        run_id = m.group(2) or "legacy"
        stem = path.stem
        df = pd.read_parquet(path)
        has_valid_col = "response_valid" in df
        ok = df["response_valid"].astype(bool) if has_valid_col else pd.Series(True, index=df.index)
        errors = df.get("request_error", pd.Series("", index=df.index)).fillna("").map(classify_error)
        manifest = data_dir / f"{stem}_manifest.json"
        ledger = {}
        if manifest.is_file():
            ledger = json.loads(manifest.read_text(encoding="utf-8")).get("budget", {}) or {}
        failure_path = data_dir / f"{stem}_failures.json"
        failures = []
        if failure_path.is_file():
            try:
                failures = json.loads(failure_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                failures = []
        report = report_by_identity.get((n, run_id))
        ledger_entry = ledger_by_identity.get((n, run_id))
        manifest_attempts = ledger.get("attempts", float("nan"))
        attempts = ledger_entry["attempts"] if ledger_entry else manifest_attempts
        if attempts != attempts and report is not None:
            attempts = getattr(report, "attempts", float("nan"))
        recorded_cost = (
            ledger_entry["cost_usd"] if ledger_entry is not None
            else (
                float(ledger.get("actual_usd", 0.0) or 0.0)
                + float(ledger.get("estimated_usd", 0.0) or 0.0)
                if "actual_usd" in ledger or "estimated_usd" in ledger
                else float(df.loc[ok, "cost"].sum(skipna=True))
            )
        )
        row = {
            "source": f"artifact:{path.name}",
            "run": RUN_LABELS.get(n, f"run n={n}"),
            "logical": n,
            "run_id": run_id,
            "identity": f"{n}:{run_id}",
            "successes": int(ok.sum()),
            "failures": int((~ok).sum()) + (0 if has_valid_col else len(failures)),
            "attempts": attempts,
            "input_tokens": float(df.loc[ok, "input_tokens"].sum()),
            "output_tokens": float(df.loc[ok, "output_tokens"].sum()),
            "cost_usd": recorded_cost,
            "status": ledger_entry["status"] if ledger_entry else "artifact",
        }
        for bucket, count in errors[~ok].value_counts().items():
            row[f"fail_{bucket}"] = int(count)
        if not has_valid_col:
            for failure in failures:
                bucket = classify_error(str(failure.get("error", "")))
                row[f"fail_{bucket}"] = row.get(f"fail_{bucket}", 0) + 1
        rows.append(row)
    artifact_ids = {(row["logical"], row["run_id"]) for row in rows}
    for (n, run_id), entry in ledger_by_identity.items():
        if (n, run_id) in artifact_ids:
            continue
        status = entry["status"]
        rows.append({
            "source": "ledger-only",
            "run": f"ledger-only ({status})",
            "logical": n,
            "run_id": run_id,
            "identity": f"{n}:{run_id}",
            "successes": float("nan"),
            "failures": float("nan"),
            "attempts": entry["attempts"],
            "input_tokens": float("nan"),
            "output_tokens": float("nan"),
            "cost_usd": entry["cost_usd"],
            "status": status,
        })
    return pd.DataFrame(rows)


def from_cache(cache_dir: Path) -> dict:
    n, cost, tin, tout = 0, 0.0, 0, 0
    for path in cache_dir.glob("*.json"):
        try:
            usage = json.loads(path.read_text(encoding="utf-8")).get("usage", {})
        except (OSError, ValueError):
            continue
        n += 1
        cost += float(usage.get("cost", 0) or 0)
        tin += int(usage.get("input_tokens", usage.get("inputTokens", 0)) or 0)
        tout += int(usage.get("output_tokens", usage.get("outputTokens", 0)) or 0)
    return {"responses": n, "cost_usd": cost, "input_tokens": tin, "output_tokens": tout}


def _get(url: str, key: str, timeout: float = 20.0) -> tuple[int, dict]:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read())
        except ValueError:
            return exc.code, {}
    except (urllib.error.URLError, TimeoutError) as exc:
        return 0, {"error": str(exc)}


def from_provider(api_key: str | None, base_url: str | None,
                  management_key: str | None = None) -> dict:
    """Per-key usage via /api/v1/key; account totals via /api/v1/credits."""
    if not api_key or not base_url:
        return {"available": False, "reason": "no JEV_API_KEY / JEV_BASE_URL in this environment"}
    origin = "{0.scheme}://{0.netloc}".format(urlsplit(base_url))
    out: dict = {"available": True}
    status, body = _get(f"{origin}/api/v1/key", api_key)
    out["key_status"] = status
    if status == 200:
        d = body.get("data", {})
        for field in ("usage", "limit", "limit_remaining", "usage_daily",
                      "usage_weekly", "usage_monthly", "byok_usage"):
            out[f"key_{field}"] = d.get(field)
    status, body = _get(f"{origin}/api/v1/credits", management_key or api_key)
    out["credits_status"] = status
    if status == 200:
        d = body.get("data", {})
        out["account_total_credits"] = d.get("total_credits")
        out["account_total_usage"] = d.get("total_usage")
        if d.get("total_credits") is not None and d.get("total_usage") is not None:
            out["account_remaining"] = d["total_credits"] - d["total_usage"]
    elif status == 403:
        out["credits_note"] = "account totals need a management key (OPENROUTER_MANAGEMENT_KEY)"
    return out


def reconcile(runs: pd.DataFrame, cache: dict, provider: dict) -> dict:
    runs_cost = float(runs["cost_usd"].sum()) if len(runs) else 0.0
    succ = float(runs["successes"].sum()) if len(runs) else 0.0
    out = {
        "run_successes": int(succ),
        "run_failures": int(runs["failures"].sum()) if len(runs) else 0,
        "run_attempts": float(runs["attempts"].sum()) if len(runs) else float("nan"),
        "run_input_tokens": float(runs["input_tokens"].sum()) if len(runs) else 0.0,
        "run_output_tokens": float(runs["output_tokens"].sum()) if len(runs) else 0.0,
        "run_cost_usd": runs_cost,
        "usd_per_success": runs_cost / succ if succ else float("nan"),
        "cache_responses": cache["responses"],
        "cache_cost_usd": cache["cost_usd"],
        # Accepted responses in the cache that no run report accounts for.
        "dev_cached_usd": max(0.0, cache["cost_usd"] - runs_cost) if cache["responses"] else float("nan"),
    }
    usage = provider.get("key_usage")
    if isinstance(usage, (int, float)):
        known = max(runs_cost, cache["cost_usd"], float(cache.get("ledger_usd", 0.0)))
        out["provider_key_usage_usd"] = float(usage)
        out["unrecorded_usd"] = float(usage) - known
    return out


def key_audit(root: Path, api_key: str | None) -> dict:
    """Check outputs for accidental key exposure without returning the key."""
    if not api_key:
        return {"configured": False, "exposed_files": []}
    exposed = []
    for folder in ("docs", "src", "tests", "config", "data"):
        base = root / folder
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix in {".parquet", ".pyc"}:
                continue
            try:
                if api_key in path.read_text(encoding="utf-8"):
                    exposed.append(str(path.relative_to(root)))
            except (OSError, UnicodeDecodeError):
                continue
    return {
        "configured": True,
        "exposed_files": exposed,
        "old_key_rotation_verifiable": False,
        "rotation_note": "No previous key was supplied; rotation cannot be proven from local files.",
    }


def quarter_audit(data_dir: Path) -> dict:
    """Return attempted/valid quarter counts from the local run artifacts."""
    out = {}
    for path in sorted(data_dir.glob("metering_*.parquet")):
        match = re.fullmatch(r"metering_(\d+)(?:_([0-9a-f]{16}))?\.parquet", path.name)
        if not match:
            continue
        n = int(match.group(1))
        stem = path.stem
        attempted_path = data_dir / f"{stem}_attempted.csv"
        composition = data_dir / f"{stem}_composition.csv"
        attempted = {}
        source = attempted_path if attempted_path.is_file() else composition
        if source.is_file():
            try:
                c = pd.read_csv(source)
                if "quarter" in c:
                    attempted = c["quarter"].value_counts().sort_index().to_dict()
            except (OSError, ValueError):
                pass
        frame = pd.read_parquet(path)
        if "quarter" in frame:
            if "response_valid" in frame:
                frame = frame[frame["response_valid"].fillna(False)]
            valid = frame["quarter"].value_counts().sort_index().to_dict()
        else:
            valid = {}
        run_id = match.group(2) or "legacy"
        identity = f"{n}:{run_id}"
        out[identity] = {
            "n": n,
            "run_id": run_id,
            "attempted": ({str(k): int(v) for k, v in attempted.items()} if attempted else "unavailable (legacy artifact)"),
            "valid": {str(k): int(v) for k, v in valid.items()},
        }
    return out


def run(root: Path, api_key: str | None = None, base_url: str | None = None,
        management_key: str | None = None, *, query_provider: bool = True) -> tuple[str, dict]:
    data_dir, docs_dir = root / "data", root / "docs"
    reports = from_reports(docs_dir)
    artifacts = from_artifacts(data_dir, reports) if data_dir.is_dir() else pd.DataFrame()
    runs = artifacts if len(artifacts) else reports
    if len(artifacts) and artifacts["source"].eq("ledger-only").all():
        source = "append-only ledger (run artifacts absent)"
    else:
        source = "run artifacts" if len(artifacts) else "committed reports (raw artifacts not on this machine)"
    cache = from_cache(data_dir / "jev_cache")
    ledger = data_dir / "jev_spend_ledger.jsonl"
    ledger_entries = _ledger_by_identity(data_dir)
    cache["ledger_runs"] = len(ledger_entries)
    cache["ledger_usd"] = sum(entry["cost_usd"] for entry in ledger_entries.values())
    provider = (
        from_provider(api_key, base_url, management_key)
        if query_provider else
        {"available": False, "reason": "offline audit: provider not queried"}
    )
    rec = reconcile(runs, cache, provider)
    security = key_audit(root, api_key)
    quarters = quarter_audit(data_dir) if data_dir.is_dir() else {}
    return _render(runs, source, cache, provider, rec, security, quarters), {**rec, "security": security, "quarters": quarters}


def _md_table(df: pd.DataFrame) -> str:
    def cell(v):
        if isinstance(v, float):
            return "" if v != v else (f"{v:,.6f}" if 0 < abs(v) < 100 and v != int(v) else f"{v:,.0f}")
        return str(v)
    head = "| " + " | ".join(df.columns) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def _render(runs, source, cache, provider, rec, security=None, quarters=None) -> str:
    def money(x):
        return "n/a" if x is None or x != x else f"${x:.6f}"

    cols = [c for c in ("run", "status", "logical", "run_id", "identity", "successes", "failures", "attempts",
                        "input_tokens", "output_tokens", "cost_usd") if c in runs]
    fail_cols = [c for c in runs.columns if c.startswith("fail_")]
    table = _md_table(runs[cols + fail_cols]) if len(runs) else "_no runs found_"
    lines = [
        "# Jev spend audit", "",
        f"Run source: **{source}**", "",
        table, "",
        "## Totals from run records", "",
        f"- Successful requests: **{rec['run_successes']:,}**",
        f"- Failed logical requests: **{rec['run_failures']:,}**",
        f"- HTTP attempts: **{rec['run_attempts']:,.0f}**" if rec["run_attempts"] == rec["run_attempts"] else "- HTTP attempts: n/a",
        f"- Input tokens: **{rec['run_input_tokens']:,.0f}**; output tokens: **{rec['run_output_tokens']:,.0f}**",
        f"- Recorded spend: **{money(rec['run_cost_usd'])}**",
        f"- Cost per successful bar: **{money(rec['usd_per_success'])}**", "",
        "## Response cache", "",
        f"- Cached accepted responses: {cache['responses']:,}, cost {money(cache['cost_usd'])}",
        f"- Accepted responses outside any run report (development): {money(rec['dev_cached_usd'])}",
        f"- Append-only ledger: {cache.get('ledger_runs', 0)} runs, {money(cache.get('ledger_usd'))} "
        "(every billed attempt since the budget guard; absent before it)", "",
        "## Credential/output audit", "",
        f"- Current key configured: **{bool((security or {}).get('configured'))}**",
        f"- Key occurrences in reports/source/test/config/data text: **{len((security or {}).get('exposed_files', []))}**",
        f"- Key rotation independently confirmed: **{bool((security or {}).get('old_key_rotation_verifiable'))}**",
        f"- Rotation note: {(security or {}).get('rotation_note', 'not checked')}", "",
        "## Quarterly sample coverage", "",
        "Attempted counts come from the stratified composition artifact; valid counts "
        "come from response-valid parquet rows.", "",
        "## OpenRouter", "",
    ]
    for identity, values in (quarters or {}).items():
        lines.insert(-2, f"- {identity}: attempted {values['attempted']}; valid {values['valid']}")
    if not provider.get("available"):
        lines.append(f"- Not queried: {provider.get('reason')}")
    else:
        lines += [
            f"- `/api/v1/key` status {provider.get('key_status')}: usage {money(provider.get('key_usage'))}, "
            f"limit {provider.get('key_limit')}, remaining {provider.get('key_limit_remaining')}, "
            f"daily {provider.get('key_usage_daily')}, monthly {provider.get('key_usage_monthly')}",
            f"- `/api/v1/credits` status {provider.get('credits_status')}: "
            + (f"purchased {money(provider.get('account_total_credits'))}, used "
               f"{money(provider.get('account_total_usage'))}, remaining {money(provider.get('account_remaining'))}"
               if provider.get("credits_status") == 200 else provider.get("credits_note", "unavailable")),
        ]
        if "unrecorded_usd" in rec:
            lines.append(
                f"- **Billed but not recorded locally: {money(rec['unrecorded_usd'])}** "
                "(provider key usage minus the larger of run records and cache)"
            )
    return "\n".join(lines) + "\n"
