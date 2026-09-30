"""真实行情进入 Jev 观察链路，不包含交易所下单路径。 / Real candles to Jev observations without order placement."""

from __future__ import annotations

import fcntl
import json
import sqlite3
import time
import urllib.error
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .features import BAR_MS
from .jev import (
    PROMPT_VERSION_OHLCV_V4, Budget, JevApiError, JevBudgetError, JevClient,
    JevConfigError, JevSchemaError, JevSettings, build_questions,
    canonical_hash, canonical_request_identity, prompt_contract,
    prompt_contract_v4,
)
from .meter import (
    DEFAULT_EST_COST_PER_REQUEST, _append_checkpoint, _append_response_event,
    _attempt_wal_state, _health_stop_reason, _read_attempt_wal,
    _read_response_events, _recover_missing_settlements, _replay_response,
)
from .markets import fetch_closed_bars, normalize_pair, server_time_ms
from .paper import PaperEngine
from .state import market_state_from_bars, market_state_from_bars_v4

MAX_SIGNAL_AGE_MS = 60_000
LIVE_MARKET = "live_market"
SYNTHETIC_FIXTURE = "synthetic_fixture"


def _budget_exhausted(client: JevClient) -> bool:
    budget = client.budget.snapshot()
    return budget["attempts"] >= budget["max_requests"] or (
        budget["committed_usd"] + client.budget.unit_usd > budget["max_usd"]
    )


def normalize_symbol(value: str) -> tuple[str, str]:
    """保留供 Binance USDT-M 调用的旧版公开助手。 / Legacy helper for Binance USDT-M callers."""
    return normalize_pair("binance", value)


def _public_json(path: str, params: dict[str, str | int]) -> object:
    from .markets import _binance_json

    return _binance_json(path, params)


def closed_klines(binance_symbol: str, server_ms: int, fetch_json=_public_json) -> pd.DataFrame:
    """在 Jev 推理前拒绝未收盘、过期或缺口 K 线。 / Reject partial, stale, or gapped candles before Jev."""
    return fetch_closed_bars("binance", binance_symbol, server_ms, fetch_json)


def _connect(path: Path, exchange: str) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute("PRAGMA synchronous=FULL")
    db.execute("""CREATE TABLE IF NOT EXISTS market_identity (
        identity_key TEXT PRIMARY KEY, identity_value TEXT NOT NULL
    )""")
    identity = db.execute(
        "SELECT identity_value FROM market_identity WHERE identity_key='exchange'"
    ).fetchone()
    if identity and identity[0] != exchange:
        db.close()
        raise RuntimeError(
            f"market database belongs to {identity[0]!r}, not requested {exchange!r}"
        )
    db.execute(
        "INSERT OR IGNORE INTO market_identity VALUES ('exchange', ?)", (exchange,)
    )
    db.execute("INSERT OR IGNORE INTO market_identity VALUES ('market_type', 'usdt-perpetual')")
    db.execute("""CREATE TABLE IF NOT EXISTS signal_runs (
        run_id TEXT NOT NULL, pair TEXT NOT NULL, exchange TEXT NOT NULL,
        market_type TEXT NOT NULL, feature_version TEXT NOT NULL,
        state_version TEXT NOT NULL, model_id TEXT NOT NULL,
        prompt_version TEXT NOT NULL, PRIMARY KEY (run_id, pair)
    )""")
    db.execute("""CREATE TABLE IF NOT EXISTS observations (
        run_id TEXT NOT NULL, symbol TEXT NOT NULL, ts INTEGER NOT NULL,
        observed_at_ms INTEGER NOT NULL, status TEXT NOT NULL,
        state_hash TEXT, model_id TEXT, prompt_version TEXT,
        latency_ms REAL, cost_usd REAL, answers_json TEXT, error TEXT,
        PRIMARY KEY (run_id, symbol, ts)
    )""")
    db.execute("""CREATE TABLE IF NOT EXISTS signal_run_provenance (
        run_id TEXT PRIMARY KEY, provenance TEXT NOT NULL,
        excluded_from_research INTEGER NOT NULL
    )""")
    db.execute("""CREATE TABLE IF NOT EXISTS research_sample_exclusions (
        run_id TEXT NOT NULL, symbol TEXT NOT NULL, ts INTEGER NOT NULL,
        reason TEXT NOT NULL, PRIMARY KEY (run_id, symbol, ts)
    )""")
    db.commit()
    return db


def _register_signal_run(db: sqlite3.Connection, run_id: str, pair: str,
                         exchange: str, settings: JevSettings,
                         data_provenance: str = LIVE_MARKET) -> None:
    if settings.prompt_version == "v3" and exchange == "binance":
        feature_version, state_version = "binance-features-v3", "jev-state-v3"
    elif settings.prompt_version == PROMPT_VERSION_OHLCV_V4:
        feature_version, state_version = "ohlcv-features-v1", "jev-state-v4"
    else:
        raise JevConfigError("signal run has no registered feature/state contract")
    identity = (
        run_id, pair, exchange, "usdt-perpetual", feature_version,
        state_version, settings.model_id, settings.prompt_version,
    )
    db.execute("INSERT OR IGNORE INTO signal_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)", identity)
    stored = db.execute(
        "SELECT run_id, pair, exchange, market_type, feature_version, "
        "state_version, model_id, prompt_version FROM signal_runs WHERE run_id=? AND pair=?",
        (run_id, pair),
    ).fetchone()
    if stored != identity:
        raise JevConfigError("run id is already bound to a different signal identity")
    provenance = db.execute(
        "SELECT provenance, excluded_from_research FROM signal_run_provenance "
        "WHERE run_id=?", (run_id,),
    ).fetchone()
    expected_provenance = (
        data_provenance, int(data_provenance == SYNTHETIC_FIXTURE),
    )
    if provenance is None:
        db.execute(
            "INSERT INTO signal_run_provenance VALUES (?, ?, ?)",
            (run_id, *expected_provenance),
        )
    elif provenance != expected_provenance:
        raise JevConfigError("run id is already bound to different data provenance")


def _record(db: sqlite3.Connection, run_id: str, symbol: str, ts: int,
            status: str, *, state_hash: str = "", model_id: str = "",
            prompt_version: str = "", latency_ms: float | None = None,
            cost_usd: float | None = None, answers: dict | None = None,
            error: str = "", data_provenance: str = LIVE_MARKET) -> None:
    db.execute("""INSERT OR IGNORE INTO observations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
        run_id, symbol, ts, int(time.time() * 1000), status, state_hash,
        model_id, prompt_version, latency_ms, cost_usd,
        json.dumps(answers, sort_keys=True) if answers is not None else None,
        error[:500],
    ))
    if data_provenance == SYNTHETIC_FIXTURE:
        db.execute(
            "INSERT OR IGNORE INTO research_sample_exclusions VALUES (?, ?, ?, ?)",
            (run_id, symbol, ts, SYNTHETIC_FIXTURE),
        )
    db.commit()


@contextmanager
def _single_process(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another shadow run is active") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def shadow_run_id(settings: JevSettings, grid: dict, exchange: str,
                  pairs: list[str], *,
                  data_provenance: str = LIVE_MARKET) -> str:
    """保留已付费 Binance v3 账本键，区分所有新批次。 / Preserve paid v3 ledger key; version new cohorts."""
    if data_provenance not in {LIVE_MARKET, SYNTHETIC_FIXTURE}:
        raise JevConfigError(f"unsupported data provenance {data_provenance!r}")
    questions = build_questions(grid)
    if exchange == "binance" and settings.prompt_version == "v3":
        if data_provenance != LIVE_MARKET:
            raise JevConfigError("synthetic fixtures cannot use the Binance v3 run identity")
        return canonical_hash({
            "kind": "shadow-v1", "url": settings.base_url,
            "model": settings.model_id, "prompt": settings.prompt_version,
            "contract": prompt_contract(grid["horizon_bars"]),
            "questions": questions,
        })[:16]
    if settings.prompt_version != PROMPT_VERSION_OHLCV_V4:
        raise JevConfigError(
            "non-legacy market runs require prompt_version jev-ohlcv-v4"
        )
    normalized_pairs = sorted({normalize_pair(exchange, pair)[0] for pair in pairs})
    return canonical_hash({
        "kind": "shadow-v2", "endpoint": settings.base_url,
        "exchange": exchange, "market_type": "usdt-perpetual",
        "pairs": normalized_pairs,
        "feature_version": "ohlcv-features-v1",
        "state_version": "jev-state-v4",
        "model_id": settings.model_id,
        "prompt_version": settings.prompt_version,
        "data_provenance": data_provenance,
        "contract": prompt_contract_v4(int(grid["horizon_bars"])),
        "questions": questions,
    })[:16]


def _budgeted_client(cfg: dict, data_dir: Path, max_requests: int,
                     max_usd: float, client: JevClient | None, *,
                     exchange: str = "binance", pairs: list[str] | None = None,
                     data_provenance: str = LIVE_MARKET,
                     ) -> tuple[JevClient, str, list[dict]]:
    settings = client.settings if client else JevSettings.from_config(cfg)
    questions = build_questions(cfg["grid"])
    run_id = shadow_run_id(
        settings, cfg["grid"], exchange, pairs or [],
        data_provenance=data_provenance,
    )
    attempt_path = data_dir / f"shadow_{run_id}_attempts.wal.jsonl"
    response_path = data_dir / f"shadow_{run_id}_responses.jsonl"
    ledger_path = data_dir / "jev_spend_ledger.jsonl"

    def append_attempt(event: dict) -> None:
        at = datetime.now(timezone.utc).isoformat()
        _append_checkpoint(attempt_path, {**event, "run_id": run_id, "at": at})
        _append_checkpoint(ledger_path, {
            "at": at, "n": 0, "status": "shadow", "run_id": run_id,
            "model_id": settings.model_id, "prompt_version": settings.prompt_version,
            "event": event["event"], "budget": event["budget"],
        })

    events = _read_attempt_wal(attempt_path, run_id)
    responses = _read_response_events(response_path, run_id)
    repairs = _recover_missing_settlements(
        events, responses, max_requests, max_usd, DEFAULT_EST_COST_PER_REQUEST
    )
    for event in repairs:
        append_attempt(event)
    events.extend(repairs)
    snapshot, _ = _attempt_wal_state(events, max_requests, max_usd)
    budget = Budget(max_requests, max_usd, DEFAULT_EST_COST_PER_REQUEST,
                    on_attempt_event=append_attempt)
    budget.restore(snapshot if events else None)
    if client is None:
        client = JevClient(settings)
    client.budget = budget
    client.on_response = lambda event: _append_response_event(response_path, event)
    return client, run_id, responses


def _recover_unrecorded(db: sqlite3.Connection, run_id: str, data_dir: Path,
                        questions: dict, responses: list[dict], *,
                        data_provenance: str = LIVE_MARKET) -> set[tuple[str, int]]:
    events = _read_attempt_wal(data_dir / f"shadow_{run_id}_attempts.wal.jsonl", run_id)
    latest_response = {
        int(event["attempt"]): event for event in responses if event.get("attempt") is not None
    }
    by_bar: dict[tuple[str, int], list[dict]] = {}
    for event in events:
        if event["event"] != "reserve":
            continue
        context = event.get("context") or {}
        if "symbol" not in context or "ts" not in context:
            continue
        key = (str(context["symbol"]), int(context["ts"]))
        by_bar.setdefault(key, []).append(event)
    for key, attempts in by_bar.items():
        if db.execute(
            "SELECT 1 FROM observations WHERE run_id=? AND symbol=? AND ts=?",
            (run_id, *key),
        ).fetchone():
            continue
        answered = next(
            (event for event in reversed(attempts)
             if int(event["attempt"]) in latest_response), None
        )
        response = latest_response[int(answered["attempt"])] if answered else None
        context = (answered or attempts[-1])["context"]
        result, error = (_replay_response(response, questions, context["state_hash"])
                         if response else (None, "request outcome uncertain; not retried"))
        if result:
            _record(db, run_id, *key, "valid", state_hash=result.state_hash,
                    model_id=result.model, cost_usd=result.usage.get("cost"),
                    answers=result.answers, data_provenance=data_provenance)
        else:
            _record(db, run_id, *key, "schema_error" if response else "uncertain",
                    state_hash=context.get("state_hash", ""), error=error or "",
                    data_provenance=data_provenance)
    return set(by_bar)


def run_once(symbols: list[str], cfg: dict, *, exchange: str = "binance",
             live_jev: bool = False,
             dry_run_trades: bool = False,
             max_requests: int | None = None, max_usd: float | None = None,
             fetch_json=None, client: JevClient | None = None,
             data_provenance: str = LIVE_MARKET) -> list[dict]:
    """观察已收盘 K 线；可选执行仅为本地 SQLite 模拟。 / Observe closed candles with local-only paper execution."""
    if dry_run_trades and not live_jev:
        raise ValueError("--dry-run-trades requires --live-jev")
    if exchange not in {"binance", "okx"}:
        raise ValueError(f"unsupported exchange {exchange!r}")
    if live_jev and (max_requests is None or max_usd is None):
        raise ValueError("live Jev observation requires --max-requests and --max-usd")
    prompt_version = (client.settings.prompt_version if client else
                      cfg.get("jev", {}).get("prompt_version", "v3"))
    if live_jev and exchange == "okx" and prompt_version != PROMPT_VERSION_OHLCV_V4:
        raise JevConfigError("OKX live inference requires prompt_version jev-ohlcv-v4")
    if live_jev and exchange == "binance" and prompt_version != "v3":
        raise JevConfigError(
            "Binance live inference requires the registered v3 feature/state contract"
        )
    if data_provenance not in {LIVE_MARKET, SYNTHETIC_FIXTURE}:
        raise JevConfigError(f"unsupported data provenance {data_provenance!r}")
    if data_provenance == SYNTHETIC_FIXTURE and not (
        live_jev and exchange == "okx" and prompt_version == PROMPT_VERSION_OHLCV_V4
    ):
        raise JevConfigError("synthetic fixture provenance is only supported for OKX v4 inference")
    data_dir = Path(cfg["data_dir"])
    if data_provenance == SYNTHETIC_FIXTURE:
        db_path = data_dir / "shadow_okx_synthetic.sqlite"
        lock_path = data_dir / "shadow_okx_synthetic.lock"
    else:
        db_path = data_dir / ("shadow.sqlite" if exchange == "binance" else "shadow_okx.sqlite")
        lock_path = data_dir / ("shadow.lock" if exchange == "binance" else "shadow_okx.lock")
    with _single_process(lock_path), closing(_connect(db_path, exchange)) as db:
        jev = None
        run_id = ("market-only:binance:usdt-perpetual:ohlcv-v1" if exchange == "binance"
                  else "market-only:okx:usdt-perpetual:ohlcv-v1")
        inference_block_reason = None
        questions = build_questions(cfg["grid"])
        attempted: set[tuple[str, int]] = set()
        health_outcomes: list[bool] = []
        if live_jev:
            jev, run_id, responses = _budgeted_client(
                cfg, data_dir, max_requests, max_usd, client,
                exchange=exchange, pairs=symbols, data_provenance=data_provenance,
            )
            attempted = _recover_unrecorded(
                db, run_id, data_dir, questions, responses,
                data_provenance=data_provenance,
            )
            health_outcomes = [row[0] == "schema_error" for row in db.execute(
                "SELECT status FROM observations WHERE run_id=? AND status IN "
                "('valid','late','schema_error') ORDER BY rowid", (run_id,),
            )]
            reason = _health_stop_reason(health_outcomes)
            if reason:
                inference_block_reason = "health_stop"

        get_time = fetch_json or None
        server_ms = server_time_ms(exchange, get_time)
        observed_at = time.monotonic()

        def current_server_ms() -> int:
            return server_ms + int((time.monotonic() - observed_at) * 1000)

        paper = PaperEngine(db, run_id, cfg["grid"], cfg["costs"]) if live_jev else None
        if paper and inference_block_reason:
            paper.block_run_entries(inference_block_reason, current_server_ms())
        output = []
        for requested in dict.fromkeys(symbols):
            symbol, native_symbol = normalize_pair(exchange, requested)
            if jev:
                _register_signal_run(
                    db, run_id, symbol, exchange, jev.settings,
                    data_provenance=data_provenance,
                )
            try:
                bars = fetch_closed_bars(exchange, native_symbol, server_ms, fetch_json)
            except urllib.error.HTTPError as exc:
                if exc.code != 400:  # 限流或故障时不持续轮询 / Do not poll through rate limits or outages.
                    raise
                if paper:
                    paper.fail_market_gap(
                        symbol, server_ms // BAR_MS * BAR_MS - BAR_MS
                    )
                output.append({"symbol": symbol, "status": "market_error", "error": str(exc)})
                continue
            except ValueError as exc:
                if paper:
                    paper.fail_market_gap(
                        symbol, server_ms // BAR_MS * BAR_MS - BAR_MS
                    )
                output.append({"symbol": symbol, "status": "market_error", "error": str(exc)})
                continue
            ts = int(bars.iloc[-1]["ts"])
            now_ms = current_server_ms()
            stale_candle = now_ms - (ts + BAR_MS) >= MAX_SIGNAL_AGE_MS
            if jev and inference_block_reason is None and _budget_exhausted(jev):
                inference_block_reason = "budget_exhausted"
                if paper:
                    paper.block_run_entries(inference_block_reason, now_ms)
            if paper:
                if inference_block_reason:
                    paper.block_entries(symbol, inference_block_reason, ts)
                elif stale_candle:
                    paper.block_entries(symbol, "stale_candle", ts)
                gap_events = paper.advance(
                    symbol, bars,
                    allow_entries=(
                        data_provenance == LIVE_MARKET
                        and inference_block_reason is None and not stale_candle
                    ),
                )
                if any(event["type"] == "market_gap" for event in gap_events):
                    paper.record_no_signal(symbol, ts, "", "market_data_gap")
                    output.append({"symbol": symbol, "ts": ts, "status": "no_signal",
                                   "reason": "market_data_gap"})
                    continue
            if db.execute(
                "SELECT 1 FROM observations WHERE run_id=? AND symbol=? AND ts=?",
                (run_id, symbol, ts),
            ).fetchone() or (symbol, ts) in attempted:
                existing = db.execute(
                    "SELECT status, state_hash, answers_json, error FROM observations "
                    "WHERE run_id=? AND symbol=? AND ts=?", (run_id, symbol, ts),
                ).fetchone()
                if paper:
                    if existing and existing[0] == "valid" and existing[2]:
                        denial = inference_block_reason or (
                            "stale_candle" if stale_candle else
                            SYNTHETIC_FIXTURE if data_provenance == SYNTHETIC_FIXTURE else None
                        )
                        signal = paper.record_signal(
                            symbol, ts, existing[1] or "", json.loads(existing[2]),
                            dry_run=dry_run_trades and denial is None,
                            allow_entry=denial is None,
                            acceptance_reason=denial or "accepted",
                        )
                        output.append({
                            "symbol": symbol, "ts": ts,
                            "status": ("inference_blocked" if inference_block_reason
                                       else "stale" if stale_candle else "already_recorded"),
                            "signal": signal,
                            **({"reason": denial} if denial else {}),
                        })
                    else:
                        reason = ("late_response" if existing and existing[0] == "late"
                                  else existing[0] if existing else "already_attempted")
                        paper.record_no_signal(symbol, ts, existing[1] or "" if existing else "",
                                               reason)
                        output.append({"symbol": symbol, "ts": ts,
                                       "status": "already_recorded", "reason": reason})
                else:
                    output.append({"symbol": symbol, "ts": ts, "status": "already_recorded"})
                continue
            if inference_block_reason:
                if paper:
                    paper.record_no_signal(symbol, ts, "", inference_block_reason)
                output.append({"symbol": symbol, "ts": ts,
                               "status": "inference_blocked",
                               "reason": inference_block_reason})
                continue
            try:
                state = (market_state_from_bars_v4(exchange, symbol, bars, cfg["grid"])
                         if exchange == "okx" else
                         market_state_from_bars(symbol, bars, cfg["grid"]))
            except ValueError as exc:
                _record(db, run_id, symbol, ts, "market_error", error=str(exc),
                        data_provenance=data_provenance)
                if paper:
                    paper.record_no_signal(symbol, ts, "", "incomplete_features")
                output.append({"symbol": symbol, "ts": ts, "status": "market_error", "error": str(exc)})
                continue
            if stale_candle:
                status = "stale"
                state_hash = canonical_hash(state)
                _record(db, run_id, symbol, ts, status, state_hash=state_hash,
                        error="closed candle older than 60s",
                        data_provenance=data_provenance)
                if paper:
                    paper.record_no_signal(symbol, ts, state_hash, "stale_candle")
            elif not live_jev:
                status = "market_ok"
                state_hash = canonical_hash(state)
                _record(db, run_id, symbol, ts, status, state_hash=state_hash,
                        data_provenance=data_provenance)
                if exchange == "okx":
                    output.append({
                        "symbol": symbol, "ts": ts, "status": status,
                        "mode": "market_only", "state_hash": state_hash,
                        "feature_version": state["feature_version"],
                        "state_version": state["state_version"],
                    })
                    continue
            else:
                state_hash = canonical_hash(canonical_request_identity(jev.settings, state, questions))
                try:
                    result = jev.decide(
                        state, questions,
                        cache_namespace=(
                            data_provenance
                            if jev.settings.prompt_version == PROMPT_VERSION_OHLCV_V4
                            else None
                        ),
                        request_context={
                            "run_id": run_id, "symbol": symbol, "ts": ts,
                            "state_hash": state_hash,
                            "data_provenance": data_provenance,
                        },
                    )
                except JevSchemaError as exc:
                    status = "schema_error"
                    _record(db, run_id, symbol, ts, status, state_hash=state_hash,
                            model_id=jev.settings.model_id,
                            prompt_version=jev.settings.prompt_version, error=str(exc),
                            data_provenance=data_provenance)
                    if paper:
                        paper.record_no_signal(symbol, ts, state_hash, "schema_error")
                except JevApiError as exc:
                    status = "api_error"
                    _record(db, run_id, symbol, ts, status, state_hash=state_hash,
                            model_id=jev.settings.model_id,
                            prompt_version=jev.settings.prompt_version, error=str(exc),
                            data_provenance=data_provenance)
                    if paper:
                        paper.record_no_signal(symbol, ts, state_hash, "api_error")
                except JevBudgetError as exc:
                    status = "budget_stopped"
                    inference_block_reason = "budget_exhausted"
                    if paper:
                        paper.block_run_entries(inference_block_reason, current_server_ms())
                    _record(db, run_id, symbol, ts, status, state_hash=state_hash,
                            model_id=jev.settings.model_id,
                            prompt_version=jev.settings.prompt_version, error=str(exc),
                            data_provenance=data_provenance)
                    if paper:
                        paper.record_no_signal(symbol, ts, state_hash, inference_block_reason)
                else:
                    status = "valid"
                    if server_ms + int((time.monotonic() - observed_at) * 1000) - (ts + BAR_MS) >= MAX_SIGNAL_AGE_MS:
                        status = "late"
                    _record(db, run_id, symbol, ts, status, state_hash=result.state_hash,
                            model_id=result.model,
                            prompt_version=jev.settings.prompt_version,
                            latency_ms=result.latency_ms,
                            cost_usd=result.usage.get("cost"), answers=result.answers,
                            data_provenance=data_provenance)
            if live_jev and status in {"valid", "late", "schema_error"}:
                health_outcomes.append(status == "schema_error")
                if _health_stop_reason(health_outcomes):
                    inference_block_reason = "health_stop"
            if (live_jev and inference_block_reason is None
                    and _budget_exhausted(jev)):
                inference_block_reason = "budget_exhausted"
            if paper and (inference_block_reason
                          or data_provenance == SYNTHETIC_FIXTURE):
                paper.block_run_entries(
                    inference_block_reason or SYNTHETIC_FIXTURE,
                    current_server_ms(),
                )
            entry_block_reason = (
                SYNTHETIC_FIXTURE
                if data_provenance == SYNTHETIC_FIXTURE
                else inference_block_reason
            )
            if paper and status == "valid":
                signal = paper.record_signal(
                    symbol, ts, result.state_hash, result.answers,
                    dry_run=dry_run_trades and entry_block_reason is None,
                    allow_entry=entry_block_reason is None,
                    acceptance_reason=entry_block_reason or "accepted",
                )
            elif paper and status == "late":
                paper.record_no_signal(symbol, ts, result.state_hash, "late_response")
            output.append({"symbol": symbol, "ts": ts, "status": status})
            if paper and status in {"valid", "late", "schema_error", "api_error", "stale"}:
                if status == "valid":
                    output[-1]["signal"] = signal
                elif status == "api_error":
                    output[-1]["reason"] = "api_error"
        return output
