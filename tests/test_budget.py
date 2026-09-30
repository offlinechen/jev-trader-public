"""支出保护必须在本地发请求前停止，不能依赖供应商余额耗尽。
测试通过伪传输驱动真实客户端；未调用传输意味着请求确实未离开本机。

The spend guard must stop locally, before money is spent -- never rely on
the provider running out of credit.

Every test drives the real JevClient through a fake transport, so "the
transport was not called" is literal proof that no request left the machine.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone

import pandas as pd
import pytest

from jev_trader.audit import (
    _render, classify_error, from_artifacts, from_reports, quarter_audit, reconcile,
    run as audit_run,
)
from jev_trader.jev import (
    Budget, JevApiError, JevBudgetError, JevClient, JevHealthError, JevSettings,
    canonical_request_identity,
)
from jev_trader.labels import level_col
from jev_trader.meter import (
    _append_attempt_event, _append_checkpoint, _attach_current_outcomes,
    _attempt_wal_state, _load_or_create_manifest, _read_attempt_wal,
    _cache_response_for_attempt, _health_stop_reason, _read_checkpoint,
    _recover_missing_settlements, _resume_health_outcomes, _replay_response,
    _run_identity, preflight, run as meter_run,
)

Q = {"smoke": {"type": "noul", "instructions": "?"}}


def settings(tmp_path, **kw):
    return JevSettings("secret", "https://example.test/api/alpha/decisions", "jev-test",
                       cache_dir=tmp_path, retry_delay_s=0, **kw)


def ok(cost=0.0003, value=0.5):
    usage = {"input_tokens": 10, "output_tokens": 20}
    if cost is not None:
        usage["cost"] = cost
    return 200, json.dumps({"model": "m", "answers": {"smoke": {"type": "noul", "noul": value}},
                            "usage": usage}).encode()


def counting(responses):
    calls = []

    def transport(url, headers, body):
        calls.append(1)
        return responses[min(len(calls), len(responses)) - 1]
    return transport, calls


def test_meter_health_stop_uses_recent_invalid_schema_rate():
    assert _health_stop_reason([True] * 2 + [False] * 8) is None
    assert _health_stop_reason([True] * 3 + [False] * 7) is not None
    assert _health_stop_reason([True] * 5 + [False] * 15) is None
    assert _health_stop_reason([False] * 9 + [True] * 6 + [False] * 5) is not None


def test_choice_barriers_cannot_enter_formal_stratified_meter():
    with pytest.raises(ValueError, match="diagnostic only"):
        meter_run(n=1, choice_barriers=True, max_requests=1, max_usd=0.01)


def test_response_replay_validates_persisted_body_without_transport():
    status, body = ok()
    result, error = _replay_response({"status": status, "body": body.decode()}, Q, "hash")
    assert error is None
    assert result.answers == {"smoke": 0.5}
    assert result.state_hash == "hash"

    status, body = ok(value=1.2)
    result, error = _replay_response({"status": status, "body": body.decode()}, Q, "hash")
    assert result is None
    assert "out of range" in error


def test_resume_restores_settlement_from_journal_once_and_preserves_hard_cap(tmp_path, monkeypatch):
    run_id = "d" * 16
    wal = tmp_path / "attempts.jsonl"
    responses = tmp_path / "responses.jsonl"
    transport, calls = counting([ok(cost=0.02), ok(cost=0.001)])
    client = JevClient(
        settings(tmp_path), transport=transport,
        on_response=lambda event: _append_checkpoint(responses, {
            **event, "context": {"run_id": run_id, "sample_index": 1},
        }),
    )
    budget = Budget(
        10, 0.03, 0.001,
        on_attempt_event=lambda event: _append_attempt_event(
            wal, tmp_path, 10, 7, client, run_id, event
        ),
    )
    client.budget = budget

    def crash_before_settle(_status, _body, _attempt=None):
        raise RuntimeError("simulated crash before settlement WAL")

    monkeypatch.setattr(budget, "settle", crash_before_settle)
    with pytest.raises(RuntimeError, match="simulated crash"):
        client.decide(
            {"bar": 1}, Q, use_cache=False,
            request_context={"run_id": run_id, "sample_index": 1},
        )
    assert calls == [1]

    attempt_events = _read_attempt_wal(wal, run_id)
    response_event = json.loads(responses.read_text().splitlines()[0])
    settlements = _recover_missing_settlements(
        attempt_events, [response_event], max_requests=10, max_usd=0.03,
        est_cost_per_request=0.001,
    )
    assert len(settlements) == 1
    _append_attempt_event(wal, tmp_path, 10, 7, client, run_id, settlements[0])

    attempt_events = _read_attempt_wal(wal, run_id)
    state, _ = _attempt_wal_state(attempt_events, 10, 0.03)
    assert state["actual_usd"] == pytest.approx(0.02)
    assert state["estimated_usd"] == pytest.approx(0)
    assert state["committed_usd"] == pytest.approx(0.02)
    assert _recover_missing_settlements(
        attempt_events, [response_event], 10, 0.03, 0.001
    ) == []
    state_again, _ = _attempt_wal_state(attempt_events, 10, 0.03)
    assert state_again["actual_usd"] == pytest.approx(0.02)

    resumed_budget = Budget(10, 0.03, 0.001)
    resumed_budget.restore(state_again)
    with pytest.raises(JevBudgetError, match="dollar cap"):
        JevClient(settings(tmp_path), transport=transport, budget=resumed_budget).decide(
            {"bar": 2}, Q, use_cache=False
        )
    assert calls == [1]


def test_historical_schema_failures_block_resume_before_any_transport(tmp_path):
    transport, calls = counting([ok()])
    client = JevClient(settings(tmp_path), transport=transport)
    checkpoint = [
        {"sample_index": index, "status": "failure", "failure": {
            "error": "Jev response TP/SL probabilities exceed 1"
        }, "record": {"response_valid": False}}
        for index in range(1, 16)
    ] + [
        {"sample_index": index, "status": "success", "failure": None,
         "record": {"response_valid": True}}
        for index in range(16, 21)
    ]
    with pytest.raises(JevHealthError, match="historical schema health"):
        _resume_health_outcomes({}, checkpoint, [], Q)
    assert calls == []
    assert _resume_health_outcomes({}, checkpoint, [], Q, allow_override=True) == []


def test_resume_health_uses_completion_order_not_sample_index(tmp_path):
    transport, calls = counting([ok()])
    client = JevClient(settings(tmp_path), transport=transport)
    completed = [
        {"sample_index": index, "status": "success", "failure": None,
         "record": {"response_valid": True}}
        for index in range(100, 130)
    ]
    completed.extend(
        {"sample_index": index, "status": "failure", "failure": {
             "kind": "schema", "error": "Jev response TP/SL probabilities exceed 1"
         }, "record": {"response_valid": False}}
        for index in range(1, 11)
    )

    def resume_one_bar():
        _resume_health_outcomes({}, completed, [], Q)
        client.decide({"bar": "next"}, Q, use_cache=False)

    with pytest.raises(JevHealthError, match="historical schema health"):
        resume_one_bar()
    assert calls == []


def test_resume_health_deduplicates_a_bar_using_its_latest_event(tmp_path):
    entries = [
        {"sample_index": index, "event_time_ns": index,
         "status": "success", "record": {"response_valid": True}}
        for index in range(100, 120)
    ]
    response_events = [{
        "status": 200, "event_time_ns": 121,
        "context": {"sample_index": 100}, "body": "[]",
    }]

    outcomes = _resume_health_outcomes({}, entries, response_events, Q)

    assert len(outcomes) == 20
    assert sum(outcomes) == 1


def test_health_stopped_manifest_is_sticky_until_explicit_override():
    manifest = {"status": "health_stopped", "health": {"stopped_reason": "bad schema"}}
    with pytest.raises(JevHealthError, match="previously health-stopped"):
        _resume_health_outcomes(manifest, [], [], Q)
    assert _resume_health_outcomes(manifest, [], [], Q, allow_override=True) == []


def test_cache_recovery_requires_a_matching_paid_attempt_window(tmp_path):
    now = datetime.now(timezone.utc).timestamp()

    def stamp(value):
        return datetime.fromtimestamp(value, timezone.utc).isoformat().replace("+00:00", "Z")
    path = tmp_path / "request.json"
    path.write_text("{}", encoding="utf-8")
    events = [
        {"event": "reserve", "attempt": 1, "at": stamp(now - 3),
         "context": {"sample_index": 4}},
        {"event": "settle", "attempt": 1, "status": 200, "at": stamp(now - 1)},
    ]
    os.utime(path, (now - 2, now - 2))
    assert _cache_response_for_attempt(path, 4, events) == {}
    assert _cache_response_for_attempt(path, 5, events) is None
    os.utime(path, (now - 10, now - 10))
    assert _cache_response_for_attempt(path, 4, events) is None


def test_request_cap_stops_before_sending(tmp_path):
    transport, calls = counting([ok()])
    budget = Budget(max_requests=3, max_usd=10, est_cost_per_request=0.0003)
    client = JevClient(settings(tmp_path), transport=transport, budget=budget)
    for i in range(3):
        client.decide({"i": i}, Q, use_cache=False)
    with pytest.raises(JevBudgetError, match="request cap"):
        client.decide({"i": 99}, Q, use_cache=False)
    assert len(calls) == 3                      # 第四次未发送 / The 4th never left the machine.


def test_dollar_cap_uses_provider_reported_cost(tmp_path):
    transport, calls = counting([ok(cost=0.004)])
    budget = Budget(max_requests=1000, max_usd=0.010, est_cost_per_request=0.0003)
    client = JevClient(settings(tmp_path), transport=transport, budget=budget)
    client.decide({"i": 0}, Q, use_cache=False)
    client.decide({"i": 1}, Q, use_cache=False)          # 已承诺 $0.008 / $0.008 committed.
    with pytest.raises(JevBudgetError, match="dollar cap"):
        client.decide({"i": 2}, Q, use_cache=False)       # 将超过 $0.010 / Would pass $0.010.
    assert len(calls) == 2
    assert budget.snapshot()["actual_usd"] == pytest.approx(0.008)


def test_unreported_cost_is_charged_at_the_estimate(tmp_path):
    transport, _ = counting([ok(cost=None)])
    budget = Budget(max_requests=10, max_usd=1, est_cost_per_request=0.0005)
    JevClient(settings(tmp_path), transport=transport, budget=budget).decide({}, Q, use_cache=False)
    snap = budget.snapshot()
    assert snap["actual_usd"] == 0 and snap["estimated_usd"] == pytest.approx(0.0005)


def test_402_and_5xx_are_counted_not_charged(tmp_path):
    transport, calls = counting([(402, b'{"error":{"message":"insufficient credits"}}')])
    budget = Budget(max_requests=10, max_usd=1, est_cost_per_request=0.0005)
    with pytest.raises(JevApiError):
        JevClient(settings(tmp_path), transport=transport, budget=budget).decide({}, Q, use_cache=False)
    snap = budget.snapshot()
    assert snap["committed_usd"] == 0 and snap["by_status"] == {"402": 1}
    assert len(calls) == 1                      # 402 不重试 / HTTP 402 is not retried.


def test_rejected_2xx_is_billed_and_not_retried_by_default(tmp_path):
    """旧盲点：无效答案付费后重试再次计费，但本地记录为零成本。

    The old blind spot: an invalid answer was paid for, retried, and paid for
    again, while local records showed zero cost for the bar."""
    transport, calls = counting([ok(cost=0.0003, value=1.7)])   # 越界答案 / Out-of-range answer.
    budget = Budget(max_requests=10, max_usd=1, est_cost_per_request=0.0003)
    with pytest.raises(JevApiError, match="out of range"):
        JevClient(settings(tmp_path), transport=transport, budget=budget).decide({}, Q, use_cache=False)
    snap = budget.snapshot()
    assert len(calls) == 1                      # 无效答案不付费重试 / No paid retry of bad answer.
    assert snap["actual_usd"] == pytest.approx(0.0003)
    assert snap["billed_rejected"] == 1


def test_budget_resume_restores_cumulative_caps_before_transport(tmp_path):
    transport, calls = counting([ok(cost=0.004), ok(cost=0.004)])
    first = Budget(max_requests=1, max_usd=0.004, est_cost_per_request=0.004)
    JevClient(settings(tmp_path), transport=transport, budget=first).decide(
        {"i": 0}, Q, use_cache=False
    )
    snapshot = first.snapshot()

    same_cap = Budget(max_requests=1, max_usd=0.004, est_cost_per_request=0.004)
    same_cap.restore(snapshot)
    with pytest.raises(JevBudgetError, match="request cap"):
        JevClient(settings(tmp_path), transport=transport, budget=same_cap).decide(
            {"i": 1}, Q, use_cache=False
        )
    assert len(calls) == 1

    raised_cap = Budget(max_requests=2, max_usd=0.008, est_cost_per_request=0.004)
    raised_cap.restore(snapshot)
    JevClient(settings(tmp_path), transport=transport, budget=raised_cap).decide(
        {"i": 1}, Q, use_cache=False
    )
    with pytest.raises(JevBudgetError, match="request cap"):
        JevClient(settings(tmp_path), transport=transport, budget=raised_cap).decide(
            {"i": 2}, Q, use_cache=False
        )
    assert len(calls) == 2


def test_meter_wal_consumes_reservations_across_crash_windows(tmp_path):
    run_id = "a" * 16
    wal = tmp_path / "attempts.wal.jsonl"
    transport, calls = counting([ok(cost=0.004)])
    client = JevClient(settings(tmp_path), transport=transport)
    budget = Budget(
        10, 1, 0.0003,
        on_attempt_event=lambda event: _append_attempt_event(
            wal, tmp_path, 1, 7, client, run_id, event
        ),
    )
    client.budget = budget

    # 持久预留后、传输前崩溃。 / Crash after durable reservation, before transport.
    budget.reserve({"run_id": run_id, "sample_index": 1})
    state, attempted = _attempt_wal_state(_read_attempt_wal(wal, run_id), 10, 1)
    assert attempted == {1}
    assert state["attempts"] == 1
    assert state["committed_usd"] == pytest.approx(0.0003)
    assert calls == []

    # 已计费响应结算后、checkpoint 前崩溃。 / Crash after billing settlement, before checkpoint.
    client.decide({"bar": 2}, Q, use_cache=False,
                  request_context={"run_id": run_id, "sample_index": 2})
    state, attempted = _attempt_wal_state(_read_attempt_wal(wal, run_id), 10, 1)
    assert attempted == {1, 2}
    assert state["attempts"] == 2
    assert state["actual_usd"] == pytest.approx(0.004)
    assert state["committed_usd"] == pytest.approx(0.0043)
    assert calls == [1]
    # 恢复时两个预留样本均已耗用，不得重发。 / Both reservations are consumed; no retry.
    assert [i for i in range(1, 5) if i not in attempted] == [3, 4]
    assert calls == [1]


def test_meter_wal_preserves_inflight_charge_in_concurrent_checkpoint(tmp_path):
    base = {
        "attempts": 2, "actual_usd": 0.004, "estimated_usd": 0.0,
        "committed_usd": 0.008, "max_requests": 3, "max_usd": 0.008,
        "billed": 1,
    }
    events = [{
        "event": "reserve", "attempt": 2, "reserve_usd": 0.004,
        "context": {"sample_index": 2},
        "budget": {"attempts": 2, "max_requests": 3, "max_usd": 0.008},
    }]
    state, attempted = _attempt_wal_state(events, 3, 0.008, base)
    assert attempted == {2}
    assert state["attempts"] == 2
    assert state["actual_usd"] == pytest.approx(0.004)
    assert state["estimated_usd"] == pytest.approx(0.0)
    assert state["committed_usd"] == pytest.approx(0.008)

    transport, calls = counting([ok(cost=0.004)])
    budget = Budget(3, 0.008, 0.004)
    budget.restore(state)
    with pytest.raises(JevBudgetError, match="dollar cap"):
        JevClient(settings(tmp_path), transport=transport, budget=budget).decide(
            {"next": True}, Q, use_cache=False
        )
    assert calls == []


def test_meter_wal_applies_settlement_after_checkpoint_to_inflight_attempt():
    events = []
    budget = Budget(4, 0.1, 0.004, on_attempt_event=events.append)
    first = budget.reserve({"sample_index": 1})
    budget.settle(200, ok(0.004)[1], first)
    second = budget.reserve({"sample_index": 2})
    checkpoint = budget.snapshot()  # 第二次已预留仍在途 / Attempt 2 is reserved and inflight.
    budget.settle(200, ok(0.009)[1], second)  # checkpoint 后持久结算 / Durable after checkpoint.

    recovered, attempted = _attempt_wal_state(events, 4, 0.1, checkpoint)
    assert attempted == {1, 2}
    assert recovered["actual_usd"] == pytest.approx(0.013)
    assert recovered["estimated_usd"] == pytest.approx(0.0)
    assert recovered["committed_usd"] == pytest.approx(0.013)
    assert recovered["by_status"] == {"200": 2}


def test_meter_wal_repairs_legacy_checkpoint_inflight_settlement():
    base = {
        "attempts": 2, "actual_usd": 0.004, "estimated_usd": 0.0,
        "committed_usd": 0.008, "max_requests": 3, "max_usd": 0.008,
        "billed": 1, "by_status": {"200": 1},
    }
    events = [
        {"event": "reserve", "attempt": 2, "reserve_usd": 0.004,
         "context": {"sample_index": 2}},
        {"event": "settle", "attempt": 2, "status": 200,
         "actual_usd": 0.009, "estimated_usd": 0.0, "billed": True},
    ]
    recovered, _ = _attempt_wal_state(events, 3, 0.008, base)
    assert recovered["actual_usd"] == pytest.approx(0.013)
    assert recovered["committed_usd"] == pytest.approx(0.013)
    assert recovered["by_status"] == {"200": 2}


def test_budget_serializes_concurrent_settle_ledger_callbacks():
    first_callback_entered = threading.Event()
    second_callback_entered = threading.Event()
    release_first_callback = threading.Event()
    events = []

    def record(event):
        if event["event"] == "settle" and event["attempt"] == 1:
            first_callback_entered.set()
            assert release_first_callback.wait(2)
        if event["event"] == "settle" and event["attempt"] == 2:
            second_callback_entered.set()
        events.append(event)

    budget = Budget(2, 1, 0.004, on_attempt_event=record)
    attempts = [budget.reserve(), budget.reserve()]
    first = threading.Thread(target=budget.settle, args=(200, ok(0.004)[1], attempts[0]))
    second = threading.Thread(target=budget.settle, args=(200, ok(0.004)[1], attempts[1]))
    first.start()
    assert first_callback_entered.wait(2)
    second.start()
    assert not second_callback_entered.wait(0.05)
    release_first_callback.set()
    first.join(2)
    second.join(2)
    assert not first.is_alive() and not second.is_alive()
    settles = [event for event in events if event["event"] == "settle"]
    assert [event["attempt"] for event in settles] == [1, 2]
    assert [event["event_seq"] for event in settles] == sorted(
        event["event_seq"] for event in settles
    )


def test_ledger_only_stopped_run_is_included_in_audit(tmp_path):
    run_id = "c" * 16
    (tmp_path / "jev_spend_ledger.jsonl").write_text(
        json.dumps({
            "n": 2000, "run_id": run_id, "status": "stopped",
            "budget": {"attempts": 17, "actual_usd": 0.012,
                       "estimated_usd": 0.001, "committed_usd": 0.015,
                       "by_status": {"200": 17}},
        }) + "\n",
        encoding="utf-8",
    )
    runs = from_artifacts(tmp_path)
    assert len(runs) == 1
    assert runs.iloc[0]["status"] == "stopped"
    assert runs.iloc[0]["attempts"] == 17
    assert runs.iloc[0]["cost_usd"] == pytest.approx(0.015)
    rec = reconcile(runs, {"responses": 0, "cost_usd": 0.0}, {})
    assert rec["run_attempts"] == 17
    assert rec["run_cost_usd"] == pytest.approx(0.015)
    rendered = _render(runs, "test", {"responses": 0, "cost_usd": 0.0}, {}, rec)
    assert "ledger-only (stopped)" in rendered
    assert run_id in rendered

def test_budget_resume_preserves_inflight_reservation_after_crash(tmp_path):
    budget = Budget(max_requests=3, max_usd=0.008, est_cost_per_request=0.004)
    budget.restore({
        "attempts": 2, "actual_usd": 0.004, "estimated_usd": 0.0,
        "committed_usd": 0.008, "max_requests": 3, "max_usd": 0.008,
        "billed": 2,
    })
    transport, calls = counting([ok(cost=0.004)])
    with pytest.raises(JevBudgetError, match="dollar cap"):
        JevClient(settings(tmp_path), transport=transport, budget=budget).decide(
            {"crashed": True}, Q, use_cache=False
        )
    assert calls == []
    assert budget.snapshot()["estimated_usd"] == pytest.approx(0.004)


def test_opt_in_invalid_retry_is_still_budgeted(tmp_path):
    transport, calls = counting([ok(cost=0.0003, value=1.7), ok(cost=0.0003)])
    budget = Budget(max_requests=10, max_usd=1, est_cost_per_request=0.0003)
    client = JevClient(settings(tmp_path, retry_invalid=True), transport=transport, budget=budget)
    client.decide({}, Q, use_cache=False)
    assert len(calls) == 2
    assert budget.snapshot()["actual_usd"] == pytest.approx(0.0006)


def test_cache_hits_are_free(tmp_path):
    transport, calls = counting([ok()])
    budget = Budget(max_requests=1, max_usd=1, est_cost_per_request=0.0003)
    client = JevClient(settings(tmp_path), transport=transport, budget=budget)
    client.decide({"a": 1}, Q)
    client.decide({"a": 1}, Q)                  # 从缓存返回，上限为 1 / Cache hit with cap 1.
    assert len(calls) == 1


def test_cap_holds_under_concurrency(tmp_path):
    """在途请求按估算预留，防止并行工作线程合计超出美元上限。

    In-flight requests are reserved at the estimate, so parallel workers
    cannot collectively overshoot the dollar cap."""
    gate = threading.Event()

    def slow(url, headers, body):
        gate.wait(2)
        return ok(cost=0.001)

    budget = Budget(max_requests=1000, max_usd=0.005, est_cost_per_request=0.001)
    client = JevClient(settings(tmp_path), transport=slow, budget=budget)
    results = []

    def worker(i):
        try:
            client.decide({"i": i}, Q, use_cache=False)
            results.append("ok")
        except JevBudgetError:
            results.append("stopped")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    gate.set()
    for t in threads:
        t.join()
    assert results.count("ok") == 5
    assert budget.snapshot()["actual_usd"] <= 0.005 + 1e-12


def test_preflight_allows_budget_capped_tranches_and_rejects_empty_allowance():
    assert preflight(1000, 1100, 1.0, 0.000284)["expected_usd"] == pytest.approx(0.284)
    staged = preflight(10_000, 750, 0.30, 0.000284)
    assert staged["request_allowance"] == 750
    assert staged["partial"] is True
    assert staged["expected_usd"] == pytest.approx(0.213)
    full = preflight(10_000, 10_000, 2.84, 0.000284)
    assert full["request_allowance"] == 10_000
    assert full["partial"] is False
    with pytest.raises(JevBudgetError, match="no remaining budget"):
        preflight(10_000, 0, 0.0, 0.000284)


def test_meter_checkpoint_identity_reuses_only_the_same_run(tmp_path):
    grid = {"horizon_bars": 16, "tp": [0.01], "sl": [0.005]}
    client = JevClient(JevSettings("secret", "https://example.test/api/alpha/decisions",
                                  "model-a", prompt_version="v3", cache_dir=tmp_path,
                                  retry_delay_s=0))
    sample_ts = [100, 200, 300]
    states = [{"ts": ts, "features": {"x": float(ts)}} for ts in sample_ts]
    questions = {"q": {"type": "noul", "instructions": "?"}}
    request_ids = [canonical_request_identity(client.settings, state, questions) for state in states]
    identity, run_id = _run_identity(
        3, 7, "BTC/USDT", client, 1, True, grid, sample_ts, sample_ts, request_ids
    )
    path = tmp_path / "manifest.json"
    manifest = _load_or_create_manifest(path, identity, run_id)
    manifest["completed"]["1"] = "success"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    resumed = _load_or_create_manifest(path, identity, run_id)
    assert resumed["completed"] == {"1": "success"}
    checkpoint = tmp_path / "checkpoint.jsonl"
    _append_checkpoint(checkpoint, {"run_id": run_id, "sample_index": 1, "record": {}})
    assert len(_read_checkpoint(checkpoint, run_id)) == 1
    checkpoint.write_text(
        checkpoint.read_text() + json.dumps({"run_id": "b" * 16, "sample_index": 2, "record": {}}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="journal identity mismatch"):
        _read_checkpoint(checkpoint, run_id)

    variants = [
        _run_identity(3, 8, "BTC/USDT", client, 1, True, grid,
                      sample_ts, sample_ts, request_ids),
        _run_identity(3, 7, "BTC/USDT", client, 1, True,
                      {"horizon_bars": 17, "tp": [0.01], "sl": [0.005]},
                      sample_ts, sample_ts, request_ids),
        _run_identity(3, 7, "BTC/USDT",
                      JevClient(JevSettings("secret", "https://example.test/api/alpha/decisions",
                                            "model-b", prompt_version="v4", cache_dir=tmp_path,
                                            retry_delay_s=0)),
                      1, True, grid, sample_ts, sample_ts, request_ids),
    ]
    assert all(other_id != run_id for _, other_id in variants)
    changed_states = [*states]
    changed_states[0] = {"ts": 100, "features": {"x": 999.0}}
    changed_request_ids = [canonical_request_identity(client.settings, state, questions)
                           for state in changed_states]
    _, changed_state_id = _run_identity(
        3, 7, "BTC/USDT", client, 8, False, grid, sample_ts, sample_ts,
        changed_request_ids,
    )
    assert changed_state_id != run_id
    _, changed_candidate_id = _run_identity(
        3, 7, "BTC/USDT", client, 1, True, grid, [100, 200, 301],
        sample_ts, request_ids,
    )
    assert changed_candidate_id != run_id
    _, changed_picked_id = _run_identity(
        3, 7, "BTC/USDT", client, 1, True, grid, sample_ts,
        [100, 200, 301], request_ids,
    )
    assert changed_picked_id != run_id
    _, changed_workers_id = _run_identity(
        3, 7, "BTC/USDT", client, 8, False, grid, sample_ts,
        sample_ts, request_ids,
    )
    assert changed_workers_id == run_id
    with pytest.raises(RuntimeError, match="identity mismatch"):
        _load_or_create_manifest(path, variants[0][0], variants[0][1])


def test_meter_recomputes_outcomes_from_current_labels_without_inference(tmp_path):
    grid = {"horizon_bars": 16, "tp": [0.01], "sl": [0.005]}
    labels = pd.DataFrame({
        level_col(0.01, "up"): [0], level_col(0.005, "dn"): [1],
        level_col(0.01, "dn"): [2], level_col(0.005, "up"): [2],
    }, index=pd.Index([100], name="ts"))
    checkpoint = pd.DataFrame({"sample_index": [1], "ts": [100], "p_long_tp100_sl50": [0.5]})
    first = _attach_current_outcomes(checkpoint, labels, grid)
    labels.loc[100, level_col(0.01, "up")] = 2
    second = _attach_current_outcomes(checkpoint, labels, grid)
    assert first.loc[0, "outcome_long_tp100_sl050"] != second.loc[0, "outcome_long_tp100_sl050"]


def test_error_classification():
    assert classify_error("Jev API request failed (402): insufficient credits") == "http_402"
    assert classify_error("Jev transport failed: timed out") == "transport"
    assert classify_error("Jev response has invalid choice probabilities: regime") == "rejected_2xx"
    assert classify_error("") == "ok"


def test_committed_reports_reconcile(tmp_path):
    (tmp_path / "inference_cost_2000_report.md").write_text(
        "- **Logical Jev requests:** 2000\n- **Successful responses:** 1953\n"
        "- **Failed responses:** 47\n- **Actual HTTP attempts:** 2006\n"
        "| Input tokens | 13,195,562 |\n| Output tokens | 4,085,677 |\n"
        "| Reported API cost | $0.554214 |\n", encoding="utf-8")
    runs = from_reports(tmp_path)
    rec = reconcile(runs, {"responses": 0, "cost_usd": 0.0}, {})
    assert rec["run_successes"] == 1953
    assert rec["usd_per_success"] == pytest.approx(0.554214 / 1953)
    rec = reconcile(runs, {"responses": 0, "cost_usd": 0.0}, {"key_usage": 2.0})
    assert rec["unrecorded_usd"] == pytest.approx(2.0 - 0.554214)
    rec = reconcile(runs, {"responses": 0, "cost_usd": 0.0, "ledger_usd": 0.8},
                    {"key_usage": 2.0})
    assert rec["unrecorded_usd"] == pytest.approx(1.2)


def test_artifact_audit_includes_failure_artifact_and_report_attempts(tmp_path):
    pd.DataFrame({"ts": [1], "input_tokens": [10], "output_tokens": [20],
                  "cost": [0.3]}).to_parquet(tmp_path / "metering_2.parquet")
    (tmp_path / "metering_2_failures.json").write_text(
        json.dumps([{"error": "Jev API request failed (402): insufficient credits"}]),
        encoding="utf-8",
    )
    reports = from_reports(tmp_path)
    # 失败数无需报告；有报告时优先取其尝试数。 / Report is optional; its attempt count wins if present.
    artifacts = from_artifacts(tmp_path, reports)
    assert artifacts.iloc[0]["successes"] == 1
    assert artifacts.iloc[0]["failures"] == 1
    assert artifacts.iloc[0]["fail_http_402"] == 1


def test_artifact_audit_keeps_same_n_run_ids_separate(tmp_path):
    runs = {"1111111111111111": (3, "Q1", 1), "2222222222222222": (7, "Q2", 2)}
    for run_id, (attempts, quarter, failures) in runs.items():
        stem = f"metering_2_{run_id}"
        pd.DataFrame({
            "ts": [1, 2], "quarter": [quarter, quarter],
            "response_valid": [True, False], "input_tokens": [1, 1],
            "output_tokens": [2, 2], "cost": [0.1, 0.0],
            "request_error": ["", "Jev API request failed (402): no credit"],
        }).to_parquet(tmp_path / f"{stem}.parquet")
        (tmp_path / f"{stem}_manifest.json").write_text(
            json.dumps({"budget": {"attempts": attempts}}), encoding="utf-8"
        )
        pd.DataFrame({"ts": [1, 2], "quarter": [quarter, quarter]}).to_csv(
            tmp_path / f"{stem}_attempted.csv", index=False
        )
        (tmp_path / "jev_spend_ledger.jsonl").open("a").write(
            json.dumps({"n": 2, "run_id": run_id, "budget": {"attempts": attempts}}) + "\n"
        )

    artifacts = from_artifacts(tmp_path)
    by_id = artifacts.set_index("run_id")
    assert set(by_id.index) == set(runs)
    assert by_id.loc["1111111111111111", "attempts"] == 3
    assert by_id.loc["2222222222222222", "attempts"] == 7
    assert by_id.loc["1111111111111111", "failures"] == 1
    assert by_id.loc["2222222222222222", "failures"] == 1
    rendered = _render(
        artifacts, "test", {"responses": 0, "cost_usd": 0.0},
        {}, reconcile(artifacts, {"responses": 0, "cost_usd": 0.0}, {}),
        {"configured": False}, quarter_audit(tmp_path),
    )
    assert "1111111111111111" in rendered and "2222222222222222" in rendered
    assert "2:1111111111111111" in rendered and "2:2222222222222222" in rendered

    quarters = quarter_audit(tmp_path)
    assert quarters["2:1111111111111111"]["attempted"] == {"Q1": 2}
    assert quarters["2:2222222222222222"]["attempted"] == {"Q2": 2}


def test_artifact_audit_uses_ledger_cost_for_rejected_2xx(tmp_path):
    pd.DataFrame({
        "ts": [1], "response_valid": [False], "input_tokens": [10],
        "output_tokens": [20], "cost": [float("nan")],
        "request_error": ["Jev response has invalid probability"],
    }).to_parquet(tmp_path / "metering_1.parquet")
    (tmp_path / "jev_spend_ledger.jsonl").write_text(
        json.dumps({
            "n": 1, "run_id": "legacy", "budget": {
                "attempts": 1, "actual_usd": 0.0003, "estimated_usd": 0.0,
                "billed_rejected": 1,
            },
        }) + "\n",
        encoding="utf-8",
    )
    artifacts = from_artifacts(tmp_path)
    assert artifacts.iloc[0]["cost_usd"] == pytest.approx(0.0003)
    rec = reconcile(artifacts, {"responses": 0, "cost_usd": 0.0}, {})
    assert rec["run_cost_usd"] == pytest.approx(0.0003)
    assert rec["run_successes"] == 0


def test_offline_audit_checks_configured_key_without_provider_query(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    (tmp_path / "docs").mkdir()

    def forbidden_query(*_args, **_kwargs):
        raise AssertionError("offline audit must not query provider")

    monkeypatch.setattr("jev_trader.audit.from_provider", forbidden_query)
    text, summary = audit_run(
        tmp_path, "local-secret", "https://example.test/api/decisions",
        query_provider=False,
    )

    assert summary["security"]["configured"] is True
    assert summary["security"]["exposed_files"] == []
    assert "Current key configured: **True**" in text
    assert "Not queried: offline audit: provider not queried" in text
    assert "local-secret" not in text
