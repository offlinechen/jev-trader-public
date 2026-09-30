from __future__ import annotations

import json

import pytest
import jev_trader.jev as jev_module

from jev_trader.jev import (
    Budget, JevApiError, JevJournalError, JevSchemaError,
    JevClient,
    JevConfigError,
    JevSettings,
    build_questions,
    canonical_request_payload,
    canonical_request_identity,
    prompt_contract_v4,
)
from jev_trader.features import OHLCV_FEATURES_V1


GRID = {
    "horizon_bars": 16,
    "tp": [0.005, 0.010, 0.015, 0.020, 0.030],
    "sl": [0.005, 0.0075, 0.010, 0.015],
}


def settings(tmp_path):
    return JevSettings("secret", "https://example.test/api/alpha/decisions", "jev-test", cache_dir=tmp_path)


def response(questions):
    return {
        "model": "jev-test-served",
        "answers": {key: {"type": "noul", "noul": 0.5} for key in questions},
        "usage": {"input_tokens": 10, "output_tokens": 20, "cost": 0.001},
    }


def test_build_questions_matches_prediction_contract():
    questions = build_questions(GRID)
    assert len(questions) == 82
    assert questions["regime"]["type"] == "choice"
    assert questions["volatility"]["type"] == "choice"
    assert "p_long_tp050_sl075" in questions
    assert "p_short_tp300_sl150" in questions
    assert sum(question["type"] == "noul" for question in questions.values()) == 80


def test_request_shape_response_validation_and_cache(tmp_path):
    calls = []

    def transport(url, headers, body):
        calls.append((url, headers, json.loads(body)))
        return 200, json.dumps(response(calls[-1][2]["questions"])).encode()

    client = JevClient(settings(tmp_path), transport=transport)
    questions = {"smoke": {"type": "noul", "instructions": "Is this a smoke test?"}}
    first = client.decide({"b": 2, "a": 1}, questions)
    second = client.decide({"a": 1, "b": 2}, questions)

    assert len(calls) == 1
    assert calls[0][0].endswith("/decisions")
    assert calls[0][1]["Authorization"] == "Bearer secret"
    assert calls[0][2]["model"] == "jev-test"
    assert calls[0][2]["state"]["market"] == {"a": 1, "b": 2}
    assert calls[0][2]["state"]["_jev_protocol"]["prompt_version"] == "v3"
    assert calls[0][2]["questions"] == questions
    assert first.answers == {"smoke": 0.5}
    assert second.cached is True
    assert second.usage["input_tokens"] == 10


def test_cache_identity_binds_endpoint_questions_and_injected_contract(tmp_path):
    calls = []

    def transport(url, headers, body):
        calls.append((url, json.loads(body)))
        return 200, json.dumps(response(calls[-1][1]["questions"])).encode()

    state = {"x": 1, "grid": {"horizon_bars": 16}}
    questions = {"smoke": {"type": "noul", "instructions": "A?"}}
    client = JevClient(settings(tmp_path), transport=transport)
    client.decide(state, questions)
    client.decide(state, questions)
    assert len(calls) == 1

    changed_questions = {"smoke": {"type": "noul", "instructions": "B?"}}
    JevClient(settings(tmp_path), transport=transport).decide(state, changed_questions)
    JevClient(settings(tmp_path), transport=transport).decide(
        {"x": 1, "grid": {"horizon_bars": 17}}, questions
    )
    prompt4 = JevSettings(
        "secret", "https://example.test/api/alpha/decisions", "jev-test",
        prompt_version="v4", cache_dir=tmp_path,
    )
    JevClient(prompt4, transport=transport).decide(state, questions)
    endpoint2 = JevSettings(
        "secret", "https://other.example.test/decisions", "jev-test",
        cache_dir=tmp_path,
    )
    JevClient(endpoint2, transport=transport).decide(state, questions)
    assert len(calls) == 5


def test_cache_namespace_prevents_cross_provenance_replay(tmp_path):
    calls = []

    def transport(url, headers, body):
        calls.append(json.loads(body))
        return 200, json.dumps(response(calls[-1]["questions"])).encode()

    client = JevClient(settings(tmp_path), transport=transport)
    state = {"pair": "BTC/USDT:USDT", "ts": 123}
    questions = {"smoke": {"type": "noul"}}
    client.decide(state, questions, cache_namespace="synthetic_fixture")
    client.decide(state, questions, cache_namespace="live_market")
    client.decide(state, questions, cache_namespace="synthetic_fixture")

    assert len(calls) == 2


def test_ohlcv_v4_prompt_and_identity_bind_common_state_and_model(tmp_path):
    state = {
        "exchange": "okx", "market_type": "usdt-perpetual",
        "pair": "BTC/USDT:USDT", "symbol": "BTC/USDT:USDT",
        "feature_version": "ohlcv-features-v1",
        "state_version": "jev-state-v4",
        "features": {key: 0.01 for key in OHLCV_FEATURES_V1},
        "grid": {"horizon_bars": 16},
    }
    questions = {"smoke": {"type": "noul", "instructions": "TP first?"}}
    settings_v4 = JevSettings(
        "secret", "https://example.test/api/alpha/decisions", "fixture/jev-v4",
        prompt_version="jev-ohlcv-v4", cache_dir=tmp_path,
    )
    payload = canonical_request_payload(settings_v4, state, questions)

    assert payload["model"] == "fixture/jev-v4"
    assert payload["state"]["_jev_protocol"] == {
        "prompt_version": "jev-ohlcv-v4",
        "feature_version": "ohlcv-features-v1",
        "state_version": "jev-state-v4",
        "contract": prompt_contract_v4(16),
    }
    assert "taker_buy_volume" not in json.dumps(payload)
    assert canonical_request_identity(settings_v4, state, questions) != (
        canonical_request_identity(settings(tmp_path), {
            "exchange": "binance", "symbol": "BTC/USDT:USDT",
            "features": {"f_buy_sell_ratio": 1.0},
            "grid": {"horizon_bars": 16},
        }, questions)
    )


def test_ohlcv_v4_prompt_rejects_v3_or_unversioned_states():
    settings_v4 = JevSettings(
        "secret", "https://example.test/api/alpha/decisions", "fixture/jev-v4",
        prompt_version="jev-ohlcv-v4",
    )
    questions = {"smoke": {"type": "noul"}}
    with pytest.raises(JevConfigError, match="exact OHLCV feature contract"):
        canonical_request_payload(settings_v4, {"grid": {"horizon_bars": 16}}, questions)


def test_ohlcv_v4_rejects_an_unregistered_or_taker_flow_feature():
    settings_v4 = JevSettings(
        "secret", "https://example.test/api/alpha/decisions", "fixture/jev-v4",
        prompt_version="jev-ohlcv-v4",
    )
    state = {
        "exchange": "okx", "market_type": "usdt-perpetual",
        "pair": "BTC/USDT:USDT", "feature_version": "ohlcv-features-v1",
        "state_version": "jev-state-v4", "features": {
            **{key: 0.0 for key in OHLCV_FEATURES_V1},
            "f_buy_sell_ratio": 1.0,
        }, "grid": {"horizon_bars": 16},
    }
    with pytest.raises(JevConfigError, match="exact OHLCV feature contract"):
        canonical_request_payload(settings_v4, state, {"smoke": {"type": "noul"}})


def test_cache_identity_binds_injected_contract_text(tmp_path, monkeypatch):
    calls = []

    def transport(url, headers, body):
        calls.append(json.loads(body))
        return 200, json.dumps(response(calls[-1]["questions"])).encode()

    state = {"x": 1, "grid": {"horizon_bars": 16}}
    questions = {"smoke": {"type": "noul", "instructions": "A?"}}
    monkeypatch.setattr(jev_module, "prompt_contract", lambda _horizon: "contract-A")
    client = JevClient(settings(tmp_path), transport=transport)
    client.decide(state, questions)
    client.decide(state, questions)
    monkeypatch.setattr(jev_module, "prompt_contract", lambda _horizon: "contract-B")
    JevClient(settings(tmp_path), transport=transport).decide(state, questions)

    assert len(calls) == 2
    assert calls[0]["state"]["_jev_protocol"]["contract"] == "contract-A"
    assert calls[1]["state"]["_jev_protocol"]["contract"] == "contract-B"


def test_retries_transient_http_once_without_printing_key(tmp_path):
    statuses = iter([(429, b'{"error":{"message":"rate limited"}}'), (200, b"")])
    sleeps = []
    questions = {"smoke": {"type": "noul", "instructions": "Is this a smoke test?"}}

    def transport(_url, _headers, _body):
        status, body = next(statuses)
        if status == 200:
            return status, json.dumps(response(questions)).encode()
        return status, body

    result = JevClient(settings(tmp_path), transport=transport, sleep=sleeps.append).decide(
        {"x": 1}, questions
    )
    assert result.answers["smoke"] == 0.5
    assert sleeps == [0.5]


def test_rejects_missing_answer_and_bad_probability(tmp_path):
    questions = {"a": {"type": "noul", "instructions": "A?"}, "b": {"type": "noul", "instructions": "B?"}}

    def transport(_url, _headers, _body):
        raw = response(questions)
        del raw["answers"]["b"]
        raw["answers"]["a"]["noul"] = 2
        return 200, json.dumps(raw).encode()

    with pytest.raises(JevApiError):
        JevClient(settings(tmp_path), transport=transport).decide({"x": 1}, questions, use_cache=False)


def test_non_object_success_body_is_a_schema_failure(tmp_path):
    def transport(_url, _headers, _body):
        return 200, b"[]"

    with pytest.raises(JevSchemaError, match="non-object"):
        JevClient(settings(tmp_path), transport=transport).decide(
            {"x": 1}, {"smoke": {"type": "noul"}}, use_cache=False
        )


def test_retries_schema_failure_before_accepting_response(tmp_path):
    """无效响应的付费重试必须主动启用，默认关闭，避免重复计费。

    Opt-in only: a paid retry of an invalid answer is off by default (see
    tests/test_budget.py) because it bills twice for a usually-identical answer."""
    questions = {
        "p_long_tp050_sl050": {"type": "noul", "instructions": "TP?"},
        "p_long_sl050_tp050": {"type": "noul", "instructions": "SL?"},
    }
    calls = 0
    sleeps = []

    def transport(_url, _headers, _body):
        nonlocal calls
        calls += 1
        raw = response(questions)
        if calls == 1:
            raw["answers"]["p_long_tp050_sl050"]["noul"] = 0.8
            raw["answers"]["p_long_sl050_tp050"]["noul"] = 0.4
        return 200, json.dumps(raw).encode()

    opted_in = JevSettings(
        "secret", "https://example.test/api/alpha/decisions", "jev-test",
        cache_dir=tmp_path, retry_invalid=True,
    )
    result = JevClient(opted_in, transport=transport, sleep=sleeps.append).decide(
        {"x": 1}, questions, use_cache=False
    )
    assert result.answers["p_long_tp050_sl050"] == 0.5
    assert calls == 2
    assert sleeps == [0.5]


def test_rejects_tp_sl_sum_after_schema_retry(tmp_path):
    questions = {
        "p_long_tp050_sl050": {"type": "noul", "instructions": "TP?"},
        "p_long_sl050_tp050": {"type": "noul", "instructions": "SL?"},
    }

    def transport(_url, _headers, _body):
        raw = response(questions)
        raw["answers"]["p_long_tp050_sl050"]["noul"] = 0.8
        raw["answers"]["p_long_sl050_tp050"]["noul"] = 0.4
        return 200, json.dumps(raw).encode()

    with pytest.raises(JevApiError, match="exceed 1"):
        JevClient(settings(tmp_path), transport=transport, sleep=lambda _: None).decide(
            {"x": 1}, questions, use_cache=False
        )


def test_journals_billed_raw_response_before_schema_rejection(tmp_path):
    questions = {
        "p_long_tp050_sl050": {"type": "noul", "instructions": "TP?"},
        "p_long_sl050_tp050": {"type": "noul", "instructions": "SL?"},
    }
    raw = response(questions)
    raw["answers"]["p_long_tp050_sl050"]["noul"] = 0.8
    raw["answers"]["p_long_sl050_tp050"]["noul"] = 0.4
    events = []
    sequence = []

    def transport(_url, _headers, _body):
        return 200, json.dumps(raw).encode()

    budget = Budget(1, 1, 0.001, on_attempt_event=lambda event: sequence.append(event["event"]))
    client = JevClient(
        settings(tmp_path), transport=transport, budget=budget,
        on_response=lambda event: (events.append(event), sequence.append("response")),
    )
    with pytest.raises(JevApiError, match="exceed 1"):
        client.decide(
            {"x": 1}, questions, use_cache=False,
            request_context={"run_id": "run-1", "sample_index": 7},
        )

    assert len(events) == 1
    assert events[0]["status"] == 200
    assert events[0]["context"] == {"run_id": "run-1", "sample_index": 7}
    assert json.loads(events[0]["body"]) == raw
    assert sequence == ["reserve", "response", "settle"]
    assert budget.snapshot()["actual_usd"] == pytest.approx(0.001)


def test_response_journal_failure_is_fatal_and_not_retried(tmp_path):
    calls = []

    def transport(_url, _headers, body):
        calls.append(1)
        return 200, json.dumps(response(json.loads(body)["questions"])).encode()

    def broken_journal(_event):
        raise OSError("disk full")

    client = JevClient(settings(tmp_path), transport=transport, on_response=broken_journal)
    with pytest.raises(JevJournalError, match="could not persist"):
        client.decide({"x": 1}, {"smoke": {"type": "noul"}}, use_cache=False)
    assert calls == [1]


def test_env_requires_all_runtime_values(tmp_path):
    with pytest.raises(JevConfigError, match="JEV_API_KEY"):
        JevSettings.from_env({"JEV_BASE_URL": "https://example.test", "JEV_MODEL_ID": "jev"}, tmp_path / ".env")


def test_dotenv_is_loaded_without_overriding_process_env(tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        'JEV_API_KEY="file-key"\nJEV_BASE_URL=https://file.test/decisions\nJEV_MODEL_ID=file-model\n',
        encoding="utf-8",
    )
    settings = JevSettings.from_env({}, dotenv)
    assert settings.api_key == "file-key"
    assert settings.base_url == "https://file.test/decisions"
    assert settings.model_id == "file-model"
