"""Small, testable client for OpenRouter's Jev Decisions API."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shlex
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

from .features import OHLCV_FEATURES_V1
from .grid import sl_key, tp_key

DEFAULT_URL = "https://openrouter.ai/api/alpha/decisions"
DEFAULT_PROMPT_VERSION = "v3"
PROMPT_VERSION_OHLCV_V4 = "jev-ohlcv-v4"


def prompt_contract(horizon_bars: int) -> str:
    return (
    "Use only information available through the t 15m candle close. "
    "The signal is produced at t close and the simulated trade enters at the "
    f"next 15m candle open. For every barrier question, estimate whether TP is "
    f"touched before SL during the next {horizon_bars} 15m candles after that entry."
    )


def prompt_contract_v4(horizon_bars: int) -> str:
    return (
        "Use only fields in the supplied state. It uses the frozen common "
        "15m OHLCV feature contract ohlcv-features-v1 and state contract "
        "jev-state-v4; it contains no taker-buy or trade-side-flow feature. "
        "Use only information available through the t 15m candle close. "
        "The signal is produced at t close and the simulated trade enters at "
        "the next 15m candle open. For every barrier question, estimate "
        "whether TP is touched before SL during the next "
        f"{horizon_bars} 15m candles after that entry."
    )


PROMPT_CONTRACT_V3 = prompt_contract(16)
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
Transport = Callable[[str, dict[str, str], bytes], tuple[int, bytes]]


class JevError(RuntimeError):
    """Base error for configuration, transport, and response failures."""


class JevConfigError(JevError):
    pass


class JevApiError(JevError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class JevSchemaError(JevApiError):
    """A successful HTTP response whose content violates the Decisions schema."""


class JevBudgetError(JevError):
    """The local spend guard tripped. Never retried; the run must stop."""


class JevHealthError(JevError):
    """The local response-quality guard stopped a metering tranche."""


class JevJournalError(JevError):
    """A response could not be durably recorded before validation."""


class Budget:
    """Hard client-side spend guard, checked before every HTTP attempt.

    The provider running out of credit is not a stop mechanism -- it arrives
    mid-run, after the money is gone, and leaves a non-random hole in the
    sample. This ledger refuses the *next* request locally once either cap
    would be crossed.

    Accounting is deliberately conservative:
    * a 2xx response is charged at its provider-reported `usage.cost`, even
      when the answer is then rejected by local validation -- the inference
      ran and was billed whether or not we keep it;
    * a 2xx without a reported cost, and a transport failure (timeout, reset),
      are charged at the estimate, because they may have been billed;
    * non-2xx statuses (402, 429, 5xx, 400) are counted but not charged.
    In-flight requests are held at the estimate until they settle, so the cap
    also holds under concurrent workers.
    """

    def __init__(self, max_requests: int, max_usd: float, est_cost_per_request: float,
                 on_attempt_event: Callable[[dict[str, Any]], None] | None = None):
        if max_requests <= 0 or max_usd <= 0 or est_cost_per_request <= 0:
            raise JevConfigError("budget limits and estimate must all be positive")
        self.max_requests = int(max_requests)
        self.max_usd = float(max_usd)
        self.est = float(est_cost_per_request)
        self.on_attempt_event = on_attempt_event
        self._lock = threading.Lock()
        self.attempts = 0
        self.event_seq = 0
        self.inflight = 0
        self.actual_usd = 0.0        # provider-reported cost of billed responses
        self.estimated_usd = 0.0     # charged at the estimate: cost unreported
        self.billed_rejected = 0     # 2xx responses billed but rejected locally
        self.billed = 0              # responses with a provider-reported cost
        self.by_status: dict[str, int] = {}

    @property
    def unit_usd(self) -> float:
        """Cost to assume for the next request: the configured estimate, or the
        running mean of what the provider has actually billed, whichever is
        higher -- so a cheap estimate cannot let real spend overshoot the cap."""
        observed = self.actual_usd / self.billed if self.billed else 0.0
        return max(self.est, observed)

    @property
    def committed_usd(self) -> float:
        return self.actual_usd + self.estimated_usd + self.inflight * self.unit_usd

    def reserve(self, context: Mapping[str, Any] | None = None) -> int:
        with self._lock:
            if self.attempts >= self.max_requests:
                raise JevBudgetError(
                    f"request cap reached: {self.attempts}/{self.max_requests} attempts"
                )
            if self.committed_usd + self.unit_usd > self.max_usd:
                raise JevBudgetError(
                    f"dollar cap reached: ${self.committed_usd:.6f} committed, "
                    f"next request would exceed ${self.max_usd:.6f}"
                )
            attempt = self.attempts + 1
            reserve_usd = self.unit_usd
            event_seq = self.event_seq + 1
            self.event_seq = event_seq
            self.attempts = attempt
            self.inflight += 1
            if self.on_attempt_event:
                self.on_attempt_event({
                    "event": "reserve", "attempt": attempt, "event_seq": event_seq,
                    "context": dict(context or {}), "reserve_usd": reserve_usd,
                    "budget": {
                        "attempts": attempt, "event_seq": event_seq,
                        "actual_usd": round(self.actual_usd, 6),
                        "estimated_usd": round(self.estimated_usd, 6),
                        "committed_usd": round(self.committed_usd, 6),
                        "max_requests": self.max_requests, "max_usd": self.max_usd,
                        "est_cost_per_request": self.est, "billed": self.billed,
                        "billed_rejected": self.billed_rejected,
                        "by_status": dict(self.by_status),
                    },
                })
            return attempt

    def settle(self, status: int | None, body: bytes | None, attempt: int | None = None) -> None:
        cost = None
        if status is not None and 200 <= status < 300 and body:
            try:
                reported = json.loads(body).get("usage", {}).get("cost")
                if isinstance(reported, (int, float)) and not isinstance(reported, bool):
                    cost = float(reported)
            except (ValueError, AttributeError):
                pass
        with self._lock:
            self.inflight -= 1
            key = "transport" if status is None else str(status)
            self.by_status[key] = self.by_status.get(key, 0) + 1
            estimated_charge = self.unit_usd if cost is None and (status is None or 200 <= status < 300) else 0.0
            if cost is not None:
                self.actual_usd += cost
                self.billed += 1
            elif estimated_charge:
                self.estimated_usd += estimated_charge
            if self.on_attempt_event:
                # Keep durable cumulative snapshots ordered across concurrent settles.
                event_seq = self.event_seq + 1
                self.event_seq = event_seq
                self.on_attempt_event({
                    "event": "settle", "attempt": attempt, "event_seq": event_seq,
                    "status": status,
                    "actual_usd": cost if cost is not None else 0.0,
                    "estimated_usd": estimated_charge,
                    "billed": cost is not None,
                    "budget": {
                        "attempts": self.attempts, "event_seq": self.event_seq,
                        "max_requests": self.max_requests,
                        "actual_usd": round(self.actual_usd, 6),
                        "estimated_usd": round(self.estimated_usd, 6),
                        "committed_usd": round(self.committed_usd, 6),
                        "max_usd": self.max_usd,
                        "est_cost_per_request": self.est,
                        "observed_cost_per_billed": round(self.actual_usd / self.billed, 8) if self.billed else None,
                        "billed": self.billed, "billed_rejected": self.billed_rejected,
                        "by_status": dict(self.by_status),
                    },
                })

    def mark_rejected(self) -> None:
        with self._lock:
            self.billed_rejected += 1

    def restore(self, snapshot: Mapping[str, Any] | None) -> None:
        """Resume one run's cumulative spend ledger under the current caps."""
        if not snapshot:
            return
        old_requests = int(snapshot.get("max_requests", self.max_requests))
        old_usd = float(snapshot.get("max_usd", self.max_usd))
        if self.max_requests < old_requests or self.max_usd < old_usd:
            raise JevBudgetError(
                "resume caps cannot be lower than the previous total caps; "
                "raise them explicitly to continue"
            )
        with self._lock:
            self.attempts = int(snapshot.get("attempts", 0))
            self.event_seq = int(snapshot.get("event_seq", 0))
            self.actual_usd = float(snapshot.get("actual_usd", 0.0))
            self.estimated_usd = float(snapshot.get("estimated_usd", 0.0))
            committed = float(snapshot.get(
                "committed_usd", self.actual_usd + self.estimated_usd
            ))
            # A crash can persist an in-flight reservation without its settle
            # event. Treat that difference as uncertain spend conservatively.
            self.estimated_usd += max(
                0.0, committed - self.actual_usd - self.estimated_usd
            )
            self.billed_rejected = int(snapshot.get("billed_rejected", 0))
            self.billed = int(snapshot.get("billed", 0))
            self.by_status = dict(snapshot.get("by_status", {}))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "attempts": self.attempts,
                "event_seq": self.event_seq,
                "max_requests": self.max_requests,
                "actual_usd": round(self.actual_usd, 6),
                "estimated_usd": round(self.estimated_usd, 6),
                "committed_usd": round(self.committed_usd, 6),
                "max_usd": self.max_usd,
                "est_cost_per_request": self.est,
                "observed_cost_per_billed": round(self.actual_usd / self.billed, 8) if self.billed else None,
                "billed": self.billed,
                "billed_rejected": self.billed_rejected,
                "by_status": dict(self.by_status),
            }


@dataclass(frozen=True)
class JevSettings:
    api_key: str
    base_url: str
    model_id: str
    prompt_version: str = DEFAULT_PROMPT_VERSION
    cache_dir: Path = Path("data/jev_cache")
    timeout_s: float = 60.0
    retry_delay_s: float = 0.5
    # Re-asking after a 2xx answer fails validation pays twice for what is
    # usually the same invalid answer, and the first bill was previously
    # invisible to local accounting. Off unless a run explicitly opts in.
    retry_invalid: bool = False

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        dotenv_path: str | Path | None = None,
        *,
        model_id: str | None = None,
        prompt_version: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> "JevSettings":
        values = dict(os.environ if env is None else env)
        path = None
        if dotenv_path:
            path = Path(dotenv_path)
        elif env is None:
            path = Path(__file__).resolve().parents[2] / ".env"
        if path:
            for key, value in _read_dotenv(path).items():
                values.setdefault(key, value)

        required = {key: values.get(key, "").strip() for key in ("JEV_API_KEY", "JEV_BASE_URL")}
        resolved_model = (model_id or values.get("JEV_MODEL_ID", "")).strip()
        if not resolved_model:
            required["JEV_MODEL_ID"] = ""
        missing = [key for key, value in required.items() if not value or value.startswith("<")]
        if missing:
            raise JevConfigError(f"missing Jev configuration: {', '.join(missing)}")
        parsed = urlsplit(required["JEV_BASE_URL"])
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise JevConfigError("JEV_BASE_URL must be an http(s) URL")

        try:
            timeout = float(values.get("JEV_TIMEOUT_S", "60"))
            retry_delay = float(values.get("JEV_RETRY_DELAY_S", "0.5"))
        except ValueError as exc:
            raise JevConfigError("JEV_TIMEOUT_S and JEV_RETRY_DELAY_S must be numbers") from exc
        if timeout <= 0 or retry_delay < 0:
            raise JevConfigError("JEV_TIMEOUT_S must be > 0 and JEV_RETRY_DELAY_S must be >= 0")

        return cls(
            api_key=required["JEV_API_KEY"],
            base_url=required["JEV_BASE_URL"].rstrip("/"),
            model_id=resolved_model,
            prompt_version=prompt_version or values.get("JEV_PROMPT_VERSION", DEFAULT_PROMPT_VERSION),
            cache_dir=Path(cache_dir or values.get("JEV_CACHE_DIR", "data/jev_cache")),
            timeout_s=timeout,
            retry_delay_s=retry_delay,
        )

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], env: Mapping[str, str] | None = None) -> "JevSettings":
        jev = cfg["jev"]
        return cls.from_env(
            env,
            model_id=jev["model_id"],
            prompt_version=jev["prompt_version"],
            cache_dir=jev["cache_dir"],
        )


@dataclass(frozen=True)
class JevResult:
    answers: dict[str, float]
    model: str
    usage: dict[str, int | float]
    latency_ms: float
    state_hash: str
    cached: bool
    raw: dict[str, Any]


class JevClient:
    """One-call Jev client with one retry and an idempotent disk cache."""

    def __init__(
        self,
        settings: JevSettings,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        budget: Budget | None = None,
        on_response: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.settings = settings
        self._transport = transport
        self._sleep = sleep
        self.budget = budget
        self.on_response = on_response
        self.request_attempts = 0
        self._attempts_lock = threading.Lock()

    def decide(
        self,
        state: Any,
        questions: Mapping[str, Mapping[str, Any]],
        *,
        use_cache: bool = True,
        cache_namespace: str | None = None,
        request_context: Mapping[str, Any] | None = None,
    ) -> JevResult:
        if not questions:
            raise JevError("questions must not be empty")
        request_identity = canonical_request_identity(self.settings, state, questions)
        payload = request_identity["payload"]
        cache_key = canonical_hash(request_identity if cache_namespace is None else {
            "request_identity": request_identity,
            "cache_namespace": cache_namespace,
        })
        cache_path = self.settings.cache_dir / f"{cache_key}.json"
        if use_cache and cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                return _result(cached, questions, cache_key, cached=True)
            except (OSError, ValueError, KeyError, TypeError, JevError):
                pass

        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        started = time.perf_counter()
        response = self._request(body, headers, questions, request_context)
        latency_ms = (time.perf_counter() - started) * 1000
        result = _result(response, questions, cache_key, latency_ms=latency_ms, cached=False)
        if use_cache:
            self._write_cache(cache_path, response)
        return result

    def _request(
        self,
        body: bytes,
        headers: dict[str, str],
        questions: Mapping[str, Mapping[str, Any]],
        request_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        last_status = None
        last_body = b""
        for attempt in range(2):
            status, response_body = self._send(body, headers, request_context)
            if 200 <= status < 300:
                try:
                    decoded = json.loads(response_body)
                    if not isinstance(decoded, dict):
                        raise JevSchemaError("Jev returned a non-object JSON response", status)
                    # Validate before accepting the response. A schema failure is
                    # retryable just like a transient HTTP failure.
                    _result(decoded, questions, "response-validation", cached=True)
                    return decoded
                except (json.JSONDecodeError, JevApiError) as exc:
                    if self.budget:
                        self.budget.mark_rejected()
                    if attempt == 0 and self.settings.retry_invalid:
                        self._sleep(self.settings.retry_delay_s)
                        continue
                    if isinstance(exc, JevApiError):
                        raise
                    raise JevSchemaError("Jev returned non-JSON success response", status) from exc
            last_status, last_body = status, response_body
            if status not in RETRYABLE_STATUS or attempt:
                break
            self._sleep(self.settings.retry_delay_s)
        detail = _error_detail(last_body, self.settings.api_key)
        raise JevApiError(f"Jev API request failed ({last_status}): {detail}", last_status)

    def _send(self, body: bytes, headers: dict[str, str],
              request_context: Mapping[str, Any] | None = None) -> tuple[int, bytes]:
        attempt = None
        if self.budget:
            attempt = self.budget.reserve(request_context)  # durable before transport
        with self._attempts_lock:
            self.request_attempts += 1
        status: int | None = None
        response_body: bytes | None = None
        try:
            status, response_body = self._send_raw(body, headers)
            if status is not None and 200 <= status < 300 and self.on_response:
                try:
                    self.on_response({
                        "status": status,
                        "attempt": attempt,
                        "event_time_ns": time.time_ns(),
                        "context": dict(request_context or {}),
                        "body": (response_body or b"").decode("utf-8", errors="replace"),
                    })
                except Exception as exc:
                    raise JevJournalError("could not persist successful Jev response") from exc
            return status, response_body
        finally:
            if self.budget:
                self.budget.settle(status, response_body, attempt)

    def _send_raw(self, body: bytes, headers: dict[str, str]) -> tuple[int, bytes]:
        if self._transport:
            return self._transport(self.settings.base_url, headers, body)
        request = urllib.request.Request(
            self.settings.base_url, data=body, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.settings.timeout_s) as response:
                return response.status, response.read(2_000_000)
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read(2_000_000)
        except (urllib.error.URLError, TimeoutError) as exc:
            raise JevApiError(f"Jev transport failed: {exc.reason if hasattr(exc, 'reason') else exc}") from exc

    def _write_cache(self, path: Path, response: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(response, fh, ensure_ascii=False, separators=(",", ":"))
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def build_questions(grid: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Build the v3 questions: 8 categorical + 80 barrier probabilities."""
    horizon = grid["horizon_bars"]
    questions: dict[str, dict[str, Any]] = {}
    questions["regime"] = {
        "type": "choice",
        "instructions": "Which single market regime best describes the current state?",
        "criteria": {
            "up": "persistent upward trend",
            "down": "persistent downward trend",
            "range": "range-bound or mean-reverting market",
            "transition": "transition between regimes",
        },
    }
    questions["volatility"] = {
        "type": "choice",
        "instructions": "Which single volatility regime best describes the current state?",
        "criteria": {
            "low": "low volatility",
            "normal": "normal volatility",
            "high": "high volatility",
            "extreme": "extreme volatility",
        },
    }
    for side in ("long", "short"):
        for tp in grid["tp"]:
            for sl in grid["sl"]:
                key = tp_key(side, tp, sl)
                direction = "up" if side == "long" else "down"
                questions[key] = {
                    "type": "noul",
                    "instructions": (
                        f"Will a {side} trade hit its {direction} take-profit at "
                        f"{tp:.4%} before its stop-loss at {sl:.4%} "
                        f"within {horizon} bars?"
                    ),
                }
                stop_key = sl_key(side, tp, sl)
                questions[stop_key] = {
                    "type": "noul",
                    "instructions": (
                        f"Will a {side} trade hit its {side} stop-loss at {sl:.4%} "
                        f"before its take-profit at {tp:.4%} within {horizon} bars?"
                    ),
                }
    return questions


def _result(
    raw: Mapping[str, Any],
    questions: Mapping[str, Mapping[str, Any]],
    state_hash: str,
    *,
    latency_ms: float = 0.0,
    cached: bool,
) -> JevResult:
    if not isinstance(raw, Mapping):
        raise JevSchemaError("Jev response must be an object")
    if not isinstance(raw.get("answers"), dict):
        raise JevSchemaError("Jev response missing object: answers")
    answers: dict[str, float] = {}
    for key in questions:
        answer = raw["answers"].get(key)
        question = questions[key]
        if not isinstance(answer, dict) or answer.get("type") != question.get("type"):
            raise JevSchemaError(f"Jev response has invalid answer: {key}")
        if question.get("type") == "noul":
            value = answer.get("noul")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise JevSchemaError(f"Jev response probability out of range: {key}")
            answers[key] = float(value)
        elif question.get("type") == "choice":
            probabilities = answer.get("probabilities")
            criteria = question.get("criteria", {})
            if not isinstance(probabilities, dict) or set(probabilities) != set(criteria):
                raise JevSchemaError(f"Jev response has invalid choice probabilities: {key}")
            values = list(probabilities.values())
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1 for value in values):
                raise JevSchemaError(f"Jev response choice probability out of range: {key}")
            if not math.isclose(sum(values), 1.0, abs_tol=1e-3):
                raise JevSchemaError(f"Jev response choice probabilities do not sum to 1: {key}")
            output_prefix = "vol" if key == "volatility" else key
            for option, value in probabilities.items():
                answers[f"{output_prefix}_{option}"] = float(value)
        else:
            raise JevSchemaError(f"unsupported Jev question type: {question.get('type')}")
    for key, value in answers.items():
        if (
            not key.startswith("p_")
            or "_tp" not in key
            or "_sl" not in key
            or key.index("_tp") > key.index("_sl")
        ):
            continue
        side, remainder = key[2:].split("_tp", 1)
        tp_code, sl_code = remainder.split("_sl", 1)
        paired = f"p_{side}_sl{sl_code}_tp{tp_code}"
        if paired in answers and value + answers[paired] > 1 + 1e-9:
            raise JevSchemaError(f"Jev response TP/SL probabilities exceed 1: {key}")
    model = raw.get("model")
    if not isinstance(model, str) or not model:
        raise JevSchemaError("Jev response missing model")
    usage = raw.get("usage")
    if not isinstance(usage, dict):
        raise JevSchemaError("Jev response missing object: usage")
    normalized_usage: dict[str, int | float] = {}
    for target, aliases in {
        "input_tokens": ("input_tokens", "inputTokens"),
        "output_tokens": ("output_tokens", "outputTokens"),
        "cache_read_input_tokens": ("cache_read_input_tokens", "cacheReadInputTokens"),
        "cache_creation_input_tokens": ("cache_creation_input_tokens", "cacheCreationInputTokens"),
    }.items():
        value = next((usage.get(alias) for alias in aliases if alias in usage), None)
        if value is None and target.startswith("cache_"):
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise JevSchemaError(f"Jev usage missing/non-negative integer: {target}")
        normalized_usage[target] = value
    if "cost" in usage:
        cost = usage["cost"]
        if isinstance(cost, bool) or not isinstance(cost, (int, float)) or cost < 0:
            raise JevSchemaError("Jev usage cost must be a non-negative number")
        normalized_usage["cost"] = float(cost)
    return JevResult(answers, model, normalized_usage, latency_ms, state_hash, cached, dict(raw))


def _read_dotenv(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        if separator and key.strip().isidentifier():
            try:
                result[key.strip()] = shlex.split(value, comments=True)[0] if value.strip() else ""
            except ValueError as exc:
                raise JevConfigError(f"invalid dotenv value for {key.strip()}") from exc
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    item = getattr(value, "item", None)
    if callable(item):
        return _jsonable(item())
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return isoformat()
    raise TypeError(f"state contains non-JSON value: {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def canonical_hash(value: Any) -> str:
    """Stable, secret-free hash for request and artifact identities."""
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def canonical_request_payload(
    settings: JevSettings,
    state: Any,
    questions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the exact payload sent to Jev, including the injected contract."""
    state_obj = _jsonable(state)
    version = settings.prompt_version
    if version == PROMPT_VERSION_OHLCV_V4:
        if (state_obj.get("state_version") != "jev-state-v4"
                or state_obj.get("feature_version") != "ohlcv-features-v1"
                or state_obj.get("market_type") != "usdt-perpetual"
                or state_obj.get("exchange") not in {"binance", "okx"}
                or not state_obj.get("pair")
                or set(state_obj.get("features", {})) != set(OHLCV_FEATURES_V1)):
            raise JevConfigError(
                "jev-ohlcv-v4 requires the exact OHLCV feature contract and "
                "versioned USDT-perpetual state"
            )
        protocol = {
            "prompt_version": version,
            "feature_version": state_obj["feature_version"],
            "state_version": state_obj["state_version"],
            "contract": prompt_contract_v4(
                int(state_obj.get("grid", {}).get("horizon_bars", 16))
            ),
        }
    else:
        if state_obj.get("state_version") == "jev-state-v4":
            raise JevConfigError("jev-state-v4 requires prompt_version jev-ohlcv-v4")
        protocol = {
            "prompt_version": version,
            "contract": prompt_contract(
                int(state_obj.get("grid", {}).get("horizon_bars", 16))
            ),
        }
    return {
        "model": settings.model_id,
        "state": {"_jev_protocol": protocol, "market": state_obj},
        "questions": _jsonable(questions),
    }


def canonical_request_identity(
    settings: JevSettings,
    state: Any,
    questions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Canonical request plus endpoint, excluding credentials."""
    return {
        "endpoint": settings.base_url,
        "payload": canonical_request_payload(settings, state, questions),
    }


def _error_detail(body: bytes, api_key: str) -> str:
    text = body.decode("utf-8", errors="replace").replace(api_key, "<redacted>")
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
            text = str(parsed["error"].get("message", parsed["error"]))
    except json.JSONDecodeError:
        pass
    return " ".join(text.split())[:500] or "empty response"
