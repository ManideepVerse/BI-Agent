"""Tests for the provider abstraction and the failover chain.

No network is touched: fake clients stand in for the real adapters.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import llm as llm_module  # noqa: E402
from src.llm import (  # noqa: E402
    ToolCall,
    FallbackLLM,
    LLMError,
    LLMReply,
    ProviderCandidate,
    _pick_gemini_model,
    _pick_model,
    _replacement_model,
    _to_gemini_schema,
    build_llm_with_fallback,
)


class FakeClient:
    """Stands in for a provider adapter. Fails a configurable number of times."""

    def __init__(self, name: str, error: str | None = None):
        self.model = f"{name}-model"
        self.name = name
        self.error = error
        self.calls = 0

    def chat(self, system, messages, tools):
        self.calls += 1
        if self.error:
            raise LLMError(self.error)
        return LLMReply(text=f"answer from {self.name}")

    def close(self):
        pass


@pytest.fixture
def fake_builder(monkeypatch):
    """Replace build_llm so FallbackLLM constructs FakeClients."""
    built: dict[str, FakeClient] = {}
    errors: dict[str, str] = {}

    def _build(provider, api_key, model=""):
        client = FakeClient(provider, errors.get(provider))
        built[provider] = client
        return client

    monkeypatch.setattr(llm_module, "build_llm", _build)
    return built, errors


# ------------------------------------------------------------------ failover
def test_primary_is_used_when_healthy(fake_builder):
    built, _errors = fake_builder
    chain = FallbackLLM([ProviderCandidate("gemini", "k1"), ProviderCandidate("openai", "k2")])
    reply = chain.chat("sys", [], [])
    assert reply.text == "answer from gemini"
    assert "openai" not in built, "the fallback must not even be constructed"


def test_falls_over_when_primary_is_out_of_quota(fake_builder):
    _built, errors = fake_builder
    errors["gemini"] = "gemini rate limit / quota exceeded."
    chain = FallbackLLM([ProviderCandidate("gemini", "k1"), ProviderCandidate("openai", "k2")])
    reply = chain.chat("sys", [], [])
    assert reply.text == "answer from openai"
    assert chain.provider == "openai"


def test_falls_over_on_a_dead_api_key(fake_builder):
    _built, errors = fake_builder
    errors["gemini"] = "gemini rejected the API key (HTTP 401)."
    chain = FallbackLLM([ProviderCandidate("gemini", "k1"), ProviderCandidate("anthropic", "k2")])
    assert chain.chat("sys", [], []).text == "answer from anthropic"


def test_stays_on_the_fallback_instead_of_retrying_a_dead_primary(fake_builder):
    built, errors = fake_builder
    errors["gemini"] = "gemini rate limit / quota exceeded."
    chain = FallbackLLM([ProviderCandidate("gemini", "k1"), ProviderCandidate("openai", "k2")])
    chain.chat("sys", [], [])
    chain.chat("sys", [], [])
    chain.chat("sys", [], [])
    assert built["gemini"].calls == 1, "a known-dead provider must not be retried every turn"
    assert built["openai"].calls == 3


def test_content_blocks_do_not_fail_over(fake_builder):
    """Retrying a policy-blocked request elsewhere just burns another quota."""
    built, errors = fake_builder
    errors["gemini"] = "Gemini blocked the request (SAFETY)."
    chain = FallbackLLM([ProviderCandidate("gemini", "k1"), ProviderCandidate("openai", "k2")])
    with pytest.raises(LLMError, match="blocked"):
        chain.chat("sys", [], [])
    assert "openai" not in built


def test_all_providers_failing_reports_every_reason(fake_builder):
    _built, errors = fake_builder
    errors["gemini"] = "gemini rate limit / quota exceeded."
    errors["openai"] = "Could not reach openai: timeout"
    chain = FallbackLLM([ProviderCandidate("gemini", "k1"), ProviderCandidate("openai", "k2")])
    with pytest.raises(LLMError) as exc:
        chain.chat("sys", [], [])
    assert "gemini" in str(exc.value) and "openai" in str(exc.value)


def test_empty_chain_is_rejected():
    with pytest.raises(LLMError):
        FallbackLLM([])


# ------------------------------------------------------------- chain building
def test_chain_puts_the_chosen_provider_first(fake_builder):
    chain = build_llm_with_fallback(
        "anthropic",
        {"gemini": "k1", "openai": "k2", "anthropic": "k3"},
    )
    assert chain.provider == "anthropic"
    assert set(chain.standby_providers) == {"gemini", "openai"}


def test_chain_skips_providers_with_no_key(fake_builder):
    chain = build_llm_with_fallback("gemini", {"gemini": "k1", "openai": "", "anthropic": None})
    assert chain.standby_providers == []


def test_chain_with_no_keys_at_all_is_an_error(fake_builder):
    with pytest.raises(LLMError, match="No API key"):
        build_llm_with_fallback("gemini", {"gemini": "", "openai": "", "anthropic": ""})


# ------------------------------------------------------- schema + model choice
def test_gemini_schema_uses_uppercase_types():
    converted = _to_gemini_schema({
        "type": "object",
        "properties": {"sql": {"type": "string"}, "limit": {"type": "integer"}},
        "required": ["sql"],
    })
    assert converted["type"] == "OBJECT"
    assert converted["properties"]["sql"]["type"] == "STRING"
    assert converted["properties"]["limit"]["type"] == "INTEGER"
    assert converted["required"] == ["sql"]


# ------------------------------------------------- Gemini model selection
GEMINI_MODELS = [
    "gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.5-flash",
    "gemini-3.6-flash", "gemini-3.6-flash-preview", "gemini-3.6-pro",
    "gemini-embedding-001", "imagen-3.0",
]


def test_gemini_picks_the_newest_stable_flash():
    assert _pick_gemini_model(GEMINI_MODELS) == "gemini-3.6-flash"


def test_gemini_selection_is_version_agnostic():
    """A model released after this code was written must be picked automatically.

    Hardcoding a preference list is what broke this in the first place: Google
    retires ids faster than anyone redeploys.
    """
    assert _pick_gemini_model([*GEMINI_MODELS, "gemini-4-flash"]) == "gemini-4-flash"


def test_gemini_skips_retired_models():
    chosen = _pick_gemini_model(GEMINI_MODELS, exclude={"gemini-3.6-flash"})
    assert chosen != "gemini-3.6-flash"


def test_gemini_avoids_previews_lite_builds_and_non_chat_models():
    assert _pick_gemini_model(["gemini-3.6-flash-preview", "gemini-2.5-flash"]) == "gemini-2.5-flash"
    assert _pick_gemini_model(["gemini-2.0-flash-lite", "gemini-2.0-flash"]) == "gemini-2.0-flash"
    assert _pick_gemini_model(["gemini-embedding-001", "imagen-3.0"]) is None


# ------------------------------------------------ recovering from a dead model
GONE_404 = (
    'gemini returned HTTP 404: {"error": {"code": 404, "message": "This model '
    'models/gemini-2.5-flash is no longer available to new users. Please update '
    'your code to use models/gemini-3.6-flash for the latest features and '
    'improvements.", "status": "NOT_FOUND"}}'
)


def test_replacement_model_is_read_from_googles_error():
    assert _replacement_model(LLMError(GONE_404)) == "gemini-3.6-flash"


def test_model_gone_without_a_named_replacement_triggers_rediscovery():
    assert _replacement_model(LLMError("gemini returned HTTP 404: not found")) == ""


@pytest.mark.parametrize("message", [
    "gemini rate limit / quota exceeded.",
    "Could not reach gemini: timeout",
    "gemini rejected the API key (HTTP 401).",
    "gemini returned HTTP 500",
])
def test_other_errors_are_not_treated_as_a_dead_model(message):
    """Only retire a model for model-specific errors, never for quota or network."""
    assert _replacement_model(LLMError(message)) is None


# ------------------------------------------------- Gemini thought signatures
class _ScriptedGemini(llm_module.GeminiLLM):
    """A GeminiLLM whose HTTP layer is replaced by a scripted response."""

    def __init__(self, response: dict):
        self._retired = set()
        self._key = "k"
        self.model = "gemini-3-flash"
        self._response = response
        self.sent: dict = {}

        class _Transport:
            def request(_s, method, url, **kwargs):
                self.sent = kwargs.get("json") or {}
                return _FakeResponse(200, __import__("json").dumps(self._response))

        self._client = _Transport()

    def _discover_model(self):
        return "gemini-3-flash"


def test_thought_signature_is_captured_from_a_function_call():
    client = _ScriptedGemini({
        "candidates": [{
            "finishReason": "STOP",
            "content": {"parts": [{
                "functionCall": {"name": "get_schema", "args": {}},
                "thoughtSignature": "SIG-ABC123",
            }]},
        }]
    })
    reply = client.chat("sys", [{"role": "user", "content": "hi"}], [])
    assert reply.tool_calls[0].signature == "SIG-ABC123"


def test_thought_signature_is_echoed_back_on_the_next_turn():
    """Gemini 3 rejects the whole request with a 400 if this is dropped."""
    client = _ScriptedGemini({"candidates": [{"content": {"parts": [{"text": "done"}]}}]})
    history = [
        {"role": "user", "content": "How many Mining deals?"},
        {
            "role": "assistant",
            "content": "",
            "signature": "TEXT-SIG",
            "tool_calls": [ToolCall(id="c1", name="get_schema", args={}, signature="SIG-ABC123")],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "get_schema", "content": '{"tables": []}'},
    ]
    client.chat("sys", history, [])

    model_turn = next(c for c in client.sent["contents"] if c["role"] == "model")
    call_part = next(p for p in model_turn["parts"] if "functionCall" in p)
    assert call_part["thoughtSignature"] == "SIG-ABC123"


def test_absent_signature_is_not_sent_as_empty_string():
    """Older models return no signature; sending an empty one is itself invalid."""
    client = _ScriptedGemini({"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})
    history = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [ToolCall(id="c1", name="get_schema", args={})],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "get_schema", "content": "{}"},
    ]
    client.chat("sys", history, [])
    model_turn = next(c for c in client.sent["contents"] if c["role"] == "model")
    call_part = next(p for p in model_turn["parts"] if "functionCall" in p)
    assert "thoughtSignature" not in call_part


def test_thought_signature_error_does_not_retire_the_model():
    """A 400 about signatures is our bug, not a dead model — don't churn models."""
    message = (
        'gemini returned HTTP 400: {"error": {"code": 400, "message": "Function call is '
        'missing a thought_signature in functionCall parts."}}'
    )
    assert _replacement_model(LLMError(message)) is None


# --------------------------------------------------------- transport retries
class _FakeResponse:
    def __init__(self, status_code: int, text: str = "{}", headers: dict | None = None):
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}

    def json(self):
        import json as _json

        return _json.loads(self.text)


class _RecordingLLM(llm_module.BaseLLM):
    """A BaseLLM whose transport returns a scripted sequence of responses."""

    provider = "fake"

    def __init__(self, responses):
        self._responses = list(responses)
        self.attempts = 0
        self._key = "k"
        self.model = "fake-model"

        class _Transport:
            def request(_self, method, url, **kwargs):
                self.attempts += 1
                item = self._responses.pop(0)
                if isinstance(item, Exception):
                    raise item
                return item

        self._client = _Transport()

    def _discover_model(self):
        return "fake-model"


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(llm_module.time, "sleep", lambda *_a, **_k: None)


def test_transient_503_is_retried_then_succeeds():
    """Google says 'spikes in demand are usually temporary' — so wait, don't bail."""
    client = _RecordingLLM([
        _FakeResponse(503, '{"error": {"code": 503, "message": "high demand"}}'),
        _FakeResponse(503, '{"error": {"code": 503, "message": "high demand"}}'),
        _FakeResponse(200, '{"ok": true}'),
    ])
    assert client._request("POST", "http://x") == {"ok": True}
    assert client.attempts == 3


def test_rate_limit_is_retried():
    client = _RecordingLLM([
        _FakeResponse(429, '{"error": "rate limit, 60 requests per minute"}'),
        _FakeResponse(200, '{"ok": true}'),
    ])
    assert client._request("POST", "http://x") == {"ok": True}


def test_hard_quota_is_not_retried():
    """No amount of waiting adds credit to an empty account."""
    client = _RecordingLLM([
        _FakeResponse(429, '{"error": {"code": "insufficient_quota", "message": "exceeded your current quota"}}'),
    ])
    with pytest.raises(LLMError, match="quota exhausted"):
        client._request("POST", "http://x")
    assert client.attempts == 1, "a credit problem must fail over immediately"


def test_network_errors_are_retried():
    import httpx

    client = _RecordingLLM([
        httpx.ConnectError("boom"),
        _FakeResponse(200, '{"ok": true}'),
    ])
    assert client._request("POST", "http://x") == {"ok": True}


def test_bad_request_is_not_retried():
    """A 400/404 is the same broken request every time."""
    client = _RecordingLLM([_FakeResponse(404, '{"error": "no such model"}')])
    with pytest.raises(LLMError, match="404"):
        client._request("POST", "http://x")
    assert client.attempts == 1


def test_auth_failure_is_not_retried():
    client = _RecordingLLM([_FakeResponse(401, "unauthorized")])
    with pytest.raises(LLMError, match="rejected the API key"):
        client._request("POST", "http://x")
    assert client.attempts == 1


def test_persistent_outage_gives_up_and_reports_it():
    client = _RecordingLLM([_FakeResponse(503, "busy") for _ in range(4)])
    with pytest.raises(LLMError, match="Gave up after"):
        client._request("POST", "http://x")
    assert client.attempts == 4


def test_model_picker_prefers_stable_over_preview():
    chosen = _pick_model(
        ["gemini-2.5-flash-preview", "gemini-2.5-flash", "gemini-2.5-pro"],
        ["gemini-2.5-flash"],
    )
    assert chosen == "gemini-2.5-flash"


def test_model_picker_excludes_non_chat_models():
    chosen = _pick_model(
        ["gemini-embedding-001", "imagen-3.0", "gemini-2.0-flash"],
        ["gemini-flash"],
    )
    assert chosen == "gemini-2.0-flash"
