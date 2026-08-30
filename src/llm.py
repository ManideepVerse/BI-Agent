"""Provider-agnostic LLM layer with tool calling.

Three providers (Google Gemini, OpenAI, Anthropic) are supported behind one
interface, talked to over plain HTTP with ``httpx``. No vendor SDKs: three thin
adapters are ~60 lines each, they keep the deployment small, and they mean a
provider outage or an exhausted free tier is a one-line config change rather
than a rewrite.

Canonical message format used everywhere else in the app::

    {"role": "user",      "content": "..."}
    {"role": "assistant", "content": "...", "tool_calls": [ToolCall, ...]}
    {"role": "tool",      "tool_call_id": "...", "name": "...", "content": "<json>"}
"""

from __future__ import annotations

import json
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from .logging_conf import get_logger

log = get_logger(__name__)

_EXCLUDE_MODEL = re.compile(
    r"embed|imagen|veo|tts|audio|whisper|dall|image|vision|moderation|rerank|"
    r"aqa|learnlm|realtime|transcribe|search|computer-use",
    re.IGNORECASE,
)


class LLMError(RuntimeError):
    """A user-presentable LLM failure."""


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict = field(default_factory=dict)
    # Gemini 3 returns an encrypted "thought signature" alongside each function
    # call and rejects the next turn if it is not echoed back verbatim. Opaque
    # to us; carried so the transcript can be replayed faithfully.
    signature: str = ""


@dataclass
class LLMReply:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    signature: str = ""


# --------------------------------------------------------------------------- #
# Schema translation
# --------------------------------------------------------------------------- #
_GEMINI_TYPES = {
    "string": "STRING", "number": "NUMBER", "integer": "INTEGER",
    "boolean": "BOOLEAN", "array": "ARRAY", "object": "OBJECT",
}


def _to_gemini_schema(schema: dict) -> dict:
    """Gemini accepts an OpenAPI subset with UPPERCASE type names."""
    out: dict[str, Any] = {}
    if "type" in schema:
        out["type"] = _GEMINI_TYPES.get(str(schema["type"]).lower(), "STRING")
    if "description" in schema:
        out["description"] = schema["description"]
    if "enum" in schema:
        out["enum"] = [str(v) for v in schema["enum"]]
    if "properties" in schema:
        out["properties"] = {k: _to_gemini_schema(v) for k, v in schema["properties"].items()}
    if "items" in schema:
        out["items"] = _to_gemini_schema(schema["items"])
    if schema.get("required"):
        out["required"] = list(schema["required"])
    return out


# A hard quota means the account is out of credit — waiting will not help, so
# the chain should move on immediately instead of backing off for 30 seconds.
# A plain rate limit (requests per minute) is the opposite: worth waiting for.
_HARD_QUOTA = re.compile(
    r"insufficient_quota|exceeded your current quota|billing|credit balance|"
    r"payment|no remaining credit|purchase",
    re.IGNORECASE,
)


def _is_hard_quota(detail: str) -> bool:
    return bool(_HARD_QUOTA.search(detail or ""))


_MODEL_GONE = re.compile(
    r"http 404|not_?found|no longer available|is not supported|has been (deprecated|retired)",
    re.IGNORECASE,
)
_NAMED_REPLACEMENT = re.compile(r"models/([A-Za-z0-9][\w.\-]*)")


def _replacement_model(exc: Exception) -> Optional[str]:
    """If an error says the model is gone, return the replacement it names.

    Returns ``""`` when the model is gone but no replacement is named (the
    caller should rediscover), and ``None`` when the error is about something
    else entirely.
    """
    message = str(exc)
    if not _MODEL_GONE.search(message):
        return None
    # Google's 404 body says: "use models/gemini-3.6-flash for the latest ...".
    # The last model it names is the recommendation, the first is the dead one.
    named = _NAMED_REPLACEMENT.findall(message)
    return named[-1] if len(named) > 1 else ""


def _model_version(name: str) -> tuple[float, ...]:
    """Extract a sortable version from a model id: gemini-3.6-flash -> (3.6,)."""
    match = re.search(r"(\d+)(?:\.(\d+))?", name)
    if not match:
        return (0.0,)
    major = float(match.group(1))
    minor = float(match.group(2) or 0) / 100
    return (major + minor,)


def _rank_gemini_models(available: list[str], exclude: set[str] | None = None) -> list[str]:
    """Rank Gemini chat models best-first.

    Deliberately version-agnostic: Google retires model ids faster than anyone
    redeploys, and a hardcoded preference list is guaranteed to go stale. This
    ranks by capability class then by version number, so a newly released
    ``gemini-4-flash`` sorts to the top automatically.
    """
    exclude = exclude or set()
    usable = [m for m in available if not _EXCLUDE_MODEL.search(m) and m not in exclude]

    def score(name: str) -> tuple:
        lowered = name.lower()
        return (
            # Stable releases beat previews and experiments.
            0 if re.search(r"preview|exp|beta", lowered) else 1,
            # Flash is the right default for a tool-calling agent: fast and cheap.
            2 if "flash" in lowered else (1 if "pro" in lowered else 0),
            # A full model beats a distilled one.
            0 if "lite" in lowered else 1,
            _model_version(lowered),
            # Prefer the plain id over dated snapshots like -001 / -latest.
            -len(name),
        )

    return sorted(usable, key=score, reverse=True)


def _pick_gemini_model(available: list[str], exclude: set[str] | None = None) -> Optional[str]:
    ranked = _rank_gemini_models(available, exclude)
    return ranked[0] if ranked else None


def _rank_by_preference(available: list[str], preferences: list[str]) -> list[str]:
    """Rank models so preferred families sort first, stable builds before dated ones."""
    def score(name: str) -> tuple:
        lowered = name.lower()
        rank = next(
            (len(preferences) - i for i, pref in enumerate(preferences) if pref in lowered),
            0,
        )
        return (
            rank,
            0 if re.search(r"preview|\d{4}-\d{2}-\d{2}|\d{8}", lowered) else 1,
            -len(name),
        )

    return sorted(available, key=score, reverse=True)


def _pick_model(available: list[str], preferences: list[str]) -> Optional[str]:
    usable = [m for m in available if not _EXCLUDE_MODEL.search(m)]
    for pref in preferences:
        exact = [m for m in usable if m == pref]
        if exact:
            return exact[0]
    for pref in preferences:
        partial = sorted(m for m in usable if pref in m)
        if partial:
            # Prefer a stable release over a preview/experimental build.
            stable = [m for m in partial if not re.search(r"preview|exp|beta|latest", m)]
            return (stable or partial)[0]
    return usable[0] if usable else None


# --------------------------------------------------------------------------- #
# Base
# --------------------------------------------------------------------------- #
class BaseLLM:
    provider = "base"
    default_preferences: list[str] = []

    def __init__(self, api_key: str, model: str = "", *, timeout: float = 120.0):
        if not api_key:
            raise LLMError(f"No API key configured for provider '{self.provider}'.")
        self._key = api_key
        self._client = httpx.Client(timeout=timeout)
        self.model = model or self._discover_model()

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # pragma: no cover
            pass

    # -- to implement ------------------------------------------------------
    def list_models(self) -> list[str]:  # pragma: no cover - overridden
        """Chat-capable models this key can use, best first."""
        raise NotImplementedError

    def _discover_model(self) -> str:
        models = self.list_models()
        if not models:
            raise LLMError(f"No usable {self.provider} chat model is available for this key.")
        log.info("%s model selected: %s", self.provider, models[0])
        return models[0]

    def chat(self, system: str, messages: list[dict], tools: list) -> LLMReply:  # pragma: no cover
        raise NotImplementedError

    # -- shared ------------------------------------------------------------
    MAX_ATTEMPTS = 4

    def _request(self, method: str, url: str, **kwargs: Any) -> dict:
        """Send one request, retrying the failures that are worth retrying.

        Providers rate-limit and shed load constantly — a 503 "high demand,
        try again later" is an instruction, not a verdict. Retrying here means
        the failover chain is reserved for a provider that is genuinely unusable
        rather than one that is briefly busy.
        """
        last_error: LLMError | None = None

        for attempt in range(self.MAX_ATTEMPTS):
            if attempt:
                delay = min(2**attempt, 20) + random.uniform(0, 0.5)
                log.warning(
                    "%s retry %s/%s in %.1fs (%s)",
                    self.provider, attempt, self.MAX_ATTEMPTS - 1, delay, last_error,
                )
                time.sleep(delay)

            try:
                response = self._client.request(method, url, **kwargs)
            except httpx.HTTPError as exc:
                last_error = LLMError(f"Could not reach {self.provider}: {exc}")
                continue

            if response.status_code in (401, 403):
                raise LLMError(
                    f"{self.provider} rejected the API key (HTTP {response.status_code}). "
                    "Check the key is correct and still active."
                )

            if response.status_code == 429:
                detail = response.text[:400]
                if _is_hard_quota(detail):
                    # Out of credit, not out of requests-per-minute. Waiting
                    # cannot fix this, so hand over to the next provider now.
                    raise LLMError(
                        f"{self.provider} quota exhausted — the account has no remaining "
                        f"credit. {detail[:160]}"
                    )
                retry_after = float(response.headers.get("Retry-After", 0) or 0)
                if retry_after:
                    time.sleep(min(retry_after, 30))
                last_error = LLMError(f"{self.provider} rate limited the request (HTTP 429).")
                continue

            if response.status_code >= 500:
                last_error = LLMError(
                    f"{self.provider} is temporarily unavailable (HTTP {response.status_code})."
                )
                continue

            if response.status_code >= 400:
                # 4xx is a problem with the request itself — a bad model id, a
                # malformed tool schema. Retrying sends the same broken request.
                raise LLMError(
                    f"{self.provider} returned HTTP {response.status_code}: {response.text[:400]}"
                )

            try:
                return response.json()
            except ValueError as exc:
                raise LLMError(f"{self.provider} returned a non-JSON response.") from exc

        assert last_error is not None
        raise LLMError(
            f"{last_error} Gave up after {self.MAX_ATTEMPTS} attempts."
        )


# --------------------------------------------------------------------------- #
# Google Gemini
# --------------------------------------------------------------------------- #
class GeminiLLM(BaseLLM):
    provider = "gemini"
    BASE = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, api_key: str, model: str = "", *, timeout: float = 120.0):
        # Populated when a model turns out to be unusable; see `chat`.
        self._retired: set[str] = set()
        super().__init__(api_key, model, timeout=timeout)

    def _headers(self) -> dict:
        return {"x-goog-api-key": self._key, "Content-Type": "application/json"}

    def list_models(self) -> list[str]:
        body = self._request("GET", f"{self.BASE}/models", headers=self._headers())
        names = [
            m["name"].split("/")[-1]
            for m in body.get("models", [])
            if "generateContent" in (m.get("supportedGenerationMethods") or [])
        ]
        return _rank_gemini_models(names, exclude=self._retired)

    def chat(self, system: str, messages: list[dict], tools: list) -> LLMReply:
        try:
            return self._generate(system, messages, tools)
        except LLMError as exc:
            replacement = _replacement_model(exc)
            if replacement is None:
                raise
            # Google's model list can advertise a model that generation then
            # rejects for this key. Retire it, take the replacement Google names
            # (or rediscover), and retry once before failing over to another
            # provider entirely.
            log.warning("Gemini model '%s' is unusable (%s). Switching to '%s'.",
                        self.model, str(exc)[:120], replacement or "next best")
            self._retired.add(self.model)
            self.model = replacement or self._discover_model()
            return self._generate(system, messages, tools)

    def _generate(self, system: str, messages: list[dict], tools: list) -> LLMReply:
        contents: list[dict] = []
        for message in messages:
            role = message["role"]
            if role == "user":
                contents.append({"role": "user", "parts": [{"text": message["content"]}]})
            elif role == "assistant":
                parts: list[dict] = []
                if message.get("content"):
                    text_part: dict[str, Any] = {"text": message["content"]}
                    if message.get("signature"):
                        text_part["thoughtSignature"] = message["signature"]
                    parts.append(text_part)
                for call in message.get("tool_calls") or []:
                    call_part: dict[str, Any] = {
                        "functionCall": {"name": call.name, "args": call.args}
                    }
                    # Must be echoed exactly as received or Gemini 3 rejects the
                    # whole request with a 400.
                    if call.signature:
                        call_part["thoughtSignature"] = call.signature
                    parts.append(call_part)
                if parts:
                    contents.append({"role": "model", "parts": parts})
            elif role == "tool":
                contents.append({
                    "role": "user",
                    "parts": [{
                        "functionResponse": {
                            "name": message["name"],
                            "response": {"result": _maybe_json(message["content"])},
                        }
                    }],
                })

        payload: dict[str, Any] = {
            "contents": contents,
            "systemInstruction": {"parts": [{"text": system}]},
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 4096},
        }
        if tools:
            payload["tools"] = [{
                "functionDeclarations": [
                    _gemini_declaration(t) for t in tools
                ]
            }]

        body = self._request(
            "POST",
            f"{self.BASE}/models/{self.model}:generateContent",
            headers=self._headers(),
            json=payload,
        )

        feedback = body.get("promptFeedback") or {}
        if feedback.get("blockReason"):
            raise LLMError(f"Gemini blocked the request ({feedback['blockReason']}).")

        candidates = body.get("candidates") or []
        if not candidates:
            raise LLMError("Gemini returned no candidates.")
        candidate = candidates[0]
        reply = LLMReply(finish_reason=candidate.get("finishReason", ""))
        for part in (candidate.get("content") or {}).get("parts") or []:
            signature = part.get("thoughtSignature") or ""
            if "text" in part and part["text"]:
                reply.text += part["text"]
                reply.signature = reply.signature or signature
            call = part.get("functionCall")
            if call:
                reply.tool_calls.append(
                    ToolCall(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        name=call["name"],
                        args=call.get("args") or {},
                        signature=signature,
                    )
                )
        if reply.finish_reason == "MAX_TOKENS" and not reply.text:
            raise LLMError("Gemini hit its output limit before producing an answer.")
        return reply


def _gemini_declaration(tool: Any) -> dict:
    declaration: dict[str, Any] = {"name": tool.name, "description": tool.description}
    params = tool.parameters or {}
    if params.get("properties"):
        declaration["parameters"] = _to_gemini_schema(params)
    return declaration


# --------------------------------------------------------------------------- #
# OpenAI
# --------------------------------------------------------------------------- #
class OpenAILLM(BaseLLM):
    provider = "openai"
    BASE = "https://api.openai.com/v1"
    default_preferences = ["gpt-4.1-mini", "gpt-4o-mini", "gpt-4.1", "gpt-4o", "gpt-5-mini", "gpt-5"]

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}

    def list_models(self) -> list[str]:
        body = self._request("GET", f"{self.BASE}/models", headers=self._headers())
        names = [
            m["id"] for m in body.get("data", [])
            if m.get("id", "").startswith(("gpt", "o1", "o3", "o4"))
            and not _EXCLUDE_MODEL.search(m.get("id", ""))
        ]
        return _rank_by_preference(names, self.default_preferences)

    def chat(self, system: str, messages: list[dict], tools: list) -> LLMReply:
        payload_messages: list[dict] = [{"role": "system", "content": system}]
        for message in messages:
            role = message["role"]
            if role == "user":
                payload_messages.append({"role": "user", "content": message["content"]})
            elif role == "assistant":
                entry: dict[str, Any] = {"role": "assistant", "content": message.get("content") or None}
                calls = message.get("tool_calls") or []
                if calls:
                    entry["tool_calls"] = [{
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": json.dumps(c.args)},
                    } for c in calls]
                payload_messages.append(entry)
            elif role == "tool":
                payload_messages.append({
                    "role": "tool",
                    "tool_call_id": message["tool_call_id"],
                    "content": message["content"],
                })

        payload: dict[str, Any] = {"model": self.model, "messages": payload_messages, "temperature": 0.1}
        if tools:
            payload["tools"] = [{
                "type": "function",
                "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
            } for t in tools]
            payload["tool_choice"] = "auto"

        body = self._request("POST", f"{self.BASE}/chat/completions", headers=self._headers(), json=payload)
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        reply = LLMReply(text=message.get("content") or "", finish_reason=choice.get("finish_reason", ""))
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            reply.tool_calls.append(
                ToolCall(id=call.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                         name=fn.get("name", ""),
                         args=_maybe_json(fn.get("arguments") or "{}") or {})
            )
        return reply


# --------------------------------------------------------------------------- #
# Anthropic
# --------------------------------------------------------------------------- #
class AnthropicLLM(BaseLLM):
    provider = "anthropic"
    BASE = "https://api.anthropic.com/v1"
    default_preferences = [
        "claude-sonnet-4-5", "claude-3-7-sonnet", "claude-3-5-sonnet",
        "claude-sonnet", "claude-haiku", "claude-opus",
    ]

    def _headers(self) -> dict:
        return {
            "x-api-key": self._key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    def list_models(self) -> list[str]:
        body = self._request("GET", f"{self.BASE}/models?limit=100", headers=self._headers())
        names = [m["id"] for m in body.get("data", []) if not _EXCLUDE_MODEL.search(m.get("id", ""))]
        return _rank_by_preference(names, self.default_preferences)

    def chat(self, system: str, messages: list[dict], tools: list) -> LLMReply:
        payload_messages: list[dict] = []
        for message in messages:
            role = message["role"]
            if role == "user":
                payload_messages.append({"role": "user", "content": message["content"]})
            elif role == "assistant":
                blocks: list[dict] = []
                if message.get("content"):
                    blocks.append({"type": "text", "text": message["content"]})
                for call in message.get("tool_calls") or []:
                    blocks.append({"type": "tool_use", "id": call.id, "name": call.name, "input": call.args})
                if blocks:
                    payload_messages.append({"role": "assistant", "content": blocks})
            elif role == "tool":
                payload_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": message["tool_call_id"],
                        "content": message["content"],
                    }],
                })

        payload: dict[str, Any] = {
            "model": self.model,
            "system": system,
            "messages": payload_messages,
            "max_tokens": 4096,
            "temperature": 0.1,
        }
        if tools:
            payload["tools"] = [{
                "name": t.name, "description": t.description, "input_schema": t.parameters,
            } for t in tools]

        body = self._request("POST", f"{self.BASE}/messages", headers=self._headers(), json=payload)
        reply = LLMReply(finish_reason=body.get("stop_reason", ""))
        for block in body.get("content") or []:
            if block.get("type") == "text":
                reply.text += block.get("text", "")
            elif block.get("type") == "tool_use":
                reply.tool_calls.append(
                    ToolCall(id=block.get("id", ""), name=block.get("name", ""), args=block.get("input") or {})
                )
        return reply


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
_PROVIDERS = {"gemini": GeminiLLM, "openai": OpenAILLM, "anthropic": AnthropicLLM}


def build_llm(provider: str, api_key: str, model: str = "") -> BaseLLM:
    cls = _PROVIDERS.get((provider or "").lower())
    if cls is None:
        raise LLMError(f"Unsupported LLM provider '{provider}'. Choose one of {sorted(_PROVIDERS)}.")
    return cls(api_key, model)


PROVIDER_LABELS = {
    "gemini": "Google Gemini",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
}

# Each vendor issues keys with a distinctive prefix, so the provider can be
# inferred rather than asked for — one less thing for a reviewer to get wrong.
_KEY_SHAPES = (
    ("sk-ant-", "anthropic"),
    ("AIza", "gemini"),
    ("sk-", "openai"),
)


def detect_provider(api_key: str) -> Optional[str]:
    """Infer the provider from an API key's prefix, or None if unrecognised."""
    key = (api_key or "").strip()
    for prefix, provider in _KEY_SHAPES:
        if key.startswith(prefix):
            return provider
    return None


def available_models(provider: str, api_key: str) -> list[str]:
    """List the chat models a key can actually use, best first.

    Used to populate the model picker. Constructed with a placeholder model so
    the client does not perform its own discovery round trip first.
    """
    client = build_llm(provider, api_key, model="__probe__")
    try:
        return client.list_models()
    finally:
        client.close()


# --------------------------------------------------------------------------- #
# Failover
# --------------------------------------------------------------------------- #
@dataclass
class ProviderCandidate:
    provider: str
    api_key: str
    model: str = ""


# A content-policy block is a property of the request, not the provider, so
# retrying the same request elsewhere just burns another quota. Everything else
# — quota exhaustion, a dead key, a network blip, a 5xx — is worth failing over.
_DO_NOT_FAILOVER = ("blocked the request",)


def _worth_failing_over(exc: LLMError) -> bool:
    message = str(exc).lower()
    return not any(marker in message for marker in _DO_NOT_FAILOVER)


class FallbackLLM:
    """Tries several providers in order, moving on when one is unusable.

    A free-tier key running out of quota mid-demo should degrade to the next
    provider, not take the app down. Once it fails over it *stays* on the new
    provider rather than retrying a known-dead one on every turn.
    """

    def __init__(self, candidates: list[ProviderCandidate]):
        if not candidates:
            raise LLMError("No LLM provider is configured. Set at least one provider API key.")
        self._candidates = candidates
        self._clients: dict[int, BaseLLM] = {}
        self._active = 0
        # Build the primary eagerly so misconfiguration surfaces at startup
        # rather than on the user's first question.
        self._client_at(0)

    def _client_at(self, index: int) -> BaseLLM:
        if index not in self._clients:
            candidate = self._candidates[index]
            self._clients[index] = build_llm(candidate.provider, candidate.api_key, candidate.model)
        return self._clients[index]

    @property
    def provider(self) -> str:
        return self._candidates[self._active].provider

    @property
    def model(self) -> str:
        client = self._clients.get(self._active)
        return client.model if client else ""

    @property
    def standby_providers(self) -> list[str]:
        return [c.provider for c in self._candidates[self._active + 1:]]

    def chat(self, system: str, messages: list[dict], tools: list) -> LLMReply:
        failures: list[str] = []
        for index in range(self._active, len(self._candidates)):
            provider = self._candidates[index].provider
            try:
                reply = self._client_at(index).chat(system, messages, tools)
            except LLMError as exc:
                if not _worth_failing_over(exc):
                    raise
                log.warning("Provider '%s' unavailable: %s", provider, exc)
                failures.append(f"{provider}: {exc}")
                continue
            if index != self._active:
                log.warning("Failed over from '%s' to '%s'.", self.provider, provider)
                self._active = index
            return reply

        raise LLMError(
            "Every configured LLM provider failed. " + " | ".join(failures)
        )

    def close(self) -> None:
        for client in self._clients.values():
            client.close()


def build_llm_with_fallback(
    primary: str,
    keys: dict[str, str],
    model: str = "",
) -> FallbackLLM:
    """Build a chain: the chosen provider first, then any other with a key set.

    ``keys`` maps provider name to API key; providers without a key are skipped.
    The explicit ``model`` only pins the primary — a fallback provider discovers
    its own best available model.
    """
    primary = (primary or "").lower()
    ordered = [primary] + [p for p in ("gemini", "openai", "anthropic") if p != primary]
    candidates = [
        ProviderCandidate(provider=name, api_key=keys[name], model=model if name == primary else "")
        for name in ordered
        if keys.get(name)
    ]
    if not candidates:
        raise LLMError(
            f"No API key found for '{primary}' or any other provider. "
            f"Set {primary.upper()}_API_KEY."
        )
    return FallbackLLM(candidates)


def _maybe_json(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value
