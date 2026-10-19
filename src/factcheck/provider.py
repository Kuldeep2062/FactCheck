"""Provider-neutral model turn plus the Gemini implementation."""
import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import httpx
from google import genai
from google.genai import errors, types

from .facts_tool import TOOL_NAME, tool_declaration_schema
from .reliability import ProviderError
from .schemas import MODEL_OUTPUT_JSON_SCHEMA


@dataclass
class ToolCall:
    name: str
    args: dict
    call_id: str | None = None


@dataclass
class ModelResponse:
    text: str | None
    tool_calls: list[ToolCall]
    usage: dict[str, int] = field(default_factory=dict)
    raw_content: Any = None
    finish_reason: str | None = None


@dataclass
class ModelRequest:
    system: str
    messages: list[dict]
    mode: Literal["tools", "final"]


class Provider(Protocol):
    name: str
    model: str

    async def generate(self, request: ModelRequest) -> ModelResponse: ...


_RETRY_DELAY = re.compile(r"retry in ([0-9.]+)s|'retryDelay': '([0-9.]+)s'", re.I)


def classify_api_error(exc: errors.APIError) -> ProviderError:
    code = exc.code or 0
    msg = f"{code} {getattr(exc, 'status', '') or ''}".strip()
    if code == 429:
        m = _RETRY_DELAY.search(str(exc))
        hint = float(m.group(1) or m.group(2)) if m else None
        return ProviderError("rate_limit", f"rate limited ({msg})", retryable=True, retry_after_s=hint)
    if code in (408, 500, 502, 503, 504):
        return ProviderError("transient", f"transient provider error ({msg})", retryable=True)
    if code in (401, 403):
        return ProviderError("auth", f"provider authentication failed ({msg})", retryable=False)
    return ProviderError("bad_request", f"provider rejected request ({msg})", retryable=False)


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str, model: str, temperature: float, thinking_level: str, timeout_s: float):
        # retry_options stays unset: the SDK then never retries, so ProviderGate owns every retry.
        self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))
        self.model = model
        self.temperature = temperature
        self.thinking_level = thinking_level

    def parameters(self) -> dict:
        return {"temperature": self.temperature, "thinking_level": self.thinking_level}

    def _contents(self, messages: list[dict]) -> list[types.Content]:
        out = []
        for m in messages:
            if m["role"] == "user":
                out.append(types.Content(role="user", parts=[types.Part.from_text(text=m["text"])]))
            elif m["role"] == "model":
                # Replay the provider's own content so Gemini thought signatures survive.
                out.append(m["raw"])
            elif m["role"] == "tool":
                out.append(types.Content(role="user", parts=[
                    types.Part.from_function_response(name=r["name"], response=r["response"]) for r in m["results"]
                ]))
        return out

    async def generate(self, request: ModelRequest) -> ModelResponse:
        config = types.GenerateContentConfig(
            system_instruction=request.system,
            temperature=self.temperature,
            thinking_config=types.ThinkingConfig(thinking_level=self.thinking_level.upper()),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        if request.mode == "tools":
            config.tools = [types.Tool(function_declarations=[types.FunctionDeclaration(
                name=TOOL_NAME,
                description="Return the read-only catalogue fact rows for one brand.",
                parameters_json_schema=tool_declaration_schema(),
            )])]
            config.tool_config = types.ToolConfig(function_calling_config=types.FunctionCallingConfig(
                mode="ANY", allowed_function_names=[TOOL_NAME]))
        else:
            config.response_mime_type = "application/json"
            config.response_json_schema = MODEL_OUTPUT_JSON_SCHEMA
        try:
            resp = await self._client.aio.models.generate_content(
                model=self.model, contents=self._contents(request.messages), config=config)
        except errors.APIError as exc:
            raise classify_api_error(exc) from None
        except httpx.TimeoutException:
            raise ProviderError("timeout", "provider HTTP timeout", retryable=True) from None
        except httpx.TransportError as exc:
            raise ProviderError("transient", f"provider transport error: {type(exc).__name__}", retryable=True) from None
        if not resp.candidates or resp.candidates[0].content is None:
            raise ProviderError("invalid_response", "provider returned no candidate content", retryable=False)
        cand = resp.candidates[0]
        calls = [ToolCall(fc.name or "", dict(fc.args or {}), fc.id) for fc in (resp.function_calls or [])]
        text = None
        if not calls:
            text = "".join(p.text for p in (cand.content.parts or []) if p.text and not p.thought) or None
        u = resp.usage_metadata
        usage = {}
        if u:
            usage = {k: v for k, v in {
                "input": u.prompt_token_count, "output": u.candidates_token_count,
                "reasoning": u.thoughts_token_count, "total": u.total_token_count}.items() if v is not None}
        finish = cand.finish_reason.name if cand.finish_reason else None
        return ModelResponse(text=text, tool_calls=calls, usage=usage, raw_content=cand.content, finish_reason=finish)


GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqProvider:
    """Groq's OpenAI-compatible chat completions API over httpx (no extra SDK)."""

    name = "groq"

    def __init__(self, api_key: str, model: str, temperature: float, reasoning_effort: str, timeout_s: float,
                 transport: httpx.AsyncBaseTransport | None = None):
        self._client = httpx.AsyncClient(timeout=timeout_s, transport=transport,
                                         headers={"Authorization": f"Bearer {api_key}"})
        self.model = model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.last_rate_limits: dict[str, str] = {}

    def parameters(self) -> dict:
        return {"temperature": self.temperature, "reasoning_effort": self._effort()}

    def _effort(self) -> str | None:
        # Only reasoning models accept reasoning_effort.
        return self.reasoning_effort if "gpt-oss" in self.model else None

    @staticmethod
    def _messages(system: str, messages: list[dict]) -> list[dict]:
        out = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["text"]})
            elif m["role"] == "model":
                out.append(m["raw"])
            elif m["role"] == "tool":
                for r in m["results"]:
                    out.append({"role": "tool", "tool_call_id": r["call_id"],
                                "content": json.dumps(r["response"], separators=(",", ":"))})
        return out

    def estimate_tokens(self, request: ModelRequest) -> int:
        """Rough prompt size (about 4 characters per token, measured 4.5 on this data) plus an output allowance."""
        body = self._body(request)
        prompt = len(json.dumps(body["messages"])) + len(json.dumps(body.get("tools") or body.get("response_format")))
        return int(prompt / 4.0) + (150 if request.mode == "tools" else 1200)

    def _body(self, request: ModelRequest) -> dict:
        body: dict[str, Any] = {"model": self.model, "temperature": self.temperature,
                                "messages": self._messages(request.system, request.messages)}
        if self._effort():
            body["reasoning_effort"] = self._effort()
        if request.mode == "tools":
            body["tools"] = [{"type": "function", "function": {
                "name": TOOL_NAME, "description": "Return the read-only catalogue fact rows for one brand.",
                "parameters": tool_declaration_schema()}}]
            body["tool_choice"] = "required"
        else:
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "claims", "schema": MODEL_OUTPUT_JSON_SCHEMA, "strict": True}}
        return body

    async def generate(self, request: ModelRequest) -> ModelResponse:
        try:
            resp = await self._client.post(GROQ_URL, json=self._body(request))
        except httpx.TimeoutException:
            raise ProviderError("timeout", "provider HTTP timeout", retryable=True) from None
        except httpx.TransportError as exc:
            raise ProviderError("transient", f"provider transport error: {type(exc).__name__}", retryable=True) from None
        self.last_rate_limits = {k: v for k, v in resp.headers.items() if k.startswith("x-ratelimit")}
        if resp.status_code != 200:
            raise classify_groq_error(resp)
        try:
            data = resp.json()
            msg = data["choices"][0]["message"]
        except (ValueError, KeyError, IndexError):
            raise ProviderError("invalid_response", "provider returned no choice message", retryable=False) from None
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                args = {"__unparsed__": fn.get("arguments")}
            calls.append(ToolCall(fn.get("name") or "", args if isinstance(args, dict) else {"__value__": args},
                                  tc.get("id")))
        raw = {"role": "assistant", "content": msg.get("content")}
        if msg.get("tool_calls"):
            raw["tool_calls"] = msg["tool_calls"]
        u = data.get("usage") or {}
        usage = {k: v for k, v in {
            "input": u.get("prompt_tokens"), "output": u.get("completion_tokens"),
            "reasoning": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            # Groq caches prompt prefixes automatically; cached tokens do not count toward rate limits.
            "cached_input": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "total": u.get("total_tokens")}.items() if v is not None}
        return ModelResponse(text=None if calls else (msg.get("content") or None), tool_calls=calls, usage=usage,
                             raw_content=raw, finish_reason=data["choices"][0].get("finish_reason"))


def classify_groq_error(resp: httpx.Response) -> ProviderError:
    code = resp.status_code
    try:
        err = resp.json().get("error", {})
    except ValueError:
        err = {}
    detail = f"{code} {err.get('code') or err.get('type') or ''}".strip()
    if code == 429:
        hint = resp.headers.get("retry-after")
        try:
            hint_s = float(hint) if hint is not None else None
        except ValueError:
            hint_s = None
        return ProviderError("rate_limit", f"rate limited ({detail})", retryable=True, retry_after_s=hint_s)
    if code in (408, 500, 502, 503, 504):
        return ProviderError("transient", f"transient provider error ({detail})", retryable=True)
    if code in (401, 403):
        return ProviderError("auth", f"provider authentication failed ({detail})", retryable=False)
    if code == 400 and err.get("code") in ("tool_use_failed", "json_validate_failed"):
        # The model produced output the API itself rejected: malformed output, never retried.
        return ProviderError("invalid_response", f"model output rejected by provider ({detail})", retryable=False)
    return ProviderError("bad_request", f"provider rejected request ({detail})", retryable=False)


def make_provider(settings) -> "Provider":
    import os
    if settings.provider == "groq":
        return GroqProvider(os.environ["GROQ_API_KEY"], settings.model, settings.temperature,
                            settings.thinking_level, settings.request_timeout_s)
    if settings.provider == "gemini":
        return GeminiProvider(os.environ["GEMINI_API_KEY"], settings.model, settings.temperature,
                              settings.thinking_level, settings.request_timeout_s)
    raise ValueError(f"unknown provider {settings.provider!r}")
