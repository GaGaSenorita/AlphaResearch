"""Generic OpenAI-compatible HTTP client retained for project utilities and tests.

The client intentionally reads credentials only from an environment variable.
No API key is accepted in YAML or written to trajectory files.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol


class ChatClient(Protocol):
    model_name: str

    def chat(self, system: str, user: str) -> str: ...


def completion_url(base_url: str) -> str:
    normalized = str(base_url).strip().rstrip("/")
    if not normalized:
        raise ValueError("LLM base URL is empty")
    if normalized.endswith("/chat/completions"):
        return normalized
    return normalized + "/chat/completions"


def effective_model_name(base_url: str, model_name: str) -> str:
    """Translate one provider-qualified alias at the official DeepSeek edge.

    Some model catalogs spell the model ``openai/deepseek-v4-pro``.  The
    official DeepSeek OpenAI-compatible endpoint rejects that catalog prefix
    and accepts ``deepseek-v4-pro``.  Keep the requested name in configuration
    and run naming, but send the provider's actual wire name.
    """

    requested = str(model_name).strip()
    hostname = urllib.parse.urlparse(str(base_url).strip()).hostname
    if hostname == "api.deepseek.com" and requested == "openai/deepseek-v4-pro":
        return "deepseek-v4-pro"
    return requested


@dataclass
class OpenAICompatibleChatClient:
    base_url: str
    model_name: str
    api_key_env: str = "ALPHARESEARCH_LLM_API_KEY"
    temperature: float = 0.7
    timeout_seconds: float = 120.0
    max_tokens: int | None = 2048
    reasoning_effort: str | None = None
    thinking_enabled: bool | None = None
    json_mode: bool = False
    max_retries: int = 3

    def chat(self, system: str, user: str) -> str:
        api_key = os.environ.get(self.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                f"missing LLM credential: export {self.api_key_env}=<key> before the run"
            )
        payload: dict[str, Any] = {
            "model": effective_model_name(self.base_url, self.model_name),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": float(self.temperature),
            "stream": False,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = int(self.max_tokens)
        if self.reasoning_effort and self.reasoning_effort != "none":
            payload["reasoning_effort"] = self.reasoning_effort
        if self.thinking_enabled is not None:
            payload["thinking"] = {
                "type": "enabled" if self.thinking_enabled else "disabled"
            }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}

        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            completion_url(self.base_url),
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(max(1, int(self.max_retries))):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    data = json.loads(response.read().decode("utf-8"))
                choices = data.get("choices") if isinstance(data, dict) else None
                if not isinstance(choices, list) or not choices:
                    raise RuntimeError("LLM response did not contain choices")
                message = choices[0].get("message", {})
                content = message.get("content") if isinstance(message, dict) else None
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("LLM returned empty message content")
                return content.strip()
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:800]
                last_error = RuntimeError(f"LLM HTTP {exc.code}: {detail}")
                if exc.code < 500 and exc.code != 429:
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
                last_error = exc
            if attempt + 1 < max(1, int(self.max_retries)):
                time.sleep(min(8.0, 1.5 * (2**attempt)))
        raise RuntimeError(f"LLM request failed after {self.max_retries} attempt(s): {last_error}")


DEFAULT_MOCK_CANDIDATES = (
    {
        "name": "NormalizedMomentum20",
        "expression": "Div(Delta($close,20),Add(Std($close,20),1e-12))",
        "reason": "Volatility-normalized medium-term price momentum.",
    },
    {
        "name": "ShortReversal5",
        "expression": "Mul(-1,Div(Delta($close,5),Add(Std($close,20),1e-12)))",
        "reason": "Short-horizon reversal normalized by recent volatility.",
    },
    {
        "name": "VolumeAcceleration",
        "expression": "Div(Mean($volume,5),Add(Mean($volume,20),1e-12))",
        "reason": "Detects recent volume expansion relative to its monthly baseline.",
    },
    {
        "name": "RangeCompression",
        "expression": "Mul(-1,Div(Mean(Sub($high,$low),5),Add(Mean($close,20),1e-12)))",
        "reason": "Rewards low recent intraday range relative to the price level.",
    },
    {
        "name": "PriceVolumeDecoupling",
        "expression": "Mul(-1,Corr($close,$volume,10))",
        "reason": "Captures short-term price-volume decoupling.",
    },
)


@dataclass
class MockChatClient:
    """Deterministic no-network client used for pipeline tests."""

    model_name: str = "mock-alphabench-llm"
    candidates: tuple[dict[str, str], ...] = DEFAULT_MOCK_CANDIDATES
    calls: list[dict[str, str]] = field(default_factory=list)
    _index: int = 0

    def chat(self, system: str, user: str) -> str:
        candidate = self.candidates[self._index % len(self.candidates)]
        self._index += 1
        response = json.dumps(candidate, ensure_ascii=False)
        self.calls.append({"system": system, "user": user, "response": response})
        return response
