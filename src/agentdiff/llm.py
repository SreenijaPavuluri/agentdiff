"""Minimal OpenAI-compatible chat client (Ollama, OpenRouter, vLLM, OpenAI).

Only used by integrations and the optional cluster namer / LLM-judge baseline; the core
library never calls a model. Keys come from environment variables only.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass

import httpx

from agentdiff.tape import LLMReply

PRESETS = {
    "ollama": ("http://localhost:11434/v1", None),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
}


class LLMUnavailable(RuntimeError):
    pass


@dataclass
class ChatModel:
    model: str
    provider: str = "ollama"
    base_url: str | None = None
    api_key_env: str | None = None
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout_s: float = 300.0
    max_retries: int = 4

    def __post_init__(self) -> None:
        url, env = PRESETS.get(self.provider, (None, None))
        self.base_url = self.base_url or os.environ.get("AGENTDIFF_BASE_URL") or url
        self.api_key_env = self.api_key_env or env
        if not self.base_url:
            raise ValueError(f"no base_url for provider {self.provider!r}")
        self._client = httpx.Client(timeout=self.timeout_s)

    def complete(self, system: str, user: str, *, temperature: float | None = None, seed: int | None = None) -> LLMReply:
        headers = {"Content-Type": "application/json"}
        if self.api_key_env:
            key = os.environ.get(self.api_key_env)
            if not key:
                raise LLMUnavailable(f"missing {self.api_key_env}")
            headers["Authorization"] = f"Bearer {key}"
        body: dict[str, object] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens,
        }
        if seed is not None:
            body["seed"] = seed
        last = ""
        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            try:
                r = self._client.post(f"{self.base_url}/chat/completions", json=body, headers=headers)
                if r.status_code in (408, 429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(f"{r.status_code}: {r.text[:200]}", request=r.request, response=r)
                r.raise_for_status()
                data = r.json()
                if "choices" not in data or not data["choices"]:
                    raise ValueError(f"empty response: {str(data)[:200]}")
                text = data["choices"][0]["message"].get("content") or ""
                usage = data.get("usage") or {}
                return LLMReply(
                    text=text, model=data.get("model", self.model),
                    tokens_in=int(usage.get("prompt_tokens", 0)), tokens_out=int(usage.get("completion_tokens", 0)),
                    latency_ms=(time.perf_counter() - started) * 1000,
                )  # fmt: skip
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                last = f"{type(exc).__name__}: {exc}"
                if attempt < self.max_retries:
                    time.sleep(min(30.0, 2.0**attempt) + random.random())
        raise LLMUnavailable(last)
