"""
Sovereign Core — Multi-Backend LLM Adapter
Supports: local tri-GPU mesh | Anthropic | OpenAI | mock

GPU Mesh:
  RTX 5050  → Qwen2.5-32B-AWQ  → :8001  (logic, generation)
  Radeon 780M → DeepSeek-Coder → :8002  (reasoning, code)
  Ryzen 7 CPU → Llama-3.2-3B   → :8003  (orchestration, routing)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Optional

import httpx

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    LOCAL = "local"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    MOCK = "mock"


class LocalEngine(str, Enum):
    """Tri-GPU inference endpoint routing."""
    LOGIC = "logic"          # RTX 5050 → Qwen2.5-32B-AWQ → :8001
    REASONING = "reasoning"  # Radeon 780M → DeepSeek-Coder → :8002
    ORCHESTRATOR = "orchestrator"  # Ryzen 7 CPU → Llama-3.2-3B → :8003


@dataclass
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    engine: str = ""
    cost_usd: float = 0.0  # Always 0.0 for local inference


@dataclass
class LLMResponse:
    content: str
    usage: LLMUsage
    model: str
    provider: str


class LocalMeshAdapter:
    """Routes requests across the tri-GPU inference mesh."""

    ENGINE_ENDPOINTS = {
        LocalEngine.LOGIC: "http://localhost:8001/v1",
        LocalEngine.REASONING: "http://localhost:8002/v1",
        LocalEngine.ORCHESTRATOR: "http://localhost:8003/v1",
    }

    ENGINE_MODELS = {
        LocalEngine.LOGIC: "Qwen2.5-32B-AWQ",
        LocalEngine.REASONING: "DeepSeek-Coder",
        LocalEngine.ORCHESTRATOR: "Llama-3.2-3B",
    }

    def __init__(self, timeout: float = 120.0):
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    def _select_engine(self, task_hint: str) -> LocalEngine:
        """Route task to optimal GPU based on task type."""
        hint = task_hint.lower()
        if any(k in hint for k in ("code", "debug", "implement", "function", "class", "bug")):
            return LocalEngine.REASONING
        if any(k in hint for k in ("route", "orchestrate", "classify", "quick", "short")):
            return LocalEngine.ORCHESTRATOR
        return LocalEngine.LOGIC  # Default: RTX 5050 for heavy generation

    async def complete(
        self,
        messages: list[dict],
        engine: Optional[LocalEngine] = None,
        task_hint: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs,
    ) -> LLMResponse:
        if engine is None:
            engine = self._select_engine(task_hint)

        endpoint = self.ENGINE_ENDPOINTS[engine]
        model = self.ENGINE_MODELS[engine]
        url = f"{endpoint}/chat/completions"

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }

        t0 = time.monotonic()
        try:
            resp = await self.client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            logger.error(f"Local mesh error [{engine.value}@{endpoint}]: {exc}")
            raise

        latency_ms = (time.monotonic() - t0) * 1000
        choice = data["choices"][0]["message"]["content"]
        usage_data = data.get("usage", {})

        return LLMResponse(
            content=choice,
            usage=LLMUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
                latency_ms=latency_ms,
                engine=engine.value,
                cost_usd=0.0,  # Zero-cost local inference
            ),
            model=model,
            provider="local",
        )

    async def stream(
        self,
        messages: list[dict],
        engine: Optional[LocalEngine] = None,
        task_hint: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        if engine is None:
            engine = self._select_engine(task_hint)

        endpoint = self.ENGINE_ENDPOINTS[engine]
        model = self.ENGINE_MODELS[engine]
        url = f"{endpoint}/chat/completions"

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        async with self.client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except (json.JSONDecodeError, KeyError):
                        continue

    async def health_check(self) -> dict[str, bool]:
        """Verify all three GPU endpoints are reachable."""
        results = {}
        for engine, endpoint in self.ENGINE_ENDPOINTS.items():
            try:
                resp = await self.client.get(f"{endpoint}/models", timeout=5.0)
                results[engine.value] = resp.status_code == 200
            except Exception:
                results[engine.value] = False
        return results

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()


class LLMAdapter:
    """
    Unified LLM adapter. Sovereign Core first tries local mesh,
    falls back to cloud provider only if explicitly configured.
    """

    def __init__(
        self,
        provider: LLMProvider = LLMProvider.LOCAL,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self._local: Optional[LocalMeshAdapter] = None

        if provider == LLMProvider.LOCAL:
            self._local = LocalMeshAdapter()

    @property
    def local(self) -> LocalMeshAdapter:
        if self._local is None:
            self._local = LocalMeshAdapter()
        return self._local

    async def complete(
        self,
        messages: list[dict],
        task_hint: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        engine: Optional[LocalEngine] = None,
        **kwargs,
    ) -> LLMResponse:
        if self.provider == LLMProvider.LOCAL:
            return await self.local.complete(
                messages, engine=engine, task_hint=task_hint,
                max_tokens=max_tokens, temperature=temperature, **kwargs
            )
        elif self.provider == LLMProvider.MOCK:
            return self._mock_response(messages)
        elif self.provider == LLMProvider.ANTHROPIC:
            return await self._anthropic_complete(messages, max_tokens, temperature)
        elif self.provider == LLMProvider.OPENAI:
            return await self._openai_complete(messages, max_tokens, temperature)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    async def stream(
        self,
        messages: list[dict],
        task_hint: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        engine: Optional[LocalEngine] = None,
    ) -> AsyncIterator[str]:
        if self.provider == LLMProvider.LOCAL:
            async for chunk in self.local.stream(
                messages, engine=engine, task_hint=task_hint,
                max_tokens=max_tokens, temperature=temperature
            ):
                yield chunk
        elif self.provider == LLMProvider.MOCK:
            yield "Mock streaming response from Sovereign Core local mesh."
        else:
            raise NotImplementedError(f"Streaming not implemented for {self.provider}")

    def _mock_response(self, messages: list[dict]) -> LLMResponse:
        last_msg = messages[-1].get("content", "") if messages else ""
        return LLMResponse(
            content=f"[MOCK] Sovereign Core local mesh response to: {last_msg[:100]}",
            usage=LLMUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30, cost_usd=0.0),
            model="mock-sovereign",
            provider="mock",
        )

    async def _anthropic_complete(
        self, messages: list[dict], max_tokens: int, temperature: float
    ) -> LLMResponse:
        """Cloud fallback — Anthropic API."""
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=self.api_key)
        model = self.model or "claude-sonnet-4-20250514"

        system_msgs = [m for m in messages if m.get("role") == "system"]
        user_msgs = [m for m in messages if m.get("role") != "system"]
        system = system_msgs[0]["content"] if system_msgs else None

        t0 = time.monotonic()
        resp = await client.messages.create(
            model=model,
            messages=user_msgs,
            system=system,
            max_tokens=max_tokens,
        )
        latency_ms = (time.monotonic() - t0) * 1000

        return LLMResponse(
            content=resp.content[0].text,
            usage=LLMUsage(
                prompt_tokens=resp.usage.input_tokens,
                completion_tokens=resp.usage.output_tokens,
                total_tokens=resp.usage.input_tokens + resp.usage.output_tokens,
                latency_ms=latency_ms,
                engine="anthropic",
                cost_usd=(resp.usage.input_tokens * 3e-6 + resp.usage.output_tokens * 15e-6),
            ),
            model=model,
            provider="anthropic",
        )

    async def _openai_complete(
        self, messages: list[dict], max_tokens: int, temperature: float
    ) -> LLMResponse:
        """Cloud fallback — OpenAI API."""
        import openai
        client = openai.AsyncOpenAI(api_key=self.api_key)
        model = self.model or "gpt-4o"

        t0 = time.monotonic()
        resp = await client.chat.completions.create(
            model=model, messages=messages,
            max_tokens=max_tokens, temperature=temperature,
        )
        latency_ms = (time.monotonic() - t0) * 1000

        return LLMResponse(
            content=resp.choices[0].message.content,
            usage=LLMUsage(
                prompt_tokens=resp.usage.prompt_tokens,
                completion_tokens=resp.usage.completion_tokens,
                total_tokens=resp.usage.total_tokens,
                latency_ms=latency_ms,
                engine="openai",
            ),
            model=model,
            provider="openai",
        )

    async def mesh_health(self) -> dict:
        """Report health of all local inference endpoints."""
        if self.provider != LLMProvider.LOCAL:
            return {"status": "cloud_provider", "provider": self.provider.value}
        return await self.local.health_check()

    async def close(self):
        if self._local:
            await self._local.close()
