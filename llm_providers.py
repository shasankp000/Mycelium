from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import requests


@dataclass
class OllamaConfig:
    """Configuration for an Ollama model.

    All fields can be overridden via environment variables so that
    context size, keep-alive time, etc. are easily tunable.
    """

    model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen2.5:14b"))
    base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    temperature: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
    num_ctx: int = int(os.getenv("OLLAMA_CONTEXT_SIZE", "4096"))
    keep_alive: str = os.getenv("OLLAMA_KEEP_ALIVE", "5m")
    request_timeout: float = float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "60"))
    extra_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HuggingFaceConfig:
    """Configuration for a Hugging Face text-generation endpoint."""

    api_url: str = field(default_factory=lambda: os.getenv("HF_API_URL", ""))
    api_key: str = field(default_factory=lambda: os.getenv("HF_API_KEY", ""))
    max_new_tokens: int = int(os.getenv("HF_MAX_NEW_TOKENS", "512"))
    temperature: float = float(os.getenv("HF_TEMPERATURE", "0.2"))


@dataclass
class LLMProviderConfig:
    """Unified config for all LLM providers.

    In this PoC, `primary` should normally be "ollama" and `secondary`
    "huggingface", but both are configurable.
    """

    primary: str = field(default_factory=lambda: os.getenv("LLM_PRIMARY", "ollama"))
    secondary: str = field(default_factory=lambda: os.getenv("LLM_SECONDARY", "huggingface"))
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)


class OllamaClient:
    def __init__(self, config: Optional[OllamaConfig] = None) -> None:
        self.config = config or OllamaConfig()

    def generate(self, prompt: str, *, system: Optional[str] = None) -> str:
        """Call the local Ollama server using the chat API.

        Exposes context size (num_ctx) and keep-alive via the config.
        """

        url = self.config.base_url.rstrip("/") + "/api/chat"
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        options: Dict[str, Any] = {
            "temperature": self.config.temperature,
            "num_ctx": self.config.num_ctx,
            "keep_alive": self.config.keep_alive,
        }
        options.update(self.config.extra_options)

        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": options,
        }

        resp = requests.post(
            url,
            json=payload,
            timeout=self.config.request_timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        # Chat API response shape: { "message": { "content": "..." }, ... }
        message = data.get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        # Fallback for generate-style APIs or unexpected shapes
        return str(data)


class HuggingFaceClient:
    def __init__(self, config: Optional[HuggingFaceConfig] = None) -> None:
        self.config = config or HuggingFaceConfig()

    def generate(self, prompt: str, *, system: Optional[str] = None) -> str:
        if not self.config.api_url:
            raise RuntimeError("HF_API_URL is not configured")

        headers = {"Accept": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        full_prompt = f"{system}\n\n{prompt}" if system else prompt

        payload = {
            "inputs": full_prompt,
            "parameters": {
                "max_new_tokens": self.config.max_new_tokens,
                "temperature": self.config.temperature,
            },
        }

        resp = requests.post(self.config.api_url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        # Inference API can return either {"generated_text": "..."} or a list
        if isinstance(data, dict) and "generated_text" in data:
            return str(data["generated_text"])
        if isinstance(data, list) and data and isinstance(data[0], dict):
            # Common shape: [{"generated_text": "..."}]
            first = data[0]
            if "generated_text" in first:
                return str(first["generated_text"])
        return str(data)


class LLMClient:
    """High-level client that routes between Ollama and Hugging Face.

    Preferred provider order is controlled by LLMProviderConfig.
    """

    def __init__(self, config: Optional[LLMProviderConfig] = None) -> None:
        self.config = config or LLMProviderConfig()
        self._ollama = OllamaClient(self.config.ollama)
        self._hf = HuggingFaceClient(self.config.huggingface)

    def generate(self, prompt: str, *, system: Optional[str] = None) -> str:
        providers = []
        for name in (self.config.primary, self.config.secondary):
            if name == "ollama":
                providers.append((name, self._ollama))
            elif name == "huggingface":
                providers.append((name, self._hf))

        last_error: Optional[Exception] = None
        for name, client in providers:
            try:
                return client.generate(prompt, system=system)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue

        if last_error is not None:
            raise last_error
        raise RuntimeError("No LLM providers configured")


def default_llm_client_from_env() -> LLMClient:
    """Helper to build an LLMClient using only environment variables."""

    return LLMClient()
