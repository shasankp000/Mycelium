from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import requests

import config_loader as cfg


@dataclass
class OllamaConfig:
    """Configuration for an Ollama model.

    Defaults are read from config.toml (via config_loader).
    Environment variables override config.toml values when set.
    """

    model: str = field(default_factory=cfg.ollama_model)
    base_url: str = field(default_factory=cfg.ollama_base_url)
    temperature: float = field(default_factory=cfg.ollama_temperature)
    num_ctx: int = field(default_factory=cfg.ollama_context_size)
    keep_alive: str = field(default_factory=cfg.ollama_keep_alive)
    request_timeout: float = field(default_factory=cfg.ollama_request_timeout)
    extra_options: Dict[str, Any] = field(default_factory=cfg.ollama_extra_options)


@dataclass
class HuggingFaceConfig:
    """Configuration for a Hugging Face text-generation endpoint."""

    api_url: str = field(default_factory=cfg.hf_api_url)
    api_key: str = field(default_factory=cfg.hf_api_key)
    max_new_tokens: int = field(default_factory=cfg.hf_max_new_tokens)
    temperature: float = field(default_factory=cfg.hf_temperature)


@dataclass
class LLMProviderConfig:
    """Unified config for all LLM providers."""

    primary: str = field(default_factory=cfg.llm_primary)
    secondary: str = field(default_factory=cfg.llm_secondary)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)


class OllamaClient:
    def __init__(self, config: Optional[OllamaConfig] = None) -> None:
        self.config = config or OllamaConfig()

    def generate(self, prompt: str, *, system: Optional[str] = None) -> str:
        """Call the local Ollama server using the chat API."""

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

        message = data.get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        return str(data)


class HuggingFaceClient:
    def __init__(self, config: Optional[HuggingFaceConfig] = None) -> None:
        self.config = config or HuggingFaceConfig()

    def generate(self, prompt: str, *, system: Optional[str] = None) -> str:
        if not self.config.api_url:
            raise RuntimeError("huggingface.api_url is not configured in config.toml")

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

        if isinstance(data, dict) and "generated_text" in data:
            return str(data["generated_text"])
        if isinstance(data, list) and data and isinstance(data[0], dict):
            first = data[0]
            if "generated_text" in first:
                return str(first["generated_text"])
        return str(data)


class LLMClient:
    """High-level client that routes between Ollama and Hugging Face."""

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
    """Compatibility helper — still works; now reads config.toml first."""
    return LLMClient()
