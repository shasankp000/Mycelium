"""Load Mycelium runtime configuration from config.toml.

Resolution order (highest priority first):
  1. Environment variable (e.g. OLLAMA_MODEL)
  2. config.toml value
  3. Hard-coded default inside each dataclass

This means you can still override individual settings with env vars
without touching config.toml — useful for CI/CD and Docker deployments.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Could not import a TOML library.  "
            "On Python < 3.11 install tomli: pip install tomli"
        ) from exc

_CONFIG_PATH = Path(__file__).parent / "config.toml"

_cached: Dict[str, Any] | None = None


def _load_raw() -> Dict[str, Any]:
    global _cached
    if _cached is not None:
        return _cached
    if _CONFIG_PATH.exists():
        with _CONFIG_PATH.open("rb") as f:
            _cached = tomllib.load(f)
    else:
        _cached = {}
    return _cached


def _section(name: str) -> Dict[str, Any]:
    return _load_raw().get(name, {})


def _str(section: str, key: str, env_var: str, default: str) -> str:
    env = os.getenv(env_var)
    if env is not None:
        return env
    return str(_section(section).get(key, default))


def _int(section: str, key: str, env_var: str, default: int) -> int:
    env = os.getenv(env_var)
    if env is not None:
        return int(env)
    raw = _section(section).get(key)
    return int(raw) if raw is not None else default


def _float(section: str, key: str, env_var: str, default: float) -> float:
    env = os.getenv(env_var)
    if env is not None:
        return float(env)
    raw = _section(section).get(key)
    return float(raw) if raw is not None else default


# ── Public accessors ──────────────────────────────────────────────────────────

def ollama_model() -> str:
    return _str("ollama", "model", "OLLAMA_MODEL", "qwen2.5:14b")


def ollama_base_url() -> str:
    return _str("ollama", "base_url", "OLLAMA_BASE_URL", "http://localhost:11434")


def ollama_temperature() -> float:
    return _float("ollama", "temperature", "OLLAMA_TEMPERATURE", 0.2)


def ollama_context_size() -> int:
    return _int("ollama", "context_size", "OLLAMA_CONTEXT_SIZE", 4096)


def ollama_keep_alive() -> str:
    return _str("ollama", "keep_alive", "OLLAMA_KEEP_ALIVE", "5m")


def ollama_request_timeout() -> float:
    return _float("ollama", "request_timeout", "OLLAMA_REQUEST_TIMEOUT", 60.0)


def ollama_extra_options() -> Dict[str, Any]:
    """Return the [ollama.extra_options] table (empty dict if absent)."""
    ollama_section = _section("ollama")
    return dict(ollama_section.get("extra_options", {}))


def hf_api_url() -> str:
    return _str("huggingface", "api_url", "HF_API_URL", "")


def hf_api_key() -> str:
    return _str("huggingface", "api_key", "HF_API_KEY", "")


def hf_max_new_tokens() -> int:
    return _int("huggingface", "max_new_tokens", "HF_MAX_NEW_TOKENS", 512)


def hf_temperature() -> float:
    return _float("huggingface", "temperature", "HF_TEMPERATURE", 0.2)


def llm_primary() -> str:
    return _str("llm", "primary", "LLM_PRIMARY", "ollama")


def llm_secondary() -> str:
    return _str("llm", "secondary", "LLM_SECONDARY", "huggingface")


def traces_dir() -> str:
    return _str("traces", "dir", "MYCELIUM_TRACES_DIR", "traces")


def api_allowed_origins() -> list[str]:
    env = os.getenv("MYCELIUM_ALLOWED_ORIGINS")
    if env:
        return [o.strip() for o in env.split(",")]
    raw = _section("api").get("allowed_origins", [])
    return list(raw)
