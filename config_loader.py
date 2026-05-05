"""
config_loader.py

Single source of truth for reading Mycelium configuration.

Resolution order for every value
---------------------------------
1. Environment variable  (highest priority – lets CI/Docker override)
2. config.toml           (the normal way to change settings)
3. Hard-coded default    (lowest priority – guarantees a working value)

All accessor functions are thin wrappers so callsites look like::

    from config_loader import cfg
    model = cfg.ollama_model()

The module-level ``cfg`` singleton re-reads ``config.toml`` on every call
(TOML parsing is fast; a stat+read of a small file is negligible vs. any
LLM call).  This means you never need to restart the server after editing
the file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Python 3.11+ ships tomllib in the stdlib; older versions need the backport.
try:
    import tomllib  # type: ignore[import]
except ModuleNotFoundError:  # Python < 3.11
    try:
        import tomli as tomllib  # type: ignore[import,no-redef]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "config_loader requires 'tomli' on Python < 3.11. "
            "Run: pip install tomli"
        ) from exc

_CONFIG_PATH = Path(__file__).parent / "config.toml"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load() -> Dict[str, Any]:
    """Parse config.toml and return the raw dict. Returns {} if absent."""
    if not _CONFIG_PATH.exists():
        return {}
    with _CONFIG_PATH.open("rb") as fh:
        return tomllib.load(fh)


def _get(section: str, key: str, default: Any, env_var: Optional[str] = None) -> Any:
    """Resolve a single setting using the three-tier priority chain."""
    # 1. Environment variable
    if env_var:
        val = os.environ.get(env_var)
        if val is not None:
            # Coerce to the same type as the default so callers get the right type.
            return _coerce(val, type(default))
    # 2. config.toml
    data = _load()
    try:
        return data[section][key]
    except KeyError:
        pass
    # 3. Hard-coded default
    return default


def _coerce(value: str, target_type: type) -> Any:
    """Best-effort string → target type coercion for env-var values."""
    if target_type is bool:
        return value.lower() in ("1", "true", "yes")
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    return value  # str or anything else – return as-is


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

def ollama_model() -> str:
    return _get("ollama", "model", "qwen2.5:14b", "OLLAMA_MODEL")

def ollama_base_url() -> str:
    return _get("ollama", "base_url", "http://localhost:11434", "OLLAMA_BASE_URL")

def ollama_temperature() -> float:
    return _get("ollama", "temperature", 0.2, "OLLAMA_TEMPERATURE")

def ollama_context_size() -> int:
    return _get("ollama", "context_size", 4096, "OLLAMA_CONTEXT_SIZE")

def ollama_keep_alive() -> str:
    return _get("ollama", "keep_alive", "5m", "OLLAMA_KEEP_ALIVE")

def ollama_request_timeout() -> int:
    return _get("ollama", "request_timeout", 60, "OLLAMA_REQUEST_TIMEOUT")

def ollama_extra_options() -> Dict[str, Any]:
    """Return the [ollama.extra_options] sub-table, or {} if absent."""
    data = _load()
    return data.get("ollama", {}).get("extra_options", {})


# ---------------------------------------------------------------------------
# HuggingFace
# ---------------------------------------------------------------------------

def hf_api_url() -> str:
    return _get("huggingface", "api_url", "", "HF_API_URL")

def hf_api_key() -> str:
    return _get("huggingface", "api_key", "", "HF_API_KEY")

def hf_max_new_tokens() -> int:
    return _get("huggingface", "max_new_tokens", 512, "HF_MAX_NEW_TOKENS")

def hf_temperature() -> float:
    return _get("huggingface", "temperature", 0.2, "HF_TEMPERATURE")


# ---------------------------------------------------------------------------
# LLM routing
# ---------------------------------------------------------------------------

def llm_primary() -> str:
    return _get("llm", "primary", "ollama")

def llm_secondary() -> str:
    return _get("llm", "secondary", "huggingface")


# ---------------------------------------------------------------------------
# Multi-lens routing thresholds
# ---------------------------------------------------------------------------

def fusion_weights() -> Dict[str, float]:
    """Return the three fusion weights as a dict."""
    return {
        "semantic":   _get("routing", "semantic_weight",   0.45, "SEMANTIC_WEIGHT"),
        "spectral":   _get("routing", "spectral_weight",   0.35, "SPECTRAL_WEIGHT"),
        "confidence": _get("routing", "confidence_weight", 0.20, "CONFIDENCE_WEIGHT"),
    }

def superposition_variance_threshold() -> float:
    return _get("routing", "superposition_variance_threshold", 0.08,
                "SUPERPOSITION_VARIANCE_THRESHOLD")

def domain_score_threshold() -> float:
    return _get("routing", "domain_score_threshold", 0.45, "DOMAIN_SCORE_THRESHOLD")

def enable_attribute_override() -> bool:
    return _get("routing", "enable_attribute_override", True, "ENABLE_ATTRIBUTE_OVERRIDE")


# ---------------------------------------------------------------------------
# Expert selection
# ---------------------------------------------------------------------------

def coverage_threshold() -> float:
    return _get("experts", "coverage_threshold", 0.70, "COVERAGE_THRESHOLD")

def soft_stop_percentage() -> float:
    return _get("experts", "soft_stop_pct", 0.90, "SOFT_STOP_PERCENTAGE")

def max_experts() -> int:
    return _get("experts", "max_experts", 3, "MAX_EXPERTS")

def expert_similarity_threshold() -> float:
    return _get("experts", "similarity_threshold", 0.45, "EXPERT_SIMILARITY_THRESHOLD")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def enable_logging() -> bool:
    return _get("logging", "enable", True, "ENABLE_LOGGING")

def log_sample_rate() -> int:
    return _get("logging", "sample_rate", 1, "LOG_SAMPLE_RATE")


# ---------------------------------------------------------------------------
# Patch-batch logger
# ---------------------------------------------------------------------------

def patch_batch_dir() -> str:
    return _get("patch_batches", "dir", "patch_batches", "PATCH_BATCH_DIR")


# ---------------------------------------------------------------------------
# Traces
# ---------------------------------------------------------------------------

def traces_dir() -> str:
    return _get("traces", "dir", "traces", "TRACES_DIR")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def api_allowed_origins() -> List[str]:
    data = _load()
    return data.get("api", {}).get(
        "allowed_origins",
        ["http://localhost:3000", "http://127.0.0.1:3000"],
    )


# ---------------------------------------------------------------------------
# Convenience singleton so callers can do `from config_loader import cfg`
# ---------------------------------------------------------------------------

class _Cfg:
    """Thin namespace so callers can do ``cfg.ollama_model()`` if preferred."""
    ollama_model                    = staticmethod(ollama_model)
    ollama_base_url                 = staticmethod(ollama_base_url)
    ollama_temperature              = staticmethod(ollama_temperature)
    ollama_context_size             = staticmethod(ollama_context_size)
    ollama_keep_alive               = staticmethod(ollama_keep_alive)
    ollama_request_timeout          = staticmethod(ollama_request_timeout)
    ollama_extra_options            = staticmethod(ollama_extra_options)
    hf_api_url                      = staticmethod(hf_api_url)
    hf_api_key                      = staticmethod(hf_api_key)
    hf_max_new_tokens               = staticmethod(hf_max_new_tokens)
    hf_temperature                  = staticmethod(hf_temperature)
    llm_primary                     = staticmethod(llm_primary)
    llm_secondary                   = staticmethod(llm_secondary)
    fusion_weights                  = staticmethod(fusion_weights)
    superposition_variance_threshold= staticmethod(superposition_variance_threshold)
    domain_score_threshold          = staticmethod(domain_score_threshold)
    enable_attribute_override       = staticmethod(enable_attribute_override)
    coverage_threshold              = staticmethod(coverage_threshold)
    soft_stop_percentage            = staticmethod(soft_stop_percentage)
    max_experts                     = staticmethod(max_experts)
    expert_similarity_threshold     = staticmethod(expert_similarity_threshold)
    enable_logging                  = staticmethod(enable_logging)
    log_sample_rate                 = staticmethod(log_sample_rate)
    patch_batch_dir                 = staticmethod(patch_batch_dir)
    traces_dir                      = staticmethod(traces_dir)
    api_allowed_origins             = staticmethod(api_allowed_origins)


cfg = _Cfg()
