"""
config_loader.py — single source of truth for Mycelium runtime configuration.

Resolution order for every value (highest wins):
  1. Matching environment variable (explicit key or MYCELIUM_*)
  2. config.toml in the repository root
  3. Hard-coded default

Usage
-----
    import config_loader as cfg

    model   = cfg.get("layer1", "embed_model")              # -> "all-MiniLM-L6-v2"
    top_k   = cfg.get_int("layer1", "lens1_top_k")          # -> 3
    weight  = cfg.get_float("layer1", "lens1_embedding_weight")  # -> 0.8
    domains = cfg.get_list("layer1.domain_list", "domains")      # -> [...]

The module-level ``cfg`` singleton re-reads ``config.toml`` on every call
(TOML parsing is fast; a stat+read of a small file is negligible vs. any
LLM call). This means you never need to restart the server after editing
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


def _env_key(section: str, key: str) -> str:
    """Map (section, key) -> MYCELIUM_SECTION_KEY (uppercase, dots → underscores)."""
    return "MYCELIUM_" + (section + "_" + key).upper().replace(".", "_")


def _coerce(value: str, target_type: type) -> Any:
    """Best-effort string → target type coercion for env-var values."""
    if target_type is bool:
        return value.lower() in ("1", "true", "yes")
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    return value  # str or anything else – return as-is


def _resolve_env(
    default: Any,
    env_var: Optional[str],
    section: str,
    key: str,
) -> Optional[Any]:
    if env_var:
        val = os.environ.get(env_var)
        if val is not None:
            return _coerce(val, type(default) if default is not None else str)
    env_val = os.environ.get(_env_key(section, key))
    if env_val is not None:
        return _coerce(env_val, type(default) if default is not None else str)
    return None


def _get(section: str, key: str, default: Any, env_var: Optional[str] = None) -> Any:
    """Resolve a single setting using env vars, config.toml, then default."""
    env_val = _resolve_env(default, env_var, section, key)
    if env_val is not None:
        return env_val

    data = _load()
    node: Any = data
    for part in section.split("."):
        if not isinstance(node, dict):
            return default
        if part not in node:
            return default
        node = node[part]
    if isinstance(node, dict) and key in node:
        return node[key]
    return default


# ---------------------------------------------------------------------------
# Generic accessors (mirror the origin/main API)
# ---------------------------------------------------------------------------

def get(section: str, key: str, default: Any = None) -> Any:
    return _get(section, key, default)


def get_int(section: str, key: str, default: int = 0) -> int:
    return int(get(section, key, default))


def get_float(section: str, key: str, default: float = 0.0) -> float:
    return float(get(section, key, default))


def get_bool(section: str, key: str, default: bool = False) -> bool:
    val = get(section, key, default)
    if isinstance(val, bool):
        return val
    return str(val).lower() in ("1", "true", "yes")


def get_list(section: str, key: str, default: Optional[List] = None) -> List:
    val = get(section, key, default)
    if val is None:
        return [] if default is None else default
    if isinstance(val, list):
        return val
    return [item.strip() for item in str(val).split(",") if item.strip()]


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
    return _get(
        "routing",
        "superposition_variance_threshold",
        0.08,
        "SUPERPOSITION_VARIANCE_THRESHOLD",
    )


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
    return _get(
        "experts",
        "similarity_threshold",
        0.45,
        "EXPERT_SIMILARITY_THRESHOLD",
    )


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
    return get_list(
        "api",
        "allowed_origins",
        ["http://localhost:3000", "http://127.0.0.1:3000"],
    )


# ---------------------------------------------------------------------------
# Convenience accessors — layer1
# ---------------------------------------------------------------------------

def layer1_embed_model() -> str:
    return get("layer1", "embed_model", "all-MiniLM-L6-v2")


def layer1_lens1_top_k() -> int:
    return get_int("layer1", "lens1_top_k", 3)


def layer1_lens1_embedding_weight() -> float:
    return get_float("layer1", "lens1_embedding_weight", 0.8)


def layer1_lens1_lexical_weight() -> float:
    return get_float("layer1", "lens1_lexical_weight", 0.2)


def layer1_tag_cluster_distance_threshold() -> float:
    return get_float("layer1", "tag_cluster_distance_threshold", 0.3)


def layer1_tag_cluster_similarity_threshold() -> float:
    return get_float("layer1", "tag_cluster_similarity_threshold", 0.7)


def layer1_tag_normalize_threshold() -> int:
    return get_int("layer1", "tag_normalize_threshold", 80)


def layer1_temporal_window_hours() -> float:
    return get_float("layer1", "temporal_window_hours", 1.0)


def layer1_temporal_max_size() -> int:
    return get_int("layer1", "temporal_max_size", 100)


def layer1_temporal_time_window_hours() -> float:
    return get_float("layer1", "temporal_time_window_hours", 24.0)


def layer1_spatial_dominance_threshold() -> float:
    return get_float("layer1", "spatial_dominance_threshold", 0.3)


def layer1_tag_clusters_filename() -> str:
    return get("layer1", "tag_clusters_filename", "tag_clusters.json")


def layer1_domain_list() -> List[str]:
    return get_list(
        "layer1.domain_list",
        "domains",
        [
            "AI", "healthcare", "logistics", "finance", "education",
            "physics", "maths", "biology", "chemistry", "technology",
            "sports", "politics", "history", "art", "music", "literature",
        ],
    )


def layer1_llama_model() -> str:
    return get("layer1.llama", "model", "llama3:8b")


def layer1_openai_model() -> str:
    return get("layer1.openai", "model", "gpt-3.5-turbo")


def layer1_openai_endpoint() -> str:
    return get("layer1.openai", "endpoint", "https://api.openai.com/v1/chat/completions")


def layer1_openai_provider() -> str:
    return get("layer1.openai", "provider", "openai")


def layer1_openai_max_tokens() -> int:
    return get_int("layer1.openai", "max_tokens", 50)


def layer1_openai_temperature() -> float:
    return get_float("layer1.openai", "temperature", 0.2)


# ---------------------------------------------------------------------------
# Convenience accessors — unified_expert
# ---------------------------------------------------------------------------

def ue_ood_svm_distance_threshold() -> float:
    return get_float("unified_expert", "ood_svm_distance_threshold", 0.3)


def ue_ood_nn_distance_threshold() -> float:
    return get_float("unified_expert", "ood_nn_distance_threshold", 0.65)


def ue_ood_isolation_score_threshold() -> float:
    return get_float("unified_expert", "ood_isolation_score_threshold", -0.05)


def ue_ood_votes_required() -> int:
    return get_int("unified_expert", "ood_votes_required", 2)


def ue_composite_weight_similarity() -> float:
    return get_float("unified_expert", "composite_weight_similarity", 0.45)


def ue_composite_weight_confidence() -> float:
    return get_float("unified_expert", "composite_weight_confidence", 0.45)


def ue_composite_weight_ood() -> float:
    return get_float("unified_expert", "composite_weight_ood", 0.10)


def ue_decision_high_threshold() -> float:
    return get_float("unified_expert", "decision_high_threshold", 0.5)


def ue_decision_mid_threshold() -> float:
    return get_float("unified_expert", "decision_mid_threshold", 0.25)


def ue_ood_penalty_detected() -> float:
    return get_float("unified_expert", "ood_penalty_detected", 0.5)


def ue_ood_penalty_not_detected() -> float:
    return get_float("unified_expert", "ood_penalty_not_detected", 0.2)


def ue_ood_tolerance_use_existing() -> float:
    return get_float("unified_expert", "ood_tolerance_use_existing", 0.45)


def ue_ood_rejection_threshold() -> float:
    return get_float("unified_expert", "ood_rejection_threshold", 0.6)


# ---------------------------------------------------------------------------
# Convenience accessors — workflow
# ---------------------------------------------------------------------------

def workflow_traces_dir() -> str:
    return get("workflow", "traces_dir", "traces")


# ---------------------------------------------------------------------------
# Convenience singleton so callers can do `from config_loader import cfg`
# ---------------------------------------------------------------------------

class _Cfg:
    """Thin namespace so callers can do ``cfg.ollama_model()`` if preferred."""

    get                            = staticmethod(get)
    get_int                        = staticmethod(get_int)
    get_float                      = staticmethod(get_float)
    get_bool                       = staticmethod(get_bool)
    get_list                       = staticmethod(get_list)
    ollama_model                   = staticmethod(ollama_model)
    ollama_base_url                = staticmethod(ollama_base_url)
    ollama_temperature             = staticmethod(ollama_temperature)
    ollama_context_size            = staticmethod(ollama_context_size)
    ollama_keep_alive              = staticmethod(ollama_keep_alive)
    ollama_request_timeout         = staticmethod(ollama_request_timeout)
    ollama_extra_options           = staticmethod(ollama_extra_options)
    hf_api_url                     = staticmethod(hf_api_url)
    hf_api_key                     = staticmethod(hf_api_key)
    hf_max_new_tokens              = staticmethod(hf_max_new_tokens)
    hf_temperature                 = staticmethod(hf_temperature)
    llm_primary                    = staticmethod(llm_primary)
    llm_secondary                  = staticmethod(llm_secondary)
    fusion_weights                 = staticmethod(fusion_weights)
    superposition_variance_threshold = staticmethod(superposition_variance_threshold)
    domain_score_threshold         = staticmethod(domain_score_threshold)
    enable_attribute_override      = staticmethod(enable_attribute_override)
    coverage_threshold             = staticmethod(coverage_threshold)
    soft_stop_percentage           = staticmethod(soft_stop_percentage)
    max_experts                    = staticmethod(max_experts)
    expert_similarity_threshold    = staticmethod(expert_similarity_threshold)
    enable_logging                 = staticmethod(enable_logging)
    log_sample_rate                = staticmethod(log_sample_rate)
    patch_batch_dir                = staticmethod(patch_batch_dir)
    traces_dir                     = staticmethod(traces_dir)
    api_allowed_origins            = staticmethod(api_allowed_origins)
    layer1_embed_model             = staticmethod(layer1_embed_model)
    layer1_lens1_top_k             = staticmethod(layer1_lens1_top_k)
    layer1_lens1_embedding_weight  = staticmethod(layer1_lens1_embedding_weight)
    layer1_lens1_lexical_weight    = staticmethod(layer1_lens1_lexical_weight)
    layer1_tag_cluster_distance_threshold = staticmethod(layer1_tag_cluster_distance_threshold)
    layer1_tag_cluster_similarity_threshold = staticmethod(layer1_tag_cluster_similarity_threshold)
    layer1_tag_normalize_threshold = staticmethod(layer1_tag_normalize_threshold)
    layer1_temporal_window_hours   = staticmethod(layer1_temporal_window_hours)
    layer1_temporal_max_size       = staticmethod(layer1_temporal_max_size)
    layer1_temporal_time_window_hours = staticmethod(layer1_temporal_time_window_hours)
    layer1_spatial_dominance_threshold = staticmethod(layer1_spatial_dominance_threshold)
    layer1_tag_clusters_filename   = staticmethod(layer1_tag_clusters_filename)
    layer1_domain_list             = staticmethod(layer1_domain_list)
    layer1_llama_model             = staticmethod(layer1_llama_model)
    layer1_openai_model            = staticmethod(layer1_openai_model)
    layer1_openai_endpoint         = staticmethod(layer1_openai_endpoint)
    layer1_openai_provider         = staticmethod(layer1_openai_provider)
    layer1_openai_max_tokens       = staticmethod(layer1_openai_max_tokens)
    layer1_openai_temperature      = staticmethod(layer1_openai_temperature)
    ue_ood_svm_distance_threshold  = staticmethod(ue_ood_svm_distance_threshold)
    ue_ood_nn_distance_threshold   = staticmethod(ue_ood_nn_distance_threshold)
    ue_ood_isolation_score_threshold = staticmethod(ue_ood_isolation_score_threshold)
    ue_ood_votes_required          = staticmethod(ue_ood_votes_required)
    ue_composite_weight_similarity = staticmethod(ue_composite_weight_similarity)
    ue_composite_weight_confidence = staticmethod(ue_composite_weight_confidence)
    ue_composite_weight_ood        = staticmethod(ue_composite_weight_ood)
    ue_decision_high_threshold     = staticmethod(ue_decision_high_threshold)
    ue_decision_mid_threshold      = staticmethod(ue_decision_mid_threshold)
    ue_ood_penalty_detected        = staticmethod(ue_ood_penalty_detected)
    ue_ood_penalty_not_detected    = staticmethod(ue_ood_penalty_not_detected)
    ue_ood_tolerance_use_existing  = staticmethod(ue_ood_tolerance_use_existing)
    ue_ood_rejection_threshold     = staticmethod(ue_ood_rejection_threshold)
    workflow_traces_dir            = staticmethod(workflow_traces_dir)


cfg = _Cfg()
