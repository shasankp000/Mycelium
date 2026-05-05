"""config_loader.py — single source of truth for Mycelium runtime configuration.

Resolution order for every value (highest wins):
  1. Matching environment variable (e.g. MYCELIUM_LAYER1_EMBED_MODEL)
  2. config.toml in the repository root
  3. Hard-coded default (identical to the defaults documented in config.toml)

Usage
-----
    import config_loader as cfg

    model  = cfg.get("layer1", "embed_model")          # -> "all-MiniLM-L6-v2"
    top_k  = cfg.get_int("layer1", "lens1_top_k")      # -> 3
    f      = cfg.get_float("layer1", "lens1_embedding_weight")  # -> 0.8
    domains = cfg.get_list("layer1.domain_list", "domains")     # -> [...]

Convenience accessors are also available (see bottom of file).
"""

from __future__ import annotations

import os
import sys
import pathlib
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# TOML loading — stdlib tomllib (Python ≥ 3.11) or third-party tomli backport
# ---------------------------------------------------------------------------
if sys.version_info >= (3, 11):
    import tomllib  # type: ignore
else:
    try:
        import tomllib  # type: ignore
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore  # pip install tomli
        except ImportError as exc:
            raise ImportError(
                "config_loader requires either Python 3.11+ (stdlib tomllib) "
                "or the 'tomli' backport: pip install tomli"
            ) from exc

# ---------------------------------------------------------------------------
# Load the config file once at import time
# ---------------------------------------------------------------------------
_CONFIG_PATH = pathlib.Path(__file__).parent / "config.toml"

_RAW: Dict[str, Any] = {}
if _CONFIG_PATH.exists():
    with open(_CONFIG_PATH, "rb") as _f:
        _RAW = tomllib.load(_f)


# ---------------------------------------------------------------------------
# Env-var name convention
# ---------------------------------------------------------------------------
def _env_key(section: str, key: str) -> str:
    """Map (section, key) -> MYCELIUM_SECTION_KEY (uppercase, dots → underscores)."""
    return "MYCELIUM_" + (section + "_" + key).upper().replace(".", "_")


# ---------------------------------------------------------------------------
# Core accessor
# ---------------------------------------------------------------------------
def get(section: str, key: str, default: Any = None) -> Any:
    """Return value for *section.key*, honouring resolution order."""
    # 1) Environment variable
    env_val = os.environ.get(_env_key(section, key))
    if env_val is not None:
        return env_val

    # 2) config.toml — section may be dot-separated ("layer1.domain_list")
    node: Any = _RAW
    for part in section.split("."):
        if not isinstance(node, dict):
            node = None
            break
        node = node.get(part)
    if isinstance(node, dict):
        val = node.get(key)
        if val is not None:
            return val

    # 3) Hard-coded default
    return default


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
        return []
    if isinstance(val, list):
        return val
    # Env var may arrive as comma-separated string
    return [item.strip() for item in str(val).split(",") if item.strip()]


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
    return get_list("layer1.domain_list", "domains", [
        "AI", "healthcare", "logistics", "finance", "education",
        "physics", "maths", "biology", "chemistry", "technology",
        "sports", "politics", "history", "art", "music", "literature"
    ])

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
