"""
model_registry.py
==================
Process-level singleton cache for all HuggingFace / SentenceTransformer models.

Usage
-----
    from model_registry import get_model

    encoder = get_model("all-mpnet-base-v2", model_type="sentence_transformer", device="cpu")

First call loads the model and caches it.  Every subsequent call with the
same (model_name, model_type, device) triple returns the cached instance
in O(1) — no disk I/O, no GPU allocation.

Thread safety
-------------
Uses double-checked locking so that if two threads race on the very first
load, only ONE thread performs the download/init and the other waits.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_lock: threading.Lock = threading.Lock()
_cache: Dict[str, Any] = {}


def get_model(
    model_name: str,
    model_type: str = "sentence_transformer",
    device: str = "cpu",
) -> Any:
    """
    Return a cached model instance, loading it on first access.

    Parameters
    ----------
    model_name:  HuggingFace model identifier (e.g. "all-mpnet-base-v2").
    model_type:  One of "sentence_transformer", "hf_pipeline", "hf_automodel".
    device:      "cpu" or "cuda".  Sentence-transformers are always pinned to
                 CPU by default to avoid competing with Ollama for VRAM.

    Returns
    -------
    The loaded model object (type depends on model_type).
    """
    cache_key = f"{model_type}:{model_name}:{device}"

    # Fast path — no lock needed for a read-only dict check after warmup
    if cache_key in _cache:
        return _cache[cache_key]

    with _lock:
        # Double-checked locking: another thread may have loaded while we waited
        if cache_key in _cache:
            return _cache[cache_key]

        logger.info(
            "ModelRegistry: loading %s (%s) on %s — first and only time",
            model_name, model_type, device,
        )
        model = _load(model_name, model_type, device)
        _cache[cache_key] = model
        logger.info(
            "ModelRegistry: %s cached — %d model(s) now resident",
            model_name, len(_cache),
        )
        return model


def _load(model_name: str, model_type: str, device: str) -> Any:
    """Internal loader — called exactly once per unique key."""
    if model_type == "sentence_transformer":
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name, device=device)

    elif model_type == "hf_pipeline":
        from transformers import pipeline
        return pipeline(model_name, device=0 if device == "cuda" else -1)

    elif model_type == "hf_automodel":
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        return {"model": model, "tokenizer": tokenizer}

    else:
        raise ValueError(f"ModelRegistry: unknown model_type={model_type!r}")


def warmup(specs: List[Dict[str, str]]) -> None:
    """
    Pre-load a list of models at startup so the first real request is warm.

    Parameters
    ----------
    specs:  List of dicts, each with keys understood by get_model():
              model_name, model_type (optional), device (optional).

    Example
    -------
        warmup([
            {"model_name": "all-mpnet-base-v2",  "model_type": "sentence_transformer", "device": "cpu"},
            {"model_name": "all-MiniLM-L6-v2",   "model_type": "sentence_transformer", "device": "cpu"},
        ])
    """
    for spec in specs:
        try:
            get_model(**spec)
        except Exception as exc:
            logger.error(
                "ModelRegistry warmup failed for %s: %s",
                spec.get("model_name", "<unknown>"), exc,
            )
    logger.info("ModelRegistry warmup complete — %d model(s) loaded", len(_cache))


def loaded_models() -> List[str]:
    """Return the cache keys of all currently resident models (for health checks)."""
    return list(_cache.keys())
