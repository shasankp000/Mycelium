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

Startup warmup
--------------
STARTUP_SPECS is the authoritative manifest of every non-LLM model weight
that Mycelium needs.  Pass it to warmup() early in the startup sequence so
that all downstream components (encoder, canonicalization pipeline, expert
filter, cross-encoder reranker) receive a hot cache hit instead of a cold
disk read on their first inference call::

    from model_registry import warmup, STARTUP_SPECS
    warmup(STARTUP_SPECS)
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_lock: threading.Lock = threading.Lock()
_cache: Dict[str, Any] = {}

# ---------------------------------------------------------------------------
# Startup manifest — the single source of truth for non-LLM model weights.
#
# Rules for adding a new entry:
#   1. Use model_type="sentence_transformer" for SentenceTransformer models.
#   2. Use model_type="hf_pipeline_cpu" for any HF pipeline that must stay on
#      CPU (classifiers, taggers, rerankers) to avoid competing with Ollama
#      for VRAM.
#   3. Use model_type="hf_automodel" for raw AutoModel+AutoTokenizer pairs
#      when you need direct access to hidden states / logits.
#   4. Always pin device="cpu" unless the model is provably GPU-only.
# ---------------------------------------------------------------------------
STARTUP_SPECS: List[Dict[str, str]] = [
    # ------------------------------------------------------------------
    # Sentence-Transformers — symmetric semantic encoders
    # Used by: embed_tags_transformer(), ExpertFilter similarity,
    #          mycelium/canonicalization/semantic_hash_pipeline (embedding_fn)
    # ------------------------------------------------------------------
    {
        "model_name": "sentence-transformers/all-mpnet-base-v2",
        "model_type": "sentence_transformer",
        "device": "cpu",
    },
    {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "model_type": "sentence_transformer",
        "device": "cpu",
    },
    # ------------------------------------------------------------------
    # Cross-encoder reranker
    # Used by: mycelium/canonicalization/canonical_form.py — scores
    #          candidate canonical representations against the original
    #          predicate-argument structure to select the best surface form.
    # ------------------------------------------------------------------
    {
        "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "model_type": "hf_pipeline_cpu",
        "device": "cpu",
    },
    # ------------------------------------------------------------------
    # Zero-shot NLI classifier
    # Used by: mycelium/canonicalization/predicate_families.py — maps
    #          a predicate lemma to its semantic family (causation,
    #          attribution, temporal, etc.) without fine-tuning.
    # ------------------------------------------------------------------
    {
        "model_name": "typeform/distilbert-base-uncased-mnli",
        "model_type": "hf_pipeline_cpu",
        "device": "cpu",
    },
    # ------------------------------------------------------------------
    # NER tagger
    # Used by: mycelium/canonicalization/srl_extractor.py — detects
    #          entity spans (PER, ORG, LOC, MISC) so they are preserved
    #          verbatim in the canonical form rather than being lemmatised.
    # ------------------------------------------------------------------
    {
        "model_name": "dslim/bert-base-NER",
        "model_type": "hf_pipeline_cpu",
        "device": "cpu",
    },
    # ------------------------------------------------------------------
    # POS tagger
    # Used by: mycelium/canonicalization/srl_extractor.py — determines
    #          argument boundary heuristics when AllenNLP SRL is absent.
    # ------------------------------------------------------------------
    {
        "model_name": "vblagoje/bert-english-uncased-finetuned-pos",
        "model_type": "hf_pipeline_cpu",
        "device": "cpu",
    },
]


def get_model(
    model_name: str,
    model_type: str = "sentence_transformer",
    device: str = "cpu",
) -> Any:
    """
    Return a cached model instance, loading it on first access.

    Parameters
    ----------
    model_name:  HuggingFace model identifier or local path.
    model_type:  One of:
                   "sentence_transformer"  — SentenceTransformer wrapper
                   "hf_pipeline"           — transformers pipeline (GPU-aware)
                   "hf_pipeline_cpu"       — transformers pipeline, always CPU
                   "hf_automodel"          — raw AutoModel + AutoTokenizer dict
    device:      "cpu" or "cuda".  Ignored for hf_pipeline_cpu (always CPU).

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

    elif model_type == "hf_pipeline_cpu":
        # Always CPU — never competes with Ollama for VRAM.
        # The task name is inferred from the model card by the pipeline factory;
        # callers that need a specific task should pass task= via get_model()
        # or call the pipeline factory directly after retrieving the cached model.
        from transformers import pipeline
        return pipeline(model=model_name, device=-1)

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
              Pass STARTUP_SPECS to load the full Mycelium non-LLM stack.

    Example
    -------
        from model_registry import warmup, STARTUP_SPECS
        warmup(STARTUP_SPECS)
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
