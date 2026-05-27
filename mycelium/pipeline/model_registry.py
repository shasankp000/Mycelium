"""
model_registry.py
==================
Process-level singleton cache for all HuggingFace / SentenceTransformer /
sklearn models.

Usage
-----
    from mycelium.pipeline.model_registry import get_model, get_embedding

    encoder = get_model("all-mpnet-base-v2", model_type="sentence_transformer", device="cpu")
    vec     = get_embedding("some text", "all-mpnet-base-v2")

Layer 0 sklearn classifiers
----------------------------
Loaded at startup via warmup_layer0() if the .joblib files are present under
    <project_root>/models/layer0/

Once loaded they are accessible via the typed helpers:
    get_layer0_classifier(name)  → dict artifact or None
    layer0_models_loaded()       → dict[name, bool]

The classifiers are NOT required to be present for the system to run — when
absent the Layer 0 components fall back to LLM arbitration transparently.
"""
from __future__ import annotations

import hashlib
import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_lock: threading.Lock = threading.Lock()
_cache: Dict[str, Any] = {}

_EMBED_CACHE_MAX: int = 4096
_embed_lock: threading.RLock = threading.RLock()
_embed_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()

# ---------------------------------------------------------------------------
# Layer 0 model paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_LAYER0_MODEL_DIR = _PROJECT_ROOT / "models" / "layer0"

_LAYER0_MODEL_FILES: Dict[str, str] = {
    "manipulation_classifier": "manipulation_classifier.joblib",
    "objectivity_classifier":  "objectivity_classifier.joblib",
    "assumption_typer":        "assumption_typer.joblib",
}

# Internal registry key prefix for layer0 sklearn artifacts
_L0_PREFIX = "layer0:"


# ---------------------------------------------------------------------------
# Embedding helpers (unchanged)
# ---------------------------------------------------------------------------

def _embed_cache_key(text: str, model_name: str) -> str:
    return hashlib.sha256(f"{model_name}\x00{text}".encode()).hexdigest()


def get_embedding(
    text: str,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    device: str = "cpu",
) -> np.ndarray:
    key = _embed_cache_key(text, model_name)
    with _embed_lock:
        if key in _embed_cache:
            _embed_cache.move_to_end(key)
            return _embed_cache[key]
    model = get_model(model_name, model_type="sentence_transformer", device=device)
    vec: np.ndarray = model.encode(text, convert_to_numpy=True)
    with _embed_lock:
        if key not in _embed_cache:
            _embed_cache[key] = vec
            if len(_embed_cache) > _EMBED_CACHE_MAX:
                _embed_cache.popitem(last=False)
    return vec


def embed_batch(
    texts: List[str],
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    device: str = "cpu",
) -> List[np.ndarray]:
    keys = [_embed_cache_key(t, model_name) for t in texts]
    result: List[Optional[np.ndarray]] = [None] * len(texts)
    missing_indices: List[int] = []
    with _embed_lock:
        for i, key in enumerate(keys):
            if key in _embed_cache:
                _embed_cache.move_to_end(key)
                result[i] = _embed_cache[key]
            else:
                missing_indices.append(i)
    if missing_indices:
        missing_texts = [texts[i] for i in missing_indices]
        model = get_model(model_name, model_type="sentence_transformer", device=device)
        vecs: np.ndarray = model.encode(missing_texts, convert_to_numpy=True)
        with _embed_lock:
            for idx, vec in zip(missing_indices, vecs):
                k = keys[idx]
                result[idx] = vec
                if k not in _embed_cache:
                    _embed_cache[k] = vec
                    if len(_embed_cache) > _EMBED_CACHE_MAX:
                        _embed_cache.popitem(last=False)
    return result  # type: ignore[return-value]


def clear_embed_cache() -> None:
    with _embed_lock:
        _embed_cache.clear()


# ---------------------------------------------------------------------------
# HuggingFace / SentenceTransformer startup specs (unchanged)
# ---------------------------------------------------------------------------

STARTUP_SPECS: List[Dict[str, str]] = [
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
    {
        "model_name": "typeform/distilbert-base-uncased-mnli",
        "model_type": "hf_pipeline_cpu",
        "device": "cpu",
    },
    {
        "model_name": "dslim/bert-base-NER",
        "model_type": "hf_pipeline_cpu",
        "device": "cpu",
    },
    {
        "model_name": "vblagoje/bert-english-uncased-finetuned-pos",
        "model_type": "hf_pipeline_cpu",
        "device": "cpu",
    },
]


# ---------------------------------------------------------------------------
# Generic model loader
# ---------------------------------------------------------------------------

def get_model(
    model_name: str,
    model_type: str = "sentence_transformer",
    device: str = "cpu",
) -> Any:
    cache_key = f"{model_type}:{model_name}:{device}"
    if cache_key in _cache:
        return _cache[cache_key]
    with _lock:
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
    if model_type == "sentence_transformer":
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name, device=device)
    elif model_type == "hf_pipeline":
        from transformers import pipeline
        return pipeline(model_name, device=0 if device == "cuda" else -1)
    elif model_type == "hf_pipeline_cpu":
        from transformers import pipeline
        return pipeline(model=model_name, device=-1)
    elif model_type == "hf_automodel":
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        return {"model": model, "tokenizer": tokenizer}
    elif model_type == "sklearn_joblib":
        # model_name is the full path to the .joblib file
        import joblib  # type: ignore
        return joblib.load(model_name)
    else:
        raise ValueError(f"ModelRegistry: unknown model_type={model_type!r}")


def warmup(specs: List[Dict[str, str]]) -> None:
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
    return list(_cache.keys())


# ---------------------------------------------------------------------------
# Layer 0 sklearn classifier helpers
# ---------------------------------------------------------------------------

def warmup_layer0() -> Dict[str, bool]:
    """
    Load all available Layer 0 sklearn classifiers from
    <project_root>/models/layer0/ into the registry cache.

    Silently skips any .joblib file that does not exist yet — the Layer 0
    components fall back to LLM arbitration when models are absent.

    Returns a dict {model_name: loaded_ok} for logging / health-check.
    """
    results: Dict[str, bool] = {}
    for name, filename in _LAYER0_MODEL_FILES.items():
        path = _LAYER0_MODEL_DIR / filename
        cache_key = f"{_L0_PREFIX}{name}"
        if cache_key in _cache:
            results[name] = True
            continue
        if not path.exists():
            logger.debug(
                "ModelRegistry [layer0]: %s not found at %s — LLM fallback active.",
                name, path,
            )
            results[name] = False
            continue
        try:
            with _lock:
                if cache_key not in _cache:
                    import joblib  # type: ignore
                    artifact = joblib.load(path)
                    _cache[cache_key] = artifact
            logger.info(
                "ModelRegistry [layer0]: loaded %s from %s", name, path
            )
            results[name] = True
        except Exception as exc:
            logger.error(
                "ModelRegistry [layer0]: failed to load %s: %s", name, exc
            )
            results[name] = False
    loaded = sum(results.values())
    logger.info(
        "ModelRegistry [layer0]: %d/%d classifiers loaded.",
        loaded, len(_LAYER0_MODEL_FILES),
    )
    return results


def get_layer0_classifier(name: str) -> Optional[Dict[str, Any]]:
    """
    Return the loaded Layer 0 classifier artifact dict, or None if not
    available (model file not present or not yet trained).

    Artifact dict structure (as saved by train_layer0_models.py):
        manipulation_classifier:  {model, label_encoder, version}
        objectivity_classifier:   {model, label_encoder, version}
        assumption_typer:         {model, active_types, all_types, version}
    """
    cache_key = f"{_L0_PREFIX}{name}"
    return _cache.get(cache_key, None)


def layer0_models_loaded() -> Dict[str, bool]:
    """Return {model_name: bool} presence map for all Layer 0 classifiers."""
    return {
        name: (f"{_L0_PREFIX}{name}" in _cache)
        for name in _LAYER0_MODEL_FILES
    }
