"""
model_registry.py
==================
Process-level singleton cache for all HuggingFace / SentenceTransformer models.

Usage
-----
    from mycelium.pipeline.model_registry import get_model, get_embedding

    encoder = get_model("all-mpnet-base-v2", model_type="sentence_transformer", device="cpu")
    vec     = get_embedding("some text", "all-mpnet-base-v2")
"""
from __future__ import annotations

import hashlib
import logging
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_lock: threading.Lock = threading.Lock()
_cache: Dict[str, Any] = {}

_EMBED_CACHE_MAX: int = 4096
_embed_lock: threading.RLock = threading.RLock()
_embed_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()


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
