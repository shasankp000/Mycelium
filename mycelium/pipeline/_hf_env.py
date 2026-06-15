# mycelium/pipeline/_hf_env.py
"""
Bootstrap: set HuggingFace cache env-vars BEFORE any HF library is imported.
Import this as the absolute FIRST thing in any entry-point script.
"""
import os
from pathlib import Path

def _bootstrap_hf_cache() -> str:
    try:
        from mycelium.pipeline import config_loader as _cfg
        cache = _cfg.hf_cache_dir()
    except Exception:
        cache = str(Path(__file__).resolve().parents[2] / "hf_cache")

    os.environ["HF_HOME"]                  = cache
    os.environ["TRANSFORMERS_CACHE"]       = cache
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache
    os.environ["HF_DATASETS_CACHE"]        = cache
    return cache

HF_CACHE_DIR: str = _bootstrap_hf_cache()
