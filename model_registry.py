# Moved to mycelium/pipeline/model_registry.py
from mycelium.pipeline.model_registry import *  # noqa: F401,F403
from mycelium.pipeline.model_registry import (
    get_model, get_embedding, embed_batch, clear_embed_cache,
    warmup, loaded_models, STARTUP_SPECS,
)  # noqa: F401
