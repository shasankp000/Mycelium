"""
expert_post_check.ollama_guard
===============================
VRAM guard: query Ollama for resident models and evict them all before
any PyTorch model is pushed onto the GPU.

Public API
----------
    evict_all(base_url)  ->  list[str]   evicted model names (may be empty)

Design notes
------------
* Uses GET /api/ps to discover *currently loaded* models -- this is the
  only reliable source; ``ollama list`` shows all downloaded models
  regardless of whether they are in VRAM.
* Eviction is done via POST /api/generate with keep_alive=0 and an empty
  prompt.  This is the Ollama-documented unload path and works on all
  Ollama versions >= 0.1.24.  The DELETE /api/delete endpoint deletes the
  model from disk, which is not what we want.
* All failures are logged as warnings and swallowed.  A failed eviction is
  never fatal -- the model will expire on its own after keep_alive elapses.
* evict_all() is safe to call at any time and from any thread.  It is
  intentionally synchronous so callers can be certain VRAM is free before
  proceeding.
"""

from __future__ import annotations

import logging
from typing import List

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434"
_EVICT_TIMEOUT   = 15   # seconds per model eviction request
_PS_TIMEOUT      = 5    # seconds for the /api/ps query


def evict_all(base_url: str = DEFAULT_BASE_URL) -> List[str]:
    """Evict every model currently loaded in Ollama from VRAM.

    Steps:
      1. GET  /api/ps               -- discover resident models
      2. POST /api/generate         -- keep_alive=0 for each model

    Args:
        base_url: Ollama server base URL (default ``http://localhost:11434``).

    Returns:
        List of model names that were successfully evicted.  Empty list if
        no models were resident or if the Ollama server is unreachable.
    """
    base_url = base_url.rstrip("/")
    resident = _get_resident_models(base_url)

    if not resident:
        logger.debug("ollama_guard.evict_all: no resident models found")
        return []

    logger.info(
        "ollama_guard.evict_all: evicting %d model(s): %s",
        len(resident), resident,
    )

    evicted: List[str] = []
    for model_name in resident:
        if _evict_model(base_url, model_name):
            evicted.append(model_name)

    if evicted:
        logger.info(
            "ollama_guard.evict_all: successfully evicted %s", evicted
        )
    return evicted


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_resident_models(base_url: str) -> List[str]:
    """Return model names currently loaded in Ollama VRAM."""
    try:
        with httpx.Client(timeout=_PS_TIMEOUT) as client:
            resp = client.get(f"{base_url}/api/ps")
            resp.raise_for_status()
            data = resp.json()

        # Ollama /api/ps response: {"models": [{"name": "...", ...}, ...]}
        models_list = data.get("models", [])
        return [m["name"] for m in models_list if "name" in m]

    except Exception as exc:
        logger.warning(
            "ollama_guard: could not query /api/ps -- %s "
            "(Ollama may not be running; skipping VRAM eviction)",
            exc,
        )
        return []


def _evict_model(base_url: str, model_name: str) -> bool:
    """Send keep_alive=0 to unload *model_name* from VRAM.

    Returns True on success, False on any error.
    """
    try:
        with httpx.Client(timeout=_EVICT_TIMEOUT) as client:
            resp = client.post(
                f"{base_url}/api/generate",
                json={"model": model_name, "prompt": "", "keep_alive": 0},
            )
        logger.info(
            "ollama_guard: evicted '%s' (status=%d)",
            model_name, resp.status_code,
        )
        return True
    except Exception as exc:
        logger.warning(
            "ollama_guard: failed to evict '%s' -- %s", model_name, exc
        )
        return False
