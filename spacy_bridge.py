"""
spacy_bridge.py
===============
**Runs inside Mycelium's .venv (Python 3.14) — never imports spaCy directly.**

All spaCy NLP work is delegated to spacy_worker.py which runs inside
.venv2 (Lexis's Python 3.11 virtual environment).  Communication is
over stdin/stdout using newline-delimited JSON.

Architecture
------------

    Mycelium (.venv, Py 3.14)
        │
        │  subprocess.run / asyncio.create_subprocess_exec
        │  stdin  = JSON task
        │  stdout = JSON result
        ┃
    spacy_worker.py (.venv2, Py 3.11)
        └── spaCy 3.x

Path resolution order for .venv2
---------------------------------
1. config.toml  [lexis] venv2_path
2. Environment variable  LEXIS_VENV2_PATH
3. Default: <project_root>/.venv2

Public API
----------
Synchronous (use from non-async callers):
    spacy_call(task, text, model="en_core_web_sm") -> dict
    tokenize(text)    -> list[str]
    pos_tag(text)     -> list[dict]
    sentences(text)   -> list[str]
    ner(text)         -> list[dict]
    noun_chunks(text) -> list[str]
    lemmatize(text)   -> list[str]
    pipeline(text)    -> dict

Asynchronous equivalents (use from async callers):
    async_spacy_call(task, text, model="en_core_web_sm") -> dict
    async_pipeline(text) -> dict

Diagnostics:
    health_check() -> bool   # returns True if worker responds correctly
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Internal helpers — path resolution
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    """Absolute path to the Mycelium project root (directory of this file)."""
    return Path(__file__).resolve().parent


def _resolve_venv2_python() -> Path:
    """
    Locate the Python interpreter inside .venv2.

    Resolution order:
    1. config.toml [lexis] venv2_path
    2. Env var LEXIS_VENV2_PATH
    3. Default <project_root>/.venv2
    """
    # --- 1. config.toml ---
    try:
        # Avoid a hard dependency on config_loader at import time; fall back
        # gracefully if the loader itself fails (e.g. toml not installed yet).
        from config_loader import get_config  # noqa: PLC0415
        cfg = get_config()
        venv2_str = (
            cfg.get("lexis", {}).get("venv2_path")
            or cfg.get("lexis", {}).get("venv2")
        )
        if venv2_str:
            venv2_root = Path(venv2_str).expanduser()
            if not venv2_root.is_absolute():
                venv2_root = _project_root() / venv2_root
            py = venv2_root / "bin" / "python"
            if py.exists():
                return py
    except Exception:  # noqa: BLE001
        pass

    # --- 2. env var ---
    env_path = os.environ.get("LEXIS_VENV2_PATH")
    if env_path:
        venv2_root = Path(env_path).expanduser()
        py = venv2_root / "bin" / "python"
        if py.exists():
            return py

    # --- 3. default ---
    return _project_root() / ".venv2" / "bin" / "python"


def _worker_script() -> Path:
    return _project_root() / "spacy_worker.py"


# ---------------------------------------------------------------------------
# Core synchronous bridge
# ---------------------------------------------------------------------------

def spacy_call(
    task: str,
    text: str,
    model: str = "en_core_web_sm",
    timeout: float = 60.0,
) -> dict[str, Any]:
    """
    Send a task to spacy_worker.py running inside .venv2 and return the
    parsed JSON result.

    Parameters
    ----------
    task    : One of tokenize | pos_tag | sentences | ner | noun_chunks |
              lemmatize | pipeline
    text    : Input text to process.
    model   : spaCy model name (must be installed in .venv2).
    timeout : Subprocess timeout in seconds.

    Returns
    -------
    dict with at minimum {"ok": True, ...} or {"ok": False, "error": str}.

    Raises
    ------
    RuntimeError   if the subprocess itself fails to start or times out.
    """
    python = _resolve_venv2_python()
    worker = _worker_script()

    if not python.exists():
        raise RuntimeError(
            f"[spacy_bridge] .venv2 Python interpreter not found at '{python}'. "
            "Please follow DUAL_VENV_SETUP.md to create .venv2 with Lexis's venv."
        )
    if not worker.exists():
        raise RuntimeError(
            f"[spacy_bridge] spacy_worker.py not found at '{worker}'."
        )

    payload = json.dumps({"task": task, "text": text, "model": model}, ensure_ascii=False)

    try:
        result = subprocess.run(
            [str(python), str(worker)],
            input=payload,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"[spacy_bridge] spacy_worker.py timed out after {timeout}s "
            f"(task='{task}', text length={len(text)})."
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"[spacy_bridge] Could not execute '{python}'. "
            "Is .venv2 correctly set up? See DUAL_VENV_SETUP.md."
        )

    if result.returncode != 0 and not result.stdout.strip():
        stderr = result.stderr.strip()
        raise RuntimeError(
            f"[spacy_bridge] Worker exited with code {result.returncode}: {stderr}"
        )

    stdout = result.stdout.strip()
    if not stdout:
        raise RuntimeError(
            "[spacy_bridge] Worker produced no output. "
            f"stderr: {result.stderr.strip()}"
        )

    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"[spacy_bridge] Could not parse worker output as JSON: {exc}. "
            f"Raw output: {stdout[:300]!r}"
        ) from exc


# ---------------------------------------------------------------------------
# Async bridge
# ---------------------------------------------------------------------------

async def async_spacy_call(
    task: str,
    text: str,
    model: str = "en_core_web_sm",
    timeout: float = 60.0,
) -> dict[str, Any]:
    """
    Async version of spacy_call().  Use from async FastAPI handlers /
    async pipeline code to avoid blocking the event loop.
    """
    python = _resolve_venv2_python()
    worker = _worker_script()

    if not python.exists():
        raise RuntimeError(
            f"[spacy_bridge] .venv2 Python not found at '{python}'. "
            "See DUAL_VENV_SETUP.md."
        )

    payload = json.dumps({"task": task, "text": text, "model": model}, ensure_ascii=False)

    proc = await asyncio.create_subprocess_exec(
        str(python), str(worker),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(input=payload.encode("utf-8")),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        raise RuntimeError(
            f"[spacy_bridge] Async worker timed out after {timeout}s "
            f"(task='{task}')."
        )

    stdout = stdout_bytes.decode("utf-8").strip()
    if not stdout:
        stderr = stderr_bytes.decode("utf-8").strip()
        raise RuntimeError(
            f"[spacy_bridge] Async worker produced no output. stderr: {stderr}"
        )

    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"[spacy_bridge] Could not parse async worker output: {exc}. "
            f"Raw: {stdout[:300]!r}"
        ) from exc


# ---------------------------------------------------------------------------
# Convenience wrappers — synchronous
# ---------------------------------------------------------------------------

def tokenize(text: str, model: str = "en_core_web_sm") -> list[str]:
    """Return a list of token strings."""
    return spacy_call("tokenize", text, model)["tokens"]


def pos_tag(text: str, model: str = "en_core_web_sm") -> list[dict]:
    """Return list of {text, pos, tag, dep} dicts."""
    return spacy_call("pos_tag", text, model)["tokens"]


def sentences(text: str, model: str = "en_core_web_sm") -> list[str]:
    """Return list of sentence strings."""
    return spacy_call("sentences", text, model)["sentences"]


def ner(text: str, model: str = "en_core_web_sm") -> list[dict]:
    """Return list of {text, label, start, end} named entity dicts."""
    return spacy_call("ner", text, model)["entities"]


def noun_chunks(text: str, model: str = "en_core_web_sm") -> list[str]:
    """Return list of noun chunk strings."""
    return spacy_call("noun_chunks", text, model)["chunks"]


def lemmatize(text: str, model: str = "en_core_web_sm") -> list[str]:
    """Return list of lemma strings (one per token)."""
    return spacy_call("lemmatize", text, model)["lemmas"]


def pipeline(text: str, model: str = "en_core_web_sm") -> dict:
    """
    Run all annotations in a single spaCy pass.

    Returns
    -------
    {
        "ok":          True,
        "tokens":      [...],
        "pos":         [{"text", "pos", "tag", "dep"}, ...],
        "sentences":   [...],
        "entities":    [{"text", "label", "start", "end"}, ...],
        "noun_chunks": [...],
        "lemmas":      [...],
    }
    """
    return spacy_call("pipeline", text, model)


# ---------------------------------------------------------------------------
# Async convenience wrappers
# ---------------------------------------------------------------------------

async def async_pipeline(text: str, model: str = "en_core_web_sm") -> dict:
    """Async version of pipeline()."""
    return await async_spacy_call("pipeline", text, model)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def health_check() -> bool:
    """
    Verify that .venv2 + spacy_worker.py are functioning correctly.

    Returns True if the worker responds with a valid result.
    Prints a diagnostic message to stderr on failure.
    """
    try:
        result = spacy_call("tokenize", "Hello world.", timeout=30.0)
        return result.get("ok") is True and "tokens" in result
    except Exception as exc:  # noqa: BLE001
        print(
            f"[spacy_bridge] health_check FAILED: {exc}\n"
            "  → Ensure .venv2 is set up per DUAL_VENV_SETUP.md",
            file=sys.stderr,
        )
        return False


# ---------------------------------------------------------------------------
# CLI smoke-test: python spacy_bridge.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("[spacy_bridge] Running smoke-test against .venv2...")
    ok = health_check()
    if ok:
        sample = pipeline("Quantum mechanics describes the behaviour of particles at the subatomic scale.")
        print(f"[spacy_bridge] Sentences : {sample['sentences']}")
        print(f"[spacy_bridge] Entities  : {sample['entities']}")
        print(f"[spacy_bridge] POS sample: {sample['pos'][:5]}")
        print("[spacy_bridge] ✅ All OK")
    else:
        sys.exit(1)
