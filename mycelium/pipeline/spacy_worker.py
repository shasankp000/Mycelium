"""
spacy_worker.py
===============
**Runs inside .venv2 (Python 3.11 / Lexis venv) — never imported by Mycelium directly.**

Mycelium's spacy_bridge.py shells out to this script via:

    .venv2/bin/python spacy_worker.py

Protocol
--------
- Reads a single JSON object from **stdin** (one line, UTF-8).
- Writes a single JSON object to **stdout** (one line, UTF-8).
- All errors are reported as {"ok": false, "error": "<message>"} on stdout.
  Nothing is written to stdout on the happy path except the result object.

Supported tasks (field: "task")
--------------------------------
tokenize        → {"ok": true, "tokens": [...]}
pos_tag         → {"ok": true, "tokens": [{"text": ..., "pos": ..., "tag": ...}, ...]}
sentences       → {"ok": true, "sentences": [...]}
ner             → {"ok": true, "entities": [{"text": ..., "label": ..., "start": ..., "end": ...}, ...]}
noun_chunks     → {"ok": true, "chunks": [...]}
lemmatize       → {"ok": true, "lemmas": [...]}
pipeline        → {"ok": true, "tokens": [...], "pos": [...], "sentences": [...],
                              "entities": [...], "noun_chunks": [...], "lemmas": [...]}

Input schema (all tasks)
-------------------------
{
    "task":  "<task_name>",
    "text":  "<input text>",
    "model": "en_core_web_sm"   // optional; defaults to en_core_web_sm
}

Usage example (from shell)
--------------------------
    echo '{"task": "pos_tag", "text": "The cat sat on the mat."}' \\
        | .venv2/bin/python spacy_worker.py
"""

from __future__ import annotations

import json
import sys


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(model_name: str):
    """Load a spaCy model, caching it in this process for the lifetime of the
    worker.  (The worker is a short-lived subprocess; the cache just avoids
    double-loading within a single invocation that calls multiple tasks via
    the 'pipeline' super-task.)"""
    if not hasattr(_load_model, "_cache"):
        _load_model._cache = {}
    if model_name not in _load_model._cache:
        import spacy  # noqa: PLC0415 — intentional late import; only available in .venv2
        _load_model._cache[model_name] = spacy.load(model_name)
    return _load_model._cache[model_name]


def _ok(**kwargs) -> dict:
    return {"ok": True, **kwargs}


def _err(message: str) -> dict:
    return {"ok": False, "error": message}


# ---------------------------------------------------------------------------
# Task handlers
# ---------------------------------------------------------------------------

def handle_tokenize(doc) -> dict:
    return _ok(tokens=[t.text for t in doc])


def handle_pos_tag(doc) -> dict:
    return _ok(tokens=[
        {"text": t.text, "pos": t.pos_, "tag": t.tag_, "dep": t.dep_}
        for t in doc
    ])


def handle_sentences(doc) -> dict:
    return _ok(sentences=[s.text for s in doc.sents])


def handle_ner(doc) -> dict:
    return _ok(entities=[
        {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
        for ent in doc.ents
    ])


def handle_noun_chunks(doc) -> dict:
    return _ok(chunks=[chunk.text for chunk in doc.noun_chunks])


def handle_lemmatize(doc) -> dict:
    return _ok(lemmas=[t.lemma_ for t in doc])


def handle_pipeline(doc) -> dict:
    """Run all analyses in a single pass — most efficient when the caller
    needs several annotations for the same text."""
    return _ok(
        tokens=[t.text for t in doc],
        pos=[{"text": t.text, "pos": t.pos_, "tag": t.tag_, "dep": t.dep_} for t in doc],
        sentences=[s.text for s in doc.sents],
        entities=[
            {"text": e.text, "label": e.label_, "start": e.start_char, "end": e.end_char}
            for e in doc.ents
        ],
        noun_chunks=[c.text for c in doc.noun_chunks],
        lemmas=[t.lemma_ for t in doc],
    )


_TASK_MAP = {
    "tokenize":    handle_tokenize,
    "pos_tag":     handle_pos_tag,
    "sentences":   handle_sentences,
    "ner":         handle_ner,
    "noun_chunks": handle_noun_chunks,
    "lemmatize":   handle_lemmatize,
    "pipeline":    handle_pipeline,
}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    raw = sys.stdin.read().strip()
    if not raw:
        print(json.dumps(_err("Empty stdin — expected a JSON task object.")), flush=True)
        return

    try:
        request = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(json.dumps(_err(f"JSON parse error: {exc}")), flush=True)
        return

    task  = request.get("task", "")
    text  = request.get("text", "")
    model = request.get("model", "en_core_web_sm")

    if not task:
        print(json.dumps(_err("Missing required field: 'task'")), flush=True)
        return

    if task not in _TASK_MAP:
        print(json.dumps(_err(
            f"Unknown task '{task}'. Valid tasks: {sorted(_TASK_MAP)}"
        )), flush=True)
        return

    if not isinstance(text, str) or not text.strip():
        print(json.dumps(_err("Missing or empty 'text' field.")), flush=True)
        return

    try:
        nlp = _load_model(model)
    except OSError as exc:
        print(json.dumps(_err(
            f"Could not load spaCy model '{model}': {exc}. "
            f"Run: .venv2/bin/python -m spacy download {model}"
        )), flush=True)
        return

    try:
        doc    = nlp(text)
        result = _TASK_MAP[task](doc)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps(_err(f"spaCy processing error: {exc}")), flush=True)
        return

    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
