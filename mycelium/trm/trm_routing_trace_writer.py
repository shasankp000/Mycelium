"""
Phase D — TRMRoutingTraceWriter
================================
Thread-safe JSONL writer that captures per-sentence routing decisions
from run_mycelium_workflow() and serialises them into the exact schema
expected by TRMTrainer.

Written to:
    training_data/routing_traces.jsonl          (80 % — training split)
    evaluation_data/routing_ground_truth.jsonl  (20 % — eval split)

Schema of each record (matches TRMTrainer dataset format exactly)
-----------------------------------------------------------------
::

    {
        "token_ids":             list[int],   # length == CONTEXT_LEN (64)
        "spectral_vec":          list[float], # length == N_DOMAINS (dynamic)
        "predicate_family_id":   int,         # routing_classification → index
        "initial_domain_probs":  list[float], # length == N_DOMAINS
        "target_domain":         int,         # selected_domain → domain index
        "halt_label":            float,       # 1.0=halt (confident), 0.0=continue
    }

halt_label derivation
---------------------
halt=1.0  Top-1 spectral score > HALT_CONFIDENT_THRESHOLD (0.50) AND
          TRM agreed (or no TRM checkpoint exists yet).
halt=0.0  Routing was uncertain or TRM was overruled.

Confidence gating (§A)
-----------------------
Only samples where TRM was uncertain (halt_confidence < GATE_THRESHOLD)
OR TRM was overruled are written.  When TRM has no checkpoint
(trm_primary_idx == -1) gating is bypassed so cold-start training sets
are still populated.  Set GATE_THRESHOLD=1.0 to write every sample.

Domain list (dynamic)
---------------------
DOMAIN_LIST is now built at import time by calling
_discover_live_domains() from layer1_router, which reads whatever expert
subdirectories exist on disk.  This means the writer automatically
numbers any new expert domain correctly instead of silently returning -1
and discarding the training sample.

Spectral scores source
-----------------------
RoutingResult stores the raw per-domain spectral scores in
    routing_context.metadata["lens_scores"]["spectral"]
not in a .spectral_scores attribute.  The writer now reads from that
path with two fallbacks:
  1. metadata["lens_scores"]["spectral"]  (primary)
  2. routing_context.fusion_scores         (secondary — fused dict)
  3. routing_context.spectral_scores       (legacy attribute)
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

# ------------------------------------------------------------------ #
# Constants matching TRMConfig defaults                               #
# ------------------------------------------------------------------ #

CONTEXT_LEN: int = 64
VOCAB_SIZE:  int = 8192
GATE_THRESHOLD: float = float(os.getenv("TRM_GATE_THRESHOLD", "1.0"))
EVAL_FRACTION:  float = 0.20

HALT_CONFIDENT_THRESHOLD: float = float(
    os.getenv("TRM_HALT_CONFIDENT_THRESHOLD", "0.50")
)


# ------------------------------------------------------------------ #
# Dynamic domain discovery                                            #
# ------------------------------------------------------------------ #

def _build_domain_list() -> List[str]:
    """
    Build DOMAIN_LIST from whatever expert subdirectories exist on disk.

    Delegates to layer1_router._discover_live_domains() so the indexing
    is always consistent with the router’s own ontology.  Falls back to
    a minimal hard-coded list if the import fails.
    """
    try:
        from mycelium.pipeline.layer1_router import _discover_live_domains
        live = _discover_live_domains()
        if live:
            return live
    except Exception:
        pass
    # Hard fallback — covers the four domains present in the original writer.
    return ["music", "physics", "chemistry", "medical"]


# Built once at import; call reload_domain_list() after adding a new expert.
DOMAIN_LIST: List[str] = _build_domain_list()
N_DOMAINS:   int       = len(DOMAIN_LIST)


def reload_domain_list() -> None:
    """Refresh DOMAIN_LIST and N_DOMAINS from disk (no process restart needed)."""
    global DOMAIN_LIST, N_DOMAINS
    DOMAIN_LIST = _build_domain_list()
    N_DOMAINS   = len(DOMAIN_LIST)


# routing_classification → predicate_family_id int
PREDICATE_FAMILY_MAP: Dict[str, int] = {
    "ATTRIBUTE_ONLY": 0,
    "CAUSAL":         1,
    "TEMPORAL":       2,
    "COMPARATIVE":    3,
    "DEFINITIONAL":   4,
    "PROCEDURAL":     5,
    "FACTUAL":        6,
    "HYPOTHETICAL":   7,
    "NEGATION":       8,
    "RELATIONAL":     9,
    "UNKNOWN":        10,
}
_DEFAULT_PREDICATE_FAMILY: int = 10  # UNKNOWN


# ------------------------------------------------------------------ #
# Tokeniser — deterministic word-hash, no model weights needed        #
# ------------------------------------------------------------------ #

def _encode_query(text: str) -> List[int]:
    """
    Deterministic word-hash tokeniser.

    Splits on whitespace, hashes each word to [0, VOCAB_SIZE),
    truncates/pads to CONTEXT_LEN.  Same text always → same token IDs.
    """
    words = text.lower().split()[:CONTEXT_LEN]
    ids = [
        int(hashlib.sha256(w.encode()).hexdigest(), 16) % VOCAB_SIZE
        for w in words
    ]
    ids += [0] * (CONTEXT_LEN - len(ids))
    return ids


# ------------------------------------------------------------------ #
# Domain index helpers                                                 #
# ------------------------------------------------------------------ #

def _domain_to_idx(domain: str) -> int:
    """Map a domain name string to its integer index.  Unknown → -1."""
    name = (domain or "").lower().strip()
    try:
        return DOMAIN_LIST.index(name)
    except ValueError:
        return -1


def _is_numeric(v: Any) -> bool:
    """Return True if v can be safely cast to float as a scalar score."""
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


# ------------------------------------------------------------------ #
# Spectral score extraction                                            #
# ------------------------------------------------------------------ #

def _extract_spectral_scores(routing_context: Any) -> Any:
    """
    Extract the raw per-domain spectral score dict from a RoutingResult.

    Priority order
    --------------
    1. routing_context.metadata["lens_scores"]["spectral"]
       The canonical location — written by MultiLensRouter.route().
    2. routing_context.fusion_scores
       The fused score dict; still per-domain floats, used as fallback.
    3. routing_context.spectral_scores
       Legacy attribute path (kept for backwards compatibility).
    """
    # Path 1 — preferred
    try:
        meta = routing_context.metadata  # type: ignore[union-attr]
        if isinstance(meta, dict):
            lens = meta.get("lens_scores", {})
            if isinstance(lens, dict):
                spectral = lens.get("spectral")
                if spectral:
                    return spectral
    except Exception:
        pass

    # Path 2 — fused scores dict
    try:
        fs = routing_context.fusion_scores  # type: ignore[union-attr]
        if fs:
            return fs
    except Exception:
        pass

    # Path 3 — legacy
    return getattr(routing_context, "spectral_scores", None)


def _spectral_vec_to_tensor(
    spectral_scores: Any,
    selected_domains: List[str],
) -> List[float]:
    """
    Build a float[N_DOMAINS] vector from routing_context spectral scores.

    Tries to read a dict / list / object from spectral_scores.
    Falls back to a soft one-hot over selected_domains if unavailable
    or if the list contains non-numeric values.
    Always normalises to sum == 1.0.
    """
    vec = [0.0] * N_DOMAINS

    # Case 1: dict keyed by domain name
    if isinstance(spectral_scores, dict):
        for domain, score in spectral_scores.items():
            idx = _domain_to_idx(domain)
            if 0 <= idx < N_DOMAINS:
                vec[idx] = float(score)

    # Case 2: numeric list/tuple of exactly N_DOMAINS elements
    elif (
        isinstance(spectral_scores, (list, tuple))
        and len(spectral_scores) == N_DOMAINS
        and all(_is_numeric(v) for v in spectral_scores)
    ):
        vec = [float(v) for v in spectral_scores]

    # Case 3: object with .scores attribute
    elif hasattr(spectral_scores, "scores"):
        scores = spectral_scores.scores
        if isinstance(scores, dict):
            for domain, score in scores.items():
                idx = _domain_to_idx(domain)
                if 0 <= idx < N_DOMAINS:
                    vec[idx] = float(score)

    # Fallback: soft one-hot over selected_domains
    if sum(vec) == 0.0 and selected_domains:
        weight = 0.7 / max(1, len(selected_domains))
        remainder = (1.0 - 0.7) / max(1, N_DOMAINS - len(selected_domains))
        vec = [remainder] * N_DOMAINS
        for d in selected_domains:
            idx = _domain_to_idx(d)
            if 0 <= idx < N_DOMAINS:
                vec[idx] = weight

    # Normalise
    total = sum(vec)
    if total > 0:
        vec = [v / total for v in vec]
    else:
        vec = [1.0 / N_DOMAINS] * N_DOMAINS

    return vec


def _initial_domain_probs(spectral_vec: List[float]) -> List[float]:
    """
    Bootstrap prior: uniform 1/N blended 50/50 with spectral_vec.
    Sums to 1.0 by construction.
    """
    uniform = 1.0 / N_DOMAINS
    blended = [(uniform + sv) / 2.0 for sv in spectral_vec]
    total = sum(blended)
    return [v / total for v in blended]


def _derive_halt_label(
    spectral_vec: List[float],
    target_idx: int,
    trm_primary_idx: int,
) -> float:
    """
    Derive a ground-truth halt label from routing confidence.

    halt=1.0  Top-1 spectral score > HALT_CONFIDENT_THRESHOLD AND
              TRM agreed (or has no checkpoint yet).
    halt=0.0  Routing uncertain or TRM overruled.
    """
    top_score = max(spectral_vec) if spectral_vec else 0.0
    spectral_confident = top_score > HALT_CONFIDENT_THRESHOLD
    trm_agreed = (trm_primary_idx < 0) or (trm_primary_idx == target_idx)
    return 1.0 if (spectral_confident and trm_agreed) else 0.0


# ------------------------------------------------------------------ #
# TRMRoutingTraceWriter                                               #
# ------------------------------------------------------------------ #

class TRMRoutingTraceWriter:
    """
    Appends TRMTrainer-ready JSONL records from live run_workflow output.

    Usage (inside run_workflow per-sentence block)::

        writer = TRMRoutingTraceWriter()   # process-wide singleton

        writer.record(
            text=text,
            routing_context=routing_context,
            selected_domain=selected_domain,
            trm_lookup_result=trm_lookup_result,  # optional, for gating
        )
    """

    def __init__(
        self,
        train_path: str = "training_data/routing_traces.jsonl",
        eval_path:  str = "evaluation_data/routing_ground_truth.jsonl",
        gate_threshold: float = GATE_THRESHOLD,
        eval_fraction:  float = EVAL_FRACTION,
    ) -> None:
        self._train_path     = Path(train_path)
        self._eval_path      = Path(eval_path)
        self._gate_threshold = gate_threshold
        self._eval_fraction  = eval_fraction
        self._lock           = threading.Lock()
        self._train_path.parent.mkdir(parents=True, exist_ok=True)
        self._eval_path.parent.mkdir(parents=True, exist_ok=True)
        self._train_count = 0
        self._eval_count  = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        *,
        text: str,
        routing_context: Any,
        selected_domain: str,
        trm_lookup_result: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Build and (conditionally) write a TRMTrainer sample.

        Returns True if the sample was written, False if gated out.
        """
        # 1. Resolve target domain index
        target_idx = _domain_to_idx(selected_domain)
        if target_idx < 0:
            # Domain not in DOMAIN_LIST — skip (would poison training)
            return False

        # 2. Confidence gating
        halt_conf: float = 1.0
        trm_primary_idx: int = -1
        if trm_lookup_result:
            _raw_halt = trm_lookup_result.get("halt_confidence")
            halt_conf = float(_raw_halt) if _raw_halt is not None else 1.0
            _raw_idx  = trm_lookup_result.get("trm_primary_domain_idx")
            trm_primary_idx = int(_raw_idx) if _raw_idx is not None else -1

        trm_active = trm_primary_idx >= 0
        if trm_active:
            overruled = trm_primary_idx != target_idx
            uncertain = halt_conf < self._gate_threshold
            if not (uncertain or overruled):
                return False  # TRM confident and correct — skip

        # 3. Build token_ids
        token_ids = _encode_query(text)

        # 4. Build spectral_vec — read from the correct attribute path
        spectral_scores = _extract_spectral_scores(routing_context)
        selected_domains: List[str] = list(
            getattr(routing_context, "selected_domains", []) or []
        )
        spectral_vec = _spectral_vec_to_tensor(spectral_scores, selected_domains)

        # 5. predicate_family_id
        classification = str(
            getattr(routing_context, "classification", "UNKNOWN") or "UNKNOWN"
        ).upper()
        pred_family_id = PREDICATE_FAMILY_MAP.get(classification, _DEFAULT_PREDICATE_FAMILY)

        # 6. initial_domain_probs
        init_probs = _initial_domain_probs(spectral_vec)

        # 7. halt_label
        halt_label = _derive_halt_label(spectral_vec, target_idx, trm_primary_idx)

        rec: Dict[str, Any] = {
            "token_ids":            token_ids,
            "spectral_vec":         spectral_vec,
            "predicate_family_id":  pred_family_id,
            "initial_domain_probs": init_probs,
            "target_domain":        target_idx,
            "halt_label":           halt_label,
        }

        # 8. Route to train or eval file
        is_eval = random.random() < self._eval_fraction
        path = self._eval_path if is_eval else self._train_path

        with self._lock:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if is_eval:
                self._eval_count += 1
            else:
                self._train_count += 1

        return True

    def counts(self) -> Dict[str, int]:
        """Return current record counts for both files."""
        with self._lock:
            return {"train": self._train_count, "eval": self._eval_count}


# Process-wide singleton
trm_trace_writer = TRMRoutingTraceWriter()
