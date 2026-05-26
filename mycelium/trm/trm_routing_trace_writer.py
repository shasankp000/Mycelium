"""
Phase D — TRMRoutingTraceWriter
================================
Thread-safe JSONL writer that captures per-sentence routing decisions
from run_mycelium_workflow() and serialises them into the exact schema
expected by TRMTrainer.

Written to:
    training_data/routing_traces.jsonl       (80 % — training split)
    evaluation_data/routing_ground_truth.jsonl (20 % — eval split)

Schema of each record (matches TRMTrainer dataset format exactly)
-----------------------------------------------------------------
::

    {
        "token_ids":             list[int],  # length == context_len (64)
        "spectral_vec":          list[float],# length == n_domains   (8)
        "predicate_family_id":   int,        # routing_classification → index
        "initial_domain_probs":  list[float],# length == n_domains   (8)
        "target_domain":         int,        # selected_domain → domain index
        "halt_label":            float,      # 1.0 = halt (confident),
                                             # 0.0 = continue (ambiguous)
    }

halt_label derivation
---------------------
The halt signal encodes whether the model should commit to its current
domain prediction or continue reasoning:

    halt=1.0   The routing was confident and unambiguous:
               - Top-1 spectral score > HALT_CONFIDENT_THRESHOLD (0.50), AND
               - TRM (if active) agreed: primary_domain == target_domain.

    halt=0.0   The routing was uncertain or multi-domain:
               - Top-1 spectral score <= HALT_CONFIDENT_THRESHOLD, OR
               - TRM was overruled (primary_domain != target_domain).

This gives the halt head a real binary supervision signal tied to routing
confidence rather than the model's own domain accuracy at training time
(which would be near-zero early in training and cause the halt head to
always predict "don't halt").

Confidence gating (long-term robustness — §A)
---------------------------------------------
Only samples where TRM was uncertain (halt_confidence < GATE_THRESHOLD)
OR where TRM was overruled (primary_domain_idx != target_domain) are
written.  This prevents the training set being dominated by easy cases
the model already handles correctly and avoids feedback-loop degradation.

When TRM has no active checkpoint (primary_domain_idx == -1 for every
query), gating is bypassed entirely so the cold-start training set is
still populated.  This is the expected behaviour during initial runs
before any TRM weights exist.

Set GATE_THRESHOLD = 1.0 to disable gating and write every sample.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

# ------------------------------------------------------------------ #
# Constants matching TRMConfig defaults                               #
# ------------------------------------------------------------------ #

N_DOMAINS: int = 8
CONTEXT_LEN: int = 64
VOCAB_SIZE: int = 8192
GATE_THRESHOLD: float = float(os.getenv("TRM_GATE_THRESHOLD", "1.0"))
EVAL_FRACTION: float = 0.20  # 20 % of samples go to eval split

# Top-1 spectral score above this threshold → halt=1 (routing is confident).
HALT_CONFIDENT_THRESHOLD: float = float(
    os.getenv("TRM_HALT_CONFIDENT_THRESHOLD", "0.50")
)

# Fixed domain list — must match initialize_unified_experts() order.
# Indices 4-7 are reserved for future expert domains.
DOMAIN_LIST: List[str] = [
    "music",       # 0
    "physics",     # 1
    "chemistry",   # 2
    "medical",     # 3
    "__reserved4", # 4
    "__reserved5", # 5
    "__reserved6", # 6
    "__reserved7", # 7
]

# routing_classification → predicate_family_id int
PREDICATE_FAMILY_MAP: Dict[str, int] = {
    "ATTRIBUTE_ONLY":       0,
    "CAUSAL":               1,
    "TEMPORAL":             2,
    "COMPARATIVE":          3,
    "DEFINITIONAL":         4,
    "PROCEDURAL":           5,
    "FACTUAL":              6,
    "HYPOTHETICAL":         7,
    "NEGATION":             8,
    "RELATIONAL":           9,
    "UNKNOWN":              10,
    # catch-all for any future classifications
}
_DEFAULT_PREDICATE_FAMILY: int = 10  # UNKNOWN

# One-shot debug flag: print spectral_scores details on the first record() call.
_SPECTRAL_DEBUG_DONE: bool = False


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
    # Pad with 0 (pad_token_id) to CONTEXT_LEN
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


def _spectral_vec_to_tensor(
    spectral_scores: Any,
    selected_domains: List[str],
) -> List[float]:
    """
    Build a float[N_DOMAINS] vector from routing_context spectral scores.

    Tries to read a dict / list / object from spectral_scores.
    Falls back to a soft one-hot over selected_domains if unavailable
    or if the list contains non-numeric values (e.g. np.str_ domain names
    instead of scores).
    Always normalises to sum == 1.0.
    """
    vec = [0.0] * N_DOMAINS

    # Case 1: dict keyed by domain name
    if isinstance(spectral_scores, dict):
        for domain, score in spectral_scores.items():
            idx = _domain_to_idx(domain)
            if 0 <= idx < N_DOMAINS:
                vec[idx] = float(score)

    # Case 2: list/tuple of length N_DOMAINS whose elements are numeric.
    # Guard: if any element is non-numeric (e.g. np.str_ domain name strings
    # passed instead of float scores) skip this branch entirely and fall
    # through to the soft one-hot fallback below.
    elif (
        isinstance(spectral_scores, (list, tuple))
        and len(spectral_scores) == N_DOMAINS
        and all(_is_numeric(v) for v in spectral_scores)
    ):
        vec = [float(v) for v in spectral_scores]

    # Case 3: object with .scores / .domain_scores attribute
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

    halt=1.0  The routing is confident and unambiguous:
              - The top spectral score exceeds HALT_CONFIDENT_THRESHOLD, AND
              - Either TRM is not active (trm_primary_idx == -1) or TRM
                agreed with the selected domain.

    halt=0.0  The routing is uncertain or TRM was overruled.

    Using spectral confidence (not the model's own domain accuracy) means
    the halt head always receives a real supervision signal, even at the
    start of training when domain accuracy is near-chance.
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

    Parameters
    ----------
    train_path : str
        Path to training JSONL file.
    eval_path : str
        Path to evaluation JSONL file.
    gate_threshold : float
        Only write samples where halt_confidence < gate_threshold OR
        TRM was overruled.  Default 1.0 = write everything.
    eval_fraction : float
        Fraction of accepted samples routed to eval file.  Default 0.20.
    """

    def __init__(
        self,
        train_path: str = "training_data/routing_traces.jsonl",
        eval_path: str = "evaluation_data/routing_ground_truth.jsonl",
        gate_threshold: float = GATE_THRESHOLD,
        eval_fraction: float = EVAL_FRACTION,
    ) -> None:
        self._train_path = Path(train_path)
        self._eval_path  = Path(eval_path)
        self._gate_threshold = gate_threshold
        self._eval_fraction  = eval_fraction
        self._lock = threading.Lock()
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

        Gating is bypassed entirely when TRM has no active checkpoint
        (trm_primary_idx == -1), i.e. during cold-start runs before any
        weights exist.  This ensures the training set is populated on
        first runs so that TRMTrainer can actually bootstrap.
        """
        global _SPECTRAL_DEBUG_DONE

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
            _raw_idx = trm_lookup_result.get("trm_primary_domain_idx")
            trm_primary_idx = int(_raw_idx) if _raw_idx is not None else -1

        # When TRM has no checkpoint, primary_domain_idx is always -1.
        # Bypass gating completely so cold-start runs still produce samples.
        trm_active = trm_primary_idx >= 0
        if trm_active:
            overruled = trm_primary_idx != target_idx
            uncertain = halt_conf < self._gate_threshold
            if not (uncertain or overruled):
                return False  # TRM was confident and correct — skip
        # else: TRM not active → always write

        # 3. Build token_ids
        token_ids = _encode_query(text)

        # 4. Build spectral_vec
        spectral_scores = getattr(routing_context, "spectral_scores", None)
        selected_domains: List[str] = list(
            getattr(routing_context, "selected_domains", []) or []
        )

        # ------------------------------------------------------------------ #
        # ONE-SHOT DIAGNOSTIC — remove after spectral_scores structure is     #
        # confirmed and the spectral_vec parsing is fixed.                    #
        # ------------------------------------------------------------------ #
        if not _SPECTRAL_DEBUG_DONE:
            _SPECTRAL_DEBUG_DONE = True
            print("[TRM-SPECTRAL-DEBUG] spectral_scores type  :", type(spectral_scores), file=sys.stderr)
            print("[TRM-SPECTRAL-DEBUG] spectral_scores value :", repr(spectral_scores)[:300], file=sys.stderr)
            print("[TRM-SPECTRAL-DEBUG] spectral_scores attrs :", [a for a in dir(spectral_scores) if not a.startswith("__")], file=sys.stderr)
            print("[TRM-SPECTRAL-DEBUG] selected_domains      :", selected_domains, file=sys.stderr)
            print("[TRM-SPECTRAL-DEBUG] routing_context attrs :", [a for a in dir(routing_context) if not a.startswith("__")][:30], file=sys.stderr)
        # ------------------------------------------------------------------ #

        spectral_vec = _spectral_vec_to_tensor(spectral_scores, selected_domains)

        # 5. predicate_family_id
        classification = str(
            getattr(routing_context, "classification", "UNKNOWN") or "UNKNOWN"
        ).upper()
        pred_family_id = PREDICATE_FAMILY_MAP.get(classification, _DEFAULT_PREDICATE_FAMILY)

        # 6. initial_domain_probs
        init_probs = _initial_domain_probs(spectral_vec)

        # 7. halt_label — derived from spectral routing confidence, not model accuracy
        halt_label = _derive_halt_label(spectral_vec, target_idx, trm_primary_idx)

        record: Dict[str, Any] = {
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
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
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
