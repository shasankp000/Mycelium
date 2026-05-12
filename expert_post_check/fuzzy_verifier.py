"""
expert_post_check.fuzzy_verifier
=================================
Fuzzy semantic verifier.

Compares two answer strings using cosine similarity on sentence embeddings
(all-mpnet-base-v2 — already used elsewhere in Mycelium).

If cosine similarity falls below the threshold the multi-lens router
hint is attached to VerificationResult.conflict_hint so the caller
can optionally escalate to a broader domain search.

VerificationResult fields
-------------------------
    match           : bool   — True when similarity >= threshold
    similarity      : float  — raw cosine similarity [0, 1]
    threshold       : float  — threshold used for this run
    preferred       : str    — "trm" | "p6" | "conflict"
    confidence_vote : float  — +delta applied to TRM weight when match=True
    conflict_hint   : str    — suggestion for multi-lens escalation on mismatch
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.72
CONFIDENCE_VOTE_DELTA = 0.05   # RL reward signal magnitude on match
_EMBED_MODEL = "all-mpnet-base-v2"


@dataclass
class VerificationResult:
    match: bool
    similarity: float
    threshold: float
    preferred: str          # "trm" | "p6" | "conflict"
    confidence_vote: float  # positive on match, negative on mismatch
    conflict_hint: str = ""


class FuzzyVerifier:
    """Semantic similarity verifier for two answer strings."""

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        embed_model: str = _EMBED_MODEL,
    ) -> None:
        self._threshold = threshold
        self._embed_model = embed_model
        self._encoder = self._load_encoder()

    def verify(
        self,
        trm_answer: str,
        p6_answer: str,
        trm_confidence: float,
        p6_confidence: float,
        domain: str = "",
    ) -> VerificationResult:
        """Compare *trm_answer* and *p6_answer* semantically.

        Args:
            trm_answer:      Answer from the TRM-slot reasoner.
            p6_answer:       Answer from the 6-phase pipeline.
            trm_confidence:  Self-reported confidence of TRM-slot.
            p6_confidence:   Self-reported confidence of P6 pipeline.
            domain:          Domain name, used in conflict hint.

        Returns:
            A ``VerificationResult``.
        """
        # Edge cases: empty answers
        if not trm_answer and not p6_answer:
            return VerificationResult(
                match=False, similarity=0.0, threshold=self._threshold,
                preferred="conflict", confidence_vote=-CONFIDENCE_VOTE_DELTA,
                conflict_hint=f"Both reasoners returned empty answers for domain={domain}",
            )
        if not trm_answer:
            return VerificationResult(
                match=False, similarity=0.0, threshold=self._threshold,
                preferred="p6", confidence_vote=-CONFIDENCE_VOTE_DELTA,
                conflict_hint="TRM-slot returned empty answer",
            )
        if not p6_answer:
            return VerificationResult(
                match=False, similarity=0.0, threshold=self._threshold,
                preferred="trm", confidence_vote=CONFIDENCE_VOTE_DELTA,
                conflict_hint="P6 pipeline returned empty answer",
            )

        similarity = self._cosine_similarity(trm_answer, p6_answer)
        match = similarity >= self._threshold

        if match:
            # Prefer higher-confidence source on agreement
            preferred = "trm" if trm_confidence >= p6_confidence else "p6"
            confidence_vote = CONFIDENCE_VOTE_DELTA
            conflict_hint = ""
        else:
            # Prefer higher-confidence source but flag conflict
            preferred = "conflict"
            confidence_vote = -CONFIDENCE_VOTE_DELTA
            conflict_hint = (
                f"Semantic mismatch (sim={similarity:.3f} < {self._threshold}) "
                f"for domain={domain}. Consider multi-lens router escalation "
                f"across semantic/spectral/confidence lenses."
            )

        logger.info(
            "FuzzyVerifier: sim=%.3f threshold=%.2f match=%s preferred=%s",
            similarity, self._threshold, match, preferred,
        )
        return VerificationResult(
            match=match,
            similarity=similarity,
            threshold=self._threshold,
            preferred=preferred,
            confidence_vote=confidence_vote,
            conflict_hint=conflict_hint,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_encoder(self):
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(self._embed_model)
            logger.debug("FuzzyVerifier loaded encoder: %s", self._embed_model)
            return model
        except ImportError:
            logger.warning(
                "sentence-transformers not available. "
                "Falling back to character-level overlap similarity."
            )
            return None

    def _cosine_similarity(self, a: str, b: str) -> float:
        if self._encoder is not None:
            return self._cosine_embedding(a, b)
        return self._char_overlap(a, b)

    def _cosine_embedding(self, a: str, b: str) -> float:
        try:
            import numpy as np
            vecs = self._encoder.encode([a, b], convert_to_numpy=True)
            va, vb = vecs[0], vecs[1]
            norm_a = float(np.linalg.norm(va))
            norm_b = float(np.linalg.norm(vb))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(np.dot(va, vb) / (norm_a * norm_b))
        except Exception as exc:
            logger.warning("Embedding cosine failed: %s", exc)
            return self._char_overlap(a, b)

    @staticmethod
    def _char_overlap(a: str, b: str) -> float:
        """Jaccard character-bigram overlap as a fallback similarity."""
        def bigrams(s: str):
            s = s.lower()
            return set(s[i:i+2] for i in range(len(s) - 1))
        ba, bb = bigrams(a), bigrams(b)
        if not ba and not bb:
            return 1.0
        if not ba or not bb:
            return 0.0
        return len(ba & bb) / len(ba | bb)
