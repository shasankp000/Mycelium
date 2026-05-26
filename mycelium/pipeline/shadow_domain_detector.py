"""
Shadow Domain Detection System
================================
Detects out-of-distribution (OOD) queries that score below a confidence
threshold across ALL live expert domains and provisionally groups them into
"shadow" clusters that accumulate evidence until they are mature enough to
be promoted to a real expert domain.

Design goals
------------
* **Zero breaking changes** — imported and called optionally; returns None
  gracefully when SentenceTransformer is unavailable.
* **Embedding-first clustering** — semantically related OOD queries merge
  into the same shadow cluster via centroid proximity so the evidence count
  for a *concept* accumulates rather than fragmenting per-query.
* **Promote signal** — once a cluster's evidence count reaches
  SHADOW_PROMOTE_THRESHOLD, observe() returns a ShadowDomainSignal with
  status='PROMOTE_TO_EXPERT' so the caller can trigger expert creation.
* **Thread-safe** — internal state is guarded by threading.RLock.
* **Graceful degradation** — when no embedding model is available the
  cluster fingerprint falls back to a lexical hash of normalised tokens,
  which is coarser but still functional.

Public API
----------
    detector = ShadowDomainDetector()
    signal   = detector.observe(text, fusion_scores)   # call after routing
    registry = detector.get_shadow_registry()          # inspection
    detector.clear_stale_shadows(max_age_hours=48)     # maintenance

ShadowDomainSignal
------------------
    {
        "shadow_id":   "shadow_a3f9c1b7",
        "status":      "ACCUMULATING" | "PROMOTE_TO_EXPERT",
        "evidence":    12,
        "top_tokens":  ["iphone", "titanium", "pro"],
        "centroid":    <np.ndarray | None>,
    }
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuneable constants  (overridable via config_loader when wired up)
# ---------------------------------------------------------------------------

# A query is OOD if its BEST score across all live domains is below this.
SHADOW_SCORE_THRESHOLD: float = 0.25

# Two shadow clusters merge when their centroid cosine similarity >= this.
SHADOW_MERGE_THRESHOLD: float = 0.82

# How many OOD observations before a cluster is promoted to expert status.
SHADOW_PROMOTE_THRESHOLD: int = 8

# Maximum number of shadow clusters kept in memory at once.
SHADOW_MAX_CLUSTERS: int = 256


# ---------------------------------------------------------------------------
# Internal cluster state
# ---------------------------------------------------------------------------

@dataclass
class _ShadowCluster:
    shadow_id: str
    centroid: Optional[np.ndarray]          # running mean embedding (may be None)
    evidence: int = 0
    top_tokens: List[str] = field(default_factory=list)
    last_seen: float = field(default_factory=time.monotonic)
    created_at: float = field(default_factory=time.monotonic)

    def update(self, embedding: Optional[np.ndarray], tokens: List[str]) -> None:
        self.evidence += 1
        self.last_seen = time.monotonic()
        # Running mean centroid
        if embedding is not None:
            if self.centroid is None:
                self.centroid = embedding.copy()
            else:
                n = self.evidence
                self.centroid = self.centroid * ((n - 1) / n) + embedding * (1.0 / n)
        # Accumulate top tokens (union, capped at 16)
        existing = set(self.top_tokens)
        for t in tokens:
            if t not in existing:
                self.top_tokens.append(t)
                existing.add(t)
        self.top_tokens = self.top_tokens[:16]


# ---------------------------------------------------------------------------
# Signal returned to callers
# ---------------------------------------------------------------------------

@dataclass
class ShadowDomainSignal:
    shadow_id: str
    status: str                              # 'ACCUMULATING' | 'PROMOTE_TO_EXPERT'
    evidence: int
    top_tokens: List[str]
    centroid: Optional[np.ndarray] = None

    def as_dict(self) -> Dict:
        return {
            "shadow_id":  self.shadow_id,
            "status":     self.status,
            "evidence":   self.evidence,
            "top_tokens": self.top_tokens,
        }


# ---------------------------------------------------------------------------
# Embedding helper (lazy, singleton per model_name)
# ---------------------------------------------------------------------------

_embed_cache: Dict[str, object] = {}   # model_name -> SentenceTransformer
_embed_lock = threading.Lock()


def _get_embedder(model_name: str = "all-MiniLM-L6-v2") -> Optional[object]:
    """Return a (possibly cached) SentenceTransformer, or None."""
    if model_name in _embed_cache:
        return _embed_cache[model_name]
    with _embed_lock:
        if model_name in _embed_cache:
            return _embed_cache[model_name]
        try:
            from mycelium.pipeline.model_registry import get_model
            model = get_model(model_name, model_type="sentence_transformer", device="cpu")
        except Exception:
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(model_name, device="cpu")
            except Exception:
                model = None
        _embed_cache[model_name] = model
        return model


def _embed(text: str, model_name: str = "all-MiniLM-L6-v2") -> Optional[np.ndarray]:
    """Embed *text* into a unit-normed float32 vector, or return None."""
    model = _get_embedder(model_name)
    if model is None:
        return None
    try:
        vec = np.array(model.encode([text])[0], dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 1e-10 else vec
    except Exception:
        return None


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two already-normalised vectors."""
    dot = float(np.dot(a, b))
    na  = float(np.linalg.norm(a))
    nb  = float(np.linalg.norm(b))
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Lexical fingerprint fallback
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _lexical_fingerprint(tokens: List[str]) -> str:
    """Stable 8-char hex fingerprint for a bag-of-words (sorted, deduped)."""
    key = " ".join(sorted(set(tokens)))
    return hashlib.sha256(key.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Main detector class
# ---------------------------------------------------------------------------

class ShadowDomainDetector:
    """
    Stateful detector that tracks OOD queries and clusters them into
    provisional "shadow" domains until they mature for expert promotion.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        score_threshold: float = SHADOW_SCORE_THRESHOLD,
        merge_threshold: float = SHADOW_MERGE_THRESHOLD,
        promote_threshold: int = SHADOW_PROMOTE_THRESHOLD,
        max_clusters: int = SHADOW_MAX_CLUSTERS,
    ) -> None:
        self._model_name      = model_name
        self._score_threshold = score_threshold
        self._merge_threshold = merge_threshold
        self._promote_threshold = promote_threshold
        self._max_clusters    = max_clusters
        self._clusters: Dict[str, _ShadowCluster] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def observe(
        self,
        text: str,
        fusion_scores: Dict[str, float],
        score_threshold: Optional[float] = None,
    ) -> Optional[ShadowDomainSignal]:
        """
        Inspect *fusion_scores* from the router.

        If the best score across all live domains is below *score_threshold*
        (default: self._score_threshold) the query is treated as OOD and
        assigned to the nearest shadow cluster (or a new one created).

        Parameters
        ----------
        text          : Raw query string.
        fusion_scores : Dict[domain_name -> float] from fusion_engine /
                        multi_lens_route lens1_candidates converted to dict.
        score_threshold : Override the instance default.

        Returns
        -------
        ShadowDomainSignal  if the query is OOD.
        None                if at least one domain score >= threshold
                            (query is within known territory).
        """
        threshold = score_threshold if score_threshold is not None else self._score_threshold

        # Check if query is within known domain territory
        best_score = max(fusion_scores.values()) if fusion_scores else 0.0
        if best_score >= threshold:
            return None   # not OOD — normal routing handles it

        tokens = _tokenize(text)
        if not tokens:
            return None

        embedding = _embed(text, model_name=self._model_name)

        with self._lock:
            cluster = self._find_or_create_cluster(tokens, embedding)
            cluster.update(embedding, tokens)

            status = (
                "PROMOTE_TO_EXPERT"
                if cluster.evidence >= self._promote_threshold
                else "ACCUMULATING"
            )

            if status == "PROMOTE_TO_EXPERT":
                logger.info(
                    "ShadowDomain %s ready for promotion — %d observations, tokens: %s",
                    cluster.shadow_id, cluster.evidence, cluster.top_tokens[:6],
                )

            return ShadowDomainSignal(
                shadow_id  = cluster.shadow_id,
                status     = status,
                evidence   = cluster.evidence,
                top_tokens = list(cluster.top_tokens),
                centroid   = cluster.centroid,
            )

    # ------------------------------------------------------------------
    # Inspection / maintenance
    # ------------------------------------------------------------------

    def get_shadow_registry(self) -> Dict[str, Dict]:
        """Return a serialisable snapshot of all current shadow clusters."""
        with self._lock:
            return {
                sid: {
                    "evidence":   c.evidence,
                    "top_tokens": c.top_tokens,
                    "age_s":      round(time.monotonic() - c.created_at, 1),
                    "last_seen_s": round(time.monotonic() - c.last_seen, 1),
                    "status": (
                        "PROMOTE_TO_EXPERT"
                        if c.evidence >= self._promote_threshold
                        else "ACCUMULATING"
                    ),
                }
                for sid, c in self._clusters.items()
            }

    def clear_stale_shadows(
        self,
        max_age_hours: float = 48.0,
    ) -> int:
        """Evict clusters not seen within *max_age_hours*. Returns count removed."""
        cutoff = time.monotonic() - max_age_hours * 3600.0
        with self._lock:
            stale = [sid for sid, c in self._clusters.items() if c.last_seen < cutoff]
            for sid in stale:
                del self._clusters[sid]
            return len(stale)

    def reset(self) -> None:
        """Clear all clusters (useful for testing)."""
        with self._lock:
            self._clusters.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_or_create_cluster(
        self,
        tokens: List[str],
        embedding: Optional[np.ndarray],
    ) -> _ShadowCluster:
        """Return the best-matching cluster, creating one if none is close enough."""

        # --- Embedding-based matching (preferred) ---
        if embedding is not None and self._clusters:
            best_sid: Optional[str] = None
            best_sim: float = -1.0
            for sid, cluster in self._clusters.items():
                if cluster.centroid is None:
                    continue
                sim = _cosine(embedding, cluster.centroid)
                if sim > best_sim:
                    best_sim = sim
                    best_sid = sid
            if best_sid is not None and best_sim >= self._merge_threshold:
                return self._clusters[best_sid]

        # --- Lexical fingerprint fallback ---
        fp = _lexical_fingerprint(tokens)
        shadow_id = f"shadow_{fp}"

        if shadow_id in self._clusters:
            return self._clusters[shadow_id]

        # --- Create new cluster ---
        self._maybe_evict()
        cluster = _ShadowCluster(shadow_id=shadow_id, centroid=None)
        self._clusters[shadow_id] = cluster
        logger.debug("ShadowDomain: new cluster %s — tokens: %s", shadow_id, tokens[:8])
        return cluster

    def _maybe_evict(self) -> None:
        """If at capacity, evict the least-recently-seen cluster."""
        if len(self._clusters) < self._max_clusters:
            return
        oldest_sid = min(self._clusters, key=lambda s: self._clusters[s].last_seen)
        del self._clusters[oldest_sid]
        logger.debug("ShadowDomain: evicted stale cluster %s", oldest_sid)


# ---------------------------------------------------------------------------
# Module-level singleton (shared across all routing calls in a process)
# ---------------------------------------------------------------------------

_DETECTOR: Optional[ShadowDomainDetector] = None
_DETECTOR_LOCK = threading.Lock()


def get_detector(
    model_name: str = "all-MiniLM-L6-v2",
    score_threshold: float = SHADOW_SCORE_THRESHOLD,
    merge_threshold: float = SHADOW_MERGE_THRESHOLD,
    promote_threshold: int = SHADOW_PROMOTE_THRESHOLD,
) -> ShadowDomainDetector:
    """Return the process-level ShadowDomainDetector singleton."""
    global _DETECTOR
    if _DETECTOR is not None:
        return _DETECTOR
    with _DETECTOR_LOCK:
        if _DETECTOR is None:
            _DETECTOR = ShadowDomainDetector(
                model_name        = model_name,
                score_threshold   = score_threshold,
                merge_threshold   = merge_threshold,
                promote_threshold = promote_threshold,
            )
    return _DETECTOR


def reset_detector() -> None:
    """Reset the singleton (primarily for testing)."""
    global _DETECTOR
    with _DETECTOR_LOCK:
        if _DETECTOR is not None:
            _DETECTOR.reset()
        _DETECTOR = None
