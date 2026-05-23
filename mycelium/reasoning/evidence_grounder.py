"""
Layer 4 — Evidence Grounder
============================
Takes the predicate sets produced by Layer 3 (PredicateGenerator) and
attempts to ground each predicate against available local evidence sources.

Evidence tiers
--------------
    TIER_1 — IRGraph node pool (in-memory, zero latency)
        The TRM GraphStore already holds IRGraphs from previous queries.
        A predicate is "grounded" at Tier 1 if any stored IRNode has a
        canonical_form that semantically matches the predicate's
        subject + relation + object triple.
        Matching is done via predicate_family equality + Jaccard overlap
        on label tokens (cosine similarity if embedding_fn is available).

    TIER_2 — External knowledge (STUB, always returns None)
        Placeholder for a future web-retrieval or vector-DB grounding step.
        Returning None means the predicate is ungrounded at Tier 2;
        HypothesisEvaluator (Layer 5) treats ungrounded predicates as
        having low evidence weight.

Grounding result per predicate
------------------------------
    grounded      : bool   — True if at least Tier 1 found a match
    tier          : int    — 1 or 2 (tier that produced the match)
    evidence_score: float  — [0,1] strength of the match
    match_node_id : str    — id of the matching IRNode (Tier 1 only)
    match_label   : str    — label of the matching node
    notes         : str    — free-text explanation

Output dict (returned by .ground())
-------------------------------------
    grounded_predicates   : list[dict]   — predicates with grounding info added
    evidence_score        : float        — mean score across all predicates
    n_grounded            : int
    n_ungrounded          : int
    domain                : str
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------

def _jaccard(a: str, b: str) -> float:
    """Word-level Jaccard similarity between two strings."""
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa and not sb:
        return 1.0
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


def _cosine(vec_a: List[float], vec_b: List[float]) -> Optional[float]:
    """Cosine similarity between two embedding vectors.  Returns None on error."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return None
    try:
        import math
        dot = sum(x * y for x, y in zip(vec_a, vec_b))
        na  = math.sqrt(sum(x * x for x in vec_a))
        nb  = math.sqrt(sum(x * x for x in vec_b))
        if na < 1e-10 or nb < 1e-10:
            return None
        return dot / (na * nb)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class EvidenceGrounder:
    """
    Layer 4: ground predicates against available evidence sources.

    Parameters
    ----------
    graph_store : GraphStore, optional
        TRM GraphStore instance.  Used for Tier 1 (in-memory) grounding.
        If None, all predicates are marked ungrounded.
    embedding_fn : callable, optional
        (str) -> list[float].  When provided, cosine similarity is used
        instead of Jaccard for matching; this improves recall significantly
        for paraphrased sub-claims.
    tier1_jaccard_threshold : float
        Minimum Jaccard similarity to count as a Tier 1 match (default 0.30).
    tier1_cosine_threshold : float
        Minimum cosine similarity for embedding-based Tier 1 match (default 0.72).
    """

    def __init__(
        self,
        *,
        graph_store: Optional[Any] = None,
        embedding_fn: Optional[Any] = None,
        tier1_jaccard_threshold: float = 0.30,
        tier1_cosine_threshold: float = 0.72,
    ) -> None:
        self._store = graph_store
        self._embed = embedding_fn
        self._jac_thresh   = tier1_jaccard_threshold
        self._cos_thresh   = tier1_cosine_threshold
        self._node_cache: Optional[List[Any]] = None   # lazy-loaded from store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ground(
        self,
        predicates: Dict,
        domain: str,
        dag_context: Optional[Dict] = None,
    ) -> Dict:
        """
        Ground every predicate in the predicate dict against evidence.

        Parameters
        ----------
        predicates : dict
            Output of PredicateGenerator.generate().  Expected key:
            ``all_predicates`` — list of predicate dicts.
        domain : str
            Routing domain hint (used for Tier 1 family filtering).
        dag_context : dict, optional
            dag_context from Layer 2 (for sub-claim node pool fallback).

        Returns
        -------
        dict  — see module docstring for schema.
        """
        all_preds: List[Dict] = predicates.get("all_predicates", [])
        grounded_preds: List[Dict] = []
        scores: List[float] = []

        # Build a local node pool: store nodes + sub_claim nodes from dag_ctx
        node_pool = self._build_node_pool(dag_context)

        for pred in all_preds:
            result = self._ground_one(pred, node_pool)
            enriched = dict(pred)
            enriched.update(result)
            grounded_preds.append(enriched)
            scores.append(result["evidence_score"])

        n_grounded   = sum(1 for p in grounded_preds if p.get("grounded"))
        n_ungrounded = len(grounded_preds) - n_grounded
        mean_score   = (sum(scores) / len(scores)) if scores else 0.0

        logger.debug(
            "EvidenceGrounder: %d/%d grounded, mean_score=%.3f, domain=%s",
            n_grounded, len(grounded_preds), mean_score, domain,
        )

        return {
            "grounded_predicates": grounded_preds,
            "evidence_score":      mean_score,
            "n_grounded":          n_grounded,
            "n_ungrounded":        n_ungrounded,
            "domain":              domain,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_node_pool(self, dag_context: Optional[Dict]) -> List[Dict]:
        """
        Collect candidate nodes from GraphStore + dag_context sub_claims.
        Returns a list of lightweight dicts: {id, label, family, embedding}.
        """
        pool: List[Dict] = []

        # Source 1: sub_claims already in dag_context (zero-cost)
        if dag_context:
            for sc in dag_context.get("sub_claims", []):
                pool.append({
                    "id":     sc.get("id", ""),
                    "label":  sc.get("label", ""),
                    "family": sc.get("predicate_family", ""),
                    "embedding": [],
                })

        # Source 2: GraphStore (Tier 1)
        if self._store is not None:
            try:
                all_graphs = self._store.list_all()   # returns list of IRGraph
                for graph in all_graphs[:50]:          # cap at 50 graphs for perf
                    for node in getattr(graph, "nodes", [])[:10]:
                        sig = getattr(node, "semantic_signature", None)
                        pool.append({
                            "id":    node.id,
                            "label": node.label,
                            "family": sig.predicate_family if sig else "",
                            "embedding": (
                                list(sig.embedding_signature)
                                if sig and sig.embedding_signature else []
                            ),
                        })
            except Exception as exc:
                logger.debug("EvidenceGrounder: GraphStore query failed (%s)", exc)

        return pool

    def _ground_one(self, pred: Dict, node_pool: List[Dict]) -> Dict:
        """
        Try to ground a single predicate against the node pool.
        Returns a grounding result dict.
        """
        query_text = f"{pred.get('subject','')} {pred.get('relation','')} {pred.get('object','')}"
        query_emb: List[float] = []
        if self._embed is not None:
            try:
                query_emb = list(self._embed(query_text))
            except Exception:
                pass

        best_score  = 0.0
        best_node_id = ""
        best_label  = ""

        for candidate in node_pool:
            # Family filter: same predicate family preferred, but don't
            # hard-exclude — cross-family matches still contribute evidence.
            family_match = (
                pred.get("predicate_family", "") == candidate.get("family", "")
            )
            family_bonus = 0.1 if family_match else 0.0

            # Similarity
            cand_emb = candidate.get("embedding", [])
            sim: float
            if query_emb and cand_emb:
                cos = _cosine(query_emb, cand_emb)
                sim = cos if cos is not None else _jaccard(query_text, candidate["label"])
                threshold = self._cos_thresh
            else:
                sim = _jaccard(query_text, candidate["label"])
                threshold = self._jac_thresh

            adjusted = min(1.0, sim + family_bonus)

            if adjusted > best_score:
                best_score   = adjusted
                best_node_id = candidate["id"]
                best_label   = candidate["label"]

            if best_score >= threshold:
                break  # good enough, stop scanning

        # Determine grounding tier
        threshold = (
            self._cos_thresh if (query_emb and best_node_id)
            else self._jac_thresh
        )
        grounded = best_score >= threshold

        return {
            "grounded":      grounded,
            "tier":          1 if grounded else 0,
            "evidence_score": best_score,
            "match_node_id": best_node_id if grounded else "",
            "match_label":   best_label   if grounded else "",
            "notes": (
                f"Tier 1 match (score={best_score:.3f})"
                if grounded
                else f"Ungrounded (best_score={best_score:.3f})"
            ),
        }
