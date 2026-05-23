"""
Layer 3 — Predicate Generator
==============================
Takes the structured dag_context produced by Layer 2 (DAGDecomposer) and
generates two predicate sets:

    positive_predicates  — claims the query is asserting as true
    negative_predicates  — claims the query is denying or negating
    neutral_predicates   — relational claims with no clear polarity

Each predicate is a dict with keys:
    subject       : str   — the entity being claimed about
    relation      : str   — the relationship type (from RELATION_TYPES)
    object        : str   — the object/outcome of the claim
    polarity      : str   — POSITIVE | NEGATIVE | NEUTRAL
    predicate_family : str  — inherited from dag_context
    source_node_id : str   — id of the IRNode this came from
    confidence    : float  — inherited from source node

Design notes
------------
- Works entirely from the dag_context dict produced by TRMOODFallback
  _run_layers_1_2(), so it never needs to touch IRNode/IRGraph objects
  directly.  This keeps Layers 3-6 decoupled from the IR type hierarchy.

- Predicate extraction uses the canonical_form strings from each
  sub-claim node.  The canonical_form already encodes
  FAMILY::subject::predicate::object::depthN (Phase B §46 format),
  so parsing it is deterministic and needs no ML model.

- When canonical_form is unavailable (older nodes), the fallback parses
  the raw label string using the same negation/scope token lists that
  ContradictionClassifier uses, keeping the vocabulary consistent.

- Polarity is determined by:
    1. Presence of _NEGATION_TOKENS in the label → NEGATIVE
    2. Presence of _POSITIVE_TOKENS  → POSITIVE
    3. Anything else                 → NEUTRAL
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# Token vocabularies (kept in sync with ContradictionClassifier)
# ---------------------------------------------------------------------------

_NEGATION_TOKENS = [
    "not", "no", "never", "neither", "nor", "cannot", "can't", "won't",
    "doesn't", "does not", "is not", "isn't", "are not", "aren't",
    "fails to", "unable to", "lack", "lacks", "without", "prevents",
    "inhibits", "blocks", "stops", "reduces risk", "decreases",
]

_POSITIVE_TOKENS = [
    "helps", "improves", "increases", "enhances", "boosts", "promotes",
    "causes", "triggers", "leads to", "results in", "produces",
    "supports", "facilitates", "enables", "strengthens", "accelerates",
    "treats", "cures", "prevents",  # 'prevents' is positive toward harm
    "correlates", "associated with", "linked to",
]

# Relation type labels — broad enough to cover all predicate families
RELATION_TYPES = [
    "CAUSES", "PREVENTS", "CORRELATES_WITH", "INHIBITS", "PROMOTES",
    "IS_A", "PART_OF", "REQUIRES", "PRODUCES", "OPPOSES", "SUPPORTS",
    "UNKNOWN",
]

# Canonical form pattern: FAMILY::subject::predicate::object::depthN
_CANONICAL_RE = re.compile(
    r"^(?P<family>[A-Z_]+)::(?P<subject>[^:]+)::(?P<predicate>[^:]+)::(?P<obj>[^:]+)::depth(?P<depth>\d+)$"
)


# ---------------------------------------------------------------------------
# Predicate dataclass (plain dict for zero-dependency cross-layer passing)
# ---------------------------------------------------------------------------

def _make_predicate(
    subject: str,
    relation: str,
    obj: str,
    polarity: str,
    predicate_family: str,
    source_node_id: str,
    confidence: float,
) -> Dict:
    return {
        "subject": subject.strip(),
        "relation": relation,
        "object": obj.strip(),
        "polarity": polarity,
        "predicate_family": predicate_family,
        "source_node_id": source_node_id,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PredicateGenerator:
    """
    Layer 3: generate structured predicate sets from a dag_context dict.

    Parameters
    ----------
    max_predicates_per_node : int
        Cap on predicates extracted from a single sub-claim node (default 3).
    """

    def __init__(self, max_predicates_per_node: int = 3) -> None:
        self.max_per_node = max_predicates_per_node

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, dag_context: Dict) -> Dict:
        """
        Extract predicates from the dag_context produced by Layer 2.

        Parameters
        ----------
        dag_context : dict
            Output of TRMOODFallback._run_layers_1_2().  Expected keys:
                sub_claims        : list[dict]  (from IRGraph.nodes)
                predicate_family  : str
                canonical_form    : str
                equivalence_family: list[str]
                domain            : str

        Returns
        -------
        dict with keys:
            positive_predicates : list[dict]
            negative_predicates : list[dict]
            neutral_predicates  : list[dict]
            all_predicates      : list[dict]  (union, for downstream ease)
            predicate_family    : str
            domain              : str
            n_nodes_processed   : int
        """
        positive: List[Dict] = []
        negative: List[Dict] = []
        neutral:  List[Dict] = []

        sub_claims      = dag_context.get("sub_claims", [])
        root_family     = dag_context.get("predicate_family", "UNKNOWN")
        domain          = dag_context.get("domain", "unknown")
        canonical_form  = dag_context.get("canonical_form", "")
        equiv_family    = dag_context.get("equivalence_family", [])

        # Always process the root canonical form first
        if canonical_form:
            preds = self._extract_from_canonical(canonical_form, root_family,
                                                  "root", 1.0)
            if not preds:
                preds = self._extract_from_label(
                    dag_context.get("root_label", canonical_form),
                    root_family, "root", 1.0
                )
            for p in preds:
                self._bucket(p, positive, negative, neutral)

        # Process each sub-claim node
        nodes_processed = 0
        for node in sub_claims:
            node_id   = node.get("id", "")
            label     = node.get("label", "")
            family    = node.get("predicate_family", root_family)
            conf      = node.get("confidence", 0.5)

            # Try canonical_form if embedded in label (Phase B format)
            preds = self._extract_from_canonical(label, family, node_id, conf)
            if not preds:
                preds = self._extract_from_label(label, family, node_id, conf)

            count = 0
            for p in preds:
                if count >= self.max_per_node:
                    break
                self._bucket(p, positive, negative, neutral)
                count += 1

            nodes_processed += 1

        # Also mine the equivalence family of the root for extra coverage
        for equiv in equiv_family[:3]:
            preds = self._extract_from_label(equiv, root_family, "equiv", 0.7)
            for p in preds[:1]:   # at most 1 per equiv form
                self._bucket(p, positive, negative, neutral)

        all_predicates = positive + negative + neutral
        logger.debug(
            "PredicateGenerator: +%d -%d ~%d predicates from %d nodes",
            len(positive), len(negative), len(neutral), nodes_processed,
        )

        return {
            "positive_predicates": positive,
            "negative_predicates": negative,
            "neutral_predicates":  neutral,
            "all_predicates":      all_predicates,
            "predicate_family":    root_family,
            "domain":              domain,
            "n_nodes_processed":   nodes_processed,
        }

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------

    def _extract_from_canonical(
        self,
        text: str,
        family: str,
        node_id: str,
        conf: float,
    ) -> List[Dict]:
        """Parse FAMILY::subject::predicate::object::depthN canonical form."""
        m = _CANONICAL_RE.match(text.strip())
        if not m:
            return []

        subject  = m.group("subject")
        predicate_str = m.group("predicate")
        obj      = m.group("obj")
        c_family = m.group("family")

        relation = self._map_relation(predicate_str)
        polarity = self._polarity(predicate_str + " " + obj)

        return [
            _make_predicate(
                subject=subject,
                relation=relation,
                obj=obj,
                polarity=polarity,
                predicate_family=c_family or family,
                source_node_id=node_id,
                confidence=conf,
            )
        ]

    def _extract_from_label(
        self,
        label: str,
        family: str,
        node_id: str,
        conf: float,
    ) -> List[Dict]:
        """Heuristic extraction from raw label text."""
        if not label:
            return []

        label_l = label.lower().strip()
        tokens  = label_l.split()

        # Rough subject/object split: take first noun phrase before the
        # first verb token as subject, rest as object.
        # This is intentionally simple — Phase B SRL already ran the
        # heavy lifting; here we just recover a flat representation.
        split_idx = len(tokens) // 2
        for i, tok in enumerate(tokens[1:], 1):
            # First obvious verb-ish token
            if any(tok.startswith(vp) for vp in [
                "help", "caus", "treat", "prev", "incr", "decr",
                "improv", "reduc", "enhanc", "block", "trigg",
            ]):
                split_idx = i
                break

        subject = " ".join(tokens[:split_idx]) or label_l
        obj     = " ".join(tokens[split_idx:]) or ""
        relation = self._map_relation(label_l)
        polarity = self._polarity(label_l)

        return [
            _make_predicate(
                subject=subject,
                relation=relation,
                obj=obj,
                polarity=polarity,
                predicate_family=family,
                source_node_id=node_id,
                confidence=conf,
            )
        ]

    # ------------------------------------------------------------------
    # Polarity + relation helpers
    # ------------------------------------------------------------------

    def _polarity(self, text: str) -> str:
        text_l = text.lower()
        if any(neg in text_l for neg in _NEGATION_TOKENS):
            return "NEGATIVE"
        if any(pos in text_l for pos in _POSITIVE_TOKENS):
            return "POSITIVE"
        return "NEUTRAL"

    def _map_relation(self, text: str) -> str:
        text_l = text.lower()
        mapping = [
            (["cause", "leads to", "results in", "triggers", "induces"], "CAUSES"),
            (["prevent", "stop", "block", "inhibit", "reduce risk"],     "PREVENTS"),
            (["correlat", "associat", "linked"],                          "CORRELATES_WITH"),
            (["inhibit", "suppress", "impair", "worsen"],                 "INHIBITS"),
            (["promot", "facilitat", "support", "boost", "enhanc"],       "PROMOTES"),
            (["is a", "are a", "type of", "kind of", "form of"],          "IS_A"),
            (["part of", "component of", "subset of"],                    "PART_OF"),
            (["requir", "needs", "depends on"],                           "REQUIRES"),
            (["produc", "generat", "creat", "output"],                    "PRODUCES"),
            (["oppos", "contra", "conflict", "disagre"],                  "OPPOSES"),
            (["help", "treat", "cur", "alleviat"],                        "SUPPORTS"),
        ]
        for keywords, rel in mapping:
            if any(kw in text_l for kw in keywords):
                return rel
        return "UNKNOWN"

    @staticmethod
    def _bucket(
        pred: Dict,
        positive: List[Dict],
        negative: List[Dict],
        neutral:  List[Dict],
    ) -> None:
        pol = pred.get("polarity", "NEUTRAL")
        if pol == "POSITIVE":
            positive.append(pred)
        elif pol == "NEGATIVE":
            negative.append(pred)
        else:
            neutral.append(pred)
