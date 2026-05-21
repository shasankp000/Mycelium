"""
Phase F.2 — Adversarial Robustness Layer
=========================================
Prevents ontology poisoning, confidence laundering, and coordinated
semantic manipulation from propagating through the IR graph.

Spec reference (§F.2):
    - AuthorTrustProfile  — per-author trust score with factual-claim history
    - AdversarialGuard    — validates inputs, decays trust, detects poisoning

Design invariants:
    1. Trust decays by ×0.90 per unverified factual claim (never rises
       automatically — re-verification is required to restore trust).
    2. When false_rate > 0.40 over recorded history, the author is flagged
       and their leverage_ceiling is hard-capped at 0.20.
    3. Silent pattern analysis begins when trust drops below 0.50.
    4. All mutations return *new* dataclass instances — no in-place mutation
       (consistent with Phase D immutability rules).
    5. Ontology proposals from flagged authors are down-weighted but not
       silently dropped; the returned OntologyRelation carries a reduced
       confidence so downstream TRM arbitration still sees the signal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from mycelium.ir.graph import IRNode
    from mycelium.ir.ontology_governor import OntologyRelation

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TRUST_DECAY_FACTOR: float = 0.90
_SILENT_ANALYSIS_THRESHOLD: float = 0.50
_POISONING_FALSE_RATE_THRESHOLD: float = 0.40
_POISONING_LEVERAGE_CEILING: float = 0.20
_MIN_HISTORY_FOR_PATTERN: int = 3          # need ≥3 claims to trigger analysis
_ONTOLOGY_FLAGGED_CONFIDENCE_PENALTY: float = 0.60   # multiply proposal confidence


# ---------------------------------------------------------------------------
# AuthorTrustProfile
# ---------------------------------------------------------------------------

@dataclass
class FactRecord:
    """One entry in an author's factual-claim history."""
    claim_hash: str          # semantic_hash of the IRNode claim
    claim_label: str         # human-readable label for audit logs
    verified: bool           # True = confirmed factual; False = refuted
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: str = "unknown"  # which ground-truth oracle verified it


@dataclass
class AuthorTrustProfile:
    """Per-author trust state.

    Attributes
    ----------
    author_id :
        Stable identifier (user ID, tool ID, source URL hash, etc.).
    trust_score :
        Float in [0.0, 1.0].  Starts at 1.0, decays on unverified claims.
        Never automatically restored — only explicit re-verification raises it.
    fact_history :
        Ordered list of FactRecord instances (oldest first).
    flagged :
        Set to True when false_rate > _POISONING_FALSE_RATE_THRESHOLD.
        Flagged authors have all claims down-weighted and their leverage
        ceiling capped at _POISONING_LEVERAGE_CEILING.
    leverage_ceiling :
        Maximum effective leverage any claim from this author can contribute.
        1.0 for trusted authors; reduced to 0.20 when flagged.
    silent_analysis_active :
        True when trust_score < _SILENT_ANALYSIS_THRESHOLD.  The guard
        then runs poisoning-pattern analysis on every subsequent input.
    """
    author_id: str
    trust_score: float = 1.0
    fact_history: List[FactRecord] = field(default_factory=list)
    flagged: bool = False
    leverage_ceiling: float = 1.0
    silent_analysis_active: bool = False

    # ------------------------------------------------------------------
    # Derived properties (read-only helpers — no mutation)
    # ------------------------------------------------------------------

    @property
    def false_rate(self) -> float:
        """Fraction of recorded claims that were refuted."""
        if not self.fact_history:
            return 0.0
        refuted = sum(1 for r in self.fact_history if not r.verified)
        return refuted / len(self.fact_history)

    @property
    def claim_count(self) -> int:
        return len(self.fact_history)

    def summary(self) -> Dict:
        return {
            "author_id": self.author_id,
            "trust_score": round(self.trust_score, 4),
            "false_rate": round(self.false_rate, 4),
            "claim_count": self.claim_count,
            "flagged": self.flagged,
            "leverage_ceiling": self.leverage_ceiling,
            "silent_analysis_active": self.silent_analysis_active,
        }


# ---------------------------------------------------------------------------
# AdversarialGuard
# ---------------------------------------------------------------------------

class AdversarialGuard:
    """Validates factual inputs against ground-truth results and maintains
    per-author trust profiles.

    All public methods take an *existing* AuthorTrustProfile and return a
    *new* one (immutable update pattern — no in-place mutation).

    Parameters
    ----------
    max_history : int
        Rolling window size for fact_history.  Oldest entries are pruned
        when the window is exceeded.  Default 100.
    """

    def __init__(self, *, max_history: int = 100) -> None:
        self._max_history = max_history

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify_factual_input(
        self,
        claim: "IRNode",
        author: AuthorTrustProfile,
        ground_truth_result: Dict,
    ) -> AuthorTrustProfile:
        """Record a ground-truth verification outcome for one claim.

        Parameters
        ----------
        claim :
            The IRNode representing the factual claim being checked.
        author :
            Existing trust profile for the claim's author.
        ground_truth_result :
            Dict with at minimum:
                ``verified`` (bool)   — did the claim pass fact-checking?
                ``source``   (str)    — which oracle performed the check
        Returns
        -------
        AuthorTrustProfile
            A *new* profile with updated trust_score, fact_history, and
            flags.  The input ``author`` object is never mutated.
        """
        verified: bool = bool(ground_truth_result.get("verified", False))
        source: str = ground_truth_result.get("source", "unknown")

        record = FactRecord(
            claim_hash=getattr(getattr(claim, "semantic_signature", None), "semantic_hash", claim.id),
            claim_label=getattr(claim, "label", str(claim.id)),
            verified=verified,
            source=source,
        )

        updated_history = list(author.fact_history) + [record]
        if len(updated_history) > self._max_history:
            updated_history = updated_history[-self._max_history:]

        updated = replace(
            author,
            fact_history=updated_history,
        )

        if not verified:
            new_trust = author.trust_score * _TRUST_DECAY_FACTOR
            updated = replace(updated, trust_score=new_trust)
            logger.debug(
                "AdversarialGuard: trust decayed for '%s' → %.4f (claim: %s)",
                author.author_id, new_trust, record.claim_label,
            )

        # Enable silent analysis once trust falls below threshold
        if updated.trust_score < _SILENT_ANALYSIS_THRESHOLD and not updated.silent_analysis_active:
            updated = replace(updated, silent_analysis_active=True)
            logger.info(
                "AdversarialGuard: silent pattern analysis activated for '%s' (trust=%.4f)",
                author.author_id, updated.trust_score,
            )

        # Run poisoning-pattern check when analysis is active or just triggered
        if updated.silent_analysis_active or not verified:
            updated = self._analyze_poisoning_pattern(updated)

        return updated

    def analyze_poisoning_pattern(
        self,
        author: AuthorTrustProfile,
    ) -> AuthorTrustProfile:
        """Public wrapper — force a poisoning-pattern check on an author profile.

        Useful for periodic audits or when an external signal triggers a
        re-evaluation (e.g., coordinated false-claim burst detected upstream).

        Returns a new profile; never mutates the input.
        """
        return self._analyze_poisoning_pattern(author)

    def verify_ontology_proposal(
        self,
        proposal: "OntologyRelation",
        author: AuthorTrustProfile,
    ) -> "OntologyRelation":
        """Down-weight an ontology proposal from a flagged or low-trust author.

        The proposal is never silently dropped — it is returned with a
        reduced confidence so TRM stabilization can still arbitrate it.
        Downstream callers should inspect ``proposal.confidence`` to decide
        how aggressively to promote the relation.

        Parameters
        ----------
        proposal :
            OntologyRelation produced by OntologyGovernor.propose_relation().
        author :
            Trust profile of the author who submitted the grounding evidence.

        Returns
        -------
        OntologyRelation
            A new OntologyRelation instance with adjusted confidence.
        """
        # Import here to avoid circular dependency at module load time
        from mycelium.ir.ontology_governor import OntologyRelation  # noqa: F401

        if author.flagged:
            penalty = _ONTOLOGY_FLAGGED_CONFIDENCE_PENALTY
            logger.warning(
                "AdversarialGuard: ontology proposal from flagged author '%s' "
                "down-weighted by %.0f%%  (%s → %s)",
                author.author_id,
                (1.0 - penalty) * 100,
                proposal.subtype,
                proposal.supertype,
            )
            return replace(proposal, confidence=proposal.confidence * penalty)

        # Proportional down-weight for low-trust (but not yet flagged) authors
        trust_factor = max(author.trust_score, 0.0)
        if trust_factor < 1.0:
            adjusted = proposal.confidence * trust_factor
            return replace(proposal, confidence=adjusted)

        return proposal

    def restore_trust(
        self,
        author: AuthorTrustProfile,
        *,
        verified_claims: int = 1,
        unflag: bool = False,
    ) -> AuthorTrustProfile:
        """Manually restore trust after successful re-verification.

        Trust is raised by +0.05 per verified claim, capped at 1.0.
        If ``unflag=True`` is explicitly passed, the flagged state is also
        cleared — this should only be done after a formal audit.

        Parameters
        ----------
        verified_claims :
            Number of positively verified claims driving the restoration.
        unflag :
            Whether to clear the ``flagged`` state.  Default False — the
            caller must consciously decide to unflag.
        """
        gain = 0.05 * verified_claims
        new_trust = min(1.0, author.trust_score + gain)
        updated = replace(author, trust_score=new_trust)

        if updated.trust_score >= _SILENT_ANALYSIS_THRESHOLD and updated.silent_analysis_active:
            updated = replace(updated, silent_analysis_active=False)

        if unflag and updated.flagged:
            updated = replace(updated, flagged=False, leverage_ceiling=1.0)
            logger.info(
                "AdversarialGuard: '%s' manually unflagged after audit",
                author.author_id,
            )

        return updated

    def get_trust_report(self, author: AuthorTrustProfile) -> Dict:
        """Return a human-readable audit report for an author profile."""
        recent = author.fact_history[-10:] if author.fact_history else []
        return {
            **author.summary(),
            "recent_claims": [
                {
                    "label": r.claim_label,
                    "verified": r.verified,
                    "timestamp": r.timestamp,
                    "source": r.source,
                }
                for r in recent
            ],
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _analyze_poisoning_pattern(
        self,
        author: AuthorTrustProfile,
    ) -> AuthorTrustProfile:
        """Core poisoning-pattern detector.

        Checks the rolling fact_history window for a false_rate that
        exceeds _POISONING_FALSE_RATE_THRESHOLD.  Requires at least
        _MIN_HISTORY_FOR_PATTERN records to avoid false positives on
        brand-new authors with a single bad claim.

        When a poisoning pattern is confirmed:
            - author.flagged = True
            - author.leverage_ceiling = _POISONING_LEVERAGE_CEILING (0.20)

        Returns a *new* AuthorTrustProfile; never mutates input.
        """
        if author.claim_count < _MIN_HISTORY_FOR_PATTERN:
            return author

        false_rate = author.false_rate
        if false_rate > _POISONING_FALSE_RATE_THRESHOLD:
            if not author.flagged:
                logger.warning(
                    "AdversarialGuard: POISONING PATTERN DETECTED for '%s' "
                    "(false_rate=%.2f > threshold=%.2f). "
                    "Flagging author and capping leverage at %.2f.",
                    author.author_id,
                    false_rate,
                    _POISONING_FALSE_RATE_THRESHOLD,
                    _POISONING_LEVERAGE_CEILING,
                )
            return replace(
                author,
                flagged=True,
                leverage_ceiling=_POISONING_LEVERAGE_CEILING,
            )

        return author
