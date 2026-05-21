"""
Phase F — OntologyGovernor
===========================
Domain-specific decay rates and ontology version management.

Referenced by LeverageEdge.decay_rate (§F note in leverage.py):
    "Phase F sets this from OntologyGovernor.DOMAIN_DECAY_RATES"

Design rationale
----------------
Not all claims decay at the same rate.  Temporal claims ("The temperature
was 20°C yesterday") expire fast; definitional claims ("A mammal is a
warm-blooded vertebrate") expire almost never.  Decay rate governs how
quickly leverage weakens over inference cycles independent of explicit
contradiction.

Ontology versioning:
    When the canonicalization pipeline changes (new SRL extractor,
    new predicate family mappings), existing graphs may carry stale
    canonical forms.  OntologyGovernor.bump_version() records the
    bump so a migration pass can identify graphs needing re-canonicalization
    (identified by ontology_version < current_version in GraphStore).

Decay rate semantics:
    Decay rate is a per-inference-cycle multiplier applied to
    LeverageEdge.stability:
        stability_next = stability * (1 - decay_rate)
    At the default rate of 0.10, a leverage edge reaches 50% stability
    after ~7 inference cycles.
"""

from __future__ import annotations

import datetime
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Domain-specific decay rates (per-inference-cycle).
# Lower = slower decay = more stable claim type.
DOMAIN_DECAY_RATES: Dict[str, float] = {
    "CAUSAL": 0.05,  # causal claims: evidence-backed, stable
    "CORRELATIONAL": 0.10,  # correlations: softer evidence, faster decay
    "TEMPORAL": 0.20,  # temporal claims: time-sensitive, fastest decay
    "DEFINITIONAL": 0.02,  # definitions: near-permanent
    "COMPARATIVE": 0.08,  # comparisons: moderately stable
    "HIERARCHICAL": 0.03,  # taxonomy: rarely changes
    "ADVERSARIAL": 0.15,  # contested claims: decay under scrutiny
    "PROCEDURAL": 0.07,  # procedures: moderately stable
}

DEFAULT_DECAY_RATE: float = 0.10  # used for unknown/unmapped families


class OntologyGovernor:
    """Manages domain decay rates and ontology version lifecycle.

    This class is designed to be used as a singleton within a
    Mycelium instance.  Phase F creates one instance shared by
    EdgeBuilder and TRMEngine.

    Parameters
    ----------
    initial_version : str
        Starting ontology version (default "v1.0").
    """

    def __init__(self, initial_version: str = "v1.0") -> None:
        self._version: str = initial_version
        self._version_history: List[Tuple[str, str, str]] = [
            # (version, timestamp, reason)
            (initial_version, _now(), "initial")
        ]
        self._custom_rates: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Decay rate API
    # ------------------------------------------------------------------

    def domain_decay_rate(self, predicate_family: str) -> float:
        """Return the per-cycle decay rate for a predicate family.

        Checks custom overrides first, then DOMAIN_DECAY_RATES,
        then DEFAULT_DECAY_RATE.
        """
        if predicate_family in self._custom_rates:
            return self._custom_rates[predicate_family]
        return DOMAIN_DECAY_RATES.get(predicate_family, DEFAULT_DECAY_RATE)

    def set_custom_rate(self, predicate_family: str, rate: float) -> None:
        """Override the decay rate for a specific predicate family.

        Useful for domain-specific tuning (e.g., medical claims decay
        faster than general definitional claims).
        """
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"decay_rate must be in [0, 1], got {rate}")
        self._custom_rates[predicate_family] = rate
        logger.debug(
            "OntologyGovernor: custom rate set %s=%.4f", predicate_family, rate
        )

    def stability_after_cycles(self, predicate_family: str, n_cycles: int) -> float:
        """Compute leverage stability after n inference cycles.

        stability_n = 1.0 * (1 - decay_rate)^n
        (Starting stability is 1.0 for a fresh LeverageEdge.)
        """
        rate = self.domain_decay_rate(predicate_family)
        return (1.0 - rate) ** n_cycles

    # ------------------------------------------------------------------
    # Ontology version API
    # ------------------------------------------------------------------

    @property
    def version(self) -> str:
        """Current ontology version string."""
        return self._version

    def bump_version(
        self,
        reason: str,
        *,
        new_version: Optional[str] = None,
    ) -> str:
        """Increment the ontology version and record the bump.

        Parameters
        ----------
        reason : str
            Human-readable reason for the bump (e.g., 'new SRL extractor
            deployed', 'predicate family CAUSAL split into CAUSAL_DIRECT
            and CAUSAL_INDIRECT').
        new_version : str, optional
            Explicit new version string.  If None, auto-increments the
            minor version (v1.0 → v1.1 → … → v1.9 → v2.0).

        Returns
        -------
        str
            The new version string.
        """
        if new_version is None:
            new_version = _auto_increment(self._version)

        self._version = new_version
        entry = (new_version, _now(), reason)
        self._version_history.append(entry)
        logger.info("OntologyGovernor: version bumped to %s — %s", new_version, reason)
        return new_version

    def version_history(self) -> List[Tuple[str, str, str]]:
        """Return full version history as list of (version, timestamp, reason)."""
        return list(self._version_history)

    def graphs_needing_migration(self, graph_store: any) -> List[str]:
        """Return graph_ids whose ontology_version is older than current.

        Parameters
        ----------
        graph_store : GraphStore
            Phase D store to scan.

        Returns
        -------
        list[str]
            Graph IDs whose latest version has an older ontology_version.
        """
        stale: List[str] = []
        for gid in graph_store.all_graph_ids():
            g = graph_store.get_latest(gid)
            if g is None:
                continue
            g_ver = getattr(g, "ontology_version", "v1.0")
            if g_ver != self._version:
                stale.append(gid)
        return stale


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def _auto_increment(version: str) -> str:
    """Increment the minor component of a vMAJOR.MINOR version string."""
    try:
        without_v = version.lstrip("v")
        parts = without_v.split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        minor += 1
        if minor >= 10:
            major += 1
            minor = 0
        return f"v{major}.{minor}"
    except (ValueError, IndexError):
        return version + ".1"
