from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional

from mycelium.domain_graph.registry import DomainGraphRegistry
from mycelium.domain_graph.state import DomainMode, DomainState, GateState

logger = logging.getLogger(__name__)


@dataclass
class ExpansionPolicy:
    """
    Controls when a domain transitions from FROZEN → EXPANSION mode.
    Triggered by DomainNoveltyPolicy returning EXPAND_EXISTING.
    """
    min_novelty_score: float = 0.30    # how novel the sub-topic must be
    min_query_count: int = 10          # domain must have seen this many queries
    max_expansions_per_domain: int = 5 # hard cap on how many times a domain can expand


@dataclass
class BootstrapPolicy:
    """
    Controls transition out of BOOTSTRAP mode into FROZEN (ready for inference).
    """
    min_training_samples: int = 500
    min_eval_f1: float = 0.70


class GateModeController:
    """
    Applies gate mode transitions to the DomainGraphRegistry.
    Called by TRMV2Trainer after each training phase completes.

    All transitions follow:
      BOOTSTRAP → FROZEN  (after bootstrap training complete)
      FROZEN    → EXPANSION (on EXPAND_EXISTING novelty decision)
      EXPANSION → FROZEN  (after expansion training complete)
      FROZEN    → THAWING (manual or drift-triggered fine-tune window)
      THAWING   → FROZEN  (after fine-tune window closes)
    """

    def __init__(
        self,
        registry: DomainGraphRegistry,
        expansion_policy: Optional[ExpansionPolicy] = None,
        bootstrap_policy: Optional[BootstrapPolicy] = None,
    ) -> None:
        self._registry = registry
        self._expansion = expansion_policy or ExpansionPolicy()
        self._bootstrap = bootstrap_policy or BootstrapPolicy()

    def bootstrap_complete(self, domain_id: str) -> None:
        """Call after bootstrap training finishes."""
        node = self._registry.require(domain_id)
        if node.mode != DomainMode.BOOTSTRAP:
            logger.warning("bootstrap_complete called on domain '%s' in mode %s", domain_id, node.mode)
            return
        self._registry.set_mode(domain_id, DomainMode.FROZEN)
        self._registry.set_state(domain_id, DomainState.HOT)
        self._registry.set_gate(domain_id, GateState.OPEN)
        logger.info("Domain '%s': BOOTSTRAP → FROZEN, gate OPEN, state HOT", domain_id)

    def trigger_expansion(self, domain_id: str, novelty_score: float) -> bool:
        """
        Returns True if expansion was triggered.
        """
        node = self._registry.require(domain_id)
        if node.mode != DomainMode.FROZEN:
            return False
        if novelty_score < self._expansion.min_novelty_score:
            return False
        if node.query_count < self._expansion.min_query_count:
            return False
        expansions = node.meta.get("expansion_count", 0)
        if expansions >= self._expansion.max_expansions_per_domain:
            logger.info("Domain '%s': expansion cap reached (%d).", domain_id, expansions)
            return False
        self._registry.set_mode(domain_id, DomainMode.EXPANSION)
        node.meta["expansion_count"] = expansions + 1
        self._registry.update(node)
        logger.info("Domain '%s': triggered EXPANSION (novelty=%.3f)", domain_id, novelty_score)
        return True

    def expansion_complete(self, domain_id: str) -> None:
        self._registry.set_mode(domain_id, DomainMode.FROZEN)
        logger.info("Domain '%s': EXPANSION → FROZEN", domain_id)

    def open_thaw_window(self, domain_id: str) -> None:
        self._registry.set_mode(domain_id, DomainMode.THAWING)
        logger.info("Domain '%s': THAWING window opened", domain_id)

    def close_thaw_window(self, domain_id: str) -> None:
        self._registry.set_mode(domain_id, DomainMode.FROZEN)
        logger.info("Domain '%s': THAWING window closed → FROZEN", domain_id)
