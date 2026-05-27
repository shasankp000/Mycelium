# mycelium/pipeline/predicates/predicate_pipeline.py
# Single-entry-point facade for the full predicate extraction + negation pass.
#
# Wires together:
#   NLPPreprocessor  (Layer 0 — spaCy bridge)
#   PredicateExtractor  (Stage 3 — SentenceAnalysis -> PredicateStore)
#   PredicateNegator    (Stage 4 — N1-N8 negation rules)
#
# Call site in run_workflow.py:
#   from mycelium.pipeline.predicates import get_pipeline
#   store = get_pipeline().run(query_text, domain_hint=domain)
#   context.predicate_store = store
#
# Nothing in this file performs LLM calls, retrieval, or I/O.
# All spaCy work is delegated through NLPPreprocessor's bridge protocol.
#
# Spec ref: implementation spec v0.2.1 — Section 7 (Pipeline Wiring)

from __future__ import annotations

import time
from typing import Optional

from mycelium.pipeline.layer0.nlp_preprocessor import NLPPreprocessor, SentenceAnalysis
from mycelium.pipeline.predicates.predicate_extractor import PredicateExtractor
from mycelium.pipeline.predicates.predicate_negator import PredicateNegator
from mycelium.pipeline.predicates.predicate_store import PredicateStore


class PredicatePipeline:
    """Full predicate extraction + negation pipeline for one request.

    Typical usage
    -------------
    ::
        from mycelium.pipeline.predicates import get_pipeline

        store = get_pipeline().run(
            "Does nuclear power cause more deaths than coal?",
            domain_hint="science",
        )
        # store.primary()        -> main predicate frame
        # store.falsifiable()    -> frames EvidenceFinder will operate on
        # store.summary()        -> dict for SSE event / logging

    Thread safety
    -------------
    PredicatePipeline itself is stateless between requests (except
    last_run_ms which is benignly overwritten).  The sub-components
    (NLPPreprocessor, PredicateExtractor, PredicateNegator) are also
    stateless.  Safe to share a singleton across threads as long as
    callers do not read last_run_ms from concurrent requests.

    If per-request timing isolation is needed, construct a new instance
    per request instead of using the singleton.
    """

    def __init__(
        self,
        preprocessor: Optional[NLPPreprocessor] = None,
        extractor: Optional[PredicateExtractor] = None,
        negator: Optional[PredicateNegator] = None,
    ) -> None:
        """Construct with optional injected sub-components.

        If not provided, the module-level singletons from each
        sub-module are used.  Injection is intended for unit testing.
        """
        self._preprocessor = preprocessor or NLPPreprocessor()
        self._extractor = extractor or PredicateExtractor()
        self._negator = negator or PredicateNegator()
        #: Wall-clock milliseconds of the most recent run() or
        #: run_from_analysis() call. 0.0 before any call.
        self.last_run_ms: float = 0.0

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def run(
        self,
        text: str,
        domain_hint: str = "general",
        spacy_model: str = "en_core_web_sm",
    ) -> PredicateStore:
        """Run the full pipeline on raw text.

        Steps
        -----
        1. NLPPreprocessor.analyse(text, model)  → SentenceAnalysis
        2. PredicateExtractor.extract(analysis, domain_hint) → PredicateStore
        3. PredicateNegator.negate_store(store)  → (in-place)
        4. Return store

        The pipeline always returns a valid PredicateStore.
        If spaCy is unavailable, the preprocessor produces a fallback
        SentenceAnalysis (spacy_available=False), which flows through
        extraction as a FACTIVE fallback frame.  The negator then
        produces its CONTRADICTS negation as normal.

        Parameters
        ----------
        text:
            Raw query or claim text.  Not pre-tokenised.
        domain_hint:
            Domain string from L1 routing.  Propagated into every
            PredicateFrame's domain_hint field.
        spacy_model:
            spaCy model name to request from the bridge.  Defaults to
            'en_core_web_sm'.  Ignored if spaCy is unavailable.

        Returns
        -------
        Populated PredicateStore with all extracted frames and their
        negated counterparts, plus all relation graph edges.
        """
        t0 = time.perf_counter()

        analysis = self._preprocessor.analyse(text, model=spacy_model)
        store = self._extractor.extract(analysis, domain_hint=domain_hint)
        self._negator.negate_store(store)

        self.last_run_ms = (time.perf_counter() - t0) * 1000.0
        return store

    # ------------------------------------------------------------------
    # Bypass entry point (Stage 7 hook)
    # ------------------------------------------------------------------

    def run_from_analysis(
        self,
        analysis: SentenceAnalysis,
        domain_hint: str = "general",
    ) -> PredicateStore:
        """Run extraction + negation on a pre-built SentenceAnalysis.

        Skips the NLPPreprocessor step entirely.  Used by EvidenceFinder
        in Stage 7 when the analysis has already been produced upstream
        (e.g. cached from a prior pipeline step or built in tests).

        Steps
        -----
        1. PredicateExtractor.extract(analysis, domain_hint) → PredicateStore
        2. PredicateNegator.negate_store(store)  → (in-place)
        3. Return store

        Parameters
        ----------
        analysis:
            Pre-built SentenceAnalysis from NLPPreprocessor or a test
            fixture.
        domain_hint:
            Domain string propagated to all frames.

        Returns
        -------
        Populated PredicateStore.
        """
        t0 = time.perf_counter()

        store = self._extractor.extract(analysis, domain_hint=domain_hint)
        self._negator.negate_store(store)

        self.last_run_ms = (time.perf_counter() - t0) * 1000.0
        return store

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"PredicatePipeline("
            f"last_run_ms={self.last_run_ms:.2f}, "
            f"preprocessor={type(self._preprocessor).__name__}, "
            f"extractor={type(self._extractor).__name__}, "
            f"negator={type(self._negator).__name__})"
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_pipeline_instance: Optional[PredicatePipeline] = None


def get_pipeline() -> PredicatePipeline:
    """Return the module-level PredicatePipeline singleton.

    Lazily instantiated on first call.  Uses the default sub-component
    singletons from each sub-module.
    """
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = PredicatePipeline()
    return _pipeline_instance
