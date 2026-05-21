"""
Phase B — Semantic Hash Pipeline: end-to-end orchestration.

Consolidation Notes §46 critical ordering rule:
    1. SRL extraction from raw text
    2. Predicate family classification
    3. Canonical form generation
    4. Semantic hash computation   ←← this is step 4, never step 1
    5. Equivalence family update
    6. IRNode construction
    7. TRM lookup

The CanonicalizeAndHash class orchestrates steps 1-6 to produce an IRNode
with a fully populated SemanticSignature.  Step 7 (TRM lookup) is Phase D.

Design divergence from impl spec (§B.4):
    The impl spec proposes using the SentenceTransformer embedding directly
    as the semantic hash seed.  The consolidation notes (§46) clarify that:
        - The semantic hash must be computed from canonical_form text
          (NOT from the embedding vector) to ensure determinism.
        - Embedding vectors are stored separately as embedding_signature.
        - Two claims with different embeddings can still be semantically
          equivalent (CAUSAL::smoking::cause::cancer::depth0) and must
          hash to the same ID.
    We implement the consolidation notes' definition, not the spec's.
"""

from __future__ import annotations

import datetime
from typing import Optional, Callable

from mycelium.ir.primitives import (
    SemanticSignature,
    ConfidenceState,
    TemporalState,
    ProvenanceChain,
)
from mycelium.ir.graph import IRNode
from mycelium.ir.serialization import compute_semantic_hash
from .srl_extractor import SRLExtractor, SRLTriple
from .canonical_form import CanonicalFormGenerator


class CanonicalizeAndHash:
    """Orchestrates the Phase B canonicalization pipeline for a single claim.

    Takes raw text → produces an IRNode with a fully populated
    SemanticSignature (semantic_hash, canonical_form, predicate_family,
    equivalence_family, embedding_signature).

    Parameters
    ----------
    embedding_fn : callable, optional
        (str) -> list[float] returning 384-dim MiniLM vector.
        If None, embedding_signature will be an empty list.
    srl_extractor : SRLExtractor, optional
        Shared SRL extractor instance (for single-instance caching).
    canonical_generator : CanonicalFormGenerator, optional
        Shared generator (for equivalence family accumulation).
    """

    def __init__(
        self,
        *,
        embedding_fn: Optional[Callable[[str], list[float]]] = None,
        srl_extractor: Optional[SRLExtractor] = None,
        canonical_generator: Optional[CanonicalFormGenerator] = None,
    ) -> None:
        self.embedding_fn = embedding_fn
        self.srl = srl_extractor or SRLExtractor()
        self.canonical_gen = canonical_generator or CanonicalFormGenerator()

    def process(
        self,
        text: str,
        *,
        node_type: str = "CLAIM",
        temporal_type: str = "UNKNOWN",
        abstraction_level: int = 0,
        source: str = "",
    ) -> list[IRNode]:
        """Process a raw text string and return a list of IRNodes.

        One IRNode is created per SRL triple extracted from the text.
        If no triples are found, a single node is created with the raw
        text as label and a CORRELATIONAL predicate family (safe fallback).

        Parameters
        ----------
        text : str
            Raw claim text to canonicalize.
        node_type : str
            IRNode type (default CLAIM).
        temporal_type : str
            TemporalState type for the resulting nodes.
        abstraction_level : int
            Depth parameter for canonical form generation.
        source : str
            Source identifier for ProvenanceChain.sources.

        Returns
        -------
        list[IRNode]
            One IRNode per extracted SRL triple.
        """
        triples = self.srl.extract(text)
        if not triples:
            # Fallback: treat the whole text as a single opaque claim
            triples = [
                SRLTriple(
                    subject=text,
                    predicate="relates to",
                    obj="",
                    raw_sentence=text,
                    confidence=0.3,
                )
            ]

        nodes: list[IRNode] = []
        for i, triple in enumerate(triples):
            canonical, family = self.canonical_gen.generate(
                triple,
                abstraction_level=abstraction_level,
                embedding_fn=self.embedding_fn,
            )
            equiv_family = self.canonical_gen.get_equivalence_family(canonical)

            # Compute embedding if available
            embedding: list[float] = []
            if self.embedding_fn is not None:
                try:
                    embedding = list(self.embedding_fn(canonical))
                except Exception:
                    embedding = []

            # Build a placeholder sig so compute_semantic_hash() can run
            placeholder_sig = SemanticSignature(
                semantic_hash="",  # computed below
                embedding_signature=embedding,
                spectral_signature=[],  # Phase C fills this
                predicate_family=family,
                abstraction_level=abstraction_level,
                canonical_form=canonical,
                equivalence_family=equiv_family,
            )

            # Temporary node just to compute the hash
            node_id = f"node_{hash(canonical) & 0xFFFFFFFF:08x}_{i}"
            tmp_node = IRNode(
                id=node_id,
                type=node_type,
                label=triple.raw_sentence or text,
                semantic_signature=placeholder_sig,
                confidence_state=ConfidenceState(overall_confidence=triple.confidence),
                temporal_state=TemporalState(type=temporal_type),
                provenance=ProvenanceChain(
                    sources=[source] if source else [],
                    reasoning_paths=["phase_b_canonicalization"],
                    decomposition_origin="srl_extraction",
                    evidence_nodes=[],
                    ontology_resolution_path=[],
                    worker_threads=["main"],
                    timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
                ),
            )

            # Compute semantic hash and write it into the signature
            semantic_hash = compute_semantic_hash(tmp_node)
            final_sig = SemanticSignature(
                semantic_hash=semantic_hash,
                embedding_signature=embedding,
                spectral_signature=[],
                predicate_family=family,
                abstraction_level=abstraction_level,
                canonical_form=canonical,
                equivalence_family=equiv_family,
            )

            node = IRNode(
                id=node_id,
                type=node_type,
                label=triple.raw_sentence or text,
                semantic_signature=final_sig,
                confidence_state=ConfidenceState(overall_confidence=triple.confidence),
                temporal_state=TemporalState(type=temporal_type),
                provenance=tmp_node.provenance,
            )
            nodes.append(node)

        return nodes
