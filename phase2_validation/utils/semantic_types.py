"""
Data classes for Phase 2.2 (Semantic Understanding) and Phase 2.3
(Expert Selection).

Provides typed containers for entities, concepts, semantic graphs,
semantic results, ranked experts, and expert selection results used
throughout the reasoning pipeline.

Example:
    >>> from phase2_validation.utils.semantic_types import (
    ...     Entity, Concept, SemanticGraph, SemanticResult,
    ... )
    >>> e = Entity(text="cancer", entity_type="medical")
    >>> print(e.confidence)
    0.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ------------------------------------------------------------------
# Phase 2.2 data classes
# ------------------------------------------------------------------


@dataclass
class Entity:
    """A recognized entity extracted from text.

    Attributes:
        text: The surface form of the entity.
        entity_type: Category of the entity (e.g. ``medical``).
        confidence: Confidence score in ``[0, 1]``.
        domain: Canonical domain the entity belongs to.
        start_pos: Character offset where the entity starts.
        end_pos: Character offset where the entity ends.
        context: Surrounding text for disambiguation.
    """

    text: str = ""
    entity_type: str = ""
    confidence: float = 0.0
    domain: str = ""
    start_pos: int = 0
    end_pos: int = 0
    context: str = ""


@dataclass
class Concept:
    """A key concept extracted via TF-IDF and semantic scoring.

    Attributes:
        text: The concept term.
        tfidf_score: TF-IDF weight of the concept.
        semantic_score: Embedding-based relevance score.
        relevance_to_tags: How well the concept aligns with
            extracted domain tags.
        overall_importance: Combined importance score.
        domain_affinity: Dictionary mapping domains to affinity
            scores.
    """

    text: str = ""
    tfidf_score: float = 0.0
    semantic_score: float = 0.0
    relevance_to_tags: float = 0.0
    overall_importance: float = 0.0
    domain_affinity: Dict[str, float] = field(
        default_factory=dict
    )


@dataclass
class SemanticGraph:
    """A graph representation of semantic relationships.

    Attributes:
        nodes: List of node labels.
        edges: List of ``(source, target, weight)`` triples.
        density: Graph density metric.
        average_degree: Average node degree.
    """

    nodes: List[str] = field(default_factory=list)
    edges: List[Tuple[str, str, float]] = field(
        default_factory=list
    )
    density: float = 0.0
    average_degree: float = 0.0


@dataclass
class SemanticResult:
    """Aggregated output of the semantic understanding pipeline.

    Attributes:
        original_text: Raw input text.
        cleaned_text: Text after normalization.
        text_embedding: Dense vector representation.
        extracted_entities: Entities found in the text.
        key_concepts: Top-K concepts with scores.
        concept_scores: Mapping of concept text to importance.
        semantic_graph: Relationship graph.
        graph_density: Density of the semantic graph.
        domain_relevance_scores: Per-domain relevance scores.
        ranked_domains: Domains sorted by relevance.
        processing_time_ms: Wall-clock processing time.
        embedding_cache_hits: Number of cache hits.
        confidence_score: Overall confidence in ``[0, 1]``.
        warnings: Non-fatal issues encountered.
    """

    original_text: str = ""
    cleaned_text: str = ""
    text_embedding: Optional[np.ndarray] = None
    extracted_entities: List[Entity] = field(
        default_factory=list
    )
    key_concepts: List[Concept] = field(default_factory=list)
    concept_scores: Dict[str, float] = field(
        default_factory=dict
    )
    semantic_graph: Optional[SemanticGraph] = None
    graph_density: float = 0.0
    domain_relevance_scores: Dict[str, float] = field(
        default_factory=dict
    )
    ranked_domains: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    embedding_cache_hits: int = 0
    confidence_score: float = 0.0
    warnings: List[str] = field(default_factory=list)


# ------------------------------------------------------------------
# Phase 2.3 data classes
# ------------------------------------------------------------------


@dataclass
class RankedExpert:
    """An expert ranked by match quality.

    Attributes:
        name: Expert identifier.
        domain: Primary domain of the expert.
        match_score: Overall match score in ``[0, 1]``.
        confidence_low: Lower bound of confidence interval.
        confidence_high: Upper bound of confidence interval.
        ranking_factors: Breakdown of scoring factors.
    """

    name: str = ""
    domain: str = ""
    match_score: float = 0.0
    confidence_low: float = 0.0
    confidence_high: float = 0.0
    ranking_factors: Dict[str, float] = field(
        default_factory=dict
    )


@dataclass
class ExpertSelectionResult:
    """Aggregated output of the expert selection pipeline.

    Attributes:
        selected_experts: Experts chosen for inference.
        filtered_out_experts: Experts that did not pass filters.
        selection_strategy: Strategy used for selection.
        total_candidates: Number of experts considered.
        total_selected: Number of experts selected.
        expert_scores: Mapping of expert name to score.
        confidence_in_selection: Overall confidence.
        processing_time_ms: Wall-clock processing time.
        cold_start_fallback_used: Whether cold-start fallback
            was triggered.
        warnings: Non-fatal issues encountered.
    """

    selected_experts: List[RankedExpert] = field(
        default_factory=list
    )
    filtered_out_experts: List[RankedExpert] = field(
        default_factory=list
    )
    selection_strategy: str = "greedy"
    total_candidates: int = 0
    total_selected: int = 0
    expert_scores: Dict[str, float] = field(
        default_factory=dict
    )
    confidence_in_selection: float = 0.0
    processing_time_ms: float = 0.0
    cold_start_fallback_used: bool = False
    warnings: List[str] = field(default_factory=list)


# ------------------------------------------------------------------
# Phase 2.4 data classes
# ------------------------------------------------------------------


@dataclass
class ExpertPrediction:
    """Single expert's prediction.

    Attributes:
        expert_name: Identifier for the expert.
        prediction: The prediction value or label.
        confidence: Confidence in ``[0, 1]``.
        latency_ms: Inference latency in milliseconds.
        metadata: Additional metadata about the prediction.
    """

    expert_name: str = ""
    prediction: object = None
    confidence: float = 0.0
    latency_ms: float = 0.0
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class PredictionStatistics:
    """Statistics across predictions.

    Attributes:
        mean: Mean of prediction confidence values.
        std: Standard deviation.
        min: Minimum confidence.
        max: Maximum confidence.
        median: Median confidence.
        iqr: Interquartile range.
    """

    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    median: float = 0.0
    iqr: float = 0.0
