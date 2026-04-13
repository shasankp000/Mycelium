"""
Phase 2.2 — Semantic Understanding Layer.

This module implements the second phase of the six-phase reasoning
pipeline.  It generates embeddings, recognizes domain-specific
entities, builds a semantic relationship graph, extracts key
concepts, and scores domain relevance.

Components:
    * EmbeddingGenerator — text/tag embedding with MD5-keyed cache
    * EntityRecognizer — domain-specific entity extraction
    * SemanticGraphBuilder — graph construction with PageRank
    * KeyConceptExtractor — TF-IDF + semantic concept extraction
    * DomainRelevanceScorer — multi-factor domain scoring
    * SemanticUnderstandingPipeline — orchestrator

Example:
    >>> from phase2_validation.phases.phase_2_2_semantic_understanding import (
    ...     SemanticUnderstandingPipeline,
    ... )
    >>> pipeline = SemanticUnderstandingPipeline()
    >>> result = pipeline.understand(normalization_result, ["medical", "physics"])
    >>> print(result.ranked_domains)
"""

import hashlib
import logging
import math
import re
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from phase2_validation.config.phase2_config import Phase2Config
from phase2_validation.utils.semantic_types import (
    Concept,
    Entity,
    SemanticGraph,
    SemanticResult,
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Custom exceptions
# ------------------------------------------------------------------


class SemanticUnderstandingError(Exception):
    """Base exception for semantic understanding failures."""


class EmbeddingError(SemanticUnderstandingError):
    """Raised when embedding generation fails."""


class EntityExtractionError(SemanticUnderstandingError):
    """Raised when entity extraction fails."""


class GraphBuildError(SemanticUnderstandingError):
    """Raised when semantic graph construction fails."""


# ------------------------------------------------------------------
# EmbeddingGenerator
# ------------------------------------------------------------------

_EMBEDDING_DIM = 384


class EmbeddingGenerator:
    """Generate and cache text embeddings.

    Uses a lightweight deterministic hash-based embedding scheme
    so that no external model download is required at test time.
    The output dimension is 384, matching all-MiniLM-L6-v2.

    Attributes:
        cache_max_size: Maximum number of cached embeddings.
    """

    def __init__(
        self,
        cache_max_size: int = 1000,
    ) -> None:
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.cache_max_size = cache_max_size
        self._cache_hits = 0

    # ---- public API ------------------------------------------------

    def generate_text_embedding(
        self, text: str
    ) -> np.ndarray:
        """Generate a 384-dim embedding for *text*.

        Args:
            text: Input text string.

        Returns:
            A numpy array of shape ``(384,)``.
        """
        key = self._md5(text)
        cached = self._cache_get(key)
        if cached is not None:
            self._cache_hits += 1
            return cached

        emb = self._deterministic_embedding(text)
        self._cache_put(key, emb)
        return emb

    def generate_tag_embeddings(
        self, tags: List[str]
    ) -> Dict[str, np.ndarray]:
        """Generate embeddings for a list of tags.

        Args:
            tags: List of tag strings.

        Returns:
            Dictionary mapping each tag to its embedding.
        """
        return {
            tag: self.generate_text_embedding(tag)
            for tag in tags
        }

    def compute_embedding_similarity(
        self, emb1: np.ndarray, emb2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            emb1: First embedding vector.
            emb2: Second embedding vector.

        Returns:
            Cosine similarity in ``[-1, 1]``.
        """
        n1 = np.linalg.norm(emb1)
        n2 = np.linalg.norm(emb2)
        if n1 == 0.0 or n2 == 0.0:
            return 0.0
        return float(np.dot(emb1, emb2) / (n1 * n2))

    @property
    def cache_hits(self) -> int:
        """Return total cache hits since creation."""
        return self._cache_hits

    def reset_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
        self._cache_hits = 0

    # ---- internals -------------------------------------------------

    def _deterministic_embedding(
        self, text: str
    ) -> np.ndarray:
        """Create a deterministic embedding from text via MD5 seed.

        The resulting vector is normalized to unit length.
        """
        digest = hashlib.md5(
            text.encode("utf-8", errors="replace")
        ).hexdigest()
        seed = int(digest, 16) % (2**31)
        rng = np.random.RandomState(seed)
        vec = rng.randn(_EMBEDDING_DIM).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    @staticmethod
    def _md5(text: str) -> str:
        return hashlib.md5(
            text.encode("utf-8", errors="replace")
        ).hexdigest()

    def _cache_get(self, key: str) -> Optional[np.ndarray]:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def _cache_put(
        self, key: str, value: np.ndarray
    ) -> None:
        self._cache[key] = value
        if len(self._cache) > self.cache_max_size:
            self._cache.popitem(last=False)


# ------------------------------------------------------------------
# EntityRecognizer
# ------------------------------------------------------------------


class EntityRecognizer:
    """Extract domain-specific entities from text.

    Supports pattern-based extraction for medical, physics,
    and chemistry domains with confidence scoring.
    """

    _DOMAIN_PATTERNS: Dict[str, List[Tuple[str, str]]] = {
        "medical": [
            (r"\b(cancer|carcinoma|tumor|tumour)\b", "disease"),
            (r"\b(chemotherapy|immunotherapy|radiation"
             r"|therapy|treatment)\b", "treatment"),
            (r"\b(patient|diagnosis|clinical|symptom"
             r"|syndrome)\b", "clinical"),
            (r"\b(drug|medication|pharmaceutical"
             r"|antibiotic)\b", "drug"),
            (r"\b(DNA|RNA|protein|gene|genome"
             r"|chromosome)\b", "molecular"),
        ],
        "physics": [
            (r"\b(quantum|quanta)\b", "quantum"),
            (r"\b(electron|proton|neutron|photon"
             r"|particle)\b", "particle"),
            (r"\b(energy|force|momentum|velocity"
             r"|acceleration)\b", "mechanics"),
            (r"\b(thermodynamic|entropy|heat"
             r"|temperature)\b", "thermodynamics"),
            (r"\b(electromagnetic|magnetic|electric"
             r"|field)\b", "electromagnetism"),
        ],
        "chemistry": [
            (r"\b(molecule|compound|element|atom)\b",
             "substance"),
            (r"\b(reaction|catalyst|reagent|oxidation"
             r"|reduction)\b", "reaction"),
            (r"\b(organic|inorganic|polymer"
             r"|biochemistry)\b", "branch"),
            (r"\b(acid|base|solution|solvent"
             r"|concentration)\b", "solution"),
            (r"\b(bond|ionic|covalent|hydrogen)\b",
             "bonding"),
        ],
    }

    def extract_entities(
        self, text: str, tags: List[str]
    ) -> List[Entity]:
        """Extract entities from *text* using domain patterns.

        Args:
            text: Input text.
            tags: Extracted domain tags for context.

        Returns:
            List of ``Entity`` objects.
        """
        entities: List[Entity] = []
        text_lower = text.lower()

        for domain, patterns in self._DOMAIN_PATTERNS.items():
            for pattern_str, entity_type in patterns:
                for match in re.finditer(
                    pattern_str, text_lower
                ):
                    confidence = self._entity_confidence(
                        match.group(), domain, tags
                    )
                    context = self._get_context(
                        text, match.start(), match.end()
                    )
                    entities.append(
                        Entity(
                            text=match.group(),
                            entity_type=entity_type,
                            confidence=confidence,
                            domain=domain,
                            start_pos=match.start(),
                            end_pos=match.end(),
                            context=context,
                        )
                    )

        # Deduplicate by (text, domain)
        seen: set = set()
        unique: List[Entity] = []
        for ent in entities:
            key = (ent.text, ent.domain)
            if key not in seen:
                seen.add(key)
                unique.append(ent)
        return unique

    def _entity_confidence(
        self,
        entity_text: str,
        domain: str,
        tags: List[str],
    ) -> float:
        """Score the confidence of a detected entity.

        Considers tag overlap and entity length.
        """
        base = 0.5
        tag_lower = [t.lower() for t in tags]

        if domain in tag_lower or any(
            domain in t for t in tag_lower
        ):
            base += 0.3

        if len(entity_text) > 5:
            base += 0.1

        return min(base, 1.0)

    @staticmethod
    def _get_context(
        text: str, start: int, end: int, window: int = 50
    ) -> str:
        """Extract surrounding context around an entity span."""
        ctx_start = max(0, start - window)
        ctx_end = min(len(text), end + window)
        return text[ctx_start:ctx_end]


# ------------------------------------------------------------------
# SemanticGraphBuilder
# ------------------------------------------------------------------


class SemanticGraphBuilder:
    """Build a semantic relationship graph from entities and tags.

    Uses a simplified PageRank to compute node importance.
    """

    def __init__(
        self,
        embedding_generator: Optional[EmbeddingGenerator] = None,
    ) -> None:
        self._emb_gen = embedding_generator or EmbeddingGenerator()

    def build_graph(
        self,
        text: str,
        entities: List[Entity],
        tags: List[str],
    ) -> SemanticGraph:
        """Construct a semantic graph.

        Nodes are entity texts and tags. Edges are weighted by
        embedding similarity.

        Args:
            text: Source text.
            entities: Extracted entities.
            tags: Domain tags.

        Returns:
            A ``SemanticGraph`` instance.
        """
        node_set: set = set()
        for ent in entities:
            node_set.add(ent.text)
        for tag in tags:
            node_set.add(tag)

        nodes = sorted(node_set)
        if not nodes:
            return SemanticGraph()

        embeddings = {
            n: self._emb_gen.generate_text_embedding(n)
            for n in nodes
        }

        edges: List[Tuple[str, str, float]] = []
        for i, n1 in enumerate(nodes):
            for n2 in nodes[i + 1:]:
                sim = self._emb_gen.compute_embedding_similarity(
                    embeddings[n1], embeddings[n2]
                )
                if sim > 0.1:
                    edges.append((n1, n2, round(sim, 4)))

        density = self._compute_density(len(nodes), len(edges))
        avg_degree = (
            (2.0 * len(edges)) / len(nodes)
            if nodes else 0.0
        )

        graph = SemanticGraph(
            nodes=nodes,
            edges=edges,
            density=round(density, 4),
            average_degree=round(avg_degree, 4),
        )

        self._apply_pagerank(graph)
        return graph

    def extract_subgraph(
        self,
        graph: SemanticGraph,
        domain: str,
    ) -> SemanticGraph:
        """Extract domain-relevant subgraph.

        Keeps nodes whose label contains *domain* (case
        insensitive) and any edges connecting them.

        Args:
            graph: Full semantic graph.
            domain: Domain string to filter by.

        Returns:
            A new ``SemanticGraph`` with filtered nodes/edges.
        """
        domain_lower = domain.lower()
        sub_nodes = [
            n for n in graph.nodes
            if domain_lower in n.lower()
        ]
        sub_set = set(sub_nodes)
        sub_edges = [
            (s, t, w) for s, t, w in graph.edges
            if s in sub_set and t in sub_set
        ]
        density = self._compute_density(
            len(sub_nodes), len(sub_edges)
        )
        avg_deg = (
            (2.0 * len(sub_edges)) / len(sub_nodes)
            if sub_nodes else 0.0
        )
        return SemanticGraph(
            nodes=sub_nodes,
            edges=sub_edges,
            density=round(density, 4),
            average_degree=round(avg_deg, 4),
        )

    @staticmethod
    def _compute_density(
        num_nodes: int, num_edges: int
    ) -> float:
        """Compute graph density."""
        max_edges = num_nodes * (num_nodes - 1) / 2
        if max_edges == 0:
            return 0.0
        return num_edges / max_edges

    @staticmethod
    def _apply_pagerank(
        graph: SemanticGraph,
        damping: float = 0.85,
        iterations: int = 20,
    ) -> None:
        """Apply simplified PageRank in-place (updates edges).

        This is a lightweight iterative computation; node
        importance values are stored as self-loop edges
        ``(node, node, importance)``.
        """
        nodes = graph.nodes
        if not nodes:
            return

        n = len(nodes)
        scores: Dict[str, float] = {nd: 1.0 / n for nd in nodes}

        adj: Dict[str, List[Tuple[str, float]]] = {
            nd: [] for nd in nodes
        }
        for src, tgt, w in graph.edges:
            adj[src].append((tgt, w))
            adj[tgt].append((src, w))

        for _ in range(iterations):
            new_scores: Dict[str, float] = {}
            for nd in nodes:
                rank = (1.0 - damping) / n
                for neighbor, w in adj[nd]:
                    out_degree = len(adj[neighbor])
                    if out_degree > 0:
                        rank += (
                            damping
                            * scores[neighbor]
                            * w
                            / out_degree
                        )
                new_scores[nd] = rank
            # Normalize
            total = sum(new_scores.values()) or 1.0
            scores = {
                k: v / total for k, v in new_scores.items()
            }

        for nd, importance in scores.items():
            graph.edges.append(
                (nd, nd, round(importance, 6))
            )


# ------------------------------------------------------------------
# KeyConceptExtractor
# ------------------------------------------------------------------


class KeyConceptExtractor:
    """Extract key concepts using TF-IDF + semantic scoring."""

    _STOP_WORDS = frozenset(
        "a an the is are was were be been being have has had "
        "do does did will would shall should may might can "
        "could must need dare to of in for on with at by from "
        "as into through during before after above below "
        "between out off over under again further then once "
        "here there when where why how all each every both "
        "few more most other some such no nor not only own "
        "same so than too very and but or if while because "
        "about against this that these those it its he she "
        "they them their what which who whom".split()
    )

    def __init__(
        self,
        embedding_generator: Optional[EmbeddingGenerator] = None,
    ) -> None:
        self._emb_gen = embedding_generator or EmbeddingGenerator()

    def extract_key_concepts(
        self, text: str, top_k: int = 5
    ) -> List[Concept]:
        """Extract the top-K concepts from *text*.

        Combines TF-IDF weight with a semantic score based on
        embedding similarity to the full text embedding.

        Args:
            text: Input text.
            top_k: Number of top concepts to return.

        Returns:
            List of ``Concept`` objects sorted by importance.
        """
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        words = [
            w for w in words if w not in self._STOP_WORDS
        ]

        if not words:
            return []

        tf = Counter(words)
        total = len(words)
        unique = list(set(words))

        text_emb = self._emb_gen.generate_text_embedding(text)

        concepts: List[Concept] = []
        for word in unique:
            tfidf = tf[word] / total
            word_emb = self._emb_gen.generate_text_embedding(
                word
            )
            sem_score = max(
                0.0,
                self._emb_gen.compute_embedding_similarity(
                    text_emb, word_emb
                ),
            )
            importance = 0.5 * tfidf + 0.5 * sem_score
            concepts.append(
                Concept(
                    text=word,
                    tfidf_score=round(tfidf, 6),
                    semantic_score=round(sem_score, 6),
                    overall_importance=round(importance, 6),
                )
            )

        concepts.sort(
            key=lambda c: c.overall_importance, reverse=True
        )
        return concepts[:top_k]

    def rank_concepts_by_relevance(
        self,
        concepts: List[Concept],
        tags: List[str],
    ) -> List[Concept]:
        """Re-rank concepts by tag relevance.

        Each concept receives a ``relevance_to_tags`` score
        based on embedding similarity to the provided tags.

        Args:
            concepts: List of concepts.
            tags: Domain tags.

        Returns:
            List of concepts sorted by tag relevance.
        """
        if not tags:
            return concepts

        tag_embs = self._emb_gen.generate_tag_embeddings(tags)

        for concept in concepts:
            c_emb = self._emb_gen.generate_text_embedding(
                concept.text
            )
            sims = [
                self._emb_gen.compute_embedding_similarity(
                    c_emb, t_emb
                )
                for t_emb in tag_embs.values()
            ]
            concept.relevance_to_tags = round(
                max(sims) if sims else 0.0, 6
            )

        concepts.sort(
            key=lambda c: c.relevance_to_tags, reverse=True
        )
        return concepts

    def filter_by_domain(
        self,
        concepts: List[Concept],
        domain: str,
    ) -> List[Concept]:
        """Filter concepts by domain affinity.

        Keeps concepts whose ``domain_affinity`` for *domain*
        exceeds 0.0, or all concepts if none have affinity data.

        Args:
            concepts: List of concepts.
            domain: Target domain.

        Returns:
            Filtered list.
        """
        filtered = [
            c for c in concepts
            if c.domain_affinity.get(domain, 0.0) > 0.0
        ]
        return filtered if filtered else concepts


# ------------------------------------------------------------------
# DomainRelevanceScorer
# ------------------------------------------------------------------


class DomainRelevanceScorer:
    """Score domain relevance using multiple factors.

    Combines tag matching (40%), semantic similarity (40%), and
    keyword presence (20%) into a single ``[0, 1]`` score.
    """

    _DOMAIN_KEYWORDS: Dict[str, List[str]] = {
        "medical": [
            "medical", "medicine", "disease", "treatment",
            "patient", "clinical", "diagnosis", "therapy",
            "cancer", "drug", "health",
        ],
        "physics": [
            "physics", "quantum", "energy", "force",
            "particle", "electron", "momentum", "wave",
            "thermodynamic", "electromagnetic",
        ],
        "chemistry": [
            "chemistry", "chemical", "molecule", "reaction",
            "compound", "element", "organic", "acid",
            "catalyst", "bond",
        ],
        "mathematics": [
            "mathematics", "algebra", "calculus", "equation",
            "theorem", "geometry", "statistics", "proof",
        ],
        "computer_science": [
            "computer", "algorithm", "software", "programming",
            "machine learning", "artificial intelligence",
        ],
        "music": [
            "music", "melody", "harmony", "rhythm",
            "instrument", "composition", "song",
        ],
    }

    def __init__(
        self,
        embedding_generator: Optional[EmbeddingGenerator] = None,
    ) -> None:
        self._emb_gen = embedding_generator or EmbeddingGenerator()

    def score_domain_relevance(
        self,
        text: str,
        domain: str,
        tags: List[str],
    ) -> float:
        """Compute relevance of *text* to *domain*.

        Multi-factor: tag_match (40%) + semantic (40%) +
        keywords (20%).

        Args:
            text: Input text.
            domain: Target domain.
            tags: Extracted domain tags.

        Returns:
            Relevance score in ``[0, 1]``.
        """
        tag_score = self._tag_match_score(tags, domain)
        semantic_score = self._semantic_score(text, domain)
        keyword_score = self._keyword_score(text, domain)

        score = (
            0.4 * tag_score
            + 0.4 * semantic_score
            + 0.2 * keyword_score
        )
        return round(min(max(score, 0.0), 1.0), 6)

    def rank_domains_by_relevance(
        self,
        text: str,
        all_domains: List[str],
        tags: List[str],
    ) -> List[Tuple[str, float]]:
        """Rank all domains by relevance to *text*.

        Args:
            text: Input text.
            all_domains: List of domain names.
            tags: Extracted domain tags.

        Returns:
            List of ``(domain, score)`` tuples sorted descending.
        """
        scored = [
            (d, self.score_domain_relevance(text, d, tags))
            for d in all_domains
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def compute_multi_domain_relevance(
        self,
        text: str,
        domains: List[str],
        tags: List[str],
    ) -> Dict[str, float]:
        """Compute relevance scores for multiple domains.

        Args:
            text: Input text.
            domains: List of domain names.
            tags: Extracted domain tags.

        Returns:
            Dictionary mapping domain to relevance score.
        """
        return {
            d: self.score_domain_relevance(text, d, tags)
            for d in domains
        }

    # ---- internals -------------------------------------------------

    def _tag_match_score(
        self, tags: List[str], domain: str
    ) -> float:
        """Score how well tags match a domain."""
        if not tags:
            return 0.0
        domain_lower = domain.lower()
        matches = sum(
            1 for t in tags
            if domain_lower in t.lower()
            or t.lower() in domain_lower
        )
        return min(matches / max(len(tags), 1), 1.0)

    def _semantic_score(
        self, text: str, domain: str
    ) -> float:
        """Score semantic similarity between text and domain."""
        text_emb = self._emb_gen.generate_text_embedding(text)
        domain_emb = self._emb_gen.generate_text_embedding(
            domain
        )
        sim = self._emb_gen.compute_embedding_similarity(
            text_emb, domain_emb
        )
        return max(0.0, sim)

    def _keyword_score(
        self, text: str, domain: str
    ) -> float:
        """Score keyword overlap with domain vocabulary."""
        keywords = self._DOMAIN_KEYWORDS.get(
            domain.lower(), []
        )
        if not keywords:
            return 0.0
        text_lower = text.lower()
        hits = sum(1 for kw in keywords if kw in text_lower)
        return min(hits / max(len(keywords), 1), 1.0)


# ------------------------------------------------------------------
# SemanticUnderstandingPipeline
# ------------------------------------------------------------------


class SemanticUnderstandingPipeline:
    """Orchestrate all Phase 2.2 components.

    Usage:
        >>> pipeline = SemanticUnderstandingPipeline()
        >>> result = pipeline.understand(norm_result, domains)
    """

    def __init__(
        self,
        config: Optional[Phase2Config] = None,
    ) -> None:
        self._config = config or Phase2Config()
        self._emb_gen = EmbeddingGenerator()
        self._entity_recognizer = EntityRecognizer()
        self._graph_builder = SemanticGraphBuilder(
            self._emb_gen
        )
        self._concept_extractor = KeyConceptExtractor(
            self._emb_gen
        )
        self._domain_scorer = DomainRelevanceScorer(
            self._emb_gen
        )
        self._last_metadata: dict = {}

    def understand(
        self,
        normalization_result: object,
        all_domains: List[str],
    ) -> SemanticResult:
        """Run the full semantic understanding pipeline.

        Args:
            normalization_result: Output of Phase 2.1. Must
                have ``cleaned_text``, ``original_text``, and
                ``extracted_tags`` attributes.
            all_domains: List of all candidate domains.

        Returns:
            A ``SemanticResult`` with all fields populated.
        """
        start = time.perf_counter()
        warnings: List[str] = []

        cleaned = getattr(
            normalization_result, "cleaned_text", ""
        )
        original = getattr(
            normalization_result, "original_text", ""
        )
        tags = getattr(
            normalization_result, "extracted_tags", []
        )

        result = SemanticResult(
            original_text=original,
            cleaned_text=cleaned,
        )

        try:
            # Step 1: text embedding
            result.text_embedding = (
                self._emb_gen.generate_text_embedding(cleaned)
            )
            logger.debug(
                "Text embedding shape: %s",
                result.text_embedding.shape,
            )

            # Step 2: entity extraction
            result.extracted_entities = (
                self._entity_recognizer.extract_entities(
                    cleaned, tags
                )
            )
            logger.info(
                "Extracted %d entities",
                len(result.extracted_entities),
            )

            # Step 3: key concept extraction
            concepts = (
                self._concept_extractor.extract_key_concepts(
                    cleaned
                )
            )
            concepts = (
                self._concept_extractor
                .rank_concepts_by_relevance(concepts, tags)
            )
            result.key_concepts = concepts
            result.concept_scores = {
                c.text: c.overall_importance
                for c in concepts
            }

            # Step 4: semantic graph
            result.semantic_graph = (
                self._graph_builder.build_graph(
                    cleaned,
                    result.extracted_entities,
                    tags,
                )
            )
            result.graph_density = (
                result.semantic_graph.density
            )

            # Step 5: domain relevance scoring
            result.domain_relevance_scores = (
                self._domain_scorer
                .compute_multi_domain_relevance(
                    cleaned, all_domains, tags
                )
            )
            ranked = (
                self._domain_scorer.rank_domains_by_relevance(
                    cleaned, all_domains, tags
                )
            )
            result.ranked_domains = [d for d, _ in ranked]

            # Step 6: confidence score
            result.confidence_score = (
                self._compute_confidence(result)
            )
            result.embedding_cache_hits = (
                self._emb_gen.cache_hits
            )

        except Exception as exc:
            logger.error(
                "Semantic understanding error: %s", exc
            )
            warnings.append(f"Pipeline error: {exc}")

        result.warnings = warnings
        elapsed = (time.perf_counter() - start) * 1000
        result.processing_time_ms = round(elapsed, 2)
        self._last_metadata = {
            "processing_time_ms": result.processing_time_ms,
            "entities_found": len(result.extracted_entities),
            "concepts_found": len(result.key_concepts),
            "domains_scored": len(
                result.domain_relevance_scores
            ),
        }
        return result

    def get_processing_metadata(self) -> dict:
        """Return metadata from the last ``understand`` call.

        Returns:
            Dictionary with timing and count information.
        """
        return dict(self._last_metadata)

    # ---- internals -------------------------------------------------

    @staticmethod
    def _compute_confidence(result: SemanticResult) -> float:
        """Derive an overall confidence score."""
        factors: List[float] = []

        if result.extracted_entities:
            avg_conf = sum(
                e.confidence
                for e in result.extracted_entities
            ) / len(result.extracted_entities)
            factors.append(avg_conf)

        if result.key_concepts:
            avg_imp = sum(
                c.overall_importance
                for c in result.key_concepts
            ) / len(result.key_concepts)
            factors.append(avg_imp)

        if result.domain_relevance_scores:
            top_score = max(
                result.domain_relevance_scores.values()
            )
            factors.append(top_score)

        if not factors:
            return 0.0
        return round(sum(factors) / len(factors), 6)
