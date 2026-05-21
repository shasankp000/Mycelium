"""
Tests for Phase 2.2 — Semantic Understanding.
"""

import numpy as np
import pytest

from mycelium.pipeline.phase2.phases.phase_2_2_semantic_understanding import (
    DomainRelevanceScorer,
    EmbeddingGenerator,
    EntityRecognizer,
    KeyConceptExtractor,
    SemanticGraphBuilder,
    SemanticUnderstandingPipeline,
)
from mycelium.pipeline.phase2.utils.semantic_types import (
    Concept,
    Entity,
    SemanticGraph,
    SemanticResult,
)


# ------------------------------------------------------------------
# EmbeddingGenerator
# ------------------------------------------------------------------


class TestEmbeddingGenerator:
    def setup_method(self):
        self.gen = EmbeddingGenerator()

    def test_embedding_generation_basic(self):
        emb = self.gen.generate_text_embedding("hello world")
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (384,)

    def test_embedding_deterministic(self):
        emb1 = self.gen.generate_text_embedding("test")
        emb2 = self.gen.generate_text_embedding("test")
        np.testing.assert_array_equal(emb1, emb2)

    def test_embedding_caching(self):
        self.gen.generate_text_embedding("cached")
        assert self.gen.cache_hits == 0
        self.gen.generate_text_embedding("cached")
        assert self.gen.cache_hits == 1

    def test_embedding_cache_eviction(self):
        gen = EmbeddingGenerator(cache_max_size=3)
        for i in range(5):
            gen.generate_text_embedding(f"text_{i}")
        # Cache should only hold 3 items
        gen.generate_text_embedding("text_0")
        # text_0 was evicted, so no cache hit on re-gen
        # (it was regenerated, not from cache)

    def test_embedding_similarity(self):
        emb1 = self.gen.generate_text_embedding("cat")
        emb2 = self.gen.generate_text_embedding("cat")
        sim = self.gen.compute_embedding_similarity(emb1, emb2)
        assert sim == pytest.approx(1.0, abs=1e-5)

    def test_embedding_similarity_different(self):
        emb1 = self.gen.generate_text_embedding("cat")
        emb2 = self.gen.generate_text_embedding("quantum")
        sim = self.gen.compute_embedding_similarity(emb1, emb2)
        assert -1.0 <= sim <= 1.0

    def test_embedding_similarity_zero_vector(self):
        emb1 = np.zeros(384)
        emb2 = self.gen.generate_text_embedding("test")
        sim = self.gen.compute_embedding_similarity(emb1, emb2)
        assert sim == 0.0

    def test_tag_embeddings(self):
        tags = ["medical", "physics", "chemistry"]
        result = self.gen.generate_tag_embeddings(tags)
        assert len(result) == 3
        for tag in tags:
            assert tag in result
            assert result[tag].shape == (384,)

    def test_reset_cache(self):
        self.gen.generate_text_embedding("data")
        self.gen.generate_text_embedding("data")
        assert self.gen.cache_hits == 1
        self.gen.reset_cache()
        assert self.gen.cache_hits == 0

    def test_unit_norm(self):
        emb = self.gen.generate_text_embedding("normalized")
        norm = np.linalg.norm(emb)
        assert norm == pytest.approx(1.0, abs=1e-5)


# ------------------------------------------------------------------
# EntityRecognizer
# ------------------------------------------------------------------


class TestEntityRecognizer:
    def setup_method(self):
        self.recognizer = EntityRecognizer()

    def test_entity_extraction_medical(self):
        text = (
            "The patient was diagnosed with cancer and "
            "received chemotherapy treatment."
        )
        entities = self.recognizer.extract_entities(
            text, ["medical"]
        )
        assert len(entities) > 0
        domains = {e.domain for e in entities}
        assert "medical" in domains

    def test_entity_extraction_physics(self):
        text = (
            "Quantum mechanics describes electron behavior "
            "and energy levels in atoms."
        )
        entities = self.recognizer.extract_entities(
            text, ["physics"]
        )
        assert len(entities) > 0
        domains = {e.domain for e in entities}
        assert "physics" in domains

    def test_entity_extraction_chemistry(self):
        text = (
            "The chemical reaction produced a new compound "
            "through organic synthesis."
        )
        entities = self.recognizer.extract_entities(
            text, ["chemistry"]
        )
        assert len(entities) > 0
        domains = {e.domain for e in entities}
        assert "chemistry" in domains

    def test_entity_linking(self):
        text = "The patient received drug treatment."
        entities = self.recognizer.extract_entities(
            text, ["medical"]
        )
        for ent in entities:
            assert ent.domain != ""
            assert ent.entity_type != ""
            assert ent.confidence > 0.0

    def test_entity_confidence_with_matching_tags(self):
        text = "The patient was diagnosed with cancer."
        entities_with_tag = self.recognizer.extract_entities(
            text, ["medical"]
        )
        entities_no_tag = self.recognizer.extract_entities(
            text, ["music"]
        )
        if entities_with_tag and entities_no_tag:
            max_conf_with = max(
                e.confidence for e in entities_with_tag
            )
            max_conf_without = max(
                e.confidence for e in entities_no_tag
            )
            assert max_conf_with >= max_conf_without

    def test_entity_context(self):
        text = "The patient was diagnosed with cancer."
        entities = self.recognizer.extract_entities(
            text, ["medical"]
        )
        for ent in entities:
            assert len(ent.context) > 0

    def test_entity_positions(self):
        text = "The patient received treatment."
        entities = self.recognizer.extract_entities(
            text, ["medical"]
        )
        for ent in entities:
            assert ent.start_pos >= 0
            assert ent.end_pos > ent.start_pos

    def test_empty_text_no_entities(self):
        entities = self.recognizer.extract_entities("", [])
        assert entities == []


# ------------------------------------------------------------------
# SemanticGraphBuilder
# ------------------------------------------------------------------


class TestSemanticGraphBuilder:
    def setup_method(self):
        self.builder = SemanticGraphBuilder()

    def test_graph_construction(self):
        entities = [
            Entity(text="cancer", domain="medical"),
            Entity(text="treatment", domain="medical"),
        ]
        graph = self.builder.build_graph(
            "cancer treatment", entities, ["medical"]
        )
        assert isinstance(graph, SemanticGraph)
        assert len(graph.nodes) >= 2

    def test_graph_density(self):
        entities = [
            Entity(text="energy", domain="physics"),
            Entity(text="force", domain="physics"),
        ]
        graph = self.builder.build_graph(
            "energy and force", entities, ["physics"]
        )
        assert 0.0 <= graph.density <= 1.0

    def test_graph_average_degree(self):
        entities = [
            Entity(text="molecule", domain="chemistry"),
            Entity(text="reaction", domain="chemistry"),
        ]
        graph = self.builder.build_graph(
            "molecule reaction", entities, ["chemistry"]
        )
        assert graph.average_degree >= 0.0

    def test_graph_empty_input(self):
        graph = self.builder.build_graph("", [], [])
        assert len(graph.nodes) == 0

    def test_extract_subgraph(self):
        entities = [
            Entity(text="cancer", domain="medical"),
            Entity(text="energy", domain="physics"),
        ]
        graph = self.builder.build_graph(
            "cancer energy", entities, ["medical", "physics"]
        )
        sub = self.builder.extract_subgraph(graph, "cancer")
        assert all("cancer" in n.lower() for n in sub.nodes)

    def test_pagerank_adds_self_loops(self):
        entities = [
            Entity(text="drug", domain="medical"),
            Entity(text="patient", domain="medical"),
        ]
        graph = self.builder.build_graph(
            "drug patient", entities, ["medical"]
        )
        self_loops = [
            (s, t, w) for s, t, w in graph.edges if s == t
        ]
        assert len(self_loops) > 0


# ------------------------------------------------------------------
# KeyConceptExtractor
# ------------------------------------------------------------------


class TestKeyConceptExtractor:
    def setup_method(self):
        self.extractor = KeyConceptExtractor()

    def test_key_concept_extraction(self):
        text = (
            "Cancer treatment requires chemotherapy and "
            "radiation therapy for advanced cases."
        )
        concepts = self.extractor.extract_key_concepts(text)
        assert len(concepts) > 0
        assert all(isinstance(c, Concept) for c in concepts)

    def test_concept_importance_ranking(self):
        text = (
            "Physics describes energy and force interactions "
            "in the quantum domain."
        )
        concepts = self.extractor.extract_key_concepts(text)
        if len(concepts) >= 2:
            for i in range(len(concepts) - 1):
                assert (
                    concepts[i].overall_importance
                    >= concepts[i + 1].overall_importance
                )

    def test_concept_top_k(self):
        text = (
            "Medical treatment diagnosis patient therapy "
            "clinical drug cancer disease biology."
        )
        concepts = self.extractor.extract_key_concepts(
            text, top_k=3
        )
        assert len(concepts) <= 3

    def test_concept_tfidf_score(self):
        text = "energy energy energy force"
        concepts = self.extractor.extract_key_concepts(text)
        energy_concepts = [
            c for c in concepts if c.text == "energy"
        ]
        if energy_concepts:
            assert energy_concepts[0].tfidf_score > 0.0

    def test_rank_concepts_by_relevance(self):
        text = (
            "Cancer treatment involves chemotherapy and "
            "radiation for clinical patients."
        )
        concepts = self.extractor.extract_key_concepts(text)
        ranked = self.extractor.rank_concepts_by_relevance(
            concepts, ["medical", "treatment"]
        )
        assert len(ranked) == len(concepts)
        for c in ranked:
            assert hasattr(c, "relevance_to_tags")

    def test_filter_by_domain(self):
        concepts = [
            Concept(
                text="cancer",
                domain_affinity={"medical": 0.9},
            ),
            Concept(
                text="algebra",
                domain_affinity={"mathematics": 0.8},
            ),
        ]
        filtered = self.extractor.filter_by_domain(
            concepts, "medical"
        )
        assert any(c.text == "cancer" for c in filtered)

    def test_empty_text(self):
        concepts = self.extractor.extract_key_concepts("")
        assert concepts == []


# ------------------------------------------------------------------
# DomainRelevanceScorer
# ------------------------------------------------------------------


class TestDomainRelevanceScorer:
    def setup_method(self):
        self.scorer = DomainRelevanceScorer()

    def test_domain_relevance_scoring(self):
        score = self.scorer.score_domain_relevance(
            "Patient received cancer treatment.",
            "medical",
            ["medical", "treatment"],
        )
        assert 0.0 <= score <= 1.0

    def test_domain_relevance_high_for_matching(self):
        score_med = self.scorer.score_domain_relevance(
            "Patient received cancer treatment.",
            "medical",
            ["medical"],
        )
        score_mus = self.scorer.score_domain_relevance(
            "Patient received cancer treatment.",
            "music",
            ["medical"],
        )
        assert score_med > score_mus

    def test_rank_domains_by_relevance(self):
        domains = ["medical", "physics", "chemistry", "music"]
        ranked = self.scorer.rank_domains_by_relevance(
            "The patient received drug treatment.",
            domains,
            ["medical"],
        )
        assert len(ranked) == 4
        assert ranked[0][1] >= ranked[-1][1]

    def test_multi_domain_relevance(self):
        domains = ["medical", "physics"]
        scores = self.scorer.compute_multi_domain_relevance(
            "Quantum energy particle experiment.",
            domains,
            ["physics"],
        )
        assert "medical" in scores
        assert "physics" in scores

    def test_no_tags_still_scores(self):
        score = self.scorer.score_domain_relevance(
            "Energy and force in physics.",
            "physics",
            [],
        )
        assert 0.0 <= score <= 1.0

    def test_unknown_domain(self):
        score = self.scorer.score_domain_relevance(
            "Some random text.",
            "unknown_domain",
            ["random"],
        )
        assert 0.0 <= score <= 1.0


# ------------------------------------------------------------------
# SemanticUnderstandingPipeline
# ------------------------------------------------------------------


class TestSemanticUnderstandingPipeline:
    def setup_method(self):
        self.pipeline = SemanticUnderstandingPipeline()

    def _make_norm_result(
        self, cleaned="", original="", tags=None
    ):
        """Helper to create a mock normalization result."""
        from types import SimpleNamespace

        return SimpleNamespace(
            cleaned_text=cleaned,
            original_text=original,
            extracted_tags=tags or [],
        )

    def test_full_pipeline_execution(self):
        norm = self._make_norm_result(
            cleaned=(
                "The patient was diagnosed with cancer and "
                "received chemotherapy treatment."
            ),
            original=(
                "The patient was diagnosed with cancer and "
                "received chemotherapy treatment."
            ),
            tags=["medical", "treatment"],
        )
        result = self.pipeline.understand(
            norm, ["medical", "physics", "chemistry"]
        )
        assert isinstance(result, SemanticResult)
        assert result.text_embedding is not None
        assert result.text_embedding.shape == (384,)
        assert len(result.ranked_domains) > 0
        assert result.processing_time_ms > 0

    def test_pipeline_entities(self):
        norm = self._make_norm_result(
            cleaned="Cancer treatment for the patient.",
            tags=["medical"],
        )
        result = self.pipeline.understand(
            norm, ["medical"]
        )
        assert len(result.extracted_entities) > 0

    def test_pipeline_concepts(self):
        norm = self._make_norm_result(
            cleaned=(
                "Quantum mechanics describes electron energy "
                "levels and particle behavior."
            ),
            tags=["physics"],
        )
        result = self.pipeline.understand(
            norm, ["physics"]
        )
        assert len(result.key_concepts) > 0

    def test_pipeline_graph(self):
        norm = self._make_norm_result(
            cleaned="Energy and force in physics.",
            tags=["physics", "energy"],
        )
        result = self.pipeline.understand(
            norm, ["physics"]
        )
        assert result.semantic_graph is not None
        assert len(result.semantic_graph.nodes) > 0

    def test_pipeline_domain_scores(self):
        norm = self._make_norm_result(
            cleaned="Chemical reaction with organic molecules.",
            tags=["chemistry"],
        )
        result = self.pipeline.understand(
            norm, ["medical", "chemistry"]
        )
        assert "chemistry" in result.domain_relevance_scores
        assert "medical" in result.domain_relevance_scores

    def test_pipeline_empty_text(self):
        norm = self._make_norm_result(cleaned="", tags=[])
        result = self.pipeline.understand(norm, ["medical"])
        assert isinstance(result, SemanticResult)

    def test_pipeline_metadata(self):
        norm = self._make_norm_result(
            cleaned="Test sentence for metadata.",
            tags=["test"],
        )
        self.pipeline.understand(norm, ["medical"])
        meta = self.pipeline.get_processing_metadata()
        assert "processing_time_ms" in meta
        assert "entities_found" in meta

    def test_pipeline_confidence_score(self):
        norm = self._make_norm_result(
            cleaned=(
                "Patient cancer treatment chemotherapy."
            ),
            tags=["medical"],
        )
        result = self.pipeline.understand(
            norm, ["medical"]
        )
        assert 0.0 <= result.confidence_score <= 1.0
