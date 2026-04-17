"""
Phase 3 — Layer 1 Contradiction Analyzer.

Analyses a TRM (Thinking and Reasoning Model) output to detect
contradictions, hallucinations, broken reasoning chains, and
evidence misalignment.

Components:
    * EmbeddingGenerator       — text → embedding vectors
    * ContradictionDetector    — answer vs reasoning cosine check
    * OODAnalyzer              — Mahalanobis-based hallucination check
    * ReasoningChainValidator  — step-to-step flow check
    * EvidenceAlignmentChecker — answer concepts vs evidence
    * ContradictionAnalyzer    — orchestrator for all sub-checks

Example:
    >>> from phase3_validation.phases.phase_3_layer_1_contradiction import (
    ...     ContradictionAnalyzer,
    ... )
    >>> analyzer = ContradictionAnalyzer()
    >>> result = analyzer.analyze(
    ...     answer="The drug reduces fever.",
    ...     reasoning_steps=["Drug X acts on pathway Y.", "Y reduces fever."],
    ...     evidence=["Drug X inhibits prostaglandin synthesis."],
    ... )
    >>> print(result.validation_result_class)
    'all_pass'
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from phase3_validation.config.validation_config import ValidationConfig
from phase3_validation.phases.layer_1_types import Layer1Result

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: cosine similarity
# ---------------------------------------------------------------------------


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity between two 1-D arrays.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in ``[-1, 1]``.  Returns 0.0 when either
        vector has zero norm.
    """
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# EmbeddingGenerator
# ---------------------------------------------------------------------------


class EmbeddingGenerator:
    """Convert text to fixed-dimension embedding vectors.

    Uses a simple character n-gram + hashing approach by default
    (``use_simple=True``) to avoid heavyweight model downloads.
    When ``use_simple=False`` the generator will attempt to load a
    ``sentence-transformers`` model specified in *config*.

    Attributes:
        config: Validation configuration.
        _cache: Optional embedding cache keyed by text.
    """

    def __init__(
        self,
        config: Optional[ValidationConfig] = None,
        use_simple: Optional[bool] = None,
    ) -> None:
        """Initialise the embedding generator.

        Args:
            config: Validation configuration.  Uses defaults when
                not provided.
            use_simple: Override ``config.use_simple_embeddings``
                when provided.
        """
        self.config = config or ValidationConfig()
        self._use_simple: bool = (
            use_simple
            if use_simple is not None
            else self.config.use_simple_embeddings
        )
        self._cache: Dict[str, np.ndarray] = {}
        self._st_model = None  # sentence-transformers model (lazy)

        logger.debug(
            "EmbeddingGenerator init: use_simple=%s dim=%d",
            self._use_simple,
            self.config.embedding_dim,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate a fixed-dimension embedding for *text*.

        Results are cached by default.

        Args:
            text: Input string to embed.

        Returns:
            1-D numpy array of shape ``(embedding_dim,)``.
        """
        if not text or not text.strip():
            return np.zeros(self.config.embedding_dim, dtype=np.float32)

        if text in self._cache:
            return self._cache[text]

        if self._use_simple:
            vec = self._simple_embedding(text)
        else:
            vec = self._sentence_transformer_embedding(text)

        self._cache[text] = vec
        return vec

    def generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a list of texts.

        Args:
            texts: Input strings.

        Returns:
            List of 1-D numpy arrays.
        """
        return [self.generate_embedding(t) for t in texts]

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------

    def _simple_embedding(self, text: str) -> np.ndarray:
        """Character n-gram hashing embedding.

        Produces a deterministic, lightweight embedding that
        captures partial lexical overlap without any model.

        Args:
            text: Input string.

        Returns:
            Normalised float32 vector of shape ``(embedding_dim,)``.
        """
        dim = self.config.embedding_dim
        vec = np.zeros(dim, dtype=np.float64)

        text_lower = text.lower()
        # Use 2-gram and 3-gram character overlaps
        for n in (2, 3):
            for i in range(len(text_lower) - n + 1):
                ngram = text_lower[i: i + n]
                idx = hash(ngram) % dim
                vec[idx] += 1.0

        # Add word-level features
        words = re.findall(r"\b\w+\b", text_lower)
        for word in words:
            idx = hash(word) % dim
            vec[idx] += 2.0  # words weighted higher

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.astype(np.float32)

    def _sentence_transformer_embedding(
        self, text: str
    ) -> np.ndarray:
        """Generate embedding using sentence-transformers.

        Falls back to simple embedding if the library or model
        is unavailable.

        Args:
            text: Input string.

        Returns:
            Float32 embedding vector.
        """
        if self._st_model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self._st_model = SentenceTransformer(
                    self.config.sentence_transformer_model
                )
                logger.info(
                    "Loaded sentence-transformer model: %s",
                    self.config.sentence_transformer_model,
                )
            except Exception as exc:
                logger.warning(
                    "Cannot load sentence-transformers (%s); "
                    "falling back to simple embeddings.",
                    exc,
                )
                self._use_simple = True
                return self._simple_embedding(text)

        try:
            vec = self._st_model.encode(
                text, convert_to_numpy=True, show_progress_bar=False
            )
            return vec.astype(np.float32)
        except Exception as exc:
            logger.warning(
                "sentence-transformers encode failed (%s); "
                "falling back to simple embeddings.",
                exc,
            )
            return self._simple_embedding(text)


# ---------------------------------------------------------------------------
# ContradictionDetector
# ---------------------------------------------------------------------------


class ContradictionDetector:
    """Detect contradictions between answer and reasoning steps.

    Compares the answer embedding against the mean of all
    reasoning-step embeddings using cosine similarity.  If
    similarity falls below the configured threshold the answer is
    considered contradictory.

    Attributes:
        config: Validation configuration.
        embedding_gen: Embedding generator to use.
    """

    def __init__(
        self,
        config: Optional[ValidationConfig] = None,
        embedding_gen: Optional[EmbeddingGenerator] = None,
    ) -> None:
        """Initialise the contradiction detector.

        Args:
            config: Validation configuration.
            embedding_gen: Shared embedding generator.  A new one is
                created if not provided.
        """
        self.config = config or ValidationConfig()
        self.embedding_gen = embedding_gen or EmbeddingGenerator(
            config=self.config
        )

    def detect(
        self,
        answer: str,
        reasoning_steps: List[str],
    ) -> Tuple[bool, float, str]:
        """Check whether *answer* contradicts *reasoning_steps*.

        Args:
            answer: The TRM answer text.
            reasoning_steps: Ordered list of reasoning step texts.

        Returns:
            Tuple of:
            * ``contradiction_found`` (bool) — True when a
              contradiction is detected.
            * ``similarity`` (float) — Cosine similarity between
              answer and mean reasoning embedding.
            * ``reason`` (str) — ``"answer_reasoning_mismatch"``
              when found, else ``""``.
        """
        if not answer or not answer.strip():
            return False, 0.0, ""

        valid_steps = [s for s in reasoning_steps if s and s.strip()]
        if not valid_steps:
            return False, 0.0, ""

        answer_emb = self.embedding_gen.generate_embedding(answer)
        step_embs = self.embedding_gen.generate_embeddings(valid_steps)

        mean_step_emb = np.mean(np.stack(step_embs), axis=0)
        similarity = _cosine_similarity(answer_emb, mean_step_emb)

        contradiction = similarity < self.config.contradiction_threshold
        reason = "answer_reasoning_mismatch" if contradiction else ""

        logger.debug(
            "ContradictionDetector: similarity=%.3f threshold=%.3f "
            "contradiction=%s",
            similarity,
            self.config.contradiction_threshold,
            contradiction,
        )
        return contradiction, similarity, reason


# ---------------------------------------------------------------------------
# OODAnalyzer
# ---------------------------------------------------------------------------


class OODAnalyzer:
    """Out-Of-Distribution analyser using Mahalanobis distance.

    Checks whether the answer embedding lies far outside the
    distribution of the evidence embeddings.  A large Mahalanobis
    distance suggests the answer was hallucinated rather than
    grounded in the evidence.

    Attributes:
        config: Validation configuration.
        embedding_gen: Embedding generator to use.
    """

    def __init__(
        self,
        config: Optional[ValidationConfig] = None,
        embedding_gen: Optional[EmbeddingGenerator] = None,
    ) -> None:
        """Initialise the OOD analyser.

        Args:
            config: Validation configuration.
            embedding_gen: Shared embedding generator.
        """
        self.config = config or ValidationConfig()
        self.embedding_gen = embedding_gen or EmbeddingGenerator(
            config=self.config
        )

    def compute_mahalanobis_distance(
        self,
        point: np.ndarray,
        corpus: np.ndarray,
    ) -> float:
        """Compute Mahalanobis distance from *point* to *corpus*.

        Falls back to scaled Euclidean distance when the covariance
        matrix is singular.

        Args:
            point: Query vector of shape ``(d,)``.
            corpus: Reference matrix of shape ``(n, d)``.

        Returns:
            Non-negative distance value.
        """
        if corpus.shape[0] < 2:
            # Only one evidence point — use Euclidean
            mean = corpus[0]
            return float(np.linalg.norm(point - mean))

        mean = np.mean(corpus, axis=0)
        centered = corpus - mean
        cov = np.cov(centered.T) + np.eye(corpus.shape[1]) * 1e-6

        try:
            inv_cov = np.linalg.inv(cov)
            diff = point - mean
            dist = float(
                np.sqrt(np.maximum(0.0, diff @ inv_cov @ diff))
            )
        except np.linalg.LinAlgError:
            # Singular covariance — fall back to scaled Euclidean
            dist = float(np.linalg.norm(point - mean))

        return dist

    def analyze(
        self,
        answer: str,
        evidence: List[str],
    ) -> Tuple[bool, float, str]:
        """Detect whether *answer* is out-of-distribution w.r.t. *evidence*.

        Args:
            answer: The TRM answer text.
            evidence: List of evidence strings.

        Returns:
            Tuple of:
            * ``hallucination_detected`` (bool)
            * ``ood_score`` (float) — Mahalanobis distance
            * ``reason`` (str) — ``"hallucination_detected"`` or ``""``
        """
        if not answer or not answer.strip():
            return False, 0.0, ""

        valid_evidence = [e for e in evidence if e and e.strip()]
        if not valid_evidence:
            # No evidence to compare against — cannot determine OOD
            return False, 0.0, ""

        answer_emb = self.embedding_gen.generate_embedding(answer)
        evidence_embs = np.stack(
            self.embedding_gen.generate_embeddings(valid_evidence)
        )

        ood_score = self.compute_mahalanobis_distance(
            answer_emb, evidence_embs
        )

        hallucination = ood_score > self.config.ood_threshold
        reason = "hallucination_detected" if hallucination else ""

        logger.debug(
            "OODAnalyzer: ood_score=%.3f threshold=%.3f "
            "hallucination=%s",
            ood_score,
            self.config.ood_threshold,
            hallucination,
        )
        return hallucination, ood_score, reason


# ---------------------------------------------------------------------------
# ReasoningChainValidator
# ---------------------------------------------------------------------------


class ReasoningChainValidator:
    """Validate that reasoning steps form a coherent chain.

    Checks consecutive step similarity.  If mean pairwise
    similarity falls below the configured threshold, the chain
    is considered broken.

    Attributes:
        config: Validation configuration.
        embedding_gen: Embedding generator to use.
    """

    def __init__(
        self,
        config: Optional[ValidationConfig] = None,
        embedding_gen: Optional[EmbeddingGenerator] = None,
    ) -> None:
        """Initialise the reasoning chain validator.

        Args:
            config: Validation configuration.
            embedding_gen: Shared embedding generator.
        """
        self.config = config or ValidationConfig()
        self.embedding_gen = embedding_gen or EmbeddingGenerator(
            config=self.config
        )

    def validate(
        self,
        reasoning_steps: List[str],
    ) -> Tuple[bool, float, str]:
        """Validate reasoning chain coherence.

        Args:
            reasoning_steps: Ordered list of reasoning step texts.

        Returns:
            Tuple of:
            * ``chain_valid`` (bool) — True when the chain is valid.
            * ``chain_validity_score`` (float) — Mean pairwise
              cosine similarity in ``[0, 1]``.
            * ``reason`` (str) — ``"broken_reasoning_chain"`` when
              invalid, else ``""``.
        """
        valid_steps = [s for s in reasoning_steps if s and s.strip()]

        if len(valid_steps) < 2:
            # Single or no steps — chain trivially valid
            return True, 1.0, ""

        step_embs = self.embedding_gen.generate_embeddings(valid_steps)
        pairwise_sims: List[float] = []

        for i in range(len(step_embs) - 1):
            sim = _cosine_similarity(step_embs[i], step_embs[i + 1])
            pairwise_sims.append(sim)

        mean_sim = float(np.mean(pairwise_sims))
        chain_valid = mean_sim >= self.config.reasoning_chain_threshold
        reason = "" if chain_valid else "broken_reasoning_chain"

        logger.debug(
            "ReasoningChainValidator: mean_sim=%.3f threshold=%.3f "
            "valid=%s",
            mean_sim,
            self.config.reasoning_chain_threshold,
            chain_valid,
        )
        return chain_valid, mean_sim, reason


# ---------------------------------------------------------------------------
# EvidenceAlignmentChecker
# ---------------------------------------------------------------------------


class EvidenceAlignmentChecker:
    """Verify that the answer is supported by the evidence.

    Extracts key phrases from the answer and checks what fraction
    of them can be found in the evidence corpus.

    Attributes:
        config: Validation configuration.
    """

    def __init__(
        self,
        config: Optional[ValidationConfig] = None,
    ) -> None:
        """Initialise the evidence alignment checker.

        Args:
            config: Validation configuration.
        """
        self.config = config or ValidationConfig()

    # ------------------------------------------------------------------
    # Key-phrase extraction
    # ------------------------------------------------------------------

    def extract_key_phrases(self, text: str) -> List[str]:
        """Extract meaningful key phrases from *text*.

        Uses a simple stop-word filter + word tokenisation.

        Args:
            text: Input text.

        Returns:
            List of lower-cased key phrases (single and multi-words).
        """
        if not text or not text.strip():
            return []

        _STOP_WORDS = {
            "a", "an", "the", "is", "are", "was", "were",
            "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "shall", "can",
            "to", "of", "in", "for", "on", "with", "at",
            "by", "from", "as", "into", "through", "about",
            "and", "or", "but", "if", "then", "that", "this",
            "it", "its", "i", "we", "you", "he", "she", "they",
        }

        words = re.findall(r"\b[a-zA-Z]\w*\b", text.lower())
        key_words = [w for w in words if w not in _STOP_WORDS and len(w) > 2]

        # Add bigrams
        phrases: List[str] = list(key_words)
        for i in range(len(key_words) - 1):
            phrases.append(f"{key_words[i]} {key_words[i + 1]}")

        return phrases

    # ------------------------------------------------------------------
    # Alignment check
    # ------------------------------------------------------------------

    def check_alignment(
        self,
        answer: str,
        evidence: List[str],
    ) -> Tuple[bool, float, str]:
        """Check whether answer concepts are present in evidence.

        Args:
            answer: The TRM answer text.
            evidence: List of evidence strings.

        Returns:
            Tuple of:
            * ``aligned`` (bool) — True when sufficient answer
              concepts are found in evidence.
            * ``alignment_score`` (float) — Fraction of answer
              concepts found in evidence.
            * ``reason`` (str) — ``"evidence_contradiction"`` when
              not aligned, else ``""``.
        """
        if not answer or not answer.strip():
            return True, 1.0, ""

        valid_evidence = [e for e in evidence if e and e.strip()]
        if not valid_evidence:
            # No evidence — alignment cannot be assessed; treat as
            # non-failure (insufficient data, not a contradiction).
            return True, 1.0, ""

        answer_phrases = self.extract_key_phrases(answer)
        if not answer_phrases:
            return True, 1.0, ""

        evidence_text = " ".join(valid_evidence).lower()
        found = sum(1 for ph in answer_phrases if ph in evidence_text)
        alignment_score = found / len(answer_phrases)

        aligned = alignment_score >= self.config.evidence_alignment_threshold
        reason = "" if aligned else "evidence_contradiction"

        logger.debug(
            "EvidenceAlignmentChecker: score=%.3f threshold=%.3f "
            "aligned=%s",
            alignment_score,
            self.config.evidence_alignment_threshold,
            aligned,
        )
        return aligned, alignment_score, reason


# ---------------------------------------------------------------------------
# ContradictionAnalyzer — main orchestrator
# ---------------------------------------------------------------------------


class ContradictionAnalyzer:
    """Orchestrate all Layer 1 contradiction checks.

    Runs the four sub-checkers in sequence and assembles a
    :class:`~phase3_validation.phases.layer_1_types.Layer1Result`.

    Attributes:
        config: Shared validation configuration.
        embedding_gen: Shared embedding generator.
        contradiction_detector: Answer-reasoning checker.
        ood_analyzer: Hallucination detector.
        chain_validator: Reasoning chain coherence checker.
        evidence_checker: Evidence alignment checker.
    """

    def __init__(
        self,
        config: Optional[ValidationConfig] = None,
    ) -> None:
        """Initialise all sub-checkers with a shared config.

        Args:
            config: Validation configuration.  Uses defaults when
                not provided.
        """
        self.config = config or ValidationConfig()
        self.embedding_gen = EmbeddingGenerator(config=self.config)
        self.contradiction_detector = ContradictionDetector(
            config=self.config,
            embedding_gen=self.embedding_gen,
        )
        self.ood_analyzer = OODAnalyzer(
            config=self.config,
            embedding_gen=self.embedding_gen,
        )
        self.chain_validator = ReasoningChainValidator(
            config=self.config,
            embedding_gen=self.embedding_gen,
        )
        self.evidence_checker = EvidenceAlignmentChecker(
            config=self.config,
        )

        logger.info("ContradictionAnalyzer initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        answer: str,
        reasoning_steps: List[str],
        evidence: List[str],
    ) -> Layer1Result:
        """Run all Layer 1 checks and return a consolidated result.

        Decision logic (in order of precedence):
        1. If ``contradiction_detected`` AND
           ``answer_reasoning_similarity`` < threshold → ``CRITICAL``
        2. If ``hallucination_detected`` (OOD > threshold) → ``CRITICAL``
        3. If chain broken AND ``chain_validity_score`` far below
           threshold → ``MAJOR``
        4. If evidence alignment low but not missing → ``MINOR``
        5. Otherwise → ``NONE`` (all pass)

        Args:
            answer: TRM answer text.
            reasoning_steps: Ordered list of reasoning step texts.
            evidence: List of evidence strings.

        Returns:
            :class:`Layer1Result` with all sub-scores and final
            classification.
        """
        affected_components: List[str] = []
        reasons: List[str] = []

        # --- Sub-check 1: Contradiction detection ---
        contradiction_found, similarity, contradiction_reason = (
            self.contradiction_detector.detect(answer, reasoning_steps)
        )
        if contradiction_found:
            affected_components.append("ContradictionDetector")
            reasons.append(contradiction_reason)

        # --- Sub-check 2: OOD / hallucination ---
        hallucination_found, ood_score, ood_reason = (
            self.ood_analyzer.analyze(answer, evidence)
        )
        if hallucination_found:
            affected_components.append("OODAnalyzer")
            reasons.append(ood_reason)

        # --- Sub-check 3: Reasoning chain ---
        chain_valid, chain_score, chain_reason = (
            self.chain_validator.validate(reasoning_steps)
        )
        if not chain_valid:
            affected_components.append("ReasoningChainValidator")
            reasons.append(chain_reason)

        # --- Sub-check 4: Evidence alignment ---
        evidence_aligned, alignment_score, alignment_reason = (
            self.evidence_checker.check_alignment(answer, evidence)
        )
        if not evidence_aligned:
            affected_components.append("EvidenceAlignmentChecker")
            reasons.append(alignment_reason)

        # --- Determine severity and classification ---
        severity, result_class, primary_reason = (
            self._classify(
                contradiction_found=contradiction_found,
                hallucination_found=hallucination_found,
                chain_valid=chain_valid,
                evidence_aligned=evidence_aligned,
                chain_score=chain_score,
                alignment_score=alignment_score,
                reasons=reasons,
            )
        )

        contradiction_detected = (
            contradiction_found or hallucination_found
        )

        problem_description = self._build_description(
            severity, primary_reason, affected_components
        )

        result = Layer1Result(
            contradiction_detected=contradiction_detected,
            severity=severity,
            reason=primary_reason,
            answer_reasoning_similarity=similarity,
            ood_score=ood_score,
            chain_validity_score=chain_score,
            evidence_alignment_score=alignment_score,
            problem_description=problem_description,
            affected_components=affected_components,
            validation_result_class=result_class,
        )

        logger.info(
            "Layer1 analysis complete: class=%s severity=%s "
            "contradiction=%s",
            result_class,
            severity,
            contradiction_detected,
        )
        return result

    # ------------------------------------------------------------------
    # Classification logic
    # ------------------------------------------------------------------

    def _classify(
        self,
        contradiction_found: bool,
        hallucination_found: bool,
        chain_valid: bool,
        evidence_aligned: bool,
        chain_score: float,
        alignment_score: float,
        reasons: List[str],
    ) -> Tuple[str, str, str]:
        """Derive severity, result class, and primary reason.

        Args:
            contradiction_found: Whether answer contradicts reasoning.
            hallucination_found: Whether OOD analysis flagged hallucination.
            chain_valid: Whether the reasoning chain is coherent.
            evidence_aligned: Whether the answer aligns with evidence.
            chain_score: Numerical chain validity score.
            alignment_score: Numerical evidence alignment score.
            reasons: List of reason strings from sub-checks.

        Returns:
            Tuple of (severity, result_class, primary_reason).
        """
        primary_reason = reasons[0] if reasons else ""

        if contradiction_found or hallucination_found:
            return "CRITICAL", "complete_failure", primary_reason

        if not chain_valid:
            # Distinguish major vs minor based on how broken the chain is
            gap = self.config.reasoning_chain_threshold - chain_score
            if gap > 0.2:
                return "MAJOR", "major_failure", primary_reason
            return "MINOR", "minor_discrepancy", primary_reason

        if not evidence_aligned:
            gap = self.config.evidence_alignment_threshold - alignment_score
            if gap > 0.3:
                return "MAJOR", "major_failure", primary_reason
            return "MINOR", "minor_discrepancy", primary_reason

        return "NONE", "all_pass", ""

    # ------------------------------------------------------------------
    # Human-readable description
    # ------------------------------------------------------------------

    @staticmethod
    def _build_description(
        severity: str,
        reason: str,
        affected_components: List[str],
    ) -> str:
        """Build a human-readable description of the detected issue.

        Args:
            severity: One of CRITICAL, MAJOR, MINOR, NONE.
            reason: Primary failure reason string.
            affected_components: Component names involved.

        Returns:
            Descriptive string, or empty when no issue.
        """
        if severity == "NONE":
            return ""

        component_str = (
            ", ".join(affected_components)
            if affected_components
            else "unknown"
        )
        reason_map = {
            "answer_reasoning_mismatch": (
                "Answer does not align with reasoning steps"
            ),
            "hallucination_detected": (
                "Answer appears to be out-of-distribution "
                "(possible hallucination)"
            ),
            "broken_reasoning_chain": (
                "Reasoning steps do not flow coherently"
            ),
            "evidence_contradiction": (
                "Answer concepts are not supported by evidence"
            ),
        }
        description = reason_map.get(reason, reason)
        return (
            f"[{severity}] {description} "
            f"(components: {component_str})"
        )
