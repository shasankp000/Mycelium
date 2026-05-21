"""
Phase 2: Fusion Engine (Score Combination & Superposition Detection)

Standalone module that:
- Normalizes lens scores into [0, 1]
- Fuses multiple lens scores via weighted averaging
- Detects superposition states (single-domain, multi-domain, ambiguous, no-expert)

Constraints:
- Optional, deterministic, non-breaking
- Works with scores only (no text)
- Missing lenses handled gracefully

Phase 6 Updates:
- Reads fusion weights from tuning_config
- Adds per-domain contribution logging
- Clamps extreme values for robustness
"""

from typing import Dict, Tuple
import sys

# Import tuning config (graceful fallback if not available)
try:
    from tuning_config import (
        FUSION_WEIGHTS,
        SUPERPOSITION_VARIANCE_THRESHOLD,
        ENABLE_LOGGING
    )
except ImportError:
    # Fallback defaults if config not available
    FUSION_WEIGHTS = {"semantic": 0.4, "spectral": 0.3, "confidence": 0.3}
    SUPERPOSITION_VARIANCE_THRESHOLD = 0.1
    ENABLE_LOGGING = False


class ScoreNormalizer:
    """
    Normalizes a dictionary of domain scores to [0, 1].

    Rules:
    - Empty dict -> return empty dict
    - Negative values -> clipped to 0
    - Values > 1 -> clipped to 1
    - Non-numeric or None -> treated as 0
    - Deterministic, no randomness
    """

    def normalize(self, scores: Dict[str, float]) -> Dict[str, float]:
        if not scores:
            return {}

        normalized: Dict[str, float] = {}
        for k, v in scores.items():
            try:
                # Convert to float safely; treat None/non-numeric as 0
                val = float(v) if v is not None else 0.0
            except (TypeError, ValueError):
                val = 0.0
            # Clip into [0, 1]
            if val < 0.0:
                val = 0.0
            if val > 1.0:
                val = 1.0
            normalized[k] = val
        return normalized


class FusionEngine:
    """
    Weighted fusion of multiple lens scores.

    Inputs are dictionaries of domain -> score for each lens, expected in [0,1].
    Missing domains or lenses default to 0.

    Fusion rule per domain:
        fused_score = w1 * semantic + w2 * spectral + w3 * confidence

    Weights auto-normalize to sum to 1.
    Output provides explainable contributions (each weight * normalized score).
    """

    def __init__(
        self,
        semantic_weight: float = None,
        spectral_weight: float = None,
        confidence_weight: float = None,
    ) -> None:
        # Use config values if not explicitly provided (Phase 6)
        self.semantic_weight = float(semantic_weight) if semantic_weight is not None else FUSION_WEIGHTS["semantic"]
        self.spectral_weight = float(spectral_weight) if spectral_weight is not None else FUSION_WEIGHTS["spectral"]
        self.confidence_weight = float(confidence_weight) if confidence_weight is not None else FUSION_WEIGHTS["confidence"]
        self.normalizer = ScoreNormalizer()
        self.enable_logging = ENABLE_LOGGING

    def _normalized_weights(self) -> Tuple[float, float, float]:
        w1, w2, w3 = self.semantic_weight, self.spectral_weight, self.confidence_weight
        total = w1 + w2 + w3
        if total <= 0.0:
            # Graceful fallback: equal weights
            return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        return (w1 / total, w2 / total, w3 / total)

    def fuse(
        self,
        semantic_scores: Dict[str, float],
        spectral_scores: Dict[str, float],
        confidence_scores: Dict[str, float],
    ) -> Dict[str, Dict[str, float]]:
        # Normalize each lens safely
        norm_sem = self.normalizer.normalize(semantic_scores or {})
        norm_spec = self.normalizer.normalize(spectral_scores or {})
        norm_conf = self.normalizer.normalize(confidence_scores or {})

        # Union of all domain keys
        domains = set(norm_sem.keys()) | set(norm_spec.keys()) | set(norm_conf.keys())
        if not domains:
            return {}

        w1, w2, w3 = self._normalized_weights()
        fused: Dict[str, Dict[str, float]] = {}
        
        for d in sorted(domains):
            s = norm_sem.get(d, 0.0)
            sp = norm_spec.get(d, 0.0)
            c = norm_conf.get(d, 0.0)

            # Phase 6: Clamp extreme values for robustness
            s = max(0.0, min(1.0, s))
            sp = max(0.0, min(1.0, sp))
            c = max(0.0, min(1.0, c))

            contrib_sem = w1 * s
            contrib_spec = w2 * sp
            contrib_conf = w3 * c
            fused_score = contrib_sem + contrib_spec + contrib_conf
            
            # Phase 6: Clamp final fused score to [0, 1]
            fused_score = max(0.0, min(1.0, fused_score))

            fused[d] = {
                "fused_score": round(fused_score, 6),
                "semantic": round(contrib_sem, 6),
                "spectral": round(contrib_spec, 6),
                "confidence": round(contrib_conf, 6),
            }
            
            # Phase 6: Log per-domain contributions (if enabled)
            if self.enable_logging and fused_score > 0:
                print(f"  [FUSION] {d}: fused={fused_score:.4f} (sem={contrib_sem:.4f}, spec={contrib_spec:.4f}, conf={contrib_conf:.4f})", file=sys.stderr)

        return fused


class SuperpositionDetector:
    """
    Detects superposition states from fused scores.

    Algorithm:
    - If fused_scores empty -> NO_EXPERT_AVAILABLE
    - Compute mean and variance across fused scores
    - Candidate domains: those with fused_score >= score_threshold
    - Decision rules:
        * No scores ≥ threshold -> NO_EXPERT_AVAILABLE
        * One score ≥ threshold -> SINGLE_DOMAIN
        * Multiple ≥ threshold AND variance ≥ variance_threshold -> MULTI_DOMAIN
        * Multiple ≥ threshold AND variance <  variance_threshold -> AMBIGUOUS
    """

    def detect(
        self,
        fused_scores: Dict[str, float],
        score_threshold: float = 0.5,
        variance_threshold: float = None,  # Phase 6: Use config default
    ) -> Dict:
        # Phase 6: Use config value if not provided
        if variance_threshold is None:
            variance_threshold = SUPERPOSITION_VARIANCE_THRESHOLD
        try:
            if not fused_scores:
                return {
                    "classification": "NO_EXPERT_AVAILABLE",
                    "candidate_domains": [],
                    "variance": 0.0,
                    "fused_scores": {},
                }

            # Convert to deterministic list in sorted key order
            domains_sorted = sorted(fused_scores.keys())
            values = [float(fused_scores[d]) for d in domains_sorted]

            # Compute mean
            mean_val = sum(values) / float(len(values))
            # Population variance (deterministic)
            var = sum((v - mean_val) ** 2 for v in values) / float(len(values))

            # Candidate domains meeting the threshold
            candidates = [d for d in domains_sorted if fused_scores[d] >= score_threshold]

            if len(candidates) == 0:
                classification = "NO_EXPERT_AVAILABLE"
            elif len(candidates) == 1:
                classification = "SINGLE_DOMAIN"
            else:
                classification = "MULTI_DOMAIN" if var >= variance_threshold else "AMBIGUOUS"

            # Sort candidates by score descending for readability
            candidates_sorted = sorted(candidates, key=lambda d: fused_scores[d], reverse=True)

            return {
                "classification": classification,
                "candidate_domains": candidates_sorted,
                "variance": round(var, 6),
                "fused_scores": {d: round(fused_scores[d], 6) for d in domains_sorted},
            }
        except Exception:
            # Graceful degradation: never raise
            return {
                "classification": "NO_EXPERT_AVAILABLE",
                "candidate_domains": [],
                "variance": 0.0,
                "fused_scores": {},
            }


def _flatten_fused_scores(fused: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Helper to extract domain -> fused_score from fusion output."""
    return {d: info.get("fused_score", 0.0) for d, info in fused.items()} if fused else {}


def _print_case(title: str, fused: Dict[str, Dict[str, float]], detection: Dict) -> None:
    print(f"\n=== {title} ===")
    print("Fused Scores:")
    # Deterministic order
    for d in sorted(fused.keys()):
        info = fused[d]
        print(
            f"  {d}: fused={info['fused_score']:.3f} | "
            f"semantic={info['semantic']:.3f} spectral={info['spectral']:.3f} confidence={info['confidence']:.3f}"
        )
    print("Classification:")
    print(
        f"  {detection.get('classification')} | "
        f"candidates={detection.get('candidate_domains')} | variance={detection.get('variance')}"
    )


if __name__ == "__main__":
    # Minimal tests: Single-domain, Multi-domain, Ambiguous, No-expert
    normalizer = ScoreNormalizer()
    fusion = FusionEngine(semantic_weight=0.4, spectral_weight=0.3, confidence_weight=0.3)
    detector = SuperpositionDetector()

    # 1) Single-domain case
    sem1 = {"astronomy": 0.9, "automobile": 0.1}
    spec1 = {"astronomy": 0.8, "automobile": 0.2}
    conf1 = {"astronomy": 0.7, "automobile": 0.3}
    fused1 = fusion.fuse(sem1, spec1, conf1)
    det1 = detector.detect(_flatten_fused_scores(fused1), score_threshold=0.5, variance_threshold=0.1)
    _print_case("Single-domain", fused1, det1)

    # 2) Multi-domain case (increase variance by adding a low-scoring third domain)
    sem2 = {"astronomy": 0.8, "physics": 0.75, "automobile": 0.05}
    spec2 = {"astronomy": 0.6, "physics": 0.7, "automobile": 0.05}
    conf2 = {"astronomy": 0.7, "physics": 0.8, "automobile": 0.05}
    fused2 = fusion.fuse(sem2, spec2, conf2)
    det2 = detector.detect(_flatten_fused_scores(fused2), score_threshold=0.5, variance_threshold=0.1)
    _print_case("Multi-domain", fused2, det2)

    # 3) Ambiguous case (multiple above threshold with low variance)
    sem3 = {"astronomy": 0.65, "physics": 0.68}
    spec3 = {"astronomy": 0.66, "physics": 0.67}
    conf3 = {"astronomy": 0.67, "physics": 0.66}
    fused3 = fusion.fuse(sem3, spec3, conf3)
    det3 = detector.detect(_flatten_fused_scores(fused3), score_threshold=0.6, variance_threshold=0.01)
    _print_case("Ambiguous", fused3, det3)

    # 4) No-expert case (empty inputs)
    fused4 = fusion.fuse({}, {}, {})
    det4 = detector.detect(_flatten_fused_scores(fused4), score_threshold=0.5, variance_threshold=0.1)
    _print_case("No-expert", fused4, det4)

    print("\n✅ Phase 2 Fusion Engine minimal tests completed.")
