"""
Phase 6: Tuning Configuration

This file contains all tunable hyperparameters for the multi-lens routing system.
All changes here are reversible and documented.

IMPORTANT: These values are conservative defaults. Adjust based on observed metrics.
"""

# ============================================================================
# FUSION WEIGHTS
# ============================================================================
# Controls the relative importance of each lens in the fusion process.
# CONSTRAINT: Must sum to 1.0
# 
# Rationale for defaults:
# - semantic (0.45): Slight increase from 0.4 - semantic similarity is most reliable
# - spectral (0.35): Increase from 0.3 - spectral helps with multi-domain detection
# - confidence (0.20): Decrease from 0.3 - reduces over-reliance on Lens 2 confidence
FUSION_WEIGHTS = {
    "semantic": 0.45,
    "spectral": 0.35,
    "confidence": 0.20
}

# ============================================================================
# SUPERPOSITION VARIANCE THRESHOLD
# ============================================================================
# Variance threshold for classifying as MULTI_DOMAIN vs SINGLE_DOMAIN/AMBIGUOUS.
# Higher values → fewer MULTI_DOMAIN classifications (more conservative)
# Lower values → more MULTI_DOMAIN classifications (more sensitive)
#
# DEFAULT: 0.08 (reduced from 0.1)
# Rationale: Phase 5 showed low variance (0.001422), reducing threshold to 0.08
# allows better multi-domain detection while remaining conservative.
SUPERPOSITION_VARIANCE_THRESHOLD = 0.08

# ============================================================================
# DOMAIN SCORE THRESHOLD
# ============================================================================
# Minimum fused score for a domain to be considered "strong enough" to override
# ATTRIBUTE_ONLY classification.
#
# DEFAULT: 0.45
# Rationale: Conservative threshold - only override ATTRIBUTE_ONLY when we have
# moderate confidence (45%) that a domain is relevant.
DOMAIN_SCORE_THRESHOLD = 0.45

# ============================================================================
# COVERAGE THRESHOLD
# ============================================================================
# Minimum coverage required before expert selection can stop.
# Lower values → fewer experts selected (more efficient)
# Higher values → more experts selected (more coverage)
#
# DEFAULT: 0.7 (reduced from 0.8)
# Rationale: Phase 5 showed coverage_met=False frequently. Reducing to 0.7
# allows earlier stopping while maintaining good coverage.
COVERAGE_THRESHOLD = 0.7

# ============================================================================
# SOFT-STOP COVERAGE THRESHOLD
# ============================================================================
# Allow stopping if coverage reaches this percentage of COVERAGE_THRESHOLD.
# This provides a "good enough" early exit.
#
# DEFAULT: 0.9 (90% of coverage threshold)
# Rationale: If we hit 90% of target coverage, additional experts may not
# provide significant value. This reduces over-selection.
SOFT_STOP_PERCENTAGE = 0.9

# ============================================================================
# MAX EXPERTS
# ============================================================================
# Maximum number of experts to select regardless of coverage.
# Prevents runaway selection in edge cases.
#
# DEFAULT: 3
# Rationale: Most real-world queries span at most 2-3 domains. Beyond this,
# we're likely adding noise.
MAX_EXPERTS = 3

# ============================================================================
# ATTRIBUTE_ONLY OVERRIDE
# ============================================================================
# Enable the rule that prevents ATTRIBUTE_ONLY classification when:
# - Lens 2 found an object-level domain
# - Fusion score ≥ DOMAIN_SCORE_THRESHOLD
#
# DEFAULT: True
# Rationale: Reduces false ATTRIBUTE_ONLY classifications when we have
# moderate confidence in a domain.
ENABLE_ATTRIBUTE_OVERRIDE = True

# ============================================================================
# LOGGING & METRICS
# ============================================================================
# Enable lightweight logging for observability.
# Set to False to disable all logging output.
ENABLE_LOGGING = True

# Log every Nth request (1 = all, 10 = every 10th, etc.)
# Useful for reducing log verbosity in production.
LOG_SAMPLE_RATE = 1

# ============================================================================
# VALIDATION
# ============================================================================
def validate_config():
    """Validate that configuration is internally consistent."""
    errors = []
    
    # Check fusion weights sum to 1.0
    weight_sum = sum(FUSION_WEIGHTS.values())
    if not (0.99 <= weight_sum <= 1.01):  # Allow floating point tolerance
        errors.append(f"Fusion weights must sum to 1.0, got {weight_sum}")
    
    # Check all weights are positive
    for key, value in FUSION_WEIGHTS.items():
        if value < 0 or value > 1:
            errors.append(f"Fusion weight '{key}' must be in [0,1], got {value}")
    
    # Check thresholds are in valid range
    if not (0 <= SUPERPOSITION_VARIANCE_THRESHOLD <= 1):
        errors.append(f"SUPERPOSITION_VARIANCE_THRESHOLD must be in [0,1], got {SUPERPOSITION_VARIANCE_THRESHOLD}")
    
    if not (0 <= DOMAIN_SCORE_THRESHOLD <= 1):
        errors.append(f"DOMAIN_SCORE_THRESHOLD must be in [0,1], got {DOMAIN_SCORE_THRESHOLD}")
    
    if not (0 <= COVERAGE_THRESHOLD <= 1):
        errors.append(f"COVERAGE_THRESHOLD must be in [0,1], got {COVERAGE_THRESHOLD}")
    
    if not (0 <= SOFT_STOP_PERCENTAGE <= 1):
        errors.append(f"SOFT_STOP_PERCENTAGE must be in [0,1], got {SOFT_STOP_PERCENTAGE}")
    
    if MAX_EXPERTS < 1:
        errors.append(f"MAX_EXPERTS must be >= 1, got {MAX_EXPERTS}")
    
    if LOG_SAMPLE_RATE < 1:
        errors.append(f"LOG_SAMPLE_RATE must be >= 1, got {LOG_SAMPLE_RATE}")
    
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return True


# Validate on import
validate_config()


if __name__ == "__main__":
    print("=" * 80)
    print("Phase 6: Tuning Configuration")
    print("=" * 80)
    print(f"\nFUSION_WEIGHTS: {FUSION_WEIGHTS}")
    print(f"  Sum: {sum(FUSION_WEIGHTS.values()):.4f}")
    print(f"\nSUPERPOSITION_VARIANCE_THRESHOLD: {SUPERPOSITION_VARIANCE_THRESHOLD}")
    print(f"DOMAIN_SCORE_THRESHOLD: {DOMAIN_SCORE_THRESHOLD}")
    print(f"COVERAGE_THRESHOLD: {COVERAGE_THRESHOLD}")
    print(f"SOFT_STOP_PERCENTAGE: {SOFT_STOP_PERCENTAGE}")
    print(f"MAX_EXPERTS: {MAX_EXPERTS}")
    print(f"\nENABLE_ATTRIBUTE_OVERRIDE: {ENABLE_ATTRIBUTE_OVERRIDE}")
    print(f"ENABLE_LOGGING: {ENABLE_LOGGING}")
    print(f"LOG_SAMPLE_RATE: {LOG_SAMPLE_RATE}")
    print(f"\n✅ Configuration validated successfully")
    print("=" * 80)
