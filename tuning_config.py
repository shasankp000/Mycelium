"""
Phase 6: Tuning Configuration

All parameters are now read from ``config.toml`` via ``config_loader``.
Edit ``config.toml`` to change any value – no restart needed.

This file re-exports the same module-level names that the rest of the
codebase imports (FUSION_WEIGHTS, ENABLE_LOGGING, etc.) so that no
other file needs to change.
"""

from config_loader import cfg

# ============================================================================
# FUSION WEIGHTS  (dict so existing consumers don't break)
# ============================================================================
FUSION_WEIGHTS = cfg.fusion_weights()

# ============================================================================
# THRESHOLDS
# ============================================================================
SUPERPOSITION_VARIANCE_THRESHOLD = cfg.superposition_variance_threshold()
DOMAIN_SCORE_THRESHOLD           = cfg.domain_score_threshold()
COVERAGE_THRESHOLD               = cfg.coverage_threshold()
SOFT_STOP_PERCENTAGE             = cfg.soft_stop_percentage()
MAX_EXPERTS                      = cfg.max_experts()

# ============================================================================
# FLAGS
# ============================================================================
ENABLE_ATTRIBUTE_OVERRIDE = cfg.enable_attribute_override()
ENABLE_LOGGING            = cfg.enable_logging()
LOG_SAMPLE_RATE           = cfg.log_sample_rate()


# ============================================================================
# VALIDATION  (preserved – still runs on import)
# ============================================================================
def validate_config():
    """Validate that configuration is internally consistent."""
    errors = []

    weight_sum = sum(FUSION_WEIGHTS.values())
    if not (0.99 <= weight_sum <= 1.01):
        errors.append(f"Fusion weights must sum to 1.0, got {weight_sum}")

    for key, value in FUSION_WEIGHTS.items():
        if value < 0 or value > 1:
            errors.append(f"Fusion weight '{key}' must be in [0,1], got {value}")

    if not (0 <= SUPERPOSITION_VARIANCE_THRESHOLD <= 1):
        errors.append(
            f"SUPERPOSITION_VARIANCE_THRESHOLD must be in [0,1], "
            f"got {SUPERPOSITION_VARIANCE_THRESHOLD}"
        )

    if not (0 <= DOMAIN_SCORE_THRESHOLD <= 1):
        errors.append(
            f"DOMAIN_SCORE_THRESHOLD must be in [0,1], got {DOMAIN_SCORE_THRESHOLD}"
        )

    if not (0 <= COVERAGE_THRESHOLD <= 1):
        errors.append(
            f"COVERAGE_THRESHOLD must be in [0,1], got {COVERAGE_THRESHOLD}"
        )

    if not (0 <= SOFT_STOP_PERCENTAGE <= 1):
        errors.append(
            f"SOFT_STOP_PERCENTAGE must be in [0,1], got {SOFT_STOP_PERCENTAGE}"
        )

    if MAX_EXPERTS < 1:
        errors.append(f"MAX_EXPERTS must be >= 1, got {MAX_EXPERTS}")

    if LOG_SAMPLE_RATE < 1:
        errors.append(f"LOG_SAMPLE_RATE must be >= 1, got {LOG_SAMPLE_RATE}")

    if errors:
        raise ValueError(
            "Configuration validation failed:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    return True


# Validate on import
validate_config()


if __name__ == "__main__":
    print("=" * 80)
    print("Phase 6: Tuning Configuration  (backed by config.toml)")
    print("=" * 80)
    print(f"\nFUSION_WEIGHTS: {FUSION_WEIGHTS}")
    print(f"  Sum: {sum(FUSION_WEIGHTS.values()):.4f}")
    print(f"\nSUPERPOSITION_VARIANCE_THRESHOLD : {SUPERPOSITION_VARIANCE_THRESHOLD}")
    print(f"DOMAIN_SCORE_THRESHOLD           : {DOMAIN_SCORE_THRESHOLD}")
    print(f"COVERAGE_THRESHOLD               : {COVERAGE_THRESHOLD}")
    print(f"SOFT_STOP_PERCENTAGE             : {SOFT_STOP_PERCENTAGE}")
    print(f"MAX_EXPERTS                      : {MAX_EXPERTS}")
    print(f"\nENABLE_ATTRIBUTE_OVERRIDE        : {ENABLE_ATTRIBUTE_OVERRIDE}")
    print(f"ENABLE_LOGGING                   : {ENABLE_LOGGING}")
    print(f"LOG_SAMPLE_RATE                  : {LOG_SAMPLE_RATE}")
    print(f"\n✅ Configuration validated successfully")
    print("=" * 80)
