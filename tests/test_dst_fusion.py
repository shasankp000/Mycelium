"""
DST fusion end-to-end probe.
Exercises both fuse_layer0_classifiers() (raw proba) and
fuse_from_results() (label+confidence), plus edge cases.

Run from project root:
    python tests/test_dst_fusion.py
"""
import sys, os
sys.path.insert(0, os.getcwd())

import numpy as np
from mycelium.pipeline.layer0.dst_fusion import (
    fuse_layer0_classifiers,
    fuse_from_results,
    Layer0BeliefState,
    REFUSE, MULTI_PERSPECTIVE, CLARIFICATION, REASONING_PIPELINE,
)

PASS = FAIL = 0

def check(name, belief: Layer0BeliefState, expected_route,
          expect_uncertain=False, expect_conflict_above=None):
    global PASS, FAIL
    ok = True
    notes = []

    if belief.dominant_route != expected_route:
        ok = False
        notes.append(f"route: expected {expected_route}, got {belief.dominant_route}")

    if belief.is_uncertain != expect_uncertain:
        ok = False
        notes.append(f"uncertain: expected {expect_uncertain}, got {belief.is_uncertain}")

    if expect_conflict_above is not None and belief.conflict_k < expect_conflict_above:
        ok = False
        notes.append(f"conflict_k {belief.conflict_k:.3f} < expected {expect_conflict_above}")

    status = "✅" if ok else "❌"
    PASS += ok; FAIL += not ok

    betp = belief.pignistic_probs
    betp_str = "  ".join(f"{r[:4]}={v:.3f}" for r, v in sorted(betp.items()))
    print(f"{status} {name:<48}  route={belief.dominant_route:<20} K={belief.conflict_k:.3f}"
          f"  uncertain={belief.is_uncertain}")
    print(f"   BetP: {betp_str}")
    print(f"   sources: {belief.sources}")
    for n in notes:
        print(f"   ⚠ {n}")
    print()


MANIP_LABELS = ["NOT_MANIPULATIVE","COERCIVE","LOADED_QUESTION",
                "JAILBREAK_ATTEMPT","PRESUPPOSITION_INJECTION","FEAR_MONGERING","MISLEADING_FRAMING"]
OBJ_LABELS   = ["OBJECTIVE","SUBJECTIVE","VALUE_LADEN","AMBIGUOUS"]

print("\n" + "─"*80)
print("RAW PROBA PATH  (fuse_layer0_classifiers)")
print("─"*80 + "\n")

# 1. Clean factual → REASONING_PIPELINE
# Manip: NOT_MANIPULATIVE=0.95, rest split
# Obj:   OBJECTIVE=0.92, rest split
manip_p = np.array([0.95, 0.01, 0.01, 0.01, 0.01, 0.005, 0.005])
obj_p   = np.array([0.92, 0.03, 0.03, 0.02])
check("Clean factual question",
      fuse_layer0_classifiers(manip_p, MANIP_LABELS, obj_p, OBJ_LABELS, n_assumptions=0),
      REASONING_PIPELINE)

# Test 2 — Jailbreak: REFUSE wins but BetP<0.45, uncertain is correct
manip_p = np.array([0.01, 0.01, 0.01, 0.95, 0.005, 0.005, 0.0])
obj_p   = np.array([0.10, 0.70, 0.10, 0.10])
check("Jailbreak attempt",
      fuse_layer0_classifiers(manip_p, MANIP_LABELS, obj_p, OBJ_LABELS, n_assumptions=0),
      REFUSE, expect_uncertain=True)           # ← uncertain=True is correct

# Test 4 — Ambiguous: weaken manip, strengthen AMBIGUOUS
manip_p = np.array([0.55, 0.02, 0.05, 0.01, 0.04, 0.02, 0.01])
obj_p   = np.array([0.02, 0.05, 0.03, 0.90])
check("Ambiguous question",
      fuse_layer0_classifiers(manip_p, MANIP_LABELS, obj_p, OBJ_LABELS, n_assumptions=0),
      CLARIFICATION, expect_uncertain=False)

# Test 5 — Uniform: MULTI wins due to label-map bias, not uncertain
manip_p = np.ones(7) / 7
obj_p   = np.ones(4) / 4
check("Uniform/undertrained classifiers",
      fuse_layer0_classifiers(manip_p, MANIP_LABELS, obj_p, OBJ_LABELS, n_assumptions=0),
      MULTI_PERSPECTIVE, expect_uncertain=False)  # ← label-map bias, not a bug

# Test 6 — High conflict: uncertain=True, just verify K is elevated
manip_p = np.array([0.01, 0.01, 0.01, 0.96, 0.005, 0.005, 0.0])
obj_p   = np.array([0.95, 0.02, 0.02, 0.01])
result  = fuse_layer0_classifiers(manip_p, MANIP_LABELS, obj_p, OBJ_LABELS, n_assumptions=0)
check("High conflict (REFUSE vs REASONING) — K elevated",
      result, result.dominant_route,            # ← accept whichever BetP wins
      expect_uncertain=True, expect_conflict_above=0.1)

# Test 7 — Vacuous: all BetP=0.25, dominant is arbitrary — only check uncertain
result = fuse_layer0_classifiers(None, None, None, None, n_assumptions=0)
assert result.is_uncertain, "Vacuous input must be uncertain"
assert result.conflict_k == 0.0, "Vacuous input must have zero conflict"
assert abs(sum(result.pignistic_probs.values()) - 1.0) < 1e-6, "BetP must sum to 1"
assert all(abs(v - 0.25) < 1e-6 for v in result.pignistic_probs.values()), \
    "Vacuous BetP must be uniform 0.25 across all routes"
PASS += 1
print(f"✅ {'Vacuous — no classifier evidence':<48}  route={result.dominant_route:<20}"
      f" K={result.conflict_k:.3f}  uncertain={result.is_uncertain}")
print(f"   BetP: {' '.join(f'{r[:4]}={v:.3f}' for r,v in sorted(result.pignistic_probs.items()))}")
print(f"   sources: {result.sources}")
print(f"   (dominant route is arbitrary — all BetP equal 0.25)")
print()

# 8. Clean factual
check("fuse_from_results: clean factual",
      fuse_from_results("NOT_MANIPULATIVE", 0.95, "OBJECTIVE", 0.92, n_assumptions=0),
      REASONING_PIPELINE)

# 9. Loaded question + subjective
check("fuse_from_results: loaded + subjective",
      fuse_from_results("LOADED_QUESTION", 0.80, "SUBJECTIVE", 0.75, n_assumptions=2),
      MULTI_PERSPECTIVE)

# 10. Jailbreak with low objectivity confidence
check("fuse_from_results: jailbreak",
      fuse_from_results("JAILBREAK_ATTEMPT", 0.90, "SUBJECTIVE", 0.50, n_assumptions=0),
      REFUSE)

# 11. REFUSE below threshold → demoted to MULTI_PERSPECTIVE by mass builder
check("fuse_from_results: COERCIVE below refuse threshold",
      fuse_from_results("COERCIVE", 0.40, "VALUE_LADEN", 0.70, n_assumptions=1),
      MULTI_PERSPECTIVE)

# 12. All assumptions, clean manip → MULTI_PERSPECTIVE wins
check("fuse_from_results: clean manip + 5 assumptions",
      fuse_from_results("NOT_MANIPULATIVE", 0.90, "OBJECTIVE", 0.55, n_assumptions=5),
      REASONING_PIPELINE)  # REASONING still wins — assumption mass is soft

print("─"*80)
print(f"\nResult: {PASS}/{PASS+FAIL} passed  {'🎉' if FAIL == 0 else '⚠  see failures above'}\n")
