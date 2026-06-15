# tests/test_full_backend.py
# Tests the entire Mycelium backend pipeline end-to-end on a single input.
# Run from project root:  python tests/test_full_backend.py

# ── HF cache bootstrap ── must be the very first lines ──────────────────
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root → sys.path

_repo_root = Path(__file__).resolve().parents[1]
try:
    from mycelium.pipeline import config_loader as _cfg
    _hf_cache = _cfg.hf_cache_dir()
except Exception:
    _hf_cache = str(_repo_root / "hf_cache")

os.environ["HF_HOME"]                    = _hf_cache
os.environ["TRANSFORMERS_CACHE"]         = _hf_cache
os.environ["SENTENCE_TRANSFORMERS_HOME"] = _hf_cache
os.environ["HF_DATASETS_CACHE"]          = _hf_cache
# ── end bootstrap ────────────────────────────────────────────────────────

# all your normal imports go here

import json, pprint
sys.path.insert(0, os.getcwd())

import logging
logging.basicConfig(level=logging.WARNING)

from mycelium.pipeline.run_workflow import run_mycelium_workflow

# ─────────────────────────────────────────────────────────────────────────────
# Input under test
# ─────────────────────────────────────────────────────────────────────────────

QUERY = "Is 9.9 less than 9.11?"

# ─────────────────────────────────────────────────────────────────────────────
# SSE event collector (captures all pipeline events without a live server)
# ─────────────────────────────────────────────────────────────────────────────

sse_events = []

def _collect_event(event: dict):
    sse_events.append(event)

# ─────────────────────────────────────────────────────────────────────────────
# Run the full workflow
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*80)
print(f"  INPUT : {QUERY!r}")
print(f"  MODE  : smart")
print("="*80 + "\n")

results, metrics = run_mycelium_workflow(
    sentences=[QUERY],
    trace_id="test-full-backend-001",
    on_event=_collect_event,
    reasoning_mode="smart",
)

# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print structured result
# ─────────────────────────────────────────────────────────────────────────────

assert results, "run_mycelium_workflow returned an empty result list"
r = results[0]

SECTION = lambda t: print(f"\n{'─'*80}\n  {t}\n{'─'*80}")

SECTION("LAYER 0 ROUTING")
l0 = r.get("layer0_routing", {})
print(f"  route       : {l0.get('route', '?')}")
print(f"  confidence  : {l0.get('confidence', '?')}")
print(f"  dst_route   : {l0.get('dst_route', '?')}")
print(f"  dst_tags    : {l0.get('dst_tags', '?')}")

SECTION("ROUTING CONTEXT (MultiLens + TRM Lens)")
rc = r.get("routing_context", {})
print(f"  classification    : {rc.get('classification', '?')}")
print(f"  selected_domains  : {rc.get('selected_domains', '?')}")
print(f"  primary_domain    : {rc.get('primary_domain', '?')}")
fusion = rc.get("fusion_scores", {})
if fusion:
    top = sorted(fusion.items(), key=lambda x: -x[1])[:5]
    print(f"  fusion_scores     : {top}")

SECTION("TRM REASONER")
trm = r.get("trm_reasoner_result", {})
if trm:
    print(f"  primary_domain    : {trm.get('primary_domain', '?')}")
    print(f"  halt_confidence   : {trm.get('halt_confidence', '?')}")
    print(f"  n_steps_taken     : {trm.get('n_steps_taken', '?')}")
    print(f"  reranked_domains  : {trm.get('reranked_domains', [])[:5]}")
    print(f"  should_escalate   : {r.get('should_escalate', '?')}")
else:
    print("  TRMReasoner not active (pre-TRM mode)")

SECTION("OOD FALLBACK")
ood = r.get("ood_fallback_result", {})
if ood:
    print(f"  triggered         : {ood.get('triggered', '?')}")
    print(f"  level             : {ood.get('level', '?')}")
    print(f"  selected_domain   : {ood.get('selected_domain', '?')}")
    print(f"  ood_confidence    : {ood.get('ood_confidence', '?')}")
    print(f"  reason            : {ood.get('reason', '?')}")
    print(f"  pre_trm_mode      : {ood.get('pre_trm_mode', False)}")
else:
    print("  (not triggered)")

SECTION("SHADOW DOMAIN SIGNAL")
shadow = r.get("shadow_signal")
if shadow:
    print(f"  status            : {shadow.get('status', '?')}")
    print(f"  shadow_id         : {shadow.get('shadow_id', '?')}")
    print(f"  evidence          : {shadow.get('evidence', '?')}")
    print(f"  top_tokens        : {shadow.get('top_tokens', [])[:6]}")
else:
    print("  (no shadow signal)")

SECTION("PREDICATE STORE  [Stage 9A]")
ps = r.get("predicate_store")
if ps:
    print(f"  total             : {ps.get('total', '?')}")
    print(f"  falsifiable       : {ps.get('falsifiable', '?')}")
else:
    print("  (PredicatePipeline not available or no frames extracted)")

SECTION("EVIDENCE RETRIEVAL  [Stage 9B]")
ev = r.get("evidence_result")
if ev:
    print(f"  bundles           : {ev.get('bundles', '?')}")
    print(f"  total_items       : {ev.get('total_items', '?')}")
else:
    print("  (EvidenceFinder not available or skipped)")

SECTION("TOOL PLAN  [Stage 9B — DomainToolPlanner]")
tp = r.get("tool_plan")
if tp:
    print(f"  selected_tools    : {tp.get('selected_tools', '?')}")
    print(f"  routing_reason    : {tp.get('routing_reason', '?')}")
    print(f"  predicate_types   : {tp.get('predicate_types', '?')}")
    stubs = tp.get("stub_resolutions", [])
    if stubs:
        print(f"  stub_resolutions  : {stubs}")
else:
    print("  (DomainToolPlanner not active or skipped)")

SECTION("EVIDENCE SCORING  [Stage 9C]")
se = r.get("scored_evidence")
if se:
    print(f"  weighted_conf     : {se.get('weighted_confidence', '?')}")
    print(f"  scored_bundles    : {se.get('scored_bundles', '?')}")
else:
    print("  (EvidenceScorer not available or skipped)")

SECTION("CONTRADICTION CHECK  [Stage 9D]")
cc = r.get("contradiction_result")
if cc:
    print(f"  pairs_evaluated   : {cc.get('pairs_evaluated', '?')}")
    print(f"  worst_type        : {cc.get('worst_type', '?')}")
    print(f"  worst_severity    : {cc.get('worst_severity', '?')}")
    print(f"  worst_scope       : {cc.get('worst_scope', '?')}")
else:
    print("  (ContradictionClassifier not active or < 2 frames)")

SECTION("DST BELIEF FUSION  [Stage 10]")
dst = r.get("evidence_dst")
if dst:
    print(f"  net_confidence    : {dst.get('net_confidence', '?')}")
    print(f"  m_true            : {dst.get('m_true', '?')}")
    print(f"  m_unknown         : {dst.get('m_unknown', '?')}")
    print(f"  is_uncertain      : {dst.get('is_genuinely_uncertain', '?')}")
else:
    print("  (EvidenceDSTAdapter not available or skipped)")

SECTION("PHASE 2 — Expert Calibration")
p2 = r.get("phase2_result", {})
print(f"  decision_label    : {p2.get('decision_label', '?')}")
print(f"  confidence        : {p2.get('confidence', '?')}")
print(f"  selected_expert   : {p2.get('selected_expert', '?')}")
print(f"  domain            : {p2.get('domain', '?')}")

SECTION("PHASE 3 — Reasoning Pipeline")
p3 = r.get("phase3_result", {})
if isinstance(p3, dict):
    print(f"  decision          : {p3.get('decision', '?')}")
    print(f"  action            : {p3.get('action', '?')}")
    print(f"  confidence        : {p3.get('confidence', '?')}")
    print(f"  expert_name       : {p3.get('expert_name', '?')}")
    reasoning = p3.get("reasoning", "")
    if reasoning:
        print(f"  reasoning         : {str(reasoning)[:200]}")
else:
    print(f"  (raw) {str(p3)[:300]}")

SECTION("UNIFIED EXPERT DECISION")
ued = r.get("expert_decision", {})
print(f"  decision_type     : {ued.get('decision_type', '?')}")
print(f"  selected_experts  : {ued.get('selected_experts', '?')}")
print(f"  expert_confidence : {ued.get('expert_confidence', '?')}")
print(f"  reasoning         : {str(ued.get('reasoning', ''))[:200]}")

SECTION("FINAL SUMMARY")
print(f"  selected_domain   : {r.get('selected_domain', '?')}")
print(f"  expert_flag       : {r.get('expert_flag', '?')}")
print(f"  decision_conf     : {r.get('decision_confidence', '?')}")
print(f"  tags              : {r.get('tags', [])}")
print(f"  timestamp         : {r.get('timestamp', '?')}")

SECTION("WORKFLOW METRICS")
print(f"  layer0_routes     : {dict(metrics.layer0_routes)}")
print(f"  routing_clf       : {dict(metrics.routing_classifications)}")
print(f"  expert_decisions  : {dict(metrics.expert_decisions)}")
print(f"  domains           : {dict(metrics.domains)}")

SECTION("SSE EVENT TRACE")
print(f"  total events fired: {len(sse_events)}")
phases_seen = [e.get("phase_name") for e in sse_events if e.get("phase_name")]
# deduplicate while preserving order
seen_set = set()
phases_ordered = []
for p in phases_seen:
    if p not in seen_set:
        phases_ordered.append(p)
        seen_set.add(p)
print(f"  phases (in order) : {phases_ordered}")

print("\n" + "="*80)
print("  ✅ Full backend test complete — no exceptions raised")
print("="*80 + "\n")
