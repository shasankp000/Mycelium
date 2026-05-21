# PHASE 1: Spectral Analysis Core

## Overview

**Phase 1** implements a standalone, non-breaking **spectral analysis module** for Layer 1 routing.

This phase learns the frequency-domain "signature" of each domain by analyzing embeddings, enabling deterministic structural comparison at runtime.

### Key Principle

> Semantic similarity alone doesn't capture structural patterns. By analyzing embedding distributions in frequency space, we gain a complementary signal that's independent of semantic content.

---

## Problem This Solves

### Current Limitation

Layer 1's existing Lens 1-2 uses:
- **Lens 1**: Semantic similarity (embeddings + fallback)
- **Lens 2**: Ontology-based explanation

Both are **semantic lenses**—they measure meaning similarity.

### Gap

A text can be structurally similar to a domain's corpus even if semantically different.

### What Phase 1 Adds

A third, **structural lens** that:
- Analyzes embedding distributions in frequency space
- Learns "how each domain looks" as a pattern
- Compares inputs against these patterns deterministically
- Returns domain similarity scores

---

> **Archived from root** during cleanup pass (2026-05-21). Original file: `PHASE_1_SPECTRAL_ANALYSIS.md`.
