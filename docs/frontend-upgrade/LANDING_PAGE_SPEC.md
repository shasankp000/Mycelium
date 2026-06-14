# Mycelium Landing Page Spec — `web-ui-prototype`

**Status:** Design-ready  
**Branch:** `web-ui-prototype`  
**Depends on:** `UI_POLISH_SPEC.md` (token system, theme engine)  
**Author:** Drafted from architecture docs, TRM v2 spec, Predicate Engine spec, and Semantic Architecture consolidation notes

---

## 1. What Mycelium Actually Is

Mycelium is not a chatbot. It is a **semantic reasoning runtime** — a system purpose-built to distinguish between what is objectively true, what depends on values, and what is an attempt at manipulation, then reason accordingly.

The system evolves around three interlocking commitments:

1. **Truth-seeking over answer-giving.** For objective questions, Mycelium pursues a definitive, evidence-grounded answer. It does not hedge to seem humble. For value-laden questions, it surfaces the full option space rather than masking a preference as fact.

2. **Epistemic honesty over confident-sounding output.** Mycelium explicitly distinguishes IS questions (verifiable claims) from OUGHT questions (value judgments). Where mainstream LLMs blend these silently, Mycelium surfaces the boundary. Bias enters when values are presented as facts — Mycelium's architecture prevents this structurally.

3. **Non-destructive, modular growth.** Inspired by biological mycelial networks, the system is built around a frozen shared backbone, isolated domain heads, and a cold-storage lifecycle. New knowledge is added without overwriting prior knowledge. This is not a feature — it is a hard architectural constraint.

### The 6-Layer Reasoning Pipeline

Every non-trivial query flows through:

L0 — Question Router (Gatekeeper)
↓ objective → L1–L6
↓ value-laden → Multi-perspective evidence engine
↓ manipulation → Refuse + explain + suggest honest reframe

L1 — Contradiction Analyzer
L2 — Claim Decomposer
L3 — Consequence Generator
L4 — Evidence Grounding (refutation-first retrieval)
L5 — Hypothesis Evaluator (DST Fusion)
L6 — Reasoning Synthesizer


The L0 Gatekeeper alone does what most systems skip entirely: it detects manipulation attempts (instruction override, conclusion-forcing, cherry-pick requests), classifies objective vs. value-laden questions, and extracts hidden value assumptions before any reasoning begins.

### The Semantic Substrate (Predicate Engine)

Beneath the 6-layer pipeline sits a structured semantic IR layer. Instead of reasoning over raw text, Mycelium converts claims into **PredicateFrames** — structured semantic objects with subject, relation, object, negation, quantifier scope, falsifiability flag, refutation burden, and full provenance lineage. This transforms Mycelium from an LLM orchestration framework into a genuine semantic reasoning runtime.

### TRM — The Memory Layer

The Temporal Reasoning Module (TRM v2) manages domain knowledge as a graph. Domains have formal lifecycle states (CREATING → HOT → WARM → COLD → REMEMBERING → DEPRECATED → ARCHIVED). Knowledge is never hard-deleted. The gate that routes queries to domain experts can be frozen, thawed, and recalibrated without global retraining. This is the anti-catastrophic-forgetting guarantee made architectural rather than heuristic.

---

## 2. Philosophy — The "No Bullshit" Commitment

The phrase originates from the earliest architecture documents and is the honest articulation of what Mycelium is for:

> *"The no bullshit promise: I will find truth where it exists. I will admit when it doesn't. I will not pretend my preferences are your facts. I will not be manipulated into bias."*

This is not a marketing claim. It is a design constraint that propagates into L0 (manipulation detection), L4 (refutation-first evidence retrieval), and the Predicate Engine (NORMATIVE predicates are structurally excluded from contradiction trees, preventing value judgments from being processed as falsifiable claims).

The philosophical foundation is Hume's Is-Ought distinction (1739): you cannot derive what ought to be from what is. Mycelium operationalises this philosophically motivated boundary as an architectural boundary.

---

## 3. The Three Reasoning Modes

The current UI exposes three modes, each corresponding to a different operating posture:

| Mode | Pipeline depth | Use case |
|------|---------------|----------|
| **Quick** | L0 + shallow L1–L2 | Fast factual lookup, low-stakes queries |
| **Smart** | Full L0–L6, bounded evidence budget | General reasoning, medium complexity |
| **Researcher** | Full L0–L6, expanded evidence budget, multi-source grounding | Deep investigation, contested claims, academic-grade rigor |

---

## 4. Landing Page Content Architecture

The landing page has one job: communicate what Mycelium is, why it is different, and get a serious user into the reasoning interface. It does not need to be a marketing page. It needs to be honest.

### 4.1 Hero Section

**Headline options** (pick one per theme — the Arc theme's gradient wordmark makes this the most expressive element):

- `Mycelium` — standalone, the name carries the weight
- `Reasoning that doesn't lie to you` — direct, confrontational in the right way
- `Truth-seeking, not answer-giving` — captures the epistemic philosophy

**Subheadline** (the one sentence that explains what this is):

> Multi-path semantic reasoning. Built to find truth where it exists, and admit when it doesn't.

**Status pill:** `Research Preview` (current) — keep as-is, honest about the state of the project.

**CTA:** `Start reasoning →` — already implemented. Keep.

**Mode pills:** Quick · Smart · Researcher — already implemented. Keep.

### 4.2 "What makes this different" Section

Three focused callouts. These should NOT be the standard AI feature grid (icon in circle + 2-line description × 3). Each callout should be a short, direct claim followed by one concrete example or mechanism.

---

**Callout 1 — The Gatekeeper**

> Before answering anything, Mycelium asks: *can this question have an objective answer?*

Objective questions enter the full 6-layer reasoning pipeline and receive a definitive, evidence-grounded response. Value-laden questions receive multi-perspective evidence presentation — no hidden opinion. Manipulation attempts are refused and explained.

*Most AI systems skip this step entirely. Mycelium makes it the foundation.*

---

**Callout 2 — Refutation-First Evidence**

> Evidence retrieval is designed to disprove the claim first, then confirm it.

This is called refutation-first retrieval: the system actively searches for counterevidence before searching for supporting evidence. For universal claims, a single strong counterexample halts retrieval immediately. This structurally prevents confirmation bias at the retrieval layer.

*Confidence scores mean something here — they reflect what the evidence actually shows.*

---

**Callout 3 — Semantic Reasoning, Not Text Autocomplete**

> Mycelium reasons over structured semantic claims, not raw text.

Every claim is decomposed into a PredicateFrame: subject, relation, object, negation, scope, falsifiability, and provenance. Reasoning propagates through a DAG of structured predicates, not a sequence of token predictions. The full reasoning graph is visible in the UI.

*You can see exactly what the system concluded, what evidence it used, and what it considered but rejected.*

---

### 4.3 Reasoning Graph Preview (Optional, Phase 6+)

If a static screenshot or animated demo of the reasoning graph overlay is available, include it here as a full-width visual between the callouts section and the CTA section. Caption: *"Every conclusion is a traversable graph. Every edge is a reasoning step."*

### 4.4 The Philosophy Statement (Optional "About" anchor)

A short block, visually distinct (slightly inset, muted text, no box shadow), anchored at `#philosophy`:

> **On epistemic honesty**
>
> David Hume observed in 1739 that you cannot derive *ought* from *is* — that facts and values are fundamentally different categories. Most AI systems blur this line constantly, presenting value judgments as factual outputs.
>
> Mycelium treats this boundary as an architectural constraint. Value-laden questions are routed to a multi-perspective evidence engine that presents the full option space with evidence for each position. The system does not have opinions on questions that require value judgments. It has evidence, and it shows it to you.

### 4.5 Footer

Already implemented: `v{version} · web-ui-prototype · MIT`. Add:
- Link to GitHub repo
- Link to `#philosophy` anchor
- Optional: `docs/` link once public docs exist

---

## 5. Visual Direction per Theme

The landing page spec inherits all theme-specific overrides already defined in `globals.css`. The following content direction complements the visual treatment:

| Theme | Tone | Headline style | Background treatment |
|-------|------|---------------|----------------------|
| **AMOLED** | Precise, minimal, high-contrast | Plain white, no gradient | Pure black, teal glow on CTA hover |
| **Anthropic** | Warm, intellectual, humanist | Off-white, slightly warm | Warm radial behind hero |
| **Apple** | Clean, confident, premium | Bright white, blue accent on hover | Subtle blue radial, vibrancy on pills |
| **Arc** | Expressive, architectural, purple | Gradient wordmark (purple → white) | Purple radial glow |

The headline should never be centered on the callouts section — left-align callout text. Center only the hero headline and subheadline.

---

## 6. Copy Principles

- **No filler phrases.** "Empowering your reasoning journey" is banned. Every sentence must be load-bearing.
- **Concrete over abstract.** "Detects manipulation attempts including instruction override, conclusion-forcing, and cherry-pick requests" is better than "Maintains integrity."
- **Honest about limitations.** This is a research preview. The landing page should not imply production-readiness.
- **First person is the system, not the company.** "Mycelium finds truth where it exists" not "We built a truth-seeking AI."

---

## 7. Implementation Notes

### Component structure

````
LandingPage.tsx
├── <NavBar /> — wordmark + ThemeSwitcher + GitHub link (already built)
├── <HeroSection /> — headline, subheadline, status pill, CTA, mode pills
├── <DifferentiatorSection /> — three callouts (new)
├── <PhilosophyBlock /> — optional #philosophy anchor (new)
└── <Footer /> — version + links (already built)
```

### New CSS classes needed in `LandingPage.module.css`

```css
/* Differentiator section */
.differentiators        — grid, 3 columns at md, 1 column at sm
.differentiatorItem     — no border, no icon circle, left-aligned
.differentiatorEyebrow  — small all-caps label, --c-accent-base, --text-xs
.differentiatorHeading  — --text-lg, --c-text-primary, font-weight 600
.differentiatorBody     — --text-base, --c-text-muted, max-width 42ch
.differentiatorNote     — italic, --text-sm, --c-text-faint, margin-top auto

/* Philosophy block */
.philosophyBlock        — max-width 640px, centered, --c-bg-1 background, --radius-lg
.philosophyHeading      — --text-md, --c-text-faint, font-weight 500
.philosophyBody         — --text-base, --c-text-muted, line-height 1.75
```

### Anti-patterns to avoid

- No icons in colored circles for the three callouts
- No card borders on differentiator items — use spacing and typography for separation
- No gradient buttons (the CTA is already a solid accent pill — keep it)
- Do not center the differentiator text — left-align creates reading gravity

---

