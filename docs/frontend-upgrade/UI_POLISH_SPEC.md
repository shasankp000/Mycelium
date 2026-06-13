# Mycelium UI Polish Spec — TRM v0.2 Visual Layer
**Document:** `docs/frontend-upgrade/UI_POLISH_SPEC.md`  
**Branch target:** `web-ui-prototype`  
**Scope:** Cosmetic / UI polish only — zero functional changes. No new data flows, no new hooks, no changes to SSE routing. Typography, colour tokens, spacing, component micro-interactions, and theme system only.  
**Status:** Pre-implementation draft — approve before any CSS edits land.

---

## 1. Motivation

The current stylesheet (`Home.module.css`, `ReasoningGraph.module.css`, `ChatBubble.module.css`, `ModeSelector.module.css`, `globals.css`) is a single hardcoded AMOLED dark palette with Inter at system fallback weight. Before TRM v0.2 adds new visual surface area (DFS step nodes, TRM decision chips, evidence fusion rows, contradiction callouts), the design token system needs to be:

1. **Themeable** — palette switching without CSS rewrite.  
2. **Systematised** — every hardcoded hex replaced by a CSS custom property.  
3. **Typographically upgraded** — loaded font instead of system fallback stack.  
4. **Micro-interaction polished** — transitions, focus rings, live-state animations consistent across every component.

---

## 2. Design Token Architecture

All tokens live in `globals.css` under `[data-theme="…"]` attribute selectors. No theme-specific classes inside component modules. Component CSS reads `var(--token-name)` exclusively — no bare hex, no bare `rgb()`, no hardcoded `opacity` values for colour.

### 2.1 Token Naming Convention

```
--c-bg-{level}        surface hierarchy (base → 1 → 2 → offset)
--c-text-{weight}     text hierarchy   (primary → muted → faint → inverse)
--c-accent-{variant}  primary accent   (base → hover → active → subtle)
--c-ok-{variant}      success/ok       (base → subtle)
--c-warn-{variant}    warning/amber    (base → subtle)
--c-err-{variant}     error/red        (base → subtle)
--c-border-{weight}   borders          (hairline → default → strong)
--c-overlay           modal/backdrop fill
--radius-{size}       border-radius tokens
--dur-{speed}         transition duration tokens
--ease-spring         standard easing curve
```

### 2.2 Font Tokens

```css
--font-body:    'Geist', 'Inter', system-ui, sans-serif;
--font-mono:    'Geist Mono', 'Fira Code', 'SF Mono', monospace;
--font-display: 'Geist', 'Inter', system-ui, sans-serif;
```

Load via `next/font/google` or local `/public/fonts`. **Do not keep the bare `@import url(...)` in `globals.css`** — it blocks render. Move to `_document.tsx` `<link rel="preconnect">` + `<link rel="stylesheet">` with `font-display: swap`.

---

## 3. Themes

Four themes ship with the spec. Each is a complete set of the tokens from §2.1 applied to `[data-theme="<id>"]`. The theme toggle persists to `localStorage` under key `"mycelium-theme"`.

---

### Theme A — `amoled` (Default · Current)

The existing palette, systematised. No visual change, purely token extraction.

```css
[data-theme="amoled"] {
  /* Surfaces */
  --c-bg-base:    #000000;
  --c-bg-1:       #080808;
  --c-bg-2:       #0d0d0d;
  --c-bg-offset:  #141414;

  /* Text */
  --c-text-primary: #e8e8e8;
  --c-text-muted:   #a0a0a0;
  --c-text-faint:   #505050;
  --c-text-inverse: #000000;

  /* Accent — Hydra Teal */
  --c-accent-base:   #4f98a3;
  --c-accent-hover:  #227f8b;
  --c-accent-active: #0e2a2e;
  --c-accent-subtle: #061618;

  /* Status */
  --c-ok-base:    #6daa45;
  --c-ok-subtle:  #1a2e18;
  --c-warn-base:  #fdab43;
  --c-warn-subtle:#2e1a08;
  --c-err-base:   #dd6974;
  --c-err-subtle: #1a0808;

  /* Borders */
  --c-border-hairline: #1a1a1a;
  --c-border-default:  #242424;
  --c-border-strong:   #2e2e2e;

  /* Overlay */
  --c-overlay: rgba(0,0,0,0.65);

  /* Radius */
  --radius-sm:   4px;
  --radius-md:   8px;
  --radius-lg:   12px;
  --radius-pill: 99px;

  /* Motion */
  --dur-fast:   120ms;
  --dur-normal: 200ms;
  --dur-slow:   400ms;
  --ease-spring: cubic-bezier(0.16, 1, 0.3, 1);
}
```

---

### Theme B — `anthropic` (Open-Design Primary)

Anthropic's visual language: warm off-white surfaces, sand/cream neutrals, terracotta/coral accent, slightly warm blacks. References Claude.ai's web interface.

```css
[data-theme="anthropic"] {
  /* Surfaces — warm paper */
  --c-bg-base:    #1a1915;   /* warm near-black, not pure */
  --c-bg-1:       #1f1e1a;
  --c-bg-2:       #252320;
  --c-bg-offset:  #2c2a26;

  /* Text — warm whites */
  --c-text-primary: #f0ede6;   /* cream-white, not cold */
  --c-text-muted:   #9e9b93;
  --c-text-faint:   #5a5750;
  --c-text-inverse: #1a1915;

  /* Accent — Anthropic Terracotta/Coral */
  --c-accent-base:   #d97757;   /* the signature Claude orange-coral */
  --c-accent-hover:  #c4623f;
  --c-accent-active: #3a1c10;
  --c-accent-subtle: #2a1508;

  /* Status — warm variants */
  --c-ok-base:    #7cb87a;
  --c-ok-subtle:  #1e2e1c;
  --c-warn-base:  #e8a84e;
  --c-warn-subtle:#2e2010;
  --c-err-base:   #e0756f;
  --c-err-subtle: #2e1010;

  /* Borders — warm stone */
  --c-border-hairline: #2a2822;
  --c-border-default:  #343129;
  --c-border-strong:   #403d34;

  --c-overlay: rgba(10,9,7,0.70);
  --radius-sm:   5px;
  --radius-md:   10px;
  --radius-lg:   14px;
  --radius-pill: 99px;
  --dur-fast:   120ms;
  --dur-normal: 200ms;
  --dur-slow:   400ms;
  --ease-spring: cubic-bezier(0.16, 1, 0.3, 1);
}
```

**Font pairing for `anthropic` theme:**  
Switch `--font-body` to `'Tiempos Text', 'Georgia', serif` for prose bubbles; keep `--font-display` as `'Söhne', 'Inter', sans-serif` for UI chrome. If web font licensing is unavailable, fall back to `Georgia` + `Inter`.

**Signature treatments:**
- User bubble background: `linear-gradient(135deg, #2a1e16, #241b14)` — deep terracotta tint.
- `--c-accent-base` (#d97757) on the send button, focus rings, and phase bar fill.
- Calibration card background: `#1f1e1a` with `1px solid #2e2b23` border.
- History item active `border-left-color`: `#d97757`.

---

### Theme C — `apple` (macOS Continuity)

SF Pro–adjacent system feel. True neutral-gray surfaces, Apple's electric blue accent, pill-shaped interactive elements. References macOS Sonoma sidebar + Notes.app.

```css
[data-theme="apple"] {
  /* Surfaces — true neutral dark */
  --c-bg-base:    #161616;
  --c-bg-1:       #1c1c1e;   /* iOS/macOS system grouped background */
  --c-bg-2:       #2c2c2e;   /* secondary grouped */
  --c-bg-offset:  #3a3a3c;   /* tertiary */

  /* Text */
  --c-text-primary: #f5f5f7;   /* Apple's headline white */
  --c-text-muted:   #aeaeb2;   /* systemGray2 */
  --c-text-faint:   #636366;   /* systemGray */
  --c-text-inverse: #000000;

  /* Accent — Apple System Blue */
  --c-accent-base:   #0a84ff;   /* iOS dark mode blue */
  --c-accent-hover:  #0070df;
  --c-accent-active: #00163a;
  --c-accent-subtle: #001228;

  /* Status */
  --c-ok-base:    #30d158;   /* Apple systemGreen dark */
  --c-ok-subtle:  #0c2e16;
  --c-warn-base:  #ffd60a;   /* Apple systemYellow */
  --c-warn-subtle:#2a2300;
  --c-err-base:   #ff453a;   /* Apple systemRed dark */
  --c-err-subtle: #2a0a08;

  /* Borders — iOS separator */
  --c-border-hairline: #2c2c2e;
  --c-border-default:  #3a3a3c;
  --c-border-strong:   #48484a;

  --c-overlay: rgba(0,0,0,0.55);
  --radius-sm:   6px;
  --radius-md:   10px;    /* Apple's signature 10px */
  --radius-lg:   14px;
  --radius-pill: 99px;
  --dur-fast:   100ms;
  --dur-normal: 180ms;
  --dur-slow:   360ms;
  --ease-spring: cubic-bezier(0.25, 0.46, 0.45, 0.94);   /* Apple's standard easing */
}
```

**Font pairing for `apple` theme:**  
`--font-body: 'SF Pro Text', -apple-system, 'Helvetica Neue', sans-serif;`  
`--font-mono: 'SF Mono', 'Fira Code', monospace;`  
(SF Pro is loaded automatically on Apple devices via `-apple-system`; non-Apple devices gracefully fall back to Helvetica Neue.)

**Signature treatments:**
- All interactive elements use `--radius-pill` instead of `--radius-md` where possible (buttons, badges, input focus ring).
- Blur backgrounds on modal overlays: `backdrop-filter: blur(20px) saturate(180%)` on sidebar + graph overlay panels.
- Phase bar fill: `linear-gradient(90deg, #0a84ff, #30d158)` — matches Apple's iOS progress colour.
- History item hover: `background: rgba(255,255,255,0.05)` — no hard colour jump.

---

### Theme D — `arc` (Arc Browser)

Arc's signature: deep purple-indigo space, gradient UI chrome, Boost-style pastel accent spots, and a strong sidebar visual weight. References Arc's command bar and Spaces.

```css
[data-theme="arc"] {
  /* Surfaces — deep purple-slate */
  --c-bg-base:    #0e0c1a;
  --c-bg-1:       #13101f;
  --c-bg-2:       #1a1628;
  --c-bg-offset:  #221e30;

  /* Text */
  --c-text-primary: #e8e4f4;
  --c-text-muted:   #9b94b8;
  --c-text-faint:   #524e6a;
  --c-text-inverse: #0e0c1a;

  /* Accent — Arc Electric Purple */
  --c-accent-base:   #9b7ff4;   /* Arc's signature violet */
  --c-accent-hover:  #7c5fe8;
  --c-accent-active: #1e1440;
  --c-accent-subtle: #140e2e;

  /* Status — pastel Arc-style */
  --c-ok-base:    #74c99a;
  --c-ok-subtle:  #102818;
  --c-warn-base:  #f4c57f;
  --c-warn-subtle:#2a1e08;
  --c-err-base:   #f47f88;
  --c-err-subtle: #2a1018;

  /* Borders — purple-tinted */
  --c-border-hairline: #1e1a2e;
  --c-border-default:  #28223e;
  --c-border-strong:   #342e4e;

  --c-overlay: rgba(5,4,14,0.72);
  --radius-sm:   5px;
  --radius-md:   10px;
  --radius-lg:   16px;   /* Arc uses generous radius */
  --radius-pill: 99px;
  --dur-fast:   140ms;
  --dur-normal: 220ms;
  --dur-slow:   460ms;
  --ease-spring: cubic-bezier(0.34, 1.56, 0.64, 1);   /* slight overshoot — Arc-like bounce */
}
```

**Font pairing for `arc` theme:**  
`--font-body: 'Geist', 'Inter', system-ui, sans-serif;`  
`--font-display: 'Geist', system-ui, sans-serif;` — keeps clean utility feel.

**Signature treatments:**
- History sidebar background: `linear-gradient(180deg, #0e0c1a 0%, #13101f 100%)` with a `box-shadow: inset -1px 0 0 #1e1a2e` right edge instead of a border.
- Header: subtle `background: linear-gradient(90deg, #0e0c1a, #13101f)` + glass shimmer on hover via `:hover` overlay `::after`.
- Send button: `background: linear-gradient(135deg, #9b7ff4, #7c5fe8)` — Arc's gradient pill.
- Phase bar fill: `linear-gradient(90deg, #9b7ff4 0%, #74c99a 100%)`.
- Graph overlay backdrop: `backdrop-filter: blur(24px)` with `--c-bg-base` at 85% opacity.
- Mode selector active pill: `background: linear-gradient(135deg, #9b7ff4 0%, #5e8cf4 100%)`.

---

## 4. Component-Level Polish Specs

### 4.1 `globals.css` Changes

```diff
- @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
+ /* Font loading moved to _document.tsx as <link> tags with rel="preconnect" */
+ /* globals.css defines only the CSS variable and the base reset */

html, body {
-  background: #0f0f0f;
-  color: #e2e2e2;
+  background: var(--c-bg-base);
+  color: var(--c-text-primary);
+  font-family: var(--font-body);
}
```

Add a `[data-theme="amoled"]` block as the **default** (applied by the theme initialisation script below). All other theme blocks follow.

**Theme initialisation script** (inline in `<head>`, before any CSS paint):
```html
<script>
  (function(){
    var t = localStorage.getItem('mycelium-theme') || 'amoled';
    document.documentElement.setAttribute('data-theme', t);
  })();
</script>
```

This prevents flash-of-wrong-theme (FOWT) without a React render cycle.

---

### 4.2 Theme Switcher Component (`ThemeSwitcher.tsx`)

New component, placed in the header right-side slot (before the graph toggle button).

**Visual:** A single icon button (palette icon, 20px, Lucide `Palette`) that opens a small popover (4 swatches in a row). Clicking a swatch:
1. Sets `document.documentElement.setAttribute('data-theme', id)`.
2. Persists to `localStorage('mycelium-theme')`.
3. Closes the popover.

**Popover:** 4 circular swatches (24×24px), labelled on hover via `title` attribute:
- `amoled` swatch: `#000000` + `#4f98a3` accent ring
- `anthropic` swatch: `#1f1e1a` + `#d97757` accent ring
- `apple` swatch: `#1c1c1e` + `#0a84ff` accent ring
- `arc` swatch: `#13101f` + `#9b7ff4` accent ring

Active swatch: `outline: 2px solid var(--c-accent-base); outline-offset: 2px`.

---

### 4.3 `Home.module.css` Token Replacements

Replace every bare hex literal with its corresponding token. Key mappings:

| Current Hex | Token |
|---|---|
| `#000000` (bg) | `var(--c-bg-base)` |
| `#080808` | `var(--c-bg-1)` |
| `#0d0d0d` | `var(--c-bg-2)` |
| `#141414` / `#1a1a1a` | `var(--c-bg-offset)` |
| `#e8e8e8` | `var(--c-text-primary)` |
| `#a0a0a0` | `var(--c-text-muted)` |
| `#505050` / `#585858` | `var(--c-text-faint)` |
| `#4f98a3` | `var(--c-accent-base)` |
| `#227f8b` | `var(--c-accent-hover)` |
| `#0e2a2e` / `#061618` | `var(--c-accent-subtle)` |
| `#6daa45` | `var(--c-ok-base)` |
| `#1a2e18` | `var(--c-ok-subtle)` |
| `#fdab43` | `var(--c-warn-base)` |
| `#2e1a08` | `var(--c-warn-subtle)` |
| `#dd6974` | `var(--c-err-base)` |
| `#1a0808` / `#120808` | `var(--c-err-subtle)` |
| `#1c1c1c` / `#2a2a2a` | `var(--c-border-default)` |
| `#2e2e2e` / `#3a3a3a` | `var(--c-border-strong)` |
| `rgba(0,0,0,0.6)` | `var(--c-overlay)` |

All `border-radius` values replaced with token equivalents:
- `3px–5px` → `var(--radius-sm)`
- `7px–10px` → `var(--radius-md)`
- `12px–14px` → `var(--radius-lg)`
- `99px` → `var(--radius-pill)`

All `transition: … 140ms` → `transition: … var(--dur-fast) var(--ease-spring)`.

---

### 4.4 Typography Scale

Add to `globals.css` (theme-independent — sizes are universal):

```css
:root {
  --text-xs:   clamp(0.6875rem, 0.65rem + 0.2vw, 0.75rem);   /* 11–12px */
  --text-sm:   clamp(0.75rem,   0.7rem  + 0.25vw, 0.8125rem); /* 12–13px */
  --text-base: clamp(0.8125rem, 0.78rem + 0.3vw,  0.875rem);  /* 13–14px — chat body */
  --text-md:   clamp(0.875rem,  0.85rem + 0.25vw, 0.9375rem); /* 14–15px */
  --text-lg:   clamp(1rem,      0.95rem + 0.5vw,  1.125rem);  /* 16–18px — headings */
  --text-xl:   clamp(1.375rem,  1.2rem  + 0.8vw,  1.6rem);    /* 22–26px — home title sub */
  --text-hero: clamp(2.2rem,    1.8rem  + 2vw,    3.4rem);    /* home title */
}
```

Apply in component CSS using `var(--text-*)` instead of `clamp(...)` or `px` literals.

---

### 4.5 `ChatBubble.module.css` Polish

**Graph chip button** (`.graphChipBtn`):
- Add `cursor: not-allowed; opacity: 0.38;` when `disabled` attribute is present.
- Change transition to `var(--dur-fast) var(--ease-spring)` across background, border-color, color, opacity.
- Focus ring: `outline: 2px solid var(--c-accent-base); outline-offset: 2px` via `:focus-visible`.

**Answer text** (`.answerText`):
- Replace hardcoded `font-family: inherit` with `font-family: var(--font-body)`.
- Add `line-height: 1.7` (up from 1.65 — more breathing room for long answers).

**Bubble meta row** (`.bubbleMeta`):
- All timestamp/label colours replaced by `var(--c-text-faint)`.

---

### 4.6 `ModeSelector.module.css` Polish

**Pill container** (`.modeSelector`):
- Background: `var(--c-bg-1)` with `border: 1px solid var(--c-border-hairline)`.
- Border-radius: `var(--radius-pill)`.

**Active option** (`.modeOptionActive`):
- Replace hardcoded `#4f98a3` fill with `var(--c-accent-base)`.
- Text: `var(--c-text-inverse)` (so all four themes get theme-matching contrast).
- Slide transition: `left/width` animated with `var(--dur-normal) var(--ease-spring)` (the existing transform approach already works; just replace the timing literal).

**Inactive option hover**:
- `color: var(--c-text-muted)` → `var(--c-text-primary)` on hover.
- Transition: `color var(--dur-fast) ease`.

---

### 4.7 `ReasoningGraph.module.css` Token Pass

All zone-colour hardcoded values to be mapped:

| Zone / Use | Current | Token or New Value |
|---|---|---|
| Panel bg | `#000000` | `var(--c-bg-base)` |
| Panel header | `#0d0d0d` | `var(--c-bg-2)` |
| Node bg default | `#0d0d0d` | `var(--c-bg-2)` |
| Node border default | `#242424` | `var(--c-border-default)` |
| Accent node fill | `#4f98a3` based | `var(--c-accent-base)` |
| Edge stroke | `#2a2a2a` | `var(--c-border-default)` |
| Highlighted edge | `#4f98a3` | `var(--c-accent-base)` |
| Ok node glow | `#6daa45` | `var(--c-ok-base)` |
| Error node glow | `#dd6974` | `var(--c-err-base)` |
| Overlay backdrop | `rgba(0,0,0,0.7)` | `var(--c-overlay)` |
| Close button hover | `#1a1a1a` | `var(--c-bg-offset)` |

---

### 4.8 Phase Indicator Polish

The `.phaseBarFill` currently animates `width` only. Add:

```css
/* In Home.module.css — phaseBarFill */
.phaseBarFill {
  background: linear-gradient(
    90deg,
    var(--c-accent-base) 0%,
    color-mix(in oklch, var(--c-accent-base) 70%, var(--c-ok-base)) 100%
  );
  box-shadow: 0 0 6px color-mix(in oklch, var(--c-accent-base) 60%, transparent);
  transition: width var(--dur-slow) var(--ease-spring);
}
```

For the `arc` theme the gradient reads `#9b7ff4 → #74c99a` automatically through the token values — no extra rules needed.

---

### 4.9 Calibration Gate Polish

The `.calibrationCard` gains:
```css
.calibrationCard {
  background: var(--c-bg-1);
  border: 1px solid var(--c-border-hairline);
  /* subtle inner light */
  box-shadow:
    inset 0 1px 0 color-mix(in oklch, var(--c-text-primary) 4%, transparent),
    0 24px 48px color-mix(in oklch, var(--c-bg-base) 90%, transparent);
}
```

`.calibrationLogo` colour: `var(--c-accent-base)`.  
`.calibrationBarFill`: `background: var(--c-accent-base)` (gradient optional per theme).

---

### 4.10 Scrollbar Styling

Add to `globals.css`:

```css
/* Thin scrollbar — themed */
::-webkit-scrollbar         { width: 5px; height: 5px; }
::-webkit-scrollbar-track   { background: transparent; }
::-webkit-scrollbar-thumb   {
  background: var(--c-border-strong);
  border-radius: var(--radius-pill);
}
::-webkit-scrollbar-thumb:hover { background: var(--c-text-faint); }

/* Firefox */
* { scrollbar-width: thin; scrollbar-color: var(--c-border-strong) transparent; }
```

---

### 4.11 Focus Ring Standardisation

Currently focus rings are absent or inconsistent. Add to `globals.css`:

```css
:focus-visible {
  outline: 2px solid var(--c-accent-base);
  outline-offset: 2px;
  border-radius: var(--radius-sm);
}

/* Inputs get a border-color transition instead of outline */
input:focus-visible,
textarea:focus-visible {
  outline: none;
  border-color: var(--c-accent-base) !important;
  box-shadow: 0 0 0 3px color-mix(in oklch, var(--c-accent-base) 20%, transparent);
}
```

---

## 5. Migration Strategy

### Phase 1 — Token extraction (no visual change)
- Extract all hex values from `Home.module.css` into `globals.css` `[data-theme="amoled"]`.
- Replace in-file with `var(--token)` across all four CSS modules.
- Ship — zero visual change, existing theme matches `amoled` 1:1.

### Phase 2 — Typography & motion
- Add `--text-*` scale, `--font-body`/`--font-mono`, `--dur-*`, `--ease-spring`.
- Update component CSS to use type tokens.
- Move font loading to `_document.tsx`.
- Standardise focus rings and scrollbars.

### Phase 3 — Additional themes
- Add `anthropic`, `apple`, `arc` token blocks to `globals.css`.
- Build `ThemeSwitcher.tsx` component.
- Wire up initialisation script in `_document.tsx` `<head>`.

### Phase 4 — Component polish pass
- Phase bar gradient + glow.
- Calibration card box-shadow.
- `ChatBubble` disabled state + focus.
- `ModeSelector` token pass.
- `ReasoningGraph` token pass.

---

## 6. Out of Scope

The following are **explicitly excluded** from this spec and belong in functional iteration tickets:
- Any change to component props, hooks, state, or data flow.
- New graph node shapes or layout algorithms for TRM v0.2 phases (covered in TRM v0.2 spec).
- Responsive layout changes beyond what token substitution naturally provides.
- Light mode / bright-background themes (none of the four themes above are light — Mycelium is a terminal-adjacent tool).
- Accessibility audit beyond focus ring standardisation (separate ticket).
- Animation changes to the ReasoningGraph SVG/Canvas layer.

---

## 7. File Locations

```
web-ui/
  styles/
    globals.css                  ← token blocks for all 4 themes + scrollbar + focus
    Home.module.css              ← token substitution + type tokens
    ChatBubble.module.css        ← token substitution + disabled state
    ModeSelector.module.css      ← token substitution + timing tokens
    ReasoningGraph.module.css    ← token substitution
  components/
    ThemeSwitcher.tsx            ← new file (Phase 3)
  pages/
    _document.tsx                ← font <link> preconnect + theme init script
```
