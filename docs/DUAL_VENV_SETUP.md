# Dual `.venv` Architecture Setup Guide

Mycelium runs on **Python 3.14** (`.venv`).  
Lexis-E's NLP pipeline requires **Python 3.11** due to spaCy compatibility constraints.  
This guide explains how to set up the two isolated environments side-by-side and wire them together via the subprocess bridge.

---

## Architecture Overview

```
mycelium/
├── .venv/              ← Mycelium (Python 3.14) — your existing venv, untouched
├── .venv2/             ← Lexis-E (Python 3.11) — copy / symlink of Lexis's venv
├── spacy_bridge.py     ← Mycelium-side shim (runs in .venv, never imports spaCy)
├── spacy_worker.py     ← Worker script (runs inside .venv2, owns all spaCy calls)
└── config.toml         ← [lexis] venv2_path controls the .venv2 location
```

Mycelium code **never imports spaCy directly**. Every NLP call goes through `spacy_bridge.py`, which spawns `.venv2/bin/python spacy_worker.py` as a subprocess, sends JSON over stdin, and reads JSON from stdout.

---

## Why Two Venvs?

| Concern | Shared venv | Isolated `.venv2` |
|---|---|---|
| spaCy requires Python 3.11 | Forces Mycelium to 3.11 | No constraint on Mycelium |
| `fastcoref` dependency pins | May conflict with Mycelium's `transformers` | Fully isolated |
| `huggingface-hub` upper-bound | Must patch Mycelium's transformers | Only patches `.venv2` |
| Breakage risk | High — one `pip install` can break both | Zero — envs never interact |

---

## Step 1 — Obtain Lexis's `.venv`

On your **testing machine** you already have Lexis's Python 3.11 venv.  
You only need to make it accessible from the Mycelium project root as `.venv2`.

### Option A — Copy (recommended for portability)

```bash
# From inside the Mycelium project root:
cp -r /path/to/lexis/.venv .venv2
```

### Option B — Symlink (saves disk space, requires both repos on the same machine)

```bash
# From inside the Mycelium project root:
ln -s /path/to/lexis/.venv .venv2
```

> **Note:** The path `/path/to/lexis/.venv` is wherever Lexis's venv lives.  
> If you cloned Lexis next to Mycelium, it would be `../Lexis/.venv`.

---

## Step 2 — Verify the Python Version

```bash
.venv2/bin/python --version
# Expected: Python 3.11.x
```

If this shows 3.14 or anything other than 3.11.x, you pointed at the wrong venv.

---

## Step 3 — Verify spaCy and Models are Installed

```bash
.venv2/bin/python -c "import spacy; print(spacy.__version__)"
# Expected: 3.7.x or later

.venv2/bin/python -m spacy validate
# Expected: ✔ en_core_web_sm  (and en_core_web_lg if installed)
```

If `en_core_web_sm` is missing:

```bash
.venv2/bin/python -m spacy download en_core_web_sm
# Optional larger model:
.venv2/bin/python -m spacy download en_core_web_lg
```

---

## Step 4 — (Optional) Override the `.venv2` Path

By default `spacy_bridge.py` looks for `.venv2` relative to the Mycelium project root.  
You can override this in two ways:

### Via `config.toml`

```toml
[lexis]
venv2_path = ".venv2"             # relative to project root
# or absolute:
# venv2_path = "/home/user/lexis/.venv"
```

### Via environment variable

```bash
export LEXIS_VENV2_PATH="/home/user/lexis/.venv"
```

The bridge checks config.toml first, then the env var, then defaults to `.venv2`.

---

## Step 5 — Smoke-Test the Bridge

Activate Mycelium's venv (Python 3.14) and run the bridge's built-in smoke-test:

```bash
source .venv/bin/activate
python spacy_bridge.py
```

Expected output:

```
[spacy_bridge] Running smoke-test against .venv2...
[spacy_bridge] Sentences : ['Quantum mechanics describes the behaviour of particles at the subatomic scale.']
[spacy_bridge] Entities  : []
[spacy_bridge] POS sample: [{'text': 'Quantum', 'pos': 'PROPN', ...}, ...]
[spacy_bridge] ✅ All OK
```

If you see `✅ All OK`, the dual-venv architecture is fully operational.

---

## Step 6 — Programmatic Health Check

Anywhere in Mycelium code you can verify the bridge at runtime:

```python
from spacy_bridge import health_check

if not health_check():
    raise RuntimeError("spaCy bridge is not available. See DUAL_VENV_SETUP.md.")
```

---

## Using the Bridge in Mycelium Code

```python
from spacy_bridge import pipeline, sentences, ner, pos_tag, tokenize, lemmatize

# Single full-pass (most efficient if you need multiple annotations):
result = pipeline("The cat sat on the mat.")
print(result["tokens"])      # ['The', 'cat', 'sat', ...]
print(result["entities"])    # []
print(result["sentences"])   # ['The cat sat on the mat.']
print(result["pos"])         # [{'text': 'The', 'pos': 'DET', ...}, ...]

# Individual helpers:
toks  = tokenize("Hello world.")      # ['Hello', 'world', '.']
sents = sentences(long_text)          # ['Sentence one.', 'Sentence two.', ...]
ents  = ner(long_text)                # [{'text': 'London', 'label': 'GPE', ...}]
lemma = lemmatize("running quickly")  # ['run', 'quickly']
```

### Async callers (FastAPI route handlers, async pipelines)

```python
from spacy_bridge import async_pipeline, async_spacy_call

# In an async function:
result = await async_pipeline(text)
ents   = (await async_spacy_call("ner", text))["entities"]
```

### Choosing the model

All helpers accept an optional `model` argument:

```python
# Faster, smaller (default):
result = pipeline(text, model="en_core_web_sm")

# More accurate, requires .venv2/bin/python -m spacy download en_core_web_lg:
result = pipeline(text, model="en_core_web_lg")
```

---

## Invariant: No Direct spaCy Imports in Mycelium

> **Rule:** No file in the Mycelium project (`.venv`) may contain `import spacy` or any
> direct reference to a spaCy object outside of `spacy_worker.py`.
>
> `spacy_worker.py` is the **only** file that imports spaCy, and it is only ever
> executed by `.venv2/bin/python`, never by Mycelium's own interpreter.

This invariant ensures that Mycelium's Python 3.14 environment is never polluted by
spaCy's Python 3.11 dependency tree, and that spaCy bugs on 3.14 never affect Mycelium.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `RuntimeError: .venv2 Python interpreter not found` | `.venv2` not created yet | Follow Steps 1–2 |
| `Could not load spaCy model 'en_core_web_sm'` | Model not downloaded in `.venv2` | Run Step 3 |
| `Worker exited with code 1` | Bad `.venv2` or missing dependency | Run `.venv2/bin/python -m spacy validate` |
| `health_check FAILED: timed out` | Very slow machine or huge text | Increase `spacy_timeout` in `config.toml` |
| `JSON parse error` | Worker printed a traceback instead of JSON | Run worker manually: `echo '{"task":"tokenize","text":"hi"}' \| .venv2/bin/python spacy_worker.py` |

---

## `.gitignore` Note

`.venv2/` is already covered by the existing `.gitignore` pattern for `.venv*/` — it will not be committed to the repository. Each developer sets up their own `.venv2` locally by copying or symlinking Lexis's venv.
