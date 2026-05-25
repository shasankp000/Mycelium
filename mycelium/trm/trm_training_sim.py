#!/usr/bin/env python3
"""
Phase D — TRM Training Simulation
====================================
Drives ``run_mycelium_workflow()`` in backend mode (no SSE) using
LLM-generated user-like queries per domain, then trains TRMReasoner on
the accumulated routing traces.

Usage
-----
::

    # Default: uses Ollama with llama3 on localhost
    python -m mycelium.trm.trm_training_sim

    # Custom LLM endpoint / model
    python -m mycelium.trm.trm_training_sim \\
        --llm-url http://localhost:11434/api/generate \\
        --llm-model mistral \\
        --queries-per-domain 50 \\
        --epochs 5

    # OpenAI-compatible endpoint (e.g. LM Studio, vLLM)
    python -m mycelium.trm.trm_training_sim \\
        --llm-url http://localhost:1234/v1/chat/completions \\
        --llm-model local-model \\
        --openai-compat

    # Skip simulation entirely — train directly from existing routing traces.
    # Use this to recover from a power cut or any interruption that left the
    # JSONL intact but prevented the training step from running.
    python -m mycelium.trm.trm_training_sim --train-only
    python -m mycelium.trm.trm_training_sim --train-only --epochs 5

Flow
----
1. For each domain in DOMAIN_LIST (skipping __reserved*):
   a. Ask LLM to generate ``--queries-per-domain`` user-like queries.
   b. Feed each query to ``run_mycelium_workflow([query], on_event=None)``.
   c. ``TRMRoutingTraceWriter`` (singleton in run_workflow) captures every
      routed sentence into ``training_data/routing_traces.jsonl``.
2. After all domains are processed, call ``TRMTrainer.train()`` on the
   accumulated traces.
3. Save checkpoint to ``mycelium/trm/trm_checkpoints/trm_latest.pt``.
   ``run_mycelium_workflow`` picks it up on next process start via
   ``_get_trm_reasoner()``.

LLM integration
---------------
The script supports two API shapes:

* **Ollama** (default): POST ``/api/generate`` with ``{"model": ..., "prompt": ...}``
* **OpenAI-compat** (``--openai-compat``): POST ``/v1/chat/completions``
  with ``{"model": ..., "messages": [{"role": "user", "content": ...}]}``

Any LLM endpoint that speaks one of these two protocols works without
changes to the rest of the pipeline.

Query generation prompt template
---------------------------------
For each domain the following prompt is sent to the LLM::

    Generate {n} realistic, diverse user questions about {domain}.
    Rules:
    - One question per line, no numbering or bullet points.
    - Vary complexity: mix simple factual, causal, procedural, and
      comparative questions.
    - Vary specificity: some broad ("what is X"), some narrow
      ("why does X happen when Y").
    - Do NOT include any preamble, explanation, or closing remarks.
    - Output ONLY the questions, one per line.

The resulting lines are stripped and filtered to remove empties and
lines that look like headers/preamble (heuristic: ≤ 3 words or ends
with ":").
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import List, Optional

from mycelium.trm.trm_routing_trace_writer import DOMAIN_LIST as _RAW_DOMAIN_LIST

ACTIVE_DOMAINS: List[str] = [
    d for d in _RAW_DOMAIN_LIST if not d.startswith("__reserved")
]

_CHECKPOINT_DIR = Path(__file__).parent / "trm_checkpoints"
_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
_CHECKPOINT_PATH = _CHECKPOINT_DIR / "trm_latest.pt"


_PROMPT_TEMPLATE = """\
Generate {n} realistic, diverse user questions about {domain}.
Rules:
- One question per line, no numbering or bullet points.
- Vary complexity: mix simple factual, causal, procedural, and comparative questions.
- Vary specificity: some broad ("what is X"), some narrow ("why does X happen when Y").
- Do NOT include any preamble, explanation, or closing remarks.
- Output ONLY the questions, one per line."""


def _clean_lines(raw: str, expected_n: int) -> List[str]:
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or len(line.split()) <= 3 or line.endswith(":"):
            continue
        for prefix in ("- ", "* ", "\u2022 "):
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
        if line[:3].rstrip(". ").isdigit():
            line = line.split(".", 1)[-1].strip()
        if line:
            lines.append(line)
    return lines[:expected_n]


def _ollama_generate(url: str, model: str, prompt: str, timeout: int = 120) -> str:
    import urllib.request
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode())
    return body.get("response", "")


def _openai_generate(url: str, model: str, prompt: str, timeout: int = 120) -> str:
    import urllib.request
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.9,
    }).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode())
    return body["choices"][0]["message"]["content"]


def generate_queries(
    domain: str, n: int, llm_url: str, llm_model: str,
    openai_compat: bool = False, timeout: int = 120, retries: int = 2,
) -> List[str]:
    prompt = _PROMPT_TEMPLATE.format(n=n, domain=domain)
    for attempt in range(retries + 1):
        try:
            raw = (_openai_generate if openai_compat else _ollama_generate)(
                llm_url, llm_model, prompt, timeout
            )
            queries = _clean_lines(raw, n)
            if queries:
                return queries
            print(f"  [warn] LLM returned no usable lines for '{domain}' (attempt {attempt+1})")
        except Exception as exc:
            print(f"  [warn] LLM call failed for '{domain}' (attempt {attempt+1}): {exc}")
        time.sleep(2)
    return []


def _run_single_query(query: str, sim_prefix: str, idx: int) -> bool:
    from mycelium.pipeline.run_workflow import run_mycelium_workflow
    trace_id = f"{sim_prefix}-{idx:06d}"
    try:
        run_mycelium_workflow([query], trace_id=trace_id, on_event=None)
        return True
    except Exception as exc:
        print(f"  [warn] workflow failed for query {idx!r}: {exc}")
        return False


def _train_trm(
    train_path: str = "training_data/routing_traces.jsonl",
    eval_path:  str = "evaluation_data/routing_ground_truth.jsonl",
    epochs: int = 3,
) -> bool:
    """
    Load accumulated routing traces and run TRMTrainer.

    Checkpoint loading handles two formats:
      - New:    {model_state, ema_state, step}  (no cfg — PyTorch 2.6 safe)
      - Legacy: {model_state, ema_state, step, cfg}  (older checkpoints)
    weights_only=False is used so that legacy checkpoints with TRMConfig
    objects (saved before this fix) still load correctly.

    trainer.step is always reset to 0 after loading so that max_train_steps
    is computed freshly from the current dataset size and the training loop
    always runs a full pass — regardless of the step count stored in the
    checkpoint.

    cfg.device is overridden to CUDA if available so training uses the GPU.
    TRMConfig defaults to "cpu" for inference but training should always
    use the fastest available device.
    """
    if not Path(train_path).exists():
        print(f"[error] Training data not found at {train_path!r} — skipping training.")
        return False

    line_count = sum(1 for _ in open(train_path, encoding="utf-8"))
    if line_count == 0:
        print("[error] Training file is empty — skipping training.")
        return False

    print(f"\n{'='*60}")
    print(f"TRM TRAINING — {line_count} samples ({train_path})")
    print(f"{'='*60}")

    try:
        import torch as _torch
        from mycelium.trm.config import TRMConfig
        from mycelium.trm.reasoner import TRMReasoner
        from mycelium.trm.trainer import TRMTrainer, _EmptyDataset

        cfg = TRMConfig()
        # Override device: always use CUDA for training if available.
        # TRMConfig defaults to "cpu" (suitable for inference), but training
        # on GPU is significantly faster and should always be preferred.
        cfg.device = "cuda" if _torch.cuda.is_available() else "cpu"
        print(f"Training device: {cfg.device}"
              + (f" ({_torch.cuda.get_device_name(0)})" if cfg.device == "cuda" else ""))

        reasoner = TRMReasoner(cfg)

        if _CHECKPOINT_PATH.exists():
            # weights_only=False: handles both new (no cfg) and legacy (with cfg)
            # checkpoints.  Safe because we wrote this file ourselves.
            ckpt = _torch.load(str(_CHECKPOINT_PATH), map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "model_state" in ckpt:
                state = ckpt["model_state"]
                _step = ckpt.get("step", "?")
                print(f"\u2705 Loaded existing checkpoint from {_CHECKPOINT_PATH} — fine-tuning (step={_step})")
            else:
                state = ckpt
                print(f"\u2705 Loaded legacy checkpoint from {_CHECKPOINT_PATH} — fine-tuning")
            reasoner.load_state_dict(state)
        else:
            print("\u26a0\ufe0f  No existing checkpoint — training from random init")

        import json as _json

        class _JSONLDataset:
            def __init__(self, path: str) -> None:
                import torch as _t
                self._records = []
                with open(path, encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        rec = _json.loads(line)
                        self._records.append({
                            "token_ids":            _t.tensor(rec["token_ids"],            dtype=_t.long),
                            "spectral_vec":         _t.tensor(rec["spectral_vec"],         dtype=_t.float32),
                            "predicate_family_id":  _t.tensor(rec["predicate_family_id"],  dtype=_t.long),
                            "initial_domain_probs": _t.tensor(rec["initial_domain_probs"], dtype=_t.float32),
                            "target_domain":        _t.tensor(rec["target_domain"],        dtype=_t.long),
                        })

            def __len__(self) -> int:
                return len(self._records)

            def __getitem__(self, idx: int) -> dict:
                return self._records[idx]

        train_dataset = _JSONLDataset(train_path)
        eval_dataset  = (
            _JSONLDataset(eval_path)
            if Path(eval_path).exists() and Path(eval_path).stat().st_size > 0
            else None
        )

        effective_batch_size = min(32, len(train_dataset))
        if effective_batch_size == 0:
            print("[error] Training dataset is empty after loading — skipping training.")
            return False

        steps_per_epoch = max(1, len(train_dataset) // effective_batch_size)
        cfg.max_train_steps = steps_per_epoch * epochs
        cfg.warmup_steps    = min(cfg.warmup_steps, cfg.max_train_steps // 10)

        trainer = TRMTrainer(
            model=reasoner,
            cfg=cfg,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            batch_size=effective_batch_size,
        )
        # Always reset step to 0 so training runs a full pass on the current
        # dataset.  The checkpoint step is informational only — using it as the
        # starting counter would cause the loop to exit immediately if
        # max_train_steps <= saved step.
        trainer.step = 0
        print(f"Training for {cfg.max_train_steps} steps ({steps_per_epoch} steps/epoch × {epochs} epochs)")

        trainer.train()
        trainer.save(str(_CHECKPOINT_PATH))
        print(f"\n\u2705 TRMReasoner checkpoint saved to {_CHECKPOINT_PATH}")

        import mycelium.pipeline.run_workflow as _rw
        _rw._trm_reasoner = None
        print("\u2705 In-process TRMReasoner singleton reset — will reload on next workflow call")

        return True
    except Exception as exc:
        print(f"[error] TRM training failed: {exc}")
        import traceback; traceback.print_exc()
        return False


def run_sim(
    llm_url: str, llm_model: str, queries_per_domain: int, epochs: int,
    openai_compat: bool, llm_timeout: int, domains: Optional[List[str]] = None,
) -> None:
    target_domains = domains or ACTIVE_DOMAINS
    sim_prefix = f"sim-{uuid.uuid4().hex[:8]}"

    print("\n" + "="*60)
    print("TRM TRAINING SIMULATION")
    print("="*60)
    print(f"LLM endpoint : {llm_url}")
    print(f"LLM model    : {llm_model}")
    print(f"Domains      : {target_domains}")
    print(f"Queries/dom  : {queries_per_domain}")
    print(f"Epochs       : {epochs}")
    print(f"Sim prefix   : {sim_prefix}")
    print(f"OpenAI compat: {openai_compat}")
    print(f"Checkpoint   : {_CHECKPOINT_PATH}")
    print("="*60 + "\n")

    total_queries = 0
    total_successes = 0
    global_idx = 0

    for domain in target_domains:
        print(f"\n[{domain.upper()}] Generating {queries_per_domain} queries via LLM...")
        queries = generate_queries(
            domain=domain, n=queries_per_domain, llm_url=llm_url,
            llm_model=llm_model, openai_compat=openai_compat, timeout=llm_timeout,
        )
        if not queries:
            print(f"  [skip] No queries generated for domain '{domain}'")
            continue

        print(f"  Generated {len(queries)} queries. Running through pipeline...")
        domain_successes = 0
        for q_idx, query in enumerate(queries, start=1):
            global_idx += 1
            ok = _run_single_query(query, sim_prefix, global_idx)
            if ok:
                domain_successes += 1
            print(f"  {'\u2705' if ok else '\u274c'} [{q_idx}/{len(queries)}] {query[:80]}")

        total_queries += len(queries)
        total_successes += domain_successes
        print(f"  Domain '{domain}' done: {domain_successes}/{len(queries)} successful")

    print(f"\n{'='*60}")
    print(
        f"Simulation complete: {total_successes}/{total_queries} queries succeeded "
        f"across {len(target_domains)} domain(s)"
    )

    from mycelium.trm.trm_routing_trace_writer import trm_trace_writer
    if trm_trace_writer:
        counts = trm_trace_writer.counts()
        print(f"TraceWriter: train={counts['train']} eval={counts['eval']} samples written")

    if total_successes == 0:
        print("[warn] No successful queries — skipping training.")
        return

    _train_trm(epochs=epochs)


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TRM training simulation: LLM-driven backend interaction loop"
    )
    parser.add_argument("--llm-url", default=os.getenv("TRM_SIM_LLM_URL", "http://localhost:11434/api/generate"))
    parser.add_argument("--llm-model", default=os.getenv("TRM_SIM_LLM_MODEL", "llama3:8b"))
    parser.add_argument("--queries-per-domain", type=int, default=int(os.getenv("TRM_SIM_QPD", "50")))
    parser.add_argument("--epochs", type=int, default=int(os.getenv("TRM_SIM_EPOCHS", "3")))
    parser.add_argument(
        "--openai-compat", action="store_true",
        default=os.getenv("TRM_SIM_OPENAI_COMPAT", "").lower() in ("1", "true", "yes"),
    )
    parser.add_argument("--llm-timeout", type=int, default=int(os.getenv("TRM_SIM_TIMEOUT", "120")))
    parser.add_argument("--domains", nargs="+", default=None,
        help=f"Subset of domains to simulate. Available: {ACTIVE_DOMAINS}")
    parser.add_argument(
        "--train-only", action="store_true",
        default=False,
        help=(
            "Skip simulation entirely and train directly from the existing "
            "routing_traces.jsonl. Use this to recover after a power cut or "
            "any interruption that left the JSONL intact but prevented the "
            "training step from running."
        ),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()

    if args.train_only:
        print("\n[--train-only] Skipping simulation — training directly from existing traces.")
        ok = _train_trm(epochs=args.epochs)
        sys.exit(0 if ok else 1)

    run_sim(
        llm_url=args.llm_url, llm_model=args.llm_model,
        queries_per_domain=args.queries_per_domain, epochs=args.epochs,
        openai_compat=args.openai_compat, llm_timeout=args.llm_timeout,
        domains=args.domains,
    )
