"""
run_workflow.py — TRM v2 CLI entrypoint.

Usage examples:
  # Interactive REPL (legacy path, TRM v2 disabled)
  python run_workflow.py

  # Backend-only single query, legacy path
  python run_workflow.py --backend-only --query "Is 9.9 bigger than 9.11?"

  # Batch input from file, legacy path, write output
  python run_workflow.py --input-file queries.txt --output results.json

  # Enable TRM v2 pipeline (feature flag), single query
  python run_workflow.py --enable-trm-v2 --backend-only --query "What is calculus?"

  # TRM v2, batch input from file
  python run_workflow.py --enable-trm-v2 --input-file queries.txt --output results.json

  # Specify domain graph path and shard root
  python run_workflow.py --enable-trm-v2 --graph-path data/graph.json \\
      --shard-root data/shards --backend-only --query "Newton's laws"
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_workflow",
        description="Mycelium TRM inference workflow runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--enable-trm-v2", action="store_true", default=False,
        help="Enable the TRM v2 pipeline. Off by default — legacy path runs instead.",
    )
    p.add_argument(
        "--backend-only", action="store_true", default=False,
        help="Run a single query and print JSON result. No REPL. Requires --query.",
    )
    p.add_argument(
        "--query", "-q", type=str, default=None,
        help="Query string.  Used for single-shot mode (--backend-only) or as an"
             " extra first sentence alongside --input-file.",
    )
    # --- Batch input / output (per IMPLEMENTATION_SPEC_TRM_V2) ------------
    p.add_argument(
        "--input-file", type=str, default=None,
        help="Path to a plain-text file of queries/sentences, one per line."
             " Activates batch mode for both legacy and TRM v2 paths.",
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="Path to write the JSON result file.  Defaults to stdout.",
    )
    # -----------------------------------------------------------------------
    p.add_argument(
        "--graph-path", type=str, default="data/domain_graph.json",
        help="Path to the domain graph JSON file.",
    )
    p.add_argument(
        "--shard-root", type=str, default="data/shards",
        help="Root directory for per-domain SQLite shards.",
    )
    p.add_argument(
        "--artifact-dir", type=str, default="artifacts/trm_v2/heads",
        help="Root directory for trained head artifacts.",
    )
    p.add_argument(
        "--log-level", type=str, default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    p.add_argument(
        "--return-embedding", action="store_true", default=False,
        help="Include the query embedding vector in JSON output (TRM v2 only).",
    )
    p.add_argument(
        "--reasoning-mode", type=str, default="smart",
        choices=["fast", "smart", "deep"],
        help="Reasoning depth for the legacy pipeline (default: smart).",
    )
    return p


# ---------------------------------------------------------------------------
# Legacy path
# ---------------------------------------------------------------------------

def _load_input_sentences(args: argparse.Namespace) -> List[str]:
    """Resolve --input-file and/or --query into an ordered sentence list."""
    sentences: List[str] = []
    if args.input_file:
        path = Path(args.input_file)
        if not path.exists():
            print(f"ERROR: --input-file '{args.input_file}' not found.", file=sys.stderr)
            sys.exit(1)
        sentences = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if args.query:
        sentences.insert(0, args.query)
    return sentences


def run_legacy(args: argparse.Namespace) -> None:
    """
    Delegates entirely to the archived TRM v1 pipeline via the shim at
    mycelium/pipeline/run_workflow.py.  This is the ONLY place legacy code
    is imported.  Delete this function when TRM v2 reaches parity.
    """
    try:
        from mycelium.pipeline.run_workflow import run as legacy_run  # type: ignore
    except ImportError:
        logger.error(
            "Legacy pipeline shim not importable. "
            "Ensure mycelium/pipeline/run_workflow.py exists, "
            "or use --enable-trm-v2."
        )
        sys.exit(1)

    sentences = _load_input_sentences(args)

    # Single-shot / batch both route through the same shim adapter.
    # backend_only=True forces the shim to skip its own REPL.
    backend_only = bool(args.backend_only or args.input_file or args.query)

    legacy_run(
        query=None,            # already folded into `sentences`
        backend_only=backend_only,
        sentences=sentences if sentences else None,
        reasoning_mode=args.reasoning_mode,
        output_path=args.output,
    )


# ---------------------------------------------------------------------------
# TRM v2 path
# ---------------------------------------------------------------------------

def build_trm_v2_pipeline(args: argparse.Namespace):
    import torch
    import torch.nn as nn
    from mycelium.domain_graph.registry import DomainGraphRegistry
    from mycelium.domain_graph.novelty import DomainNoveltyPolicy, NoveltyThresholds
    from mycelium.trm.v2.encoder import SharedEncoder
    from mycelium.trm.v2.heads import HeadRegistry
    from mycelium.trm.v2.gating import DomainGate
    from mycelium.trm.v2.halt import HaltControllerV2
    from mycelium.trm.v2.inference import TRMV2InferenceEngine
    from mycelium.trm.v2.cold.manager import ColdStorageManager
    from mycelium.trm.v2.store.query_store import QueryStore
    from mycelium.trm.v2.pipeline import TRMV2Pipeline

    registry = DomainGraphRegistry(args.graph_path)

    try:
        from mycelium.trm.network import TRMNetwork  # type: ignore
        backbone = TRMNetwork()
        output_dim = backbone.output_dim
        logger.info("Loaded TRMNetwork backbone.")
    except (ImportError, AttributeError):
        logger.warning(
            "TRMNetwork not available — using EmbeddingBag stub (vocab=30522, DIM=128)."
        )
        output_dim = 128
        backbone = nn.EmbeddingBag(30522, output_dim, mode="mean", sparse=False)

    encoder = SharedEncoder(backbone, output_dim=output_dim, frozen=True)

    head_registry = HeadRegistry()
    _load_heads_from_artifacts(head_registry, args.artifact_dir, output_dim, registry)

    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        logger.info("Loaded bert-base-uncased tokenizer.")
    except Exception as exc:
        logger.warning(
            "Tokenizer unavailable (%s) — pass pre-tokenised input_ids manually.", exc
        )

    policy = DomainNoveltyPolicy(registry, NoveltyThresholds())
    gate   = DomainGate(head_registry, registry)
    halt   = HaltControllerV2()
    engine = TRMV2InferenceEngine(
        encoder=encoder, gate=gate, novelty_policy=policy,
        domain_registry=registry, halt_controller=halt, device="cpu",
        tokenizer=tokenizer,
    )

    cold  = ColdStorageManager(registry)
    store = QueryStore(shard_root=args.shard_root)
    return TRMV2Pipeline(engine, cold, store, registry, enabled=True)


def _load_heads_from_artifacts(head_registry, artifact_dir, output_dim, registry) -> None:
    from pathlib import Path
    from mycelium.trm.v2.heads import DomainHead
    from mycelium.trm.v2.checkpointing import load_head, load_manifest

    root = Path(artifact_dir)
    if not root.exists():
        logger.info("Artifact dir '%s' does not exist — no heads loaded.", artifact_dir)
        return
    for domain_dir in root.iterdir():
        if not domain_dir.is_dir():
            continue
        domain_id = domain_dir.name
        manifest = load_manifest(artifact_dir, domain_id)
        if manifest is None:
            continue
        entry = manifest.get(domain_id)
        if entry is None or not entry.artifact_path:
            continue
        try:
            head = DomainHead(
                domain_id, input_dim=entry.input_dim,
                hidden_dim=entry.hidden_dim, output_dim=entry.output_dim,
            )
            load_head(head, entry.artifact_path)
            head_registry.register(head)
            logger.info("Loaded head for '%s'.", domain_id)
        except Exception as exc:
            logger.warning("Failed to load head for '%s': %s", domain_id, exc)


def run_trm_v2(args: argparse.Namespace) -> None:
    pipeline = build_trm_v2_pipeline(args)

    # ------------------------------------------------------------------
    # Batch mode: --input-file (and optionally --query as first sentence)
    # ------------------------------------------------------------------
    if args.input_file:
        sentences = _load_input_sentences(args)
        if not sentences:
            print("ERROR: --input-file produced no sentences.", file=sys.stderr)
            sys.exit(1)
        batch_results = []
        for q in sentences:
            result = pipeline.run(q, return_embedding=args.return_embedding)
            r = result.route
            entry: dict = {
                "query": q,
                "selected_domain": r.selected_domain_id,
                "novelty_decision": r.novelty_decision.value,
                "gate_confidence": round(r.gate_confidence, 4),
                "novelty_similarity": round(r.novelty_similarity, 4),
                "ood_fallback": r.ood_fallback,
                "halted": r.halt_state.halted,
                "halt_reason": r.halt_state.reason,
                "stored_query_id": result.stored_query_id,
                "drift_triggered": (
                    result.drift_signal.triggered if result.drift_signal else False
                ),
                "top_k_domains": r.top_k_domains,
            }
            if args.return_embedding:
                entry["embedding"] = r.embedding
            batch_results.append(entry)
        payload = json.dumps(batch_results, indent=2)
        if args.output:
            Path(args.output).write_text(payload, encoding="utf-8")
            print(f"TRM v2 batch output written to {args.output}")
        else:
            print(payload)
        pipeline._store.close_all()
        return

    # ------------------------------------------------------------------
    # Single-shot mode: --backend-only --query
    # ------------------------------------------------------------------
    if args.backend_only:
        if not args.query:
            print("ERROR: --query is required with --backend-only", file=sys.stderr)
            sys.exit(1)
        result = pipeline.run(args.query, return_embedding=args.return_embedding)
        output = {
            "query": args.query,
            "selected_domain": result.route.selected_domain_id,
            "novelty_decision": result.route.novelty_decision.value,
            "gate_confidence": round(result.route.gate_confidence, 4),
            "novelty_similarity": round(result.route.novelty_similarity, 4),
            "ood_fallback": result.route.ood_fallback,
            "halted": result.route.halt_state.halted,
            "halt_reason": result.route.halt_state.reason,
            "stored_query_id": result.stored_query_id,
            "drift_triggered": (
                result.drift_signal.triggered if result.drift_signal else False
            ),
            "top_k_domains": result.route.top_k_domains,
        }
        if args.return_embedding:
            output["embedding"] = result.route.embedding
        payload = json.dumps(output, indent=2)
        if args.output:
            Path(args.output).write_text(payload, encoding="utf-8")
            print(f"TRM v2 output written to {args.output}")
        else:
            print(payload)
        pipeline._store.close_all()
        return

    # ------------------------------------------------------------------
    # Interactive REPL
    # ------------------------------------------------------------------
    print("Mycelium TRM v2 — interactive mode. Type 'quit' to exit.\n")
    try:
        while True:
            try:
                query = input("query> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break
            if query.lower() in ("quit", "exit", "q"):
                break
            if not query:
                continue
            result = pipeline.run(query)
            r = result.route
            print(f"  \u2192 domain    : {r.selected_domain_id or 'OOD'}")
            print(f"    decision  : {r.novelty_decision.value}")
            print(f"    confidence: {r.gate_confidence:.4f}")
            print(f"    similarity: {r.novelty_similarity:.4f}")
            if result.drift_signal and result.drift_signal.triggered:
                print(f"  \u26a0  drift    : {result.drift_signal.reason}")
            print()
    finally:
        pipeline._store.close_all()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.enable_trm_v2:
        run_trm_v2(args)
    else:
        run_legacy(args)


if __name__ == "__main__":
    main()
