"""
run_workflow.py — TRM v2 CLI entrypoint.

Usage examples:
  # Interactive REPL (legacy path, TRM v2 disabled)
  python run_workflow.py

  # Backend-only single query, legacy path
  python run_workflow.py --backend-only --query "Is 9.9 bigger than 9.11?"

  # Enable TRM v2 pipeline (feature flag)
  python run_workflow.py --enable-trm-v2 --backend-only --query "What is calculus?"

  # Specify domain graph path and shard root
  python run_workflow.py --enable-trm-v2 --graph-path data/graph.json \
      --shard-root data/shards --backend-only --query "Newton's laws"
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from typing import Optional

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_workflow",
        description="Mycelium TRM inference workflow runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--enable-trm-v2", action="store_true", default=False,
                   help="Enable the TRM v2 pipeline. Off by default — legacy path runs instead.")
    p.add_argument("--backend-only", action="store_true", default=False,
                   help="Run a single query and print JSON result. No REPL.")
    p.add_argument("--query", "-q", type=str, default=None,
                   help="Query string (required when --backend-only is set).")
    p.add_argument("--graph-path", type=str, default="data/domain_graph.json",
                   help="Path to the domain graph JSON file.")
    p.add_argument("--shard-root", type=str, default="data/shards",
                   help="Root directory for per-domain SQLite shards.")
    p.add_argument("--artifact-dir", type=str, default="artifacts/trm_v2/heads",
                   help="Root directory for trained head artifacts.")
    p.add_argument("--log-level", type=str, default="WARNING",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--return-embedding", action="store_true", default=False,
                   help="Include the query embedding vector in JSON output.")
    return p


def run_legacy(query: Optional[str], backend_only: bool) -> None:
    """
    Delegates entirely to the existing pipeline — this is the ONLY place
    legacy code is imported. Delete this function when TRM v2 reaches parity.
    """
    try:
        from mycelium.pipeline.run_workflow import run as legacy_run  # type: ignore
        legacy_run(query=query, backend_only=backend_only)
    except ImportError:
        logger.error("Legacy pipeline not available. Use --enable-trm-v2.")
        sys.exit(1)


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
        logger.warning("TRMNetwork not available — using EmbeddingBag stub (vocab=30522, DIM=128).")
        output_dim = 128
        # bert-base-uncased vocab size = 30522
        backbone = nn.EmbeddingBag(30522, output_dim, mode="mean", sparse=False)

    encoder = SharedEncoder(backbone, output_dim=output_dim, frozen=True)

    head_registry = HeadRegistry()
    _load_heads_from_artifacts(head_registry, args.artifact_dir, output_dim, registry)

    # --- Tokenizer ---
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        logger.info("Loaded bert-base-uncased tokenizer.")
    except Exception as exc:
        logger.warning("Tokenizer unavailable (%s) — pass pre-tokenised input_ids manually.", exc)

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
            head = DomainHead(domain_id, input_dim=entry.input_dim,
                              hidden_dim=entry.hidden_dim, output_dim=entry.output_dim)
            load_head(head, entry.artifact_path)
            head_registry.register(head)
            logger.info("Loaded head for '%s'.", domain_id)
        except Exception as exc:
            logger.warning("Failed to load head for '%s': %s", domain_id, exc)


def run_trm_v2(args: argparse.Namespace) -> None:
    pipeline = build_trm_v2_pipeline(args)

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
            "drift_triggered": result.drift_signal.triggered if result.drift_signal else False,
            "top_k_domains": result.route.top_k_domains,
        }
        if args.return_embedding:
            output["embedding"] = result.route.embedding
        print(json.dumps(output, indent=2))
        pipeline._store.close_all()
        return

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
            print(f"  → domain    : {r.selected_domain_id or 'OOD'}")
            print(f"    decision  : {r.novelty_decision.value}")
            print(f"    confidence: {r.gate_confidence:.4f}")
            print(f"    similarity: {r.novelty_similarity:.4f}")
            if result.drift_signal and result.drift_signal.triggered:
                print(f"  ⚠  drift    : {result.drift_signal.reason}")
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
        run_legacy(args.query, args.backend_only)


if __name__ == "__main__":
    main()
