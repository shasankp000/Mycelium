#!/usr/bin/env python
"""
scripts/onboard_new_domain.py

CLI tool to onboard a new domain into the TRM v2 system.

What it does
------------
1. Validates the domain_id (slug format, no collisions with the existing graph).
2. Creates the SQLite shard file at artifacts/sqlite_experts/<domain_id>.sqlite.
3. Ingests seed documents supplied via --seed-dir or --seed-file (optional).
4. Registers the new DomainNode in the DomainGraphRegistry with state=CREATING.
5. Snapshots the updated graph to the configured graph_path.
6. Prints a summary and next-step instructions.

Usage
-----
    python scripts/onboard_new_domain.py \\
        --domain-id astrophysics \\
        --label "Astrophysics" \\
        --seed-dir data/astrophysics_seed \\
        --creation-reason "New domain requested by team" \\
        --created-by "alice"

    # Dry-run (validate only, no writes)
    python scripts/onboard_new_domain.py --domain-id astrophysics --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import sys
import time
import uuid
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("onboard_new_domain")

# ---------------------------------------------------------------------------
# Repo root resolution  (scripts/ lives one level below repo root)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from mycelium.pipeline.config_loader import load_trm_v2_config  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_SLUG_RE = re.compile(r"^[a-z][a-z0-9_-]{0,62}[a-z0-9]$|^[a-z]$")


def _validate_domain_id(domain_id: str) -> None:
    if not _SLUG_RE.match(domain_id):
        raise ValueError(
            f"Invalid domain_id '{domain_id}'. "
            "Must be lowercase, start with a letter, 1-64 chars, "
            "alphanumeric / hyphens / underscores only."
        )


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _create_shard(shard_path: Path) -> None:
    """Initialise an empty SQLite shard with the TRM v2 schema."""
    # Import here so the script works even if the full package isn't installed yet.
    try:
        from mycelium.domain_store.schema import SCHEMA_DDL  # noqa: PLC0415
    except ImportError:
        # Minimal inline schema for bootstrap scenarios
        SCHEMA_DDL = """
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY, text TEXT NOT NULL, embedding BLOB,
            source TEXT, created_at TEXT, updated_at TEXT, metadata_json TEXT
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
            USING fts5(id UNINDEXED, text, content='chunks', content_rowid='rowid');
        CREATE TABLE IF NOT EXISTS entities (
            entity_id TEXT PRIMARY KEY, label TEXT NOT NULL,
            entity_type TEXT, metadata_json TEXT
        );
        CREATE TABLE IF NOT EXISTS relations (
            relation_id TEXT PRIMARY KEY, src_id TEXT NOT NULL,
            rel_type TEXT NOT NULL, dst_id TEXT NOT NULL,
            weight REAL DEFAULT 1.0, metadata_json TEXT
        );
        CREATE TABLE IF NOT EXISTS replay_samples (
            sample_id TEXT PRIMARY KEY, text TEXT NOT NULL,
            label_json TEXT, embedding BLOB, metadata_json TEXT
        );
        CREATE TABLE IF NOT EXISTS shard_meta (
            key TEXT PRIMARY KEY, value TEXT
        );
        """

    conn = sqlite3.connect(shard_path)
    try:
        for stmt in SCHEMA_DDL.split(";"):
            stmt = stmt.strip()
            if stmt:
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError as exc:
                    # FTS5 may not be available in all SQLite builds
                    if "fts5" in str(exc).lower():
                        log.warning("FTS5 not available: %s — skipping virtual table", exc)
                    else:
                        raise
        conn.commit()
    finally:
        conn.close()
    log.info("Created shard: %s", shard_path)


def _set_shard_meta(shard_path: Path, domain_id: str, label: str) -> None:
    conn = sqlite3.connect(shard_path)
    try:
        for k, v in [
            ("domain_id", domain_id),
            ("label", label),
            ("schema_version", "1.0"),
            ("created_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
        ]:
            conn.execute(
                "INSERT OR REPLACE INTO shard_meta (key, value) VALUES (?, ?)", (k, v)
            )
        conn.commit()
    finally:
        conn.close()


def _ingest_seed_dir(shard_path: Path, seed_dir: Path) -> int:
    """Ingest plain-text .txt files from seed_dir into the shard. Returns chunk count."""
    if not seed_dir.is_dir():
        log.warning("--seed-dir '%s' does not exist or is not a directory — skipping.", seed_dir)
        return 0

    try:
        from mycelium.domain_store.store import SQLiteDomainStore  # noqa: PLC0415
        store = SQLiteDomainStore(shard_path)
        use_store = True
    except ImportError:
        use_store = False
        conn = sqlite3.connect(shard_path)

    count = 0
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for txt_file in sorted(seed_dir.rglob("*.txt")):
        text = txt_file.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            continue
        chunk_id = str(uuid.uuid4())
        if use_store:
            store.upsert_chunk({
                "id": chunk_id,
                "text": text,
                "source": str(txt_file.relative_to(seed_dir)),
                "created_at": now,
                "updated_at": now,
            })
        else:
            conn.execute(
                "INSERT OR REPLACE INTO chunks "
                "(id, text, source, created_at, updated_at) VALUES (?,?,?,?,?)",
                (chunk_id, text, str(txt_file.relative_to(seed_dir)), now, now),
            )
        count += 1

    if not use_store:
        conn.commit()
        conn.close()

    log.info("Ingested %d seed document(s) from '%s'", count, seed_dir)
    return count


def _register_domain_node(
    graph_path: Path,
    domain_id: str,
    label: str,
    shard_path: Path,
    creation_reason: str,
    created_by: str,
) -> dict:
    """Add DomainNode to the JSON graph, or create the graph file if absent."""
    try:
        from mycelium.domain_graph.models import DomainNode  # noqa: PLC0415
        from mycelium.domain_graph.registry import DomainGraphRegistry  # noqa: PLC0415
        registry = DomainGraphRegistry(graph_path=graph_path)
        node = DomainNode(
            domain_id=domain_id,
            label=label,
            state="CREATING",
            sqlite_shard_path=str(shard_path),
            creation_reason=creation_reason,
            created_by=created_by,
            creation_timestamp=time.time(),
            updated_at=time.time(),
        )
        registry.add_domain(node)
        snapshot_path = registry.snapshot()
        log.info("DomainGraphRegistry snapshot written to %s", snapshot_path)
        return {"registered": True, "via": "DomainGraphRegistry"}
    except ImportError:
        # Fallback: raw JSON manipulation when domain_graph module not yet importable
        if graph_path.exists():
            graph = json.loads(graph_path.read_text(encoding="utf-8"))
        else:
            graph = {"schema_version": "1.0", "nodes": {}, "edges": []}
        graph["nodes"][domain_id] = {
            "domain_id": domain_id,
            "label": label,
            "state": "CREATING",
            "sqlite_shard_path": str(shard_path),
            "creation_reason": creation_reason,
            "created_by": created_by,
            "creation_timestamp": time.time(),
            "updated_at": time.time(),
        }
        _ensure_dir(graph_path.parent)
        graph_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")
        log.info("Raw JSON graph updated at %s (DomainGraphRegistry not available)", graph_path)
        return {"registered": True, "via": "raw_json"}


def _check_collision(graph_path: Path, domain_id: str) -> bool:
    """Return True if domain_id already exists in the graph."""
    if not graph_path.exists():
        return False
    try:
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        return domain_id in graph.get("nodes", {})
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="onboard_new_domain",
        description="Onboard a new domain into the TRM v2 system.",
    )
    p.add_argument("--domain-id", required=True, help="Slug identifier for the new domain.")
    p.add_argument("--label", default="", help="Human-readable label (defaults to domain_id).")
    p.add_argument("--seed-dir", type=Path, default=None,
                   help="Directory of .txt seed documents to ingest.")
    p.add_argument("--seed-file", type=Path, default=None,
                   help="Single .txt file to ingest as the seed document.")
    p.add_argument("--creation-reason", default="",
                   help="Human note explaining why this domain is being created.")
    p.add_argument("--created-by", default="system",
                   help="Username or system name creating the domain.")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate inputs and print plan without writing anything.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing shard and graph entry if domain already exists.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    domain_id: str = args.domain_id.lower().strip()
    label: str = args.label or domain_id

    # --- Validate ---
    try:
        _validate_domain_id(domain_id)
    except ValueError as exc:
        log.error("%s", exc)
        return 1

    trm_cfg = load_trm_v2_config()
    sqlite_root = Path(trm_cfg.sqlite_experts.root_dir)
    graph_path = Path(trm_cfg.domain_graph.graph_path)
    shard_path = sqlite_root / f"{domain_id}.sqlite"

    # Collision check
    if not args.force and _check_collision(graph_path, domain_id):
        log.error(
            "Domain '%s' already exists in %s. Use --force to overwrite.",
            domain_id, graph_path,
        )
        return 1

    log.info("=" * 60)
    log.info("Onboarding domain: %s (%s)", domain_id, label)
    log.info("  shard_path   : %s", shard_path)
    log.info("  graph_path   : %s", graph_path)
    log.info("  dry_run      : %s", args.dry_run)
    log.info("=" * 60)

    if args.dry_run:
        log.info("[DRY RUN] No files written.")
        return 0

    # --- Create shard ---
    _ensure_dir(sqlite_root)
    _create_shard(shard_path)
    _set_shard_meta(shard_path, domain_id, label)

    # --- Seed ingestion ---
    if args.seed_dir:
        _ingest_seed_dir(shard_path, args.seed_dir)
    if args.seed_file and args.seed_file.is_file():
        seed_tmp = shard_path.parent / "_seed_tmp"
        seed_tmp.mkdir(exist_ok=True)
        (seed_tmp / args.seed_file.name).write_bytes(args.seed_file.read_bytes())
        _ingest_seed_dir(shard_path, seed_tmp)
        import shutil
        shutil.rmtree(seed_tmp, ignore_errors=True)

    # --- Register in graph ---
    result = _register_domain_node(
        graph_path=graph_path,
        domain_id=domain_id,
        label=label,
        shard_path=shard_path,
        creation_reason=args.creation_reason,
        created_by=args.created_by,
    )

    log.info("Onboarding complete.")
    log.info("  Registration method : %s", result.get("via"))
    log.info("")
    log.info("Next steps:")
    log.info("  1. Add seed documents:  python scripts/onboard_new_domain.py "
             "--domain-id %s --seed-dir data/%s_seed", domain_id, domain_id)
    log.info("  2. Set state HOT once trained:")
    log.info("     from mycelium.domain_graph.registry import DomainGraphRegistry")
    log.info("     reg = DomainGraphRegistry(graph_path='%s')", graph_path)
    log.info("     reg.mark_state('%s', 'HOT')", domain_id)
    log.info("  3. Enable TRM v2 in config.toml: [trm_v2] enabled = true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
