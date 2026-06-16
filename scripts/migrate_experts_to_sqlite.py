#!/usr/bin/env python
"""
scripts/migrate_experts_to_sqlite.py

Offline migration script: reads the legacy UnifiedBERTExpert knowledge stores
and writes one SQLite shard per domain under artifacts/sqlite_experts/.

This script is deliberately read-only from the legacy side — it never modifies
or deletes any existing expert files.

Usage
-----
    # Migrate all domains discovered under [experts].dir
    python scripts/migrate_experts_to_sqlite.py

    # Migrate specific domains only
    python scripts/migrate_experts_to_sqlite.py --domains physics chemistry

    # Dry-run: show what would be migrated without writing anything
    python scripts/migrate_experts_to_sqlite.py --dry-run

    # Validate existing shards without re-migrating
    python scripts/migrate_experts_to_sqlite.py --validate-only

Output
------
    artifacts/sqlite_experts/<domain_id>.sqlite   — one shard per domain
    artifacts/sqlite_experts/migration_report.json — summary report
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
import uuid
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("migrate_experts_to_sqlite")

# ---------------------------------------------------------------------------
# Repo root
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from mycelium.pipeline.config_loader import load_trm_v2_config, discover_live_domains  # noqa: E402

# ---------------------------------------------------------------------------
# Minimal inline schema (identical to domain_store.schema.SCHEMA_DDL)
# ---------------------------------------------------------------------------
_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS chunks (
    id           TEXT PRIMARY KEY,
    text         TEXT NOT NULL,
    embedding    BLOB,
    source       TEXT,
    created_at   TEXT,
    updated_at   TEXT,
    metadata_json TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
    USING fts5(id UNINDEXED, text, content='chunks', content_rowid='rowid');
CREATE TABLE IF NOT EXISTS entities (
    entity_id    TEXT PRIMARY KEY,
    label        TEXT NOT NULL,
    entity_type  TEXT,
    metadata_json TEXT
);
CREATE TABLE IF NOT EXISTS relations (
    relation_id  TEXT PRIMARY KEY,
    src_id       TEXT NOT NULL,
    rel_type     TEXT NOT NULL,
    dst_id       TEXT NOT NULL,
    weight       REAL DEFAULT 1.0,
    metadata_json TEXT
);
CREATE TABLE IF NOT EXISTS replay_samples (
    sample_id    TEXT PRIMARY KEY,
    text         TEXT NOT NULL,
    label_json   TEXT,
    embedding    BLOB,
    metadata_json TEXT
);
CREATE TABLE IF NOT EXISTS shard_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


# ---------------------------------------------------------------------------
# Shard helpers
# ---------------------------------------------------------------------------

def _init_shard(shard_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(shard_path)
    for stmt in _SCHEMA_DDL.split(";"):
        stmt = stmt.strip()
        if not stmt:
            continue
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError as exc:
            if "fts5" in str(exc).lower():
                log.warning("FTS5 unavailable (%s) — virtual table skipped", exc)
            else:
                raise
    conn.commit()
    return conn


def _set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO shard_meta (key, value) VALUES (?, ?)", (key, value))


def _insert_chunk(
    conn: sqlite3.Connection,
    text: str,
    source: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    chunk_id = str(uuid.uuid4())
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    conn.execute(
        "INSERT OR REPLACE INTO chunks "
        "(id, text, source, created_at, updated_at, metadata_json) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (chunk_id, text, source, now, now, json.dumps(metadata or {})),
    )
    return chunk_id


def _validate_shard(shard_path: Path) -> dict[str, Any]:
    """Return a lightweight validation report for an existing shard."""
    if not shard_path.exists():
        return {"ok": False, "error": "file_not_found", "path": str(shard_path)}
    try:
        conn = sqlite3.connect(shard_path)
        chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        meta = dict(conn.execute("SELECT key, value FROM shard_meta").fetchall())
        conn.close()
        return {
            "ok": True,
            "path": str(shard_path),
            "chunk_count": chunk_count,
            "meta": meta,
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "path": str(shard_path)}


# ---------------------------------------------------------------------------
# Legacy expert corpus extraction
# ---------------------------------------------------------------------------

def _extract_corpus_via_migrator(
    domain_id: str,
    experts_root: Path,
) -> list[dict[str, Any]]:
    """Try to use LegacyExpertMigrator; fall back to directory scan."""
    try:
        from mycelium.domain_store.migration import LegacyExpertMigrator  # noqa: PLC0415
        from mycelium.domain_store.embedder import OfflineEmbedder         # noqa: PLC0415
        embedder = OfflineEmbedder()
        migrator = LegacyExpertMigrator(
            bert_expert=None,  # migrator uses file-based extraction path
            store_root=experts_root.parent / "sqlite_experts",
            embedder=embedder,
        )
        return migrator.export_domain_corpus(domain_id)
    except (ImportError, Exception) as exc:  # noqa: BLE001
        log.debug("LegacyExpertMigrator unavailable (%s); using directory scan.", exc)

    # Fallback: scan domain directory for .txt / .json files
    domain_dir = experts_root / domain_id
    if not domain_dir.is_dir():
        log.warning("No directory found for domain '%s' under '%s'", domain_id, experts_root)
        return []

    chunks: list[dict[str, Any]] = []
    for txt_file in sorted(domain_dir.rglob("*.txt")):
        text = txt_file.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            chunks.append({"text": text, "source": str(txt_file.relative_to(domain_dir))})

    for json_file in sorted(domain_dir.rglob("*.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "text" in item:
                        chunks.append({
                            "text": item["text"],
                            "source": str(json_file.relative_to(domain_dir)),
                            "metadata": {k: v for k, v in item.items() if k != "text"},
                        })
            elif isinstance(data, dict) and "text" in data:
                chunks.append({
                    "text": data["text"],
                    "source": str(json_file.relative_to(domain_dir)),
                })
        except Exception:  # noqa: BLE001
            pass

    return chunks


# ---------------------------------------------------------------------------
# Per-domain migration
# ---------------------------------------------------------------------------

def _migrate_domain(
    domain_id: str,
    experts_root: Path,
    shard_dir: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    shard_path = shard_dir / f"{domain_id}.sqlite"
    corpus = _extract_corpus_via_migrator(domain_id, experts_root)

    report: dict[str, Any] = {
        "domain_id": domain_id,
        "shard_path": str(shard_path),
        "chunks_found": len(corpus),
        "chunks_written": 0,
        "dry_run": dry_run,
        "ok": True,
        "error": None,
    }

    if dry_run:
        log.info("[DRY RUN] %s: would write %d chunks to %s",
                 domain_id, len(corpus), shard_path)
        return report

    try:
        shard_dir.mkdir(parents=True, exist_ok=True)
        conn = _init_shard(shard_path)
        _set_meta(conn, "domain_id", domain_id)
        _set_meta(conn, "schema_version", "1.0")
        _set_meta(conn, "migrated_at",
                  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        _set_meta(conn, "source", "legacy_expert_migration")

        for chunk in corpus:
            _insert_chunk(
                conn,
                text=chunk.get("text", ""),
                source=chunk.get("source", ""),
                metadata=chunk.get("metadata"),
            )

        conn.commit()
        conn.close()
        report["chunks_written"] = len(corpus)
        log.info("Migrated '%s': %d chunk(s) → %s", domain_id, len(corpus), shard_path)
    except Exception as exc:  # noqa: BLE001
        report["ok"] = False
        report["error"] = str(exc)
        log.error("Failed to migrate '%s': %s", domain_id, exc)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="migrate_experts_to_sqlite",
        description="Migrate legacy expert stores to SQLite shards for TRM v2.",
    )
    p.add_argument("--domains", nargs="+", default=None,
                   help="Domain IDs to migrate. Defaults to all discovered domains.")
    p.add_argument("--experts-dir", type=Path, default=None,
                   help="Override the [experts].dir path from config.toml.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Override the [sqlite_experts].root_dir from config.toml.")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be migrated without writing any files.")
    p.add_argument("--validate-only", action="store_true",
                   help="Validate existing shards and print a report without migrating.")
    p.add_argument("--report-path", type=Path, default=None,
                   help="Write JSON migration report to this path.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    trm_cfg = load_trm_v2_config()

    # Resolve paths
    from mycelium.pipeline.config_loader import experts_dir  # noqa: PLC0415
    experts_root = args.experts_dir or Path(experts_dir())
    shard_dir = args.output_dir or Path(trm_cfg.sqlite_experts.root_dir)

    # Resolve domain list
    if args.domains:
        domains = [d.lower().strip() for d in args.domains]
    else:
        try:
            domains = discover_live_domains()
        except Exception as exc:  # noqa: BLE001
            log.warning("discover_live_domains() failed (%s); "
                        "scanning experts_root directly.", exc)
            if experts_root.is_dir():
                domains = [d.name for d in sorted(experts_root.iterdir()) if d.is_dir()]
            else:
                log.error("experts_root '%s' is not a directory.", experts_root)
                return 1

    if not domains:
        log.error("No domains found to migrate.")
        return 1

    log.info("Domains to process: %s", domains)

    # --- Validate-only mode ---
    if args.validate_only:
        reports = []
        for domain_id in domains:
            shard_path = shard_dir / f"{domain_id}.sqlite"
            rep = _validate_shard(shard_path)
            rep["domain_id"] = domain_id
            reports.append(rep)
            status = "OK" if rep["ok"] else "FAIL"
            log.info("[%s] %s — chunks=%s",
                     status, domain_id, rep.get("chunk_count", "N/A"))
        _write_report(reports, args.report_path, shard_dir)
        return 0

    # --- Migration ---
    reports = []
    for domain_id in domains:
        rep = _migrate_domain(
            domain_id=domain_id,
            experts_root=experts_root,
            shard_dir=shard_dir,
            dry_run=args.dry_run,
        )
        reports.append(rep)

    # Summary
    ok_count = sum(1 for r in reports if r["ok"])
    fail_count = len(reports) - ok_count
    total_chunks = sum(r.get("chunks_written", 0) for r in reports)

    log.info("=" * 60)
    log.info("Migration complete: %d ok, %d failed, %d total chunks written",
             ok_count, fail_count, total_chunks)
    if args.dry_run:
        log.info("(dry-run — no files were written)")

    _write_report(reports, args.report_path, shard_dir)
    return 0 if fail_count == 0 else 1


def _write_report(
    reports: list[dict],
    report_path: Path | None,
    shard_dir: Path,
) -> None:
    if report_path is None:
        report_path = shard_dir / "migration_report.json"
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
        log.info("Report written to %s", report_path)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not write report: %s", exc)


if __name__ == "__main__":
    sys.exit(main())
