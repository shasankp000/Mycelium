"""
scripts/seed_sqlite_experts.py

Idempotent seeder for the TRM v2 per-domain SQLite expert shards.

What it does
------------
1. For each domain listed in DOMAINS, open (or create) a DomainShard at
   <shard_root>/<domain>.db.
2. Discover every CSV or JSONL file under data/<domain>/.
3. Insert every row as an ExpertRecord (INSERT OR IGNORE — safe to re-run).
4. Call QueryStore.index_domain() so the shard is registered in the
   central query-store index.

Supported dataset formats
-------------------------
CSV  — must contain at least one of: text, question, sentence, query,
        content, input.  Optional: label, confidence columns.
JSONL — one JSON object per line, same field names as CSV.

Usage
-----
  python scripts/seed_sqlite_experts.py
  python scripts/seed_sqlite_experts.py --shard-root data/shards --data-root data
  python scripts/seed_sqlite_experts.py --domains physics chemistry --verbose
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domains to seed.  Extend this list when new domains are added.
# ---------------------------------------------------------------------------
DEFAULT_DOMAINS: List[str] = ["physics", "chemistry"]

# Column names we accept as the primary text field (checked in order).
_TEXT_FIELDS = ("text", "question", "sentence", "query", "content", "input")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick_text(row: dict) -> Optional[str]:
    """Return the first non-empty recognised text field from *row*, or None."""
    for field in _TEXT_FIELDS:
        val = row.get(field) or row.get(field.upper()) or row.get(field.capitalize())
        if val and str(val).strip():
            return str(val).strip()
    return None


def _iter_csv(path: Path):
    """Yield dicts from a CSV file."""
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            yield dict(row)


def _iter_jsonl(path: Path):
    """Yield dicts from a JSONL (one JSON object per line) file."""
    with path.open(encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
                else:
                    logger.debug("%s:%d — skipping non-dict JSON value", path.name, lineno)
            except json.JSONDecodeError as exc:
                logger.warning("%s:%d — JSON parse error: %s", path.name, lineno, exc)


def _discover_datasets(data_root: Path, domain: str) -> List[Path]:
    """Return all CSV and JSONL files under data_root/<domain>/."""
    domain_dir = data_root / domain
    if not domain_dir.exists():
        logger.warning("Data directory '%s' does not exist — skipping domain.", domain_dir)
        return []
    files = sorted(
        p for p in domain_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in (".csv", ".jsonl", ".ndjson")
    )
    return files


# ---------------------------------------------------------------------------
# Core seeder
# ---------------------------------------------------------------------------

def seed_domain(
    domain: str,
    data_root: Path,
    shard_root: Path,
    verbose: bool = False,
) -> int:
    """
    Seed *domain* shard from all datasets found under data_root/<domain>/.
    Returns the number of rows inserted.
    """
    # Lazy import so the script can be imported without torch installed.
    try:
        from mycelium.trm.v2.store.shard import DomainShard
        from mycelium.trm.v2.store.query_store import QueryStore
    except ImportError as exc:
        logger.error(
            "Cannot import DomainShard / QueryStore: %s\n"
            "Make sure you are running from the project root with the venv active.",
            exc,
        )
        sys.exit(1)

    shard_root.mkdir(parents=True, exist_ok=True)
    shard_path = shard_root / f"{domain}.db"
    shard = DomainShard(domain_id=domain, db_path=str(shard_path))

    datasets = _discover_datasets(data_root, domain)
    if not datasets:
        logger.warning("No CSV/JSONL datasets found for domain '%s'.", domain)
        return 0

    total_inserted = 0

    for dataset_path in datasets:
        if verbose:
            print(f"  [{domain}] seeding from {dataset_path.relative_to(data_root.parent)} ...",
                  flush=True)

        suffix = dataset_path.suffix.lower()
        row_iter = _iter_csv(dataset_path) if suffix == ".csv" else _iter_jsonl(dataset_path)

        inserted = 0
        skipped = 0
        for row in row_iter:
            text = _pick_text(row)
            if not text:
                skipped += 1
                continue

            label_raw = row.get("label") or row.get("category") or ""
            conf_raw  = row.get("confidence") or row.get("score") or 1.0
            try:
                confidence = float(conf_raw)
            except (TypeError, ValueError):
                confidence = 1.0

            try:
                shard.insert(
                    text=text,
                    label=str(label_raw) if label_raw else domain,
                    confidence=confidence,
                    source=dataset_path.name,
                )
                inserted += 1
            except Exception as exc:
                logger.debug("Insert error (%s): %s", dataset_path.name, exc)
                skipped += 1

        total_inserted += inserted
        if verbose:
            print(f"     inserted={inserted}  skipped={skipped}")

    # Register in the central QueryStore index.
    try:
        store = QueryStore(shard_root=str(shard_root))
        store.index_domain(domain)
        store.close_all()
        if verbose:
            print(f"  [{domain}] QueryStore index updated.")
    except Exception as exc:
        logger.warning("QueryStore.index_domain('%s') failed: %s", domain, exc)

    return total_inserted


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed TRM v2 SQLite expert shards from local datasets."
    )
    parser.add_argument(
        "--domains", nargs="+", default=DEFAULT_DOMAINS,
        metavar="DOMAIN",
        help=f"Domains to seed (default: {DEFAULT_DOMAINS}).",
    )
    parser.add_argument(
        "--shard-root", default="data/shards",
        help="Root directory for SQLite shard files (default: data/shards).",
    )
    parser.add_argument(
        "--data-root", default="data",
        help="Root directory containing per-domain dataset folders (default: data).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-file progress.",
    )
    parser.add_argument(
        "--log-level", default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )

    data_root  = Path(args.data_root)
    shard_root = Path(args.shard_root)

    print(f"Seeding SQLite expert shards: {args.domains}")
    print(f"  data-root  : {data_root.resolve()}")
    print(f"  shard-root : {shard_root.resolve()}\n")

    grand_total = 0
    for domain in args.domains:
        print(f"\u25b6 {domain}")
        n = seed_domain(domain, data_root, shard_root, verbose=args.verbose)
        print(f"  \u2713 {domain}: {n} row(s) inserted")
        grand_total += n

    print(f"\n\u2705 Done. Total rows inserted: {grand_total}")


if __name__ == "__main__":
    main()
