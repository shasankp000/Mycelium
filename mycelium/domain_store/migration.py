from __future__ import annotations
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from mycelium.domain_store.store import SQLiteDomainStore
from mycelium.domain_store.ingest import DomainIngestor
from mycelium.domain_store.embedder import OfflineEmbedder

logger = logging.getLogger(__name__)


class LegacyExpertMigrator:
    """
    Reads knowledge from the old BERT-based expert flat files and writes
    them into SQLite domain shards.

    This is a READ-ONLY consumer of the legacy system — it never imports
    UnifiedBERTExpert or any model code. It only reads the text corpus
    files that experts were trained on.

    Usage (offline script only):
        migrator = LegacyExpertMigrator(
            legacy_expert_dir="artifacts/experts",
            sqlite_root="artifacts/sqlite_experts",
        )
        migrator.migrate_all()
    """

    def __init__(
        self,
        legacy_expert_dir: str,
        sqlite_root: str,
        embedder: Optional[OfflineEmbedder] = None,
        chunk_size: int = 512,
        overlap: int = 64,
        dry_run: bool = False,
    ) -> None:
        self._legacy_dir = Path(legacy_expert_dir)
        self._sqlite_root = sqlite_root
        self._embedder = embedder
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._dry_run = dry_run

    def discover_domains(self) -> List[str]:
        """
        Returns domain_ids inferred from the legacy expert directory layout.
        Expects: artifacts/experts/{domain_id}/ or artifacts/experts/{domain_id}.json
        """
        if not self._legacy_dir.exists():
            logger.warning("Legacy expert dir not found: %s", self._legacy_dir)
            return []
        domains = []
        for entry in sorted(self._legacy_dir.iterdir()):
            if entry.is_dir():
                domains.append(entry.name)
            elif entry.suffix == ".json":
                domains.append(entry.stem)
        logger.info("Discovered %d legacy domain(s): %s", len(domains), domains)
        return domains

    def migrate_domain(self, domain_id: str) -> int:
        """
        Migrates one domain. Returns number of chunks written.
        """
        corpus_texts = self._load_corpus(domain_id)
        if not corpus_texts:
            logger.warning("No corpus found for domain '%s' — skipping.", domain_id)
            return 0

        if self._dry_run:
            total_chars = sum(len(t) for t in corpus_texts)
            logger.info("[dry-run] Would migrate domain '%s': %d texts, ~%d chars",
                        domain_id, len(corpus_texts), total_chars)
            return 0

        store = SQLiteDomainStore(self._sqlite_root, domain_id)
        ingestor = DomainIngestor(
            store=store,
            embedder=self._embedder,
            chunk_size=self._chunk_size,
            overlap=self._overlap,
        )
        total_chunks = 0
        for idx, text in enumerate(corpus_texts):
            ids = ingestor.ingest_text(text, source=f"legacy:{domain_id}:doc{idx}")
            total_chunks += len(ids)
        store.set_meta("migrated_from", "legacy_bert_expert")
        store.set_meta("migration_chunk_count", str(total_chunks))
        store.close()
        logger.info("Migrated domain '%s': %d chunks written.", domain_id, total_chunks)
        return total_chunks

    def migrate_all(self) -> Dict[str, int]:
        """Migrates all discovered domains. Returns {domain_id: chunk_count}."""
        results: Dict[str, int] = {}
        for domain_id in self.discover_domains():
            results[domain_id] = self.migrate_domain(domain_id)
        return results

    # ------------------------------------------------------------------
    # Corpus loading — extend this as the actual legacy layout is known
    # ------------------------------------------------------------------

    def _load_corpus(self, domain_id: str) -> List[str]:
        texts: List[str] = []

        # Layout 1: artifacts/experts/{domain_id}/corpus.txt
        txt_path = self._legacy_dir / domain_id / "corpus.txt"
        if txt_path.exists():
            texts.append(txt_path.read_text(encoding="utf-8", errors="replace"))

        # Layout 2: artifacts/experts/{domain_id}/*.txt
        txt_dir = self._legacy_dir / domain_id
        if txt_dir.is_dir():
            for f in sorted(txt_dir.glob("*.txt")):
                if f != txt_path:
                    texts.append(f.read_text(encoding="utf-8", errors="replace"))

        # Layout 3: artifacts/experts/{domain_id}.json → {"texts": [...]}
        json_path = self._legacy_dir / f"{domain_id}.json"
        if json_path.exists():
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    texts.extend(str(x) for x in data)
                elif isinstance(data, dict):
                    for key in ("texts", "corpus", "documents", "data"):
                        if key in data and isinstance(data[key], list):
                            texts.extend(str(x) for x in data[key])
                            break
            except Exception as exc:
                logger.warning("Failed to parse %s: %s", json_path, exc)

        return texts
