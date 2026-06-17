"""
DynamicSignatureManager
=======================
Keeps spectral signatures in sync with the live expert registry.

Call sync_signatures(registered_domains) once at startup (after
UnifiedExpertSystem initialises).  Returns an up-to-date
RuntimeSpectralAnalyzer ready for the MultiLensRouter to use.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from mycelium.pipeline.spectral_analyzer import SpectralSignatureGenerator, RuntimeSpectralAnalyzer

# ---------------------------------------------------------------------------
# Project-root-relative path helper (plan §5.2)
# ---------------------------------------------------------------------------
# This file lives at mycelium/pipeline/dynamic_signature_manager.py
# so parents[2] resolves to the project root.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Domain -> corpus source mapping — loaded from domain_corpora.json
# ---------------------------------------------------------------------------
# domain_corpora.json lives in the same directory as this file.
# To add a new domain or update corpus paths, edit that file — no source
# changes required.
# ---------------------------------------------------------------------------

_DOMAIN_CORPORA_CONFIG = Path(__file__).resolve().parent / "domain_corpora.json"


def active_domains() -> List[str]:
    """
    Return the currently live domain list from config_loader.discover_live_domains().
    This is the single source of truth for which domains need signatures.
    Falls back to domain_corpora.json keys if config_loader is unavailable.
    """
    try:
        from mycelium.pipeline import config_loader as _cfg
        return _cfg.discover_live_domains()
    except Exception:
        pass
    # secondary fallback: keys from domain_corpora.json
    src = _load_domain_corpus_sources()
    return sorted(src.keys())


def _load_domain_corpus_sources() -> Dict[str, Dict]:
    """Load domain corpus configuration from domain_corpora.json.
    Falls back to an empty dict if the file is missing — unknown domains
    will receive generic placeholder sentences from build_corpus_for_domain.
    """
    if not _DOMAIN_CORPORA_CONFIG.exists():
        import logging as _log
        _log.getLogger(__name__).warning(
            "domain_corpora.json not found at %s — all domains will use generic fallback texts.",
            _DOMAIN_CORPORA_CONFIG,
        )
        return {}
    try:
        import json as _json
        data = _json.loads(_DOMAIN_CORPORA_CONFIG.read_text(encoding="utf-8"))
        # Resolve relative CSV paths to absolute using _PROJECT_ROOT
        for domain, cfg in data.items():
            csv_path = cfg.get("csv")
            if csv_path and not Path(csv_path).is_absolute():
                cfg["csv"] = str(_PROJECT_ROOT / csv_path)
        return data
    except Exception as exc:
        import logging as _log
        _log.getLogger(__name__).error(
            "Failed to load domain_corpora.json: %s", exc
        )
        return {}


DOMAIN_CORPUS_SOURCES: Dict[str, Dict] = _load_domain_corpus_sources()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_lfs_pointer(path: str) -> bool:
    """Return True if the file is a Git LFS pointer rather than real data."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.readline().strip().startswith(
                "version https://git-lfs.github.com/spec/v1"
            )
    except Exception:
        return True


def build_corpus_for_domain(domain: str) -> List[str]:
    """
    Return a list of representative texts for *domain*.

    Resolution order:
    1. CSV file defined in DOMAIN_CORPUS_SOURCES (if present and not LFS).
    2. Hardcoded fallback_texts in DOMAIN_CORPUS_SOURCES.
    3. Generic placeholder sentences (unknown / future domains).
    """
    config = DOMAIN_CORPUS_SOURCES.get(domain.lower())

    if config is None:
        return [
            f"This is a question about {domain}.",
            f"{domain} is an area of scientific and academic study.",
            f"Research in {domain} involves systematic inquiry and analysis.",
            f"Key concepts in {domain} are studied by domain experts.",
            f"Advanced topics in {domain} require specialised knowledge.",
        ]

    csv_path: Optional[str] = config.get("csv")
    column: Optional[str] = config.get("column")
    sample_size: int = config.get("sample_size", 50)

    if csv_path and column and Path(csv_path).exists() and not _is_lfs_pointer(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if column in df.columns:
                available = df[column].dropna()
                texts = (
                    available
                    .sample(min(sample_size, len(available)), random_state=42)
                    .tolist()
                )
                if texts:
                    return texts
        except Exception as exc:
            print(f"\u26a0\ufe0f  Could not read corpus CSV for '{domain}': {exc}")

    fallback = config.get("fallback_texts", [])
    if fallback:
        return fallback

    return [
        f"{domain} involves the study of complex systems and phenomena.",
        f"Experts in {domain} apply rigorous methods to solve problems.",
        f"The field of {domain} has many open research questions.",
        f"{domain} intersects with mathematics, logic, and empirical science.",
        f"Core principles of {domain} are taught at university level.",
    ]


# ---------------------------------------------------------------------------
# DynamicSignatureManager
# ---------------------------------------------------------------------------

class DynamicSignatureManager:
    """Keeps spectral signatures in sync with the live expert registry."""

    MANIFEST_FILENAME = "manifest.json"

    def __init__(
        self,
        signature_dir: str = "signatures",
        model_name: str = "all-MiniLM-L6-v2",
        force_regen: bool = False,
    ) -> None:
        self.signature_dir = Path(signature_dir)
        self.signature_dir.mkdir(exist_ok=True)
        self.model_name = model_name
        self.force_regen = force_regen

        self._generator = SpectralSignatureGenerator(
            model_name=model_name,
            signature_dir=signature_dir,
        )
        self._manifest: Dict[str, str] = self._load_manifest()

    @property
    def _manifest_path(self) -> Path:
        return self.signature_dir / self.MANIFEST_FILENAME

    def _load_manifest(self) -> Dict[str, str]:
        if self._manifest_path.exists():
            try:
                return json.loads(self._manifest_path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"\u26a0\ufe0f  Could not read signature manifest: {exc}")
        return {}

    def _save_manifest(self) -> None:
        try:
            self._manifest_path.write_text(
                json.dumps(self._manifest, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            print(f"\u26a0\ufe0f  Could not save signature manifest: {exc}")

    @staticmethod
    def _corpus_hash(texts: List[str]) -> str:
        payload = "|||".join(sorted(texts)).encode("utf-8")
        return hashlib.md5(payload).hexdigest()[:12]

    def _signature_path(self, domain: str) -> Path:
        return self.signature_dir / f"{domain}_spectral_signature.npy"

    def _signature_exists(self, domain: str) -> bool:
        return self._signature_path(domain).exists()

    def _needs_regen(self, domain: str, corpus_hash: str) -> bool:
        if self.force_regen:
            return True
        if not self._signature_exists(domain):
            return True
        return self._manifest.get(domain) != corpus_hash

    def sync_signatures(self, registered_domains: Set[str]) -> RuntimeSpectralAnalyzer:
        """
        Ensure every domain in *registered_domains* has a current spectral
        signature on disk, then return a hot-loaded RuntimeSpectralAnalyzer.
        """
        domains_to_generate: Dict[str, List[str]] = {}

        for domain in sorted(registered_domains):
            corpus = build_corpus_for_domain(domain)
            h = self._corpus_hash(corpus)

            if self._needs_regen(domain, h):
                print(f"\U0001f504 Queuing spectral signature generation for: {domain}")
                domains_to_generate[domain] = corpus
                self._manifest[domain] = h
            else:
                print(f"\u2705 Spectral signature up-to-date: {domain}")

        results: Dict[str, Dict] = {}
        if domains_to_generate:
            print(
                f"\n\U0001f9ec Generating {len(domains_to_generate)} spectral signature(s): "
                f"{sorted(domains_to_generate.keys())}\n"
            )
            results = self._generator.generate_domain_signatures(domains_to_generate)

            for domain, result in results.items():
                if result.get("status") != "saved":
                    self._manifest.pop(domain, None)
                    print(
                        f"\u274c Signature generation failed for '{domain}': "
                        f"{result.get('error', 'unknown error')}"
                    )

            self._save_manifest()
        else:
            print("\u2705 All spectral signatures are current -- skipping generation.")

        analyzer = RuntimeSpectralAnalyzer(
            signature_dir=str(self.signature_dir),
            model_name=self.model_name,
        )

        if results:
            injected = 0
            for domain, result in results.items():
                if result.get("status") == "saved":
                    sig = result.get("_signature")
                    if sig is not None and isinstance(sig, np.ndarray):
                        analyzer.signatures[domain] = sig
                        injected += 1
            if injected:
                print(
                    f"\u26a1 Hot-populated {injected} signature(s) from memory "
                    "(skipped disk re-read)."
                )
                analyzer._rebuild_sig_matrix()

        return analyzer

    def get_manifest(self) -> Dict[str, str]:
        return dict(self._manifest)

    def invalidate(self, domain: str) -> None:
        self._manifest.pop(domain, None)
        self._save_manifest()
        print(f"\U0001f6ab Invalidated spectral signature for: {domain}")
