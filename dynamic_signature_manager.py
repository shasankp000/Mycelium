"""
DynamicSignatureManager
=======================
Keeps spectral signatures in sync with the live expert registry.

Call sync_signatures(registered_domains) once at startup (after
UnifiedExpertSystem initialises).  Returns an up-to-date
RuntimeSpectralAnalyzer ready for the MultiLensRouter to use.

Features
--------
- Auto-generates missing signatures for every registered domain.
- Content-addressable manifest prevents redundant regeneration.
- Graceful fallback texts when CSV corpora are absent or LFS-only.
- force_regen=True for unconditional rebuild (CI / management commands).
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

from spectral_analyzer import SpectralSignatureGenerator, RuntimeSpectralAnalyzer


# ---------------------------------------------------------------------------
# Domain → corpus source mapping
# ---------------------------------------------------------------------------
# Add a new entry here whenever a new expert domain is registered.
# Keys must match the domain names returned by UnifiedExpertSystem.experts.

DOMAIN_CORPUS_SOURCES: Dict[str, Dict] = {
    "physics": {
        "csv": "dummy_models/Physics/physics_data.csv",
        "column": "Comment",
        "sample_size": 50,
        "fallback_texts": [
            "Quantum entanglement occurs when particles remain correlated.",
            "String theory posits that fundamental particles are tiny vibrating strings.",
            "The Higgs boson gives other particles their mass via the Higgs field.",
            "Einstein's general relativity describes gravity as spacetime curvature.",
            "Wave-particle duality means electrons exhibit both wave and particle behaviour.",
            "The standard model unifies the electromagnetic, weak, and strong forces.",
            "Thermodynamics governs energy transfer and entropy in physical systems.",
            "Nuclear fusion releases energy by combining light atomic nuclei.",
            "Superconductivity is the zero-resistance flow of electrons at low temperatures.",
            "Photoelectric effect demonstrates that light is quantised into photons.",
        ],
    },
    "medical": {
        "csv": "dummy_models/Medical/medical_dataset.csv",
        "column": "sentence",
        "sample_size": 50,
        "fallback_texts": [
            "Metastatic carcinoma requires systemic chemotherapy.",
            "The blood-brain barrier restricts passage of large molecules.",
            "Insulin resistance is a hallmark of type 2 diabetes mellitus.",
            "MRI imaging provides high-resolution soft-tissue contrast.",
            "Antibiotic resistance arises from selective evolutionary pressure.",
            "The immune system deploys T-cells to target infected cells.",
            "Hypertension increases the risk of stroke and myocardial infarction.",
            "Synaptic plasticity underlies learning and memory formation.",
            "CRISPR-Cas9 enables precise editing of genomic sequences.",
            "Vaccines stimulate adaptive immunity without causing disease.",
        ],
    },
    "music": {
        "csv": "dummy_models/Music/music_classification_dataset.csv",
        "column": "sentence",
        "sample_size": 50,
        "fallback_texts": [
            "A major chord consists of a root, major third, and perfect fifth.",
            "Counterpoint involves the interplay of independent melodic lines.",
            "The tempo marking allegro indicates a fast and lively pace.",
            "Jazz harmony relies heavily on extended chords and modal scales.",
            "Polyphony describes music with two or more independent melodic voices.",
            "The circle of fifths illustrates harmonic relationships between keys.",
            "Synthesisers generate sound electronically by modulating waveforms.",
            "Rhythm is the pattern of sounds and silences in time.",
            "Dynamic markings such as forte and piano indicate volume levels.",
            "Timbre distinguishes the sound quality of different instruments.",
        ],
    },
    "chemistry": {
        "csv": None,
        "column": None,
        "sample_size": 0,
        "fallback_texts": [
            "Covalent bonds form when atoms share electrons.",
            "Ionic compounds dissolve readily in polar solvents.",
            "The periodic table organises elements by atomic number.",
            "Exothermic reactions release energy to the surroundings.",
            "Acids donate protons while bases accept them in Brønsted-Lowry theory.",
            "Catalysts lower activation energy without being consumed.",
            "Organic chemistry focuses on carbon-containing compounds.",
            "Redox reactions involve the transfer of electrons between species.",
            "Polymers are large molecules built from repeating monomer units.",
            "Electronegativity measures an atom's tendency to attract electrons.",
        ],
    },
    "astronomy": {
        "csv": None,
        "column": None,
        "sample_size": 0,
        "fallback_texts": [
            "Earth orbits the Sun at a distance of 150 million kilometres.",
            "Neutron stars are the remnants of supernova explosions.",
            "Black holes have gravitational fields from which light cannot escape.",
            "The Milky Way galaxy contains over 200 billion stars.",
            "Dark matter does not interact with the electromagnetic force.",
            "Stellar nucleosynthesis forges heavy elements inside dying stars.",
            "The cosmic microwave background is relic radiation from the Big Bang.",
            "Exoplanets orbit stars outside our solar system.",
            "Gravitational waves are ripples in spacetime caused by massive events.",
            "Quasars are extremely luminous active galactic nuclei.",
        ],
    },
    "automobile": {
        "csv": None,
        "column": None,
        "sample_size": 0,
        "fallback_texts": [
            "A car with a 700cc engine has a tubeless tyre system.",
            "Manual transmission provides direct control over gear selection.",
            "Modern motorcycles feature advanced suspension systems.",
            "Turbochargers force more air into the combustion chamber.",
            "ABS prevents wheel lockup during emergency braking.",
            "Electric vehicles use regenerative braking to recover energy.",
            "The catalytic converter reduces harmful exhaust emissions.",
            "Torque vectoring distributes power between wheels for better handling.",
            "Hybrid powertrains combine an internal combustion engine with an electric motor.",
            "Chassis rigidity directly affects vehicle handling and safety.",
        ],
    },
}


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
        # Unknown / future domain — produce generic placeholder corpus.
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
    """
    Keeps spectral signatures in sync with the live expert registry.

    Usage
    -----
    .. code-block:: python

        expert_system = UnifiedExpertSystem()
        registered_domains = set(expert_system.experts.keys())

        sig_manager = DynamicSignatureManager()
        analyzer = sig_manager.sync_signatures(registered_domains)

        router = MultiLensRouter(spectral_analyzer=analyzer)

    The returned *analyzer* is a freshly initialised
    :class:`RuntimeSpectralAnalyzer` that has loaded all (including newly
    generated) signatures from disk.
    """

    MANIFEST_FILENAME = "manifest.json"

    def __init__(
        self,
        signature_dir: str = "signatures",
        model_name: str = "all-MiniLM-L6-v2",
        force_regen: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        signature_dir:
            Directory that holds ``*_spectral_signature.npy`` files and the
            ``manifest.json`` cache.
        model_name:
            SentenceTransformer model used for encoding.  Must match the model
            used by :class:`RuntimeSpectralAnalyzer` in MultiLensRouter.
        force_regen:
            When *True*, regenerate every signature regardless of whether the
            manifest shows it as current.  Useful in CI or after a model change.
        """
        self.signature_dir = Path(signature_dir)
        self.signature_dir.mkdir(exist_ok=True)
        self.model_name = model_name
        self.force_regen = force_regen

        self._generator = SpectralSignatureGenerator(
            model_name=model_name,
            signature_dir=signature_dir,
        )
        self._manifest: Dict[str, str] = self._load_manifest()

    # ------------------------------------------------------------------
    # Manifest helpers
    # ------------------------------------------------------------------

    @property
    def _manifest_path(self) -> Path:
        return self.signature_dir / self.MANIFEST_FILENAME

    def _load_manifest(self) -> Dict[str, str]:
        """Load the corpus-hash manifest from disk (empty dict if absent)."""
        if self._manifest_path.exists():
            try:
                return json.loads(self._manifest_path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"\u26a0\ufe0f  Could not read signature manifest: {exc}")
        return {}

    def _save_manifest(self) -> None:
        """Persist the corpus-hash manifest to disk."""
        try:
            self._manifest_path.write_text(
                json.dumps(self._manifest, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            print(f"\u26a0\ufe0f  Could not save signature manifest: {exc}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _corpus_hash(texts: List[str]) -> str:
        """Stable 12-char MD5 fingerprint of a corpus text list."""
        payload = "|||".join(sorted(texts)).encode("utf-8")
        return hashlib.md5(payload).hexdigest()[:12]

    def _signature_path(self, domain: str) -> Path:
        return self.signature_dir / f"{domain}_spectral_signature.npy"

    def _signature_exists(self, domain: str) -> bool:
        return self._signature_path(domain).exists()

    def _needs_regen(self, domain: str, corpus_hash: str) -> bool:
        """Return True if the signature must be (re)generated."""
        if self.force_regen:
            return True
        if not self._signature_exists(domain):
            return True
        return self._manifest.get(domain) != corpus_hash

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sync_signatures(self, registered_domains: Set[str]) -> RuntimeSpectralAnalyzer:
        """
        Ensure every domain in *registered_domains* has a current spectral
        signature on disk, then return a hot-loaded
        :class:`RuntimeSpectralAnalyzer`.

        Algorithm
        ---------
        1. For each domain, build its corpus and compute a content hash.
        2. Skip domains whose ``.npy`` file exists **and** whose hash matches
           the manifest entry (corpus unchanged).
        3. Batch-generate all stale / missing signatures via
           :class:`SpectralSignatureGenerator`.
        4. Update the manifest and flush to disk.
        5. Instantiate and return a fresh :class:`RuntimeSpectralAnalyzer`
           so the caller always gets up-to-date signatures.
        """
        domains_to_generate: Dict[str, List[str]] = {}

        for domain in sorted(registered_domains):  # sorted for deterministic logs
            corpus = build_corpus_for_domain(domain)
            h = self._corpus_hash(corpus)

            if self._needs_regen(domain, h):
                print(f"\U0001f504 Queuing spectral signature generation for: {domain}")
                domains_to_generate[domain] = corpus
                # Optimistically update manifest; rollback on failure below.
                self._manifest[domain] = h
            else:
                print(f"\u2705 Spectral signature up-to-date: {domain}")

        if domains_to_generate:
            print(
                f"\n\U0001f9ec Generating {len(domains_to_generate)} spectral signature(s): "
                f"{sorted(domains_to_generate.keys())}\n"
            )
            results = self._generator.generate_domain_signatures(domains_to_generate)

            # Roll back manifest entries for any failed domains so they are
            # retried on the next startup.
            for domain, result in results.items():
                if result.get("status") != "saved":
                    self._manifest.pop(domain, None)
                    print(
                        f"\u274c Signature generation failed for '{domain}': "
                        f"{result.get('error', 'unknown error')}"
                    )

            self._save_manifest()
        else:
            print("\u2705 All spectral signatures are current — skipping generation.")

        # Hot-reload: picks up any .npy files written in this call.
        analyzer = RuntimeSpectralAnalyzer(
            signature_dir=str(self.signature_dir),
            model_name=self.model_name,
        )
        return analyzer

    def get_manifest(self) -> Dict[str, str]:
        """Return a copy of the current domain → corpus-hash manifest."""
        return dict(self._manifest)

    def invalidate(self, domain: str) -> None:
        """
        Force regeneration of a specific domain's signature on next sync.

        Useful after manually updating a domain's training corpus.
        """
        self._manifest.pop(domain, None)
        self._save_manifest()
        print(f"\U0001f6ab Invalidated spectral signature for: {domain}")
