"""
Phase 1: Spectral Analysis Core
Deterministic frequency-domain analysis of domain embeddings.

This module is STANDALONE and NON-BREAKING:
- No integration with existing routing
- No fusion, optimization, or expert selection
- Optional pre-training phase for domain signatures
- Runtime analyzer for structural comparison

Graceful degradation: Returns {} if any component unavailable.

Model loading
-------------
All SentenceTransformer instantiation is delegated to ModelRegistry so
that each unique (model_name, device) pair is loaded exactly once per
process, regardless of how many SpectralSignatureGenerator or
RuntimeSpectralAnalyzer objects are created.

Vectorised scoring (M4 item 7.2)
---------------------------------
After loading all domain signatures, RuntimeSpectralAnalyzer stacks them
into a single matrix ``_sig_matrix`` (shape: [D, L]) and a precomputed
norm vector ``_sig_norms`` (shape: [D]).  analyze_text() then performs
one matrix-vector multiply (BLAS sgemv) instead of a Python loop over D
domains, reducing CPU time from O(D) serial dot products to a single
vectorised O(D × L) call.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Graceful imports with fallbacks
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import scipy.fftpack as fftpack
except ImportError:
    fftpack = None


def _get_sentence_transformer(model_name: str) -> Optional[object]:
    """
    Return a SentenceTransformer instance via ModelRegistry.
    Falls back to direct instantiation if ModelRegistry is unavailable.
    Always pins device='cpu' to avoid competing with Ollama for VRAM.
    """
    try:
        from model_registry import get_model
        return get_model(model_name, model_type="sentence_transformer", device="cpu")
    except ImportError:
        pass

    # Direct fallback
    if SentenceTransformer is not None:
        try:
            return SentenceTransformer(model_name)
        except Exception as e:
            print(f"\u26a0\ufe0f  Failed to load SentenceTransformer: {e}")
    return None


class SpectralSignatureGenerator:
    """
    Pre-training phase: Generate and save spectral domain signatures.

    Process:
    1. Load corpus texts for each domain
    2. Encode using SentenceTransformer
    3. Compute FFT + PSD per embedding dimension
    4. Average across dimensions and texts
    5. Normalize to [0, 1]
    6. Save to disk as .npy files
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", signature_dir: str = "signatures"):
        """
        Initialize signature generator.

        Args:
            model_name: SentenceTransformer model name
            signature_dir: Directory to save signatures
        """
        self.model_name = model_name
        self.signature_dir = Path(signature_dir)
        self.signature_dir.mkdir(exist_ok=True)

        # Delegate to ModelRegistry — free after first call
        self.model = _get_sentence_transformer(model_name)
        if self.model is None:
            print("\u26a0\ufe0f  SentenceTransformer not available; signature generation will fail gracefully")

    def _compute_psd_for_dimension(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute Power Spectral Density for a single 1D signal.

        Args:
            signal: 1D array (e.g., one embedding dimension across all texts)

        Returns:
            PSD array (one-sided, normalized)
        """
        if fftpack is None:
            return np.ones(len(signal) // 2 + 1)

        try:
            fft_vals = fftpack.fft(signal)
            power = np.abs(fft_vals) ** 2 / len(signal)
            psd = power[:len(signal) // 2 + 1]
            return psd
        except Exception as e:
            print(f"\u26a0\ufe0f  PSD computation failed: {e}")
            return np.ones(len(signal) // 2 + 1)

    def generate_domain_signatures(
        self,
        domain_corpus: Dict[str, List[str]]
    ) -> Dict[str, Dict]:
        """
        Generate and save spectral signatures for multiple domains.

        Args:
            domain_corpus: Dict mapping domain names to lists of texts

        Returns:
            Dict with metadata for each domain
        """
        if self.model is None:
            print("\u274c Signature generation skipped: SentenceTransformer unavailable")
            return {}

        results = {}

        for domain_name, texts in domain_corpus.items():
            if not texts:
                print(f"\u26a0\ufe0f  Skipping empty corpus for domain: {domain_name}")
                continue

            try:
                print(f"\U0001f4dd Generating signature for domain: {domain_name}")

                embeddings = self.model.encode(texts)  # (num_texts, embedding_dim)
                num_texts, embedding_dim = embeddings.shape

                average_psd = None

                for dim_idx in range(embedding_dim):
                    signal = embeddings[:, dim_idx]
                    psd = self._compute_psd_for_dimension(signal)
                    if average_psd is None:
                        average_psd = psd
                    else:
                        average_psd = (average_psd * dim_idx + psd) / (dim_idx + 1)

                if average_psd is not None and np.max(average_psd) > 0:
                    signature = average_psd / np.max(average_psd)
                else:
                    signature = average_psd if average_psd is not None else np.zeros(1)

                signature_path = self.signature_dir / f"{domain_name}_spectral_signature.npy"
                np.save(signature_path, signature)

                results[domain_name] = {
                    "signature_path": str(signature_path),
                    "num_texts": num_texts,
                    "embedding_dim": embedding_dim,
                    "status": "saved",
                    # Expose the in-memory array so DynamicSignatureManager can
                    # populate the analyzer cache without a redundant disk read.
                    "_signature": signature,
                }

                print(f"\u2705 Saved signature: {signature_path}")

            except Exception as e:
                print(f"\u274c Failed to generate signature for {domain_name}: {e}")
                results[domain_name] = {"status": "failed", "error": str(e)}

        return results


class RuntimeSpectralAnalyzer:
    """
    Runtime phase: Analyze input text and compare against domain signatures.

    Process:
    1. Load all available domain signatures
    2. Encode input text
    3. Compute input's PSD (same method as training)
    4. Batch cosine similarity against all domain signatures (vectorised)
    5. Return normalized scores [0, 1]

    Vectorised scoring (M4 item 7.2)
    ---------------------------------
    _rebuild_sig_matrix() stacks all loaded signatures into a float32 matrix
    _sig_matrix of shape [D, L] (zero-padded to the longest signature) and
    precomputes per-row L2 norms into _sig_norms [D].  analyze_text() then
    performs the whole similarity computation with a single np.dot call
    (BLAS sgemv) instead of a Python loop over D domains.
    """

    def __init__(self, signature_dir: str = "signatures", model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize runtime analyzer.

        Args:
            signature_dir: Directory containing .npy signatures
            model_name: SentenceTransformer model name
        """
        self.signature_dir = Path(signature_dir)
        self.model_name = model_name

        # Delegate to ModelRegistry — free after first call
        self.model = _get_sentence_transformer(model_name)

        # Load all available signatures into the in-memory cache
        self.signatures: Dict[str, np.ndarray] = {}
        self._load_signatures()

        # Build the vectorised scoring matrix from whatever was loaded
        self._sig_matrix: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self._sig_domains: List[str] = []
        self._sig_norms: np.ndarray = np.empty(0, dtype=np.float32)
        self._rebuild_sig_matrix()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_signatures(self) -> None:
        """Load all *_spectral_signature.npy files from signature_dir."""
        if not self.signature_dir.exists():
            print(f"\u26a0\ufe0f  Signature directory not found: {self.signature_dir}")
            return

        loaded = 0
        for sig_file in self.signature_dir.glob("*_spectral_signature.npy"):
            domain_name = sig_file.stem.replace("_spectral_signature", "")
            try:
                self.signatures[domain_name] = np.load(sig_file)
                loaded += 1
            except Exception as e:
                print(f"\u26a0\ufe0f  Failed to load signature for {domain_name}: {e}")

        if loaded:
            print(f"\u2705 Loaded {loaded} spectral signature(s) from {self.signature_dir}")
        else:
            print(f"\u26a0\ufe0f  No spectral signatures found in {self.signature_dir}")

    def _rebuild_sig_matrix(self) -> None:
        """
        (Re)build the vectorised scoring matrix from self.signatures.

        Called once after _load_signatures() and again by
        DynamicSignatureManager after injecting freshly-generated signatures
        into self.signatures.

        Populates:
            self._sig_domains  — ordered list of domain names [D]
            self._sig_matrix   — float32 array [D, L] (zero-padded)
            self._sig_norms    — float32 array [D] (L2 norms per row)
        """
        if not self.signatures:
            self._sig_domains = []
            self._sig_matrix = np.empty((0, 0), dtype=np.float32)
            self._sig_norms = np.empty(0, dtype=np.float32)
            return

        domains = sorted(self.signatures.keys())  # deterministic ordering
        max_len = max(v.shape[0] for v in self.signatures.values())

        matrix = np.zeros((len(domains), max_len), dtype=np.float32)
        for i, d in enumerate(domains):
            sig = self.signatures[d].astype(np.float32)
            matrix[i, : sig.shape[0]] = sig

        norms = np.linalg.norm(matrix, axis=1).astype(np.float32)  # [D]

        self._sig_domains = domains
        self._sig_matrix = matrix
        self._sig_norms = norms

    def _compute_psd_for_input(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute PSD for a single 1D input signal.

        Args:
            signal: 1D array

        Returns:
            PSD array (one-sided, normalized)
        """
        if fftpack is None:
            return np.ones(len(signal) // 2 + 1)

        try:
            fft_vals = fftpack.fft(signal)
            power = np.abs(fft_vals) ** 2 / len(signal)
            return power[:len(signal) // 2 + 1]
        except Exception as e:
            print(f"\u26a0\ufe0f  PSD computation failed: {e}")
            return np.ones(len(signal) // 2 + 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze input text against all loaded domain signatures.

        Scoring is performed with a single vectorised batch cosine operation
        (M4 item 7.2) instead of a per-domain Python loop.

        Args:
            text: Input query string

        Returns:
            Dict mapping domain names to similarity scores in [0, 1].
            Returns {} on any failure.
        """
        if self.model is None or not self.signatures or self._sig_matrix.shape[0] == 0:
            return {}

        try:
            embedding = self.model.encode([text])[0]  # (embedding_dim,)

            # Compute average PSD across embedding dimensions
            input_psd: Optional[np.ndarray] = None
            for dim_idx in range(len(embedding)):
                psd = self._compute_psd_for_input(np.array([embedding[dim_idx]]))
                if input_psd is None:
                    input_psd = psd
                else:
                    input_psd = (input_psd * dim_idx + psd) / (dim_idx + 1)

            if input_psd is None:
                return {}

            # Normalise
            if np.max(input_psd) > 0:
                input_psd = input_psd / np.max(input_psd)

            # -------------------------------------------------------
            # Vectorised batch cosine (M4 item 7.2)
            # -------------------------------------------------------
            L = self._sig_matrix.shape[1]
            psd_f32 = input_psd.astype(np.float32)

            # Pad or truncate query PSD to match matrix width
            if len(psd_f32) < L:
                query_vec = np.zeros(L, dtype=np.float32)
                query_vec[: len(psd_f32)] = psd_f32
            else:
                query_vec = psd_f32[:L]

            query_norm = float(np.linalg.norm(query_vec))
            if query_norm < 1e-10:
                return {d: 0.0 for d in self._sig_domains}

            dots = self._sig_matrix @ query_vec            # [D]  — single BLAS call
            denom = self._sig_norms * query_norm           # [D]
            # Avoid divide-by-zero for any zero-norm signatures
            safe_denom = np.where(denom < 1e-10, 1e-10, denom)
            cosines = dots / safe_denom                    # [D] in [-1, 1]

            # Map from [-1, 1] to [0, 1]
            scores_arr = ((cosines + 1.0) / 2.0).clip(0.0, 1.0)
            return dict(zip(self._sig_domains, scores_arr.tolist()))

        except Exception as e:
            print(f"\u26a0\ufe0f  Spectral analysis failed: {e}")
            return {}

    def get_available_domains(self) -> List[str]:
        """Return list of domains with loaded signatures."""
        return list(self.signatures.keys())

    def get_top_domains(
        self,
        text: str,
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        Return the top-k (domain, score) pairs for a given text.

        Args:
            text:  Input query string.
            top_k: Number of top domains to return.

        Returns:
            List of (domain_name, score) tuples sorted descending by score.
        """
        scores = self.analyze_text(text)
        if not scores:
            return []
        sorted_domains = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_domains[:top_k]
