"""
Phase 1: Spectral Analysis Core
Deterministic frequency-domain analysis of domain embeddings.

This module is STANDALONE and NON-BREAKING:
- No integration with existing routing
- No fusion, optimization, or expert selection
- Optional pre-training phase for domain signatures
- Runtime analyzer for structural comparison

Graceful degradation: Returns {} if any component unavailable.
"""

import os
import json
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
        
        # Try to load model; gracefully degrade if unavailable
        try:
            if SentenceTransformer is not None:
                self.model = SentenceTransformer(model_name)
            else:
                print("⚠️  SentenceTransformer not available; signature generation will fail gracefully")
                self.model = None
        except Exception as e:
            print(f"⚠️  Failed to load SentenceTransformer: {e}")
            self.model = None
    
    def _compute_psd_for_dimension(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute Power Spectral Density for a single 1D signal.
        
        Args:
            signal: 1D array (e.g., one embedding dimension across all texts)
            
        Returns:
            PSD array (one-sided, normalized)
        """
        if fftpack is None:
            # Fallback: return uniform array
            return np.ones(len(signal) // 2 + 1)
        
        try:
            # Compute FFT
            fft_vals = fftpack.fft(signal)
            
            # Compute power spectrum (magnitude squared, normalized)
            power = np.abs(fft_vals) ** 2 / len(signal)
            
            # Return one-sided PSD
            psd = power[:len(signal) // 2 + 1]
            
            return psd
        except Exception as e:
            print(f"⚠️  PSD computation failed: {e}")
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
            print("❌ Signature generation skipped: SentenceTransformer unavailable")
            return {}
        
        results = {}
        
        for domain_name, texts in domain_corpus.items():
            if not texts:
                print(f"⚠️  Skipping empty corpus for domain: {domain_name}")
                continue
            
            try:
                print(f"📝 Generating signature for domain: {domain_name}")
                
                # Encode all texts
                embeddings = self.model.encode(texts)  # (num_texts, embedding_dim)
                num_texts, embedding_dim = embeddings.shape
                
                # Compute PSD for each dimension, across all texts
                average_psd = None
                
                for dim_idx in range(embedding_dim):
                    # Extract signal for this dimension across all texts
                    signal = embeddings[:, dim_idx]
                    
                    # Compute PSD
                    psd = self._compute_psd_for_dimension(signal)
                    
                    # Accumulate
                    if average_psd is None:
                        average_psd = psd
                    else:
                        average_psd = (average_psd * dim_idx + psd) / (dim_idx + 1)
                
                # Normalize to [0, 1]
                if average_psd is not None and np.max(average_psd) > 0:
                    signature = average_psd / np.max(average_psd)
                else:
                    signature = average_psd if average_psd is not None else np.zeros(1)
                
                # Save signature
                signature_path = self.signature_dir / f"{domain_name}_spectral_signature.npy"
                np.save(signature_path, signature)
                
                results[domain_name] = {
                    "signature_path": str(signature_path),
                    "num_texts": num_texts,
                    "embedding_dim": embedding_dim,
                    "status": "saved"
                }
                
                print(f"✅ Saved signature: {signature_path}")
                
            except Exception as e:
                print(f"❌ Failed to generate signature for {domain_name}: {e}")
                results[domain_name] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        return results


class RuntimeSpectralAnalyzer:
    """
    Runtime phase: Analyze input text and compare against domain signatures.
    
    Process:
    1. Load all available domain signatures
    2. Encode input text
    3. Compute input's PSD (same method as training)
    4. Cross-correlate with each domain signature
    5. Return normalized scores [0, 1]
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
        
        # Try to load model
        try:
            if SentenceTransformer is not None:
                self.model = SentenceTransformer(model_name)
            else:
                self.model = None
        except Exception as e:
            print(f"⚠️  Failed to load SentenceTransformer: {e}")
            self.model = None
        
        # Load all available signatures
        self.signatures: Dict[str, np.ndarray] = {}
        self._load_signatures()
    
    def _load_signatures(self):
        """Load all available domain signatures from disk."""
        if not self.signature_dir.exists():
            print(f"⚠️  Signature directory not found: {self.signature_dir}")
            return
        
        for sig_file in self.signature_dir.glob("*_spectral_signature.npy"):
            domain_name = sig_file.stem.replace("_spectral_signature", "")
            try:
                self.signatures[domain_name] = np.load(sig_file)
                print(f"✅ Loaded signature: {domain_name}")
            except Exception as e:
                print(f"⚠️  Failed to load signature for {domain_name}: {e}")
    
    def _compute_psd_for_signal(self, signal: np.ndarray) -> np.ndarray:
        """Same PSD computation as training phase."""
        if fftpack is None:
            return np.ones(len(signal) // 2 + 1)
        
        try:
            fft_vals = fftpack.fft(signal)
            power = np.abs(fft_vals) ** 2 / len(signal)
            psd = power[:len(signal) // 2 + 1]
            return psd
        except Exception:
            return np.ones(len(signal) // 2 + 1)
    
    def _normalized_cross_correlation(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """
        Compute normalized cross-correlation between two signals.
        
        Returns score in [0, 1].
        """
        try:
            # Pad to equal length
            max_len = max(len(sig1), len(sig2))
            sig1_padded = np.pad(sig1, (0, max_len - len(sig1)), mode='constant')
            sig2_padded = np.pad(sig2, (0, max_len - len(sig2)), mode='constant')
            
            # Compute correlation coefficient
            correlation = np.corrcoef(sig1_padded, sig2_padded)[0, 1]
            
            # Handle NaN
            if np.isnan(correlation):
                correlation = 0.0
            
            # Normalize to [0, 1]
            score = (correlation + 1.0) / 2.0
            
            return float(max(0.0, min(1.0, score)))
        except Exception as e:
            print(f"⚠️  Cross-correlation failed: {e}")
            return 0.0
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze input text and compute spectral scores per domain.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dict mapping domain names to scores in [0, 1]
        """
        if self.model is None:
            print("⚠️  Model unavailable; returning empty scores")
            return {}
        
        if not text or not text.strip():
            return {domain: 0.0 for domain in self.signatures.keys()}
        
        try:
            # Encode input
            input_embedding = self.model.encode(text)  # (embedding_dim,)
            
            # Compute input's PSD signature
            # Treat embedding as a signal (one dimension across embedding values)
            input_signature = np.abs(input_embedding)  # Magnitude
            input_signature = input_signature / (np.max(input_signature) + 1e-10)  # Normalize
            
            # Cross-correlate with each domain
            scores = {}
            for domain_name, domain_signature in self.signatures.items():
                score = self._normalized_cross_correlation(input_signature, domain_signature)
                scores[domain_name] = score
            
            return scores
        
        except Exception as e:
            print(f"⚠️  Text analysis failed: {e}")
            return {}
    
    def is_ready(self) -> bool:
        """Check if analyzer is ready (has model and signatures)."""
        return self.model is not None and len(self.signatures) > 0
    
    def get_available_domains(self) -> List[str]:
        """Get list of domains with loaded signatures."""
        return sorted(self.signatures.keys())


# ============================================================================
# Minimal Testing
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("PHASE 1: Spectral Analysis Core - Minimal Test")
    print("="*70)
    
    # Create tiny test corpus
    print("\n📝 Creating test corpus...")
    test_corpus = {
        "astronomy": [
            "Earth orbits the Sun at a distance of 150 million kilometers",
            "The moon influences Earth's tides through gravitational forces",
            "Stars are massive celestial bodies that emit light and heat",
        ],
        "automobile": [
            "A car with a 700cc engine has a tubeless tyre system",
            "Manual transmission provides direct control over gear selection",
            "Modern motorcycles feature advanced suspension systems",
        ],
    }
    
    # Test signature generation
    print("\n🔄 Pre-training signatures...")
    generator = SpectralSignatureGenerator(signature_dir="signatures")
    
    if generator.model is None:
        print("⚠️  SentenceTransformer not available; skipping pre-training test")
        print("   (This is expected in environments without transformers)")
    else:
        gen_results = generator.generate_domain_signatures(test_corpus)
        print(f"Generated: {list(gen_results.keys())}")
        for domain, result in gen_results.items():
            print(f"  {domain}: {result}")
    
    # Test runtime analyzer
    print("\n🔍 Runtime analysis...")
    analyzer = RuntimeSpectralAnalyzer(signature_dir="signatures")
    
    print(f"Ready: {analyzer.is_ready()}")
    print(f"Loaded domains: {analyzer.get_available_domains()}")
    
    if analyzer.is_ready():
        # Test matching-domain text
        text1 = "The planet orbits a star in space"
        print(f"\nAnalyzing (expected astronomy): '{text1}'")
        scores1 = analyzer.analyze_text(text1)
        for domain, score in sorted(scores1.items(), key=lambda x: x[1], reverse=True):
            print(f"  {domain}: {score:.4f}")
        
        # Test non-matching text
        text2 = "My car has a red finish"
        print(f"\nAnalyzing (expected automobile): '{text2}'")
        scores2 = analyzer.analyze_text(text2)
        for domain, score in sorted(scores2.items(), key=lambda x: x[1], reverse=True):
            print(f"  {domain}: {score:.4f}")
    else:
        print("⚠️  Analyzer not ready (no signatures or model)")
    
    print("\n" + "="*70)
    print("✅ Phase 1 Test Complete")
    print("="*70)
