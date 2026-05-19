#!/usr/bin/env python3
"""
Unified Expert Pre-Check Layer: Integration of K-Medoids, Calibration, and OOD Detection

This module combines three powerful systems:
1. K-Medoids clustering for domain representation and similarity
2. Calibration for well-calibrated confidence scores
3. OOD detection for distribution shift identification

The unified system provides the most robust expert decision-making possible.

Changes (2026-05-19 — M5 patch)
---------------------------------
- M5 §9.1: _ExpertLRUCache class added above UnifiedExpertSystem.
  Keeps at most ``max_resident_experts`` (default 8, configurable via
  config.toml [unified_expert] max_resident_experts) UnifiedExpert
  instances fully loaded in RAM. When the limit is exceeded the
  least-recently-used expert is pickle-serialised to
  <project_root>/lru_cache/<domain>.pkl and replaced by a lazy
  placeholder.  Re-accessing an evicted expert transparently
  deserialises it. Thread-safe via a single RLock.
- UnifiedExpertSystem.__init__ wraps self.experts in _ExpertLRUCache
  after _initialize_all_experts() completes.
- _initialize_all_experts() populates self._raw_experts (plain dict)
  instead of self.experts directly so the LRU wrapper can be applied
  once, after all experts are loaded.
"""

# ---------------------------------------------------------------------------
# Offline guard: prefer local HuggingFace cache over network downloads.
# Set BEFORE any sentence_transformers / transformers import so the library
# picks up the env-var at import time.  Only activates when the var is not
# already set in the environment (so callers can still override to "0" if
# they explicitly want to pull a fresh model).
# ---------------------------------------------------------------------------
import os as _os
_os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
_os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import uuid
import pickle
import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, brier_score_loss, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

from core.types import ExpertDecisionResult, RoutingResult


# ---------------------------------------------------------------------------
# M5 §9.1 — _ExpertLRUCache: evict least-recently-used experts to disk
# ---------------------------------------------------------------------------
import threading as _threading
import pathlib as _pathlib

class _ExpertLRUCache:
    """LRU cache for UnifiedExpert instances.

    Keeps at most *max_k* experts fully loaded in memory.  When the limit is
    exceeded, the least-recently-used expert is serialised to
    ``<evict_dir>/lru_cache/<domain>.pkl`` and removed from the in-memory
    dict.  Re-accessing an evicted expert transparently deserialises it.

    Thread-safe via a single reentrant lock so the per-sentence routing loop
    can call __getitem__ concurrently with background prefetch.

    Design notes
    ------------
    - Uses collections.OrderedDict as an O(1) LRU map (move_to_end on
      access, popitem(last=False) for eviction).
    - Pickle serialisation failures are caught and the expert is kept in
      memory rather than silently lost.
    - Evicted domain pickle files persist across server restarts, giving
      free warm reload of rarely-used experts.
    """

    def __init__(self, experts: dict, max_k: int = 8, evict_dir: str = "."):
        self._max_k = max_k
        self._lock = _threading.RLock()
        self._cache_dir = _pathlib.Path(evict_dir) / "lru_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        from collections import OrderedDict
        self._loaded: OrderedDict = OrderedDict()
        for domain, expert in experts.items():
            self._loaded[domain] = expert
        # Evict down to max_k right away if seeded with more experts than limit.
        self._evict_if_needed()

    # ------------------------------------------------------------------
    # Dict-like interface
    # ------------------------------------------------------------------
    def keys(self):
        with self._lock:
            return list(self._loaded.keys()) + self._evicted_domains()

    def values(self):
        with self._lock:
            for domain in list(self.keys()):
                yield self[domain]

    def items(self):
        with self._lock:
            for domain in list(self.keys()):
                yield domain, self[domain]

    def __contains__(self, domain):
        with self._lock:
            return domain in self._loaded or self._evict_path(domain).exists()

    def __len__(self):
        with self._lock:
            return len(self._loaded) + len(self._evicted_domains())

    def __iter__(self):
        return iter(self.keys())

    def get(self, domain, default=None):
        try:
            return self[domain]
        except KeyError:
            return default

    # ------------------------------------------------------------------
    # Main access: transparent lazy deserialisation
    # ------------------------------------------------------------------
    def __getitem__(self, domain: str):
        with self._lock:
            if domain in self._loaded:
                self._loaded.move_to_end(domain)  # mark as most-recently-used
                return self._loaded[domain]

            evict_path = self._evict_path(domain)
            if evict_path.exists():
                with evict_path.open("rb") as fh:
                    expert = pickle.load(fh)
                self._loaded[domain] = expert
                self._loaded.move_to_end(domain)
                self._evict_if_needed()
                return expert

            raise KeyError(domain)

    def __setitem__(self, domain: str, expert):
        with self._lock:
            self._loaded[domain] = expert
            self._loaded.move_to_end(domain)
            self._evict_if_needed()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _evict_path(self, domain: str) -> _pathlib.Path:
        safe = domain.replace("/", "_").replace("\\", "_")
        return self._cache_dir / f"{safe}.pkl"

    def _evicted_domains(self):
        """Return domain names currently serialised on disk (not in RAM)."""
        loaded = set(self._loaded.keys())
        return [p.stem for p in self._cache_dir.glob("*.pkl") if p.stem not in loaded]

    def _evict_if_needed(self):
        """Evict the LRU expert to disk until len(_loaded) <= max_k."""
        import logging as _logging
        while len(self._loaded) > self._max_k:
            domain, expert = self._loaded.popitem(last=False)  # pop LRU
            try:
                with self._evict_path(domain).open("wb") as fh:
                    pickle.dump(expert, fh, protocol=pickle.HIGHEST_PROTOCOL)
                _logging.getLogger(__name__).debug(
                    "ExpertLRU: evicted '%s' to disk", domain
                )
            except Exception as exc:
                # Serialisation failed — keep in memory rather than lose the expert
                self._loaded[domain] = expert
                self._loaded.move_to_end(domain, last=False)  # restore LRU position
                _logging.getLogger(__name__).warning(
                    "ExpertLRU: could not evict '%s' to disk: %s", domain, exc
                )
                break  # avoid infinite loop if every expert fails to serialise


class UnifiedExpert:
    """
    Unified Expert with K-Medoids clustering, Calibration, and OOD detection.
    
    This is the most advanced version of the Expert class, combining all three
    systems for maximum decision-making quality.
    """
    
    def __init__(self, domain, model_path, dataset_path, text_column='sentence', 
                 vectorizer_path=None, enable_calibration=True, enable_ood_detection=True):
        """
        Initialize unified expert with all three systems.
        
        Args:
            domain: Expert domain (e.g., 'medical', 'physics', 'music')
            model_path: Path to trained SVM model
            dataset_path: Path to training dataset
            text_column: Name of text column in dataset
            vectorizer_path: Path to TF-IDF vectorizer
            enable_calibration: Whether to enable calibration system
            enable_ood_detection: Whether to enable OOD detection
        """
        self.domain = domain
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.text_column = text_column
        self.vectorizer_path = vectorizer_path
        self.enable_calibration = enable_calibration
        self.enable_ood_detection = enable_ood_detection
        
        # Generate UUIDs for tracking
        self.model_uuid = str(uuid.uuid4())
        self.dataset_uuid = str(uuid.uuid4())
        self.centroid_uuid = str(uuid.uuid4())
        
        # Initialize core components
        self.model = None
        self.vectorizer = None
        self.calibrated_model = None
        self.calibration_score = None
        self.label_encoder = None
        
        # K-Medoids components
        self.centroid = None
        self.medoids = None
        self.medoid = None
        self.medoid_text = None
        self.centroid_path = None
        
        # OOD Detection components
        self.ood_detector = None
        self.training_features = None
        self.training_embeddings = None
        self.isolation_forest = None
        self.nn_detector = None

        # ---------------------------------------------------------------------------
        # Lazy-cached SentenceTransformer: loaded once per UnifiedExpert instance
        # and reused across _setup_ood_detection_system, _calculate_centroid,
        # calculate_similarity_to_centroid, and detect_ood.  This eliminates the
        # previous pattern of calling SentenceTransformer('all-MiniLM-L6-v2') in
        # each method, which triggered repeated HuggingFace Hub connectivity checks
        # and full weight re-loads for every expert during calibration setup.
        # ---------------------------------------------------------------------------
        self._sentence_transformer = None
        
        # Statistics for analysis
        self.system_stats = {
            'k_medoids_enabled': True,
            'calibration_enabled': enable_calibration,
            'ood_detection_enabled': enable_ood_detection,
            'initialization_status': {}
        }
        
        # Initialize all systems
        self._initialize_unified_expert()

    # ---------------------------------------------------------------------------
    # Lazy SentenceTransformer accessor
    # ---------------------------------------------------------------------------
    def _get_sentence_transformer(self, model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
        """Return a cached SentenceTransformer, loading it only once per instance.

        Using a per-instance cache means:
        - The model weights are loaded from the local HuggingFace cache exactly
          once per UnifiedExpert, not once per method call.
        - With TRANSFORMERS_OFFLINE=1 (set at module top) no network round-trip
          is attempted after the first download.
        - If a different model_name is requested the cache is invalidated so the
          correct model is always returned.
        """
        if (
            self._sentence_transformer is None
            or getattr(self._sentence_transformer, '_model_name', None) != model_name
        ):
            local_cache = os.path.join(
                os.path.expanduser("~"), ".cache", "huggingface", "hub"
            )
            self._sentence_transformer = SentenceTransformer(
                model_name, cache_folder=local_cache
            )
            # Store the name so we can detect if a different model is requested later.
            self._sentence_transformer._model_name = model_name
        return self._sentence_transformer

    def _initialize_unified_expert(self):
        """Initialize all three systems in sequence."""
        print(f"\n\U0001f680 Initializing Unified Expert for {self.domain} domain...")
        print("="*60)
        
        try:
            # 1. Core model and vectorizer
            print("1\ufe0f\u20e3 Loading core SVM model and vectorizer...")
            self._load_trained_model()
            self._load_vectorizer()
            self.system_stats['initialization_status']['core_model'] = '\u2705 Success'
            
            # 2. K-Medoids clustering system
            print("2\ufe0f\u20e3 Setting up K-Medoids clustering...")
            self._setup_k_medoids_system()
            self.system_stats['initialization_status']['k_medoids'] = '\u2705 Success'
            
            # 3. Calibration system (optional)
            if self.enable_calibration:
                print("3\ufe0f\u20e3 Setting up calibration system...")
                self._setup_calibration_system()
                self.system_stats['initialization_status']['calibration'] = '\u2705 Success'
            else:
                print("3\ufe0f\u20e3 Calibration disabled - using fallback confidence")
                self._setup_fallback_confidence()
                self.system_stats['initialization_status']['calibration'] = '\u26a0\ufe0f Disabled'
            
            # 4. OOD Detection system (optional)
            if self.enable_ood_detection:
                print("4\ufe0f\u20e3 Setting up OOD detection system...")
                self._setup_ood_detection_system()
                self.system_stats['initialization_status']['ood_detection'] = '\u2705 Success'
            else:
                print("4\ufe0f\u20e3 OOD detection disabled")
                self.system_stats['initialization_status']['ood_detection'] = '\u26a0\ufe0f Disabled'
            
            print(f"\U0001f389 Unified Expert initialization complete for {self.domain}!")
            print("="*60)
            
        except Exception as e:
            print(f"\u274c Error initializing unified expert for {self.domain}: {e}")
            raise
    
    def _load_trained_model(self):
        """Load the trained SVM model."""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
        else:
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
    
    def _load_vectorizer(self):
        """Load the TF-IDF vectorizer."""
        if self.vectorizer_path and os.path.exists(self.vectorizer_path):
            with open(self.vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
        else:
            # Try to find vectorizer in same directory as model
            base_dir = os.path.dirname(self.model_path)
            vectorizer_files = [f for f in os.listdir(base_dir) if 'vectorizer' in f and f.endswith('.pkl')]
            if vectorizer_files:
                self.vectorizer_path = os.path.join(base_dir, vectorizer_files[0])
                with open(self.vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
            else:
                raise FileNotFoundError(f"Vectorizer file not found")
    
    def _setup_k_medoids_system(self):
        """Set up K-Medoids clustering system."""
        # Check for existing centroid
        base_dir = os.path.dirname(self.model_path)
        existing_centroid = self._find_existing_centroid(base_dir)
        
        if existing_centroid:
            print(f"   Found existing centroid: {existing_centroid}")
            self.centroid_path = existing_centroid
            self.load_centroid()
        else:
            print("   No existing centroid found, calculating new one...")
            self._calculate_centroid()
            self._save_centroid()
    
    def _find_existing_centroid(self, base_dir):
        """Find existing centroid file."""
        for file in os.listdir(base_dir):
            if file.startswith('centroid_') and file.endswith('.pkl'):
                return os.path.join(base_dir, file)
        return None
    
    def _setup_calibration_system(self):
        """Set up calibration system."""
        try:
            # Load dataset
            df = pd.read_csv(self.dataset_path)
            X = df[self.text_column].dropna()
            
            # Find label column
            label_col = self._find_label_column(df)
            y = df[df[self.text_column].notna()][label_col]
            
            # Handle label encoding
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Check for sufficient samples
            unique_labels, counts = np.unique(y_encoded, return_counts=True)
            min_count = min(counts)
            
            if min_count < 2:
                print(f"   \u26a0\ufe0f Warning: Insufficient samples for calibration (min: {min_count})")
                self._setup_fallback_confidence()
                return
            
            # Create train/validation/test split
            test_size = 0.3 if len(X) > 100 else 0.5
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
            )
            
            val_size = 0.5 if len(X_temp) > 50 else 0.6
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
            )
            
            # Vectorize
            X_train_tfidf = self.vectorizer.transform(X_train)
            X_val_tfidf = self.vectorizer.transform(X_val)
            X_test_tfidf = self.vectorizer.transform(X_test)
            
            # Create calibrated classifier.
            # NOTE: cv='prefit' was removed in scikit-learn 1.2 and raises a
            # ValueError in 1.4+.  The correct approach for an already-trained
            # estimator is to omit cv entirely and fit the wrapper only on the
            # held-out validation split — which is exactly what the .fit() call
            # below does.
            self.calibrated_model = CalibratedClassifierCV(
                self.model, method='isotonic'
            )
            
            # Fit calibration on validation set
            self.calibrated_model.fit(X_val_tfidf, y_val)
            
            # Evaluate calibration
            calibrated_proba = self.calibrated_model.predict_proba(X_test_tfidf)
            y_pred_calibrated = self.calibrated_model.predict(X_test_tfidf)
            
            self.calibration_score = self._evaluate_calibration(y_test, y_pred_calibrated, calibrated_proba)
            print(f"   Calibration score: {self.calibration_score:.4f}")
            
        except Exception as e:
            print(f"   \u26a0\ufe0f Calibration setup failed: {e}")
            self._setup_fallback_confidence()
    
    def _find_label_column(self, df):
        """Find the label column in the dataset."""
        possible_cols = ['label', 'binary_label', 'target', 'class', 'category']
        for col in possible_cols:
            if col in df.columns:
                return col
        # Default to second column
        return df.columns[1] if len(df.columns) > 1 else df.columns[0]
    
    def _setup_fallback_confidence(self):
        """Setup fallback confidence when calibration is disabled/failed."""
        self.calibrated_model = self.model
        self.calibration_score = 0.5
        print("   Using fallback confidence estimation")
    
    def _setup_ood_detection_system(self):
        """Set up OOD detection system."""
        try:
            # Load and prepare training data
            df = pd.read_csv(self.dataset_path)
            training_texts = df[self.text_column].dropna().tolist()
            
            # Sample for computational efficiency
            if len(training_texts) > 500:
                sample_indices = np.random.choice(len(training_texts), 500, replace=False)
                training_texts = [training_texts[i] for i in sample_indices]
            
            # Prepare TF-IDF features
            self.training_features = self.vectorizer.transform(training_texts)
            
            # Generate embeddings — reuse cached transformer, no new Hub check
            self.training_embeddings = self._get_sentence_transformer().encode(training_texts)
            
            # Setup OOD detectors
            self.isolation_forest = IsolationForest(contamination=0.1, random_state=42, n_estimators=50)
            self.isolation_forest.fit(self.training_embeddings)
            
            self.nn_detector = NearestNeighbors(n_neighbors=3, metric='cosine')
            self.nn_detector.fit(self.training_embeddings)
            
            print(f"   OOD detection ready (training sample: {len(training_texts)})")
            
        except Exception as e:
            print(f"   \u26a0\ufe0f OOD detection setup failed: {e}")
            self.enable_ood_detection = False
    
    def _calculate_centroid(self, model_name="all-MiniLM-L6-v2", k=10):
        """Calculate centroid using K-Medoids clustering."""
        # Load dataset
        df = pd.read_csv(self.dataset_path)
        texts = df[self.text_column].dropna().tolist()
        
        # Generate embeddings — reuse cached transformer
        embeddings = self._get_sentence_transformer(model_name).encode(texts)
        
        # Run K-Medoids clustering
        medoids, medoid_indices, final_assignment = self._pure_python_k_medoids(texts, embeddings, k)
        
        # Store results
        self.medoids = medoids
        self.medoid = medoids[0] if len(medoids) > 0 else np.mean(embeddings, axis=0)
        self.centroid = self.medoid  # Primary centroid
        self.medoid_text = texts[medoid_indices[0]] if len(medoid_indices) > 0 else "No text available"
        
        print(f"   K-Medoids clustering complete: {len(medoids)} medoids")
        print(f"   Primary medoid text: '{self.medoid_text[:50]}...'")
    
    def _pure_python_k_medoids(self, texts, embeddings, k=3):
        """Pure Python implementation of PAM k-medoids algorithm."""
        n_samples = len(embeddings)
        k = min(k, n_samples)
        
        # Random initialization
        np.random.seed(42)
        medoid_indices = np.random.choice(n_samples, k, replace=False)
        
        for iteration in range(10):  # Max iterations
            # Assign points to closest medoids
            distances = cosine_similarity(embeddings, embeddings[medoid_indices])
            assignments = np.argmax(distances, axis=1)
            
            # Update medoids
            new_medoid_indices = []
            for cluster_id in range(k):
                cluster_points = np.where(assignments == cluster_id)[0]
                if len(cluster_points) > 0:
                    # Find point that minimizes total distance within cluster
                    best_medoid = cluster_points[0]
                    best_cost = float('inf')
                    
                    for candidate in cluster_points:
                        cost = np.sum(1 - cosine_similarity(
                            embeddings[candidate:candidate+1], 
                            embeddings[cluster_points]
                        ))
                        if cost < best_cost:
                            best_cost = cost
                            best_medoid = candidate
                    
                    new_medoid_indices.append(best_medoid)
                else:
                    new_medoid_indices.append(medoid_indices[cluster_id])
            
            # Check for convergence
            if set(new_medoid_indices) == set(medoid_indices):
                print(f"   K-Medoids converged after {iteration + 1} iterations")
                break
            
            medoid_indices = np.array(new_medoid_indices)
        
        return embeddings[medoid_indices], medoid_indices, assignments
    
    def _save_centroid(self):
        """Save centroid and medoids to file."""
        base_dir = os.path.dirname(self.model_path)
        centroid_filename = f"centroid_{self.centroid_uuid}.pkl"
        self.centroid_path = os.path.join(base_dir, centroid_filename)
        
        centroid_data = {
            'centroid': self.centroid,
            'medoids': self.medoids,
            'medoid_text': self.medoid_text,
            'domain': self.domain,
            'created_at': pd.Timestamp.now().isoformat()
        }
        
        with open(self.centroid_path, 'wb') as f:
            pickle.dump(centroid_data, f)
    
    def load_centroid(self):
        """Load centroid and medoids from file."""
        if self.centroid_path and os.path.exists(self.centroid_path):
            with open(self.centroid_path, 'rb') as f:
                centroid_data = pickle.load(f)
            
            self.centroid = centroid_data['centroid']
            self.medoids = centroid_data.get('medoids', [self.centroid])
            self.medoid = self.medoids[0] if len(self.medoids) > 0 else self.centroid
            self.medoid_text = centroid_data.get('medoid_text', 'Unknown')
            
            print(f"   Loaded {len(self.medoids)} medoids")
            print(f"   Primary medoid: '{self.medoid_text[:50]}...'")
        else:
            raise FileNotFoundError(f"Centroid file not found: {self.centroid_path}")
    
    def _evaluate_calibration(self, y_true, y_pred, y_proba):
        """Evaluate calibration quality."""
        try:
            if y_proba.shape[1] >= 2:
                positive_proba = y_proba[:, 1]
                brier = brier_score_loss(y_true, positive_proba)
                return 1 - brier  # Higher is better
            else:
                return 0.5
        except:
            return 0.5
    
    def calculate_similarity_to_centroid(self, text, model_name="all-MiniLM-L6-v2"):
        """Calculate similarity using K-Medoids (best among all medoids)."""
        # Reuse cached transformer — no HuggingFace check on each query
        text_embedding = self._get_sentence_transformer(model_name).encode([text])
        
        # Calculate similarity with all medoids and take maximum
        similarities = []
        medoids_array = self.medoids if hasattr(self, 'medoids') and len(self.medoids) > 0 else [self.medoid]
        
        for medoid in medoids_array:
            similarity = cosine_similarity(text_embedding, medoid.reshape(1, -1))[0][0]
            similarities.append(similarity)
        
        return float(max(similarities))
    
    def get_confidence_prediction(self, text):
        """Get calibrated confidence prediction."""
        if self.calibrated_model is None or self.vectorizer is None:
            return {'prediction': None, 'confidence': 0.0}
        
        text_tfidf = self.vectorizer.transform([text])
        
        if hasattr(self.calibrated_model, 'predict_proba'):
            probabilities = self.calibrated_model.predict_proba(text_tfidf)[0]
            prediction = self.calibrated_model.predict(text_tfidf)[0]
            confidence = max(probabilities)
        else:
            prediction = self.calibrated_model.predict(text_tfidf)[0]
            decision_value = abs(self.calibrated_model.decision_function(text_tfidf)[0])
            confidence = min(1.0, decision_value / 2.0)
        
        # Apply calibration score
        adjusted_confidence = confidence * (0.5 + 0.5 * self.calibration_score)
        
        return {
            'prediction': prediction,
            'confidence': float(adjusted_confidence),
            'raw_confidence': float(confidence),
            'calibration_score': float(self.calibration_score)
        }
    
    def detect_ood(self, text):
        """Perform OOD detection using multiple methods."""
        if not self.enable_ood_detection:
            return {
                'is_ood': False,
                'ood_confidence': 0.0,
                'ood_scores': {},
                'reason': 'OOD detection disabled'
            }
        
        try:
            # Prepare features
            text_features = self.vectorizer.transform([text])
            
            # Generate embedding — reuse cached transformer
            text_embedding = self._get_sentence_transformer().encode([text])
            
            ood_scores = {}
            ood_flags = []
            
            # 1. SVM Decision Function Distance
            decision_distance = abs(self.model.decision_function(text_features)[0])
            ood_scores['svm_distance'] = decision_distance
            ood_flags.append(decision_distance < 0.3)
            
            # 2. Nearest Neighbors Distance
            nn_distances, _ = self.nn_detector.kneighbors(text_embedding)
            avg_nn_distance = np.mean(nn_distances[0])
            ood_scores['nn_distance'] = avg_nn_distance
            ood_flags.append(avg_nn_distance > 0.65)
            
            # 3. Isolation Forest
            isolation_score = self.isolation_forest.decision_function(text_embedding)[0]
            ood_scores['isolation_score'] = isolation_score
            ood_flags.append(isolation_score < -0.05)
            
            # Majority vote: 2 out of 3 methods must agree
            ood_count = sum(ood_flags)
            ood_confidence = ood_count / len(ood_flags)
            is_ood = ood_count >= 2
            
            return {
                'is_ood': is_ood,
                'ood_confidence': ood_confidence,
                'ood_scores': ood_scores,
                'ood_flags': ood_flags,
                'reason': f'{ood_count}/{len(ood_flags)} methods flagged as OOD'
            }
            
        except Exception as e:
            return {
                'is_ood': False,
                'ood_confidence': 0.0,
                'ood_scores': {},
                'reason': f'OOD detection failed: {e}'
            }
    
    def unified_decision_analysis(self, text):
        """
        Comprehensive analysis using all three systems.
        
        Returns detailed analysis from K-Medoids, Calibration, and OOD Detection.
        """
        analysis = {
            'input_text': text[:100] + '...' if len(text) > 100 else text,
            'domain': self.domain,
            'timestamp': pd.Timestamp.now().isoformat(),
            'systems_analysis': {},
            'unified_scores': {},
            'recommendation': {}
        }
        
        # 1. K-Medoids Similarity Analysis
        similarity_score = self.calculate_similarity_to_centroid(text)
        analysis['systems_analysis']['k_medoids'] = {
            'similarity_score': similarity_score,
            'num_medoids': len(self.medoids) if hasattr(self, 'medoids') else 1,
            'primary_medoid_text': self.medoid_text[:50] + '...' if self.medoid_text else 'N/A'
        }
        
        # 2. Calibration Analysis
        confidence_result = self.get_confidence_prediction(text)
        analysis['systems_analysis']['calibration'] = {
            'enabled': self.enable_calibration,
            'prediction': confidence_result.get('prediction'),
            'confidence_score': confidence_result.get('confidence', 0.0),
            'raw_confidence': confidence_result.get('raw_confidence', 0.0),
            'calibration_quality': self.calibration_score
        }
        
        # 3. OOD Detection Analysis
        ood_result = self.detect_ood(text)
        analysis['systems_analysis']['ood_detection'] = {
            'enabled': self.enable_ood_detection,
            'is_ood': ood_result['is_ood'],
            'ood_confidence': ood_result['ood_confidence'],
            'ood_scores': ood_result['ood_scores'],
            'reason': ood_result['reason']
        }
        
        # 4. Unified Scoring
        analysis['unified_scores'] = self._calculate_unified_scores(
            similarity_score, 
            confidence_result.get('confidence', 0.0),
            ood_result
        )
        
        # 5. Final Recommendation
        analysis['recommendation'] = self._make_unified_recommendation(analysis['unified_scores'])
        
        return analysis
    
    def _calculate_unified_scores(self, similarity, confidence, ood_result):
        """Calculate unified scores combining all three systems."""
        
        base_similarity = similarity
        base_confidence = confidence
        
        if ood_result['is_ood']:
            ood_penalty = ood_result['ood_confidence'] * 0.5
        else:
            ood_penalty = ood_result['ood_confidence'] * 0.2
        
        adjusted_confidence = confidence * (1 - ood_penalty)
        adjusted_similarity = similarity * (1 - 0.3 * ood_penalty)
        
        composite_score = (adjusted_similarity * 0.45 + 
                          adjusted_confidence * 0.45 + 
                          (1 - ood_penalty) * 0.1)
        
        quality_indicators = []
        if hasattr(self, 'medoids') and len(self.medoids) >= 5:
            quality_indicators.append(0.2)
        if self.enable_calibration and self.calibration_score > 0.7:
            quality_indicators.append(0.3)
        if self.enable_ood_detection:
            quality_indicators.append(0.2)
        
        quality_score = sum(quality_indicators) + 0.3
        
        return {
            'base_similarity': base_similarity,
            'base_confidence': base_confidence,
            'ood_penalty': ood_penalty,
            'adjusted_similarity': adjusted_similarity,
            'adjusted_confidence': adjusted_confidence,
            'composite_score': composite_score,
            'quality_score': min(1.0, quality_score),
            'systems_enabled': {
                'k_medoids': True,
                'calibration': self.enable_calibration,
                'ood_detection': self.enable_ood_detection
            }
        }
    
    def _make_unified_recommendation(self, unified_scores):
        """Make final recommendation based on unified analysis."""
        
        composite = unified_scores['composite_score']
        quality = unified_scores['quality_score']
        ood_penalty = unified_scores['ood_penalty']
        
        high_threshold = 0.5 * quality
        medium_threshold = 0.25 * quality
        ood_rejection_threshold = 0.6
        
        recommendation = {
            'decision': 'create_new_expert',
            'confidence_in_decision': 0.0,
            'reasoning': [],
            'thresholds_used': {
                'high_threshold': high_threshold,
                'medium_threshold': medium_threshold,
                'ood_rejection_threshold': ood_rejection_threshold
            }
        }
        
        if ood_penalty > ood_rejection_threshold:
            recommendation['decision'] = 'create_new_expert'
            recommendation['confidence_in_decision'] = 0.9
            recommendation['reasoning'].append(f"High OOD penalty ({ood_penalty:.3f}) suggests input is out-of-distribution")
        
        elif composite >= high_threshold and ood_penalty < 0.45:
            recommendation['decision'] = 'use_existing_expert'
            recommendation['confidence_in_decision'] = composite * quality
            recommendation['reasoning'].append(f"High composite score ({composite:.3f}) with acceptable OOD risk")
            
        elif composite >= medium_threshold and ood_penalty < 0.55:
            recommendation['decision'] = 'create_new_patch'
            recommendation['confidence_in_decision'] = composite * quality * 0.8
            recommendation['reasoning'].append(f"Medium composite score ({composite:.3f}) suitable for patch creation")
            
        else:
            recommendation['decision'] = 'create_new_expert'
            recommendation['confidence_in_decision'] = 0.7
            recommendation['reasoning'].append(f"Low composite score ({composite:.3f}) or high OOD risk")
        
        if unified_scores['adjusted_similarity'] < 0.2:
            recommendation['reasoning'].append("Low similarity to existing medoids")
        
        if self.enable_calibration and unified_scores['base_confidence'] < 0.5:
            recommendation['reasoning'].append("Low calibrated confidence")
        
        if unified_scores['ood_penalty'] > 0.3:
            recommendation['reasoning'].append("OOD detection flagged potential distribution shift")
        
        if quality > 0.8:
            recommendation['reasoning'].append("High-quality analysis (all systems enabled and well-calibrated)")
        elif quality < 0.5:
            recommendation['reasoning'].append("Lower quality analysis (some systems disabled or poorly calibrated)")
        
        return recommendation
    
    def get_system_status(self):
        """Get comprehensive status of all three systems."""
        return {
            'domain': self.domain,
            'systems_status': self.system_stats,
            'k_medoids': {
                'num_medoids': len(self.medoids) if hasattr(self, 'medoids') else 0,
                'primary_medoid_available': self.medoid is not None
            },
            'calibration': {
                'enabled': self.enable_calibration,
                'calibration_score': self.calibration_score,
                'model_type': 'CalibratedClassifierCV' if self.enable_calibration else 'Raw SVM'
            },
            'ood_detection': {
                'enabled': self.enable_ood_detection,
                'methods': ['svm_distance', 'nn_distance', 'isolation_forest'] if self.enable_ood_detection else [],
                'training_samples': (
                    len(self.training_embeddings)
                    if getattr(self, "training_embeddings", None) is not None
                    else 0
                )
            }
        }
    
    def __repr__(self):
        """String representation showing all enabled systems."""
        systems = []
        systems.append("K-Medoids")
        if self.enable_calibration:
            systems.append("Calibration")
        if self.enable_ood_detection:
            systems.append("OOD-Detection")
        
        return f"UnifiedExpert(domain='{self.domain}', systems=[{', '.join(systems)}])"


def create_unified_expert_from_domain_folder(domain_folder_path, domain_name, 
                                           text_column='sentence', 
                                           enable_calibration=True, 
                                           enable_ood_detection=True):
    """
    Create a unified expert from a domain folder.
    """
    model_file = None
    vectorizer_file = None
    dataset_file = None
    
    for file in os.listdir(domain_folder_path):
        if file.startswith('svm_model_') and file.endswith('.pkl'):
            model_file = os.path.join(domain_folder_path, file)
        elif 'vectorizer' in file and file.endswith('.pkl'):
            vectorizer_file = os.path.join(domain_folder_path, file)
        elif file.endswith('.csv'):
            dataset_file = os.path.join(domain_folder_path, file)
    
    if not all([model_file, dataset_file]):
        raise FileNotFoundError(f"Required files not found in {domain_folder_path}")
    
    return UnifiedExpert(
        domain=domain_name,
        model_path=model_file,
        dataset_path=dataset_file,
        text_column=text_column,
        vectorizer_path=vectorizer_file,
        enable_calibration=enable_calibration,
        enable_ood_detection=enable_ood_detection
    )


def initialize_unified_experts(enable_calibration=True, enable_ood_detection=True):
    """
    Initialize unified experts for all available domains (SVM and BERT).
    """
    experts = {}
    base_dir = os.path.dirname(__file__)
    dummy_models_dir = os.path.join(base_dir, 'dummy_models')
    
    svm_domain_configs = {
        'music': {
            'folder': 'Music', 
            'text_column': 'sentence',
            'type': 'svm'
        }
    }
    
    bert_domain_configs = {
        'physics': {
            'folder': 'Physics_BERT',
            'text_column': 'text',
            'type': 'bert'
        },
        'chemistry': {
            'folder': 'Chemistry_BERT',
            'text_column': 'text',
            'type': 'bert'
        },
        'medical': {
            'folder': 'Medical_BERT',
            'text_column': 'Text',
            'type': 'bert'
        }
    }
    
    for domain_name, config in svm_domain_configs.items():
        domain_folder = os.path.join(dummy_models_dir, config['folder'])
        if os.path.exists(domain_folder):
            try:
                expert = create_unified_expert_from_domain_folder(
                    domain_folder,
                    domain_name,
                    config['text_column'],
                    enable_calibration,
                    enable_ood_detection
                )
                experts[domain_name] = expert
                print(f"\u2705 Unified SVM expert created for {domain_name}")
            except Exception as e:
                print(f"\u274c Failed to create SVM expert for {domain_name}: {e}")
    
    from unified_bert_expert import create_unified_bert_expert_from_folder
    
    for domain_name, config in bert_domain_configs.items():
        domain_folder = os.path.join(dummy_models_dir, config['folder'])
        if os.path.exists(domain_folder):
            try:
                expert = create_unified_bert_expert_from_folder(
                    domain_folder,
                    domain_name,
                    config['text_column'],
                    enable_calibration,
                    enable_ood_detection
                )
                experts[domain_name] = expert
                print(f"\u2705 Unified BERT expert created for {domain_name}")
            except Exception as e:
                print(f"\u274c Failed to create BERT expert for {domain_name}: {e}")
    
    return experts


def make_unified_expert_decision(input_text, experts, 
                               similarity_threshold_high=0.3,
                               similarity_threshold_medium=0.2):
    """
    Make expert decision using unified analysis from all experts.
    """
    
    decision_result = {
        "input_text": input_text,
        "timestamp": pd.Timestamp.now().isoformat(),
        "expert_analyses": {},
        "unified_decision": {},
        "system_summary": {
            "experts_analyzed": len(experts),
            "systems_enabled": {
                "k_medoids": True,
                "calibration": any(expert.enable_calibration for expert in experts.values()),
                "ood_detection": any(expert.enable_ood_detection for expert in experts.values())
            }
        }
    }
    
    expert_results = []
    for domain, expert in experts.items():
        analysis = expert.unified_decision_analysis(input_text)
        decision_result["expert_analyses"][domain] = analysis
        expert_results.append((domain, expert, analysis))
    
    best_expert = None
    best_composite_score = 0
    best_analysis = None
    
    for domain, expert, analysis in expert_results:
        composite_score = analysis['unified_scores']['composite_score']
        quality_score = analysis['unified_scores']['quality_score']
        weighted_score = composite_score * quality_score
        
        if weighted_score > best_composite_score:
            best_composite_score = weighted_score
            best_expert = expert
            best_analysis = analysis
    
    if best_analysis:
        decision_result["unified_decision"] = {
            "selected_domain": best_expert.domain if best_expert else None,
            "composite_score": best_composite_score,
            "recommendation": best_analysis['recommendation'],
            "quality_score": best_analysis['unified_scores']['quality_score']
        }
    else:
        decision_result["unified_decision"] = {
            "selected_domain": None,
            "composite_score": 0.0,
            "recommendation": {"decision": "create_new_expert"},
            "quality_score": 0.0
        }
    
    return decision_result


# ---------------------------------------------------------------------------
# UnifiedExpertSystem — top-level orchestrator
# ---------------------------------------------------------------------------

class UnifiedExpertSystem:
    """
    Orchestrates multiple UnifiedExpert instances across all domains.

    M5 §9.1 — Experts are held in an _ExpertLRUCache that evicts the
    least-recently-used expert to disk when the in-memory count exceeds
    ``max_resident_experts`` (default 8, configurable in config.toml
    under ``[unified_expert] max_resident_experts``).
    """

    def __init__(self, enable_calibration=True, enable_ood_detection=True):
        """Initialize the unified expert system with multiple experts."""
        self.enable_calibration = enable_calibration
        self.enable_ood_detection = enable_ood_detection

        # Resolve max_resident_experts from config with safe fallbacks
        try:
            import config_loader as _cfg
            if hasattr(_cfg, 'get'):
                self._max_resident = int(_cfg.get("unified_expert", "max_resident_experts", 8))
            elif hasattr(_cfg, 'UNIFIED_EXPERT_CONFIG'):
                self._max_resident = int(_cfg.UNIFIED_EXPERT_CONFIG.get("max_resident_experts", 8))
            else:
                self._max_resident = 8
        except Exception:
            self._max_resident = 8

        # _raw_experts is populated by _initialize_all_experts(); then wrapped
        self._raw_experts: dict = {}
        self._initialize_all_experts()

        # Wrap in LRU cache after all experts are loaded so the initial
        # population doesn't trigger spurious evictions.
        self.experts = _ExpertLRUCache(
            self._raw_experts,
            max_k=self._max_resident,
            evict_dir=os.path.dirname(os.path.abspath(__file__)),
        )

    def _initialize_all_experts(self):
        """Initialize all available expert domains."""
        print("\U0001f680 UNIFIED EXPERT SYSTEM INITIALIZATION")
        print("="*60)
        
        try:
            self._raw_experts = initialize_unified_experts(
                enable_calibration=self.enable_calibration,
                enable_ood_detection=self.enable_ood_detection
            )
            print(f"\n\u2705 Total experts loaded: {len(self._raw_experts)}")
            for domain in self._raw_experts:
                print(f"   - {domain}: {self._raw_experts[domain]}")
        except Exception as e:
            print(f"\u274c Expert initialization failed: {e}")
            self._raw_experts = {}

    def _registered_domains(self):
        """Return the set of all registered domain names."""
        return set(self._raw_experts.keys())

    def analyze_query(self, input_text):
        """Analyze a query against all experts and return the best decision."""
        if not self.experts:
            return {
                "decision": "no_experts_available",
                "selected_domain": None,
                "confidence": 0.0
            }
        
        return make_unified_expert_decision(input_text, self.experts)

    def get_system_status(self):
        """Get status of the entire expert system."""
        resident_count = len(self.experts._loaded) if hasattr(self.experts, '_loaded') else len(self._raw_experts)
        evicted_count = len(self.experts._evicted_domains()) if hasattr(self.experts, '_evicted_domains') else 0
        return {
            "total_experts": len(self._raw_experts),
            "resident_experts": resident_count,
            "evicted_experts": evicted_count,
            "max_resident": self._max_resident,
            "domains": list(self._raw_experts.keys()),
            "calibration_enabled": self.enable_calibration,
            "ood_detection_enabled": self.enable_ood_detection,
        }


_unified_system: UnifiedExpertSystem = None


def get_unified_expert_system(
    enable_calibration: bool = True,
    enable_ood_detection: bool = True,
) -> UnifiedExpertSystem:
    """Return the module-level singleton UnifiedExpertSystem."""
    global _unified_system
    if _unified_system is None:
        _unified_system = UnifiedExpertSystem(
            enable_calibration=enable_calibration,
            enable_ood_detection=enable_ood_detection,
        )
    return _unified_system
