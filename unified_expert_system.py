#!/usr/bin/env python3

"""
Unified Expert Pre-Check Layer: Integration of K-Medoids, Calibration, and OOD Detection

This module combines three powerful systems:
1. K-Medoids clustering for domain representation and similarity
2. Calibration for well-calibrated confidence scores
3. OOD detection for distribution shift identification

The unified system provides the most robust expert decision-making possible.
"""

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
        
        # Statistics for analysis
        self.system_stats = {
            'k_medoids_enabled': True,
            'calibration_enabled': enable_calibration,
            'ood_detection_enabled': enable_ood_detection,
            'initialization_status': {}
        }
        
        # Initialize all systems
        self._initialize_unified_expert()
    
    def _initialize_unified_expert(self):
        """Initialize all three systems in sequence."""
        print(f"\n🚀 Initializing Unified Expert for {self.domain} domain...")
        print("="*60)
        
        try:
            # 1. Core model and vectorizer
            print("1️⃣ Loading core SVM model and vectorizer...")
            self._load_trained_model()
            self._load_vectorizer()
            self.system_stats['initialization_status']['core_model'] = '✅ Success'
            
            # 2. K-Medoids clustering system
            print("2️⃣ Setting up K-Medoids clustering...")
            self._setup_k_medoids_system()
            self.system_stats['initialization_status']['k_medoids'] = '✅ Success'
            
            # 3. Calibration system (optional)
            if self.enable_calibration:
                print("3️⃣ Setting up calibration system...")
                self._setup_calibration_system()
                self.system_stats['initialization_status']['calibration'] = '✅ Success'
            else:
                print("3️⃣ Calibration disabled - using fallback confidence")
                self._setup_fallback_confidence()
                self.system_stats['initialization_status']['calibration'] = '⚠️ Disabled'
            
            # 4. OOD Detection system (optional)
            if self.enable_ood_detection:
                print("4️⃣ Setting up OOD detection system...")
                self._setup_ood_detection_system()
                self.system_stats['initialization_status']['ood_detection'] = '✅ Success'
            else:
                print("4️⃣ OOD detection disabled")
                self.system_stats['initialization_status']['ood_detection'] = '⚠️ Disabled'
            
            print(f"🎉 Unified Expert initialization complete for {self.domain}!")
            print("="*60)
            
        except Exception as e:
            print(f"❌ Error initializing unified expert for {self.domain}: {e}")
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
                print(f"   ⚠️ Warning: Insufficient samples for calibration (min: {min_count})")
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
            
            # Create calibrated classifier
            self.calibrated_model = CalibratedClassifierCV(
                self.model, method='isotonic', cv='prefit'
            )
            
            # Fit calibration on validation set
            self.calibrated_model.fit(X_val_tfidf, y_val)
            
            # Evaluate calibration
            calibrated_proba = self.calibrated_model.predict_proba(X_test_tfidf)
            y_pred_calibrated = self.calibrated_model.predict(X_test_tfidf)
            
            self.calibration_score = self._evaluate_calibration(y_test, y_pred_calibrated, calibrated_proba)
            print(f"   Calibration score: {self.calibration_score:.4f}")
            
        except Exception as e:
            print(f"   ⚠️ Calibration setup failed: {e}")
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
            
            # Generate embeddings
            sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            self.training_embeddings = sentence_transformer.encode(training_texts)
            
            # Setup OOD detectors
            self.isolation_forest = IsolationForest(contamination=0.1, random_state=42, n_estimators=50)
            self.isolation_forest.fit(self.training_embeddings)
            
            self.nn_detector = NearestNeighbors(n_neighbors=3, metric='cosine')
            self.nn_detector.fit(self.training_embeddings)
            
            print(f"   OOD detection ready (training sample: {len(training_texts)})")
            
        except Exception as e:
            print(f"   ⚠️ OOD detection setup failed: {e}")
            self.enable_ood_detection = False
    
    def _calculate_centroid(self, model_name="all-MiniLM-L6-v2", k=10):
        """Calculate centroid using K-Medoids clustering."""
        # Load dataset
        df = pd.read_csv(self.dataset_path)
        texts = df[self.text_column].dropna().tolist()
        
        # Generate embeddings
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts)
        
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
        model = SentenceTransformer(model_name)
        text_embedding = model.encode([text])
        
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
            
            # Generate embedding
            sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            text_embedding = sentence_transformer.encode([text])
            
            ood_scores = {}
            ood_flags = []
            
            # 1. SVM Decision Function Distance
            decision_distance = abs(self.model.decision_function(text_features)[0])
            ood_scores['svm_distance'] = decision_distance
            ood_flags.append(decision_distance < 0.5)  # More sensitive threshold
            
            # 2. Nearest Neighbors Distance
            nn_distances, _ = self.nn_detector.kneighbors(text_embedding)
            avg_nn_distance = np.mean(nn_distances[0])
            ood_scores['nn_distance'] = avg_nn_distance
            ood_flags.append(avg_nn_distance > 0.5)  # More sensitive threshold
            
            # 3. Isolation Forest
            isolation_score = self.isolation_forest.decision_function(text_embedding)[0]
            ood_scores['isolation_score'] = isolation_score
            ood_flags.append(isolation_score < 0.0)  # More sensitive threshold
            
            # Calculate overall OOD decision
            ood_count = sum(ood_flags)
            ood_confidence = ood_count / len(ood_flags)
            is_ood = ood_count >= 1  # More sensitive: any method flagging as OOD
            
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
        
        # Base scores
        base_similarity = similarity
        base_confidence = confidence
        # Always apply OOD penalty based on confidence, not just when is_ood is True
        ood_penalty = ood_result['ood_confidence'] * 0.8  # Scale the penalty
        
        # Adjusted scores (apply OOD penalty)
        adjusted_confidence = confidence * (1 - ood_penalty)
        adjusted_similarity = similarity * (1 - 0.5 * ood_penalty)  # Less penalty for similarity
        
        # Composite score (combination of all factors)
        composite_score = (adjusted_similarity * 0.4 + 
                          adjusted_confidence * 0.4 + 
                          (1 - ood_penalty) * 0.2)
        
        # Quality score (confidence in the composite score)
        quality_indicators = []
        if hasattr(self, 'medoids') and len(self.medoids) >= 5:
            quality_indicators.append(0.2)  # Good medoid coverage
        if self.enable_calibration and self.calibration_score > 0.7:
            quality_indicators.append(0.3)  # Good calibration
        if self.enable_ood_detection:
            quality_indicators.append(0.2)  # OOD protection enabled
        
        quality_score = sum(quality_indicators) + 0.3  # Base quality
        
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
        
        # Decision thresholds (adaptive based on quality)
        high_threshold = 0.6 * quality  # Higher quality allows lower thresholds
        medium_threshold = 0.3 * quality
        ood_rejection_threshold = 0.4  # More sensitive to OOD
        
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
        
        # Decision logic with explicit reasoning
        if ood_penalty > ood_rejection_threshold:
            recommendation['decision'] = 'create_new_expert'
            recommendation['confidence_in_decision'] = 0.9
            recommendation['reasoning'].append(f"High OOD penalty ({ood_penalty:.3f}) suggests input is out-of-distribution")
        
        elif composite >= high_threshold and ood_penalty < 0.2:  # More conservative OOD threshold
            recommendation['decision'] = 'use_existing_expert'
            recommendation['confidence_in_decision'] = composite * quality
            recommendation['reasoning'].append(f"High composite score ({composite:.3f}) with low OOD risk")
            
        elif composite >= medium_threshold and ood_penalty < 0.3:  # More conservative OOD threshold
            recommendation['decision'] = 'create_new_patch'
            recommendation['confidence_in_decision'] = composite * quality * 0.8
            recommendation['reasoning'].append(f"Medium composite score ({composite:.3f}) suitable for patch creation")
            
        else:
            recommendation['decision'] = 'create_new_expert'
            recommendation['confidence_in_decision'] = 0.7
            recommendation['reasoning'].append(f"Low composite score ({composite:.3f}) or high OOD risk")
        
        # Add system-specific reasoning
        if unified_scores['adjusted_similarity'] < 0.2:
            recommendation['reasoning'].append("Low similarity to existing medoids")
        
        if self.enable_calibration and unified_scores['base_confidence'] < 0.5:
            recommendation['reasoning'].append("Low calibrated confidence")
        
        if unified_scores['ood_penalty'] > 0.3:
            recommendation['reasoning'].append("OOD detection flagged potential distribution shift")
        
        # Quality assessment
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
                'training_samples': len(self.training_embeddings) if hasattr(self, 'training_embeddings') else 0
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
    
    Args:
        domain_folder_path: Path to domain folder containing model files
        domain_name: Name of the domain
        text_column: Name of text column in dataset
        enable_calibration: Whether to enable calibration
        enable_ood_detection: Whether to enable OOD detection
    
    Returns:
        UnifiedExpert instance
    """
    
    # Find required files
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
    Initialize unified experts for all available domains.
    
    Args:
        enable_calibration: Whether to enable calibration for all experts
        enable_ood_detection: Whether to enable OOD detection for all experts
    
    Returns:
        Dictionary mapping domain names to UnifiedExpert instances
    """
    experts = {}
    base_dir = os.path.dirname(__file__)
    dummy_models_dir = os.path.join(base_dir, 'dummy_models')
    
    # Domain configurations
    domain_configs = {
        'medical': {
            'folder': 'Medical',
            'text_column': 'sentence'
        },
        'music': {
            'folder': 'Music', 
            'text_column': 'sentence'
        },
        'physics': {
            'folder': 'Physics',
            'text_column': 'Comment'
        }
    }
    
    for domain_name, config in domain_configs.items():
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
                print(f"✅ Unified expert created for {domain_name}")
            except Exception as e:
                print(f"❌ Failed to create unified expert for {domain_name}: {e}")
    
    return experts


def make_unified_expert_decision(input_text, experts, 
                               similarity_threshold_high=0.3,
                               similarity_threshold_medium=0.2):
    """
    Make expert decision using unified analysis from all experts.
    
    Args:
        input_text: Text to analyze
        experts: Dictionary of UnifiedExpert instances
        similarity_threshold_high: High similarity threshold
        similarity_threshold_medium: Medium similarity threshold
    
    Returns:
        Comprehensive decision with analysis from all systems
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
    
    # Analyze with each expert
    expert_results = []
    for domain, expert in experts.items():
        analysis = expert.unified_decision_analysis(input_text)
        decision_result["expert_analyses"][domain] = analysis
        expert_results.append((domain, expert, analysis))
    
    # Find best expert based on unified scores
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
    
    # Make unified decision
    if best_analysis:
        decision_result["unified_decision"] = {
            "selected_domain": best_analysis['domain'],
            "decision_flag": best_analysis['recommendation']['decision'],
            "confidence_in_decision": best_analysis['recommendation']['confidence_in_decision'],
            "composite_score": best_analysis['unified_scores']['composite_score'],
            "quality_score": best_analysis['unified_scores']['quality_score'],
            "reasoning": best_analysis['recommendation']['reasoning'],
            "ood_analysis": {
                "is_ood": best_analysis['systems_analysis']['ood_detection']['is_ood'],
                "ood_confidence": best_analysis['systems_analysis']['ood_detection']['ood_confidence']
            },
            "similarity_analysis": {
                "similarity_score": best_analysis['systems_analysis']['k_medoids']['similarity_score'],
                "adjusted_similarity": best_analysis['unified_scores']['adjusted_similarity']
            },
            "confidence_analysis": {
                "calibrated_confidence": best_analysis['systems_analysis']['calibration']['confidence_score'],
                "adjusted_confidence": best_analysis['unified_scores']['adjusted_confidence']
            }
        }
    else:
        decision_result["unified_decision"] = {
            "decision_flag": "create_new_expert",
            "reasoning": ["No suitable expert found"],
            "confidence_in_decision": 0.9
        }
    
    return decision_result


class UnifiedExpertSystem:
    """
    System manager for multiple UnifiedExpert instances.
    Provides simplified interface for workflow integration.
    """
    
    def __init__(self, enable_calibration=True, enable_ood_detection=True):
        """Initialize the unified expert system with multiple experts."""
        self.enable_calibration = enable_calibration
        self.enable_ood_detection = enable_ood_detection
        self.experts = {}
        self._initialize_all_experts()
    
    def _initialize_all_experts(self):
        """Initialize all available expert domains."""
        print("🚀 UNIFIED EXPERT SYSTEM INITIALIZATION")
        print("="*60)
        
        try:
            self.experts = initialize_unified_experts(
                enable_calibration=self.enable_calibration,
                enable_ood_detection=self.enable_ood_detection
            )
            print(f"\n✅ Successfully initialized {len(self.experts)} unified experts")
            
            # Print system status
            for domain, expert in self.experts.items():
                status = expert.get_system_status()
                print(f"\n🔧 {domain.upper()} Expert:")
                print(f"   K-Medoids: {status['k_medoids']['num_medoids']} medoids")
                print(f"   Calibration: {'✅ Enabled' if status['calibration']['enabled'] else '⚠️ Disabled'} (score: {status['calibration']['calibration_score']:.3f})")
                print(f"   OOD Detection: {'✅ Enabled' if status['ood_detection']['enabled'] else '⚠️ Disabled'} ({len(status['ood_detection']['methods'])} methods)")
                
        except Exception as e:
            print(f"❌ Failed to initialize unified expert system: {e}")
            raise
    
    def unified_decision_analysis(self, input_text):
        """
        Analyze input text with all experts and return unified decision.
        
        Args:
            input_text: Text to analyze
            
        Returns:
            Dictionary with decision analysis from all three systems
        """
        if not self.experts:
            return {
                'decision': {
                    'decision_flag': 'create_new_expert',
                    'selected_domain': 'unknown',
                    'confidence_in_decision': 0.9,
                    'reasoning': ['No experts available']
                }
            }
        
        # Use the unified decision function
        return make_unified_expert_decision(input_text, self.experts)
    
    def get_system_status(self):
        """Get status of all experts in the system."""
        return {
            'total_experts': len(self.experts),
            'expert_domains': list(self.experts.keys()),
            'system_configuration': {
                'calibration_enabled': self.enable_calibration,
                'ood_detection_enabled': self.enable_ood_detection
            },
            'expert_details': {
                domain: expert.get_system_status() 
                for domain, expert in self.experts.items()
            }
        }
    
    def __repr__(self):
        """String representation of the system."""
        return f"UnifiedExpertSystem(experts={len(self.experts)}, calibration={self.enable_calibration}, ood={self.enable_ood_detection})"


if __name__ == "__main__":
    print("🚀 UNIFIED EXPERT SYSTEM - K-MEDOIDS + CALIBRATION + OOD DETECTION")
    print("="*80)
    
    # Test with different configurations
    test_configs = [
        {"calibration": True, "ood": True, "name": "Full System"},
        {"calibration": False, "ood": True, "name": "K-Medoids + OOD"},
        {"calibration": True, "ood": False, "name": "K-Medoids + Calibration"}
    ]
    
    for config in test_configs:
        print(f"\n🧪 Testing Configuration: {config['name']}")
        print("-" * 50)
        
        try:
            system = UnifiedExpertSystem(
                enable_calibration=config['calibration'],
                enable_ood_detection=config['ood']
            )
            
            # Test with sample text
            test_text = "The patient showed symptoms of acute myocardial infarction."
            result = system.unified_decision_analysis(test_text)
            
            print(f"📊 Decision: {result['unified_decision']['decision_flag']}")
            print(f"🎯 Domain: {result['unified_decision']['selected_domain']}")
            print(f"🔢 Confidence: {result['unified_decision']['confidence_in_decision']:.3f}")
            
        except Exception as e:
            print(f"❌ Configuration failed: {e}")
    
    print(f"\n🎉 Unified Expert System testing complete!")