#!/usr/bin/env python3

"""
Unified BERT Expert - Extends UnifiedExpert for BERT-based models

This module provides a BERT-compatible version of UnifiedExpert that:
1. Inherits all 3-phase decision logic (K-Medoids, Calibration, OOD)
2. Only overrides model loading and prediction methods
3. Maintains identical decision flags and behavior as SVM experts
"""

import logging
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import pandas as pd
from unified_expert_system import UnifiedExpert
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Minimum free VRAM (bytes) required before we attempt to place BERT on the
# GPU.  BERT-base-uncased is ~420 MiB; 512 MiB gives a small safety margin.
# If less than this is free we fall back to CPU automatically.
_MIN_VRAM_BYTES = 512 * 1024 * 1024   # 512 MiB


def _safe_device() -> torch.device:
    """Return the safest device for BERT inference.

    Logic:
      1. If CUDA is unavailable -> CPU
      2. Query Ollama for resident models and evict them all (best-effort)
      3. Flush PyTorch's VRAM cache
      4. If free VRAM >= _MIN_VRAM_BYTES -> CUDA
      5. Otherwise -> CPU (fast enough on Zen 4 for BERT-base at seq=128)
    """
    if not torch.cuda.is_available():
        return torch.device('cpu')

    # Best-effort: evict any Ollama model from VRAM before measuring headroom.
    try:
        from expert_post_check.ollama_guard import evict_all
        evicted = evict_all()
        if evicted:
            logger.info("_safe_device: evicted Ollama models before BERT load: %s", evicted)
    except Exception as exc:
        logger.warning("_safe_device: ollama_guard.evict_all failed -- %s", exc)

    # Release any cached (but unused) VRAM back to the OS allocator.
    torch.cuda.empty_cache()

    free_bytes, _ = torch.cuda.mem_get_info(0)
    logger.info(
        "_safe_device: free VRAM = %.1f MiB (threshold = %.1f MiB)",
        free_bytes / 1024**2, _MIN_VRAM_BYTES / 1024**2,
    )

    if free_bytes >= _MIN_VRAM_BYTES:
        return torch.device('cuda')

    logger.warning(
        "_safe_device: insufficient free VRAM (%.1f MiB < %.1f MiB) -- "
        "falling back to CPU for BERT inference.",
        free_bytes / 1024**2, _MIN_VRAM_BYTES / 1024**2,
    )
    return torch.device('cpu')


class UnifiedBERTExpert(UnifiedExpert):
    """
    BERT-based expert that inherits all 3-phase decision logic from UnifiedExpert.
    
    Only the model loading and prediction methods are overridden.
    Everything else (K-Medoids, Calibration, OOD, decision logic) is inherited unchanged.
    """
    
    def __init__(self, domain, model_path, dataset_path, text_column='text', 
                 enable_calibration=True, enable_ood_detection=True):
        """
        Initialize BERT expert with 3-phase system.
        
        Args:
            domain: Expert domain (e.g., 'physics', 'chemistry')
            model_path: Path to BERT model directory (contains config.json, pytorch_model.bin, etc.)
            dataset_path: Path to training dataset CSV
            text_column: Name of text column in dataset
            enable_calibration: Whether to enable calibration system
            enable_ood_detection: Whether to enable OOD detection
        """
        # device is resolved lazily at model-load time via _safe_device();
        # set a placeholder here so the parent constructor does not crash if
        # it references self.device before _ensure_model_loaded() is called.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.bert_model = None
        self.max_length = 128
        self.model_loaded = False  # Lazy loading flag
        
        # Call parent constructor
        # This will call our overridden _load_trained_model() and _load_vectorizer()
        super().__init__(
            domain=domain,
            model_path=model_path,
            dataset_path=dataset_path,
            text_column=text_column,
            vectorizer_path=None,  # BERT doesn't use separate vectorizer
            enable_calibration=enable_calibration,
            enable_ood_detection=enable_ood_detection
        )
    
    def _load_trained_model(self):
        """
        Override: LAZY LOAD - Only load tokenizer for expert pre-check.
        Full BERT model is loaded only when actually needed (use_existing_expert).
        """
        import os
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"BERT model directory not found: {self.model_path}")
        
        print(f"   Preparing BERT expert from {self.model_path}...")
        # Only load tokenizer for pre-check phase (lightweight)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        
        # DO NOT load full BERT model yet (lazy loading)
        self.bert_model = None
        self.model_loaded = False
        
        # Store as self.model for compatibility with parent class methods
        self.model = self  # Point to self so parent methods can call predict()
        
        print(f"   ✅ BERT tokenizer loaded (model will be loaded on-demand)")
    
    def _ensure_model_loaded(self):
        """
        Lazy-load the full BERT model when actually needed.

        Before pushing weights to the GPU we:
          1. Query Ollama /api/ps and evict any resident model (ollama_guard)
          2. Flush PyTorch's VRAM cache
          3. Re-evaluate the safest device (_safe_device)

        If free VRAM is still below the 512 MiB threshold after eviction,
        BERT is placed on CPU instead.  On a Ryzen 9 8945HX inference at
        max_length=128 takes ~50-150 ms on CPU, which is acceptable.
        """
        if not self.model_loaded:
            # Re-evaluate device at load time (not at __init__ time) so we
            # always get an accurate picture of current VRAM availability.
            self.device = _safe_device()
            logger.info(
                "_ensure_model_loaded: loading BERT for domain '%s' on %s",
                self.domain, self.device,
            )
            print(f"   📦 Loading full BERT model for {self.domain} (on-demand) on {self.device}...")
            self.bert_model = BertForSequenceClassification.from_pretrained(self.model_path)
            self.bert_model.to(self.device)
            self.bert_model.eval()
            self.model_loaded = True
            print(f"   ✅ BERT model loaded on {self.device}")
    
    def _load_vectorizer(self):
        """Override: BERT uses tokenizer, not TF-IDF vectorizer."""
        # Create a dummy vectorizer object that has a transform method
        # This allows parent class methods that expect self.vectorizer to work
        class BERTTokenizerWrapper:
            def __init__(self, tokenizer, device, max_length):
                self.tokenizer = tokenizer
                self.device = device
                self.max_length = max_length
            
            def transform(self, texts):
                """
                Transform texts for BERT.
                Returns the texts themselves (BERT will tokenize them later).
                """
                return texts
        
        self.vectorizer = BERTTokenizerWrapper(self.tokenizer, self.device, self.max_length)
        print(f"   ✅ BERT tokenizer wrapper ready")
    
    def _find_text_column(self, df):
        """Find the text column in the dataset."""
        text_candidates = ['text', 'sentence', 'content', 'input', 'query']
        for col in text_candidates:
            if col in df.columns:
                return col
        # Return first string column
        for col in df.columns:
            if df[col].dtype == 'object':
                return col
        return df.columns[0]  # Fallback
    
    def _find_label_column(self, df):
        """Find the label column in the dataset."""
        label_candidates = ['label', 'target', 'class', 'category', 'y']
        for col in label_candidates:
            if col in df.columns:
                return col
        # Return last column as fallback
        return df.columns[-1]
    
    def _normalize_labels(self, labels):
        """
        Normalize labels to numeric format (0, 1).
        Handles both string labels (e.g., "Physics", "Not Physics") and numeric labels.
        
        Args:
            labels: List of labels (can be strings or numbers)
            
        Returns:
            List of numeric labels (0 or 1)
        """
        import numpy as np
        
        # If already numeric, convert to int
        if all(isinstance(label, (int, np.integer)) for label in labels):
            return [int(label) for label in labels]
        
        # If mixed or string, normalize
        # Assume positive class contains the domain name
        positive_keywords = [self.domain.lower(), '1', 'true', 'yes', 'positive']
        
        normalized = []
        for label in labels:
            label_str = str(label).lower().strip()
            
            # Check if it's a positive label
            is_positive = any(keyword in label_str for keyword in positive_keywords)
            normalized.append(1 if is_positive else 0)
        
        return normalized
    
    def predict(self, text):
        """
        Override: Make predictions using BERT instead of SVM.
        LAZY LOADS the model if not already loaded.
        
        Args:
            text: Input text string or list of texts
            
        Returns:
            Predicted class (integer or string depending on model)
        """
        # Ensure model is loaded
        self._ensure_model_loaded()
        
        # Handle both single text and list of texts
        if isinstance(text, str):
            texts = [text]
            single_input = True
        else:
            texts = text
            single_input = False
        
        # Tokenize
        encodings = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
        
        predictions = predictions.cpu().numpy()
        
        if single_input:
            return predictions[0]
        return predictions
    
    def decision_function(self, text):
        """
        Override: Get decision function scores (confidence) for BERT.
        Used by OOD detection.
        LAZY LOADS the model if not already loaded.
        
        Args:
            text: Input text (single string or list)
            
        Returns:
            Decision scores (numpy array)
        """
        # Ensure model is loaded
        self._ensure_model_loaded()
        
        # Handle both single text and list of texts
        if isinstance(text, str):
            texts = [text]
            single_input = True
        else:
            texts = text
            single_input = False
        
        # Tokenize
        encodings = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        # Get raw logits (before softmax)
        with torch.no_grad():
            outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Use max logit as decision score (similar to SVM decision function)
            scores = torch.max(logits, dim=1)[0]
        
        scores = scores.cpu().numpy()
        
        if single_input:
            return scores
        return scores
    
    def predict_proba(self, text):
        """
        Get probability predictions for BERT.
        Used by calibration system.
        LAZY LOADS the model if not already loaded.
        
        Args:
            text: Input text (single string or list)
            
        Returns:
            Probability array (n_samples, n_classes)
        """
        # Ensure model is loaded
        self._ensure_model_loaded()
        
        # Handle both single text and list of texts
        if isinstance(text, str):
            texts = [text]
            single_input = True
        else:
            texts = text
            single_input = False
        
        # Tokenize
        encodings = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        # Get probabilities
        with torch.no_grad():
            outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
        
        probs = probs.cpu().numpy()
        
        if single_input:
            return probs
        return probs
    
    def get_confidence_prediction(self, text):
        """
        Override: Get calibrated confidence prediction for BERT.
        
        This method is called by the parent class's unified_decision_analysis().
        
        LOADS THE MODEL and gets REAL per-input confidence predictions.
        This is necessary for accurate use_existing_expert decisions.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary with prediction and confidence (real inference)
        """
        # Load model and get REAL predictions (not proxy)
        # This is what we want for accurate decisions
        probs = self.predict_proba(text)
        prediction = self.predict(text)
        confidence = np.max(probs)
        
        # Apply calibration adjustment
        # The calibration_score represents model quality (validation accuracy)
        # We use it to scale the raw confidence appropriately
        if self.calibration_score is not None:
            # Scale confidence based on calibration quality
            # High calibration score = trust the confidence more
            # Low calibration score = be more conservative
            adjusted_confidence = confidence * (0.5 + 0.5 * self.calibration_score)
        else:
            adjusted_confidence = confidence
        
        return {
            'prediction': int(prediction),
            'confidence': float(adjusted_confidence),
            'raw_confidence': float(confidence),
            'calibration_score': float(self.calibration_score) if self.calibration_score else 0.5,
            'is_proxy': False  # This is real inference
        }
    
    def _setup_calibration_system(self):
        """
        Override: Setup calibration for BERT model using VALIDATION split.
        
        Strategy:
        1. Generate model fingerprint (hash of model files + config)
        2. Check for cached calibration metrics with matching fingerprint
        3. If cache invalid/missing, load model ONCE, compute metrics, cache them
        4. Use cached metrics for all future pre-checks (no model loading)
        """
        try:
            import os
            import pickle
            
            # Generate model fingerprint
            model_fingerprint = self._generate_model_fingerprint()
            
            # Check for cached calibration
            cache_path = os.path.join(self.model_path, 'calibration_cache.pkl')
            
            cache_valid = False
            if os.path.exists(cache_path):
                # Load and verify cached calibration metrics
                with open(cache_path, 'rb') as f:
                    cached_metrics = pickle.load(f)
                
                # Verify fingerprint matches
                cached_fingerprint = cached_metrics.get('model_fingerprint', {})
                if self._verify_fingerprint(model_fingerprint, cached_fingerprint):
                    self.calibration_score = cached_metrics.get('validation_accuracy', 0.5)
                    self.calibration_metrics = cached_metrics
                    cache_valid = True
                    
                    print(f"   ✅ Loaded cached calibration metrics")
                    print(f"   Validation accuracy: {self.calibration_score:.4f}")
                    print(f"   (Cached from {cached_metrics.get('num_samples', 'unknown')} validation samples)")
                    print(f"   Cache date: {cached_metrics.get('timestamp', 'unknown')}")
                else:
                    print(f"   ⚠️ Cached calibration fingerprint mismatch - model has changed")
                    print(f"   Will recompute calibration metrics...")
            
            if not cache_valid:
                # Need to compute calibration - do ONE-TIME model loading
                print(f"   📊 Computing calibration metrics (one-time setup)...")
                self._compute_and_cache_calibration(cache_path, model_fingerprint)
            
            # Store the model itself as calibrated_model for compatibility
            self.calibrated_model = self.model
            
        except Exception as e:
            print(f"   ⚠️ Calibration setup failed: {e}")
            import traceback
            traceback.print_exc()
            self._setup_fallback_confidence()
    
    def _generate_model_fingerprint(self):
        """
        Generate a unique fingerprint for the current model.
        
        Returns:
            Dictionary with model identification info
        """
        import os
        import hashlib
        import json
        
        fingerprint = {
            'model_path': self.model_path,
            'domain': self.domain,
            'cache_version': '1.0',  # Version of calibration method
        }
        
        # Hash model directory path
        path_hash = hashlib.md5(self.model_path.encode()).hexdigest()[:8]
        fingerprint['path_hash'] = path_hash
        
        # Get timestamps of key model files
        key_files = ['config.json', 'pytorch_model.bin', 'model.safetensors']
        file_timestamps = {}
        
        for filename in key_files:
            filepath = os.path.join(self.model_path, filename)
            if os.path.exists(filepath):
                mtime = os.path.getmtime(filepath)
                file_timestamps[filename] = mtime
        
        fingerprint['file_timestamps'] = file_timestamps
        
        # Hash config.json if available
        config_path = os.path.join(self.model_path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_content = f.read()
                config_hash = hashlib.md5(config_content.encode()).hexdigest()[:8]
                fingerprint['config_hash'] = config_hash
        
        # Create composite hash
        fingerprint_str = json.dumps(fingerprint, sort_keys=True)
        composite_hash = hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]
        fingerprint['composite_hash'] = composite_hash
        
        return fingerprint
    
    def _verify_fingerprint(self, current_fingerprint, cached_fingerprint):
        """
        Verify if cached fingerprint matches current model.
        
        Returns:
            True if fingerprints match (cache is valid)
        """
        # Check cache version
        if current_fingerprint.get('cache_version') != cached_fingerprint.get('cache_version'):
            return False
        
        # Check composite hash (most reliable)
        if current_fingerprint.get('composite_hash') != cached_fingerprint.get('composite_hash'):
            return False
        
        # Check path hash
        if current_fingerprint.get('path_hash') != cached_fingerprint.get('path_hash'):
            return False
        
        # Check config hash if available
        if 'config_hash' in current_fingerprint and 'config_hash' in cached_fingerprint:
            if current_fingerprint['config_hash'] != cached_fingerprint['config_hash']:
                return False
        
        # All checks passed
        return True
    
    def _compute_and_cache_calibration(self, cache_path, model_fingerprint):
        """
        Load model ONCE, compute real calibration metrics, cache them with fingerprint.
        This is a one-time operation per model.
        
        Args:
            cache_path: Path to save calibration cache
            model_fingerprint: Model identification dictionary
        """
        import os
        import pickle
        
        # Find validation dataset
        val_path = self.dataset_path.replace('train.csv', 'val.csv')
        if not os.path.exists(val_path):
            val_path = self.dataset_path.replace('train.csv', 'validation.csv')
        
        if not os.path.exists(val_path):
            print(f"   ⚠️ No validation set found, using dataset balance as proxy")
            # Fall back to simple balance calculation
            df = pd.read_csv(self.dataset_path)
            label_col = self._find_label_column(df)
            if label_col in df.columns:
                positive_ratio = np.mean(df[label_col].values == 1)
                self.calibration_score = 1 - abs(0.5 - positive_ratio) * 2
            else:
                self.calibration_score = 0.5
            
            # Cache even the fallback value with fingerprint
            calibration_metrics = {
                'validation_accuracy': self.calibration_score,
                'method': 'dataset_balance_fallback',
                'num_samples': len(df),
                'timestamp': pd.Timestamp.now().isoformat(),
                'model_fingerprint': model_fingerprint
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(calibration_metrics, f)
            
            return
        
        # Load validation data
        df = pd.read_csv(val_path)
        text_col = self._find_text_column(df)
        label_col = self._find_label_column(df)
        
        if text_col not in df.columns or label_col not in df.columns:
            print(f"   ⚠️ Could not find text/label columns")
            self.calibration_score = 0.5
            return
        
        # Load model for ONE-TIME calibration
        print(f"   🔧 Loading BERT model for calibration computation...")
        self._ensure_model_loaded()
        
        # Get predictions on validation set
        texts = df[text_col].tolist()[:500]  # Max 500 samples for speed
        labels_raw = df[label_col].tolist()[:500]
        
        # Normalize labels to numeric (0, 1)
        labels = self._normalize_labels(labels_raw)
        
        predictions = []
        confidences = []
        
        print(f"   🔍 Running inference on {len(texts)} validation samples...")
        for i, text in enumerate(texts):
            if i % 100 == 0 and i > 0:
                print(f"       Progress: {i}/{len(texts)} samples...")
            pred = self.predict(text)
            probs = self.predict_proba(text)
            predictions.append(int(pred))  # Ensure numeric
            confidences.append(np.max(probs))
        
        # Compute metrics
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        accuracy = accuracy_score(labels, predictions)
        
        # Handle binary vs multi-class
        if len(set(labels)) == 2:
            f1 = f1_score(labels, predictions, average='binary', zero_division=0)
            precision = precision_score(labels, predictions, average='binary', zero_division=0)
            recall = recall_score(labels, predictions, average='binary', zero_division=0)
        else:
            f1 = f1_score(labels, predictions, average='macro', zero_division=0)
            precision = precision_score(labels, predictions, average='macro', zero_division=0)
            recall = recall_score(labels, predictions, average='macro', zero_division=0)
        
        avg_confidence = np.mean(confidences)
        
        # Cache metrics with fingerprint
        calibration_metrics = {
            'validation_accuracy': accuracy,
            'validation_f1': f1,
            'validation_precision': precision,
            'validation_recall': recall,
            'avg_confidence': avg_confidence,
            'num_samples': len(texts),
            'method': 'full_validation_inference',
            'timestamp': pd.Timestamp.now().isoformat(),
            'model_fingerprint': model_fingerprint
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(calibration_metrics, f)
        
        self.calibration_score = accuracy
        self.calibration_metrics = calibration_metrics
        
        print(f"   ✅ Calibration computed and cached")
        print(f"   Validation accuracy: {accuracy:.4f}")
        print(f"   Validation F1: {f1:.4f}")
        print(f"   Validation precision: {precision:.4f}")
        print(f"   Validation recall: {recall:.4f}")
        print(f"   Average confidence: {avg_confidence:.4f}")
        print(f"   Fingerprint: {model_fingerprint['composite_hash']}")
        
        # Unload model to save memory (will be lazy-loaded when needed)
        print(f"   💾 Unloading model to save memory (will lazy-load when needed)")
        self.bert_model = None
        self.model_loaded = False
    
    def invalidate_calibration_cache(self):
        """
        Manually invalidate cached calibration metrics.
        Useful when you know the model has changed or want to force recomputation.
        """
        import os
        
        cache_path = os.path.join(self.model_path, 'calibration_cache.pkl')
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print(f"   🗑️ Calibration cache invalidated for {self.domain}")
            return True
        else:
            print(f"   ℹ️ No calibration cache found for {self.domain}")
            return False
    
    def get_calibration_cache_info(self):
        """
        Get information about the cached calibration metrics.
        
        Returns:
            Dictionary with cache info or None if no cache exists
        """
        import os
        import pickle
        
        cache_path = os.path.join(self.model_path, 'calibration_cache.pkl')
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                cached_metrics = pickle.load(f)
            
            return {
                'exists': True,
                'cache_path': cache_path,
                'timestamp': cached_metrics.get('timestamp', 'unknown'),
                'validation_accuracy': cached_metrics.get('validation_accuracy', 'unknown'),
                'num_samples': cached_metrics.get('num_samples', 'unknown'),
                'method': cached_metrics.get('method', 'unknown'),
                'fingerprint_hash': cached_metrics.get('model_fingerprint', {}).get('composite_hash', 'unknown')
            }
        else:
            return {'exists': False, 'cache_path': cache_path}
    
    def __repr__(self):
        """String representation."""
        systems = ["K-Medoids"]
        if self.enable_calibration:
            systems.append("Calibration")
        if self.enable_ood_detection:
            systems.append("OOD-Detection")
        
        return f"UnifiedBERTExpert(domain='{self.domain}', systems=[{', '.join(systems)}], device='{self.device}')"


def create_unified_bert_expert_from_folder(domain_folder_path, domain_name,
                                           text_column='text',
                                           enable_calibration=True,
                                           enable_ood_detection=True):
    """
    Create a UnifiedBERTExpert from a domain folder.
    
    Args:
        domain_folder_path: Path to folder containing BERT model and data
        domain_name: Name of the domain (e.g., 'physics', 'chemistry')
        text_column: Column name for text in dataset
        enable_calibration: Enable calibration
        enable_ood_detection: Enable OOD detection
        
    Returns:
        UnifiedBERTExpert instance
    """
    import os
    
    # BERT models are stored directly in the folder
    model_path = domain_folder_path
    
    # Look for dataset (prefer train.csv for medoid computation)
    possible_datasets = [
        os.path.join(domain_folder_path, 'train.csv'),
        os.path.join(domain_folder_path, f'{domain_name}_combined_dataset.csv'),
        os.path.join(domain_folder_path, 'dataset.csv')
    ]
    
    dataset_path = None
    for path in possible_datasets:
        if os.path.exists(path):
            dataset_path = path
            break
    
    if dataset_path is None:
        raise FileNotFoundError(f"No dataset found in {domain_folder_path}")
    
    return UnifiedBERTExpert(
        domain=domain_name,
        model_path=model_path,
        dataset_path=dataset_path,
        text_column=text_column,
        enable_calibration=enable_calibration,
        enable_ood_detection=enable_ood_detection
    )
