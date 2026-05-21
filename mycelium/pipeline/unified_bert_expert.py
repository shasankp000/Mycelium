#!/usr/bin/env python3
"""
Unified BERT Expert - Extends UnifiedExpert for BERT-based models
"""

import logging
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import pandas as pd
from mycelium.pipeline.unified_expert_system import UnifiedExpert
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

_MIN_VRAM_BYTES = 512 * 1024 * 1024  # 512 MiB


def _safe_device() -> torch.device:
    """Return the safest device for BERT inference."""
    if not torch.cuda.is_available():
        return torch.device('cpu')
    try:
        from expert_post_check.ollama_guard import evict_all
        evicted = evict_all()
        if evicted:
            logger.info("_safe_device: evicted Ollama models before BERT load: %s", evicted)
    except Exception as exc:
        logger.warning("_safe_device: ollama_guard.evict_all failed -- %s", exc)
    torch.cuda.empty_cache()
    free_bytes, _ = torch.cuda.mem_get_info(0)
    logger.info(
        "_safe_device: free VRAM = %.1f MiB (threshold = %.1f MiB)",
        free_bytes / 1024**2, _MIN_VRAM_BYTES / 1024**2,
    )
    if free_bytes >= _MIN_VRAM_BYTES:
        return torch.device('cuda')
    logger.warning(
        "_safe_device: insufficient free VRAM (%.1f MiB < %.1f MiB) -- falling back to CPU.",
        free_bytes / 1024**2, _MIN_VRAM_BYTES / 1024**2,
    )
    return torch.device('cpu')


class UnifiedBERTExpert(UnifiedExpert):
    """BERT-based expert that inherits all 3-phase decision logic from UnifiedExpert."""

    def __init__(self, domain, model_path, dataset_path, text_column='text',
                 enable_calibration=True, enable_ood_detection=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.bert_model = None
        self.max_length = 128
        self.model_loaded = False
        super().__init__(
            domain=domain,
            model_path=model_path,
            dataset_path=dataset_path,
            text_column=text_column,
            vectorizer_path=None,
            enable_calibration=enable_calibration,
            enable_ood_detection=enable_ood_detection,
        )

    def _load_trained_model(self):
        import os
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"BERT model directory not found: {self.model_path}")
        print(f"   Preparing BERT expert from {self.model_path}...")
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.bert_model = None
        self.model_loaded = False
        self.model = self
        print("   \u2705 BERT tokenizer loaded (model will be loaded on-demand)")

    def _ensure_model_loaded(self):
        if not self.model_loaded:
            self.device = _safe_device()
            logger.info("_ensure_model_loaded: loading BERT for domain '%s' on %s", self.domain, self.device)
            print(f"   \U0001f4e6 Loading full BERT model for {self.domain} (on-demand) on {self.device}...")
            self.bert_model = BertForSequenceClassification.from_pretrained(self.model_path)
            self.bert_model.to(self.device)
            self.bert_model.eval()
            self.model_loaded = True
            print(f"   \u2705 BERT model loaded on {self.device}")

    def _load_vectorizer(self):
        class BERTTokenizerWrapper:
            def __init__(self, tokenizer, device, max_length):
                self.tokenizer = tokenizer
                self.device = device
                self.max_length = max_length

            def transform(self, texts):
                return texts

        self.vectorizer = BERTTokenizerWrapper(self.tokenizer, self.device, self.max_length)
        print("   \u2705 BERT tokenizer wrapper ready")

    def _find_text_column(self, df):
        text_candidates = ['text', 'sentence', 'content', 'input', 'query']
        for col in text_candidates:
            if col in df.columns:
                return col
        for col in df.columns:
            if df[col].dtype == 'object':
                return col
        return df.columns[0]

    def _find_label_column(self, df):
        label_candidates = ['label', 'target', 'class', 'category', 'y']
        for col in label_candidates:
            if col in df.columns:
                return col
        return df.columns[-1]

    def _normalize_labels(self, labels):
        import numpy as np
        if all(isinstance(label, (int, np.integer)) for label in labels):
            return [int(label) for label in labels]
        positive_keywords = [self.domain.lower(), '1', 'true', 'yes', 'positive']
        normalized = []
        for label in labels:
            label_str = str(label).lower().strip()
            is_positive = any(keyword in label_str for keyword in positive_keywords)
            normalized.append(1 if is_positive else 0)
        return normalized

    def predict(self, text):
        self._ensure_model_loaded()
        if isinstance(text, str):
            texts = [text]
            single_input = True
        else:
            texts = text
            single_input = False
        encodings = self.tokenizer(
            texts, padding='max_length', truncation=True,
            max_length=self.max_length, return_tensors='pt',
        )
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        with torch.no_grad():
            outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
        predictions = predictions.cpu().numpy()
        return predictions[0] if single_input else predictions

    def decision_function(self, text):
        self._ensure_model_loaded()
        if isinstance(text, str):
            texts = [text]
            single_input = True
        else:
            texts = text
            single_input = False
        encodings = self.tokenizer(
            texts, padding='max_length', truncation=True,
            max_length=self.max_length, return_tensors='pt',
        )
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        with torch.no_grad():
            outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
            scores = torch.max(outputs.logits, dim=1)[0]
        scores = scores.cpu().numpy()
        return scores[0] if single_input else scores

    def predict_proba(self, text):
        self._ensure_model_loaded()
        if isinstance(text, str):
            texts = [text]
            single_input = True
        else:
            texts = text
            single_input = False
        encodings = self.tokenizer(
            texts, padding='max_length', truncation=True,
            max_length=self.max_length, return_tensors='pt',
        )
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        with torch.no_grad():
            outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
        probs = probs.cpu().numpy()
        return probs[0] if single_input else probs

    def get_confidence_prediction(self, text):
        probs = self.predict_proba(text)
        prediction = self.predict(text)
        confidence = np.max(probs)
        if self.calibration_score is not None:
            adjusted_confidence = confidence * (0.5 + 0.5 * self.calibration_score)
        else:
            adjusted_confidence = confidence
        return {
            'prediction': int(prediction),
            'confidence': float(adjusted_confidence),
            'raw_confidence': float(confidence),
            'calibration_score': float(self.calibration_score) if self.calibration_score else 0.5,
            'is_proxy': False,
        }

    def _setup_calibration_system(self):
        try:
            import os
            import pickle
            model_fingerprint = self._generate_model_fingerprint()
            cache_path = os.path.join(self.model_path, 'calibration_cache.pkl')
            cache_valid = False
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    cached_metrics = pickle.load(f)
                cached_fingerprint = cached_metrics.get('model_fingerprint', {})
                if self._verify_fingerprint(model_fingerprint, cached_fingerprint):
                    self.calibration_score = cached_metrics.get('validation_accuracy', 0.5)
                    self.calibration_metrics = cached_metrics
                    cache_valid = True
                    print("   \u2705 Loaded cached calibration metrics")
                    print(f"   Validation accuracy: {self.calibration_score:.4f}")
                    print(f"   (Cached from {cached_metrics.get('num_samples', 'unknown')} validation samples)")
                    print(f"   Cache date: {cached_metrics.get('timestamp', 'unknown')}")
                else:
                    print("   \u26a0\ufe0f Cached calibration fingerprint mismatch - model has changed")
            if not cache_valid:
                print("   \U0001f4ca Computing calibration metrics (one-time setup)...")
                self._compute_and_cache_calibration(cache_path, model_fingerprint)
            self.calibrated_model = self.model
        except Exception as e:
            print(f"   \u26a0\ufe0f Calibration setup failed: {e}")
            import traceback
            traceback.print_exc()
            self._setup_fallback_confidence()

    def _generate_model_fingerprint(self):
        import os
        import hashlib
        import json
        fingerprint = {
            'model_path': self.model_path,
            'domain': self.domain,
            'cache_version': '1.0',
        }
        path_hash = hashlib.md5(self.model_path.encode()).hexdigest()[:8]
        fingerprint['path_hash'] = path_hash
        key_files = ['config.json', 'pytorch_model.bin', 'model.safetensors']
        file_timestamps = {}
        for filename in key_files:
            filepath = os.path.join(self.model_path, filename)
            if os.path.exists(filepath):
                file_timestamps[filename] = os.path.getmtime(filepath)
        fingerprint['file_timestamps'] = file_timestamps
        config_path = os.path.join(self.model_path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_content = f.read()
            fingerprint['config_hash'] = hashlib.md5(config_content.encode()).hexdigest()[:8]
        fingerprint_str = json.dumps(fingerprint, sort_keys=True)
        fingerprint['composite_hash'] = hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]
        return fingerprint

    def _verify_fingerprint(self, current_fingerprint, cached_fingerprint):
        if current_fingerprint.get('cache_version') != cached_fingerprint.get('cache_version'):
            return False
        if current_fingerprint.get('composite_hash') != cached_fingerprint.get('composite_hash'):
            return False
        if current_fingerprint.get('path_hash') != cached_fingerprint.get('path_hash'):
            return False
        if 'config_hash' in current_fingerprint and 'config_hash' in cached_fingerprint:
            if current_fingerprint['config_hash'] != cached_fingerprint['config_hash']:
                return False
        return True

    def _compute_and_cache_calibration(self, cache_path, model_fingerprint):
        import os
        import pickle
        val_path = self.dataset_path.replace('train.csv', 'val.csv')
        if not os.path.exists(val_path):
            val_path = self.dataset_path.replace('train.csv', 'validation.csv')
        if not os.path.exists(val_path):
            print("   \u26a0\ufe0f No validation set found, using dataset balance as proxy")
            df = pd.read_csv(self.dataset_path)
            label_col = self._find_label_column(df)
            if label_col in df.columns:
                positive_ratio = np.mean(df[label_col].values == 1)
                self.calibration_score = 1 - abs(0.5 - positive_ratio) * 2
            else:
                self.calibration_score = 0.5
            calibration_metrics = {
                'validation_accuracy': self.calibration_score,
                'method': 'dataset_balance_fallback',
                'num_samples': len(df),
                'timestamp': pd.Timestamp.now().isoformat(),
                'model_fingerprint': model_fingerprint,
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(calibration_metrics, f)
            return
        df = pd.read_csv(val_path)
        text_col = self._find_text_column(df)
        label_col = self._find_label_column(df)
        if text_col not in df.columns or label_col not in df.columns:
            print("   \u26a0\ufe0f Could not find text/label columns")
            self.calibration_score = 0.5
            return
        print("   \U0001f527 Loading BERT model for calibration computation...")
        self._ensure_model_loaded()
        texts = df[text_col].tolist()[:500]
        labels_raw = df[label_col].tolist()[:500]
        labels = self._normalize_labels(labels_raw)
        predictions = []
        confidences = []
        print(f"   \U0001f50d Running inference on {len(texts)} validation samples...")
        for i, text in enumerate(texts):
            if i % 100 == 0 and i > 0:
                print(f"       Progress: {i}/{len(texts)} samples...")
            pred = self.predict(text)
            probs = self.predict_proba(text)
            predictions.append(int(pred))
            confidences.append(np.max(probs))
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        accuracy = accuracy_score(labels, predictions)
        if len(set(labels)) == 2:
            f1 = f1_score(labels, predictions, average='binary', zero_division=0)
            precision = precision_score(labels, predictions, average='binary', zero_division=0)
            recall = recall_score(labels, predictions, average='binary', zero_division=0)
        else:
            f1 = f1_score(labels, predictions, average='macro', zero_division=0)
            precision = precision_score(labels, predictions, average='macro', zero_division=0)
            recall = recall_score(labels, predictions, average='macro', zero_division=0)
        avg_confidence = np.mean(confidences)
        calibration_metrics = {
            'validation_accuracy': accuracy,
            'validation_f1': f1,
            'validation_precision': precision,
            'validation_recall': recall,
            'avg_confidence': avg_confidence,
            'num_samples': len(texts),
            'method': 'full_validation_inference',
            'timestamp': pd.Timestamp.now().isoformat(),
            'model_fingerprint': model_fingerprint,
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(calibration_metrics, f)
        self.calibration_score = accuracy
        self.calibration_metrics = calibration_metrics
        print(f"   \u2705 Calibration computed and cached")
        print(f"   Validation accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        print(f"   Fingerprint: {model_fingerprint['composite_hash']}")
        print("   \U0001f4be Unloading model to save memory (will lazy-load when needed)")
        self.bert_model = None
        self.model_loaded = False

    def invalidate_calibration_cache(self):
        import os
        cache_path = os.path.join(self.model_path, 'calibration_cache.pkl')
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print(f"   \U0001f5d1\ufe0f Calibration cache invalidated for {self.domain}")
            return True
        print(f"   \u2139\ufe0f No calibration cache found for {self.domain}")
        return False

    def get_calibration_cache_info(self):
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
                'fingerprint_hash': cached_metrics.get('model_fingerprint', {}).get('composite_hash', 'unknown'),
            }
        return {'exists': False, 'cache_path': cache_path}

    def __repr__(self):
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
    import os
    model_path = domain_folder_path
    possible_datasets = [
        os.path.join(domain_folder_path, 'train.csv'),
        os.path.join(domain_folder_path, f'{domain_name}_combined_dataset.csv'),
        os.path.join(domain_folder_path, 'dataset.csv'),
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
        enable_ood_detection=enable_ood_detection,
    )
