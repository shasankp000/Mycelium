# This module is now importable only. Use evaluate_all_models() or evaluate_model() from another script.

class ExpertModel:
    """
    Wrapper class to provide the expected interface for expert models.
    """
    def __init__(self, model, vectorizer, domain):
        self.model = model
        self.vectorizer = vectorizer
        self.domain = domain
        self.model_id = f"{domain}_expert"
    
    def predict(self, input_text):
        """Make a prediction on the input text."""
        text_tfidf = self.vectorizer.transform([input_text])
        prediction = self.model.predict(text_tfidf)[0]
        return prediction
    
    def score(self, input_text, tags):
        """
        Calculate metrics for the input text.
        For now, returns dummy metrics as the original implementation.
        """
        import random
        # Simulate higher scores if domain matches tags
        if self.domain in [tag.lower() for tag in tags]:
            precision = round(random.uniform(0.7, 0.95), 2)
            recall = round(random.uniform(0.7, 0.95), 2)
            f1 = round((2 * precision * recall) / (precision + recall), 2)
            mse = round(random.uniform(0.01, 0.1), 2)
            mae = round(random.uniform(0.01, 0.1), 2)
        else:
            precision = round(random.uniform(0.2, 0.5), 2)
            recall = round(random.uniform(0.2, 0.5), 2)
            f1 = round((2 * precision * recall) / (precision + recall), 2)
            mse = round(random.uniform(0.2, 0.5), 2)
            mae = round(random.uniform(0.2, 0.5), 2)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mse": mse,
            "mae": mae
        }

# Returns the correct expert model for a given domain string
def get_expert_model(domain):
    """
    Returns an ExpertModel instance for the given domain string.
    Supported domains: 'medical', 'music', 'physics'.
    """
    domain = domain.lower()
    if domain in ["medical", "healthcare"]:
        return ExpertModel(medical_model, medical_vectorizer, "medical")
    elif domain == "music":
        return ExpertModel(music_model, music_vectorizer, "music")
    elif domain == "physics":
        return ExpertModel(physics_model, physics_vectorizer, "physics")
    return None

# Layer 2: Import and evaluate real models from dummy_models
import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error

# Import model modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'dummy_models', 'Medical'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'dummy_models', 'Music'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'dummy_models', 'Physics'))


# Load pre-trained models from pickle files
import pickle

base_dir = os.path.dirname(__file__)

# Load medical model and vectorizer
with open(os.path.join(base_dir, 'dummy_models', 'Medical', 'svm_model_medical.pkl'), 'rb') as f:
    medical_model = pickle.load(f)
with open(os.path.join(base_dir, 'dummy_models', 'Medical', 'vectorizer_medical.pkl'), 'rb') as f:
    medical_vectorizer = pickle.load(f)

# Load music model and vectorizer
with open(os.path.join(base_dir, 'dummy_models', 'Music', 'svm_model_music.pkl'), 'rb') as f:
    music_model = pickle.load(f)
with open(os.path.join(base_dir, 'dummy_models', 'Music', 'vectorizer_music.pkl'), 'rb') as f:
    music_vectorizer = pickle.load(f)

# Load physics model and vectorizer
with open(os.path.join(base_dir, 'dummy_models', 'Physics', 'svm_model_physics.pkl'), 'rb') as f:
    physics_model = pickle.load(f)
with open(os.path.join(base_dir, 'dummy_models', 'Physics', 'vectorizer_physics.pkl'), 'rb') as f:
    physics_vectorizer = pickle.load(f)

import pandas as pd

def evaluate_model(expert_model, data_path, text_col, label_col):
    df = pd.read_csv(data_path)
    X = df[text_col]
    y_true = df[label_col]
    y_pred = [expert_model.predict(x) for x in X]
    # Convert to numpy arrays for metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # If labels are not numeric, encode them
    if y_true.dtype == object or y_pred.dtype == object:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_true = le.fit_transform(y_true)
        y_pred = le.transform(y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mse': mse,
        'mae': mae
    }

def evaluate_all_models():
    results = {}
    # Medical
    medical_expert = ExpertModel(medical_model, medical_vectorizer, "medical")
    results['medical'] = evaluate_model(
        medical_expert,
        os.path.join(os.path.dirname(__file__), 'dummy_models', 'Medical', 'medical_dataset.csv'),
        text_col='sentence',
        label_col='label'
    )
    # Music
    music_expert = ExpertModel(music_model, music_vectorizer, "music")
    results['music'] = evaluate_model(
        music_expert,
        os.path.join(os.path.dirname(__file__), 'dummy_models', 'Music', 'music_classification_dataset.csv'),
        text_col='sentence',
        label_col='label'
    )
    # Physics
    physics_expert = ExpertModel(physics_model, physics_vectorizer, "physics")
    results['physics'] = evaluate_model(
        physics_expert,
        os.path.join(os.path.dirname(__file__), 'dummy_models', 'Physics', 'physics_data.csv'),
        text_col='Comment',
        label_col='binary_label'
    )
    return results


# This module is now importable only. Use evaluate_all_models() or evaluate_model() from another script.
