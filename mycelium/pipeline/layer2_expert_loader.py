# This module is now importable only. Use evaluate_all_models() or evaluate_model() from another script.

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
)

# ---------------------------------------------------------------------------
# Path resolution — use __file__ so this works regardless of cwd (plan §5.2)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
# dummy_models lives at project root, three levels up from mycelium/pipeline/
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_DUMMY_MODELS = os.path.join(_PROJECT_ROOT, "dummy_models")

sys.path.append(os.path.join(_DUMMY_MODELS, "Medical"))
sys.path.append(os.path.join(_DUMMY_MODELS, "Music"))
sys.path.append(os.path.join(_DUMMY_MODELS, "Physics"))


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
        import random

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
            "mae": mae,
        }


_MODEL_CACHE = {
    "medical": None,
    "music": None,
    "physics": None,
}
_VECTORIZER_CACHE = {
    "medical": None,
    "music": None,
    "physics": None,
}


def _load_model_assets(domain: str):
    domain_dir = {
        "medical": "Medical",
        "music": "Music",
        "physics": "Physics",
    }.get(domain)
    if not domain_dir:
        return None, None
    model_path = os.path.join(_DUMMY_MODELS, domain_dir, f"svm_model_{domain}.pkl")
    vectorizer_path = os.path.join(
        _DUMMY_MODELS, domain_dir, f"vectorizer_{domain}.pkl"
    )
    if not (os.path.exists(model_path) and os.path.exists(vectorizer_path)):
        return None, None
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def get_expert_model(domain):
    """
    Returns an ExpertModel instance for the given domain string.
    Supported domains: 'medical', 'music', 'physics'.
    """
    domain = domain.lower()
    if domain == "healthcare":
        domain = "medical"
    if domain not in _MODEL_CACHE:
        return None
    if _MODEL_CACHE[domain] is None or _VECTORIZER_CACHE[domain] is None:
        model, vectorizer = _load_model_assets(domain)
        _MODEL_CACHE[domain] = model
        _VECTORIZER_CACHE[domain] = vectorizer
    model = _MODEL_CACHE[domain]
    vectorizer = _VECTORIZER_CACHE[domain]
    if model is None or vectorizer is None:
        return None
    return ExpertModel(model, vectorizer, domain)


def evaluate_model(expert_model, data_path, text_col, label_col):
    df = pd.read_csv(data_path)
    X = df[text_col]
    y_true = df[label_col]
    y_pred = [expert_model.predict(x) for x in X]
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if y_true.dtype == object or y_pred.dtype == object:
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y_true = le.fit_transform(y_true)
        y_pred = le.transform(y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return {"precision": precision, "recall": recall, "f1": f1, "mse": mse, "mae": mae}


def evaluate_all_models():
    results = {}
    medical_expert = ExpertModel(medical_model, medical_vectorizer, "medical")
    results["medical"] = evaluate_model(
        medical_expert,
        os.path.join(_DUMMY_MODELS, "Medical", "medical_dataset.csv"),
        text_col="sentence",
        label_col="label",
    )
    music_expert = ExpertModel(music_model, music_vectorizer, "music")
    results["music"] = evaluate_model(
        music_expert,
        os.path.join(_DUMMY_MODELS, "Music", "music_classification_dataset.csv"),
        text_col="sentence",
        label_col="label",
    )
    physics_expert = ExpertModel(physics_model, physics_vectorizer, "physics")
    results["physics"] = evaluate_model(
        physics_expert,
        os.path.join(_DUMMY_MODELS, "Physics", "physics_data.csv"),
        text_col="Comment",
        label_col="binary_label",
    )
    return results
