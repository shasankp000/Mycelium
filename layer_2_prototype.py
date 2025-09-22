import random

class DomainExpertBase:
    def __init__(self, domain):
        self.domain = domain
        self.model_id = f"{domain}_expert"
    def predict(self, input_text):
        # Dummy prediction logic
        return f"Prediction for '{input_text}' in domain '{self.domain}'"
    def score(self, input_text, tags):
        # Dummy metrics for demonstration
        # Simulate higher scores if input matches domain
        if self.domain in tags:
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

# Create 5 dummy expert models
EXPERT_TAGS = ["AI", "healthcare", "finance", "music", "physics"]
EXPERT_MODELS = {tag: DomainExpertBase(tag) for tag in EXPERT_TAGS}

def get_expert_model(domain):
    return EXPERT_MODELS.get(domain)

def get_all_expert_models():
    return EXPERT_MODELS
