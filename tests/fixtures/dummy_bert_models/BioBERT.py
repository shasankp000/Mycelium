import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from tqdm import tqdm
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class BiologyDataset(Dataset):
    """Custom Dataset for biology text classification"""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BioBERTTrainer:
    """BioBERT Trainer for Biology Domain"""
    def __init__(self, model_name='dmis-lab/biobert-base-cased-v1.1', output_dir='dummy_models/BioBERT'):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
        print(f"📦 Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.num_labels = None

    def load_data(self, train_path, val_path, test_path):
        print("\n📂 Loading datasets...")
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        
        print(f"Train: {len(train_df)} samples")
        print(f"Validation: {len(val_df)} samples")
        print(f"Test: {len(test_df)} samples")
        
        # Convert labels to numeric if needed
        if train_df['Type'].dtype == 'object':
            unique_labels = sorted(train_df['Type'].unique())
            self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
            self.inverse_label_map = {idx: label for label, idx in self.label_map.items()}
            
            train_df['label_encoded'] = train_df['Type'].map(self.label_map)
            val_df['label_encoded'] = val_df['Type'].map(self.label_map)
            test_df['label_encoded'] = test_df['Type'].map(self.label_map)
        else:
            train_df['label_encoded'] = train_df['Type']
            val_df['label_encoded'] = val_df['Type']
            test_df['label_encoded'] = test_df['Type']
            self.label_map = None
            self.inverse_label_map = None
        
        self.num_labels = train_df['label_encoded'].nunique()
        print(f"   Number of classes: {self.num_labels}")
        
        return train_df, val_df, test_df

    def create_data_loaders(self, train_df, val_df, test_df, batch_size=16, max_length=128):
        print(f"\n🔄 Creating data loaders (batch_size={batch_size}, max_length={max_length})...")
        train_dataset = BiologyDataset(
            train_df['Text'].values,
            train_df['label_encoded'].values,
            self.tokenizer,
            max_length
        )
        val_dataset = BiologyDataset(
            val_df['Text'].values,
            val_df['label_encoded'].values,
            self.tokenizer,
            max_length
        )
        test_dataset = BiologyDataset(
            test_df['Text'].values,
            test_df['label_encoded'].values,
            self.tokenizer,
            max_length
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader

    def initialize_model(self):
        print(f"\n🤖 Initializing BioBERT model for {self.num_labels} classes...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )
        self.model.to(self.device)
        print(f"   Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def train_epoch(self, train_loader, optimizer, scheduler):
        self.model.train()
        total_loss = 0
        predictions = []
        true_labels = []
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())
            progress_bar.set_postfix({'loss': loss.item()})
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(true_labels, predictions)
        return avg_loss, accuracy

    def evaluate(self, data_loader, dataset_name="Validation"):
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Evaluating {dataset_name}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                total_loss += loss.item()
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.extend(preds)
                true_labels.extend(labels.cpu().numpy())
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted', zero_division=0
        )
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': predictions,
            'true_labels': true_labels
        }

    def train(self, train_loader, val_loader, epochs=3, learning_rate=2e-5):
        print(f"\n🚀 Starting training for {epochs} epochs...")
        print(f"   Learning rate: {learning_rate}")
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, eps=1e-8)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': []
        }
        best_val_f1 = 0.0
        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"{'='*60}")
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, scheduler)
            print(f"📊 Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
            val_metrics = self.evaluate(val_loader, "Validation")
            print(f"📊 Val Loss: {val_metrics['loss']:.4f}, Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"   Val Precision: {val_metrics['precision']:.4f}, Val Recall: {val_metrics['recall']:.4f}, Val F1: {val_metrics['f1']:.4f}")
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['val_f1'].append(val_metrics['f1'])
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                self.save_model(suffix='best')
                print(f"💾 Saved best model (F1: {best_val_f1:.4f})")
        self.save_model(suffix='final')
        self.save_training_history(history)
        return history

    def save_model(self, suffix='final'):
        save_dir = self.output_dir / f"model_{suffix}"
        save_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        if self.label_map:
            with open(save_dir / 'label_map.json', 'w') as f:
                json.dump(self.label_map, f, indent=2)

    def save_training_history(self, history):
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"\n💾 Training history saved to {history_path}")

    def test_model(self, test_loader):
        print(f"\n{'='*60}")
        print("🧪 Testing on held-out test set...")
        print(f"{'='*60}")
        test_metrics = self.evaluate(test_loader, "Test")
        print("\n📊 Test Results:")
        print(f"   Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"   Precision: {test_metrics['precision']:.4f}")
        print(f"   Recall: {test_metrics['recall']:.4f}")
        print(f"   F1 Score: {test_metrics['f1']:.4f}")
        # Detailed classification report
        if self.inverse_label_map:
            target_names = [self.inverse_label_map[i] for i in range(self.num_labels)]
        else:
            target_names = [f"Class {i}" for i in range(self.num_labels)]
        print("\n📋 Detailed Classification Report:")
        print(classification_report(
            test_metrics['true_labels'],
            test_metrics['predictions'],
            target_names=target_names,
            zero_division=0
        ))
        # Save test results
        test_results = {
            'accuracy': float(test_metrics['accuracy']),
            'precision': float(test_metrics['precision']),
            'recall': float(test_metrics['recall']),
            'f1': float(test_metrics['f1'])
        }
        with open(self.output_dir / 'test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)
        return test_metrics

def main():
    print("🚀 BioBERT-based Biology Expert Training")
    print("="*60)
    DATA_DIR = Path(".")
    MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
    OUTPUT_DIR = "dummy_models/BioBERT"
    BATCH_SIZE = 16
    MAX_LENGTH = 128
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    train_path = DATA_DIR / "train.csv"
    val_path = DATA_DIR / "validation.csv"
    test_path = DATA_DIR / "test.csv"
    if not all([train_path.exists(), val_path.exists(), test_path.exists()]):
        print("❌ Error: Training data not found!")
        print(f"   Expected location: {DATA_DIR}")
        return
    trainer = BioBERTTrainer(model_name=MODEL_NAME, output_dir=OUTPUT_DIR)
    train_df, val_df, test_df = trainer.load_data(train_path, val_path, test_path)
    train_loader, val_loader, test_loader = trainer.create_data_loaders(
        train_df, val_df, test_df,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH
    )
    trainer.initialize_model()
    trainer.train(
        train_loader,
        val_loader,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE
    )
    trainer.test_model(test_loader)
    print("\n" + "="*60)
    print("🎉 Training complete!")
    print(f"📁 Model saved to: {OUTPUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()
