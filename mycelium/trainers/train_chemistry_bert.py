"""
BERT-based Chemistry Expert Training Script
Trains a BERT model to classify chemistry vs non-chemistry content
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

class ChemistryDataset(Dataset):
    """Dataset class for chemistry classification"""
    
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
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class ChemistryBERTTrainer:
    """Trainer class for BERT chemistry classifier"""
    
    def __init__(self, model_name='bert-base-uncased', device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = None
        self.label_map = {'Chemistry': 1, 'Non-Chemistry': 0}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        
        print(f"🔧 Using device: {self.device}")
        print(f"📦 Loading tokenizer: {model_name}")
    
    def load_data(self, train_path, val_path, test_path):
        """Load datasets from CSV files"""
        print("\n📂 Loading datasets...")
        
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        
        print(f"Train: {len(train_df)} samples")
        print(f"Validation: {len(val_df)} samples")
        print(f"Test: {len(test_df)} samples")
        
        # Convert labels to integers
        train_labels = train_df['label'].map(self.label_map).values
        val_labels = val_df['label'].map(self.label_map).values
        test_labels = test_df['label'].map(self.label_map).values
        
        print(f"   Number of classes: {len(self.label_map)}")
        
        return (train_df['text'].values, train_labels,
                val_df['text'].values, val_labels,
                test_df['text'].values, test_labels)
    
    def create_data_loaders(self, train_texts, train_labels, val_texts, val_labels, 
                           test_texts, test_labels, batch_size=16, max_length=128):
        """Create DataLoader objects"""
        print(f"\n🔄 Creating data loaders (batch_size={batch_size}, max_length={max_length})...")
        
        train_dataset = ChemistryDataset(train_texts, train_labels, self.tokenizer, max_length)
        val_dataset = ChemistryDataset(val_texts, val_labels, self.tokenizer, max_length)
        test_dataset = ChemistryDataset(test_texts, test_labels, self.tokenizer, max_length)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        return train_loader, val_loader, test_loader
    
    def initialize_model(self, num_labels=2):
        """Initialize BERT model for classification"""
        print(f"\n🤖 Initializing BERT model for {num_labels} classes...")
        
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels
        )
        self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"   Model parameters: {total_params:,}")
    
    def train_epoch(self, train_loader, optimizer, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        predictions = []
        true_labels = []
        
        progress_bar = tqdm(train_loader, desc=f"Training")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            optimizer.zero_grad()
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Get predictions
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(true_labels, predictions)
        
        return avg_loss, accuracy
    
    def evaluate(self, data_loader, dataset_name="Validation"):
        """Evaluate the model"""
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
                
                total_loss += outputs.loss.item()
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted', zero_division=0
        )
        
        return avg_loss, accuracy, precision, recall, f1, predictions, true_labels
    
    def train(self, train_loader, val_loader, epochs=3, learning_rate=2e-5, save_dir="dummy_models/Chemistry_BERT"):
        """Train the model"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        print(f"\n🚀 Starting training for {epochs} epochs...")
        print(f"   Learning rate: {learning_rate}")
        
        best_f1 = 0
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': []
        }
        
        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print('='*60)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, epoch)
            print(f"📊 Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
            
            # Validate
            val_loss, val_acc, val_precision, val_recall, val_f1, _, _ = self.evaluate(val_loader)
            print(f"📊 Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
            print(f"   Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)
            
            # Save best model
            if val_f1 > best_f1:
                best_f1 = val_f1
                self.model.save_pretrained(save_path)
                self.tokenizer.save_pretrained(save_path)
                print(f"💾 Saved best model (F1: {val_f1:.4f})")
        
        # Save training history
        history_path = save_path / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"\n💾 Training history saved to {history_path}")
        
        return history
    
    def test(self, test_loader):
        """Test the model on held-out test set"""
        print(f"\n{'='*60}")
        print("🧪 Testing on held-out test set...")
        print('='*60)
        
        test_loss, test_acc, test_precision, test_recall, test_f1, predictions, true_labels = self.evaluate(
            test_loader, "Test"
        )
        
        print(f"\n📊 Test Results:")
        print(f"   Accuracy: {test_acc:.4f}")
        print(f"   Precision: {test_precision:.4f}")
        print(f"   Recall: {test_recall:.4f}")
        print(f"   F1 Score: {test_f1:.4f}")
        
        # Detailed classification report
        print(f"\n📋 Detailed Classification Report:")
        target_names = [self.reverse_label_map[i] for i in sorted(self.reverse_label_map.keys())]
        print(classification_report(true_labels, predictions, target_names=target_names))
        
        return test_acc, test_precision, test_recall, test_f1

def main():
    print("🧪 BERT-based Chemistry Expert Training - Phase 1")
    print("="*60)
    
    # Paths
    data_dir = Path("training_data/chemistry")
    train_path = data_dir / "train.csv"
    val_path = data_dir / "validation.csv"
    test_path = data_dir / "test.csv"
    
    # Initialize trainer
    trainer = ChemistryBERTTrainer()
    
    # Load data
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = trainer.load_data(
        train_path, val_path, test_path
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = trainer.create_data_loaders(
        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels,
        batch_size=16, max_length=128
    )
    
    # Initialize model
    trainer.initialize_model(num_labels=2)
    
    # Train
    history = trainer.train(
        train_loader, val_loader,
        epochs=3,
        learning_rate=2e-5,
        save_dir="dummy_models/Chemistry_BERT"
    )
    
    # Test
    test_acc, test_precision, test_recall, test_f1 = trainer.test(test_loader)
    
    print("\n" + "="*60)
    print("🎉 Training complete!")
    print(f"📁 Model saved to: dummy_models/Chemistry_BERT")
    print("="*60)

if __name__ == "__main__":
    main()
