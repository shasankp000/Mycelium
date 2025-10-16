#!/usr/bin/env python3
"""
Physics Dataset Downloader and Preprocessor for BERT Training
Downloads high-quality physics datasets and prepares them for BERT fine-tuning
"""

import os
import pandas as pd
import requests
from datasets import load_dataset
import json
from pathlib import Path

class PhysicsDatasetDownloader:
    def __init__(self, output_dir="training_data/physics"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def download_hendrycks_physics(self):
        """Download STEM physics competition dataset from HuggingFace"""
        print("📥 Downloading Hendrycks Competition Physics dataset...")
        try:
            # Load the dataset
            dataset = load_dataset("cais/mmlu", "college_physics")
            
            # Combine train, validation, test
            all_data = []
            physics_count = 0
            for split in ['train', 'validation', 'test']:
                if split in dataset:
                    for item in dataset[split]:
                        text = f"Question: {item['question']}\nAnswer: {item['choices'][item['answer']]}"
                        all_data.append({
                            'text': text,
                            'label': 'Physics',
                            'source': 'hendrycks_mmlu',
                            'category': 'college_physics'
                        })
                        physics_count += 1
            
            # Also get high school physics
            dataset_hs = load_dataset("cais/mmlu", "high_school_physics")
            for split in ['train', 'validation', 'test']:
                if split in dataset_hs:
                    for item in dataset_hs[split]:
                        text = f"Question: {item['question']}\nAnswer: {item['choices'][item['answer']]}"
                        all_data.append({
                            'text': text,
                            'label': 'Physics',
                            'source': 'hendrycks_mmlu',
                            'category': 'high_school_physics'
                        })
                        physics_count += 1
            
            # Add non-physics STEM subjects for contrast
            non_physics_subjects = ['college_biology', 'college_chemistry', 'high_school_biology']
            non_physics_count = 0
            target_non_physics = physics_count  # Balance the classes
            
            for subject in non_physics_subjects:
                if non_physics_count >= target_non_physics:
                    break
                try:
                    dataset_np = load_dataset("cais/mmlu", subject)
                    for split in ['train', 'validation', 'test']:
                        if split in dataset_np and non_physics_count < target_non_physics:
                            for item in dataset_np[split]:
                                if non_physics_count >= target_non_physics:
                                    break
                                text = f"Question: {item['question']}\nAnswer: {item['choices'][item['answer']]}"
                                all_data.append({
                                    'text': text,
                                    'label': 'Non-Physics',
                                    'source': 'hendrycks_mmlu',
                                    'category': subject
                                })
                                non_physics_count += 1
                except:
                    continue
            
            # Save to CSV
            df = pd.DataFrame(all_data)
            output_path = self.output_dir / "hendrycks_physics.csv"
            df.to_csv(output_path, index=False)
            print(f"✅ Saved {len(df)} samples to {output_path}")
            print(f"   Physics: {physics_count}, Non-Physics: {non_physics_count}")
            return df
            
        except Exception as e:
            print(f"❌ Error downloading Hendrycks dataset: {e}")
            return None
    
    def download_sciq_physics(self):
        """Download SciQ dataset and filter physics questions"""
        print("📥 Downloading SciQ dataset...")
        try:
            dataset = load_dataset("sciq")
            
            physics_keywords = [
                'force', 'energy', 'momentum', 'velocity', 'acceleration',
                'mass', 'gravity', 'motion', 'wave', 'light', 'electricity',
                'magnetism', 'quantum', 'atom', 'particle', 'thermodynamics',
                'pressure', 'volume', 'temperature', 'newton', 'einstein'
            ]
            
            # Non-physics keywords for filtering
            non_physics_keywords = [
                'cell', 'organ', 'tissue', 'species', 'evolution', 'gene',
                'chemical', 'molecule', 'compound', 'reaction', 'organism',
                'planet', 'earth', 'climate', 'weather', 'soil', 'rock'
            ]
            
            all_data = []
            physics_count = 0
            non_physics_count = 0
            max_per_class = 2000  # Balance the classes
            
            for split in ['train', 'validation', 'test']:
                if split in dataset:
                    for item in dataset[split]:
                        if physics_count >= max_per_class and non_physics_count >= max_per_class:
                            break
                            
                        question_lower = item['question'].lower()
                        support_lower = item['support'].lower()
                        
                        # Check if physics-related
                        is_physics = any(keyword in question_lower or keyword in support_lower 
                                       for keyword in physics_keywords)
                        
                        # Check if non-physics science
                        is_non_physics = any(keyword in question_lower or keyword in support_lower 
                                           for keyword in non_physics_keywords)
                        
                        text = f"{item['support']} Question: {item['question']} Answer: {item['correct_answer']}"
                        
                        if is_physics and physics_count < max_per_class:
                            all_data.append({
                                'text': text,
                                'label': 'Physics',
                                'source': 'sciq',
                                'category': 'science_qa'
                            })
                            physics_count += 1
                        elif is_non_physics and not is_physics and non_physics_count < max_per_class:
                            all_data.append({
                                'text': text,
                                'label': 'Non-Physics',
                                'source': 'sciq',
                                'category': 'science_qa'
                            })
                            non_physics_count += 1
            
            df = pd.DataFrame(all_data)
            output_path = self.output_dir / "sciq_physics.csv"
            df.to_csv(output_path, index=False)
            print(f"✅ Saved {len(df)} samples to {output_path}")
            print(f"   Physics: {physics_count}, Non-Physics: {non_physics_count}")
            return df
            
        except Exception as e:
            print(f"❌ Error downloading SciQ dataset: {e}")
            return None
    
    def download_wikipedia_physics(self):
        """Download physics articles from Wikipedia via HuggingFace"""
        print("📥 Downloading Wikipedia physics articles...")
        try:
            # Using wikipedia dataset from HuggingFace
            dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
            
            physics_categories = [
                'physics', 'quantum', 'mechanics', 'thermodynamics', 
                'electromagnetism', 'optics', 'relativity', 'particle'
            ]
            
            all_data = []
            count = 0
            max_articles = 10000  # Limit for initial download
            
            for item in dataset:
                if count >= max_articles:
                    break
                    
                title_lower = item['title'].lower()
                text_lower = item['text'].lower()
                
                # Check if physics-related
                is_physics = any(cat in title_lower or cat in text_lower[:500] 
                               for cat in physics_categories)
                
                if is_physics and len(item['text']) > 200:
                    # Split into chunks for BERT (max 512 tokens ≈ 2000 chars)
                    text_chunks = [item['text'][i:i+2000] 
                                 for i in range(0, len(item['text']), 2000)]
                    
                    for chunk in text_chunks[:3]:  # Max 3 chunks per article
                        all_data.append({
                            'text': chunk,
                            'label': 'Physics',
                            'source': 'wikipedia',
                            'category': 'encyclopedia'
                        })
                    count += 1
                    
                    if count % 100 == 0:
                        print(f"   Processed {count} physics articles...")
            
            df = pd.DataFrame(all_data)
            output_path = self.output_dir / "wikipedia_physics.csv"
            df.to_csv(output_path, index=False)
            print(f"✅ Saved {len(df)} physics text chunks to {output_path}")
            return df
            
        except Exception as e:
            print(f"❌ Error downloading Wikipedia dataset: {e}")
            print("   Note: This dataset is large. Consider downloading manually if needed.")
            return None
    
    def combine_datasets(self):
        """Combine all downloaded datasets into a single training file"""
        print("\n🔄 Combining all datasets...")
        
        all_files = list(self.output_dir.glob("*.csv"))
        if not all_files:
            print("❌ No datasets found to combine")
            return None
        
        dfs = []
        for file in all_files:
            try:
                df = pd.read_csv(file)
                dfs.append(df)
                print(f"   Loaded {len(df)} samples from {file.name}")
            except Exception as e:
                print(f"   ⚠️ Could not load {file.name}: {e}")
        
        if not dfs:
            return None
        
        # Combine all datasets
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Remove duplicates
        original_len = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['text'])
        print(f"   Removed {original_len - len(combined_df)} duplicate samples")
        
        # Filter by length (50-512 tokens ≈ 200-2000 characters)
        combined_df = combined_df[combined_df['text'].str.len().between(200, 2000)]
        print(f"   Filtered to {len(combined_df)} samples with appropriate length")
        
        # Save combined dataset
        output_path = self.output_dir / "physics_combined_dataset.csv"
        combined_df.to_csv(output_path, index=False)
        print(f"\n✅ Combined dataset saved to {output_path}")
        print(f"📊 Total samples: {len(combined_df)}")
        
        # Create train/val/test splits
        self.create_splits(combined_df)
        
        return combined_df
    
    def create_splits(self, df):
        """Create train/validation/test splits"""
        print("\n📂 Creating train/validation/test splits...")
        
        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split: 80% train, 10% validation, 10% test
        train_size = int(0.8 * len(df))
        val_size = int(0.1 * len(df))
        
        train_df = df[:train_size]
        val_df = df[train_size:train_size + val_size]
        test_df = df[train_size + val_size:]
        
        # Save splits
        train_df.to_csv(self.output_dir / "train.csv", index=False)
        val_df.to_csv(self.output_dir / "validation.csv", index=False)
        test_df.to_csv(self.output_dir / "test.csv", index=False)
        
        print(f"   Train: {len(train_df)} samples")
        print(f"   Validation: {len(val_df)} samples")
        print(f"   Test: {len(test_df)} samples")
        print("✅ Splits saved successfully")

def main():
    print("🚀 Physics Dataset Downloader for BERT Training")
    print("=" * 60)
    
    downloader = PhysicsDatasetDownloader()
    
    # Download datasets
    print("\n📥 Starting dataset downloads...\n")
    
    # 1. Hendrycks MMLU Physics (Quick, ~2K samples)
    hendrycks_df = downloader.download_hendrycks_physics()
    
    # 2. SciQ Physics subset (Quick, ~3K samples)
    sciq_df = downloader.download_sciq_physics()
    
    # 3. Wikipedia Physics (Slow, can be skipped for quick start)
    # Uncomment if you want Wikipedia data:
    # wikipedia_df = downloader.download_wikipedia_physics()
    
    # Combine all datasets
    print("\n" + "=" * 60)
    combined_df = downloader.combine_datasets()
    
    if combined_df is not None:
        print("\n" + "=" * 60)
        print("🎉 Dataset preparation complete!")
        print(f"📊 Ready for BERT training with {len(combined_df)} samples")
        print(f"📁 Location: {downloader.output_dir}")
    else:
        print("\n❌ Failed to prepare datasets")

if __name__ == "__main__":
    main()
