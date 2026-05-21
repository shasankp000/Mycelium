#!/usr/bin/env python3
"""
Physics Dataset Downloader and Preprocessor for BERT Training
Downloads high-quality physics datasets and prepares them for BERT fine-tuning.

.. note::
   Moved from repository root to scripts/ during cleanup pass (2026-05-21).
"""

import pandas as pd
from datasets import load_dataset
from pathlib import Path


class PhysicsDatasetDownloader:
    def __init__(self, output_dir="training_data/physics"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_hendrycks_physics(self):
        """Download MMLU college + high-school physics datasets."""
        print("📥 Downloading Hendrycks Physics datasets...")
        all_data = []
        physics_count = 0
        for subject in ['college_physics', 'high_school_physics']:
            try:
                dataset = load_dataset("cais/mmlu", subject)
                for split in ['train', 'validation', 'test']:
                    if split in dataset:
                        for item in dataset[split]:
                            text = f"Question: {item['question']}\nAnswer: {item['choices'][item['answer']]}"
                            all_data.append({'text': text, 'label': 'Physics', 'source': 'hendrycks_mmlu', 'category': subject})
                            physics_count += 1
            except Exception as e:
                print(f"   ⚠️ {subject}: {e}")
        non_physics_count = 0
        for subject in ['college_biology', 'college_chemistry', 'high_school_biology']:
            if non_physics_count >= physics_count:
                break
            try:
                dataset = load_dataset("cais/mmlu", subject)
                for split in ['train', 'validation', 'test']:
                    if split in dataset and non_physics_count < physics_count:
                        for item in dataset[split]:
                            if non_physics_count >= physics_count:
                                break
                            text = f"Question: {item['question']}\nAnswer: {item['choices'][item['answer']]}"
                            all_data.append({'text': text, 'label': 'Non-Physics', 'source': 'hendrycks_mmlu', 'category': subject})
                            non_physics_count += 1
            except Exception:
                continue
        df = pd.DataFrame(all_data)
        output_path = self.output_dir / "hendrycks_physics.csv"
        df.to_csv(output_path, index=False)
        print(f"✅ Saved {len(df)} samples to {output_path}")
        return df

    def download_sciq_physics(self):
        """Download SciQ dataset and filter physics questions."""
        print("📥 Downloading SciQ dataset...")
        try:
            dataset = load_dataset("sciq")
            physics_keywords = [
                'force', 'energy', 'momentum', 'velocity', 'acceleration', 'mass', 'gravity',
                'motion', 'wave', 'light', 'electricity', 'magnetism', 'quantum', 'atom',
                'particle', 'thermodynamics', 'pressure', 'volume', 'temperature', 'newton', 'einstein'
            ]
            non_physics_keywords = [
                'cell', 'organ', 'tissue', 'species', 'evolution', 'gene',
                'chemical', 'molecule', 'compound', 'reaction', 'organism',
                'planet', 'earth', 'climate', 'weather', 'soil', 'rock'
            ]
            all_data, physics_count, non_physics_count, max_per_class = [], 0, 0, 2000
            for split in ['train', 'validation', 'test']:
                if split in dataset:
                    for item in dataset[split]:
                        if physics_count >= max_per_class and non_physics_count >= max_per_class:
                            break
                        q, s = item['question'].lower(), item['support'].lower()
                        is_phys = any(k in q or k in s for k in physics_keywords)
                        is_non = any(k in q or k in s for k in non_physics_keywords)
                        text = f"{item['support']} Question: {item['question']} Answer: {item['correct_answer']}"
                        if is_phys and physics_count < max_per_class:
                            all_data.append({'text': text, 'label': 'Physics', 'source': 'sciq', 'category': 'science_qa'})
                            physics_count += 1
                        elif is_non and not is_phys and non_physics_count < max_per_class:
                            all_data.append({'text': text, 'label': 'Non-Physics', 'source': 'sciq', 'category': 'science_qa'})
                            non_physics_count += 1
            df = pd.DataFrame(all_data)
            output_path = self.output_dir / "sciq_physics.csv"
            df.to_csv(output_path, index=False)
            print(f"✅ Saved {len(df)} samples")
            return df
        except Exception as e:
            print(f"❌ Error: {e}")
            return None

    def combine_datasets(self):
        """Combine all downloaded datasets into a single training file with 80/10/10 splits."""
        all_files = list(self.output_dir.glob("*.csv"))
        if not all_files:
            return None
        dfs = []
        for file in all_files:
            try:
                dfs.append(pd.read_csv(file))
            except Exception:
                continue
        if not dfs:
            return None
        combined_df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=['text'])
        combined_df = combined_df[combined_df['text'].str.len().between(200, 2000)]
        combined_df.to_csv(self.output_dir / "physics_combined_dataset.csv", index=False)
        df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        train_size, val_size = int(0.8 * len(df)), int(0.1 * len(df))
        df[:train_size].to_csv(self.output_dir / "train.csv", index=False)
        df[train_size:train_size + val_size].to_csv(self.output_dir / "validation.csv", index=False)
        df[train_size + val_size:].to_csv(self.output_dir / "test.csv", index=False)
        print(f"✅ Combined: {len(combined_df)} samples. Splits saved.")
        return combined_df


def main():
    downloader = PhysicsDatasetDownloader()
    downloader.download_hendrycks_physics()
    downloader.download_sciq_physics()
    downloader.combine_datasets()


if __name__ == "__main__":
    main()
