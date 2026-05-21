"""
Chemistry Dataset Downloader for BERT Training
Downloads and prepares chemistry datasets from various sources.

.. note::
   Moved from repository root to scripts/ during cleanup pass (2026-05-21).
"""

import pandas as pd
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


class ChemistryDatasetDownloader:
    def __init__(self, output_dir="training_data/chemistry"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Output directory: {self.output_dir}")

    def download_hendrycks_chemistry(self):
        """Download Hendrycks MMLU chemistry datasets."""
        print("📥 Downloading Hendrycks MMLU chemistry datasets...")
        chemistry_subjects = ['college_chemistry', 'high_school_chemistry']
        non_chemistry_subjects = [
            'college_physics', 'high_school_physics', 'college_mathematics',
            'high_school_mathematics', 'college_biology', 'high_school_biology',
        ]
        all_data = []
        for subject in tqdm(chemistry_subjects, desc="Chemistry"):
            try:
                dataset = load_dataset("cais/mmlu", subject, split="test")
                for item in dataset:
                    question, choices, answer_idx = item['question'], item['choices'], item['answer']
                    text = f"Question: {question} Choices: {', '.join(choices)}. Answer: {choices[answer_idx]}. This is a chemistry question."
                    all_data.append({'text': text, 'label': 'Chemistry', 'source': 'hendrycks_mmlu', 'category': subject})
            except Exception as e:
                print(f"   ⚠️ Could not load {subject}: {e}")
        for subject in tqdm(non_chemistry_subjects, desc="Non-Chemistry"):
            try:
                dataset = load_dataset("cais/mmlu", subject, split="test")
                for item in dataset:
                    question, choices, answer_idx = item['question'], item['choices'], item['answer']
                    text = f"Question: {question} Choices: {', '.join(choices)}. Answer: {choices[answer_idx]}. Not related to chemistry."
                    all_data.append({'text': text, 'label': 'Non-Chemistry', 'source': 'hendrycks_mmlu', 'category': subject})
            except Exception as e:
                print(f"   ⚠️ Could not load {subject}: {e}")
        df = pd.DataFrame(all_data)
        output_path = self.output_dir / "hendrycks_chemistry.csv"
        df.to_csv(output_path, index=False)
        print(f"✅ Saved {len(df)} samples to {output_path}")
        return df

    def download_sciq_chemistry(self):
        """Download SciQ dataset and filter for chemistry questions."""
        print("📥 Downloading SciQ dataset (filtering for chemistry)...")
        try:
            dataset = load_dataset("allenai/sciq", split="train")
            chemistry_keywords = [
                'atom', 'molecule', 'chemical', 'reaction', 'element', 'compound',
                'bond', 'electron', 'proton', 'neutron', 'ion', 'acid', 'base',
                'pH', 'oxidation', 'reduction', 'catalyst', 'solution', 'solvent',
                'periodic table', 'carbon', 'hydrogen', 'oxygen', 'nitrogen',
                'metal', 'nonmetal', 'organic', 'inorganic', 'chemistry',
                'mole', 'molarity', 'concentration', 'titration', 'precipitate',
                'solubility', 'valence', 'isotope', 'radioactive', 'nuclear'
            ]
            non_chemistry_keywords = [
                'force', 'motion', 'velocity', 'acceleration', 'gravity',
                'electromagnetic', 'light wave', 'sound wave', 'frequency',
                'cell', 'organism', 'DNA', 'protein synthesis', 'evolution',
                'photosynthesis', 'mitosis', 'meiosis', 'ecosystem'
            ]
            chemistry_data, non_chemistry_data = [], []
            for item in tqdm(dataset, desc="Filtering"):
                question, answer, support = item.get('question',''), item.get('correct_answer',''), item.get('support','')
                combined = f"{question} {answer} {support}".lower()
                is_chem = any(k in combined for k in chemistry_keywords)
                is_non_chem = any(k in combined for k in non_chemistry_keywords)
                text = f"Question: {question} Answer: {answer}." + (f" Context: {support}" if support else "")
                if is_chem and not is_non_chem:
                    chemistry_data.append({'text': text, 'label': 'Chemistry', 'source': 'sciq', 'category': 'science_qa'})
                elif is_non_chem and not is_chem:
                    non_chemistry_data.append({'text': text, 'label': 'Non-Chemistry', 'source': 'sciq', 'category': 'science_qa'})
            min_count = min(len(chemistry_data), len(non_chemistry_data))
            df = pd.DataFrame(chemistry_data[:min_count] + non_chemistry_data[:min_count])
            output_path = self.output_dir / "sciq_chemistry.csv"
            df.to_csv(output_path, index=False)
            print(f"✅ Saved {len(df)} samples")
            return df
        except Exception as e:
            print(f"❌ Error: {e}")
            return None

    def combine_datasets(self):
        """Combine all downloaded datasets into a single training file with 80/10/10 splits."""
        print("\n🔄 Combining all datasets...")
        all_files = [f for f in self.output_dir.glob("*.csv")
                     if f.name not in ['train.csv', 'validation.csv', 'test.csv', 'chemistry_combined_dataset.csv']]
        if not all_files:
            print("❌ No datasets found to combine")
            return None
        dfs = []
        for file in all_files:
            try:
                dfs.append(pd.read_csv(file))
            except Exception as e:
                print(f"   ⚠️ Could not load {file.name}: {e}")
        if not dfs:
            return None
        combined_df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=['text'])
        combined_df = combined_df[combined_df['text'].str.len().between(200, 2000)]
        min_count = min(
            len(combined_df[combined_df['label'] == 'Chemistry']),
            len(combined_df[combined_df['label'] == 'Non-Chemistry'])
        )
        combined_df = pd.concat([
            combined_df[combined_df['label'] == 'Chemistry'].sample(n=min_count, random_state=42),
            combined_df[combined_df['label'] == 'Non-Chemistry'].sample(n=min_count, random_state=42)
        ], ignore_index=True)
        combined_df.to_csv(self.output_dir / "chemistry_combined_dataset.csv", index=False)
        df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        train_size, val_size = int(0.8 * len(df)), int(0.1 * len(df))
        df[:train_size].to_csv(self.output_dir / "train.csv", index=False)
        df[train_size:train_size + val_size].to_csv(self.output_dir / "validation.csv", index=False)
        df[train_size + val_size:].to_csv(self.output_dir / "test.csv", index=False)
        print(f"✅ Combined dataset: {len(combined_df)} samples. Splits saved.")
        return combined_df


def main():
    downloader = ChemistryDatasetDownloader()
    downloader.download_hendrycks_chemistry()
    downloader.download_sciq_chemistry()
    downloader.combine_datasets()


if __name__ == "__main__":
    main()
