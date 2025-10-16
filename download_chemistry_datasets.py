"""
Chemistry Dataset Downloader for BERT Training
Downloads and prepares chemistry datasets from various sources
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
        """Download Hendrycks MMLU chemistry datasets"""
        print("📥 Downloading Hendrycks MMLU chemistry datasets...")
        
        chemistry_subjects = [
            'college_chemistry',
            'high_school_chemistry',
        ]
        
        # Also include non-chemistry subjects for contrast
        non_chemistry_subjects = [
            'college_physics',
            'high_school_physics',
            'college_mathematics',
            'high_school_mathematics',
            'college_biology',
            'high_school_biology',
        ]
        
        all_data = []
        
        # Download chemistry data
        print("   Loading chemistry subjects...")
        for subject in tqdm(chemistry_subjects, desc="Chemistry"):
            try:
                dataset = load_dataset("cais/mmlu", subject, split="test")
                
                for item in dataset:
                    question = item['question']
                    choices = item['choices']
                    answer_idx = item['answer']
                    answer = choices[answer_idx]
                    
                    # Format as a complete text
                    text = f"Question: {question} Choices: {', '.join(choices)}. Answer: {answer}. This is a chemistry question testing understanding of chemical principles, reactions, molecular structures, and laboratory techniques."
                    
                    all_data.append({
                        'text': text,
                        'label': 'Chemistry',
                        'source': 'hendrycks_mmlu',
                        'category': subject
                    })
            except Exception as e:
                print(f"      ⚠️ Could not load {subject}: {e}")
        
        # Download non-chemistry data for contrast
        print("   Loading non-chemistry subjects for contrast...")
        for subject in tqdm(non_chemistry_subjects, desc="Non-Chemistry"):
            try:
                dataset = load_dataset("cais/mmlu", subject, split="test")
                
                for item in dataset:
                    question = item['question']
                    choices = item['choices']
                    answer_idx = item['answer']
                    answer = choices[answer_idx]
                    
                    # Format as a complete text
                    text = f"Question: {question} Choices: {', '.join(choices)}. Answer: {answer}. This question tests knowledge in science and mathematics but is not related to chemistry."
                    
                    all_data.append({
                        'text': text,
                        'label': 'Non-Chemistry',
                        'source': 'hendrycks_mmlu',
                        'category': subject
                    })
            except Exception as e:
                print(f"      ⚠️ Could not load {subject}: {e}")
        
        df = pd.DataFrame(all_data)
        output_path = self.output_dir / "hendrycks_chemistry.csv"
        df.to_csv(output_path, index=False)
        print(f"✅ Saved {len(df)} samples to {output_path}")
        return df
    
    def download_sciq_chemistry(self):
        """Download SciQ dataset and filter for chemistry questions"""
        print("📥 Downloading SciQ dataset (filtering for chemistry)...")
        
        try:
            dataset = load_dataset("allenai/sciq", split="train")
            
            # Keywords to identify chemistry questions
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
            
            chemistry_data = []
            non_chemistry_data = []
            
            print("   Processing SciQ questions...")
            for item in tqdm(dataset, desc="Filtering"):
                question = item.get('question', '')
                answer = item.get('correct_answer', '')
                support = item.get('support', '')
                
                combined_text = f"{question} {answer} {support}".lower()
                
                # Check for chemistry keywords
                is_chemistry = any(keyword in combined_text for keyword in chemistry_keywords)
                is_non_chemistry = any(keyword in combined_text for keyword in non_chemistry_keywords)
                
                # Format the text
                if support:
                    text = f"Question: {question} Answer: {answer}. Context: {support}"
                else:
                    text = f"Question: {question} Answer: {answer}."
                
                if is_chemistry and not is_non_chemistry:
                    chemistry_data.append({
                        'text': text,
                        'label': 'Chemistry',
                        'source': 'sciq',
                        'category': 'science_qa'
                    })
                elif is_non_chemistry and not is_chemistry:
                    non_chemistry_data.append({
                        'text': text,
                        'label': 'Non-Chemistry',
                        'source': 'sciq',
                        'category': 'science_qa'
                    })
            
            # Balance the classes
            min_count = min(len(chemistry_data), len(non_chemistry_data))
            all_data = chemistry_data[:min_count] + non_chemistry_data[:min_count]
            
            df = pd.DataFrame(all_data)
            output_path = self.output_dir / "sciq_chemistry.csv"
            df.to_csv(output_path, index=False)
            print(f"✅ Saved {len(df)} samples ({len(chemistry_data[:min_count])} Chemistry, {len(non_chemistry_data[:min_count])} Non-Chemistry)")
            return df
            
        except Exception as e:
            print(f"❌ Error downloading SciQ: {e}")
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
            if file.name in ['train.csv', 'validation.csv', 'test.csv', 'chemistry_combined_dataset.csv']:
                continue  # Skip previously created splits
            
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
        
        # Filter by length (200-2000 characters for BERT)
        combined_df = combined_df[combined_df['text'].str.len().between(200, 2000)]
        print(f"   Filtered to {len(combined_df)} samples with appropriate length")
        
        # Balance classes if needed
        chemistry_count = len(combined_df[combined_df['label'] == 'Chemistry'])
        non_chemistry_count = len(combined_df[combined_df['label'] == 'Non-Chemistry'])
        print(f"   Chemistry: {chemistry_count}, Non-Chemistry: {non_chemistry_count}")
        
        # Balance by taking equal samples from each class
        min_count = min(chemistry_count, non_chemistry_count)
        chemistry_df = combined_df[combined_df['label'] == 'Chemistry'].sample(n=min_count, random_state=42)
        non_chemistry_df = combined_df[combined_df['label'] == 'Non-Chemistry'].sample(n=min_count, random_state=42)
        combined_df = pd.concat([chemistry_df, non_chemistry_df], ignore_index=True)
        print(f"   Balanced to {len(combined_df)} samples ({min_count} per class)")
        
        # Save combined dataset
        output_path = self.output_dir / "chemistry_combined_dataset.csv"
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
    print("🧪 Chemistry Dataset Downloader for BERT Training")
    print("=" * 60)
    
    downloader = ChemistryDatasetDownloader()
    
    # Download datasets
    print("\n📥 Starting dataset downloads...\n")
    
    # 1. Hendrycks MMLU chemistry datasets
    hendrycks_df = downloader.download_hendrycks_chemistry()
    
    # 2. SciQ chemistry questions
    sciq_df = downloader.download_sciq_chemistry()
    
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
