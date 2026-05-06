"""
Unified BERT Domain Expert Training Script
Trains a BERT model to classify domain vs non-domain content.

Usage:
    python train_bert.py --domain medical
    python train_bert.py --domain chemistry --epochs 5
    python train_bert.py --domain physics --train_data ./custom/train.csv

All flags are optional if set in config.toml under [bert_training].
"""

import argparse
import json
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config loader (reads config.toml if available, no hard crash if absent)
# ---------------------------------------------------------------------------

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # pip install tomli
    except ImportError:
        tomllib = None

_CONFIG_PATH = Path(__file__).parent / "config.toml"

def _load_config() -> dict:
    if tomllib is None or not _CONFIG_PATH.exists():
        return {}
    with open(_CONFIG_PATH, "rb") as f:
        return tomllib.load(f)

_cfg = _load_config()
_bert_cfg = _cfg.get("bert_training", {})

def _cfg_get(key, fallback):
    """Return value from [bert_training] table, else fallback."""
    return _bert_cfg.get(key, fallback)

# ---------------------------------------------------------------------------
# Domain registry — extend this dict to add new domains without touching code
# ---------------------------------------------------------------------------

DOMAIN_REGISTRY = {
    # Medical_BERT CSVs use Biology / Non-Biology labels (see label_map.json)
    "medical": {
        "positive_label": "Biology",
        "negative_label": "Non-Biology",
        "default_data_dir": "dummy_models/Medical_BERT",
        "default_output_dir": "dummy_models/Medical_BERT",
    },
    "chemistry": {
        "positive_label": "Chemistry",
        "negative_label": "Non-Chemistry",
        "default_data_dir": "dummy_models/Chemistry_BERT",
        "default_output_dir": "dummy_models/Chemistry_BERT",
    },
    "physics": {
        "positive_label": "Physics",
        "negative_label": "Non-Physics",
        "default_data_dir": "dummy_models/Physics_BERT",
        "default_output_dir": "dummy_models/Physics_BERT",
    },
}

# Candidate column names searched in order (case-insensitive) — first match wins.
# Add any new column name variants here if needed.
_LABEL_COL_CANDIDATES = ["label", "type", "category", "class", "target", "is_domain"]
_TEXT_COL_CANDIDATES  = ["text", "sentence", "content", "input", "question"]


def _detect_column(df: pd.DataFrame, candidates: list, kind: str) -> str:
    """
    Return the first column name from `candidates` that exists in df
    (case-insensitive match). Raises a descriptive ValueError if none match.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    raise ValueError(
        f"Could not find a {kind} column in the CSV.\n"
        f"  Columns found : {list(df.columns)}\n"
        f"  Expected one of: {candidates}\n"
        f"  Tip: use --label_col / --text_col flags to specify explicitly."
    )

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DomainDataset(Dataset):
    """Generic binary classification dataset for any domain."""

    def __init__(self, texts, labels, tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class DomainBERTTrainer:
    """BERT fine-tuner for binary domain classification."""

    def __init__(
        self,
        domain: str,
        model_name: str = "bert-base-uncased",
        device=None,
        label_col: str = None,
        text_col: str = None,
    ):
        if domain not in DOMAIN_REGISTRY:
            raise ValueError(
                f"Unknown domain '{domain}'. "
                f"Available: {list(DOMAIN_REGISTRY.keys())}"
            )
        self.domain = domain
        self.meta = DOMAIN_REGISTRY[domain]
        self.label_map = {
            self.meta["positive_label"]: 1,
            self.meta["negative_label"]: 0,
        }
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}

        # Explicit overrides (from CLI); None means auto-detect per CSV
        self._label_col_override = label_col
        self._text_col_override  = text_col

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = None

        print(f"🔧 Domain       : {domain}")
        print(f"🔧 Device       : {self.device}")
        print(f"📦 Base model   : {model_name}")
        print(f"🏷️  Labels       : {self.meta['positive_label']} → 1 "
              f"| {self.meta['negative_label']} → 0")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _extract(self, df: pd.DataFrame, split: str):
        """
        Detect or use the overridden text/label columns, map label strings
        to ints, and return (texts, int_labels).
        """
        label_col = self._label_col_override or _detect_column(
            df, _LABEL_COL_CANDIDATES, "label"
        )
        text_col = self._text_col_override or _detect_column(
            df, _TEXT_COL_CANDIDATES, "text"
        )
        print(f"  [{split}] using columns → text='{text_col}', label='{label_col}'")

        mapped = df[label_col].map(self.label_map)
        unknown = df[label_col][mapped.isna()].unique().tolist()
        if unknown:
            print(
                f"  ⚠️  [{split}] Unknown label values — these rows will be "
                f"DROPPED: {unknown}"
            )
            df = df[~df[label_col].isin(unknown)].copy()
            mapped = df[label_col].map(self.label_map)

        return df[text_col].values, mapped.values.astype(int)

    def load_data(self, train_path, val_path, test_path=None):
        """Load CSVs. test_path is optional."""
        print("\n📂 Loading datasets...")
        train_df = pd.read_csv(train_path)
        val_df   = pd.read_csv(val_path)

        train_texts, train_labels = self._extract(train_df, "train")
        val_texts,   val_labels   = self._extract(val_df,   "val")

        print(f"  Train      : {len(train_texts)} samples")
        print(f"  Validation : {len(val_texts)} samples")

        test_texts = test_labels = None
        if test_path and Path(test_path).exists():
            test_df = pd.read_csv(test_path)
            test_texts, test_labels = self._extract(test_df, "test")
            print(f"  Test       : {len(test_texts)} samples")
        else:
            print("  Test       : (skipped — file not found)")

        return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels

    def create_data_loaders(
        self,
        train_texts, train_labels,
        val_texts,   val_labels,
        test_texts=None, test_labels=None,
        batch_size: int = 16,
        max_length: int = 128,
    ):
        print(
            f"\n🔄 Creating data loaders "
            f"(batch={batch_size}, max_len={max_length})..."
        )
        mk = lambda t, l: DomainDataset(t, l, self.tokenizer, max_length)  # noqa
        train_loader = DataLoader(mk(train_texts, train_labels), batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(mk(val_texts,   val_labels),   batch_size=batch_size)
        test_loader  = None
        if test_texts is not None:
            test_loader = DataLoader(mk(test_texts, test_labels), batch_size=batch_size)
        return train_loader, val_loader, test_loader

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def initialize_model(self, num_labels: int = 2):
        print(f"\n🤖 Initialising BERT ({num_labels} classes)...")
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name, num_labels=num_labels
        )
        self.model.to(self.device)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"   Parameters : {total:,}")

    # ------------------------------------------------------------------
    # Train / evaluate loops
    # ------------------------------------------------------------------

    def _train_epoch(self, loader, optimizer):
        self.model.train()
        total_loss, preds_all, labels_all = 0.0, [], []
        for batch in tqdm(loader, desc="  Training", leave=False):
            ids  = batch["input_ids"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
            lbls = batch["labels"].to(self.device)

            optimizer.zero_grad()
            out = self.model(input_ids=ids, attention_mask=mask, labels=lbls)
            out.loss.backward()
            optimizer.step()

            total_loss += out.loss.item()
            preds_all.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
            labels_all.extend(lbls.cpu().numpy())

        return total_loss / len(loader), accuracy_score(labels_all, preds_all)

    def _evaluate(self, loader, split_name: str = "Validation"):
        self.model.eval()
        total_loss, preds_all, labels_all = 0.0, [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"  {split_name}", leave=False):
                ids  = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                lbls = batch["labels"].to(self.device)

                out = self.model(input_ids=ids, attention_mask=mask, labels=lbls)
                total_loss += out.loss.item()
                preds_all.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
                labels_all.extend(lbls.cpu().numpy())

        loss = total_loss / len(loader)
        acc  = accuracy_score(labels_all, preds_all)
        prec, rec, f1, _ = precision_recall_fscore_support(
            labels_all, preds_all, average="weighted", zero_division=0
        )
        return loss, acc, prec, rec, f1, preds_all, labels_all

    # ------------------------------------------------------------------
    # Public train / test
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader,
        val_loader,
        epochs: int = 3,
        learning_rate: float = 2e-5,
        save_dir: str = None,
    ):
        save_path = Path(save_dir or self.meta["default_output_dir"])
        save_path.mkdir(parents=True, exist_ok=True)

        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        best_f1   = 0.0
        history   = {
            "domain": self.domain,
            "train_loss": [], "train_acc": [],
            "val_loss": [],   "val_acc": [],  "val_f1": [],
        }

        print(f"\n🚀 Training for {epochs} epoch(s) | lr={learning_rate}")

        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print("=" * 60)

            train_loss, train_acc = self._train_epoch(train_loader, optimizer)
            print(f"📊 Train  — loss: {train_loss:.4f}  acc: {train_acc:.4f}")

            val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = self._evaluate(
                val_loader, "Validation"
            )
            print(
                f"📊 Val    — loss: {val_loss:.4f}  acc: {val_acc:.4f}  "
                f"P: {val_prec:.4f}  R: {val_rec:.4f}  F1: {val_f1:.4f}"
            )

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_f1"].append(val_f1)

            if val_f1 > best_f1:
                best_f1 = val_f1
                self.model.save_pretrained(save_path)
                self.tokenizer.save_pretrained(save_path)
                print(f"💾 Best model saved (F1: {val_f1:.4f}) → {save_path}")

        hist_path = save_path / "training_history.json"
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"\n💾 Training history → {hist_path}")
        return history

    def test(self, test_loader):
        if test_loader is None:
            print("\n⏭️  No test loader — skipping test evaluation.")
            return None
        print(f"\n{'='*60}")
        print("🧪 Final test evaluation")
        print("=" * 60)

        loss, acc, prec, rec, f1, preds, labels = self._evaluate(test_loader, "Test")
        print(
            f"\n📊 Test — acc: {acc:.4f}  P: {prec:.4f}  "
            f"R: {rec:.4f}  F1: {f1:.4f}"
        )
        target_names = [
            self.reverse_label_map[i] for i in sorted(self.reverse_label_map.keys())
        ]
        print("\n📋 Classification Report:")
        print(classification_report(labels, preds, target_names=target_names))
        return acc, prec, rec, f1

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unified BERT domain-expert trainer. "
                    "Flags override config.toml [bert_training] values."
    )
    p.add_argument(
        "--domain",
        default=_cfg_get("domain", None),
        choices=list(DOMAIN_REGISTRY.keys()),
        required=("domain" not in _bert_cfg),
        help="Domain to train (medical / chemistry / physics …)",
    )
    p.add_argument(
        "--train_data",
        default=_cfg_get("train_data", None),
        help="Path to train CSV.",
    )
    p.add_argument(
        "--val_data",
        default=_cfg_get("val_data", None),
        help="Path to validation CSV.",
    )
    p.add_argument(
        "--test_data",
        default=_cfg_get("test_data", None),
        help="Path to test CSV (optional).",
    )
    p.add_argument(
        "--output_dir",
        default=_cfg_get("output_dir", None),
        help="Directory to save the trained model.",
    )
    p.add_argument(
        "--model_name",
        default=_cfg_get("model_name", "bert-base-uncased"),
        help="HuggingFace model identifier.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=_cfg_get("epochs", 3),
        help="Number of training epochs.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=_cfg_get("batch_size", 16),
        help="Batch size.",
    )
    p.add_argument(
        "--max_length",
        type=int,
        default=_cfg_get("max_length", 128),
        help="Max token sequence length.",
    )
    p.add_argument(
        "--learning_rate",
        type=float,
        default=_cfg_get("learning_rate", 2e-5),
        help="AdamW learning rate.",
    )
    p.add_argument(
        "--label_col",
        default=_cfg_get("label_col", None),
        help=(
            "CSV column that holds the label strings. "
            f"Auto-detected from {_LABEL_COL_CANDIDATES} if not set."
        ),
    )
    p.add_argument(
        "--text_col",
        default=_cfg_get("text_col", None),
        help=(
            "CSV column that holds the text/sentence. "
            f"Auto-detected from {_TEXT_COL_CANDIDATES} if not set."
        ),
    )
    return p


def main():
    args = build_parser().parse_args()

    domain = args.domain
    meta   = DOMAIN_REGISTRY[domain]

    data_dir   = Path(meta["default_data_dir"])
    train_path = Path(args.train_data) if args.train_data else data_dir / "train.csv"
    val_path   = Path(args.val_data)   if args.val_data   else data_dir / "validation.csv"
    test_path  = Path(args.test_data)  if args.test_data  else data_dir / "test.csv"
    output_dir = args.output_dir or meta["default_output_dir"]

    print(f"\n{'='*60}")
    print(f"  BERT Domain Expert Trainer — {domain.upper()}")
    print(f"{'='*60}")
    print(f"  Train      : {train_path}")
    print(f"  Validation : {val_path}")
    print(
        f"  Test       : {test_path} "
        f"{'(skipped if missing)' if not test_path.exists() else ''}"
    )
    print(f"  Output     : {output_dir}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Max length : {args.max_length}")
    print(f"  LR         : {args.learning_rate}")
    if args.label_col:
        print(f"  Label col  : {args.label_col} (explicit)")
    if args.text_col:
        print(f"  Text col   : {args.text_col} (explicit)")

    trainer = DomainBERTTrainer(
        domain=domain,
        model_name=args.model_name,
        label_col=args.label_col,
        text_col=args.text_col,
    )

    (
        train_texts, train_labels,
        val_texts,   val_labels,
        test_texts,  test_labels,
    ) = trainer.load_data(train_path, val_path, test_path)

    train_loader, val_loader, test_loader = trainer.create_data_loaders(
        train_texts, train_labels,
        val_texts,   val_labels,
        test_texts,  test_labels,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    trainer.initialize_model(num_labels=2)

    trainer.train(
        train_loader,
        val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        save_dir=output_dir,
    )

    trainer.test(test_loader)

    print(f"\n{'='*60}")
    print(f"🎉 Done! Model saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
