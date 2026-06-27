"""
Step 5 — DistilBERT Fine-Tuning

Fixes three reference-notebook issues simultaneously:
  1. Uses distilbert-base-uncased (66M params, 6 layers) vs bert-tiny (4.4M, 2 layers)
  2. Feeds natural-language text (text_transformer) — no stemming or stopword removal
  3. Handles class imbalance with WeightedTrainer — no data discarded via undersampling

Three-way split: train/val/test. Val drives best-checkpoint selection
(load_best_model_at_end=True). Not early stopping — all TRAIN_EPOCHS are run.
Test is evaluated once after training.
Final metrics are written to outputs/metrics/transformer.json for the comparison table.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).parent))
import config
from utils import get_logger, save_metrics, set_seed, timer

logger = get_logger(__name__)


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ReviewDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer) -> None:
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=config.MAX_SEQ_LEN,
        )
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


class WeightedTrainer(Trainer):
    """Trainer that applies class weights to the cross-entropy loss."""

    def __init__(self, class_weights: torch.Tensor, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weights = self.class_weights.to(outputs.logits.device)
        loss = CrossEntropyLoss(weight=weights, ignore_index=-100)(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average="weighted"),
        "recall": recall_score(labels, preds, average="weighted"),
    }


@timer
def train_transformer() -> None:
    set_seed(config.RANDOM_SEED)

    src = config.DATA_CLEAN / "labeled_dataset.parquet"
    if not src.exists():
        raise FileNotFoundError(f"Run build_labels.py first — {src} not found")
    df = pd.read_parquet(src, engine=config.PARQUET_ENGINE).dropna(subset=["text_transformer", "sentiment_label"])
    df["sentiment_label"] = df["sentiment_label"].astype(int)
    logger.info(f"Loaded: {len(df):,} rows")

    # Optional stratified sample — preserves class ratios, keeps training feasible on MPS/CPU
    if config.MAX_TRAIN_SAMPLES and len(df) > config.MAX_TRAIN_SAMPLES:
        df, _ = train_test_split(
            df,
            train_size=config.MAX_TRAIN_SAMPLES,
            random_state=config.RANDOM_SEED,
            stratify=df["sentiment_label"],
        )
        df = df.reset_index(drop=True)
        logger.info(f"Sampled {len(df):,} rows (stratified, MAX_TRAIN_SAMPLES={config.MAX_TRAIN_SAMPLES:,})")

    # Three-way stratified split: 70% train / 10% val / 20% test
    train_val_df, test_df = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=df["sentiment_label"],
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=config.VAL_SIZE / (1 - config.TEST_SIZE),
        random_state=config.RANDOM_SEED,
        stratify=train_val_df["sentiment_label"],
    )
    logger.info(f"Train: {len(train_df):,}  |  Val: {len(val_df):,}  |  Test: {len(test_df):,}")

    tokenizer = AutoTokenizer.from_pretrained(config.TRANSFORMER_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.TRANSFORMER_MODEL,
        num_labels=len(config.LABEL_NAMES),
    )

    train_ds = ReviewDataset(train_df["text_transformer"].tolist(), train_df["sentiment_label"].tolist(), tokenizer)
    val_ds = ReviewDataset(val_df["text_transformer"].tolist(), val_df["sentiment_label"].tolist(), tokenizer)
    test_ds = ReviewDataset(test_df["text_transformer"].tolist(), test_df["sentiment_label"].tolist(), tokenizer)

    raw_weights = compute_class_weight(
        "balanced",
        classes=np.array(sorted(df["sentiment_label"].unique())),
        y=train_df["sentiment_label"].values,
    )
    class_weights = torch.tensor(raw_weights, dtype=torch.float)
    logger.info(f"Class weights: {dict(zip(config.LABEL_NAMES, raw_weights.round(3)))}")

    device = _get_device()
    logger.info(f"Training device: {device}")

    use_cuda = device.type == "cuda"

    model_out = config.OUTPUTS / "transformer_model"
    training_args = TrainingArguments(
        output_dir=str(model_out),
        num_train_epochs=config.TRAIN_EPOCHS,
        per_device_train_batch_size=config.BATCH_TRAIN,
        per_device_eval_batch_size=config.BATCH_EVAL,
        learning_rate=config.LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=200,
        seed=config.RANDOM_SEED,
        report_to="none",
        bf16=use_cuda,       # BF16 for CUDA only — MPS uses float32
        fp16=False,
        dataloader_num_workers=0,  # avoids multiprocessing issues on macOS
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )

    logger.info("Training started...")
    trainer.train()

    # Evaluate on held-out test set exactly once
    logger.info("Evaluating on test set...")
    test_preds_output = trainer.predict(test_ds)
    y_pred = np.argmax(test_preds_output.predictions, axis=1)
    y_true = test_df["sentiment_label"].values

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1_per_class = f1_score(y_true, y_pred, average=None)

    logger.info(f"Test Accuracy: {acc:.4f}  |  F1 (weighted): {f1:.4f}")
    for name, val in zip(config.LABEL_NAMES, f1_per_class):
        logger.info(f"  F1 [{name}]: {val:.4f}")
    logger.info(f"Confusion matrix:\n{confusion_matrix(y_true, y_pred)}")
    logger.info(
        "Note: reference numbers (LSTM 0.94, bert-tiny 0.889) are binary is_recommended — "
        "a different task with a higher random baseline (50% vs 33%). Not directly comparable."
    )

    save_metrics({
        "model": config.TRANSFORMER_MODEL,
        "label_source": "rating_3class",
        "accuracy": round(acc, 4),
        "f1_weighted": round(f1, 4),
        "precision_weighted": round(precision, 4),
        "recall_weighted": round(recall, 4),
        "f1_per_class": {n: round(float(v), 4) for n, v in zip(config.LABEL_NAMES, f1_per_class)},
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "epochs": config.TRAIN_EPOCHS,
        "device": str(device),
        "notes": "WeightedTrainer, natural-language input, 3-class rating label",
    }, "transformer")

    model.save_pretrained(str(model_out))
    tokenizer.save_pretrained(str(model_out))
    logger.info(f"Model saved → {model_out}")


if __name__ == "__main__":
    train_transformer()
