"""
Step 6 — Disagreement Analysis

Compares model-predicted text sentiment against is_recommended for every review
in the transformer's held-out test set.

Restricting to the test set matters: the disagreement analysis on training-set reviews
reflects memorisation, not generalisation. The 1.76% rate from the original full-dataset
run was inflated because the model's confidence on its own training examples is artificially high.

This script also updates transformer.json with per-class F1, since it already has the
true labels and predicted labels for the test set.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
import config
from utils import batch_predict, get_device, get_logger, get_transformer_test_df, load_metrics, save_metrics, timer

logger = get_logger(__name__)


@timer
def disagreement_analysis() -> None:
    model_path = config.OUTPUTS / "transformer_model"
    if not model_path.exists():
        raise FileNotFoundError(f"Run train_transformer.py first — {model_path} not found")

    labeled = pd.read_parquet(config.DATA_CLEAN / "labeled_dataset.parquet", engine=config.PARQUET_ENGINE)

    # Reproduce the exact test split used during training (deterministic, seed=42)
    test_df = get_transformer_test_df(labeled)
    logger.info(f"Transformer test set: {len(test_df):,} rows (reproduced from same seed/split as training)")

    # Keep only rows where is_recommended is available for disagreement analysis
    disagree_df = test_df.dropna(subset=["is_recommended"]).copy()
    disagree_df["is_recommended"] = disagree_df["is_recommended"].astype(int)
    logger.info(f"Rows with is_recommended in test set: {len(disagree_df):,}")

    device = get_device()
    logger.info(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path)).to(device)

    # Run inference on the full test set first (for per-class metrics)
    logger.info("Running inference on full test set for per-class metrics...")
    test_preds = batch_predict(
        test_df["text_transformer"].tolist(), model, tokenizer, device,
        batch_size=64, max_length=config.MAX_SEQ_LEN,
    )
    y_true = test_df["sentiment_label"].values
    y_pred = np.array(test_preds)

    acc = accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    f1_per_class = f1_score(y_true, y_pred, average=None)

    logger.info(f"\n{classification_report(y_true, y_pred, target_names=config.LABEL_NAMES)}")

    # Update transformer.json with per-class F1 (previously only saved weighted aggregate)
    transformer_metrics = load_metrics("transformer")
    transformer_metrics.update({
        "accuracy": round(acc, 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "f1_per_class": {n: round(float(v), 4) for n, v in zip(config.LABEL_NAMES, f1_per_class)},
    })
    save_metrics(transformer_metrics, "transformer")
    logger.info("transformer.json updated with per-class F1")

    # Run inference on the disagree subset (is_recommended available)
    logger.info("Running inference on disagreement subset...")
    disagree_df["predicted_sentiment"] = batch_predict(
        disagree_df["text_transformer"].tolist(), model, tokenizer, device,
        batch_size=64, max_length=config.MAX_SEQ_LEN,
    )

    # Flag disagreements (neutral predictions are intentionally excluded)
    disagree_df["recommended_but_negative"] = (
        (disagree_df["is_recommended"] == 1) & (disagree_df["predicted_sentiment"] == 0)
    )
    disagree_df["not_recommended_but_positive"] = (
        (disagree_df["is_recommended"] == 0) & (disagree_df["predicted_sentiment"] == 2)
    )
    disagree_df["is_disagreement"] = (
        disagree_df["recommended_but_negative"] | disagree_df["not_recommended_but_positive"]
    )

    total = len(disagree_df)
    n_disagree = disagree_df["is_disagreement"].sum()
    n_rec_neg = disagree_df["recommended_but_negative"].sum()
    n_norec_pos = disagree_df["not_recommended_but_positive"].sum()

    logger.info(f"Disagreement rate (test set only): {n_disagree / total * 100:.2f}% ({n_disagree:,} / {total:,})")
    logger.info(f"  Recommended but negative text: {n_rec_neg:,} ({n_rec_neg / total * 100:.2f}%)")
    logger.info(f"  Not recommended but positive text: {n_norec_pos:,} ({n_norec_pos / total * 100:.2f}%)")

    examples = disagree_df[disagree_df["is_disagreement"]].nlargest(10, "predicted_sentiment")[
        [c for c in ["review_text", "rating", "is_recommended", "predicted_sentiment", "product_name", "brand_name"]
         if c in disagree_df.columns]
    ]
    logger.info(f"\nTop 10 disagreement cases:\n{examples.to_string()}")

    disagreements = disagree_df[disagree_df["is_disagreement"]].sort_values("product_id")
    save_cols = [c for c in [
        "review_text", "rating", "is_recommended", "predicted_sentiment",
        "product_id", "product_name", "brand_name", "skin_type", "skin_tone",
        "recommended_but_negative", "not_recommended_but_positive",
    ] if c in disagreements.columns]

    out = config.OUTPUTS / "disagreement_cases.csv"
    disagreements[save_cols].to_csv(out, index=False)
    logger.info(f"Saved → {out}")

    save_metrics({
        "scope": "transformer_test_set_only",
        "total_reviews_analyzed": total,
        "disagreement_count": int(n_disagree),
        "disagreement_rate_pct": round(n_disagree / total * 100, 2),
        "recommended_but_negative": int(n_rec_neg),
        "not_recommended_but_positive": int(n_norec_pos),
        "note": "Neutral predictions (class 1) excluded from both disagreement categories by design",
    }, "disagreement")


if __name__ == "__main__":
    disagreement_analysis()
