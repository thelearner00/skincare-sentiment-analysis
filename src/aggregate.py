"""
Step 10 — Aggregation & Model Comparison Table

Produces:
  - Brand-level summary (mean sentiment + per-aspect scores)
  - Product-level summary
  - Model comparison table: this project's measured results vs reference notebook's reported numbers
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config
from utils import get_logger, load_metrics, timer

logger = get_logger(__name__)

ASPECT_COLS = list(config.ASPECT_KEYWORDS.keys())


@timer
def aggregate() -> None:
    src = config.DATA_CLEAN / "aspect_scored.parquet"
    if not src.exists():
        raise FileNotFoundError(f"Run aspect_sentiment.py first — {src} not found")
    df = pd.read_parquet(src, engine=config.PARQUET_ENGINE)
    logger.info(f"Loaded: {len(df):,} rows")

    summary_cols = ASPECT_COLS + ["sentiment_label", "rating"]

    # Brand summary
    brand_summary = (
        df.groupby("brand_name")[summary_cols]
        .agg(["mean", "count"])
    )
    brand_summary.columns = [f"{col}_{stat}" for col, stat in brand_summary.columns]
    brand_summary = brand_summary.reset_index()

    out_brand = config.OUTPUTS / "brand_summary.csv"
    brand_summary.to_csv(out_brand, index=False)
    logger.info(f"Brand summary saved → {out_brand} ({len(brand_summary):,} brands)")

    # Product summary
    product_summary = (
        df.groupby(["product_id", "product_name", "brand_name"])[summary_cols]
        .agg(["mean", "count"])
    )
    product_summary.columns = [f"{col}_{stat}" for col, stat in product_summary.columns]
    product_summary = product_summary.reset_index()

    out_product = config.OUTPUTS / "product_summary.csv"
    product_summary.to_csv(out_product, index=False)
    logger.info(f"Product summary saved → {out_product} ({len(product_summary):,} products)")

    # Model comparison table
    baseline = load_metrics("baseline")
    transformer = load_metrics("transformer")
    label_dist = load_metrics("label_distribution")

    majority_baseline_acc = label_dist.get("majority_class_baseline_accuracy", 0.832)

    # NOTE: Reference rows (LSTM, bert-tiny) use binary is_recommended labels.
    # Our rows use 3-class rating labels. Random baseline is 50% vs 33% respectively.
    # The majority-class baseline (always predict "positive") is the correct lower bound
    # for our 3-class results — at 83.2% it is close to the TF-IDF number, which is
    # why per-class and weighted F1 matter more than raw accuracy here.
    comparison = pd.DataFrame([
        {
            "model": "Majority-class baseline (always predict positive)",
            "label_type": "rating (3-class)",
            "num_classes": 3,
            "accuracy": majority_baseline_acc,
            "f1_weighted": None,
            "n_test": baseline.get("test_size"),
            "notes": "Lower bound — predict the most frequent class every time",
        },
        {
            "model": "This project — TF-IDF + LogReg",
            "label_type": "rating (3-class)",
            "num_classes": 3,
            "accuracy": baseline.get("accuracy"),
            "f1_weighted": baseline.get("f1_weighted"),
            "n_test": baseline.get("test_size"),
            "notes": "class_weight=balanced, negations preserved; shared-test F1="
                     + str(baseline.get("shared_test_f1_weighted", "—")),
        },
        {
            "model": f"This project — {config.TRANSFORMER_MODEL}",
            "label_type": "rating (3-class)",
            "num_classes": 3,
            "accuracy": transformer.get("accuracy"),
            "f1_weighted": transformer.get("f1_weighted"),
            "n_test": transformer.get("test_size"),
            "notes": "WeightedTrainer, natural-language input, 66M params",
        },
        {
            "model": "— Reference — LSTM (DIFFERENT TASK)",
            "label_type": "is_recommended (binary) ← not comparable",
            "num_classes": 2,
            "accuracy": 0.94,
            "f1_weighted": None,
            "n_test": None,
            "notes": "Binary label; random baseline=50% vs 33% for our task. Not directly comparable.",
        },
        {
            "model": "— Reference — bert-tiny (DIFFERENT TASK)",
            "label_type": "is_recommended (binary) ← not comparable",
            "num_classes": 2,
            "accuracy": 0.889,
            "f1_weighted": None,
            "n_test": None,
            "notes": "bert-tiny 4.4M params, stemmed input; binary label. Not directly comparable.",
        },
    ])

    out_comparison = config.OUTPUTS / "model_comparison.csv"
    comparison.to_csv(out_comparison, index=False)
    logger.info(f"\nModel comparison:\n{comparison.to_string()}")
    logger.info(f"Saved → {out_comparison}")


if __name__ == "__main__":
    aggregate()
