"""
Step 3 — Build Sentiment Labels

Derives a 3-class sentiment label from the star rating.
Keeps is_recommended as a separate column for disagreement analysis (Step 6).

The reference notebook used is_recommended as the label directly, which means
the model learned to predict a behavioral checkbox rather than text-expressed sentiment.
We fix this by using the rating — and the disagreement analysis becomes possible precisely
because is_recommended is NOT the label here.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config
from utils import get_logger, timer

logger = get_logger(__name__)


@timer
def build_labels() -> pd.DataFrame:
    src = config.DATA_CLEAN / "clean_dataset.parquet"
    if not src.exists():
        raise FileNotFoundError(f"Run clean.py first — {src} not found")
    df = pd.read_parquet(src, engine=config.PARQUET_ENGINE)
    logger.info(f"Loaded: {len(df):,} rows")

    # Drop rows with missing rating
    before = len(df)
    df = df.dropna(subset=["rating"])
    # Validate before cast — non-integer ratings (e.g. 3.5) would silently truncate
    valid_ratings = df["rating"].isin([1, 2, 3, 4, 5])
    if not valid_ratings.all():
        bad = df[~valid_ratings]["rating"].unique().tolist()
        raise ValueError(f"Unexpected rating values outside {{1,2,3,4,5}}: {bad}")
    df["rating"] = df["rating"].astype(int)
    logger.info(f"Rows with valid rating: {len(df):,} (dropped {before - len(df):,})")

    df["sentiment_label"] = df["rating"].map(config.LABEL_MAP)
    if df["sentiment_label"].isna().any():
        raise ValueError("Unexpected rating value produced NaN label — extend LABEL_MAP")
    if df["sentiment_label"].nunique() != 3:
        raise ValueError(f"Expected 3 sentiment classes, got {df['sentiment_label'].nunique()}")

    dist = df["sentiment_label"].value_counts().sort_index()
    class_fractions = {}
    for label_id, count in dist.items():
        pct = count / len(df) * 100
        class_fractions[config.LABEL_NAMES[label_id]] = round(pct / 100, 4)
        logger.info(f"  {config.LABEL_NAMES[label_id]:>10}: {count:>7,} ({pct:.1f}%)")

    majority = dist.max()
    minority = dist.min()
    majority_class_accuracy = round(majority / len(df), 4)
    logger.info(f"Imbalance ratio (majority/minority): {majority / minority:.1f}x — handled via class weighting at training")
    logger.info(f"Majority-class baseline accuracy: {majority_class_accuracy:.4f} (always predict positive)")

    # is_recommended stays as a separate column — confirmed present
    if "is_recommended" in df.columns:
        rec_rate = df["is_recommended"].mean()
        logger.info(f"is_recommended non-null rate: {df['is_recommended'].notna().mean()*100:.1f}%, positive rate: {rec_rate*100:.1f}%")
    else:
        logger.warning("is_recommended column not found — disagreement analysis will not be possible")

    # Persist class distribution so downstream readers don't need to recompute
    from utils import save_metrics
    save_metrics({
        "class_fractions": class_fractions,
        "majority_class_baseline_accuracy": majority_class_accuracy,
        "imbalance_ratio": round(majority / minority, 2),
        "total_rows": len(df),
    }, "label_distribution")

    out = config.DATA_CLEAN / "labeled_dataset.parquet"
    df.to_parquet(out, index=False, engine=config.PARQUET_ENGINE)
    logger.info(f"Saved → {out}")
    return df


if __name__ == "__main__":
    build_labels()
