"""
Step 8 — Skin-Type & Skin-Tone Segmentation

For each product, computes per-aspect sentiment broken out by skin_type and skin_tone,
then finds products where a specific segment diverges notably from the product's overall average.

Aspect scores are P(positive) in [0, 1]; DIVERGENCE_THRESHOLD is expressed in the same scale
(0.15 = 15 percentage-point divergence, previously 0.3 on the [0-2] ordinal scale).

Three improvements over the initial version:
  - Flagging covers ALL six aspects, not just breakouts_irritation.
  - low_confidence uses the minimum count across ALL aspects (conservative): if any aspect
    has fewer than MIN_SEGMENT_SIZE mentions in the segment, the row is low-confidence.
  - product_name is included in the output CSV for dashboard readability.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config
from utils import get_logger, timer

logger = get_logger(__name__)

ASPECT_COLS = list(config.ASPECT_KEYWORDS.keys())


def compute_divergence(df: pd.DataFrame, segment_col: str, out_filename: str) -> None:
    df_valid = df.dropna(subset=[segment_col])
    logger.info(
        f"{segment_col}: {df_valid[segment_col].nunique()} unique values, "
        f"{df_valid[segment_col].notna().mean() * 100:.1f}% non-null"
    )

    # Per-product overall means (across all reviewers, all skin types)
    product_overall = (
        df_valid.groupby("product_id")[ASPECT_COLS]
        .mean()
        .rename(columns={c: f"{c}_overall" for c in ASPECT_COLS})
    )

    # Per-product, per-segment: mean and non-null count per aspect
    agg_dict = {}
    for col in ASPECT_COLS:
        agg_dict[col] = ["mean", "count"]  # count = reviews that mentioned this aspect

    segment_agg = (
        df_valid.groupby(["product_id", segment_col])[ASPECT_COLS]
        .agg(["mean", "count"])
    )
    # Flatten MultiIndex columns: (aspect, stat) → "aspect_mean" / "aspect_count"
    segment_agg.columns = [f"{col}_{stat}" for col, stat in segment_agg.columns]
    segment_agg = segment_agg.reset_index()

    # Join overall means and compute delta per aspect
    merged = segment_agg.merge(product_overall, on="product_id", how="left")
    for col in ASPECT_COLS:
        merged[f"{col}_delta"] = merged[f"{col}_mean"] - merged[f"{col}_overall"]

    # Add product_name for readability in the dashboard (product_id alone is opaque)
    product_name_map = df[["product_id", "product_name"]].drop_duplicates("product_id")
    merged = merged.merge(product_name_map, on="product_id", how="left")

    # low_confidence: based on the count of the PRIMARY flagging aspect (the one with the
    # largest |delta|), not a single proxy or the minimum across all aspects.
    # This avoids false confidence when a flagged aspect has effective n=1,
    # while not over-penalising segments whose flagging aspect has plenty of mentions.
    delta_cols_list = [f"{c}_delta" for c in ASPECT_COLS]
    primary_delta_col = merged[delta_cols_list].fillna(0).abs().idxmax(axis=1)
    primary_count_col = primary_delta_col.str.replace("_delta", "_count")
    primary_count = pd.Series(
        [merged.at[i, col] if col in merged.columns else 0
         for i, col in primary_count_col.items()],
        index=merged.index,
    )
    merged["low_confidence"] = primary_count < config.MIN_SEGMENT_SIZE

    # Flag any segment where ANY aspect delta exceeds the threshold (not just breakouts)
    delta_cols = [f"{c}_delta" for c in ASPECT_COLS]
    merged["flagged"] = merged[delta_cols].abs().max(axis=1) > config.DIVERGENCE_THRESHOLD

    # Separate flag for breakouts specifically (for backward compatibility)
    merged["breakouts_flagged"] = merged["breakouts_irritation_delta"].abs() > config.DIVERGENCE_THRESHOLD

    flagged = merged[merged["flagged"] & ~merged["low_confidence"]].sort_values(
        "breakouts_irritation_delta"
    )
    logger.info(
        f"Flagged {len(flagged):,} {segment_col} segments with "
        f"any |aspect_delta| > {config.DIVERGENCE_THRESHOLD}"
    )

    if not flagged.empty:
        cols_to_show = (
            ["product_id", segment_col, "breakouts_irritation_delta",
             "hydration_delta", "breakouts_irritation_count"]
        )
        cols_to_show = [c for c in cols_to_show if c in flagged.columns]
        logger.info(f"\nTop flagged ({segment_col}):\n{flagged[cols_to_show].head(10).to_string()}")

    out = config.OUTPUTS / out_filename
    merged.to_csv(out, index=False)
    logger.info(f"Saved → {out} ({len(merged):,} rows)")


@timer
def segment_analysis() -> None:
    src = config.DATA_CLEAN / "aspect_scored.parquet"
    if not src.exists():
        raise FileNotFoundError(f"Run aspect_sentiment.py first — {src} not found")
    df = pd.read_parquet(src, engine=config.PARQUET_ENGINE)
    logger.info(f"Loaded: {len(df):,} rows")

    compute_divergence(df, "skin_type", "skin_type_divergence.csv")
    compute_divergence(df, "skin_tone", "skin_tone_divergence.csv")


if __name__ == "__main__":
    segment_analysis()
