"""
Step 1 — Load & Merge

Concatenates all five review files, joins to the skincare product metadata,
and writes a single merged CSV for all downstream steps.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config
from utils import get_logger, timer

logger = get_logger(__name__)


def load_products() -> pd.DataFrame:
    path = config.DATA_RAW / "product_info_skincare.csv"
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    before = len(df)
    df = df.drop_duplicates(subset="product_id")
    if len(df) < before:
        logger.warning(f"Dropped {before - len(df)} duplicate product rows before join")
    logger.info(f"Products loaded: {df.shape[0]} rows, {df.shape[1]} cols")
    return df


def load_reviews() -> pd.DataFrame:
    files = sorted(config.DATA_RAW.glob("reviews_*_masked.csv"))
    if not files:
        raise FileNotFoundError(f"No review files found in {config.DATA_RAW}")
    parts = []
    for f in files:
        part = pd.read_csv(f)
        part = part.loc[:, ~part.columns.str.startswith("Unnamed")]
        logger.info(f"  {f.name}: {len(part):,} rows")
        parts.append(part)
    reviews = pd.concat(parts, ignore_index=True)
    logger.info(f"Reviews combined: {len(reviews):,} rows")
    return reviews


@timer
def merge_and_save(reviews: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    n_before = len(reviews)

    product_cols = ["product_id", "brand_name", "primary_category", "secondary_category",
                    "tertiary_category", "price_usd", "ingredients"]
    merged = reviews.merge(
        products[product_cols],
        on="product_id",
        how="inner",          # keep only reviews that match a skincare product
        suffixes=("", "_product"),
    )

    n_after = len(merged)
    dropped = n_before - n_after
    pct_dropped = dropped / n_before * 100
    logger.info(f"Merge complete: {n_after:,} rows kept, {dropped:,} dropped ({pct_dropped:.1f}% — non-skincare products)")
    if n_after == 0:
        raise ValueError("Merge produced zero rows — check product_id alignment between reviews and products")
    if pct_dropped > 50:
        logger.warning(f"Over 50% of reviews dropped — verify data alignment (expected if dataset includes non-skincare products)")

    # Null audit
    null_rates = merged.isnull().mean().sort_values(ascending=False)
    high_null = null_rates[null_rates > 0.5]
    if not high_null.empty:
        logger.warning(f"Columns with >50% nulls:\n{high_null.to_string()}")

    out = config.DATA_CLEAN / "merged_raw.parquet"
    config.DATA_CLEAN.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False, engine=config.PARQUET_ENGINE)
    logger.info(f"Saved → {out}")
    return merged


if __name__ == "__main__":
    products = load_products()
    reviews = load_reviews()
    merge_and_save(reviews, products)
