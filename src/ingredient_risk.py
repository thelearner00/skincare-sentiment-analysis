"""
Step 9 — Ingredient Risk-Signal Analysis

Cross-references product ingredients against breakout/irritation complaint patterns.
Identifies ingredients that appear disproportionately often in high-complaint products
versus low-complaint products.

IMPORTANT: this is a frequency/association signal, not causal evidence. The output CSV
includes an explicit column marking this distinction. Frame it this way in the README
and in the dashboard tooltip.

Thresholds are on aspect scores as P(positive) in [0, 1]:
  HIGH_COMPLAINT_THRESHOLD = 0.35  → products where reviewers rarely report positive breakout outcomes
  LOW_COMPLAINT_THRESHOLD  = 0.65  → products where reviewers mostly report positive outcomes
  MIN_INGREDIENT_COUNT = 5        → minimum appearances in high-complaint products to enter the table
"""

import ast
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from scipy.stats import fisher_exact

sys.path.insert(0, str(Path(__file__).parent))
import config
from utils import get_logger, timer

logger = get_logger(__name__)


def parse_ingredients(raw: str) -> list[str]:
    """Parse the Python-list-literal ingredients field into individual ingredient names."""
    try:
        items = ast.literal_eval(raw)
    except Exception:
        return []

    ingredients = []
    for item in items:
        if not isinstance(item, str):
            continue
        item = item.strip()
        # Variation labels: short, no commas, ends with ':'
        if item.endswith(":") and "," not in item:
            continue
        for ing in item.split(","):
            cleaned = ing.strip().lower()
            cleaned = re.sub(r"\(.*?\)", "", cleaned).strip()
            cleaned = re.sub(r"\s+", " ", cleaned)
            if len(cleaned) > 3:
                ingredients.append(cleaned)
    return ingredients


def build_ingredient_matrix(products: pd.DataFrame, product_ids) -> Counter:
    """Count ingredient appearances across a set of product_ids. Vectorised (no iterrows)."""
    subset = products[products["product_id"].isin(product_ids)]["ingredients"].dropna()
    counter: Counter = Counter()
    for raw in subset:
        counter.update(parse_ingredients(str(raw)))
    return counter


@timer
def ingredient_risk() -> None:
    products = pd.read_csv(config.DATA_RAW / "product_info_skincare.csv")
    products = products.loc[:, ~products.columns.str.startswith("Unnamed")]

    aspect_df = pd.read_parquet(
        config.DATA_CLEAN / "aspect_scored.parquet", engine=config.PARQUET_ENGINE
    )
    if "breakouts_irritation" not in aspect_df.columns:
        raise ValueError("Run aspect_sentiment.py first — breakouts_irritation column missing")

    # Per-product mean P(positive) for breakouts_irritation
    product_complaint = (
        aspect_df.groupby("product_id")["breakouts_irritation"]
        .mean()
        .dropna()
    )
    logger.info(f"Products with aspect data: {len(product_complaint):,}")
    logger.info(f"breakouts_irritation score distribution:\n{product_complaint.describe().round(3)}")

    high_ids = product_complaint[product_complaint < config.HIGH_COMPLAINT_THRESHOLD].index
    low_ids = product_complaint[product_complaint >= config.LOW_COMPLAINT_THRESHOLD].index
    excluded = len(product_complaint) - len(high_ids) - len(low_ids)

    logger.info(
        f"High-complaint products (P(positive) < {config.HIGH_COMPLAINT_THRESHOLD}): {len(high_ids):,}"
    )
    logger.info(
        f"Low-complaint products  (P(positive) >= {config.LOW_COMPLAINT_THRESHOLD}): {len(low_ids):,}"
    )
    logger.info(
        f"Excluded (grey zone {config.HIGH_COMPLAINT_THRESHOLD}–{config.LOW_COMPLAINT_THRESHOLD}): "
        f"{excluded:,} products"
    )

    high_freq = build_ingredient_matrix(products, high_ids)
    low_freq = build_ingredient_matrix(products, low_ids)

    if not high_freq:
        logger.warning("No ingredient data found for high-complaint products — check ingredient parsing")
        return

    n_high = max(len(high_ids), 1)
    n_low = max(len(low_ids), 1)

    rows = []
    for ingredient, high_count in high_freq.most_common():
        if high_count < config.MIN_INGREDIENT_COUNT:
            continue
        low_count = low_freq.get(ingredient, 0)

        # Risk ratio: how much more common per product in high-complaint vs low-complaint group
        high_rate = high_count / n_high
        low_rate = low_count / n_low
        risk_ratio = high_rate / low_rate if low_rate > 0 else float("inf")

        # Fisher's exact test: 2×2 contingency [high_with, high_without; low_with, low_without]
        contingency = [
            [high_count, n_high - high_count],
            [low_count, n_low - low_count],
        ]
        if n_high - high_count >= 0 and n_low - low_count >= 0:
            _, p_value = fisher_exact(contingency, alternative="greater")
        else:
            p_value = float("nan")

        rows.append({
            "ingredient": ingredient,
            "count_in_high_complaint_products": high_count,
            "count_in_low_complaint_products": low_count,
            "risk_ratio_association_only": round(risk_ratio, 2),
            "p_value_fishers_exact": round(p_value, 4),
            "association_only_not_causal": True,
        })

    if not rows:
        logger.warning(
            f"No ingredients passed the MIN_INGREDIENT_COUNT={config.MIN_INGREDIENT_COUNT} filter. "
            f"High-complaint group may be too small."
        )
        return

    risk_table = pd.DataFrame(rows).sort_values("risk_ratio_association_only", ascending=False)

    out = config.OUTPUTS / "ingredient_risk_signal.csv"
    risk_table.to_csv(out, index=False)
    logger.info(f"Saved → {out} ({len(risk_table):,} ingredients after min_count filter)")
    logger.info(f"\nTop 15 flagged ingredients:\n{risk_table.head(15).to_string()}")


if __name__ == "__main__":
    ingredient_risk()
