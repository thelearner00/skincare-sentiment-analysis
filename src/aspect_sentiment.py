"""
Step 7 — Aspect-Based Sentiment Extraction

Scores each review across 6 product aspects: hydration, breakouts_irritation,
scent, packaging, price_value, texture_application.

Approach: keyword-seeded sentence-level scoring using the fine-tuned transformer.

Three-phase design for efficiency:
  Phase 1 — Extract all matching sentences across all reviews (pure Python, fast).
  Phase 2 — Single large batched inference pass over all collected sentences.
  Phase 3 — Map predictions back to per-review, per-aspect mean scores.

Scores are softmax P(positive class) in [0, 1] — NOT ordinal class indices.
Averaging class indices (0/1/2) conflates "one negative + one positive" with
"two neutrals". Averaging P(positive) preserves confidence and is on a continuous scale.
"""

import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config
from utils import batch_predict, get_device, get_logger, timer

logger = get_logger(__name__)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(str(text)) if len(s.strip()) > 5]


def sentences_for_aspect(sentences: list[str], keywords: list[str]) -> list[str]:
    return [s for s in sentences if any(kw in s.lower() for kw in keywords)]


@timer
def run_aspect_sentiment(sample_n: Optional[int] = None) -> None:
    model_path = config.OUTPUTS / "transformer_model"
    if not model_path.exists():
        raise FileNotFoundError(f"Run train_transformer.py first — {model_path} not found")

    src = config.DATA_CLEAN / "labeled_dataset.parquet"
    df = pd.read_parquet(src, engine=config.PARQUET_ENGINE).dropna(subset=["text_transformer"])
    df = df.reset_index(drop=True)
    logger.info(f"Loaded: {len(df):,} rows")

    if sample_n is not None:
        df = df.sample(n=sample_n, random_state=config.RANDOM_SEED).reset_index(drop=True)
        logger.info(f"Using sample of {sample_n:,} — omit --sample for full dataset")

    device = get_device()
    logger.info(f"Device: {device}")

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path)).to(device)
    model.eval()

    # ── Phase 1: collect all matching (sentence, review_idx, aspect) triples ──
    logger.info("Phase 1 — extracting aspect-matching sentences from all reviews...")
    flat_sentences: list[str] = []
    flat_review_idxs: list[int] = []
    flat_aspects: list[str] = []

    from tqdm import tqdm
    for review_idx, text in enumerate(tqdm(df["text_transformer"].tolist(), desc="Extracting")):
        sentences = split_sentences(text)
        for aspect, keywords in config.ASPECT_KEYWORDS.items():
            for sent in sentences_for_aspect(sentences, keywords):
                flat_sentences.append(sent)
                flat_review_idxs.append(review_idx)
                flat_aspects.append(aspect)

    logger.info(f"Total sentence-aspect pairs to score: {len(flat_sentences):,}")

    # ── Phase 2: single batched inference pass — returns P(positive) in [0, 1] ──
    logger.info("Phase 2 — batched inference (returning softmax P(positive))...")
    all_scores = batch_predict(
        flat_sentences, model, tokenizer, device,
        batch_size=256, max_length=64, return_proba=True,
    )

    # ── Phase 3: aggregate predictions back to per-review per-aspect means ─────
    # Using P(positive) means a mean of 0.5 = borderline; 0.9 = strongly positive.
    # This is continuous and avoids the ordinal-averaging problem (mean([0,2]) = 1 ≠ neutral).
    score_accumulator: dict[tuple[int, str], list[float]] = defaultdict(list)
    for review_idx, aspect, score in zip(flat_review_idxs, flat_aspects, all_scores):
        score_accumulator[(review_idx, aspect)].append(score)

    n_rows = len(df)
    for aspect in config.ASPECT_KEYWORDS:
        df[aspect] = [
            float(np.mean(score_accumulator[(i, aspect)]))
            if (i, aspect) in score_accumulator
            else None
            for i in range(n_rows)
        ]

    # Null-rate audit — most reviews won't mention all 6 aspects
    for aspect in config.ASPECT_KEYWORDS:
        null_pct = df[aspect].isna().mean() * 100
        logger.info(f"  {aspect}: {null_pct:.1f}% reviews have no keyword match")

    out = config.DATA_CLEAN / "aspect_scored.parquet"
    df.to_parquet(out, index=False, engine=config.PARQUET_ENGINE)
    logger.info(f"Saved → {out} ({len(df):,} rows)")
    logger.info("Aspect scores are P(positive) in [0, 1] — 0=negative sentiment, 1=positive sentiment")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Rows to sample for a quick validation run (omit for full dataset)",
    )
    args = parser.parse_args()
    run_aspect_sentiment(sample_n=args.sample)
