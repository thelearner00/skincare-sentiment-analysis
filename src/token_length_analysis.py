"""
Token Length Analysis — standalone diagnostic script.

Loads the saved DistilBERT tokenizer and measures what fraction of reviews
exceed MAX_SEQ_LEN=128 tokens. Run after train_transformer.py has saved its
tokenizer to outputs/transformer_model/.

Results are logged and saved to outputs/metrics/token_length.json.
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config
from utils import get_logger, save_metrics, timer

logger = get_logger(__name__)

SAMPLE_SIZE = 5_000  # enough for stable percentiles, completes in ~30 s on CPU


@timer
def analyse_token_lengths() -> None:
    tokenizer_path = config.OUTPUTS / "transformer_model"
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Run train_transformer.py first — tokenizer not found at {tokenizer_path}"
        )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    logger.info(f"Loaded tokenizer from {tokenizer_path}")

    src = config.DATA_CLEAN / "labeled_dataset.parquet"
    if not src.exists():
        raise FileNotFoundError(f"Run build_labels.py first — {src} not found")
    df = pd.read_parquet(src, engine=config.PARQUET_ENGINE)
    texts = df["text_transformer"].dropna().tolist()

    # Sample for speed; shuffle so we don't sample only short reviews
    import random
    random.seed(config.RANDOM_SEED)
    random.shuffle(texts)
    sample = texts[:SAMPLE_SIZE]
    logger.info(f"Tokenizing {len(sample):,} reviews (sample from {len(texts):,} total)")

    lengths = sorted(
        len(tokenizer.encode(t, add_special_tokens=True)) for t in sample
    )
    n = len(lengths)
    max_len = config.MAX_SEQ_LEN
    truncated = sum(1 for l in lengths if l > max_len)
    truncation_pct = truncated / n * 100

    p50  = lengths[n // 2]
    p75  = lengths[int(n * 0.75)]
    p90  = lengths[int(n * 0.90)]
    p95  = lengths[int(n * 0.95)]
    p99  = lengths[int(n * 0.99)]
    p_max = lengths[-1]

    logger.info(
        f"Token length distribution (n={n:,}):\n"
        f"  P50={p50}  P75={p75}  P90={p90}  P95={p95}  P99={p99}  max={p_max}"
    )
    logger.info(
        f"Reviews truncated at {max_len} tokens: {truncated:,} / {n:,} = {truncation_pct:.1f}%"
    )

    if truncation_pct > 30:
        logger.warning(
            f"High truncation rate ({truncation_pct:.1f}%). Consider increasing MAX_SEQ_LEN "
            f"or verify that sentiment-bearing content appears early in reviews."
        )

    save_metrics(
        {
            "sample_size": n,
            "max_seq_len": max_len,
            "truncation_rate_pct": round(truncation_pct, 2),
            "p50_tokens": p50,
            "p75_tokens": p75,
            "p90_tokens": p90,
            "p95_tokens": p95,
            "p99_tokens": p99,
            "max_tokens": p_max,
        },
        "token_length",
    )


if __name__ == "__main__":
    analyse_token_lengths()
