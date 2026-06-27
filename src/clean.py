"""
Step 2 — Clean Data & Build Two Text Variants

Path A (text_transformer): minimal cleaning — natural language for DistilBERT.
Path B (text_classic):     full NLP pipeline — for TF-IDF/LogReg baseline only.

The reference notebook applies full NLP cleaning to BERT input, discarding the
casing and function words that transformer attention relies on. We fix this by
keeping two separate pipelines.
"""

import re
import sys
import warnings
from pathlib import Path

import nltk
import pandas as pd
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

sys.path.insert(0, str(Path(__file__).parent))
import config
from utils import get_logger, timer

logger = get_logger(__name__)

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# Negation words are removed from the standard NLTK stopword list.
# Stripping "not", "no", "never" etc. would turn "not recommend" → "recommend",
# "never broke out" → "broke out" — a known fatal error in sentiment pipelines.
_NEGATIONS = {
    "no", "not", "nor", "never", "neither", "nothing", "nobody", "nowhere",
    "without", "cannot", "n't",
}
_stop_words = set(stopwords.words("english")) - _NEGATIONS
_lemmatizer = WordNetLemmatizer()


def _strip_html(text: str) -> str:
    return BeautifulSoup(text, "html.parser").get_text()


def clean_for_transformer(text: str) -> str:
    """Minimal cleaning: strip HTML/URLs, normalize whitespace. Preserve natural language."""
    text = _strip_html(str(text))
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_for_classic(text: str) -> str:
    """Full NLP pipeline for bag-of-words models only."""
    text = clean_for_transformer(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = [_lemmatizer.lemmatize(w) for w in text.split() if w not in _stop_words]
    return " ".join(tokens)


@timer
def build_clean_dataset() -> pd.DataFrame:
    src = config.DATA_CLEAN / "merged_raw.parquet"
    if not src.exists():
        raise FileNotFoundError(f"Run load_data.py first — {src} not found")
    df = pd.read_parquet(src, engine=config.PARQUET_ENGINE)
    logger.info(f"Loaded: {len(df):,} rows")

    # Common filters
    before = len(df)
    df = df.dropna(subset=["review_text"])
    df = df[df["review_text"].str.len() >= 10]
    # Deduplicate on (product_id, lowercased review_text) — pure text dedup
    # would drop two different products that happen to share a copied review.
    dup_key = df["product_id"].astype(str) + "||" + df["review_text"].str.lower()
    df = df[~dup_key.duplicated(keep="first")]
    logger.info(f"After filters: {len(df):,} rows (removed {before - len(df):,})")

    # Build both text paths
    logger.info("Building text_transformer (minimal cleaning)...")
    df["text_transformer"] = df["review_text"].apply(clean_for_transformer)

    logger.info("Building text_classic (full NLP cleaning)...")
    df["text_classic"] = df["text_transformer"].apply(clean_for_classic)

    out = config.DATA_CLEAN / "clean_dataset.parquet"
    df.to_parquet(out, index=False, engine=config.PARQUET_ENGINE)
    logger.info(f"Saved → {out} ({len(df):,} rows)")
    return df


if __name__ == "__main__":
    build_clean_dataset()
