from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_CLEAN = ROOT / "data" / "clean"
OUTPUTS = ROOT / "outputs"
METRICS = OUTPUTS / "metrics"

# Intermediate pipeline files use Parquet to safely handle complex text fields
# (ingredients, review_text) that contain embedded newlines and special chars.
# Final human-readable outputs in outputs/ remain CSV.
PARQUET_ENGINE = "pyarrow"

RANDOM_SEED = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.10  # fraction of train set, after test split

# Cap training rows for transformer fine-tuning on CPU/MPS.
# Set to None to use the full dataset (use on a CUDA GPU for full training).
# 50k gives good F1 in ~25 min on Apple Silicon MPS.
MAX_TRAIN_SAMPLES = 50_000

# Sentiment label mapping: rating → 0 (negative), 1 (neutral), 2 (positive)
LABEL_MAP = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}
LABEL_NAMES = ["negative", "neutral", "positive"]

# TF-IDF baseline
TFIDF_MAX_FEATURES = 20_000
TFIDF_NGRAM_RANGE = (1, 2)

# Transformer
TRANSFORMER_MODEL = "distilbert-base-uncased"
MAX_SEQ_LEN = 128
TRAIN_EPOCHS = 3
BATCH_TRAIN = 16
BATCH_EVAL = 32
LEARNING_RATE = 2e-5

# Aspect keyword seeds — single source of truth used by aspect_sentiment and dashboard
ASPECT_KEYWORDS: dict[str, list[str]] = {
    "hydration": ["moistur", "hydrat", "dry", "dewy", "plump"],
    "breakouts_irritation": ["breakout", "acne", "irritat", "redness", "rash", "clog", "sensit"],
    "scent": ["smell", "scent", "fragrance", "odor"],
    "packaging": ["bottle", "pump", "packaging", "leak", "container", "tube"],
    "price_value": ["expensive", "worth", "price", "value", "cheap", "overprice"],
    "texture_application": ["texture", "greasy", "absorb", "sticky", "lightweight", "thick"],
}

# Segment analysis — minimum reviews per segment to be considered reliable
MIN_SEGMENT_SIZE = 30
# Aspect scores are softmax P(positive) in [0, 1]; 0.15 delta ≈ 15 percentage-point divergence
DIVERGENCE_THRESHOLD = 0.15

# Ingredient risk — thresholds on mean P(positive) for breakouts_irritation aspect
# A product with mean P(positive) < 0.35 has predominantly negative/neutral breakout mentions
HIGH_COMPLAINT_THRESHOLD = 0.35
LOW_COMPLAINT_THRESHOLD = 0.65
# Minimum appearances in high-complaint products to be included in the risk table
MIN_INGREDIENT_COUNT = 5
