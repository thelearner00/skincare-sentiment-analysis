# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What We're Building

A skincare review sentiment analysis project that directly addresses specific weaknesses in a reference Kaggle notebook (`skincare-products-eda-sentiment-analysis.ipynb`, included in the repo). The goal is not to replicate the reference — it's to fix its methodological gaps and add a personalization layer it never builds, then prove it with measured results side by side.

## Python Version

**System Python is 3.9.6** (`/usr/bin/python3`). There is no virtualenv — install directly with `pip3`. This means:
- Use `Optional[int]` from `typing`, **not** `int | None` (PEP 604 union syntax requires Python 3.10+)
- Built-in generic aliases like `list[str]`, `dict[str, Any]` are fine (PEP 585, Python 3.9+)

## Commands

```bash
# One-time setup
pip3 install -r skincare-sentiment-analysis/requirements.txt
python3 -m nltk.downloader stopwords wordnet

# Run pipeline from skincare-sentiment-analysis/
cd skincare-sentiment-analysis
python3 src/load_data.py        # Step 1 — merge reviews + products
python3 src/clean.py            # Step 2 — build two text variants
python3 src/build_labels.py     # Step 3 — rating-derived sentiment labels
python3 src/train_baseline.py   # Step 4 — TF-IDF + Logistic Regression
python3 src/train_transformer.py   # Step 5 — DistilBERT fine-tune (~47 min on Apple Silicon MPS)
python3 src/disagreement.py        # Step 6 — text sentiment vs. is_recommended
python3 src/aspect_sentiment.py    # Step 7 — aspect-level scoring (full dataset)
python3 src/aspect_sentiment.py --sample 5000  # quick validation run
python3 src/segment_analysis.py    # Step 8 — skin-type divergence
python3 src/ingredient_risk.py     # Step 9 — ingredient association
python3 src/aggregate.py           # Step 10 — summary tables + model comparison CSV

# Dashboard
streamlit run app/dashboard.py
```

## Intermediate Files

All pipeline intermediates are **Parquet** (not CSV) — the `review_text` and `ingredients` fields contain embedded newlines and special characters that break CSV parsing. Final human-readable outputs in `outputs/` are CSV.

```
data/clean/merged_raw.parquet      ← Step 1 output
data/clean/clean_dataset.parquet   ← Step 2 output
data/clean/labeled_dataset.parquet ← Step 3 output
data/clean/aspect_scored.parquet   ← Step 7 output (adds 6 aspect columns)
```

## Key Architectural Decisions (and why)

### 1. Rating-derived labels, not `is_recommended`
The reference notebook uses `is_recommended` (a behavioral checkbox) as the sentiment label. This makes the model predict a separate explicit signal rather than text-expressed sentiment — an easier, less meaningful task. We derive labels from star rating: 1–2 → negative (0), 3 → neutral (1), 4–5 → positive (2).

`is_recommended` is kept as an independent column for disagreement analysis in Step 6 — comparing predicted text sentiment against it produces findings the reference approach can never generate.

### 2. Two separate text-preprocessing paths
- **`text_transformer`**: minimal cleaning only (strip HTML/URLs, normalize whitespace). Keeps casing, punctuation, stopwords — what DistilBERT expects.
- **`text_classic`**: full NLP pipeline (lowercase, remove punctuation, lemmatize, remove stopwords) — for TF-IDF/LogReg only.

The reference notebook applies full NLP cleaning to BERT input, discarding the contextual cues the model was pretrained to use.

### 3. Class weighting, not undersampling
The reference notebook undersamples the positive class down to match the negative class, discarding the majority of data. We use `class_weight="balanced"` in LogReg and a `WeightedTrainer` (custom `compute_loss`) in HuggingFace Trainer to handle imbalance without data loss.

### 4. DistilBERT, not bert-tiny
The reference uses `prajjwal1/bert-tiny` (2 transformer layers), which actually underperforms their own LSTM (88.9% vs 93–94%). We use `distilbert-base-uncased` — a proper production-scale model.

### 5. Globally-batched aspect inference
`aspect_sentiment.py` uses a 3-phase design to avoid per-review model calls:
- Phase 1: extract all keyword-matching sentences across all reviews (pure Python)
- Phase 2: single batched inference pass (batch_size=256) over all collected sentences
- Phase 3: aggregate scores back to per-review per-aspect means

## Measured Results

| Model | Label | Accuracy | F1 (weighted) |
|---|---|---|---|
| Reference LSTM | is_recommended (binary) | 94.0% | — |
| Reference bert-tiny | is_recommended (binary) | 88.9% | — |
| TF-IDF + LogReg (ours) | rating (3-class) | 84.7% | 86.6% |
| DistilBERT-base (ours) | rating (3-class) | **90.7%** | **91.2%** |

Our models solve a harder task (3-class vs binary) — direct accuracy comparison is misleading. The `model_comparison.csv` documents the methodological difference per model.

Disagreement analysis (Step 6): 1.76% of reviews have text sentiment that conflicts with `is_recommended` (3,431 / 194,723 analyzed).

## Data

All raw data lives in `data /` (note: trailing space in the directory name — use `data /` in file paths when referencing the original location before the pipeline copies to `data/raw/`).

**Reviews** (five files, ~285k rows total, join on `product_id`):
- Key sentiment signals: `rating`, `is_recommended`, `review_text`, `review_title`
- Reviewer attributes (sparse): `skin_tone`, `skin_type`, `eye_color`, `hair_color`
- `product_name` is in the reviews CSVs (not just the products CSV)

**Products** (`product_info_skincare.csv`, ~1,800 skincare products):
- Key fields: `product_id`, `brand_name`, `price_usd`, `ingredients`, `primary_category`, `secondary_category`, `tertiary_category`
- `ingredients` is stored as a Python list literal string — use `ast.literal_eval()` to parse
- Both files have an unnamed index column as their first column — drop with `df.loc[:, ~df.columns.str.startswith("Unnamed")]`

After the inner join (reviews → products on `product_id`), the merged frame has ~244k rows. Columns with >50% nulls are warned on at load time (`skin_tone`, `skin_type`, `eye_color`, `hair_color`, `helpfulness`).

## What Gets Built Beyond the Reference Notebook

| Feature | Reference | This project |
|---|---|---|
| Sentiment label source | `is_recommended` (binary) | Star rating (3-class) |
| Imbalance handling | Undersampling | Class weighting |
| Transformer size | bert-tiny (2 layers) | DistilBERT-base |
| BERT input text | Stemmed + stopwords removed | Natural language |
| Aspect-level sentiment | None | 6 aspects (hydration, breakouts, scent, packaging, price-value, texture) |
| Skin-type/tone segmentation | Collected, never used | Divergence analysis per product |
| Ingredient analysis | None | Frequency association with complaint patterns |
| Disagreement analysis | Impossible (label = `is_recommended`) | Text sentiment vs. `is_recommended` flags |
| Output format | Static notebook | Streamlit dashboard |

## Aspect Keyword Seeds (Step 7)

Defined once in `src/config.py` as `ASPECT_KEYWORDS`. Downstream scripts and the dashboard both import from there — do not duplicate.

```python
ASPECT_KEYWORDS = {
    "hydration": ["moistur", "hydrat", "dry", "dewy", "plump"],
    "breakouts_irritation": ["breakout", "acne", "irritat", "redness", "rash", "clog", "sensit"],
    "scent": ["smell", "scent", "fragrance", "odor"],
    "packaging": ["bottle", "pump", "packaging", "leak", "container", "tube"],
    "price_value": ["expensive", "worth", "price", "value", "cheap", "overprice"],
    "texture_application": ["texture", "greasy", "absorb", "sticky", "lightweight", "thick"],
}
```

## Training Notes

- `MAX_TRAIN_SAMPLES = 50_000` in `config.py` caps the transformer fine-tune at 50k rows (stratified). Set to `None` for the full dataset on a CUDA GPU.
- Training uses `eval_strategy="epoch"` (not the deprecated `evaluation_strategy`).
- `WeightedTrainer` overrides `compute_loss` to apply `CrossEntropyLoss(weight=class_weights)`.
- Model/tokenizer saved with `save_pretrained` to `outputs/transformer_model/`. Downstream steps load from this path.
- `processing_class=tokenizer` (not `tokenizer=tokenizer`, deprecated in transformers 4.43+).

## Known Limitations (document these in the README)

- Ingredient analysis is frequency/association, not clinical causation — frame it explicitly as a signal worth investigating.
- Aspect extraction uses keyword seeds, so it misses unusually phrased mentions.
- `skin_tone`/`skin_type` have significant nulls — small segments should be flagged as low-confidence.
- Dataset is a fixed March 2023 snapshot; findings don't reflect current products or reviews.
