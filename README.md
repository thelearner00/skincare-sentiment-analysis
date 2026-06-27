# Skincare Review Sentiment Analysis

**Fixing the reference Kaggle notebook's methodological gaps — then going further with aspect-level personalisation, ingredient risk signals, and a live Streamlit dashboard.**

---

## Results at a Glance

| Model | Task | Accuracy | Weighted F1 | Test n |
|---|---|---|---|---|
| Majority-class baseline | rating (3-class) | 82.85% | — | 56,969 |
| TF-IDF + LogReg *(ours)* | rating (3-class) | **87.29%** | **88.52%** | 56,969 |
| DistilBERT-base *(ours)* | rating (3-class) | **91.17%** | **91.46%** | 10,000 |
| Reference LSTM | is_recommended (binary) ⚠️ | 94.0% | — | — |
| Reference bert-tiny | is_recommended (binary) ⚠️ | 88.9% | — | — |

> ⚠️ **The reference rows are not directly comparable.** They solve an easier binary task (random baseline = 50%) while ours solve a harder 3-class problem (majority-class ceiling = 82.85%). Our DistilBERT surpasses the reference bert-tiny's binary accuracy on the harder task.

---

## What This Project Fixes

The [reference Kaggle notebook](https://www.kaggle.com/code/eward96/skincare-products-eda-sentiment-analysis) contains five significant methodological errors this project addresses head-on:

| Issue | Reference | This project |
|---|---|---|
| **Sentiment label** | `is_recommended` (behavioural checkbox) | Star rating → 3 classes |
| **Imbalance handling** | Undersampling — discards 80% of data | Class-weighted loss, no data lost |
| **Transformer size** | bert-tiny (4.4M params, 2 layers) | DistilBERT-base (66M params, 6 layers) |
| **BERT input text** | Stemmed + stopwords removed | Natural language (casing + punctuation preserved) |
| **Negation handling** | "not recommend" → "recommend" | Negations explicitly preserved in TF-IDF pipeline |

The label choice matters most: by using `is_recommended` as the label, the reference model learns to predict a separate explicit signal rather than text-expressed sentiment — making disagreement analysis impossible. Here, `is_recommended` is kept as an independent column and compared against predicted sentiment in Step 6.

---

## Beyond the Reference: Personalisation Layer

Five analyses the reference notebook never builds:

1. **Aspect-level sentiment** — 6 aspects (hydration, breakouts/irritation, scent, packaging, price-value, texture) scored per-review using the fine-tuned DistilBERT as a sentence-level scorer. Scores are softmax P(positive) in [0, 1], not ordinal class averages.

2. **Skin-type divergence** — Products where oily-skin or dry-skin reviewers report meaningfully different outcomes. Flagged when any aspect delta > 0.15 P(positive) with per-aspect mention counts for confidence gating.

3. **Ingredient risk signal** — Ingredients that appear disproportionately in high-complaint products (mean P(positive breakout) < 0.35) vs low-complaint products, tested with Fisher's exact test. Only **parfum** passes both the minimum count (≥5) and significance (p = 0.0021) thresholds — all other flagged ingredients have p > 0.20.

4. **Disagreement analysis** — Reviews where the model's predicted sentiment conflicts with the reviewer's `is_recommended` flag. Found in **2.17% of the held-out test set** (174 / 8,012 reviewers). This analysis is only possible because `is_recommended` is NOT the training label.

5. **Streamlit dashboard** — Five interactive sections with brand/product filters, download buttons, and a per-class F1 breakdown that exposes where each model actually struggles (the neutral class).

---

## Pipeline Architecture

```
data/raw/           ← 5 review CSVs + product CSV (~285k reviews, ~1,800 products)
    │
    ▼ Step 1: load_data.py       merge on product_id → 284,844 rows
    │
    ▼ Step 2: clean.py           two text paths:
    │                              text_transformer  (natural language — for DistilBERT)
    │                              text_classic      (lemmatized, negations preserved — for TF-IDF)
    │
    ▼ Step 3: build_labels.py    rating → 3-class label  (neg 9.7% / neu 7.4% / pos 82.9%)
    │
    ├─▶ Step 4: train_baseline.py     TF-IDF + LogReg, 3-fold grid search over C & max_features
    │
    ├─▶ Step 5: train_transformer.py  DistilBERT fine-tune, WeightedTrainer, 35k samples, ~25 min MPS
    │
    ▼ Step 6: disagreement.py    text sentiment vs is_recommended (test set only)
    │
    ▼ Step 7: aspect_sentiment.py  per-review per-aspect P(positive) via batched sentence inference
    │
    ▼ Step 8: segment_analysis.py  skin-type / skin-tone divergence per product
    │
    ▼ Step 9: ingredient_risk.py   Fisher's exact test, ingredient ↔ complaint association
    │
    ▼ Step 10: aggregate.py       brand/product summaries + model comparison CSV
    │
    ▼ streamlit run app/dashboard.py
```

---

## Key Findings

- **Parfum is the only statistically significant ingredient signal** (risk ratio 5.82, Fisher's exact p = 0.0021). All other flagged ingredients have p > 0.20 — the associations are likely noise. Treat this as a signal to investigate, not a causal claim.
- **2.17% of reviewers say one thing and click another** — their text predicts negative or neutral sentiment but they still tick `is_recommended`. These edge cases are only detectable because `is_recommended` is not the training label.
- **Neutral class is the hardest**: DistilBERT achieves F1 = 53.4% on neutral vs 96.6% on positive. The 11x class imbalance (82.9% positive) means the neutral class is the true test of the model.
- **12.6% truncation at seq_len=128** — median review is only 63 tokens, so truncation has minimal impact on this dataset. P90 is 140 tokens.

---

## Setup & Quickstart

**Requires Python 3.9+**

```bash
# 1. Clone
git clone https://github.com/<your-username>/skincare-sentiment-analysis.git
cd skincare-sentiment-analysis

# 2. Install dependencies
pip3 install -r requirements.txt
python3 -m nltk.downloader stopwords wordnet

# 3. Add your data files to data/raw/
#    Required files:
#      reviews_0-250_masked.csv, reviews_250-500_masked.csv,
#      reviews_500-750_masked.csv, reviews_750-1250_masked.csv,
#      reviews_1250-end_masked.csv, product_info_skincare.csv

# 4. Run the pipeline (Steps 1–10)
python3 src/load_data.py
python3 src/clean.py
python3 src/build_labels.py
python3 src/train_baseline.py          # ~2 min (includes 3-fold grid search)
python3 src/train_transformer.py       # ~25 min on Apple Silicon MPS
python3 src/disagreement.py
python3 src/aspect_sentiment.py
python3 src/segment_analysis.py
python3 src/ingredient_risk.py
python3 src/aggregate.py

# 5. Launch the dashboard
streamlit run app/dashboard.py
```

> **Note:** The trained DistilBERT weights (255 MB per checkpoint) are not included in this repo. Run `train_transformer.py` to generate them locally, or explore results directly from the pre-computed output CSVs in `outputs/`.

---

## Project Structure

```
skincare-sentiment-analysis/
├── src/
│   ├── config.py                  # Single source of truth — all thresholds and paths
│   ├── utils.py                   # Shared helpers: batch_predict, timer, save_metrics
│   ├── load_data.py               # Step 1 — merge reviews + products
│   ├── clean.py                   # Step 2 — two preprocessing paths
│   ├── build_labels.py            # Step 3 — rating → 3-class labels
│   ├── train_baseline.py          # Step 4 — TF-IDF + LogReg with grid search
│   ├── train_transformer.py       # Step 5 — DistilBERT fine-tune
│   ├── disagreement.py            # Step 6 — text vs is_recommended
│   ├── aspect_sentiment.py        # Step 7 — globally-batched, 3-phase design
│   ├── segment_analysis.py        # Step 8 — skin-type / skin-tone divergence
│   ├── ingredient_risk.py         # Step 9 — Fisher's exact test
│   ├── aggregate.py               # Step 10 — summaries + comparison table
│   └── token_length_analysis.py   # Standalone diagnostic
├── app/
│   └── dashboard.py               # Streamlit: 5 sections, brand filters, download buttons
├── outputs/
│   ├── metrics/                   # JSON metrics files (baseline, transformer, disagreement, etc.)
│   └── *.csv                      # Result tables (model comparison, brand/product summaries, etc.)
├── requirements.txt
├── CLAUDE.md                      # Development guide (architecture decisions, commands)
├── LICENSE
└── README.md
```

---

## Limitations

- **Ingredient analysis is associative, not causal.** Frequency co-occurrence in complaint clusters does not establish that any ingredient causes breakouts or irritation.
- **Aspect extraction uses keyword seeds**, so it misses unusual phrasings. Coverage varies by aspect — scent and packaging have fewer keyword matches than hydration or breakouts.
- **`skin_type` and `skin_tone` are sparse** — ~20–40% null in the raw data. Small segments are flagged `low_confidence` when the per-aspect mention count falls below 30.
- **Dataset is a fixed 2023 Sephora snapshot.** Findings do not reflect current products, reformulations, or reviews posted after that date.
- **DistilBERT was fine-tuned on 35,000 samples** (14% of available data) for compute feasibility on Apple Silicon MPS. Set `MAX_TRAIN_SAMPLES = None` in `src/config.py` for a full-data run on a CUDA GPU.

---

## Tech Stack

`Python 3.9` · `PyTorch` · `HuggingFace Transformers` · `scikit-learn` · `NLTK` · `pandas` · `Streamlit` · `Plotly` · `scipy`

---

*Dataset: Sephora skincare reviews (~285k reviews, ~1,800 products). Reference notebook: [Kaggle — skincare-products-eda-sentiment-analysis](https://www.kaggle.com/code/eward96/skincare-products-eda-sentiment-analysis).*
