"""
Step 4 — TF-IDF + Logistic Regression Baseline

Fast, interpretable comparison point. Uses class_weight="balanced" to handle
class imbalance — equivalent intent to the reference notebook's undersampling
but without discarding any data.

Inspect clf.coef_ after training for top-weighted words per class.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).parent))
import config
from utils import get_logger, save_metrics, set_seed, timer

logger = get_logger(__name__)


def log_top_features(clf: LogisticRegression, vectorizer: TfidfVectorizer, n: int = 15) -> None:
    feature_names = vectorizer.get_feature_names_out()
    for i, label_name in enumerate(config.LABEL_NAMES):
        top_idx = np.argsort(clf.coef_[i])[-n:][::-1]
        top_words = ", ".join(feature_names[top_idx])
        logger.info(f"  Top words [{label_name}]: {top_words}")


@timer
def train_baseline() -> None:
    set_seed(config.RANDOM_SEED)

    src = config.DATA_CLEAN / "labeled_dataset.parquet"
    if not src.exists():
        raise FileNotFoundError(f"Run build_labels.py first — {src} not found")
    df = pd.read_parquet(src, engine=config.PARQUET_ENGINE).dropna(subset=["text_classic", "sentiment_label"])
    df["sentiment_label"] = df["sentiment_label"].astype(int)
    logger.info(f"Loaded: {len(df):,} rows")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text_classic"],
        df["sentiment_label"],
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=df["sentiment_label"],
    )
    logger.info(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # Hyperparameter search over TF-IDF max_features and LogReg C.
    # Scored on f1_weighted (not accuracy) — appropriate for class-imbalanced data.
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=config.TFIDF_NGRAM_RANGE)),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000,
                                   random_state=config.RANDOM_SEED)),
    ])
    param_grid = {
        "tfidf__max_features": [10_000, 20_000],
        "clf__C": [0.1, 1.0, 10.0],
    }
    gs = GridSearchCV(pipe, param_grid, cv=3, scoring="f1_weighted", n_jobs=-1, verbose=1)
    gs.fit(X_train.fillna(""), y_train)
    best_params = gs.best_params_
    best_cv_f1 = gs.best_score_
    logger.info(f"Grid search best params: {best_params}  |  CV F1: {best_cv_f1:.4f}")

    # Refit standalone vectorizer + clf with best params for feature inspection and pickling
    best_max_features = best_params["tfidf__max_features"]
    best_C = best_params["clf__C"]
    vectorizer = TfidfVectorizer(
        max_features=best_max_features,
        ngram_range=config.TFIDF_NGRAM_RANGE,
    )
    X_train_vec = vectorizer.fit_transform(X_train.fillna(""))
    X_test_vec = vectorizer.transform(X_test.fillna(""))
    logger.info(f"Vocabulary size: {len(vectorizer.vocabulary_):,}")

    clf = LogisticRegression(
        C=best_C,
        class_weight="balanced",
        max_iter=1000,
        random_state=config.RANDOM_SEED,
    )
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1_per_class = f1_score(y_test, y_pred, average=None)

    logger.info(f"\n{classification_report(y_test, y_pred, target_names=config.LABEL_NAMES)}")
    logger.info(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")
    logger.info(f"Accuracy: {acc:.4f}  |  F1 (weighted): {f1:.4f}")
    for name, val in zip(config.LABEL_NAMES, f1_per_class):
        logger.info(f"  F1 [{name}]: {val:.4f}")

    log_top_features(clf, vectorizer)

    # Evaluate on the transformer's exact test set for a valid head-to-head comparison.
    # Both models see the same 10k rows; the transformer's 90.7% is then comparable to
    # the baseline's number on this shared set.
    from utils import get_transformer_test_df
    trans_test = get_transformer_test_df(df)
    X_shared = vectorizer.transform(trans_test["text_classic"].fillna(""))
    y_shared_true = trans_test["sentiment_label"].values
    y_shared_pred = clf.predict(X_shared)
    shared_acc = accuracy_score(y_shared_true, y_shared_pred)
    shared_f1 = f1_score(y_shared_true, y_shared_pred, average="weighted")
    logger.info(
        f"Shared test set (n={len(trans_test):,}, same as transformer): "
        f"accuracy={shared_acc:.4f}  F1={shared_f1:.4f}"
    )

    metrics = {
        "model": "TF-IDF + LogisticRegression",
        "label_source": "rating_3class",
        "accuracy": round(acc, 4),
        "f1_weighted": round(f1, 4),
        "precision_weighted": round(precision, 4),
        "recall_weighted": round(recall, 4),
        "f1_per_class": {n: round(float(v), 4) for n, v in zip(config.LABEL_NAMES, f1_per_class)},
        "train_size": len(X_train),
        "test_size": len(X_test),
        # Shared-test metrics allow direct comparison with the transformer
        "shared_test_size": len(trans_test),
        "shared_test_accuracy": round(shared_acc, 4),
        "shared_test_f1_weighted": round(shared_f1, 4),
        "grid_search_best_params": best_params,
        "grid_search_best_cv_f1": round(best_cv_f1, 4),
        "notes": (
            "class_weight=balanced, no undersampling, negations preserved in stopword removal; "
            "C and max_features from 3-fold grid search. "
            "WARNING: shared_test_* metrics are inflated — the TF-IDF trains on ~80% of the full "
            "284k corpus, so ~80% of the 10k shared rows were already in its training set. "
            "The primary accuracy/f1_weighted from the TF-IDF's own 57k held-out set are the honest numbers."
        ),
    }
    save_metrics(metrics, "baseline")

    config.OUTPUTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, config.OUTPUTS / "baseline_model.pkl")
    joblib.dump(vectorizer, config.OUTPUTS / "tfidf_vectorizer.pkl")
    logger.info(f"Model saved → {config.OUTPUTS / 'baseline_model.pkl'}")


if __name__ == "__main__":
    train_baseline()
