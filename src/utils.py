import json
import logging
import random
import time
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import numpy as np

from config import MAX_TRAIN_SAMPLES, METRICS, OUTPUTS, RANDOM_SEED, TEST_SIZE


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(OUTPUTS / "pipeline.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def timer(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__name__} completed in {elapsed:.1f}s")
        return result
    return wrapper


def save_metrics(metrics: dict[str, Any], name: str) -> None:
    METRICS.mkdir(parents=True, exist_ok=True)
    metrics["_timestamp"] = datetime.now(timezone.utc).isoformat()
    path = METRICS / f"{name}.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    get_logger("utils").info(f"Metrics saved → {path}")


def load_metrics(name: str) -> dict[str, Any]:
    path = METRICS / f"{name}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def get_device():
    """Return the best available torch device (CUDA > MPS > CPU)."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def batch_predict(
    texts: list[str],
    model,
    tokenizer,
    device,
    batch_size: int = 64,
    max_length: int = 128,
    return_proba: bool = False,
) -> list:
    """Run batched inference over a flat list of texts.

    Args:
        return_proba: If True, returns softmax P(positive class=2) as floats in [0, 1].
                      If False (default), returns argmax class indices.
    """
    import torch
    from tqdm import tqdm

    if not texts:
        return []
    model.eval()
    results: list = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting", leave=False):
        batch = texts[i: i + batch_size]
        inputs = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        if return_proba:
            probs = torch.softmax(logits, dim=1)
            results.extend(probs[:, 2].cpu().tolist())  # P(class=positive)
        else:
            results.extend(torch.argmax(logits, dim=1).cpu().tolist())
    return results


def get_transformer_test_df(df):
    """Reproduce the exact test split from train_transformer.py (deterministic, seed=42).

    Replicating the same sampling + split logic with the same seed guarantees we get
    the identical 10,000-row test set that was used for transformer evaluation, without
    re-running the expensive fine-tuning step.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    work = df.dropna(subset=["text_transformer", "sentiment_label"]).copy()
    work["sentiment_label"] = work["sentiment_label"].astype(int)

    if MAX_TRAIN_SAMPLES and len(work) > MAX_TRAIN_SAMPLES:
        work, _ = train_test_split(
            work,
            train_size=MAX_TRAIN_SAMPLES,
            random_state=RANDOM_SEED,
            stratify=work["sentiment_label"],
        )
        work = work.reset_index(drop=True)

    _, test_df = train_test_split(
        work,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=work["sentiment_label"],
    )
    return test_df.reset_index(drop=True)
