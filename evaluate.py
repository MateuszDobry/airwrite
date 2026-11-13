"""
Metrics (accuracy, macro-F1, per-class F1), averaging over seeds,
confusion matrices, and the statistics used by metrics_analysis.py.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import plotting
from model import predict_batch


def compute_metrics(y_true, y_pred) -> dict:
    """Accuracy, macro-F1 and F1 for each digit."""
    per_class = f1_score(y_true, y_pred, labels=range(10), average=None,
                         zero_division=0)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=range(10),
                             average="macro", zero_division=0),
        "n": len(y_true),
    }
    for digit, score in enumerate(per_class):
        metrics[f"f1_{digit}"] = score
    return metrics


def evaluate(model, images, labels) -> dict:
    """Metrics for a CNN on preprocessed images (N, 28, 28)."""
    preds, _ = predict_batch(model, images)
    return compute_metrics(labels, preds)


def summarise(runs: pd.DataFrame, group_cols, metric_cols=None) -> pd.DataFrame:
    """
    Per-seed rows -> mean and std per configuration.

    group_cols are the columns that define a configuration, e.g. ["variant"].
    Output has <metric>_mean, <metric>_std and the number of seeds.
    """
    metric_cols = metric_cols or ["accuracy", "macro_f1"]
    grouped = runs.groupby(group_cols, sort=False)
    out = grouped[metric_cols].agg(["mean", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out["seeds"] = grouped.size()
    return out.reset_index()


def save_confusion_matrix(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=range(10))
    plotting.confusion(cm, title, path)
    return cm


# Statistics for metrics_analysis.py.
# The std over seeds only shows training noise. The test set is small
# (807 images, 1 error = 0.12 pp), so we also need confidence intervals
# and paired tests.

def bootstrap_accuracy_ci(correct, n_boot=2000, level=0.95, seed=0):
    """
    Bootstrap (percentile) CI for accuracy.

    correct: bool array (n_images,) or (n_seeds, n_images). With several
    seeds the same images are resampled for all of them, and the interval
    is for the accuracy averaged over seeds.

    Returns (low, high).
    """
    correct = np.atleast_2d(np.asarray(correct, dtype=float))
    n = correct.shape[1]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    # accuracy of each resample, mean over seeds
    stats = correct[:, idx].mean(axis=(0, 2))
    alpha = (1 - level) / 2
    return float(np.quantile(stats, alpha)), float(np.quantile(stats, 1 - alpha))


def mcnemar_test(correct_a, correct_b):
    """
    Exact McNemar test for two models on the same images.

    b = A right, B wrong; c = A wrong, B right. If both models are equally
    good, b ~ Binomial(b + c, 0.5), so we do a two-sided binomial test.
    """
    from scipy.stats import binomtest

    a = np.asarray(correct_a, dtype=bool)
    bb = np.asarray(correct_b, dtype=bool)
    b = int(np.sum(a & ~bb))
    c = int(np.sum(~a & bb))
    p = 1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue
    return {"a_right_b_wrong": b, "a_wrong_b_right": c, "p_value": float(p)}


def gap_closed(acc_variant, acc_minimal, acc_source):
    """
    How much of the domain gap a variant closes.

    0 = same as minimal preprocessing, 1 = as good as on MNIST test,
    > 1 = better than MNIST.
    """
    gap = acc_source - acc_minimal
    return float("nan") if gap == 0 else (acc_variant - acc_minimal) / gap


def rule_of_three_upper(errors, n):
    """
    Approx. 95% upper bound on the error rate.

    0 errors -> rule of three (3/n), otherwise Wilson upper bound.
    """
    if n == 0:
        return float("nan")
    if errors == 0:
        return 3.0 / n
    z = 1.96
    p = errors / n
    centre = p + z * z / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((centre + margin) / (1 + z * z / n))
