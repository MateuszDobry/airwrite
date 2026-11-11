"""
Metrics, seed aggregation and confusion matrices.

Owner: Kirill Riabkov (person C).

Reported metrics: accuracy, macro-F1 and per-class F1. Every configuration
is run with 3 seeds and reported as mean +/- std, so that differences between
variants can be told apart from run-to-run noise.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import plotting
from model import predict_batch


def compute_metrics(y_true, y_pred) -> dict:
    """Accuracy, macro-F1 and the ten per-class F1 scores."""
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
    """
    Return a dict of metrics for one model/dataset pair.

    Args:
        model: a trained DigitCNN.
        images: float32 (N, 28, 28), already preprocessed.
        labels: int (N,).
    """
    preds, _ = predict_batch(model, images)
    return compute_metrics(labels, preds)


def summarise(runs: pd.DataFrame, group_cols, metric_cols=None) -> pd.DataFrame:
    """
    Collapse per-seed rows into mean and std per configuration.

    Args:
        runs: one row per (configuration, seed).
        group_cols: columns identifying a configuration, e.g. ["variant"].

    Returns:
        One row per configuration with <metric>_mean and <metric>_std columns
        and the number of seeds that went into it.
    """
    metric_cols = metric_cols or ["accuracy", "macro_f1"]
    grouped = runs.groupby(group_cols, sort=False)
    out = grouped[metric_cols].agg(["mean", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out["seeds"] = grouped.size()
    return out.reset_index()


def save_confusion_matrix(y_true, y_pred, title, path):
    """Plot a row-normalised confusion matrix and save it to path."""
    cm = confusion_matrix(y_true, y_pred, labels=range(10))
    plotting.confusion(cm, title, path)
    return cm


# --------------------------------------------------------------------------
# Uncertainty and significance (used by metrics_analysis.py)
# --------------------------------------------------------------------------
#
# The std over seeds above only reflects training randomness. Our test set
# is small (807 images, one error = 0.12 points), so the choice of test
# images is an equally important source of noise. The functions below
# quantify it.

def bootstrap_accuracy_ci(correct, n_boot=2000, level=0.95, seed=0):
    """
    Percentile bootstrap confidence interval for accuracy.

    Args:
        correct: bool array (n_images,) for one model, or (n_seeds, n_images)
                 for several seeds evaluated on the SAME images. Images are
                 resampled together across seeds, so the interval covers the
                 test-set sampling noise of the seed-averaged accuracy.
        n_boot: number of bootstrap resamples.
        level: confidence level.
        seed: random seed, for reproducibility.

    Returns:
        (low, high) bounds of the interval.
    """
    correct = np.atleast_2d(np.asarray(correct, dtype=float))
    n = correct.shape[1]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    # Accuracy of each resample, averaged over seeds.
    stats = correct[:, idx].mean(axis=(0, 2))
    alpha = (1 - level) / 2
    return float(np.quantile(stats, alpha)), float(np.quantile(stats, 1 - alpha))


def mcnemar_test(correct_a, correct_b):
    """
    Exact McNemar test for two classifiers evaluated on the same images.

    Only the discordant images matter: b = A right and B wrong,
    c = A wrong and B right. Under the null hypothesis (equal accuracy)
    each discordant image is a fair coin flip, so the p-value is a two-sided
    binomial test of b against n = b + c.

    Returns:
        dict with b, c and the two-sided p-value.
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
    Share of the domain gap closed by a mitigation.

    0 = no better than minimal preprocessing, 1 = back to the accuracy on the
    source domain (MNIST test), above 1 = better than on MNIST.
    """
    gap = acc_source - acc_minimal
    return float("nan") if gap == 0 else (acc_variant - acc_minimal) / gap


def rule_of_three_upper(errors, n):
    """
    Approximate 95% upper bound on the error rate.

    With zero errors in n images the usual interval collapses to 0; the
    "rule of three" gives 3/n instead. With errors > 0 a Wilson upper bound
    is used.
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
