"""
Extended evaluation of the zero-shot results: uncertainty, significance,
per-writer breakdown, error analysis and robustness.

Owner: Kirill Riabkov (person C, evaluation).

Why this script exists
----------------------
ablation.py reports accuracy and macro-F1 as mean +/- std over 3 seeds.
That std only captures training randomness. With 807 test images, one
error is 0.12 accuracy points, so the test set itself is a large source of
uncertainty: we need intervals and paired tests before claiming that one
preprocessing variant beats another. This script adds that, plus the
analyses listed in docs/remaining-work.md, section 2.

It only runs inference with existing checkpoints (no training), so it takes
a minute or two on a laptop CPU. Whatever checkpoints exist in models/ are
used; results/metrics/checkpoints_used.csv records which ones.

Usage:
    python metrics_analysis.py            # our data, real checkpoints
    python metrics_analysis.py --quick    # smoke-test checkpoints from
                                          # `train.py --quick`; outputs go to
                                          # results/smoke/metrics/

Outputs (results/metrics/):
    predictions.csv            one row per (model, variant, seed, image)
    checkpoints_used.csv       which checkpoints were found
    confidence_intervals.csv   accuracy with 95% bootstrap CI per config
    gap_closed.csv             share of the MNIST -> ours gap each variant closes
    mcnemar.csv                paired significance tests between configs
    per_writer.csv             zero-shot accuracy for every writer
    confusion_pairs.csv        most frequent (true -> predicted) mistakes
    confidence.csv             model confidence on correct vs. wrong answers
    stroke_ratio_sensitivity.csv   accuracy vs. TARGET_STROKE_RATIO
    ceiling.csv                fine-tuning / LOWO error counts (ceiling check)
    augmentation_stroke_width.csv  stroke width: MNIST vs. augmented MNIST
Figures (figures/metrics/): gap_closed, per_writer, stroke_ratio_sensitivity,
confidence.
"""

import argparse

import joblib
import numpy as np
import pandas as pd

import config
import plotting
import preprocessing
from ablation import VARIANTS
from data_utils import load_target
from evaluate import (bootstrap_accuracy_ci, gap_closed, mcnemar_test,
                      rule_of_three_upper)
from model import checkpoint_path, load_model, predict_batch
from preprocessing import preprocess_many

# Short names used in every table: model tag in models/ -> report name.
CNN_TAGS = {"cnn": "mnist", "aug": "aug"}

# Paired comparisons for McNemar's test. Each tuple is
# ((model, variant), (model, variant)); "A" is the first, "B" the second.
COMPARISONS = [
    (("cnn", "minimal"), ("cnn", "thickness")),
    (("cnn", "minimal"), ("cnn", "centering")),
    (("cnn", "minimal"), ("cnn", "full")),
    (("cnn", "thickness"), ("cnn", "centering")),
    (("cnn", "centering"), ("cnn", "full")),
    (("cnn", "full"), ("aug", "full")),
    (("cnn", "minimal"), ("aug", "minimal")),
    (("logreg", "minimal"), ("logreg", "full")),
]

# Values of preprocessing.TARGET_STROKE_RATIO for the sensitivity curve.
# The main results use 0.12, which was fixed before the data was collected.
# This curve is reported as a robustness check only: picking the best value
# from it would be tuning on the test set.
STROKE_RATIOS = [0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18]

# A prediction with at least this softmax probability counts as "confident".
CONFIDENT = 0.9


# --------------------------------------------------------------------------
# Predictions
# --------------------------------------------------------------------------

def available_checkpoints(seeds):
    """Return {(model, seed): path} for every checkpoint that exists."""
    found = {}
    for seed in seeds:
        for name, tag in CNN_TAGS.items():
            path = checkpoint_path(tag, seed)
            if path.exists():
                found[(name, seed)] = path
        path = config.MODELS_DIR / f"logreg_seed{seed}.joblib"
        if path.exists():
            found[("logreg", seed)] = path
    return found


def predict_all(prepared, labels, writers, checkpoints):
    """
    Run every available model on every preprocessing variant.

    Returns a long DataFrame with one row per image and configuration,
    including the softmax confidence of the predicted class.
    """
    frames = []
    for (name, seed), path in sorted(checkpoints.items()):
        if name == "logreg":
            clf = joblib.load(path)
        else:
            net = load_model(CNN_TAGS[name], seed)
        for variant, x in prepared.items():
            if name == "logreg":
                probs = clf.predict_proba(x.reshape(len(x), -1))
                preds = clf.classes_[probs.argmax(axis=1)]
            else:
                preds, probs = predict_batch(net, x)
            frames.append(pd.DataFrame({
                "model": name, "variant": variant, "seed": seed,
                "image": np.arange(len(labels)), "writer": writers,
                "label": labels, "pred": preds,
                "confidence": probs.max(axis=1),
            }))
    out = pd.concat(frames, ignore_index=True)
    out["correct"] = out["label"] == out["pred"]
    return out


def correct_matrix(pred, model, variant):
    """Boolean matrix (n_seeds, n_images) of correct answers for one config."""
    sub = pred[(pred.model == model) & (pred.variant == variant)]
    return sub.pivot(index="seed", columns="image", values="correct").to_numpy()


# --------------------------------------------------------------------------
# Analyses
# --------------------------------------------------------------------------

def confidence_intervals(pred):
    """Accuracy, error count and 95% bootstrap CI for every config."""
    rows = []
    for (model, variant), _ in pred.groupby(["model", "variant"], sort=False):
        mat = correct_matrix(pred, model, variant)
        lo, hi = bootstrap_accuracy_ci(mat, seed=config.SEED)
        rows.append({"model": model, "variant": variant, "seeds": len(mat),
                     "n": mat.shape[1], "accuracy": mat.mean(),
                     "errors_per_seed": (~mat).sum(axis=1).mean(),
                     "ci95_low": lo, "ci95_high": hi})
    return pd.DataFrame(rows)


def gap_table(ci, mnist_runs, pred):
    """Share of the domain gap closed by each variant, per model."""
    rows = []
    for model, sub in ci.groupby("model", sort=False):
        # MNIST accuracy over exactly the seeds evaluated on our data.
        seeds = set(pred.loc[pred.model == model, "seed"])
        runs = mnist_runs[(mnist_runs.model == model)
                          & mnist_runs.seed.isin(seeds)]
        if runs.empty:
            continue
        source = {model: runs.accuracy.mean()}
        minimal = sub.loc[sub.variant == "minimal", "accuracy"].iloc[0]
        for _, r in sub.iterrows():
            rows.append({
                "model": model, "variant": r.variant,
                "mnist_accuracy": source[model],
                "ours_accuracy": r.accuracy,
                "gap_points": 100 * (source[model] - r.accuracy),
                "error_ratio_vs_mnist": (1 - r.accuracy) / (1 - source[model]),
                "gap_closed": gap_closed(r.accuracy, minimal, source[model]),
            })
    return pd.DataFrame(rows)


def mcnemar_table(pred):
    """McNemar's exact test for each comparison and seed."""
    rows = []
    present = set(zip(pred.model, pred.variant))
    for a, b in COMPARISONS:
        if a not in present or b not in present:
            continue
        ma, mb = correct_matrix(pred, *a), correct_matrix(pred, *b)
        seeds = sorted(set(pred[(pred.model == a[0])].seed)
                       & set(pred[(pred.model == b[0])].seed))
        for i, seed in enumerate(seeds):
            res = mcnemar_test(ma[i], mb[i])
            rows.append({"config_a": f"{a[0]}/{a[1]}",
                         "config_b": f"{b[0]}/{b[1]}", "seed": seed,
                         "acc_a": ma[i].mean(), "acc_b": mb[i].mean(), **res})
    out = pd.DataFrame(rows)
    if len(out):
        # Several comparisons per seed: Holm correction keeps the chance of
        # any false "significant" result at 5% within each seed.
        out["p_holm"] = out.groupby("seed").p_value.transform(holm)
        out["significant_5pct"] = out.p_holm < 0.05
    return out


def holm(p):
    """Holm-Bonferroni adjusted p-values for one family of tests."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(running, 1.0)
    return adjusted


def per_writer_table(pred, domain_stats):
    """Zero-shot accuracy for each writer, joined with their stroke width."""
    acc = (pred.groupby(["model", "variant", "writer", "seed"]).correct.mean()
           .groupby(["model", "variant", "writer"]).agg(["mean", "std"])
           .rename(columns={"mean": "accuracy_mean", "std": "accuracy_std"})
           .reset_index())
    n = pred.groupby(["model", "variant", "writer"]).image.nunique()
    acc["n"] = acc.set_index(["model", "variant", "writer"]).index.map(n)
    widths = domain_stats.set_index("writer")["stroke_width_median"]
    acc["stroke_width_median"] = acc.writer.map(widths)
    return acc


def confusion_pairs(pred, top=10):
    """Most frequent (true -> predicted) mistakes, averaged over seeds."""
    wrong = pred[~pred.correct]
    seeds = pred.groupby(["model", "variant"]).seed.nunique()
    counts = (wrong.groupby(["model", "variant", "label", "pred"]).size()
              .rename("count").reset_index())
    counts["count_per_seed"] = counts["count"] / counts.set_index(
        ["model", "variant"]).index.map(seeds)
    per_class = pred.groupby(["model", "variant", "label"]).image.nunique()
    counts["share_of_true_class"] = counts["count_per_seed"] / counts.set_index(
        ["model", "variant", "label"]).index.map(per_class)
    return (counts.sort_values(["model", "variant", "count"],
                               ascending=[True, True, False])
            .groupby(["model", "variant"], sort=False).head(top))


def confidence_table(pred):
    """How confident is each model when it is right and when it is wrong?"""
    rows = []
    for (model, variant), sub in pred.groupby(["model", "variant"], sort=False):
        wrong = sub[~sub.correct]
        n_seeds = sub.seed.nunique()
        rows.append({
            "model": model, "variant": variant,
            "mean_conf_correct": sub[sub.correct].confidence.mean(),
            "mean_conf_wrong": wrong.confidence.mean() if len(wrong) else np.nan,
            "errors_per_seed": len(wrong) / n_seeds,
            "share_of_errors_confident": (
                (wrong.confidence >= CONFIDENT).mean() if len(wrong) else np.nan),
        })
    return pd.DataFrame(rows)


def stroke_ratio_sensitivity(raw, labels, checkpoints):
    """CNN accuracy (full pipeline) for several TARGET_STROKE_RATIO values."""
    seeds = sorted(s for (m, s) in checkpoints if m == "cnn")
    nets = {s: load_model("mnist", s) for s in seeds}
    original = preprocessing.TARGET_STROKE_RATIO
    rows = []
    try:
        for ratio in STROKE_RATIOS:
            preprocessing.TARGET_STROKE_RATIO = ratio
            x = preprocess_many(raw, VARIANTS["full"])
            for s, net in nets.items():
                preds, _ = predict_batch(net, x)
                rows.append({"stroke_ratio": ratio, "seed": s,
                             "accuracy": (preds == labels).mean(),
                             "used_in_main_results": ratio == original})
    finally:
        preprocessing.TARGET_STROKE_RATIO = original   # never leak the change
    return pd.DataFrame(rows)


def ceiling_table():
    """Error counts of the fine-tuning and LOWO experiments (from ablation.py)."""
    rows = []
    runs_path = config.RESULTS_DIR / "ablation_runs.csv"
    if runs_path.exists():
        runs = pd.read_csv(runs_path)
        for _, r in runs[runs.experiment == "finetune"].iterrows():
            errors = int(round(r.n * (1 - r.accuracy)))
            rows.append({"experiment": "finetune", "group": r.model,
                         "stage": r.stage, "seed": r.seed, "n": r.n,
                         "errors": errors,
                         "error_upper95": rule_of_three_upper(errors, r.n)})
    lowo_path = config.RESULTS_DIR / "lowo_runs.csv"
    if lowo_path.exists():
        for _, r in pd.read_csv(lowo_path).iterrows():
            errors = int(round(r.n * (1 - r.accuracy)))
            rows.append({"experiment": "lowo", "group": r.writer,
                         "stage": r.stage, "seed": r.seed, "n": r.n,
                         "errors": errors,
                         "error_upper95": rule_of_three_upper(errors, r.n)})
    return pd.DataFrame(rows)


def augmentation_stroke_width(n=2000):
    """
    Median stroke width of MNIST test digits before and after augment().

    Tests the hypothesis that augmentation hurts because it makes strokes
    thicker, while our strokes are thinner than MNIST. Returns None if MNIST
    is not available offline.
    """
    try:
        from analysis import stroke_width
        from data_utils import load_mnist
        images, _ = load_mnist(train=False)
    except Exception as exc:                     # no MNIST, no network
        print(f"  skipped augmentation stroke width: {exc}")
        return None
    rng = np.random.default_rng(config.SEED)
    idx = rng.choice(len(images), size=n, replace=False)
    plain = [stroke_width(images[i] / 255.0) for i in idx]
    augmented = [stroke_width(preprocessing.augment(images[i], rng) / 255.0)
                 for i in idx]
    return pd.DataFrame({
        "source": ["MNIST", "MNIST augmented"],
        "stroke_width_median": [np.median(plain), np.median(augmented)],
        "stroke_width_mean": [np.mean(plain), np.mean(augmented)],
        "n": [n, n],
    })


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def make_figures(gap, writers_tab, sens, conf, fig_dir):
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)

    # Share of the domain gap closed by each preprocessing variant.
    variants = [v for v in VARIANTS if v != "minimal"]
    models = [m for m in ("logreg", "cnn", "aug") if m in set(gap.model)]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    width = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        sub = gap[gap.model == m].set_index("variant").reindex(variants)
        ax.bar(np.arange(len(variants)) + i * width, 100 * sub.gap_closed,
               width, label=m)
    ax.axhline(100, color="black", lw=0.8, ls="--")
    ax.set_xticks(np.arange(len(variants)) + width * (len(models) - 1) / 2,
                  variants)
    ax.set_ylabel("domain gap closed, %")
    ax.set_title("Share of the MNIST -> in-air gap closed by preprocessing")
    ax.legend(frameon=False)
    plotting.save(fig, fig_dir / "gap_closed")

    # Per-writer accuracy against stroke width, CNN.
    sub = writers_tab[writers_tab.model == "cnn"]
    if len(sub):
        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        for variant, marker in (("minimal", "o"), ("full", "s")):
            s = sub[sub.variant == variant].sort_values("stroke_width_median")
            # Markers only: four writers are separate points, not a curve.
            ax.plot(s.stroke_width_median, 100 * s.accuracy_mean, marker=marker,
                    ls="none", markersize=7, label=f"CNN, {variant}")
            if variant == "minimal":
                for _, r in s.iterrows():
                    ax.annotate(r.writer, (r.stroke_width_median,
                                           100 * r.accuracy_mean),
                                textcoords="offset points", xytext=(4, -10),
                                fontsize=8)
        ax.set_xlabel("median stroke width at 28x28, px")
        ax.set_ylabel("accuracy, %")
        ax.set_title("Zero-shot accuracy per writer")
        ax.legend(frameon=False)
        plotting.save(fig, fig_dir / "per_writer")

    # Sensitivity to TARGET_STROKE_RATIO.
    if len(sens):
        agg = sens.groupby("stroke_ratio").accuracy.agg(["mean", "std"])
        fig, ax = plt.subplots(figsize=(5.6, 3.4))
        ax.errorbar(agg.index, 100 * agg["mean"], yerr=100 * agg["std"].fillna(0),
                    marker="o", capsize=3)
        ax.axvline(preprocessing.TARGET_STROKE_RATIO, color="black", lw=0.8,
                   ls="--", label="value used (fixed in advance)")
        ax.set_xlabel("TARGET_STROKE_RATIO")
        ax.set_ylabel("CNN accuracy, full pipeline, %")
        ax.set_title("Robustness to the stroke-width hyperparameter")
        ax.legend(frameon=False)
        plotting.save(fig, fig_dir / "stroke_ratio_sensitivity")

    # Confidence on correct vs. wrong predictions.
    c = conf[conf.model.isin(["cnn", "aug"])]
    if len(c):
        labels = [f"{m}/{v}" for m, v in zip(c.model, c.variant)]
        x = np.arange(len(c))
        fig, ax = plt.subplots(figsize=(7.0, 3.4))
        ax.bar(x - 0.2, c.mean_conf_correct, 0.4, label="correct")
        ax.bar(x + 0.2, c.mean_conf_wrong, 0.4, label="wrong")
        ax.set_xticks(x, labels, rotation=30, ha="right")
        ax.set_ylabel("mean softmax confidence")
        ax.set_ylim(0, 1)
        ax.set_title("Model confidence on correct vs. wrong answers")
        ax.legend(frameon=False)
        plotting.save(fig, fig_dir / "confidence")


# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--quick", action="store_true",
                        help="use smoke-test checkpoints; outputs to */smoke/")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=list(config.SEEDS))
    args = parser.parse_args()
    if args.quick:
        config.redirect_outputs("smoke")

    out_dir = config.RESULTS_DIR / "metrics"
    fig_dir = config.FIGURES_DIR / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw, labels, writers = load_target(synthetic=False)
    print(f"{len(raw)} images from {len(set(writers))} writers")

    checkpoints = available_checkpoints(args.seeds)
    if not checkpoints:
        raise SystemExit("No checkpoints in models/. Run train.py first.")
    pd.DataFrame([{"model": m, "seed": s, "file": p.name}
                  for (m, s), p in sorted(checkpoints.items())]).to_csv(
        out_dir / "checkpoints_used.csv", index=False)
    print("checkpoints:", sorted(checkpoints))

    prepared = {name: preprocess_many(raw, steps)
                for name, steps in VARIANTS.items()}
    pred = predict_all(prepared, labels, writers, checkpoints)
    pred.to_csv(out_dir / "predictions.csv", index=False)

    ci = confidence_intervals(pred)
    ci.to_csv(out_dir / "confidence_intervals.csv", index=False)

    mnist_runs = pd.read_csv(config.RESULTS_DIR / "mnist_test_runs.csv")
    gap = gap_table(ci, mnist_runs, pred)
    gap.to_csv(out_dir / "gap_closed.csv", index=False)

    mcn = mcnemar_table(pred)
    mcn.to_csv(out_dir / "mcnemar.csv", index=False)

    domain = pd.read_csv(config.RESULTS_DIR / "domain_stats.csv")
    wt = per_writer_table(pred, domain)
    wt.to_csv(out_dir / "per_writer.csv", index=False)

    confusion_pairs(pred).to_csv(out_dir / "confusion_pairs.csv", index=False)
    conf = confidence_table(pred)
    conf.to_csv(out_dir / "confidence.csv", index=False)

    print("stroke-ratio sensitivity...")
    sens = stroke_ratio_sensitivity(raw, labels, checkpoints)
    sens.to_csv(out_dir / "stroke_ratio_sensitivity.csv", index=False)

    ceiling_table().to_csv(out_dir / "ceiling.csv", index=False)

    aug_width = augmentation_stroke_width()
    if aug_width is not None:
        aug_width.to_csv(out_dir / "augmentation_stroke_width.csv", index=False)

    make_figures(gap, wt, sens, conf, fig_dir)

    with pd.option_context("display.width", 140, "display.precision", 4):
        print(ci.to_string(index=False))
        print(gap.to_string(index=False))
    print(f"\nwritten to {out_dir} and {fig_dir}")


if __name__ == "__main__":
    main()
