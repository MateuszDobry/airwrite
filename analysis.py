"""
Figures, domain statistics and the main numbers (headline.json).

Usage:
    python analysis.py                 # after train.py and ablation.py
    python analysis.py --synthetic     # for the synthetic data

Model results are read from the CSVs written by train.py and ablation.py,
nothing is recomputed here.

Figures (figures/):
    learning_curves          CNN and augmented CNN on MNIST
    samples_by_writer        the same digits from MNIST and from every writer
    preprocessing_steps      what each preprocessing step does
    augmentation_examples    what training-time augmentation produces
    domain_stroke_width      stroke-width distribution, MNIST vs. ours
    domain_fill_ratio        ink coverage distribution, MNIST vs. ours
    ablation_cnn             accuracy per mitigation, with error bars
    ablation_models          logistic regression vs. CNN vs. augmented CNN
    lowo                     cross-writer: before vs. after fine-tuning

Tables (results/):
    domain_stats.csv, headline.json
"""

import argparse
import json

import cv2
import numpy as np
import pandas as pd

import config
import plotting
from data_utils import load_mnist, load_target
from preprocessing import (
    ALL_STEPS, MINIMAL_STEPS, augment, preprocess, preprocess_many,
    skeletonize,
)

MNIST_SAMPLE = 2000      # MNIST images used for the distribution plots


def to_display(x):
    """Undo the normalization so the image can be displayed."""
    return np.clip(x * config.MNIST_STD + config.MNIST_MEAN, 0, 1)


# Domain statistics (MNIST vs. ours)

def stroke_width(img01):
    """
    Mean stroke width (px) of a 28x28 image in [0, 1].

    Distance transform on the skeleton = half of the stroke width.
    """
    # computed on a 4x bigger image - at 28x28 there are only a few possible
    # values and the histogram looks bad. Divided by 4 at the end.
    big = cv2.resize(img01.astype(np.float32), None, fx=4, fy=4,
                     interpolation=cv2.INTER_LINEAR)
    binary = (big > 0.5).astype(np.uint8) * 255
    if not binary.any():
        return 0.0
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    skel = skeletonize(binary) > 0
    return float(2 * dist[skel].mean() / 4) if skel.any() else 0.0


def fill_ratio(img01):
    """Fraction of pixels that are ink."""
    return float((img01 > 0.5).mean())


def domain_statistics(mnist_imgs, ours, writers):
    """Stroke width and fill ratio for MNIST and each writer."""
    rows = []
    for img in mnist_imgs:
        rows.append({"domain": "MNIST", "writer": "MNIST",
                     "stroke_width": stroke_width(img),
                     "fill_ratio": fill_ratio(img)})
    for img, w in zip(ours, writers):
        rows.append({"domain": "ours", "writer": w,
                     "stroke_width": stroke_width(img),
                     "fill_ratio": fill_ratio(img)})
    return pd.DataFrame(rows)


# Figures

def fig_learning_curves():
    """Learning curves of the CNN and augmented CNN (3 seeds)."""
    histories = {}
    for tag, label in (("mnist", "CNN"), ("aug", "CNN + augmentation")):
        runs = [pd.read_csv(p) for p in
                sorted(config.RESULTS_DIR.glob(f"history_{tag}_seed*.csv"))]
        if runs:
            histories[label] = runs
    if histories:
        plotting.learning_curves(histories,
                                 config.FIGURES_DIR / "learning_curves")


def fig_samples(mnist_x, mnist_y, ours01, labels, writers):
    """Rows: MNIST + each writer, columns: digits 0-9."""
    rows, names = [], []
    rows.append([mnist_x[np.flatnonzero(mnist_y == d)[0]] for d in range(10)])
    names.append("MNIST")
    for w in sorted(set(writers)):
        row = []
        for d in range(10):
            idx = np.flatnonzero((writers == w) & (labels == d))
            row.append(ours01[idx[0]] if len(idx) else None)
        rows.append(row)
        names.append(w)
    plotting.digit_grid(rows, names, config.FIGURES_DIR / "samples_by_writer",
                        "Same digits: MNIST vs. each writer "
                        "(minimal preprocessing)")


def fig_preprocessing_steps(raw, labels):
    """A few drawings with more and more preprocessing steps."""
    stages = [
        ("raw", None),
        ("minimal", MINIMAL_STEPS),
        ("+ binarize", ["binarize", "resize_padded"]),
        ("+ thickness", ["binarize", "normalize_thickness", "resize_padded"]),
        ("+ centre of mass", [s for s in ALL_STEPS if s != "normalize_pixels"]),
    ]
    rng = np.random.default_rng(config.SEED)
    picks = rng.choice(len(raw), size=min(4, len(raw)), replace=False)
    rows = []
    for i in picks:
        row = []
        for name, steps in stages:
            if steps is None:
                row.append(raw[i])
            else:
                out = preprocess(raw[i], steps)
                row.append(to_display(out) if "normalize_pixels" in steps
                           else out)
        rows.append(row)
    plotting.digit_grid(rows, [f"label {labels[i]}" for i in picks],
                        config.FIGURES_DIR / "preprocessing_steps",
                        "Preprocessing: " + " | ".join(n for n, _ in stages),
                        cell=1.0)


def fig_augmentation(mnist_x):
    """MNIST digits and 9 augmented versions of each."""
    rng = np.random.default_rng(config.SEED)
    rows = []
    for img in mnist_x[:3]:
        rows.append([img] + [augment(img, rng) for _ in range(9)])
    plotting.digit_grid(rows, ["", "", ""],
                        config.FIGURES_DIR / "augmentation_examples",
                        "Training-time augmentation: original, then 9 "
                        "random distortions")


def fig_domain(stats, writers):
    """Histograms of stroke width and fill ratio, MNIST vs. writers."""
    colours = [plotting.GREY] + [plotting.writer_colours(writers)[w]
                                 for w in sorted(set(writers))]
    for metric, label, name in (
        ("stroke_width", "stroke width after resizing [px]",
         "domain_stroke_width"),
        ("fill_ratio", "share of image covered by ink", "domain_fill_ratio"),
    ):
        data = {"MNIST": stats.loc[stats.domain == "MNIST", metric].values}
        for w in sorted(set(writers)):
            data[w] = stats.loc[stats.writer == w, metric].values
        plotting.distributions(data, f"{label.split(' [')[0].capitalize()}: "
                               "MNIST vs. our writers", label,
                               config.FIGURES_DIR / name, colours)


def fig_ablation(summary, mnist_acc):
    """Main result: accuracy for each mitigation."""
    def pick(model, variant, experiment="zero_shot", stage="-"):
        # one row of the summary table, or None
        row = summary[(summary.experiment == experiment)
                      & (summary.model == model)
                      & (summary.variant == variant)
                      & (summary.stage == stage)]
        return row.iloc[0] if len(row) else None

    bars = [
        ("minimal", pick("cnn", "minimal"), plotting.ORANGE),
        ("+ thickness", pick("cnn", "thickness"), plotting.BLUE),
        ("+ centering", pick("cnn", "centering"), plotting.BLUE),
        ("+ both", pick("cnn", "full"), plotting.BLUE),
        ("augmented\ntraining", pick("aug", "full"), plotting.AQUA),
        ("fine-tuned\n(30% test)", pick("cnn", "full", "finetune", "after"),
         plotting.VIOLET),
    ]
    bars = [b for b in bars if b[1] is not None]
    plotting.bar_with_errors(
        [b[0] for b in bars],
        [b[1]["accuracy_mean"] for b in bars],
        [np.nan_to_num(b[1]["accuracy_std"]) for b in bars],
        "CNN accuracy on our data, mean ± std over seeds",
        config.FIGURES_DIR / "ablation_cnn",
        colours=[b[2] for b in bars],
        reference=mnist_acc, reference_label="MNIST test")

    zs = summary[summary.experiment == "zero_shot"]
    models = [m for m in ("logreg", "cnn", "aug") if m in set(zs.model)]
    variants = list(dict.fromkeys(zs.variant))
    vals = np.array([[zs[(zs.model == m) & (zs.variant == v)]
                      .accuracy_mean.iloc[0] for v in variants] for m in models])
    errs = np.array([[np.nan_to_num(zs[(zs.model == m) & (zs.variant == v)]
                      .accuracy_std.iloc[0]) for v in variants] for m in models])
    names = {"logreg": "logistic regression", "cnn": "CNN",
             "aug": "CNN + augmentation"}
    plotting.grouped_bars(variants, [names[m] for m in models], vals, errs,
                          "Zero-shot accuracy on our data by model and "
                          "preprocessing", config.FIGURES_DIR / "ablation_models",
                          colours=[plotting.GREY, plotting.BLUE, plotting.AQUA])


def fig_lowo():
    """LOWO accuracy before/after fine-tuning."""
    path = config.RESULTS_DIR / "lowo_summary.csv"
    if not path.exists():
        return
    s = pd.read_csv(path)
    writers = list(dict.fromkeys(s.writer))
    vals, errs = [], []
    for stage in ("before", "after"):
        sub = s[s.stage == stage].set_index("writer").loc[writers]
        vals.append(sub.accuracy_mean.values)
        errs.append(np.nan_to_num(sub.accuracy_std.values))
    plotting.grouped_bars(
        writers, ["MNIST model (zero-shot)", "fine-tuned on other writers"],
        np.array(vals), np.array(errs),
        "Leave-one-writer-out: accuracy on the held-out writer",
        config.FIGURES_DIR / "lowo", colours=[plotting.GREY, plotting.VIOLET])


def write_headline(summary, mnist_runs):
    """Main numbers for the report -> headline.json"""
    def mean_acc(model, variant, experiment="zero_shot", stage="-"):
        # mean accuracy, or None if this config wasn't run
        row = summary[(summary.experiment == experiment)
                      & (summary.model == model)
                      & (summary.variant == variant)
                      & (summary.stage == stage)]
        return float(row.accuracy_mean.iloc[0]) if len(row) else None

    mnist_cnn = float(mnist_runs[mnist_runs.model == "cnn"].accuracy.mean())
    ours_minimal = mean_acc("cnn", "minimal")
    headline = {
        "mnist_test_accuracy_cnn": mnist_cnn,
        "mnist_test_accuracy_logreg": float(
            mnist_runs[mnist_runs.model == "logreg"].accuracy.mean())
        if "logreg" in set(mnist_runs.model) else None,
        "ours_accuracy_cnn_minimal": ours_minimal,
        "domain_gap_points": (mnist_cnn - ours_minimal) * 100
        if ours_minimal is not None else None,
        "ours_accuracy_cnn_full_preprocessing": mean_acc("cnn", "full"),
        "ours_accuracy_aug_full_preprocessing": mean_acc("aug", "full"),
        "ours_accuracy_finetuned": mean_acc("cnn", "full", "finetune", "after"),
    }
    with open(config.RESULTS_DIR / "headline.json", "w") as f:
        json.dump(headline, f, indent=2)
    for k, v in headline.items():
        print(f"  {k:40s} {v if v is None else round(v, 4)}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    # MNIST results are always in the main folder, results on our data go
    # to the synthetic/ subfolder with --synthetic
    if args.quick:
        config.redirect_outputs("smoke")
    mnist_runs = pd.read_csv(config.RESULTS_DIR / "mnist_test_runs.csv")
    fig_learning_curves()
    if args.synthetic:
        sub = "smoke/synthetic" if args.quick else "synthetic"
        config.redirect_outputs(sub, models=False)
        print("*** SYNTHETIC STAND-IN DATA - not results on our dataset ***")

    mnist_x, mnist_y = load_mnist(train=False)
    raw, labels, writers = load_target(args.synthetic)
    ours01 = preprocess_many(raw, ["resize_padded"])   # [0, 1], for plots

    print("figures: samples, preprocessing, augmentation")
    fig_samples(mnist_x, mnist_y, ours01, labels, writers)
    fig_preprocessing_steps(raw, labels)
    fig_augmentation(mnist_x)

    print("domain statistics")
    rng = np.random.default_rng(config.SEED)
    sample = rng.choice(len(mnist_x), size=MNIST_SAMPLE, replace=False)
    stats = domain_statistics(mnist_x[sample] / 255.0, ours01, writers)
    table = stats.groupby("writer")[["stroke_width", "fill_ratio"]].agg(
        ["mean", "std", "median"])
    table.columns = [f"{a}_{b}" for a, b in table.columns]
    table.to_csv(config.RESULTS_DIR / "domain_stats.csv")
    print(table.round(3).to_string())
    fig_domain(stats, writers)

    summary_path = config.RESULTS_DIR / "ablation_summary.csv"
    if summary_path.exists():
        print("figures: ablation, cross-writer")
        summary = pd.read_csv(summary_path)
        mnist_acc = float(mnist_runs[mnist_runs.model == "cnn"].accuracy.mean())
        fig_ablation(summary, mnist_acc)
        fig_lowo()
        print("headline numbers")
        write_headline(summary, mnist_runs)
    else:
        print("no ablation_summary.csv yet - run ablation.py first")

    print(f"\nfigures written to {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
