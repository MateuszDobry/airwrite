"""
Error analysis (docs/remaining-work.md, section 2).

    1. per-class F1 for every model and variant (mean of 3 seeds)
    2. how the common confusions change from minimal to full preprocessing
    3. gallery of the main confusions: raw, minimal input, full input
    4. which writers' "1"s the augmented CNN predicts as "7"
    5. shift test on MNIST: CNN vs. logistic regression accuracy when the
       digits are shifted by k pixels (to check why centering helps logreg
       so much more than the CNN)

1, 2, 4 use results/metrics/predictions.csv from metrics_analysis.py.
3 and 5 run the saved checkpoints (no training), a few minutes on CPU.

Usage:
    python metrics_analysis.py   # first, if results/metrics/ is missing
    python error_analysis.py

Outputs (results/analysis/):
    per_class_f1_all.csv     F1 per digit, per model x variant (mean over seeds)
    confusion_change.csv     errors per seed for each (true, predicted) pair,
                             minimal vs. full preprocessing, per model
    ones_as_sevens.csv       share of each writer's "1"s predicted as "7"
    shift_sensitivity.csv    MNIST test accuracy vs. shift in pixels
Figures (figures/analysis/): per_class_f1, confusion_gallery,
shift_sensitivity.
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import config
import plotting
from data_utils import load_mnist, load_target
from model import load_model, predict_batch
from preprocessing import FULL_STEPS, MINIMAL_STEPS, preprocess

OUT_DIR = config.RESULTS_DIR / "analysis"
FIG_DIR = config.FIGURES_DIR / "analysis"
PREDICTIONS = config.RESULTS_DIR / "metrics" / "predictions.csv"

MODELS = ["logreg", "cnn", "aug"]
VARIANT_ORDER = ["minimal", "thickness", "centering", "full"]

# (model, true, predicted) - most common errors of the CNN with minimal
# preprocessing (confusion_pairs.csv) + the main error of the augmented CNN
GALLERY_PAIRS = [("cnn", 6, 5), ("cnn", 7, 1), ("cnn", 1, 7), ("cnn", 9, 3),
                 ("aug", 1, 7)]
GALLERY_PER_PAIR = 6

# shifts in pixels for the MNIST shift test
SHIFTS = [0, 1, 2, 3, 4, 5, 6]


# 1. Per-class F1

def per_class_f1(pred):
    """F1 per digit for each (model, variant, seed), then mean/std."""
    rows = []
    for (m, v, s), g in pred.groupby(["model", "variant", "seed"]):
        f1 = f1_score(g.label, g.pred, labels=range(10), average=None,
                      zero_division=0)
        for digit, value in enumerate(f1):
            rows.append({"model": m, "variant": v, "seed": s,
                         "digit": digit, "f1": value})
    runs = pd.DataFrame(rows)
    return (runs.groupby(["model", "variant", "digit"])["f1"]
            .agg(f1_mean="mean", f1_std="std").reset_index())


def fig_per_class_f1(table):
    """Heatmap per model: variant x digit."""
    fig, axes = plt.subplots(1, len(MODELS), figsize=(13, 2.9),
                             constrained_layout=True)
    for ax, m in zip(axes, MODELS):
        sub = table[table.model == m].pivot(index="variant", columns="digit",
                                            values="f1_mean")
        sub = sub.loc[VARIANT_ORDER]
        # separate color range per panel, otherwise all CNN cells look
        # the same (all close to 1)
        im = ax.imshow(sub.to_numpy(), cmap="Blues",
                       vmin=sub.to_numpy().min(), vmax=1)
        ax.grid(False)
        for i in range(sub.shape[0]):
            for j in range(10):
                val = sub.iat[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if val > (sub.to_numpy().min() + 1) / 2
                        else plotting.INK)
        ax.set_xticks(range(10))
        ax.set_yticks(range(len(VARIANT_ORDER)))
        ax.set_yticklabels(VARIANT_ORDER)
        ax.set_xlabel("digit")
        ax.set_title({"logreg": "Logistic regression", "cnn": "CNN",
                      "aug": "Augmented CNN"}[m])
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle("Per-class F1 on our digits (mean of 3 seeds)",
                 fontsize=11, fontweight="bold")
    return plotting.save(fig, FIG_DIR / "per_class_f1")


# 2. Confusions: minimal vs. full

def confusion_change(pred):
    """
    Errors per seed for each (true, predicted) pair, minimal vs. full.
    status = removed / reduced / unchanged or worse / new.
    """
    wrong = pred[~pred.correct & pred.variant.isin(["minimal", "full"])]
    n_seeds = pred.groupby("model")["seed"].nunique()
    counts = (wrong.groupby(["model", "variant", "label", "pred"])
              .size().rename("count").reset_index())
    counts["per_seed"] = counts["count"] / counts["model"].map(n_seeds)
    table = counts.pivot_table(index=["model", "label", "pred"],
                               columns="variant", values="per_seed",
                               fill_value=0.0).reset_index()
    for col in ["minimal", "full"]:
        if col not in table:
            table[col] = 0.0
    table = table.rename(columns={"minimal": "errors_minimal",
                                  "full": "errors_full"})

    def status(row):
        if row.errors_minimal == 0:
            return "new"
        if row.errors_full == 0:
            return "removed"
        if row.errors_full < row.errors_minimal:
            return "reduced"
        return "unchanged or worse"

    table["status"] = table.apply(status, axis=1)
    table.columns.name = None
    return table.sort_values(["model", "errors_minimal"],
                             ascending=[True, False])


# 3. Gallery of the main confusions

def to_display(img28):
    """Undo the normalization for display (0..1)."""
    return np.clip(img28 * config.MNIST_STD + config.MNIST_MEAN, 0, 1)


def fig_confusion_gallery(pred, raw):
    """
    Up to GALLERY_PER_PAIR examples for each pair in GALLERY_PAIRS (seed 42).
    Each column: raw drawing, minimal input, full input + predictions.
    """
    seed = config.SEED
    p42 = pred[pred.seed == seed].set_index(["model", "variant", "image"])
    columns = []          # [raw, minimal, full] per example
    captions = []
    for m, true, wrong in GALLERY_PAIRS:
        # aug 1->7 is shown with full preprocessing (it stays there),
        # CNN errors with minimal
        variant = "full" if m == "aug" else "minimal"
        sub = pred[(pred.seed == seed) & (pred.model == m)
                   & (pred.variant == variant)
                   & (pred.label == true) & (pred.pred == wrong)]
        for idx in sub.image.to_numpy()[:GALLERY_PER_PAIR]:
            pm = int(p42.loc[(m, "minimal", idx), "pred"])
            pf = int(p42.loc[(m, "full", idx), "pred"])
            columns.append([raw[idx] / 255.0,
                            to_display(preprocess(raw[idx], MINIMAL_STEPS)),
                            to_display(preprocess(raw[idx], FULL_STEPS))])
            captions.append((f"{m} {true}→{wrong}", f"min: {pm}",
                             f"full: {pf}"))
    if not columns:
        return None

    n = len(columns)
    fig, axes = plt.subplots(3, n, figsize=(n * 0.85 + 0.8, 3.4),
                             squeeze=False)
    row_names = ["raw", "minimal", "full"]
    for c, (images, caps) in enumerate(zip(columns, captions)):
        for r in range(3):
            ax = axes[r, c]
            ax.imshow(images[r], cmap="gray", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.set_title(caps[r], fontsize=6, fontweight="normal",
                         color=plotting.INK_SECONDARY)
        for r in range(3):
            if c == 0:
                axes[r, 0].set_ylabel(row_names[r], fontsize=8,
                                      color=plotting.INK_SECONDARY)
    fig.suptitle("Main confusions (seed 42): raw drawing and model inputs",
                 fontsize=11, fontweight="bold")
    return plotting.save(fig, FIG_DIR / "confusion_gallery")


# 4. "1" predicted as "7", per writer

def ones_as_sevens(pred):
    """Fraction of each writer's "1"s predicted as "7"."""
    ones = pred[pred.label == 1]
    table = (ones.assign(as_seven=ones.pred == 7)
             .groupby(["model", "variant", "writer"])["as_seven"]
             .agg(share_as_seven="mean", n="size").reset_index())
    # n is images x seeds, we want images
    table["n"] = table["n"] // pred.seed.nunique()
    return table


# 5. Shift test on MNIST

def shift(images, dx, dy):
    """Shift images by (dx, dy), fill with black."""
    out = np.zeros_like(images)
    h, w = images.shape[1:]
    src_y = slice(max(0, -dy), h - max(0, dy))
    dst_y = slice(max(0, dy), h - max(0, -dy))
    src_x = slice(max(0, -dx), w - max(0, dx))
    dst_x = slice(max(0, dx), w - max(0, -dx))
    out[:, dst_y, dst_x] = images[:, src_y, src_x]
    return out


def shift_sensitivity(seeds):
    """
    MNIST test accuracy with digits shifted by k pixels. Average of the
    4 directions (right, left, down, up).
    """
    images, labels = load_mnist(train=False)
    rows = []
    for seed in seeds:
        cnn = load_model("mnist", seed)
        logreg = joblib.load(config.MODELS_DIR / f"logreg_seed{seed}.joblib")
        for k in SHIFTS:
            directions = [(k, 0), (-k, 0), (0, k), (0, -k)] if k else [(0, 0)]
            acc = {"cnn": [], "logreg": []}
            for dx, dy in directions:
                x = shift(images, dx, dy).astype(np.float32) / 255.0
                x = (x - config.MNIST_MEAN) / config.MNIST_STD
                p_cnn, _ = predict_batch(cnn, x)
                p_lr = logreg.predict(x.reshape(len(x), -1))
                acc["cnn"].append(np.mean(p_cnn == labels))
                acc["logreg"].append(np.mean(p_lr == labels))
            for m, values in acc.items():
                rows.append({"model": m, "seed": seed, "shift_px": k,
                             "accuracy": float(np.mean(values))})
        print(f"  shift test seed {seed} done")
    return pd.DataFrame(rows)


def fig_shift(table):
    """Accuracy vs. shift, mean/std over seeds."""
    summary = (table.groupby(["model", "shift_px"])["accuracy"]
               .agg(["mean", "std"]).reset_index())
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for m, colour, name in [("cnn", plotting.BLUE, "CNN"),
                            ("logreg", plotting.ORANGE,
                             "logistic regression")]:
        s = summary[summary.model == m]
        ax.errorbar(s.shift_px, s["mean"] * 100, yerr=s["std"] * 100,
                    marker="o", color=colour, label=name, capsize=3)
    ax.set_xlabel("shift of the MNIST test digit (pixels at 28×28)")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Translation sensitivity on MNIST")
    ax.legend(frameon=False)
    return plotting.save(fig, FIG_DIR / "shift_sensitivity")


def main():
    if not PREDICTIONS.exists():
        raise SystemExit(f"{PREDICTIONS} not found. "
                         "Run `python metrics_analysis.py` first.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(PREDICTIONS)
    seeds = sorted(pred.seed.unique())
    print(f"{len(pred)} predictions, seeds {seeds}")

    f1 = per_class_f1(pred)
    f1.to_csv(OUT_DIR / "per_class_f1_all.csv", index=False)
    fig_per_class_f1(f1)

    confusion_change(pred).to_csv(OUT_DIR / "confusion_change.csv",
                                  index=False)
    ones_as_sevens(pred).to_csv(OUT_DIR / "ones_as_sevens.csv", index=False)

    raw, labels, _ = load_target(synthetic=False)
    # predictions.csv uses the load_target() order, make sure it matches
    first = pred[(pred.model == "cnn") & (pred.variant == "minimal")
                 & (pred.seed == seeds[0])].sort_values("image")
    assert np.array_equal(first.label.to_numpy(), labels), \
        "image order differs from predictions.csv; rerun metrics_analysis.py"
    fig_confusion_gallery(pred, raw)

    print("shift test on MNIST...")
    sens = shift_sensitivity(seeds)
    sens.to_csv(OUT_DIR / "shift_sensitivity.csv", index=False)
    fig_shift(sens)

    print(f"written to {OUT_DIR} and {FIG_DIR}")


if __name__ == "__main__":
    main()
