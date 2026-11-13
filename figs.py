"""
Figures for the report (pipeline diagram, zoomed axes, CIs, per-writer plot).

Usage: python figs.py <folder with the result csv files> <output folder>
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

D = Path(sys.argv[1])
F = Path(sys.argv[2])
F.mkdir(exist_ok=True, parents=True)

# same colors as in plotting.py
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, GREEN, VIOLET, RED = "#e87ba4", "#008300", "#6250d6", "#e34948"
GREY, INK, INK2 = "#a3a29b", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 220, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelcolor": INK2, "axes.edgecolor": "#c9c8c2",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False,
    "lines.linewidth": 1.8,
})

# Fig 1: pipeline
fig, ax = plt.subplots(figsize=(6.6, 2.9))
ax.set_xlim(0, 10); ax.set_ylim(0, 4.4); ax.axis("off"); ax.grid(False)
def box(x, y, w, h, title, sub, colour):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=colour, ec="none", alpha=0.16))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="none", ec=colour, lw=1.4))
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=8.2,
            fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=6.9, color=INK2)
def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9,
                                 color=INK2, lw=1.2))
w, h = 2.1, 1.25
xs = [0.1, 2.65, 5.2, 7.75]
row1, row2 = 2.8, 0.55
box(xs[0], row1, w, h, "Webcam frame", "mirrored, 960×540", BLUE)
box(xs[1], row1, w, h, "MediaPipe Hands", "21 hand landmarks", BLUE)
box(xs[2], row1, w, h, "Gesture rules", "finger state → mode", BLUE)
box(xs[3], row1, w, h, "Canvas", "smoothed tip, 12 px pen", BLUE)
box(xs[3], row2, w, h, "Raw crop", "20 px margin, PNG", ORANGE)
box(xs[2], row2, w, h, "Preprocessing", "4 variants", ORANGE)
box(xs[1], row2, w, h, "Classifier", "LogReg / CNN", AQUA)
box(xs[0], row2, w, h, "Digit + confidence", "softmax probability", AQUA)
for i in range(3):
    arrow(xs[i] + w + 0.04, row1 + h / 2, xs[i + 1] - 0.04, row1 + h / 2)
arrow(xs[3] + w / 2, row1 - 0.04, xs[3] + w / 2, row2 + h + 0.04)
for i in (3, 2, 1):
    arrow(xs[i] - 0.04, row2 + h / 2, xs[i - 1] + w + 0.04, row2 + h / 2)
fig.savefig(F / "fig_pipeline.png"); plt.close(fig)

# Fig 4: learning curves
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
for tag, label, colour in (("mnist", "CNN", BLUE), ("aug", "CNN + augmentation", AQUA)):
    runs = [pd.read_csv(D / f"history_{tag}_seed{s}.csv") for s in (42, 43, 44)]
    ep = runs[0].epoch.values
    for ax, metric in zip(axes, ("loss", "acc")):
        for split, style in (("train", "--"), ("val", "-")):
            st = np.stack([r[f"{split}_{metric}"].values for r in runs])
            ax.plot(ep, st.mean(0), style, color=colour, label=f"{label} ({split})")
            ax.fill_between(ep, st.min(0), st.max(0), color=colour, alpha=0.13, lw=0)
axes[0].set_title("Loss"); axes[0].set_yscale("log")
axes[1].set_title("Accuracy"); axes[1].set_ylim(0.86, 1.0)
for ax in axes:
    ax.set_xlabel("epoch"); ax.set_xticks(range(1, 9))
axes[1].legend(fontsize=6.8, loc="lower right")
fig.tight_layout(); fig.savefig(F / "fig_learning_curves.png"); plt.close(fig)

# Fig 5: ablation
ci = pd.read_csv(D / "confidence_intervals.csv")
gap = pd.read_csv(D / "gap_closed.csv")
variants = ["minimal", "thickness", "centering", "full"]
fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), gridspec_kw={"width_ratios": [1.15, 1]})
x = np.arange(4)
for model, colour, dx, label in (("cnn", BLUE, -0.09, "CNN"), ("aug", AQUA, 0.09, "CNN + augmentation")):
    sub = ci[ci.model == model].set_index("variant").loc[variants]
    axes[0].errorbar(x + dx, 100 * sub.accuracy,
                     yerr=[100 * (sub.accuracy - sub.ci95_low), 100 * (sub.ci95_high - sub.accuracy)],
                     fmt="o", color=colour, capsize=3, ms=5, label=label)
    ref = gap[gap.model == model].mnist_accuracy.iloc[0] * 100
    axes[0].axhline(ref, color=colour, ls=":", lw=1.1)
axes[0].set_xticks(x, variants); axes[0].set_ylim(92.5, 100.2)
axes[0].set_ylabel("accuracy on our data, %"); axes[0].set_title("CNN models")
axes[0].legend(loc="lower right", fontsize=7)
sub = ci[ci.model == "logreg"].set_index("variant").loc[variants]
axes[1].errorbar(x, 100 * sub.accuracy,
                 yerr=[100 * (sub.accuracy - sub.ci95_low), 100 * (sub.ci95_high - sub.accuracy)],
                 fmt="o", color=GREY, capsize=3, ms=5)
axes[1].axhline(gap[gap.model == "logreg"].mnist_accuracy.iloc[0] * 100, color=GREY, ls=":", lw=1.1)
axes[1].set_xticks(x, variants); axes[1].set_ylim(50, 96); axes[1].set_title("Logistic regression")
fig.text(0.5, -0.02, "dotted line: accuracy of the same model on the MNIST test set; "
         "bars: 95% bootstrap CI", ha="center", fontsize=7, color=INK2)
fig.tight_layout(); fig.savefig(F / "fig_ablation.png"); plt.close(fig)

# Fig 6: confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.1))
for ax, var, title in zip(axes, ("minimal", "full"), ("minimal preprocessing", "full preprocessing")):
    cm = pd.read_csv(D / f"confusion_cnn_{var}_seed42.csv", index_col=0).values
    norm = cm / cm.sum(1, keepdims=True)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
    for i in range(10):
        for j in range(10):
            if cm[i, j] and i != j:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7, color=RED, fontweight="bold")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"CNN, {title}\n({int((cm.sum() - np.trace(cm)))} errors of {int(cm.sum())})", fontsize=8.5)
fig.tight_layout(); fig.savefig(F / "fig_confusion.png"); plt.close(fig)

# Fig 9: per writer
pw = pd.read_csv(D / "per_writer.csv")
fig, ax = plt.subplots(figsize=(4.8, 2.9))
cols = {"kacper": BLUE, "kirill": ORANGE, "mateusz": AQUA, "roman": VIOLET}
for w, c in cols.items():
    s = pw[(pw.model == "cnn") & (pw.writer == w)].set_index("variant").loc[variants]
    ax.errorbar(x, 100 * s.accuracy_mean, yerr=100 * s.accuracy_std, marker="o", ms=4.5,
                color=c, capsize=2.5, label=f"{w.capitalize()} ({s.stroke_width_median.iloc[0]:.2f} px)")
ax.set_xticks(x, variants); ax.set_ylim(91.5, 100.4)
ax.set_ylabel("accuracy, %"); ax.set_title("CNN, accuracy per writer")
ax.legend(title="writer (median stroke width)", fontsize=7, title_fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig(F / "fig_per_writer.png"); plt.close(fig)

# Fig 10: sensitivity
sn = pd.read_csv(D / "stroke_ratio_sensitivity.csv")
agg = sn.groupby("stroke_ratio").accuracy.agg(["mean", "std"])
fig, ax = plt.subplots(figsize=(4.6, 2.7))
ax.errorbar(agg.index, 100 * agg["mean"], yerr=100 * agg["std"], marker="o", ms=4.5,
            color=BLUE, capsize=3)
ax.axvline(0.12, color=INK, lw=0.9, ls="--")
ax.text(0.1215, 98.62, "value used\n(fixed in advance)", fontsize=7, color=INK2, va="bottom")
ax.set_xlabel("TARGET_STROKE_RATIO"); ax.set_ylabel("accuracy, %")
ax.set_title("CNN, full preprocessing")
fig.tight_layout(); fig.savefig(F / "fig_sensitivity.png"); plt.close(fig)
print("figures written to", F)
