"""
Common plot style and helper functions for all figures.

Figures are saved as PNG and PDF. Colors: each writer always has the same
color, blue scale for confusion matrices, grey for reference values
(MNIST, "before").
"""

import matplotlib

matplotlib.use("Agg")                    # no GUI needed
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

# color palette (colorblind-friendly)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, GREEN, VIOLET, RED = "#e87ba4", "#008300", "#6250d6", "#e34948"
CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
GREY = "#a3a29b"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelcolor": INK_SECONDARY,
    "axes.edgecolor": "#c9c8c2",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e6e5e0",
    "grid.linewidth": 0.6,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "legend.frameon": False,
    "lines.linewidth": 2,
})


def save(fig, path):
    """Save as .png and .pdf and close the figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path.with_suffix(".png")


def writer_colours(writers):
    """writer -> color, by alphabetical order"""
    return {w: CATEGORICAL[i % len(CATEGORICAL)]
            for i, w in enumerate(sorted(set(writers)))}


def _show_digit(ax, img, title=None, colour=INK):
    """Show one digit image without axes."""
    ax.imshow(img, cmap="gray", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title is not None:
        ax.set_title(title, fontsize=8, fontweight="normal", color=colour)


def confusion(cm, title, path):
    """
    Confusion matrix, colors normalized per row (classes have different
    sizes), raw counts written in the cells.
    """
    rows = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, rows, out=np.zeros_like(cm, dtype=float),
                     where=rows > 0)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    for i in range(10):
        for j in range(10):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.55 else INK)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("predicted digit")
    ax.set_ylabel("true digit")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, label="share of true class")
    return save(fig, path)


def learning_curves(histories, path):
    """
    Loss and accuracy per epoch, train and val.

    histories: dict label -> list of history DataFrames (one per seed).
    Line = mean over seeds, shaded area = min-max.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for (label, runs), colour in zip(histories.items(), CATEGORICAL):
        for ax, metric in zip(axes, ("loss", "acc")):
            for split, style in (("train", "--"), ("val", "-")):
                col = f"{split}_{metric}"
                stacked = np.stack([h[col].values for h in runs])
                epochs = runs[0]["epoch"].values
                ax.plot(epochs, stacked.mean(0), style, color=colour,
                        label=f"{label} ({split})")
                if len(runs) > 1:
                    ax.fill_between(epochs, stacked.min(0), stacked.max(0),
                                    color=colour, alpha=0.15, linewidth=0)
    axes[0].set_title("Loss")
    axes[1].set_title("Accuracy")
    for ax in axes:
        ax.set_xlabel("epoch")
    axes[1].legend(fontsize=8)
    return save(fig, path)


def bar_with_errors(labels, means, stds, title, path, ylabel="accuracy",
                    colours=None, reference=None, reference_label=None):
    """
    Bars with mean +/- std over seeds. `reference` adds a dashed line
    (e.g. MNIST test accuracy).
    """
    fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(labels) + 2), 3.8))
    x = np.arange(len(labels))
    colours = colours or [BLUE] * len(labels)
    ax.bar(x, means, yerr=stds, color=colours, width=0.62, capsize=4,
           error_kw={"elinewidth": 1.2, "ecolor": INK_SECONDARY})
    for xi, m, s in zip(x, means, stds):
        ax.text(xi, m + (s if np.isfinite(s) else 0) + 0.015, f"{m:.3f}",
                ha="center", va="bottom", fontsize=8, color=INK)
    if reference is not None:
        ax.axhline(reference, color=GREY, linestyle="--", linewidth=1.2)
        ax.text(len(labels) - 0.5, reference, f" {reference_label}",
                va="bottom", ha="right", fontsize=8, color=INK_SECONDARY)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    return save(fig, path)


def grouped_bars(groups, series, values, errors, title, path,
                 colours=None, ylabel="accuracy"):
    """
    Grouped bar chart. values/errors have shape (len(series), len(groups)).
    """
    colours = colours or CATEGORICAL
    n = len(series)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(max(5, 1.3 * len(groups) + 2), 3.8))
    x = np.arange(len(groups))
    for i, (name, colour) in enumerate(zip(series, colours)):
        pos = x - 0.4 + width * (i + 0.5)
        ax.bar(pos, values[i], yerr=errors[i], width=width * 0.92,
               color=colour, label=name, capsize=3,
               error_kw={"elinewidth": 1, "ecolor": INK_SECONDARY})
        for p, v in zip(pos, values[i]):
            ax.text(p, v + 0.02, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", visible=False)
    return save(fig, path)


def distributions(data, title, xlabel, path, colours=None, bins=30):
    """Overlapping histograms (density). data: dict label -> values."""
    colours = colours or [GREY] + CATEGORICAL
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    all_values = np.concatenate([np.asarray(v) for v in data.values()])
    edges = np.histogram_bin_edges(all_values, bins=bins)
    for (label, values), colour in zip(data.items(), colours):
        ax.hist(values, bins=edges, density=True, histtype="stepfilled",
                alpha=0.35, color=colour, edgecolor=colour, linewidth=1.5,
                label=f"{label} (median {np.median(values):.2f})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return save(fig, path)


def digit_grid(rows, row_labels, path, title=None, cell=0.75):
    """
    Grid of digits. rows = list of lists of images (None = empty cell),
    row_labels are shown on the left.
    """
    n_rows, n_cols = len(rows), max(len(r) for r in rows)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(cell * n_cols + 1.2, cell * n_rows + 0.4),
                             squeeze=False)
    for r, (images, label) in enumerate(zip(rows, row_labels)):
        for c in range(n_cols):
            ax = axes[r, c]
            img = images[c] if c < len(images) else None
            if img is None:
                ax.axis("off")
                continue
            _show_digit(ax, img)
        axes[r, 0].set_ylabel(label, rotation=0, ha="right", va="center",
                              fontsize=8, color=INK_SECONDARY)
        axes[r, 0].axis("on")
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.subplots_adjust(wspace=0.05, hspace=0.08)
    return save(fig, path)


def gallery(images, captions, path, title=None, cols=10, caption_colour=RED):
    """Images with a caption under each (e.g. misclassified digits)."""
    n = len(images)
    if n == 0:
        return None
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 0.95, rows * 1.15),
                             squeeze=False)
    for k, ax in enumerate(axes.flat):
        if k < n:
            _show_digit(ax, images[k], captions[k], caption_colour)
        else:
            ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold")
    return save(fig, path)
