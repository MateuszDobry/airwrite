"""
Dataset loading.

load_mnist()               MNIST via torchvision
load_airwrite()            our digits from data/raw/
make_synthetic_airwrite()  fake "air" digits made from MNIST test images.
                           Only for testing the pipeline before we had real
                           data - not a result.
"""

import csv

import cv2
import numpy as np

import config
from preprocessing import elastic_deform, skeletonize


def load_mnist(train: bool):
    """MNIST as numpy arrays: uint8 images (N, 28, 28) and int64 labels."""
    from torchvision.datasets import MNIST    # slow import, only load when needed

    ds = MNIST(root=str(config.DATA_MNIST), train=train, download=True)
    return ds.data.numpy(), ds.targets.numpy().astype(np.int64)


def scan_raw_folder():
    """
    List samples in data/raw/<writer>/<digit>/*.png.

    Returns sorted (path, label, writer) tuples - sorted so that splits made
    later are reproducible.
    """
    rows = []
    for path in sorted(config.DATA_RAW.glob("*/*/*.png")):
        digit_dir = path.parent.name
        if not digit_dir.isdigit():
            continue
        rows.append((path, int(digit_dir), path.parent.parent.name))
    return rows


def write_labels_csv(rows):
    """Write data/labels.csv (path, label, writer)."""
    out = config.PROJECT_ROOT / "data" / "labels.csv"
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "writer"])
        for path, label, who in rows:
            writer.writerow([path.relative_to(config.PROJECT_ROOT).as_posix(),
                             label, who])
    return out


def load_airwrite():
    """
    Load our raw digits (not resized yet).

    Returns a list of uint8 images of different sizes, labels and writer
    names. Also regenerates data/labels.csv.
    """
    rows = scan_raw_folder()
    if not rows:
        raise FileNotFoundError(
            "No collected digits found in data/raw/. Run collect_data.py "
            "first, or pass --synthetic to test the pipeline on stand-in "
            "data.")
    write_labels_csv(rows)

    images = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p, _, _ in rows]
    labels = np.array([r[1] for r in rows], dtype=np.int64)
    writers = np.array([r[2] for r in rows])
    return images, labels, writers


# Synthetic data (development only).
# Four fake writers with different settings so leave-one-writer-out has
# something to work with.
_SYNTHETIC_WRITERS = {
    "synth_a": dict(thickness=(10, 14), slant=-0.10, tremor=3.0, joins=0.3),
    "synth_b": dict(thickness=(14, 20), slant=0.15, tremor=5.0, joins=0.5),
    "synth_c": dict(thickness=(8, 12), slant=0.00, tremor=2.0, joins=0.2),
    "synth_d": dict(thickness=(16, 22), slant=0.25, tremor=6.0, joins=0.6),
}


def _air_stroke(digit28, rng, style):
    """Make an MNIST digit look like it was drawn with a finger."""
    # upscale to webcam size and take the skeleton
    size = int(rng.integers(160, 260))
    big = cv2.resize(digit28, (size, size), interpolation=cv2.INTER_CUBIC)
    big = (big > 100).astype(np.uint8) * 255
    skel = skeletonize(big)

    # slant + random horizontal stretch
    stretch = rng.uniform(0.8, 1.25)
    shear = np.float32([[stretch, style["slant"], 0], [0, 1, 0]])
    width = int(size * (stretch + abs(style["slant"]))) + 1
    skel = cv2.warpAffine(skel, shear, (width, size), borderValue=0)
    skel = (skel > 0).astype(np.uint8) * 255

    # redraw with a thick stroke
    lo, hi = style["thickness"]
    t = int(rng.integers(lo, hi + 1))
    out = cv2.dilate(skel, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (t, t)))

    # sometimes add a connecting line (you can't lift the "pen" in the air)
    if rng.random() < style["joins"]:
        ys, xs = np.nonzero(skel)
        i, j = rng.integers(0, len(ys), size=2)
        cv2.line(out, (int(xs[i]), int(ys[i])), (int(xs[j]), int(ys[j])),
                 255, max(2, t // 2), cv2.LINE_AA)

    # hand tremor
    out = elastic_deform(out, rng, alpha=style["tremor"] * 8, sigma=12.0)

    # crop like the whiteboard does
    ys, xs = np.nonzero(out)
    m = config.CROP_MARGIN
    out = np.pad(out, m)
    return out[ys.min():ys.max() + 2 * m + 1, xs.min():xs.max() + 2 * m + 1]


def make_synthetic_airwrite(per_writer=100, seed=config.SEED):
    """Synthetic dataset in the same format as load_airwrite(), from MNIST test."""
    rng = np.random.default_rng(seed)
    mnist_x, mnist_y = load_mnist(train=False)
    images, labels, writers = [], [], []
    for name, style in _SYNTHETIC_WRITERS.items():
        for k in range(per_writer):
            digit = k % 10                       # same count for every digit
            pool = np.flatnonzero(mnist_y == digit)
            idx = int(rng.choice(pool))
            images.append(_air_stroke(mnist_x[idx], rng, style))
            labels.append(digit)
            writers.append(name)
    return images, np.array(labels, dtype=np.int64), np.array(writers)


def load_target(synthetic: bool):
    """Our data, or the synthetic one if synthetic=True."""
    return make_synthetic_airwrite() if synthetic else load_airwrite()
