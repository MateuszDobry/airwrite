"""
Preprocessing pipeline: raw in-air drawing -> MNIST-like 28x28 image.

Owner: Kirill Riabkov (person C).

The design requirement that matters: every step is switchable independently.
The ablation study measures what each step contributes on its own, which is
impossible if the steps are fused into one monolithic function. preprocess()
therefore takes a list of step names, and the order in which enabled steps
run is fixed here so that results never depend on how the list was written.

The module also contains augment(), the training-time distortion used to
test whether the domain gap can be closed without seeing any target data.
"""

import cv2
import numpy as np

import config

# Step names accepted by preprocess(), in the order they are applied.
ALL_STEPS = [
    "binarize",             # threshold to pure black and white
    "normalize_thickness",  # skeletonise, then redraw at a constant width
    "resize_padded",        # scale to 20x20 preserving aspect, pad to 28x28
    "center_by_mass",       # move the centre of mass to the image centre
    "normalize_pixels",     # standardise with MNIST mean / std
]

# Minimal pipeline used to measure the raw domain gap. Anything more would
# already be a mitigation and would understate the gap. normalize_pixels is
# not a mitigation: the network was trained on standardised inputs, so
# leaving it out would only measure a scaling bug.
MINIMAL_STEPS = ["resize_padded", "normalize_pixels"]

# Every step enabled.
FULL_STEPS = list(ALL_STEPS)

# Stroke width after normalisation, as a fraction of the larger side of the
# digit's bounding box. MNIST strokes are roughly 2-3 px in a 20 px box,
# i.e. about 12 % of the box; this value reproduces that after resizing.
TARGET_STROKE_RATIO = 0.12


# --------------------------------------------------------------------------
# Individual steps. Each takes and returns a uint8 image, white on black,
# except normalize_pixels which produces the final float32 array.
# --------------------------------------------------------------------------

def binarize(img):
    """Threshold with Otsu's method, which picks the cut-off automatically."""
    _, out = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return out


def _bounding_box(img):
    """(y0, y1, x0, x1) of all non-zero pixels, or None for an empty image."""
    ys, xs = np.nonzero(img)
    if len(ys) == 0:
        return None
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


def skeletonize(img):
    """
    Morphological skeleton (Lantuejoul's method) using only OpenCV.

    Repeatedly erodes the shape; at every step the pixels that an opening
    would remove are part of the skeleton. The result is a one-pixel-wide
    centre line that no longer carries any information about the original
    stroke width - which is exactly what we want to normalise away.
    """
    binary = (img > 127).astype(np.uint8) * 255
    skeleton = np.zeros_like(binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    current = binary.copy()
    while current.any():
        eroded = cv2.erode(current, kernel)
        opened = cv2.dilate(eroded, kernel)
        skeleton |= cv2.subtract(current, opened)
        current = eroded
    return skeleton


def estimate_stroke_width(img):
    """
    Typical stroke width of a binary drawing, in pixels.

    The distance transform gives, for every ink pixel, the distance to the
    nearest background pixel. On the skeleton - the centre line of the
    stroke - that distance is half of the local stroke width, so twice its
    median is a robust estimate of the overall width.
    """
    binary = (img > 127).astype(np.uint8) * 255
    if not binary.any():
        return 0.0
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    centre = skeletonize(binary) > 0
    return float(2 * np.median(dist[centre])) if centre.any() else 0.0


def normalize_thickness(img):
    """
    Give every drawing the same relative stroke width.

    In-air strokes are thick and uneven (the finger speeds up and slows
    down). We measure the current width and then erode or dilate the drawing
    so that it becomes a fixed fraction of the digit's size; after resizing
    to 20x20 this matches typical MNIST strokes.

    Eroding or dilating the existing shape (rather than redrawing its
    skeleton) keeps the outline smooth: a morphological skeleton has small
    spurs that would turn into visible blobs when thickened again.
    """
    box = _bounding_box(img)
    if box is None:
        return img
    y0, y1, x0, x1 = box
    longest = max(y1 - y0, x1 - x0)

    binary = (img > 127).astype(np.uint8) * 255
    current = estimate_stroke_width(binary)
    target = TARGET_STROKE_RATIO * longest
    diff = int(round(current - target))         # > 0: too thick
    if abs(diff) < 2:
        return binary                           # already close enough

    size = abs(diff) + 1
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    if diff > 0:
        thinner = cv2.erode(binary, disc)
        # Erosion can wipe out parts of the digit that were drawn thinner
        # than the rest. If too much ink disappears, thin more gently.
        while (cv2.countNonZero(thinner) < 0.25 * cv2.countNonZero(binary)
               and size > 2):
            size -= 2
            disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            thinner = cv2.erode(binary, disc)
        return thinner
    return cv2.dilate(binary, disc)


def resize_padded(img):
    """
    Crop to the digit, scale the longer side to 20 px, paste into 28x28.

    Preserving the aspect ratio matters: a "1" stretched to a square would
    look nothing like an MNIST "1". INTER_AREA is used because it averages
    pixels when shrinking, producing the soft anti-aliased edges that MNIST
    digits also have.
    """
    size, box_size = config.IMG_SIZE, config.DIGIT_BOX
    box = _bounding_box(img)
    if box is None:
        return np.zeros((size, size), dtype=np.uint8)
    y0, y1, x0, x1 = box
    crop = img[y0:y1, x0:x1]

    h, w = crop.shape
    scale = box_size / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    small = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Centre the bounding box geometrically. center_by_mass, if enabled,
    # refines this afterwards.
    out = np.zeros((size, size), dtype=np.uint8)
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    out[top:top + new_h, left:left + new_w] = small
    return out


def center_by_mass(img):
    """
    Shift the image so that its centre of mass sits at the image centre.

    This is the second half of the original MNIST normalisation procedure
    (LeCun et al., 1998). Without it, a digit with a long tail (a "7" or a
    "9") sits off-centre compared with the training data.
    """
    total = img.sum()
    if total == 0:
        return img
    ys, xs = np.indices(img.shape)
    cy = (ys * img).sum() / total
    cx = (xs * img).sum() / total
    centre = (img.shape[0] - 1) / 2
    shift = np.float32([[1, 0, centre - cx], [0, 1, centre - cy]])
    return cv2.warpAffine(img, shift, (img.shape[1], img.shape[0]),
                          flags=cv2.INTER_LINEAR, borderValue=0)


def normalize_pixels(img):
    """Scale to [0, 1] and standardise exactly like the training data."""
    x = img.astype(np.float32) / 255.0
    return (x - config.MNIST_MEAN) / config.MNIST_STD


# --------------------------------------------------------------------------
# The configurable pipeline
# --------------------------------------------------------------------------

_STEP_FUNCTIONS = {
    "binarize": binarize,
    "normalize_thickness": normalize_thickness,
    "resize_padded": resize_padded,
    "center_by_mass": center_by_mass,
    "normalize_pixels": normalize_pixels,
}


def preprocess(img: np.ndarray, steps: list[str]) -> np.ndarray:
    """
    Convert a raw drawing into a model-ready image.

    Args:
        img: grayscale uint8 array of arbitrary size, white strokes on black,
             exactly as written by collect_data.py.
        steps: subset of ALL_STEPS to apply. Order in the list is ignored;
               enabled steps always run in the order of ALL_STEPS.

    Returns:
        float32 array of shape (IMG_SIZE, IMG_SIZE). If normalize_pixels is
        enabled the values are MNIST-standardised, otherwise in [0, 1].
    """
    unknown = set(steps) - set(ALL_STEPS)
    if unknown:
        raise ValueError(f"unknown preprocessing steps: {sorted(unknown)}")

    out = img
    if out.ndim == 3:                       # tolerate colour PNGs
        out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    out = out.astype(np.uint8)

    for name in ALL_STEPS:
        if name == "normalize_pixels":
            continue                        # applied last, below
        if name in steps:
            out = _STEP_FUNCTIONS[name](out)

    # Whatever happened above, the model needs exactly 28x28. Without
    # resize_padded the drawing is simply squashed to that size - that is
    # the "naive" baseline, and it is only used if explicitly requested.
    if out.shape != (config.IMG_SIZE, config.IMG_SIZE):
        out = cv2.resize(out, (config.IMG_SIZE, config.IMG_SIZE),
                         interpolation=cv2.INTER_AREA)

    if "normalize_pixels" in steps:
        return normalize_pixels(out)
    return out.astype(np.float32) / 255.0


def preprocess_many(images, steps):
    """Apply preprocess() to a list of raw images; returns (N, 28, 28)."""
    return np.stack([preprocess(im, steps) for im in images]).astype(np.float32)


# --------------------------------------------------------------------------
# Training-time augmentation
# --------------------------------------------------------------------------

def elastic_deform(img, rng, alpha=34.0, sigma=4.0):
    """
    Elastic distortion as described by Simard et al. (2003).

    A random displacement field is smoothed with a Gaussian and scaled by
    alpha. It imitates the small, local wobbles of a hand-drawn stroke - and
    in our case, of a trembling finger in mid-air.
    """
    h, w = img.shape
    dx = rng.uniform(-1, 1, size=(h, w)).astype(np.float32)
    dy = rng.uniform(-1, 1, size=(h, w)).astype(np.float32)
    dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    return cv2.remap(img, xs + dx, ys + dy, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Random training-time distortion: dilation, rotation, shift, elastic warp.

    Args:
        img: one MNIST image, uint8, 28x28.
        rng: numpy random generator (seeded by the caller for reproducibility).

    Returns:
        A distorted uint8 28x28 image.

    Each distortion targets one of the differences we expect between MNIST
    and in-air digits: thicker strokes (dilation), a tilted hand (rotation),
    imprecise placement (shift and scale) and tremor (elastic warp).
    """
    out = img

    # Thicker strokes, half of the time.
    if rng.random() < 0.5:
        k = int(rng.integers(2, 4))                 # 2x2 or 3x3
        out = cv2.dilate(out, np.ones((k, k), np.uint8))

    # Rotation +-10 degrees, scale 0.9-1.1, shift +-2 px, in one affine warp.
    angle = rng.uniform(-10, 10)
    scale = rng.uniform(0.9, 1.1)
    centre = (config.IMG_SIZE / 2, config.IMG_SIZE / 2)
    matrix = cv2.getRotationMatrix2D(centre, angle, scale)
    matrix[:, 2] += rng.uniform(-2, 2, size=2)
    out = cv2.warpAffine(out, matrix, (config.IMG_SIZE, config.IMG_SIZE),
                         flags=cv2.INTER_LINEAR, borderValue=0)

    # Elastic tremor, half of the time.
    if rng.random() < 0.5:
        out = elastic_deform(out, rng)

    return out
