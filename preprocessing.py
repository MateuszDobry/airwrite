"""
Preprocessing: raw air drawing -> 28x28 MNIST-like image.

Each step can be turned on/off separately, because the ablation needs to
measure them one by one. preprocess() gets a list of step names and always
runs them in the order of ALL_STEPS.

augment() (training-time augmentation) is also here.
"""

import cv2
import numpy as np

import config

# Available steps, in the order they are applied.
ALL_STEPS = [
    "binarize",             # Otsu threshold
    "normalize_thickness",  # make the stroke width relative to digit size
    "resize_padded",        # fit into 20x20 keeping aspect ratio, pad to 28x28
    "center_by_mass",       # move center of mass to the middle
    "normalize_pixels",     # MNIST mean/std
]

# Used to measure the raw domain gap, so no mitigation steps here.
# normalize_pixels stays because the network was trained on normalized input.
MINIMAL_STEPS = ["resize_padded", "normalize_pixels"]

FULL_STEPS = list(ALL_STEPS)

# Target stroke width as a fraction of the longer side of the bounding box.
# MNIST strokes are about 2-3 px in a 20 px box, so ~0.12.
TARGET_STROKE_RATIO = 0.12


# Steps. All take and return uint8 images (white on black), except
# normalize_pixels which returns float32.

def binarize(img):
    _, out = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return out


def _bounding_box(img):
    """(y0, y1, x0, x1) of non-zero pixels, None if the image is empty."""
    ys, xs = np.nonzero(img)
    if len(ys) == 0:
        return None
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


def skeletonize(img):
    """
    Morphological skeleton (Lantuejoul), OpenCV only.

    Erode repeatedly and collect what an opening removes at each step.
    Gives a 1 px wide center line.
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
    Stroke width in pixels.

    On the skeleton, the distance transform equals half of the local
    stroke width, so we take 2 * median.
    """
    binary = (img > 127).astype(np.uint8) * 255
    if not binary.any():
        return 0.0
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    centre = skeletonize(binary) > 0
    return float(2 * np.median(dist[centre])) if centre.any() else 0.0


def normalize_thickness(img):
    """
    Erode or dilate so the stroke width is TARGET_STROKE_RATIO of the digit
    size.

    We erode/dilate the original shape instead of redrawing the skeleton,
    because the skeleton has small spurs that turn into blobs when made
    thicker.
    """
    box = _bounding_box(img)
    if box is None:
        return img
    y0, y1, x0, x1 = box
    longest = max(y1 - y0, x1 - x0)

    binary = (img > 127).astype(np.uint8) * 255
    current = estimate_stroke_width(binary)
    target = TARGET_STROKE_RATIO * longest
    diff = int(round(current - target))         # > 0 means too thick
    if abs(diff) < 2:
        return binary

    size = abs(diff) + 1
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    if diff > 0:
        thinner = cv2.erode(binary, disc)
        # thin parts of the digit can disappear completely, so if we lose
        # too much ink use a smaller kernel
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

    Keeps the aspect ratio (otherwise a "1" gets stretched into a block).
    INTER_AREA gives soft edges similar to MNIST.
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

    # center the bounding box (center_by_mass can fix it later)
    out = np.zeros((size, size), dtype=np.uint8)
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    out[top:top + new_h, left:left + new_w] = small
    return out


def center_by_mass(img):
    """
    Shift the image so the center of mass is in the middle.

    Same as in the original MNIST (LeCun et al., 1998).
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
    """Scale to [0, 1] and normalize with MNIST mean/std."""
    x = img.astype(np.float32) / 255.0
    return (x - config.MNIST_MEAN) / config.MNIST_STD


_STEP_FUNCTIONS = {
    "binarize": binarize,
    "normalize_thickness": normalize_thickness,
    "resize_padded": resize_padded,
    "center_by_mass": center_by_mass,
    "normalize_pixels": normalize_pixels,
}


def preprocess(img: np.ndarray, steps: list[str]) -> np.ndarray:
    """
    Run the selected steps on a raw drawing.

    img: grayscale uint8, any size, white on black (as saved by
         collect_data.py).
    steps: subset of ALL_STEPS, order doesn't matter.

    Returns float32 28x28 - normalized if normalize_pixels is on, otherwise
    in [0, 1].
    """
    unknown = set(steps) - set(ALL_STEPS)
    if unknown:
        raise ValueError(f"unknown preprocessing steps: {sorted(unknown)}")

    out = img
    if out.ndim == 3:                       # in case of a color png
        out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    out = out.astype(np.uint8)

    for name in ALL_STEPS:
        if name == "normalize_pixels":
            continue                        # done at the end
        if name in steps:
            out = _STEP_FUNCTIONS[name](out)

    # without resize_padded the image is just squashed to 28x28
    if out.shape != (config.IMG_SIZE, config.IMG_SIZE):
        out = cv2.resize(out, (config.IMG_SIZE, config.IMG_SIZE),
                         interpolation=cv2.INTER_AREA)

    if "normalize_pixels" in steps:
        return normalize_pixels(out)
    return out.astype(np.float32) / 255.0


def preprocess_many(images, steps):
    """preprocess() for a list of images -> (N, 28, 28)"""
    return np.stack([preprocess(im, steps) for im in images]).astype(np.float32)


# Augmentation (training only)

def elastic_deform(img, rng, alpha=34.0, sigma=4.0):
    """
    Elastic distortion (Simard et al., 2003).

    Random displacement field, smoothed with a Gaussian and scaled by alpha.
    Simulates small wobbles of the stroke.
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
    Random augmentation of one uint8 28x28 MNIST image.

    dilation   - thicker strokes
    rotation   - tilted hand
    shift/scale - digit not placed precisely
    elastic    - tremor
    """
    out = img

    # thicker strokes, 50% of the time
    if rng.random() < 0.5:
        k = int(rng.integers(2, 4))                 # 2x2 or 3x3
        out = cv2.dilate(out, np.ones((k, k), np.uint8))

    # rotation +-10 deg, scale 0.9-1.1, shift +-2 px
    angle = rng.uniform(-10, 10)
    scale = rng.uniform(0.9, 1.1)
    centre = (config.IMG_SIZE / 2, config.IMG_SIZE / 2)
    matrix = cv2.getRotationMatrix2D(centre, angle, scale)
    matrix[:, 2] += rng.uniform(-2, 2, size=2)
    out = cv2.warpAffine(out, matrix, (config.IMG_SIZE, config.IMG_SIZE),
                         flags=cv2.INTER_LINEAR, borderValue=0)

    # elastic, 50% of the time
    if rng.random() < 0.5:
        out = elastic_deform(out, rng)

    return out
