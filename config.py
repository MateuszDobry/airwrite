"""Constants and paths shared by all scripts."""

from pathlib import Path

# Seeds. Every experiment runs once per seed in SEEDS and we report
# mean +/- std. SEED is used where a single run is enough (figures, demo).
SEED = 42
SEEDS = (42, 43, 44)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_RAW = PROJECT_ROOT / "data" / "raw"          # our collected digits
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_MNIST = PROJECT_ROOT / "data" / "mnist"      # torchvision downloads here
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"

for _d in (DATA_RAW, DATA_PROCESSED, DATA_MNIST, MODELS_DIR, FIGURES_DIR,
           RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Image format: 28x28, white digit on black, same as MNIST.
IMG_SIZE = 28
# MNIST digits are scaled to fit a 20x20 box and then padded to 28x28.
DIGIT_BOX = 20

# MNIST pixel mean/std (pixels scaled to [0, 1])
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

# Training
EPOCHS = 8
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
VAL_SIZE = 6000           # 54k train / 6k val

FINETUNE_FRACTION = 0.7   # part of our data used for fine-tuning
FINETUNE_EPOCHS = 15
FINETUNE_LR = 1e-4

# Webcam / whiteboard
CAMERA_INDEX = 0          # try 1 if the wrong camera opens
FRAME_WIDTH = 960
FRAME_HEIGHT = 540

STROKE_THICKNESS = 12     # px at camera resolution
SMOOTHING_WINDOW = 5      # fingertip positions to average
CROP_MARGIN = 20          # px around the digit when cropping

# MediaPipe hand landmark ids
WRIST = 0
THUMB_TIP, THUMB_IP = 4, 3
INDEX_TIP, INDEX_PIP, INDEX_MCP = 8, 6, 5
MIDDLE_TIP, MIDDLE_PIP = 12, 10
RING_TIP, RING_PIP = 16, 14
PINKY_TIP, PINKY_PIP, PINKY_MCP = 20, 18, 17


def redirect_outputs(subdir: str, models: bool = True):
    """
    Write results/figures (and models, unless models=False) to a subfolder.

    Used by --quick and --synthetic so test runs don't overwrite the real
    results. Other modules read config.RESULTS_DIR etc. at call time, so
    changing them here is enough.
    """
    global MODELS_DIR, FIGURES_DIR, RESULTS_DIR
    FIGURES_DIR = PROJECT_ROOT / "figures" / subdir
    RESULTS_DIR = PROJECT_ROOT / "results" / subdir
    if models:
        MODELS_DIR = PROJECT_ROOT / "models" / subdir
    for d in (MODELS_DIR, FIGURES_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
