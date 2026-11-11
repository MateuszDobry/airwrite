"""
CNN architecture, checkpoint helpers and the predict() entry point.

Owner: person B.

The network is deliberately small. The study is about data, not about
architecture, and a small network keeps all training runs cheap enough to
repeat with three seeds.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn

import config


def get_device():
    """Use the GPU when there is one (e.g. on Colab), otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _block(c_in, c_out):
    """Conv -> BatchNorm -> ReLU -> MaxPool: halves the spatial size."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class DigitCNN(nn.Module):
    """
    Compact CNN for 28x28 grayscale digits.

    3 x (Conv -> BatchNorm -> ReLU -> MaxPool), dropout, one dense layer.
    Spatial size: 28 -> 14 -> 7 -> 3, so the dense layer sees 128*3*3
    features. About 94k parameters.
    """

    def __init__(self, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            _block(1, 32),
            _block(32, 64),
            _block(64, 128),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128 * 3 * 3, 10),
        )

    def forward(self, x):
        # Accept (N, 28, 28) as well as (N, 1, 28, 28).
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.classifier(self.features(x))


# --------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------

def checkpoint_path(tag: str, seed: int) -> Path:
    """models/cnn_<tag>_seed<seed>.pt, e.g. models/cnn_mnist_seed42.pt"""
    return config.MODELS_DIR / f"cnn_{tag}_seed{seed}.pt"


def save_model(model, tag, seed):
    """Save the weights to models/cnn_<tag>_seed<seed>.pt and return the path."""
    path = checkpoint_path(tag, seed)
    torch.save(model.state_dict(), path)
    return path


def load_model(tag="mnist", seed=config.SEED, device=None):
    """Load a trained checkpoint in evaluation mode."""
    device = device or get_device()
    path = checkpoint_path(tag, seed)
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found. Run `python train.py` first.")
    model = DigitCNN().to(device)
    # weights_only=True: the file holds only tensors, so no pickled code runs.
    model.load_state_dict(torch.load(path, map_location=device,
                                     weights_only=True))
    model.eval()
    return model


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------

@torch.no_grad()
def predict_batch(model, images: np.ndarray, batch_size=512):
    """
    Classify many preprocessed images at once.

    Args:
        images: float32 (N, 28, 28), output of preprocessing.preprocess().

    Returns:
        (predictions, probabilities): int64 (N,) and float32 (N, 10).
    """
    device = next(model.parameters()).device
    model.eval()
    probs = []
    for i in range(0, len(images), batch_size):
        x = torch.from_numpy(np.asarray(images[i:i + batch_size],
                                        dtype=np.float32)).to(device)
        probs.append(torch.softmax(model(x), dim=1).cpu().numpy())
    probs = np.concatenate(probs) if probs else np.zeros((0, 10), np.float32)
    return probs.argmax(axis=1), probs


_cached_model = None


def predict(img: np.ndarray) -> tuple[int, float]:
    """
    Classify one preprocessed image with the default MNIST checkpoint.

    Args:
        img: float32 array of shape (28, 28), already run through
             preprocessing.preprocess().

    Returns:
        (digit, confidence) with digit in 0..9 and confidence in [0, 1].

    The model is loaded once and cached, so this is cheap enough to call on
    every frame of the live whiteboard demo.
    """
    global _cached_model
    if _cached_model is None:
        _cached_model = load_model("mnist", config.SEED)
    preds, probs = predict_batch(_cached_model, img[None])
    return int(preds[0]), float(probs[0, preds[0]])
