"""CNN model, saving/loading checkpoints and prediction."""

from pathlib import Path

import numpy as np
import torch
from torch import nn

import config


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _block(c_in, c_out):
    """conv -> batchnorm -> relu -> maxpool (halves the size)"""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class DigitCNN(nn.Module):
    """
    Small CNN for 28x28 digits, ~94k parameters.

    Three conv blocks (28 -> 14 -> 7 -> 3), dropout, one linear layer.
    We kept it small because the project is about the data, and it has to
    be trained 3 times (3 seeds).
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
        # add the channel dim if input is (N, 28, 28)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.classifier(self.features(x))


def checkpoint_path(tag: str, seed: int) -> Path:
    """e.g. models/cnn_mnist_seed42.pt"""
    return config.MODELS_DIR / f"cnn_{tag}_seed{seed}.pt"


def save_model(model, tag, seed):
    path = checkpoint_path(tag, seed)
    torch.save(model.state_dict(), path)
    return path


def load_model(tag="mnist", seed=config.SEED, device=None):
    """Load a checkpoint and put the model in eval mode."""
    device = device or get_device()
    path = checkpoint_path(tag, seed)
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found. Run `python train.py` first.")
    model = DigitCNN().to(device)
    model.load_state_dict(torch.load(path, map_location=device,
                                     weights_only=True))
    model.eval()
    return model


@torch.no_grad()
def predict_batch(model, images: np.ndarray, batch_size=512):
    """
    Predict a batch of preprocessed images, float32 (N, 28, 28).

    Returns (predictions, probabilities) with shapes (N,) and (N, 10).
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
    Predict one preprocessed 28x28 image with the default MNIST model.

    Returns (digit, confidence). The model is loaded only on the first call.
    """
    global _cached_model
    if _cached_model is None:
        _cached_model = load_model("mnist", config.SEED)
    preds, probs = predict_batch(_cached_model, img[None])
    return int(preds[0]), float(probs[0, preds[0]])
