"""
Training on MNIST: logistic regression baseline, CNN, and CNN with
augmentation. fine_tune() is used later by ablation.py.

Usage:
    python train.py                         # all models, 3 seeds
    python train.py --models cnn --seeds 42 # one model, one seed
    python train.py --quick                 # quick test on CPU (~1 min)

Outputs:
    models/cnn_mnist_seed<s>.pt, models/cnn_aug_seed<s>.pt,
    models/logreg_seed<s>.joblib
    results/history_<tag>_seed<s>.csv     learning curves
    results/mnist_test_runs.csv           MNIST test metrics per run
"""

import argparse
import random
import time

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import config
from data_utils import load_mnist
from evaluate import compute_metrics
from model import DigitCNN, get_device, predict_batch, save_model
from preprocessing import augment


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic cuDNN - a bit slower but runs are repeatable
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def standardise(images_uint8):
    """uint8 -> float32 normalized with MNIST mean/std"""
    x = images_uint8.astype(np.float32) / 255.0
    return (x - config.MNIST_MEAN) / config.MNIST_STD


def split_train_val(images, labels, seed, val_size=config.VAL_SIZE):
    """Random 54k/6k split of MNIST train."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(images))
    val, train = order[:val_size], order[val_size:]
    return images[train], labels[train], images[val], labels[val]


class DigitDataset(Dataset):
    """
    MNIST images, optionally augmented.

    Augmentation is applied on the fly to the uint8 image (before
    normalization), so each epoch gets different distortions.
    """

    def __init__(self, images_uint8, labels, augment_seed=None):
        self.images = images_uint8
        self.labels = labels
        # one rng for the whole dataset; with num_workers=0 this is
        # reproducible
        self.rng = (np.random.default_rng(augment_seed)
                    if augment_seed is not None else None)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = self.images[i]
        if self.rng is not None:
            img = augment(img, self.rng)
        x = standardise(img)
        return torch.from_numpy(x), int(self.labels[i])


class TensorDigits(Dataset):
    """Already preprocessed float32 images (for fine-tuning)."""

    def __init__(self, images_float, labels):
        self.x = torch.from_numpy(np.asarray(images_float, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(labels, dtype=np.int64))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def run_epoch(model, loader, loss_fn, optimiser=None):
    """One epoch. Trains if optimiser is given, otherwise just evaluates.
    Returns (mean loss, accuracy)."""
    device = next(model.parameters()).device
    training = optimiser is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            if training:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
    return total_loss / seen, correct / seen


def fit(model, train_loader, val_loader, epochs, lr, log_name=None):
    """Adam + cosine LR schedule. Returns history (one row per epoch)."""
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, epochs)
    history = []

    for epoch in range(1, epochs + 1):
        start = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, loss_fn, optimiser)
        scheduler.step()
        row = {"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc}
        if val_loader is not None:
            row["val_loss"], row["val_acc"] = run_epoch(model, val_loader,
                                                        loss_fn)
        row["seconds"] = time.time() - start
        history.append(row)

        if log_name:
            val = (f" | val loss {row['val_loss']:.4f} "
                   f"acc {row['val_acc']:.4f}" if val_loader else "")
            print(f"  [{log_name}] epoch {epoch}/{epochs}  "
                  f"train loss {tr_loss:.4f} acc {tr_acc:.4f}{val}  "
                  f"({row['seconds']:.1f}s)")
    return pd.DataFrame(history)


def train_cnn(seed, augmented, epochs, train_subset=None):
    """
    Train one CNN on MNIST. train_subset limits the number of training
    images (for --quick). Returns (model, history).
    """
    set_seed(seed)
    images, labels = load_mnist(train=True)
    x_tr, y_tr, x_val, y_val = split_train_val(images, labels, seed)
    if train_subset:
        x_tr, y_tr = x_tr[:train_subset], y_tr[:train_subset]

    device = get_device()
    generator = torch.Generator().manual_seed(seed)   # shuffle order
    train_loader = DataLoader(
        DigitDataset(x_tr, y_tr, augment_seed=seed if augmented else None),
        batch_size=config.BATCH_SIZE, shuffle=True, generator=generator)
    val_loader = DataLoader(DigitDataset(x_val, y_val),
                            batch_size=1024)

    model = DigitCNN().to(device)
    tag = "aug" if augmented else "mnist"
    history = fit(model, train_loader, val_loader, epochs,
                  config.LEARNING_RATE, log_name=f"{tag} seed {seed}")
    return model, history


def train_logreg(seed, train_subset=None):
    """Logistic regression on raw normalized pixels (baseline)."""
    from sklearn.linear_model import LogisticRegression

    images, labels = load_mnist(train=True)
    x_tr, y_tr, _, _ = split_train_val(images, labels, seed)
    if train_subset:
        x_tr, y_tr = x_tr[:train_subset], y_tr[:train_subset]
    clf = LogisticRegression(max_iter=200, random_state=seed)
    clf.fit(standardise(x_tr).reshape(len(x_tr), -1), y_tr)
    return clf


def fine_tune(model, images, labels, seed, epochs=config.FINETUNE_EPOCHS,
              lr=config.FINETUNE_LR):
    """
    Continue training on our own (preprocessed) digits.

    LR is 10x smaller than for MNIST - we only have a few hundred images,
    so we want to adjust the model, not retrain it.
    """
    set_seed(seed)
    loader = DataLoader(TensorDigits(images, labels), batch_size=32,
                        shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    fit(model, loader, None, epochs, lr)
    model.eval()
    return model


def upsert_rows(path, new_rows, key_cols):
    """
    Append rows to a CSV, replacing old rows with the same key.

    So you can rerun one model or seed without losing the rest.
    """
    new = pd.DataFrame(new_rows)
    if path.exists():
        old = pd.read_csv(path)
        keys = set(map(tuple, new[key_cols].astype(str).values))
        keep = [tuple(r) not in keys
                for r in old[key_cols].astype(str).values]
        new = pd.concat([old[keep], new], ignore_index=True)
    new.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["logreg", "cnn", "aug"],
                        choices=["logreg", "cnn", "aug"])
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=list(config.SEEDS))
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--quick", action="store_true",
                        help="quick test: 5k images, 1 epoch, outputs go to "
                             "*/smoke/")
    args = parser.parse_args()

    subset = None
    if args.quick:
        config.redirect_outputs("smoke")
        subset, args.epochs = 5000, 1

    print(f"device: {get_device()}")
    test_x, test_y = load_mnist(train=False)
    test_std = standardise(test_x)
    rows = []

    for seed in args.seeds:
        if "logreg" in args.models:
            start = time.time()
            clf = train_logreg(seed, subset)
            joblib.dump(clf, config.MODELS_DIR / f"logreg_seed{seed}.joblib")
            preds = clf.predict(test_std.reshape(len(test_std), -1))
            rows.append({"model": "logreg", "seed": seed,
                         "train_seconds": time.time() - start,
                         **compute_metrics(test_y, preds)})
            print(f"logreg seed {seed}: test acc {rows[-1]['accuracy']:.4f}")

        for name in ("cnn", "aug"):
            if name not in args.models:
                continue
            start = time.time()
            model, history = train_cnn(seed, augmented=(name == "aug"),
                                       epochs=args.epochs,
                                       train_subset=subset)
            tag = "aug" if name == "aug" else "mnist"
            save_model(model, tag, seed)
            history.to_csv(config.RESULTS_DIR / f"history_{tag}_seed{seed}.csv",
                           index=False)
            preds, _ = predict_batch(model, test_std)
            rows.append({"model": name, "seed": seed,
                         "train_seconds": time.time() - start,
                         **compute_metrics(test_y, preds)})
            print(f"{name} seed {seed}: test acc {rows[-1]['accuracy']:.4f}")

    out = config.RESULTS_DIR / "mnist_test_runs.csv"
    upsert_rows(out, rows, ["model", "seed"])
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
