"""
All experiments on our data. Needs the checkpoints from train.py.

Usage:
    python ablation.py                 # our data (data/raw/)
    python ablation.py --synthetic     # synthetic data, for testing;
                                       # outputs go to results/synthetic/

Experiments (each for every seed in config.SEEDS):

    1. Zero-shot: MNIST models on all our data, for each preprocessing
       variant. "minimal" gives the raw domain gap.
           minimal     resize + normalize only
           thickness   + stroke thickness normalization
           centering   + center of mass
           full        both
       Models: logistic regression, CNN, CNN with augmentation.
       No retraining, only the test-time preprocessing changes.

    2. Fine-tuning: stratified 70/30 split of our data, CNN tested on the
       30% before and after fine-tuning on the 70%.

    3. Leave-one-writer-out: fine-tune on 3 writers, test on the 4th.

Outputs (results/):
    ablation_runs.csv, ablation_summary.csv     experiments 1 and 2
    lowo_runs.csv, lowo_summary.csv             experiment 3
    per_class_f1.csv                            CNN, per variant
plus confusion matrices and misclassified examples in figures/.
"""

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config
import plotting
from data_utils import load_target
from evaluate import compute_metrics, save_confusion_matrix, summarise
from model import load_model, predict_batch
from preprocessing import preprocess_many
from train import fine_tune, set_seed

# preprocessing variants compared in the ablation
VARIANTS = {
    "minimal": ["resize_padded", "normalize_pixels"],
    "thickness": ["binarize", "normalize_thickness", "resize_padded",
                  "normalize_pixels"],
    "centering": ["resize_padded", "center_by_mass", "normalize_pixels"],
    "full": ["binarize", "normalize_thickness", "resize_padded",
             "center_by_mass", "normalize_pixels"],
}

# variant used for fine-tuning and LOWO
FINETUNE_VARIANT = "full"


def logreg_predict(seed, images):
    """Logistic regression predictions, None if there is no model file."""
    path = config.MODELS_DIR / f"logreg_seed{seed}.joblib"
    if not path.exists():
        return None
    clf = joblib.load(path)
    return clf.predict(images.reshape(len(images), -1))


def zero_shot(prepared, labels, seeds):
    """Experiment 1: every model x every variant."""
    rows = []
    for seed in seeds:
        cnn = load_model("mnist", seed)
        aug = load_model("aug", seed)
        for variant, x in prepared.items():
            for name, predictor in (
                ("logreg", lambda x: logreg_predict(seed, x)),
                ("cnn", lambda x: predict_batch(cnn, x)[0]),
                ("aug", lambda x: predict_batch(aug, x)[0]),
            ):
                preds = predictor(x)
                if preds is None:
                    continue
                rows.append({"experiment": "zero_shot", "model": name,
                             "variant": variant, "stage": "-", "seed": seed,
                             **compute_metrics(labels, preds)})
        print(f"  zero-shot seed {seed} done")
    return rows


def finetune_split(x, labels, seeds):
    """Experiment 2: 70/30 split, CNN before/after fine-tuning."""
    rows = []
    for seed in seeds:
        idx_tr, idx_te = train_test_split(
            np.arange(len(labels)), train_size=config.FINETUNE_FRACTION,
            stratify=labels, random_state=seed)
        for tag, name in (("mnist", "cnn"), ("aug", "aug")):
            model = load_model(tag, seed)
            before = predict_batch(model, x[idx_te])[0]
            rows.append({"experiment": "finetune", "model": name,
                         "variant": FINETUNE_VARIANT, "stage": "before",
                         "seed": seed, **compute_metrics(labels[idx_te], before)})
            fine_tune(model, x[idx_tr], labels[idx_tr], seed)
            after = predict_batch(model, x[idx_te])[0]
            rows.append({"experiment": "finetune", "model": name,
                         "variant": FINETUNE_VARIANT, "stage": "after",
                         "seed": seed, **compute_metrics(labels[idx_te], after)})
            if seed == config.SEED and name == "cnn":
                save_confusion_matrix(
                    labels[idx_te], after,
                    "CNN after fine-tuning, our data (30% test)",
                    config.FIGURES_DIR / "confusion_ours_finetuned")
        print(f"  fine-tuning seed {seed} done")
    return rows


def leave_one_writer_out(x, labels, writers, seeds):
    """Experiment 3: fine-tune on all writers except one, test on that one."""
    rows = []
    for held_out in sorted(set(writers)):
        test = writers == held_out
        for seed in seeds:
            model = load_model("mnist", seed)
            before = predict_batch(model, x[test])[0]
            rows.append({"writer": held_out, "stage": "before", "seed": seed,
                         **compute_metrics(labels[test], before)})
            fine_tune(model, x[~test], labels[~test], seed)
            after = predict_batch(model, x[test])[0]
            rows.append({"writer": held_out, "stage": "after", "seed": seed,
                         **compute_metrics(labels[test], after)})
        print(f"  held out {held_out} done")
    return rows


def misclassified_gallery(raw, prepared, labels, preds, path, title, limit=30):
    """Grid of wrong predictions, captioned 'true -> predicted'."""
    wrong = np.flatnonzero(preds != labels)[:limit]
    images = [prepared[i] for i in wrong]
    captions = [f"{labels[i]} -> {preds[i]}" for i in wrong]
    return plotting.gallery(images, captions, path, title)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--synthetic", action="store_true",
                        help="use synthetic data; outputs go to */synthetic/")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=list(config.SEEDS))
    parser.add_argument("--quick", action="store_true",
                        help="use the checkpoints from `train.py --quick`")
    parser.add_argument("--skip-lowo", action="store_true")
    args = parser.parse_args()

    if args.quick:
        config.redirect_outputs("smoke")
    if args.synthetic:
        # models are trained on MNIST anyway, so only move results/figures
        sub = "smoke/synthetic" if args.quick else "synthetic"
        models_dir = config.MODELS_DIR
        config.redirect_outputs(sub, models=False)
        config.MODELS_DIR = models_dir
        print("*** SYNTHETIC STAND-IN DATA - not results on our dataset ***")

    set_seed(config.SEED)
    raw, labels, writers = load_target(args.synthetic)
    print(f"{len(raw)} images from {len(set(writers))} writers: "
          f"{dict(zip(*np.unique(writers, return_counts=True)))}")

    prepared = {name: preprocess_many(raw, steps)
                for name, steps in VARIANTS.items()}

    print("experiment 1: zero-shot ablation")
    rows = zero_shot(prepared, labels, args.seeds)
    print("experiment 2: fine-tuning on 70% of our data")
    rows += finetune_split(prepared[FINETUNE_VARIANT], labels, args.seeds)

    runs = pd.DataFrame(rows)
    runs.to_csv(config.RESULTS_DIR / "ablation_runs.csv", index=False)
    summary = summarise(runs, ["experiment", "model", "variant", "stage"])
    summary.to_csv(config.RESULTS_DIR / "ablation_summary.csv", index=False)
    print(summary.to_string(index=False))

    # per-class F1 of the CNN
    per_class = runs[(runs.experiment == "zero_shot") & (runs.model == "cnn")]
    f1_cols = [f"f1_{d}" for d in range(10)]
    per_class.groupby("variant", sort=False)[f1_cols].mean().to_csv(
        config.RESULTS_DIR / "per_class_f1.csv")

    # confusion matrices + mistakes, default seed only
    cnn = load_model("mnist", config.SEED)
    for variant in ("minimal", "full"):
        preds = predict_batch(cnn, prepared[variant])[0]
        save_confusion_matrix(labels, preds,
                              f"CNN on our data, {variant} preprocessing",
                              config.FIGURES_DIR / f"confusion_ours_{variant}")
        misclassified_gallery(raw, prepared[variant], labels, preds,
                              config.FIGURES_DIR / f"misclassified_{variant}",
                              f"Misclassified ({variant} preprocessing): "
                              "true -> predicted")

    if not args.skip_lowo and len(set(writers)) >= 2:
        print("experiment 3: leave-one-writer-out")
        lowo = pd.DataFrame(leave_one_writer_out(
            prepared[FINETUNE_VARIANT], labels, writers, args.seeds))
        lowo.to_csv(config.RESULTS_DIR / "lowo_runs.csv", index=False)
        lowo_summary = summarise(lowo, ["writer", "stage"])
        lowo_summary.to_csv(config.RESULTS_DIR / "lowo_summary.csv",
                            index=False)
        print(lowo_summary.to_string(index=False))

    print(f"\nresults written to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
