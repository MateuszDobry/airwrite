# CLAUDE.md

Context for Claude Code working in this repository.

## Communication

The repository owner is Polish and writes in Polish. **Reply in Polish.**
Code, comments, docstrings, commit messages and the final report are in
**English**, because the course report must be in English.

## What this project is

A course project for *Programming with Python, Fall 2026* at Beijing
Institute of Technology. Group of 4 students. Hard deadline:
**13 November 2026, 23:59.**

**Title:** Writing in the Air — Measuring and Mitigating Domain Shift in
Handwritten Digit Recognition.

**Research question:** How large is the accuracy drop when an MNIST-trained
classifier is applied to digits written in the air with a finger, and which
interventions — stroke normalisation, centre-of-mass alignment, training-time
augmentation, or fine-tuning — close that gap most effectively?

The gesture whiteboard is **not the point of the project**. It is the
instrument used to collect the target-domain dataset, and the live demo for
the presentation. The deliverable that earns marks is the empirical study:
measured domain gap, ablation over mitigations, error analysis.

## Non-negotiable constraints

These come from the course brief (`docs/course-requirements.md`). Do not
quietly relax them.

- **Results must never be fabricated or altered.** Every number quoted in the
  report must be produced by code in this repo and written to `results/`.
  If an experiment was not run, the report does not claim it.
- Reproducible: all seeds fixed via `config.SEED`, library versions pinned.
- At least 100 lines of code, at least 2 AI/data libraries (PyTorch,
  scikit-learn, NumPy, pandas, Matplotlib are all used).
- Code must contain sufficient comments to be readable by a grader.
- The report must follow the exact 5-section template in
  `docs/course-requirements.md`. Do not invent a different structure.

## Repository layout

```
config.py           shared constants, paths, seeds      (person B owns)
data_utils.py       MNIST, our data, synthetic stand-in (B / D)
plotting.py         shared figure style                 (person D)
airwrite_colab.ipynb  runs train -> ablation -> analysis on Colab
gestures.py         gesture logic, unit-testable        (person A)
whiteboard.py       hand tracking + canvas              (person A)
collect_data.py     dataset collection tool             (person A)
model.py            CNN architecture                    (person B)
train.py            training loop, baseline             (person B)
preprocessing.py    image normalisation pipeline        (person C)
evaluate.py         metrics, confusion matrices         (person C)
ablation.py         runs all experiment variants        (person C)
analysis.py         figures, cross-writer study         (person D)
metrics_analysis.py CIs, McNemar, per-writer, sensitivity (person C)
error_analysis.py   per-class F1, confusions, shift test

data/raw/<writer>/<digit>/*.png   our own collected digits (raw, uncropped size)
data/processed/                   generated, not committed
models/                           checkpoints
figures/                          all report figures
results/                          metric tables as CSV/JSON
docs/                             course brief, plan, division of work
```

## Module contracts

These interfaces were agreed so the four members can work in parallel.
**Do not change a signature without updating this file.**

```python
# preprocessing.py  (C -> B, C)
def preprocess(img: np.ndarray, steps: list[str]) -> np.ndarray:
    """Raw grayscale drawing -> 28x28 float32, normalised like MNIST."""

# model.py / train.py  (B -> C, D)
def predict(img: np.ndarray) -> tuple[int, float]:
    """img: 28x28 float32. Returns (digit, confidence)."""
```

Data written by `collect_data.py` is **raw and uncropped to 28x28 on
purpose**. Every preprocessing variant must start from the same source
images, otherwise the ablation is meaningless. Never "helpfully" resize at
collection time.

## Status

- [x] Phase 0 - repo scaffold, config, requirements (pinned for Python 3.11)
- [x] Phase 1 - whiteboard, gestures, collection tool (verified with a real
      webcam; `whiteboard.py --predict` is the live demo)
- [x] Phase 2 - CNN, logistic-regression baseline, training (`train.py`)
- [x] Phase 3 - preprocessing pipeline, augmentation (`preprocessing.py`)
- [x] Phase 4 - data collection: 807 digits from 4 writers (20 per digit)
- [x] Phase 5 code - `ablation.py` (zero-shot ablation, fine-tuning,
      leave-one-writer-out).
- [x] Phase 6 code - `analysis.py`, `metrics_analysis.py`, `error_analysis.py`.
- [x] Phase 5/6 runs on real data: official run 7 Nov 2026 (CPU, Python 3.11),
      all 9 checkpoints committed in `models/`
- [ ] Phase 7 - report
- [ ] Phase 8 - presentation (10 min, all members on stage)
- [ ] Phase 9 - submission (two places, see below)

Team task list: `docs/team-tasks.md`.

### Test modes

- `--quick` (train, ablation, analysis): tiny run, outputs in `*/smoke/`.
- `--synthetic` (ablation, analysis): stand-in data made by distorting
  MNIST test digits, outputs in `results/synthetic/`, `figures/synthetic/`.
  **These numbers must never appear in the report.**

See `docs/project-plan.md` for the step-by-step breakdown.

## Experiments to run

| Experiment | Purpose |
|---|---|
| Baseline (logistic regression) vs. CNN on MNIST | source-domain performance |
| MNIST model on our data, minimal preprocessing | **the central result** |
| Ablation over preprocessing steps | isolate each step's contribution |
| Augmented training | close the gap without target data |
| Fine-tuning on 70% of our data | upper bound |
| Leave-one-writer-out | generalisation to an unseen writing style |

Every configuration runs with **3 seeds**, reported as mean ± std. Most
preprocessing variants need no retraining — load the checkpoint and
re-evaluate. Only the baseline CNN and the augmented CNN need full training
runs (6 total).

## Working notes

- Compute is not the bottleneck: ~15-20 min total on a Colab T4, ~1.5-3 h on
  a laptop CPU. Measure one epoch before planning around it. If too slow,
  reduce epochs (8-10 is plenty for MNIST) before shrinking the network.
- MediaPipe wheels are not available for every Python version. Python 3.10 or
  3.11 is the safe choice.
- MNIST download from the original host can be slow from China; torchvision
  has mirrors.
- `python gestures.py` runs a self-test of the gesture logic with synthetic
  landmarks, no webcam required. Keep it passing.

## Submission (easy to get wrong)

Two destinations, different contents:

1. **python123.io** — report and code **only**. Do *not* upload data.
2. **Email `python_projects@163.com`** — report, code and data, **packaged
   as three separate archives**. One email per group.
   Subject format:
   `CourseProjectFall2026-Name1-ID1-Name2-ID2-Name3-ID3-Name4-ID4`
   Body must list the project title and every member's name, student ID,
   class, responsibilities and contribution.

## Placeholders still to fill

- Group member names, student IDs and classes (currently `A`-`D` and
  `Name 1`-`Name 4` in `docs/` and in the proposal).
