# Writing in the Air

Measuring and mitigating domain shift in handwritten digit recognition.

Course project for *Programming with Python, Fall 2026*, Beijing Institute of
Technology. Deadline: **13 November 2026**.

## The question

A CNN trained on MNIST reaches above 99% accuracy. Does it still work on
digits written in the air with a finger? Those digits have uneven stroke
width, hand tremor, distorted proportions, and unwanted connecting strokes
because the pen cannot be lifted.

We measure how much accuracy is lost, then test which interventions recover
it: stroke normalisation, centre-of-mass alignment, training-time
augmentation, and fine-tuning.

The gesture whiteboard is the instrument for collecting that data, not the
project itself.

## Setup

Python 3.11 (3.10 also works). MediaPipe 0.10.21 has no wheels for 3.13+.

```bash
py -3.11 -m venv .venv            # macOS/Linux: python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running

```bash
python gestures.py     # gesture logic self-test, no webcam needed
python whiteboard.py   # free drawing
python whiteboard.py --predict   # live demo: classifies the digit when the hand pauses
```

In `--predict` mode the digit is classified as soon as you stop drawing (fist,
two fingers, or hand out of view) with `models/cnn_mnist_seed42.pt` and the
same `preprocess(..., FULL_STEPS)` as in the study. The top-right corner shows
the prediction, the top-3 probabilities and the 28x28 image the model sees.
`p` switches between full and minimal preprocessing on the same drawing,
`c` clears, `q` quits. `--model aug` uses the augmented CNN instead.

Gestures: index finger draws, fist pauses, index+middle moves the cursor
without drawing, open palm clears.

Collecting data:

```bash
python collect_data.py --writer <yourname> --per-digit 20
```

Walks through digits 0-9. SPACE saves, `c` clears, `n` next digit, `b` back,
`q` quits. Safe to stop and resume — numbering continues and nothing is
overwritten.

**Write naturally.** Do not imitate the MNIST style; measuring the real
difference between the two domains is the entire point.

## Experiments

```bash
python train.py        # MNIST: logistic regression, CNN, CNN + augmentation, 3 seeds
python ablation.py     # our data: ablation, fine-tuning, leave-one-writer-out
python analysis.py     # figures/ and results/headline.json
python metrics_analysis.py   # CIs, McNemar tests, per-writer, sensitivity -> results/metrics/
python error_analysis.py     # per-class F1, confusions, shift test -> results/analysis/
```

The last two only run inference with the committed checkpoints in `models/`,
so they take a few minutes and need no training.

On Colab, `airwrite_colab.ipynb` runs all three. On a laptop CPU the full
`train.py` takes roughly an hour; on a Colab T4 about 15-25 minutes.

Self-test without collected data (about 2 minutes, outputs go to `*/smoke/`):

```bash
python train.py --quick
python ablation.py --quick --synthetic
python analysis.py --quick --synthetic
```

`--synthetic` uses stand-in digits made by distorting MNIST. It exists only
to test the pipeline; its numbers are **not** results.

## Tuning

If the whiteboard behaves badly, adjust in `config.py`:

| Setting | Symptom |
|---|---|
| `STROKE_THICKNESS` | strokes too thick or too thin for the digit size |
| `SMOOTHING_WINDOW` | raise if lines are ragged, lower if drawing lags |
| `CAMERA_INDEX` | the wrong webcam opens |

## Documentation

- `CLAUDE.md` — project context, module contracts, current status
- `docs/project-plan.md` — step-by-step plan across all phases
- `docs/division-of-work.md` — who owns which files
- `docs/SETUP_PL.md` — instalacja i zbieranie danych dla zespołu (po polsku)
- `docs/team-tasks.md` — concrete tasks per person for the remaining days
- `docs/remaining-work.md` — checklist for analysis, report and submission
- `docs/analysis-notes.md` — written interpretation of the results (draft text for the report)
- `docs/course-requirements.md` — course brief as a checklist (Polish)
- `docs/course-brief.pdf` — the original brief
- `docs/project_proposal.pdf` — two-page English proposal

## Status

- The whiteboard and the collection tool work with a real webcam
  (MediaPipe 0.10.21, Python 3.11).
- Data collected: 807 digits from 4 writers (`data/raw/`, `data/labels.csv`).
- Official run on 7 November 2026: `train.py` (seeds 42, 43, 44) ->
  `ablation.py` -> `analysis.py` -> `metrics_analysis.py`, Python 3.11 with
  the pinned `requirements.txt`, CPU. All 9 checkpoints are in `models/`.
  Every number in the report comes from `results/`, every figure from
  `figures/`.
- Still to do: report, presentation, submission (see `docs/remaining-work.md`).
