# Analysis notes (draft text for the report)

Our notes on the results, used for sections 3 and 4 of the report. Every number
comes from a file in `results/`.

All numbers are from the official run (7 November 2026, seeds 42-44, 807 of our digits)
and are the mean over 3 seeds unless stated otherwise. Sources:

- `results/ablation_summary.csv`, `results/headline.json`, `results/domain_stats.csv`
- `results/metrics/*.csv` (`metrics_analysis.py`)
- `results/analysis/*.csv` (`error_analysis.py`)

Figures referred to: `figures/ablation_*.png`, `figures/confusion_ours_*.png`,
`figures/metrics/*.png`, `figures/analysis/per_class_f1.png`,
`figures/analysis/confusion_gallery.png`, `figures/analysis/shift_sensitivity.png`.

---

## 1. Error analysis: what the CNN gets wrong, and why

At minimal preprocessing the CNN makes 28 errors per seed on 807 images
(`results/metrics/confidence_intervals.csv`). The four most frequent confusions
(`results/analysis/confusion_change.csv`, errors per seed) are:

| True → predicted | Minimal | Full | Our reading of the images (`confusion_gallery.png`) |
|---|---|---|---|
| 6 → 5 | 6.33 | 0.33 | The upper stroke of the 6 is long and the loop is small and sits low, so the top half looks like the flag of a 5. |
| 7 → 1 | 6.00 | 0 | Thin, narrow 7s with a very short top bar: after a plain resize the bar is only a few pixels wide. |
| 1 → 7 | 3.67 | 1.67 | Our 1s are written with a long upstroke ("flag") at the top, the continental-European style. MNIST 1s are mostly a single vertical bar, so a flagged 1 is closer to an MNIST 7. |
| 9 → 3 | 3.67 | 0.33 | 9s with an open loop: the left side of the loop is missing or very thin, leaving a shape like the top of a 3. |

Full preprocessing removes or nearly removes the first, second and fourth pair. The
1 → 7 confusion is the one that survives (1.67 per seed), which fits the explanation: it
comes from *how* the digit is written, not from stroke width or position, so no
normalisation step can fix it. At full preprocessing the CNN is left with 4.33 errors
per seed.

The gallery also shows that most raw drawings are clean and readable to a human. The
errors are caused by differences in style (flagged 1, short 7 bar) and by geometry that
preprocessing changes (thin strokes, off-centre mass), not by noisy capture.

## 2. Per-class F1

`results/analysis/per_class_f1_all.csv`, `figures/analysis/per_class_f1.png`.

For the CNN, the weakest digits at minimal preprocessing are 1 (F1 0.930), 7 (0.932),
6 (0.953) and 9 (0.954). Each step affects them differently:

- **Stroke thickness** helps 1, 7 and 9 (1: 0.959, 7: 0.963, 9: 0.965) but not 6 (0.953).
  Thin strokes hurt digits whose identity depends on a short, thin detail (the bar of the 7,
  the loop of the 9).
- **Centring** helps 6 most (0.982) and 7 (0.988).
- **Full** brings every digit to at least 0.985 (digit 1 is the lowest, consistent with the
  surviving 1 → 7 confusion).

For logistic regression the picture is extreme: at minimal preprocessing digit 7 has
F1 0.06 and digit 9 has 0.02. Centring alone lifts 7 to 0.86, but 9 stays at 0.24 even
with full preprocessing, because the linear model sends most 9s to 3
(48.67 errors per seed at full, `confusion_change.csv`).

## 3. Ablation: what each step contributes

`results/ablation_summary.csv`, `results/metrics/gap_closed.csv`,
`results/metrics/mcnemar.csv`, `figures/ablation_models.png`, `figures/metrics/gap_closed.png`.

| Model | Minimal | Thickness | Centring | Full |
|---|---|---|---|---|
| Logistic regression | 55.47 % | 63.28 % | 74.56 % | 79.97 % |
| CNN | 96.53 % | 97.48 % | 98.43 % | 99.46 % |
| Augmented CNN | 94.34 % | 94.96 % | 97.56 % | 97.19 % |

For the CNN, centring alone closes 68 % of the MNIST → ours gap and stroke thickness 34 %;
together they close all of it (99.46 % vs. 99.33 % on MNIST, a difference within the
confidence intervals). McNemar's test (Holm-corrected) finds minimal → centring and
minimal → full significant for all three seeds, but minimal → thickness significant for
none, so the thickness step on its own is a trend, not a proven effect at this data size.

**Why centring matters much more for the linear model.** Centring adds 19.1 points for
logistic regression but only 1.9 points for the CNN. Logistic regression learns one
weight per pixel, so a digit moved by a few pixels lands on weights learned for other
digits. The CNN's convolutions and max-pooling give it partial translation invariance.

`error_analysis.py` tests this directly on MNIST, without our data: every MNIST test digit
is shifted by k pixels (`results/analysis/shift_sensitivity.csv`,
`figures/analysis/shift_sensitivity.png`).

| Shift (px) | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| CNN | 99.33 % | 99.20 % | 98.22 % | 94.97 % | 85.48 % |
| Logistic regression | 92.35 % | 86.19 % | 62.44 % | 35.00 % | 18.56 % |

A 2-pixel shift costs the CNN about 1 point and the linear model about 30 points. This
supports the explanation. Note that the CNN is not fully shift-invariant either: at
3-4 pixels it also drops sharply, which is why centring still helps it.

## 4. Domain statistics and the steps that fix them

`results/domain_stats.csv`, `figures/domain_stroke_width.png`, `figures/domain_fill_ratio.png`,
`results/metrics/per_writer.csv`, `figures/metrics/per_writer.png`.

- **Stroke width.** The median stroke width at 28×28 is 2.44 px for MNIST and
  0.90-1.55 px for our writers. This is what the thickness-normalisation step targets.
- **Fill ratio** (share of ink pixels, median): 0.130 for MNIST, 0.052-0.081 for our writers. The
  same cause: thinner strokes mean less ink.
- **Per writer.** The writer with the thinnest strokes (Roman, median 0.90 px) is the
  hardest at minimal preprocessing: CNN 92.91 % vs. 97.2-98.3 % for the other three, and
  logistic regression 43.16 % vs. 52.3-65.0 %. With full preprocessing the CNN reaches
  99.19 % on the same writer, in the same range as the others (99.0-99.8 %). The
  per-writer gap is closed by the step that targets the measured difference.

## 5. Why augmentation hurts (hypothesis)

The augmented CNN is worse than the plain CNN on our data at every preprocessing level
(97.19 % vs. 99.46 % at full; McNemar significant for all three seeds at full,
`mcnemar.csv`). On MNIST it is almost as good (99.12 % vs. 99.33 %,
`results/mnist_test_runs.csv`), so it did not simply learn worse.

Almost all of the difference is one confusion: the augmented CNN reads our 1s as 7s
(18.0 errors per seed at full preprocessing, 29.3 at minimal; `confusion_change.csv`),
and its F1 for digit 1 is 0.84 at full vs. 0.985 for the plain CNN. The error is
concentrated in the writers whose 1s have the longest flag: at full preprocessing
45 % of Kirill's 1s and 27 % of Mateusz's are read as 7, against 5 % of Kacper's
(`results/analysis/ones_as_sevens.csv`).

Two measured facts point to a likely cause, which we state as a hypothesis:

1. Augmentation makes the training strokes thicker (median 2.44 → 3.24 px,
   `results/metrics/augmentation_stroke_width.csv`), while our strokes are thinner than
   MNIST and full preprocessing normalises them to MNIST width, not to augmented width.
   Augmentation moves the training data *away* from our data on the one axis where we
   measured the biggest difference.
2. Rotation and elastic warping turn some MNIST 7s into near-vertical strokes with a short
   slanted top, which is close to our flagged 1. That would teach the model that this
   shape is a 7.

We did not run the experiment that would confirm either cause (e.g. training with each
augmentation separately), so the report should present this as a hypothesis.
Augmentation was designed for noisy, distorted input; our whiteboard produces clean,
upright strokes, so the augmented training data is less like our test data than plain MNIST.

## 6. Ceiling effect in fine-tuning and leave-one-writer-out

`results/metrics/ceiling.csv`, `results/lowo_summary.csv`, `figures/lowo.png`.

- **Fine-tuning** (70/30 split, 243 test images): the CNN makes 0, 1 and 0 errors before
  fine-tuning and 0 after, for the three seeds. With 0 errors in 243 images the 95 % upper
  bound on the error rate is still 1.2 % (rule of three), so "100 %" here means "no
  measurable difference", not "fine-tuning solved the problem".
- **Leave-one-writer-out**: before fine-tuning, accuracy per held-out writer is already
  99.0-99.8 % (0-2 errors in 200 images); after, 99.4-100 %.

Both experiments start from a model that is already at the ceiling on our full-preprocessed
data, so they cannot rank the interventions or tell us how well the model generalises to
a new writer. The report should say this explicitly and not present 100 % as a result.
A harder version (LOWO at minimal preprocessing, where a gap remains) would be needed to
make these experiments informative; it was not run.

## 7. Confidence of the errors

`results/metrics/confidence.csv`, `figures/metrics/confidence.png`.

The CNN's wrong answers have lower confidence than its correct answers (0.68 vs. 0.96 at
minimal preprocessing), but 17 % of its errors at minimal preprocessing still have
confidence ≥ 0.9. A confidence threshold would catch most errors, not all, which matters
for the live demo: a confident answer is not always a correct one.

---

## Points for the Limitations section

- 807 images, 4 writers, one webcam, one session per writer: one error is 0.12 points.
- The whiteboard renders clean strokes of constant width. This is probably why the CNN's
  gap is small (2.8 points); real pen or touch input would be noisier.
- The augmentation explanation is a hypothesis, not an ablated result.
- Fine-tuning and LOWO are at the ceiling and give no information at this data size.
- The stroke-ratio value (0.12) was fixed before the data was collected;
  `results/metrics/stroke_ratio_sensitivity.csv` is reported as a robustness curve only
  (98.4-99.6 % per seed across 0.06-0.18), not used to pick a value.
