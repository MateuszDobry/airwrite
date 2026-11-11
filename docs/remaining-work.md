# Remaining work

Status on 5 November 2026. Deadline: **13 November 2026, 23:59** (report + code + data),
oral presentation in weeks 9-11.

Done: code, data collection (4 writers, 807 images), one full run of
`train.py` → `ablation.py` → `analysis.py` → `metrics_analysis.py` (official run, 7 November 2026,
Python 3.11 with the pinned `requirements.txt`, CPU; all 9 checkpoints are in `models/`). Every number in the report must come from
`results/`; every figure from `figures/`.

---

## 1. What the results say (read this first)

From `results/headline.json` and `results/ablation_summary.csv`, mean of 3 seeds:

| Setting | Accuracy |
|---|---|
| CNN on MNIST test | 99.33 % |
| Logistic regression on MNIST test | 92.35 % |
| CNN on our digits, minimal preprocessing | 96.53 % |
| CNN on our digits, full preprocessing | 99.46 % |
| Augmented CNN on our digits, full preprocessing | 97.19 % |
| Logistic regression on our digits, minimal → full | 55.47 % → 79.97 % |
| Fine-tuned CNN (70/30 split) | 100 % (already 99.86 % before fine-tuning) |

The story is different from the proposal, and that is fine as long as we say so:

- The CNN loses only **2.8 points** on in-air digits; full preprocessing removes the gap.
  Centring is the strongest single step (68 % of the gap closed), then stroke thickness (34 %);
  see `results/metrics/gap_closed.csv`.
- The **linear baseline is not robust at all**: a 37-point drop. This is the clearest
  contrast in the study (translation and stroke-width sensitivity vs. a CNN).
- **Augmentation hurts** on our data (97.19 % vs. 99.46 %).
- **Fine-tuning and leave-one-writer-out hit the ceiling** (~99-100 % before and after),
  so they cannot rank interventions. Say this explicitly; do not present it as a win.
- Our strokes are much thinner than MNIST (median width 0.9-1.6 px vs. 2.4 px at 28×28,
  `results/domain_stats.csv`). Roman's are the thinnest.

---

## 2. Analysis to do

Work on a branch and open a PR. Any new number must be produced by a script in the repo
that writes to `results/` (not computed by hand or in a notebook cell that is not committed).

**Must have**

- [ ] **Error analysis** from `figures/confusion_ours_minimal.png` and
      `misclassified_minimal.png`. Main confusions at minimal preprocessing:
      6 → 5 (6.3 per seed), 7 → 1 (6.0), 1 → 7 (3.7), 9 → 3 (3.7) of about 80 per digit
      (`results/metrics/confusion_pairs.csv`). Explain each from the images
      (e.g. a 1 written with a serif/flag looks like a European 7; a 6 with an open loop).
      Then show which of these disappear with full preprocessing (`confusion_ours_full.png`).
- [ ] **Per-class F1** (`results/per_class_f1.csv`): digits 1, 7, 9 are weakest at
      minimal preprocessing; show how each step changes them. A small table or heatmap.
- [ ] **Ablation interpretation** (`figures/ablation_cnn.png`, `ablation_models.png`):
      contribution of each step, for all three models. Explain why centring helps the
      linear model so much more than the CNN (pooling gives the CNN some translation
      invariance; logistic regression has none).
- [ ] **Domain statistics** (`domain_stroke_width.png`, `domain_fill_ratio.png`): link
      the measured differences to the preprocessing steps that fix them.
- [ ] **Why augmentation hurts**: look at `augmentation_examples.png`. Our digits are
      clean and upright; rotation, elastic warp and dilation make the training data
      *less* like our test data. State this as a hypothesis, not a proven fact.
- [ ] **Ceiling effect**: say that fine-tuning and LOWO are saturated, so they give no
      information about generalisation to a new writer at this data size.

**Should have (strengthens "sufficient analysis" and the hyperparameter point in the template)**

- [x] **Per-writer zero-shot accuracy** (done: `results/metrics/per_writer.csv`) at minimal and full preprocessing (is the writer
      with the thinnest strokes the hardest?). Needs a short script reading
      `data/labels.csv` and the existing checkpoints; output `results/per_writer.csv`.
- [x] **Confidence intervals** (done: `results/metrics/confidence_intervals.csv`, plus McNemar tests in `mcnemar.csv`): 807 test images means one error = 0.12 points. A bootstrap
      or Wilson interval around the headline accuracies shows which differences are real
      (2.8-point gap: yes; 99.33 vs 99.46: no). Output `results/confidence_intervals.csv`.
- [x] **Hyperparameter robustness** (done: `results/metrics/stroke_ratio_sensitivity.csv`): accuracy as a function of `TARGET_STROKE_RATIO`
      (e.g. 0.06-0.18). Report it as a **sensitivity curve only**. The value used in the
      main results (0.12) was fixed before seeing the data and must stay unchanged;
      picking the best value from this curve would be tuning on the test set.
- [ ] **Harder LOWO** (optional): leave-one-writer-out with *minimal* preprocessing,
      where there is still a gap, so the experiment can show something.

---

## 3. Report (Word or LaTeX, English, concise)

Use **exactly** this structure (course template). Header: title, all names, student IDs, classes.

### 1. Introduction
- [ ] Problem: classifiers trained on clean scanned digits used on a different input
      device; why domain shift matters (touchless input, AR/VR, accessibility).
- [ ] Our approach in 2-3 sentences: own gesture whiteboard → own dataset → measure the gap →
      ablation of mitigations.
- [ ] Main findings in 2 sentences (numbers from section 1 above).

### 2. Method
- [ ] **Data acquisition**: MediaPipe Hands, gestures, smoothing, canvas, `collect_data.py`,
      collection protocol ("write naturally"), 4 writers × 20 per digit.
- [ ] **Models with justification against common practice**: logistic regression as the
      classic linear baseline; small CNN (3 conv blocks) as the standard choice for MNIST.
      Training setup: epochs, optimiser, seeds 42/43/44, train/val split.
- [ ] **Preprocessing pipeline**: each step (binarise, stroke thickness normalisation,
      centre of mass, resize/pad) with `figures/preprocessing_steps.png`.
- [ ] **Augmentation**: dilation, rotation, shift, elastic warp (Simard et al. 2003).
- [ ] **Own contributions** stated explicitly: the whiteboard tool, the dataset, the
      thickness-normalisation step, the ablation design.
- [ ] One pipeline diagram (camera → hand landmarks → strokes → image → preprocessing → model).

### 3. Experimental Results and Analysis
- [ ] Datasets: MNIST (60k/10k) and ours (807 images, 4 writers, class balance,
      `samples_by_writer.png`).
- [ ] Evaluation criteria: accuracy and macro-F1, mean ± std over 3 seeds.
- [ ] MNIST results and `learning_curves.png` (training converges by ~epoch 6-8).
- [ ] The gap, the ablation table, confusion matrices, everything from section 2 above.
- [ ] Every table/figure numbered, captioned, and referred to in the text.

### 4. Discussion and Conclusion
- [ ] Summary of findings.
- [ ] Original contribution: new in-air digit dataset, measured domain gap, ablation.
- [ ] Strengths: own real-world data, controlled ablation, 3 seeds, everything reproducible.
- [ ] **Limitations (be honest, it raises the mark)**: 4 writers, 807 images, one webcam
      and lighting, one session per writer, the whiteboard renders clean uniform strokes
      (which is likely why the gap is small), ceiling effect in fine-tuning/LOWO.
- [ ] Lessons learned; future work (more writers, raw trajectories instead of rendered
      images, sequence models, noisier capture conditions).

### 5. References (one consistent style)
- [ ] LeCun et al. 1998 (MNIST / LeNet), Simard et al. 2003 (elastic distortions),
      Zhang et al. 2020 (MediaPipe Hands), PyTorch, scikit-learn, OpenCV, NumPy, pandas,
      Matplotlib. Cite any reused code.

### Report checks
- [ ] Every number appears in a file under `results/` (no rounding errors, no numbers from
      `smoke/` or `synthetic/`).
- [ ] Whoever wrote the code behind a section checks that section for accuracy.
- [ ] Short "Contributions" paragraph (also needed in the submission email).

---

## 4. Course requirements checklist

| Requirement | Status |
|---|---|
| ≤ 4 students | ✅ |
| ≥ 100 lines of code | ✅ (well over) |
| ≥ 2 AI/data libraries | ✅ PyTorch, scikit-learn, NumPy, pandas, Matplotlib |
| Code runs correctly | ✅ full run completed; re-check on a fresh clone before submitting |
| Reproducible (seeds, pinned versions) | ✅ seeds fixed, `requirements.txt` pinned |
| Sufficient comments | ☐ review every `.py` file: module docstring, function docstrings, comments on non-obvious steps |
| No fabricated/modified results | ✅ keep it that way: numbers only from `results/` |
| AI / data analysis applied to our field of study | ☐ state the link explicitly in the Introduction |
| Clear goal of the program | ✅ research question in README / CLAUDE.md |
| Sufficient data analysis | ☐ section 2 of this file |
| Sufficient visualisation | ✅ 14 figures; pick the best 6-8 for the report |
| Some innovation | ✅ own acquisition tool + own dataset; say it explicitly |
| Real-world dataset | ✅ our own recordings + MNIST |
| Report follows the 5-section template | ☐ |
| All sources cited, consistent format | ☐ |
| Oral presentation, 10 min, all four on stage | ☐ |

---

## 5. Code and repository housekeeping

- [ ] `whiteboard.py --predict`: live prediction for the demo. Show the predicted digit and
      confidence while the hand pauses (fist), using
      `predict(preprocess(board.extract_digit(), FULL_STEPS))` and `models/cnn_mnist_seed42.pt`.
- [ ] Record a 30-60 s backup video of the demo in case the camera fails on stage.
- [ ] Update `README.md` / `CLAUDE.md` status (webcam now verified, data collected,
      results in). Add how the results were produced (CPU, Python 3.11, seeds 42-44).
- [ ] Fill in names, student IDs, classes in `division-of-work.md`.
- [ ] Final test on a fresh clone: `pip install -r requirements.txt`, `python gestures.py`,
      `python train.py --quick`.

---

## 6. Presentation (10 minutes, everyone on stage)

- [ ] Slides: problem (1), method and tool (2), results (3), live demo (1), conclusions (1).
- [ ] Every member speaks; each explains the part they worked on.
- [ ] One rehearsal with a timer.

---

## 7. Submission (by 13 Nov 23:59)

Two places, different contents:

1. **python123.io**, correct course unit: **report + code only. No data.**
2. **Email to `python_projects@163.com`**, one per group:
   - [ ] Subject: `CourseProjectFall2026-Name1-ID1-Name2-ID2-Name3-ID3-Name4-ID4`
   - [ ] Body: project title; every member's name, student ID, class,
         responsibilities and contribution.
   - [ ] Three **separate** archives: `report.zip`, `code.zip` (no `.venv`, no `data/`,
         no `models/smoke`), `data.zip` (`data/raw/` + `data/labels.csv`).
