# results/metrics

Written by `metrics_analysis.py` (Kirill Riabkov, person C). Read `results_analysis.ipynb` for
tables, figures and conclusions.

These files come from the official run: all 9 checkpoints in `models/` (CNN, augmented CNN and
logistic regression, seeds 42-44), the same checkpoints behind `results/ablation_summary.csv`.
`checkpoints_used.csv` lists them. To reproduce, run `python metrics_analysis.py` from the
repository root; it only runs inference and takes about a minute on a laptop CPU.
