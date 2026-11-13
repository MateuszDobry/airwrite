"""
Derived numbers for the report. Reads only the team's result files (data/*.csv)
and writes derived.json plus a few small CSVs next to it. Nothing is estimated
by hand: every number quoted in the report that is not already in a result
file comes from this script.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(sys.argv[1])          # folder with the copied result files
out = {}

# ---------------------------------------------------------------- MNIST runs
m = pd.read_csv(D / "mnist_test_runs.csv")
g = m.groupby("model")
out["mnist"] = {
    k: {"acc_mean": g.accuracy.mean()[k], "acc_std": g.accuracy.std()[k],
        "f1_mean": g.macro_f1.mean()[k], "f1_std": g.macro_f1.std()[k],
        "seconds_mean": g.train_seconds.mean()[k],
        "acc_per_seed": m[m.model == k].sort_values("seed").accuracy.tolist()}
    for k in ["logreg", "cnn", "aug"]}

# ---------------------------------------------------------------- learning curves
hist = {}
for tag in ["mnist", "aug"]:
    runs = [pd.read_csv(D / f"history_{tag}_seed{s}.csv") for s in (42, 43, 44)]
    hist[tag] = {
        "val_acc_epoch8": [r.val_acc.iloc[-1] for r in runs],
        "train_acc_epoch8": [r.train_acc.iloc[-1] for r in runs],
        "sec_per_epoch_min": min(r.seconds.min() for r in runs),
        "sec_per_epoch_max": max(r.seconds.max() for r in runs),
        "sec_per_epoch_mean": float(np.mean([r.seconds.mean() for r in runs])),
    }
out["history"] = hist

# ---------------------------------------------------------------- predictions
p = pd.read_csv(D / "predictions.csv")
p["correct"] = p["correct"].astype(bool)
s42 = p[(p.model == "cnn") & (p.variant == "full") & (p.seed == 42)]
out["dataset"] = {
    "n": int(len(s42)),
    "per_class": {int(k): int(v) for k, v in s42.label.value_counts().sort_index().items()},
    "per_writer": {k: int(v) for k, v in s42.writer.value_counts().items()},
}

# thickness step, CNN: mean errors per seed by writer, minimal vs thickness
cnn = p[p.model == "cnn"]
err = (cnn.assign(err=~cnn.correct)
       .groupby(["variant", "writer", "seed"]).err.sum()
       .groupby(["variant", "writer"]).mean().unstack(0))
err = err[["minimal", "thickness", "centering", "full"]]
err["thickness_gain_images"] = err["minimal"] - err["thickness"]
out["cnn_errors_per_seed_by_writer"] = err.round(4).to_dict(orient="index")
out["thickness_total_gain_images"] = float(err["thickness_gain_images"].sum())
out["thickness_total_gain_points"] = float(100 * err["thickness_gain_images"].sum() / out["dataset"]["n"])
out["thickness_gain_roman_share"] = float(err.loc["roman", "thickness_gain_images"] / err["thickness_gain_images"].sum())

# augmented model: where are the errors at full preprocessing?
for var in ["minimal", "full"]:
    a = p[(p.model == "aug") & (p.variant == var) & (~p.correct)]
    out[f"aug_{var}_errors_total_3seeds"] = int(len(a))
    out[f"aug_{var}_errors_class1_3seeds"] = int((a.label == 1).sum())
    out[f"aug_{var}_errors_class1_share"] = float((a.label == 1).mean())
    c = p[(p.model == "cnn") & (p.variant == var) & (~p.correct)]
    out[f"cnn_{var}_errors_total_3seeds"] = int(len(c))
    out[f"cnn_{var}_errors_class1_3seeds"] = int((c.label == 1).sum())
    out[f"cnn_{var}_errors_by_writer_3seeds"] = {k: int(v) for k, v in c.writer.value_counts().items()}

# class 1 recall for cnn vs aug at full, per seed mean
for model in ["cnn", "aug"]:
    sub = p[(p.model == model) & (p.variant == "full") & (p.label == 1)]
    out[f"{model}_full_class1_recall"] = float(sub.correct.mean())

# confusion matrices for seed 42 (for the figure)
for var in ["minimal", "full"]:
    sub = p[(p.model == "cnn") & (p.variant == var) & (p.seed == 42)]
    cm = pd.crosstab(sub.label, sub.pred).reindex(index=range(10), columns=range(10), fill_value=0)
    cm.to_csv(D / f"confusion_cnn_{var}_seed42.csv")
    out[f"cnn_{var}_seed42_errors"] = int((sub.label != sub.pred).sum())

# ---------------------------------------------------------------- McNemar summary
mc = pd.read_csv(D / "mcnemar.csv")
rows = []
for (a, b), sub in mc.groupby(["config_a", "config_b"], sort=False):
    sub = sub.sort_values("seed")
    rows.append({"a": a, "b": b,
                 "acc_a": sub.acc_a.mean(), "acc_b": sub.acc_b.mean(),
                 "p_holm": [float(x) for x in sub.p_holm],
                 "significant": [bool(x) for x in sub.significant_5pct],
                 "n_sig": int(sub.significant_5pct.sum())})
out["mcnemar"] = rows

# ---------------------------------------------------------------- sensitivity
s = pd.read_csv(D / "stroke_ratio_sensitivity.csv")
agg = s.groupby("stroke_ratio").accuracy.agg(["mean", "std"])
out["sensitivity"] = {f"{k:.2f}": {"mean": float(v["mean"]), "std": float(v["std"])}
                      for k, v in agg.iterrows()}

# ---------------------------------------------------------------- fine-tune / LOWO
ab = pd.read_csv(D / "ablation_summary.csv")
out["finetune"] = ab[ab.experiment == "finetune"][["model", "stage", "accuracy_mean", "accuracy_std"]].to_dict(orient="records")
ce = pd.read_csv(D / "ceiling.csv")
ft = ce[ce.experiment == "finetune"].groupby(["group", "stage"]).agg(errors_mean=("errors", "mean"), n=("n", "first"))
out["finetune_errors"] = {f"{i[0]}/{i[1]}": {"errors_mean": float(r.errors_mean), "n": int(r.n)} for i, r in ft.iterrows()}
lw = ce[ce.experiment == "lowo"].groupby(["group", "stage"]).agg(errors_mean=("errors", "mean"), n=("n", "first"))
out["lowo_errors"] = {f"{i[0]}/{i[1]}": {"errors_mean": float(r.errors_mean), "n": int(r.n)} for i, r in lw.iterrows()}

# ---------------------------------------------------------------- confidence
cf = pd.read_csv(D / "confidence.csv")
out["confidence"] = cf.set_index(["model", "variant"]).round(4).reset_index().to_dict(orient="records")

# ---------------------------------------------------------------- model size
# parameter count from the layer shapes in model.py (conv without bias, BN has weight+bias, linear with bias)
convs = [(1, 32), (32, 64), (64, 128)]
n = sum(i * o * 9 + 2 * o for i, o in convs) + 128 * 3 * 3 * 10 + 10
out["cnn_parameters"] = int(n)

# ---------------------------------------------------------------- dump
def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o

(D / "derived.json").write_text(json.dumps(clean(out), indent=1))
print(json.dumps(clean(out), indent=1))
