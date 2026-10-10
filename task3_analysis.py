"""Task 3 extras: extra experiments, multi-seed tables, patient-level statistics,
feature analysis and Grad-CAM.

Nothing here changes the training code. This module only
  1. defines EXTRA experiments (ablations, seeds) that reuse experiments.run_experiment, and
  2. analyses the files that finished runs already wrote
     (results.json, config.json, history.csv, test_predictions.csv, best.pt).

The heavy imports (torch, the model code) happen inside the functions that need them, so
the tables and statistics only read files and are fast. Run this file's functions from the
repo folder, next to utils.py and config.py.
"""
import glob
import os
import re
import shutil

import numpy as np
import pandas as pd

import config
from utils import deep_merge, load_json

ABLATION_GROUP = "task3_ablation"   # extra experiments that explain *why* fine-tuning works
SEED_GROUP = "task3_seeds"          # repeats of existing experiments with other seeds

# Experiments we repeat with extra seeds (the headline comparison + scratch reference).
HEADLINE_IDS = ["t2s_cosine", "t3_resnet18_frozen", "t3_resnet18_partial", "t3_resnet18_full"]
# Cheap data-efficiency runs that are also repeated (the 50% runs are skipped to save time).
EFFICIENCY_SEED_IDS = ["t3eff_scratch_10", "t3eff_resnet18_frozen_10", "t3eff_resnet18_full_10",
                       "t3eff_scratch_25", "t3eff_resnet18_frozen_25", "t3eff_resnet18_full_25"]

METRICS = ["test_bal_acc", "test_macro_f1", "test_acc", "test_patient_acc",
           "test_binary_sensitivity", "test_binary_specificity", "peak_val_bal_acc"]


# ----------------------------------------------------------------------------------------
# 1. Extra experiments (they are built from registry experiments, experiments.json is untouched)
# ----------------------------------------------------------------------------------------
def _derive(base_id, new_id, group, description=None, **overrides):
    """Config of an existing registry experiment, renamed and moved to another group."""
    import experiments as ex
    cfg = deep_merge(ex.get_experiment(base_id), overrides)
    cfg.update(id=new_id, group=group, also_in=[])
    if description:
        cfg["description"] = description
    return cfg


def ablation_experiments():
    """Backbone learning-rate multiplier and number of unfrozen blocks (ResNet-18)."""
    out = []
    # (a) backbone LR multiplier for full fine-tuning. x0.1 is the registry run t3_resnet18_full.
    for mult in (1.0, 0.01):
        out.append(_derive("t3_resnet18_full", f"t3x_resnet18_full_mult{mult:g}", ABLATION_GROUP,
                           f"resnet18 full fine-tune, backbone LR x{mult:g}",
                           model={"backbone_lr_mult": mult}))
    # (b) number of unfrozen blocks. 1 block (layer4) is the registry run t3_resnet18_partial.
    for k in (2, 3):
        out.append(_derive("t3_resnet18_partial", f"t3x_resnet18_partial_u{k}", ABLATION_GROUP,
                           f"resnet18 partial: last {k} residual stages unfrozen",
                           model={"unfreeze_blocks": k}))
    return out


def seed_experiments(base_ids=None, seeds=(1, 2)):
    """Same experiment, different seed (new id, so it is not skipped as 'already done')."""
    base_ids = base_ids or HEADLINE_IDS
    return [_derive(b, f"{b}_s{s}", SEED_GROUP, seed=s) for b in base_ids for s in seeds]


def pending(cfgs, out_root=None):
    """ids of the experiments in cfgs that have no results.json yet (i.e. still to be trained)."""
    out_root = out_root or config.out_dir()
    return [c["id"] for c in cfgs
            if not os.path.exists(os.path.join(out_root, c["group"], c["id"], "results.json"))]


def run_cfgs(cfgs, out_root=None, only=None, **kw):
    """Run a list of resolved configs through the normal pipeline (finished ones are skipped)."""
    import experiments as ex
    results = []
    for cfg in cfgs:
        if only and cfg["id"] not in only:
            continue
        results.append(ex.run_experiment(cfg, out_root, **kw))
    return results


# ----------------------------------------------------------------------------------------
# 2. Reading finished runs, multi-seed tables
# ----------------------------------------------------------------------------------------
def load_all(out_root=None):
    """One row per finished run: results.json fields + seed + base id + folder."""
    out_root = out_root or config.out_dir()
    rows = []
    for p in glob.glob(os.path.join(out_root, "*", "*", "results.json")):
        r = load_json(p)
        d = os.path.dirname(p)
        cfg_path = os.path.join(d, "config.json")
        r["seed"] = load_json(cfg_path).get("seed", config.SEED) if os.path.exists(cfg_path) else config.SEED
        r["base_id"] = re.sub(r"_s\d+$", "", r["id"])      # "t3_resnet18_full_s1" -> "t3_resnet18_full"
        r["dir"] = d
        r["family"] = "scratch CNN" if r["model_name"] == "simple_cnn" else f'{r["model_name"]} {r["model_mode"]}'
        rows.append(r)
    return pd.DataFrame(rows)


def find_dir(run_id, out_root=None):
    hits = glob.glob(os.path.join(out_root or config.out_dir(), "*", run_id))
    if not hits:
        raise FileNotFoundError(f"No finished or started run folder for '{run_id}'")
    return hits[0]


def summary_table(df, ids=None, metrics=METRICS, extra=("params_trainable", "train_time_min",
                                                        "epochs_to_90pct_best")):
    """mean +- std over seeds for every base experiment (the 42-seed run counts as one seed)."""
    if ids is not None:
        df = df[df["base_id"].isin(ids)]
    rows = []
    for base, g in df.groupby("base_id", sort=False):
        row = {"experiment": base, "n_seeds": len(g)}
        for m in metrics:
            if m not in g:
                continue
            mean, std = g[m].mean(), g[m].std(ddof=1)
            row[m] = f"{mean:.3f}" if len(g) == 1 or np.isnan(std) else f"{mean:.3f} +- {std:.3f}"
        for m in extra:
            if m in g:
                row[m] = round(float(g[m].mean()), 2)
        rows.append(row)
    out = pd.DataFrame(rows)
    if ids is not None:                                    # keep the order the caller asked for
        out["_o"] = out["experiment"].map({k: i for i, k in enumerate(ids)})
        out = out.sort_values("_o").drop(columns="_o")
    return out.reset_index(drop=True)


def best_by_validation(df, metric="peak_val_bal_acc"):
    """Pick the winner on the VALIDATION set (never on test)."""
    g = df.groupby("base_id")[metric].mean().sort_values(ascending=False)
    return g.index[0], g


def efficiency_frame(df):
    """Rows for the data-efficiency plot: scratch / frozen / full ResNet-18 vs fraction of train data."""
    ref = {"t2s_cosine", "t3_resnet18_frozen", "t3_resnet18_full"}
    keep = df[(df["group"] == "task3_efficiency") | df["base_id"].isin(ref)]
    return keep[["family", "train_fraction", "n_train", "seed", "test_bal_acc", "test_macro_f1",
                 "test_acc", "peak_val_bal_acc"]].copy()


def aggregate_efficiency(eff, y="test_bal_acc"):
    g = eff.groupby(["family", "train_fraction"])[y].agg(["mean", "std", "count"]).reset_index()
    g["std"] = g["std"].fillna(0.0)
    return g


# ----------------------------------------------------------------------------------------
# 3. Statistics that respect the patient as the unit of independence
# ----------------------------------------------------------------------------------------
def _scores(t, p):
    """accuracy and balanced accuracy (mean recall over the classes present in t)."""
    acc = float((t == p).mean())
    bal = float(np.mean([(p[t == c] == c).mean() for c in np.unique(t)]))
    return acc, bal


def _prep(csv_path):
    d = pd.read_csv(csv_path)
    pats = d["patient"].to_numpy()
    idx = {p: np.where(pats == p)[0] for p in np.unique(pats)}
    return d["target"].to_numpy(), d["pred"].to_numpy(), idx


def patient_bootstrap(csv_path, n_boot=1000, seed=0):
    """95% CI for accuracy and balanced accuracy, resampling PATIENTS (images of one patient
    are not independent). Returns {"acc": (point, lo, hi), "bal_acc": (point, lo, hi)}."""
    t, p, idx = _prep(csv_path)
    rng = np.random.RandomState(seed)
    patients = list(idx)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(len(patients), len(patients), replace=True)
        rows = np.concatenate([idx[patients[i]] for i in pick])
        vals.append(_scores(t[rows], p[rows]))
    vals = np.array(vals)
    point = _scores(t, p)
    out = {}
    for j, name in enumerate(("acc", "bal_acc")):
        lo, hi = np.percentile(vals[:, j], [2.5, 97.5])
        out[name] = (point[j], float(lo), float(hi))
    return out


def paired_bootstrap(csv_a, csv_b, n_boot=1000, seed=0):
    """Is model A better than model B on the same test patients?
    Returns the mean difference in balanced accuracy, its 95% CI and P(A > B)."""
    a, b = pd.read_csv(csv_a), pd.read_csv(csv_b)
    m = a.merge(b, on="filename", suffixes=("_a", "_b"))
    t, pa, pb = m["target_a"].to_numpy(), m["pred_a"].to_numpy(), m["pred_b"].to_numpy()
    pats = m["patient_a"].to_numpy()
    idx = {p: np.where(pats == p)[0] for p in np.unique(pats)}
    patients = list(idx)
    rng = np.random.RandomState(seed)
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(len(patients), len(patients), replace=True)
        rows = np.concatenate([idx[patients[i]] for i in pick])
        diffs.append(_scores(t[rows], pa[rows])[1] - _scores(t[rows], pb[rows])[1])
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"mean_diff": float(diffs.mean()), "ci95": (float(lo), float(hi)),
            "p_a_better": float((diffs > 0).mean())}


def bootstrap_table(run_ids, out_root=None, n_boot=1000):
    rows = []
    for rid in run_ids:
        r = patient_bootstrap(os.path.join(find_dir(rid, out_root), "test_predictions.csv"), n_boot)
        rows.append({"run": rid,
                     "acc [95% CI]": "%.3f [%.3f, %.3f]" % r["acc"],
                     "bal acc [95% CI]": "%.3f [%.3f, %.3f]" % r["bal_acc"]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------
# 4. Re-running the test split once to get probabilities and features
# ----------------------------------------------------------------------------------------
def get_test_outputs(run_dir, with_features=False, device=None):
    """Reload best.pt, run the test split once, cache probs (and penultimate features) next to it."""
    cache = os.path.join(run_dir, "test_outputs.npz")
    if os.path.exists(cache):                       # reuse the saved outputs: no model, no GPU needed
        z = np.load(cache, allow_pickle=True)
        if not with_features or "feats" in z.files:
            return {k: z[k] for k in z.files}

    import torch
    import torch.nn as nn
    from breakhis_data import get_dataloaders
    from evaluation import load_checkpoint
    from utils import get_device

    device = device or get_device()
    model, ckpt = load_checkpoint(os.path.join(run_dir, "best.pt"), device)
    loaders, info = get_dataloaders(deep_merge(ckpt["exp_config"], {"num_workers": 2}))

    buf = []
    hook = None
    if with_features:   # input of the last Linear layer = the representation the classifier sees
        last_linear = [m for m in model.modules() if isinstance(m, nn.Linear)][-1]
        hook = last_linear.register_forward_hook(lambda m, i, o: buf.append(i[0].detach().float().cpu()))

    probs, targets = [], []
    with torch.no_grad():
        for x, y in loaders["test"]:
            probs.append(torch.softmax(model(x.to(device)).float(), 1).cpu())
            targets.append(y)
    if hook:
        hook.remove()

    out = {"probs": torch.cat(probs).numpy(), "targets": torch.cat(targets).numpy(),
           "patients": info["test_df"]["patient"].astype(str).to_numpy(),
           "class_names": np.array(info["class_names"])}
    if with_features:
        out["feats"] = torch.cat(buf).numpy()
    np.savez(cache, **out)
    return out


def binary_metrics(out):
    """Benign-vs-malignant from the 8-class probabilities: P(malignant) = sum of malignant classes."""
    from sklearn.metrics import roc_auc_score
    names = list(out["class_names"])
    mal = np.array([c not in config.BENIGN_SUBTYPES for c in names])
    p_mal = out["probs"][:, mal].sum(1)
    t_mal = mal[out["targets"]]
    pred = p_mal >= 0.5
    res = {"sensitivity": float((pred & t_mal).sum() / max(t_mal.sum(), 1)),
           "specificity": float((~pred & ~t_mal).sum() / max((~t_mal).sum(), 1)),
           "roc_auc": float(roc_auc_score(t_mal, p_mal)) if 0 < t_mal.sum() < len(t_mal) else float("nan")}
    # patient level: average P(malignant) over each patient's test images
    d = pd.DataFrame({"p": p_mal, "t": t_mal, "patient": out["patients"]}).groupby("patient").mean()
    res["patient_acc_binary"] = float(((d["p"] >= 0.5) == (d["t"] >= 0.5)).mean())
    return res


def binary_table(run_ids, out_root=None):
    rows = []
    for rid in run_ids:
        m = binary_metrics(get_test_outputs(find_dir(rid, out_root)))
        rows.append({"run": rid, **{k: round(v, 3) for k, v in m.items()}})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------
# 5. Plots
# ----------------------------------------------------------------------------------------
def plot_time_to_accuracy(run_ids, out_root=None, metric="val_bal_acc", path=None):
    """Validation score against cumulative TRAINING TIME (convergence speed in minutes)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for rid in run_ids:
        h = pd.read_csv(os.path.join(find_dir(rid, out_root), "history.csv"))
        ax.plot(h["epoch_time_s"].cumsum() / 60, h[metric], marker="o", ms=3, label=rid)
    ax.set_xlabel("cumulative training time (min)"), ax.set_ylabel(metric)
    ax.grid(alpha=0.3), ax.legend(fontsize=8), ax.set_title("Convergence speed in wall-clock time")
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_efficiency_ci(agg, y_label="test balanced accuracy (mean +- std over seeds)", path=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for fam, g in agg.groupby("family"):
        g = g.sort_values("train_fraction")
        ax.errorbar(g["train_fraction"] * 100, g["mean"], yerr=g["std"], marker="o", capsize=3, label=fam)
    ax.set_xlabel("% of training images"), ax.set_ylabel(y_label), ax.grid(alpha=0.3), ax.legend()
    ax.set_title("Data efficiency: scratch vs frozen vs fine-tuned")
    fig.tight_layout()
    _save(fig, path)
    return fig


def show_run(run_id, out_root=None):
    """Curves, per-class accuracy and normalised confusion matrix of ANY finished run (also the extra ones)."""
    import matplotlib.pyplot as plt
    import plots
    d = find_dir(run_id, out_root)
    res = load_json(os.path.join(d, "results.json"))
    hist = pd.read_csv(os.path.join(d, "history.csv")).to_dict("records")
    plots.plot_learning_curves(hist, run_id)
    plots.plot_per_class_accuracy(hist, res["class_names"], run_id)
    plots.plot_confusion_matrix(res["confusion_matrix_normalized"], res["class_names"], True, f"{run_id} - test")
    plt.show()
    keys = ["test_acc", "test_bal_acc", "test_macro_f1", "test_patient_acc", "test_binary_sensitivity",
            "test_binary_specificity", "peak_val_acc", "generalization_gap_acc", "params_total",
            "params_trainable", "latency_ms_bs1", "train_time_min"]
    return pd.Series({k: res[k] for k in keys if k in res}, name=run_id)


def plot_roc(run_ids, out_root=None, path=None):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for rid in run_ids:
        out = get_test_outputs(find_dir(rid, out_root))
        mal = np.array([c not in config.BENIGN_SUBTYPES for c in out["class_names"]])
        fpr, tpr, _ = roc_curve(mal[out["targets"]], out["probs"][:, mal].sum(1))
        ax.plot(fpr, tpr, label=f"{rid} (AUC {binary_metrics(out)['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("false positive rate (benign called malignant)"), ax.set_ylabel("sensitivity (malignant found)")
    ax.legend(fontsize=7), ax.grid(alpha=0.3), ax.set_title("Benign vs malignant ROC (from 8-class model)")
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_tsne(run_ids, out_root=None, color_by="subtype", max_points=1500, seed=0, path=None):
    """t-SNE of the penultimate-layer features on the TEST split, one panel per model.
    color_by='patient' shows whether the model groups images by patient instead of by tumour type."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score

    fig, axes = plt.subplots(1, len(run_ids), figsize=(5.2 * len(run_ids), 4.8), squeeze=False)
    for ax, rid in zip(axes[0], run_ids):
        out = get_test_outputs(find_dir(rid, out_root), with_features=True)
        rng = np.random.RandomState(seed)
        pick = rng.choice(len(out["feats"]), min(max_points, len(out["feats"])), replace=False)
        X = out["feats"][pick]
        names = np.array(out["class_names"])[out["targets"][pick]]
        labels = names if color_by == "subtype" else out["patients"][pick]
        sil = silhouette_score(X, names) if len(set(names)) > 1 else float("nan")
        if X.shape[1] > 50:
            X = PCA(50, random_state=seed).fit_transform(X)
        Z = TSNE(2, perplexity=min(30, (len(X) - 1) / 3), init="pca", random_state=seed).fit_transform(X)
        for lab in sorted(set(labels)):
            m = labels == lab
            ax.scatter(Z[m, 0], Z[m, 1], s=6, label=lab if color_by == "subtype" else None)
        ax.set_title(f"{rid}\nsilhouette (subtype) = {sil:.2f}", fontsize=9), ax.set_xticks([]), ax.set_yticks([])
        if color_by == "subtype":
            ax.legend(fontsize=6, markerscale=2)
    fig.tight_layout()
    _save(fig, path)
    return fig


# ----------------------------------------------------------------------------------------
# 6. Grad-CAM: which pixels pushed the model towards its decision
# ----------------------------------------------------------------------------------------
def _cam_layer(model):
    """Last convolutional stage whose activation we explain."""
    if hasattr(model, "backbone"):
        name = model.model_config["backbone"]
        return model.backbone.layer4 if name.startswith("resnet") else model.backbone.features[-1]
    return model.features[-2]          # scratch CNN: the last activation, just before the final MaxPool


def gradcam(model, x, class_idx=None):
    """Grad-CAM (Selvaraju et al.): weight each feature map by the mean gradient of the class score,
    sum, ReLU, normalise to [0, 1], upsample to the input size. Returns (cam [B,H,W], logits, class_idx)."""
    import torch
    import torch.nn.functional as F
    model.eval()
    store = {}

    def fwd(_, __, out):
        store["a"] = out
        out.register_hook(lambda g: store.__setitem__("g", g))

    handle = _cam_layer(model).register_forward_hook(fwd)
    x = x.clone().requires_grad_(True)       # frozen backbones need an input gradient to build a graph
    logits = model(x)
    if class_idx is None:
        class_idx = logits.argmax(1)
    model.zero_grad(set_to_none=True)
    logits.gather(1, class_idx.view(-1, 1)).sum().backward()
    handle.remove()
    a, g = store["a"].detach(), store["g"]
    cam = torch.relu((g.mean(dim=(2, 3), keepdim=True) * a).sum(1))
    cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
    cam = F.interpolate(cam[:, None], size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    return cam.detach().cpu(), logits.detach(), class_idx


def show_gradcam(run_id, out_root=None, per_class=1, device=None, path=None):
    """For every subtype present in the test split show its most confident CORRECT test image
    and the Grad-CAM heat map of the model's decision."""
    import matplotlib.pyplot as plt
    import torch
    from breakhis_data import denormalize, get_dataloaders
    from evaluation import load_checkpoint
    from utils import get_device

    device = device or get_device()
    run_dir = find_dir(run_id, out_root)
    model, ckpt = load_checkpoint(os.path.join(run_dir, "best.pt"), device)
    loaders, info = get_dataloaders(deep_merge(ckpt["exp_config"], {"num_workers": 0}))
    ds = loaders["test"].dataset
    out = get_test_outputs(run_dir)
    names = list(out["class_names"])
    pred, conf = out["probs"].argmax(1), out["probs"].max(1)

    chosen = []
    for c in range(len(names)):
        ok = np.where((out["targets"] == c) & (pred == c))[0]
        chosen += list(ok[np.argsort(-conf[ok])][:per_class])
    if not chosen:
        raise ValueError(f"{run_id}: no correctly classified test image found for any class")
    x = torch.stack([ds[int(i)][0] for i in chosen]).to(device)
    cam, _, cls = gradcam(model, x)
    img = denormalize(x.cpu(), info["mean"], info["std"])

    fig, axes = plt.subplots(2, len(chosen), figsize=(2.2 * len(chosen), 4.8), squeeze=False)
    for j, i in enumerate(chosen):
        axes[0, j].imshow(img[j].permute(1, 2, 0)); axes[0, j].set_title(names[out["targets"][i]], fontsize=9)
        axes[1, j].imshow(img[j].permute(1, 2, 0)); axes[1, j].imshow(cam[j], cmap="jet", alpha=0.45)
        axes[1, j].set_title(f"conf {conf[i]:.2f}", fontsize=8)
        axes[0, j].axis("off"); axes[1, j].axis("off")
    fig.suptitle(f"Grad-CAM: {run_id}")
    fig.tight_layout()
    _save(fig, path)
    return fig


# ----------------------------------------------------------------------------------------
# 7. Local-disk workflow (fast) + sync with the shared Drive folder, and the defense helper
# ----------------------------------------------------------------------------------------
def _finished_runs(root):
    return [os.path.dirname(p) for p in glob.glob(os.path.join(root, "*", "*", "results.json"))]


def sync_down(drive_out, local_out):
    """Copy finished runs from Drive to local disk so they are skipped, not retrained."""
    n = 0
    for d in _finished_runs(drive_out):
        dst = os.path.join(local_out, os.path.relpath(d, drive_out))
        if not os.path.exists(os.path.join(dst, "results.json")):
            shutil.copytree(d, dst, dirs_exist_ok=True)
            n += 1
    print(f"sync_down: {n} finished runs copied to {local_out}")


def sync_up(local_out, drive_out):
    """Copy finished local runs to Drive (without last.pt). Never overwrites a finished run on Drive."""
    n = 0
    for d in _finished_runs(local_out):
        dst = os.path.join(drive_out, os.path.relpath(d, local_out))
        if os.path.exists(os.path.join(dst, "results.json")):
            continue
        shutil.copytree(d, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("last.pt"))
        n += 1
    print(f"sync_up: {n} new finished runs copied to {drive_out}")


def quick_eval(ckpt_path, n_batches=2, batch_size=32, device=None):
    """Defense demo: load a saved checkpoint and classify a few unseen test mini-batches."""
    import torch
    from breakhis_data import get_dataloaders
    from evaluation import load_checkpoint
    from utils import get_device

    device = device or get_device()
    model, ckpt = load_checkpoint(ckpt_path, device)
    cfg = deep_merge(ckpt["exp_config"], {"num_workers": 0, "batch_size": batch_size})
    loaders, info = get_dataloaders(cfg)
    names, correct, total = info["class_names"], 0, 0
    with torch.no_grad():
        for b, (x, y) in enumerate(loaders["test"]):
            if b >= n_batches:
                break
            pred = model(x.to(device)).argmax(1).cpu()
            correct += int((pred == y).sum()); total += len(y)
            print(f"batch {b}: acc {(pred == y).float().mean():.2f} | true {[names[i] for i in y[:8]]} "
                  f"| pred {[names[i] for i in pred[:8]]}")
    print(f"{correct}/{total} correct on {n_batches} test mini-batches ({ckpt['exp_config']['id']})")


def _save(fig, path):
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
