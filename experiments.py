import glob
import os
import shutil

import matplotlib.pyplot as plt
import pandas as pd
import torch

import config
import plots
from breakhis_data import get_dataloaders
from engine import fit
from evaluation import model_summary, results_table, test_report
from models import build_model
from utils import Timer, deep_merge, get_device, load_json, save_json, set_seed

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "experiments.json")

# Registry
def load_registry(path=CONFIG_PATH):
    return load_json(path)


def resolve(defaults, exp):
    cfg = deep_merge(defaults, {k: v for k, v in exp.items() if k != "model"})
    model = exp.get("model", {})
    if model.get("name", defaults["model"]["name"]) != defaults["model"]["name"]:
        cfg["model"] = dict(model)
    else:
        cfg["model"] = deep_merge(defaults["model"], model)
    cfg.setdefault("also_in", [])
    return cfg


def get_experiment(exp_id, path=CONFIG_PATH):
    reg = load_registry(path)
    for exp in reg["experiments"]:
        if exp["id"] == exp_id:
            return resolve(reg["defaults"], exp)
    raise KeyError(f"Unknown experiment '{exp_id}'. Known: {[e['id'] for e in reg['experiments']]}")


def list_experiments(group=None, include_inactive=False, include_refs=False, path=CONFIG_PATH):
    reg = load_registry(path)
    out = []
    for exp in reg["experiments"]:
        member = group is None or exp["group"] == group or (include_refs and group in exp.get("also_in", []))
        if not member:
            continue
        if not include_inactive and not exp.get("active", True):
            continue
        out.append(resolve(reg["defaults"], exp))
    return out


def describe(group=None, out_root=None):
    out_root = out_root or config.out_dir()
    rows = []
    for e in list_experiments(group, include_inactive=True, include_refs=True):
        m = e["model"]
        done = os.path.exists(os.path.join(out_root, e["group"], e["id"], "results.json"))
        rows.append({"id": e["id"], "group": e["group"], "active": e.get("active", True), "done": done,
                     "model": m["name"], "optimizer": e["optimizer"], "lr": e["lr"],
                     "scheduler": e["scheduler"], "aug": e["augmentation"], "balance": e["balance"],
                     "epochs": e["epochs"], "reference": e["group"] != group and group is not None,
                     "description": e.get("description", "")})
    return pd.DataFrame(rows)


def exp_dir_of(cfg, out_root=None):
    return os.path.join(out_root or config.out_dir(), cfg["group"], cfg["id"])


# Run
def _derived_metrics(history, best_epoch):
    h = pd.DataFrame(history)
    best = h[h["epoch"] == best_epoch].iloc[0]
    target = 0.9 * h["val_bal_acc"].max()
    return {
        "epochs_run": int(len(h)),
        "best_epoch": int(best_epoch),
        "peak_val_acc": float(h["val_acc"].max()),
        "peak_val_bal_acc": float(h["val_bal_acc"].max()),
        "peak_val_macro_f1": float(h["val_macro_f1"].max()),
        "val_acc_at_best": float(best["val_acc"]),
        
        "generalization_gap_acc": float(best["train_clean_acc"] - best["val_acc"]),
        "generalization_gap_loss": float(best["val_loss"] - best["train_clean_loss"]),
   
        "epochs_to_90pct_best": int(h.loc[h["val_bal_acc"] >= target, "epoch"].iloc[0]),
        "train_time_min": float(h["epoch_time_s"].sum() / 60),
        "mean_epoch_time_s": float(h["epoch_time_s"].mean()),
    }


def run_experiment(exp, out_root=None, force=False, overrides=None, verbose=True, keep_last=False):

    cfg = get_experiment(exp) if isinstance(exp, str) else dict(exp)
    cfg = deep_merge(cfg, overrides)
    exp_dir = exp_dir_of(cfg, out_root)
    res_path = os.path.join(exp_dir, "results.json")
    if os.path.exists(res_path) and not force:
        print(f"[skip] {cfg['id']} already done -> {res_path}")
        return load_json(res_path)
    if force and os.path.isdir(exp_dir):
        shutil.rmtree(exp_dir)
    os.makedirs(exp_dir, exist_ok=True)

    set_seed(cfg.get("seed", config.SEED))
    device = get_device()
    torch.backends.cudnn.benchmark = True

    loaders, info = get_dataloaders(cfg)
    model = build_model(cfg["model"], info["num_classes"]).to(device)
    save_json(cfg, os.path.join(exp_dir, "config.json"))
    print(f"\n=== {cfg['id']} === {cfg.get('description', '')}")
    print(f"  device={device} | train/val/test = {info['n_train']}/{info['n_val']}/{info['n_test']} | "
          f"params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    history, best_epoch = fit(model, loaders, info, cfg, exp_dir, device, verbose)

    # test the BEST checkpoint on the unseen test split 
    best = torch.load(os.path.join(exp_dir, "best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    rep = test_report(model, loaders["test"], info, device)
    summary = model_summary(model, cfg.get("img_size", 224), device)

    test_df = info["test_df"][["filename", "patient", "subtype", "magnification"]].copy()
    test_df["target"] = [info["class_names"][i] for i in rep["_targets"]]
    test_df["pred"] = [info["class_names"][i] for i in rep["_preds"]]
    test_df["confidence"] = rep["_probs"].max(1)
    test_df.to_csv(os.path.join(exp_dir, "test_predictions.csv"), index=False)

    results = {"id": cfg["id"], "group": cfg["group"], "also_in": cfg.get("also_in", []),
               "description": cfg.get("description", ""),
               "class_names": info["class_names"], "n_train": info["n_train"],
               "train_fraction": cfg.get("train_fraction", 1.0), "model_name": cfg["model"]["name"],
               "model_mode": cfg["model"].get("mode", "scratch"),
               **summary, **_derived_metrics(history, best_epoch),
               **{k: v for k, v in rep.items() if not k.startswith("_")}}
    save_json(results, res_path)

    title = f"{cfg['id']}"
    for fig in (
        plots.plot_learning_curves(history, title, os.path.join(exp_dir, "curves.png")),
        plots.plot_per_class_accuracy(history, info["class_names"], title, os.path.join(exp_dir, "per_class.png")),
        plots.plot_confusion_matrix(rep["confusion_matrix_normalized"], info["class_names"], True,
                                    f"{title} - test", os.path.join(exp_dir, "confusion_matrix.png")),
    ):
        plt.close(fig)

    if not keep_last and os.path.exists(os.path.join(exp_dir, "last.pt")):
        os.remove(os.path.join(exp_dir, "last.pt"))  # saves Drive space
    print(f"  TEST acc={results['test_acc']:.3f} bal_acc={results['test_bal_acc']:.3f} "
          f"macroF1={results['test_macro_f1']:.3f} patient_acc={results['test_patient_acc']:.3f} "
          f"| {results['train_time_min']:.1f} min")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def run_group(group, out_root=None, only=None, force=False, overrides=None, include_refs=False):
    
    results = []
    for cfg in list_experiments(group, include_refs=include_refs):
        if only and cfg["id"] not in only:
            continue
        results.append(run_experiment(cfg, out_root, force=force, overrides=overrides))
    return results


# Collect / display

def collect_results(group=None, out_root=None, include_refs=True):

    out_root = out_root or config.out_dir()
    groups = None if group is None else ([group] if isinstance(group, str) else list(group))
    order = {e["id"]: i for i, e in enumerate(load_registry()["experiments"])}
    res = []
    for p in glob.glob(os.path.join(out_root, "*", "*", "results.json")):
        r = load_json(p)
        member = groups is None or r["group"] in groups or (
            include_refs and any(g in r.get("also_in", []) for g in groups))
        if member:
            res.append(r)
    return sorted(res, key=lambda r: order.get(r["id"], 10 ** 6))


def benchmark_table(group=None, out_root=None):

    res = collect_results(group, out_root)
    if not res:
        print("No finished experiments found.")
        return pd.DataFrame()
    return results_table(res).round(4)


def load_history(exp_id, out_root=None):
    cfg = get_experiment(exp_id)
    return pd.read_csv(os.path.join(exp_dir_of(cfg, out_root), "history.csv"))


def show_experiment(exp_id, out_root=None):
    cfg = get_experiment(exp_id)
    d = exp_dir_of(cfg, out_root)
    res = load_json(os.path.join(d, "results.json"))
    hist = pd.read_csv(os.path.join(d, "history.csv")).to_dict("records")
    plots.plot_learning_curves(hist, exp_id)
    plots.plot_per_class_accuracy(hist, res["class_names"], exp_id)
    plots.plot_confusion_matrix(res["confusion_matrix_normalized"], res["class_names"], True, f"{exp_id} - test")
    plt.show()
    keys = ["test_acc", "test_bal_acc", "test_macro_f1", "test_patient_acc", "peak_val_acc",
            "generalization_gap_acc", "params_total", "latency_ms_bs1", "train_time_min"]
    return pd.Series({k: res[k] for k in keys}, name=exp_id)


def export_checkpoints(dest, out_root=None):
    out_root = out_root or config.out_dir()
    os.makedirs(dest, exist_ok=True)
    n = 0
    for p in sorted(glob.glob(os.path.join(out_root, "*", "*", "best.pt"))):
        shutil.copy(p, os.path.join(dest, os.path.basename(os.path.dirname(p)) + ".pt"))
        n += 1
    print(f"Copied {n} checkpoints to {dest}")
