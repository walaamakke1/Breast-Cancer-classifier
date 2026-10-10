import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix

import config
from breakhis_data import get_transforms
from engine import evaluate
from models import build_model, count_parameters
from utils import Timer, get_device


def load_checkpoint(path, device=None):
    device = device or get_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["exp_config"]
    model = build_model(cfg["model"], num_classes=len(ckpt["class_names"]), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), ckpt


@torch.no_grad()
def predict_images(model, images, ckpt, device=None):
    
    device = device or next(model.parameters()).device
    tf = get_transforms("none", ckpt["img_size"], ckpt["mean"], ckpt["std"], train=False)
    batch = []
    for im in images:
        if isinstance(im, str):
            im = Image.open(im)
        batch.append(tf(im.convert("RGB")))
    probs = torch.softmax(model(torch.stack(batch).to(device)).float(), 1).cpu().numpy()
    return [ckpt["class_names"][i] for i in probs.argmax(1)], probs


@torch.no_grad()
def measure_latency(model, img_size=224, device=None, batch_size=1, warmup=10, runs=50):
    device = device or next(model.parameters()).device
    model.eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    for _ in range(warmup):
        model(x)
    with Timer() as t:
        for _ in range(runs):
            model(x)
    return 1000 * t.seconds / (runs * batch_size)


def test_report(model, loader, info, device=None):

    device = device or get_device()
    num_classes, names = info["num_classes"], info["class_names"]
    metrics, preds, targets, probs = evaluate(model, loader, nn.CrossEntropyLoss(), device, num_classes)

    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    test_df = info["test_df"].copy()
    test_df["pred"], test_df["target"] = preds, targets
    test_df["correct"] = test_df["pred"] == test_df["target"]
    per_mag = test_df.groupby("magnification")["correct"].mean().to_dict()
    patient_vote = test_df.groupby("patient").agg(
        target=("target", lambda s: s.value_counts().index[0]),
        pred=("pred", lambda s: s.value_counts().index[0]))
    patient_acc = float((patient_vote["target"] == patient_vote["pred"]).mean())

    # Derived benign vs malignant result: merge the 8 subtypes into 2 classes.
    # Subtypes 0-3 (A, F, PT, TA) are benign, 4-7 (DC, LC, MC, PC) are malignant.
    binary = {}
    if num_classes == len(config.SUBTYPES):
        is_malignant = [s not in config.BENIGN_SUBTYPES for s in config.SUBTYPES]
        t_bin = np.array([is_malignant[t] for t in targets], dtype=int)
        p_bin = np.array([is_malignant[p] for p in preds], dtype=int)
        cm_bin = confusion_matrix(t_bin, p_bin, labels=[0, 1])
        binary = {
            "test_binary_acc": float((t_bin == p_bin).mean()),
            "test_binary_bal_acc": float(balanced_accuracy_score(t_bin, p_bin)),
            "test_binary_sensitivity": float(cm_bin[1, 1] / max(cm_bin[1].sum(), 1)),  # malignant found
            "test_binary_specificity": float(cm_bin[0, 0] / max(cm_bin[0].sum(), 1)),  # benign found
            "binary_confusion_matrix": cm_bin.tolist(),
        }

    return {
        "test_loss": metrics["loss"], "test_acc": metrics["acc"], "test_bal_acc": metrics["bal_acc"],
        "test_macro_f1": metrics["macro_f1"],
        "test_per_class_acc": dict(zip(names, metrics["per_class_acc"])),
        "test_acc_per_magnification": {f"{int(k)}X": float(v) for k, v in per_mag.items()},
        "test_patient_acc": patient_acc,
        **binary,
        "confusion_matrix": cm.tolist(), "confusion_matrix_normalized": cm_norm.round(4).tolist(),
        "classification_report": classification_report(
            targets, preds, labels=list(range(num_classes)), target_names=names, zero_division=0,
            output_dict=True),
        "_preds": preds, "_targets": targets, "_probs": probs,
    }


def model_summary(model, img_size=224, device=None):
    return {
        "params_total": count_parameters(model),
        "params_trainable": count_parameters(model, trainable_only=True),
        "latency_ms_bs1": measure_latency(model, img_size, device, batch_size=1),
        "latency_ms_per_img_bs64": measure_latency(model, img_size, device, batch_size=64, runs=10),
    }


def results_table(results_list):
    rows = []
    for r in results_list:
        rows.append({
            "id": r["id"], "group": r["group"], "description": r.get("description", ""),
            "params (M)": r["params_total"] / 1e6, "trainable (M)": r["params_trainable"] / 1e6,
            "latency bs1 (ms)": r["latency_ms_bs1"], "latency bs64 (ms/img)": r["latency_ms_per_img_bs64"],
            "train time (min)": r["train_time_min"], "epochs run": r["epochs_run"], "best epoch": r["best_epoch"],
            "peak val acc": r["peak_val_acc"], "peak val bal acc": r["peak_val_bal_acc"],
            "gen gap (acc)": r["generalization_gap_acc"], "gen gap (loss)": r["generalization_gap_loss"],
            "epochs to 90% best": r["epochs_to_90pct_best"],
            "test acc": r["test_acc"], "test bal acc": r["test_bal_acc"], "test macro F1": r["test_macro_f1"],
            "test patient acc": r["test_patient_acc"],
            "benign/malignant acc": r.get("test_binary_acc"),
        })
    return pd.DataFrame(rows)
