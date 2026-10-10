import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score

from breakhis_data import get_batch_mixer
from models import TransferNet
from utils import Timer, save_json

# Loss
class FocalLoss(nn.Module):

    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits, target):
        num_classes = logits.shape[1]
        log_p = F.log_softmax(logits.float(), dim=1)
        if target.ndim == 1:
            target = F.one_hot(target, num_classes).float()
        if self.label_smoothing > 0:
            target = target * (1 - self.label_smoothing) + self.label_smoothing / num_classes
        loss = -target * (1 - log_p.exp()) ** self.gamma * log_p
        if self.weight is not None:
            loss = loss * self.weight
        return loss.sum(dim=1).mean()


def build_criterion(cfg, class_weights=None):
    weight = class_weights if cfg.get("balance") == "loss_weights" else None
    ls = cfg.get("label_smoothing", 0.0)
    if cfg.get("loss", "ce") == "focal":
        return FocalLoss(gamma=cfg.get("focal_gamma", 2.0), weight=weight, label_smoothing=ls)
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=ls)


# Optimiser + scheduler

def build_optimizer(model, cfg):
    lr, wd = cfg.get("lr", 1e-3), cfg.get("weight_decay", 0.0)
    if isinstance(model, TransferNet):
        params = model.param_groups(lr, cfg.get("model", {}).get("backbone_lr_mult", 1.0))
    else:
        params = [p for p in model.parameters() if p.requires_grad]
    name = cfg.get("optimizer", "adam").lower()
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=cfg.get("momentum", 0.9), weight_decay=wd,
                               nesterov=cfg.get("nesterov", False))
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    raise ValueError(f"Unknown optimizer '{name}' (sgd / adam / adamw)")


def build_scheduler(optimizer, cfg):
    """ 'none' 
        'step' (x0.1 every epochs/3) 
        'cosine' (to 0 over all epochs) 
        'plateau' (x0.1 on val-loss plateau) """
    name, epochs = cfg.get("scheduler", "none"), cfg.get("epochs", 30)
    if name == "none":
        return None
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3)
    raise ValueError(f"Unknown scheduler '{name}' (none / step / cosine / plateau)")

# One epoch

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, mixer=None):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    use_amp = scaler is not None
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        targets = labels
        if mixer is not None:
            images, targets = mixer(images, labels)  # soft labels [B, C]

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        n += labels.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss, n = 0.0, 0
    all_probs, all_targets = [], []
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            outputs = model(images)
        loss = F.cross_entropy(outputs.float(), labels) if criterion is None else criterion(outputs.float(), labels)
        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)
        all_probs.append(torch.softmax(outputs.float(), 1).cpu())
        all_targets.append(labels.cpu())
    probs = torch.cat(all_probs).numpy()
    targets = torch.cat(all_targets).numpy()
    preds = probs.argmax(1)
    per_class = []
    for c in range(num_classes):
        mask = targets == c
        per_class.append(float((preds[mask] == c).mean()) if mask.any() else float("nan"))
    metrics = {
        "loss": total_loss / max(n, 1),
        "acc": float((preds == targets).mean()),
        "bal_acc": float(balanced_accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro", labels=list(range(num_classes)),
                                   zero_division=0)),
        "per_class_acc": per_class,
    }
    return metrics, preds, targets, probs

def make_checkpoint(model, cfg, info, epoch, metrics):
    return {
        "model_state": model.state_dict(),
        "exp_config": cfg,
        "model_kind": getattr(model, "model_kind", "unknown"),
        "class_names": info["class_names"],
        "mean": info["mean"], "std": info["std"],
        "img_size": cfg.get("img_size", 224),
        "epoch": epoch, "val_metrics": metrics,
    }


def fit(model, loaders, info, cfg, exp_dir, device, verbose=True):
    os.makedirs(exp_dir, exist_ok=True)
    num_classes, class_names = info["num_classes"], info["class_names"]
    weights = torch.tensor(info["class_weights"], dtype=torch.float32).to(device)
    criterion = build_criterion(cfg, weights).to(device)
    eval_criterion = nn.CrossEntropyLoss()  # val loss always plain CE -> comparable across runs
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler("cuda") if (device.type == "cuda" and cfg.get("amp", True)) else None
    mixer = get_batch_mixer(num_classes, cfg.get("mixup", 0.0), cfg.get("cutmix", 0.0))
    monitor = cfg.get("monitor", "val_bal_acc")
    patience = cfg.get("patience", 0) or 10 ** 9

    history, best_score, best_epoch, start_epoch, bad_epochs = [], -np.inf, -1, 0, 0
    last_path, best_path = os.path.join(exp_dir, "last.pt"), os.path.join(exp_dir, "best.pt")
    if os.path.exists(last_path):
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        if scheduler is not None and state.get("scheduler_state"):
            scheduler.load_state_dict(state["scheduler_state"])
        if scaler is not None and state.get("scaler_state"):
            scaler.load_state_dict(state["scaler_state"])
        history, best_score, best_epoch = state["history"], state["best_score"], state["best_epoch"]
        start_epoch, bad_epochs = state["epoch"] + 1, state["bad_epochs"]
        print(f"  Resuming from epoch {start_epoch + 1}")

    epochs = cfg.get("epochs", 30)
    for epoch in range(start_epoch, epochs):
        with Timer() as t:
            tr_loss, tr_acc = train_one_epoch(model, loaders["train"], criterion, optimizer, device, scaler, mixer)
        val, _, _, _ = evaluate(model, loaders["val"], eval_criterion, device, num_classes)
        tr_clean, _, _, _ = evaluate(model, loaders["train_eval"], eval_criterion, device, num_classes)
        lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step(val["loss"]) if cfg.get("scheduler") == "plateau" else scheduler.step()

        row = {
            "epoch": epoch + 1, "lr": lr, "epoch_time_s": t.seconds,
            "train_loss": tr_loss, "train_acc": tr_acc,
            "train_clean_loss": tr_clean["loss"], "train_clean_acc": tr_clean["acc"],
            "val_loss": val["loss"], "val_acc": val["acc"], "val_bal_acc": val["bal_acc"],
            "val_macro_f1": val["macro_f1"],
        }
        for name, acc in zip(class_names, val["per_class_acc"]):
            row[f"val_acc_{name}"] = acc
        history.append(row)

        score = row[monitor]
        improved = score > best_score
        if improved:
            best_score, best_epoch, bad_epochs = score, epoch + 1, 0
            torch.save(make_checkpoint(model, cfg, info, epoch + 1, row), best_path)
        else:
            bad_epochs += 1

        if verbose:
            print(f"  ep {epoch + 1:3d}/{epochs} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
                  f"val loss {val['loss']:.3f} acc {val['acc']:.3f} bal {val['bal_acc']:.3f} "
                  f"f1 {val['macro_f1']:.3f} | {t.seconds:5.1f}s{'  *' if improved else ''}")

        pd.DataFrame(history).to_csv(os.path.join(exp_dir, "history.csv"), index=False)
        torch.save({
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "history": history, "best_score": best_score, "best_epoch": best_epoch,
            "epoch": epoch, "bad_epochs": bad_epochs,
        }, last_path)

        if bad_epochs >= patience:
            print(f"  Early stopping: no {monitor} improvement for {patience} epochs.")
            break

    save_json({"history": history, "best_epoch": best_epoch, "best_score": best_score, "monitor": monitor},
              os.path.join(exp_dir, "history.json"))
    return history, best_epoch
