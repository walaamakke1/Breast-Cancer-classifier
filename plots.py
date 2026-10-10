import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torchvision

from breakhis_data import denormalize


def _save(fig, path):
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")


def plot_learning_curves(history, title="", path=None):
    #Combined learning curves: loss and accuracy across epochs
    h = pd.DataFrame(history)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(h["epoch"], h["train_loss"], label="train loss")
    ax[0].plot(h["epoch"], h["val_loss"], label="val loss")
    ax[0].set_xlabel("epoch"), ax[0].set_ylabel("loss"), ax[0].set_title("Loss")
    ax[1].plot(h["epoch"], h["train_acc"], label="train acc (augmented)", alpha=0.6)
    if "train_clean_acc" in h:
        ax[1].plot(h["epoch"], h["train_clean_acc"], label="train acc (clean)")
    ax[1].plot(h["epoch"], h["val_acc"], label="val acc")
    ax[1].plot(h["epoch"], h["val_bal_acc"], "--", label="val balanced acc")
    ax[1].set_xlabel("epoch"), ax[1].set_ylabel("accuracy"), ax[1].set_title("Accuracy")
    for a in ax:
        a.grid(alpha=0.3), a.legend()
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_per_class_accuracy(history, class_names, title="", path=None):
    #Validation accuracy of every class across epochs
    h = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(8, 4))
    for name in class_names:
        ax.plot(h["epoch"], h[f"val_acc_{name}"], label=name)
    ax.set_xlabel("epoch"), ax.set_ylabel("val accuracy"), ax.set_ylim(-0.02, 1.02)
    ax.set_title(title or "Per-class validation accuracy"), ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_confusion_matrix(cm, class_names, normalized=True, title="", path=None):
    cm = np.asarray(cm, dtype=float)
    fig, ax = plt.subplots(figsize=(1.0 * len(class_names) + 2, 0.9 * len(class_names) + 1.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalized else None)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted"), ax.set_ylabel("true")
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i, j]:.2f}" if normalized else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(title or ("Normalised confusion matrix" if normalized else "Confusion matrix"))
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_class_distribution(df, path=None):
    #Images per subtype / split and per magnification
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    pd.crosstab(df["subtype"], df["split"]).loc[
        [s for s in ["A", "F", "PT", "TA", "DC", "LC", "MC", "PC"] if s in set(df["subtype"])]
    ].plot.bar(ax=ax[0], stacked=True)
    ax[0].set_title("Images per subtype and split"), ax[0].set_ylabel("images")
    pd.crosstab(df["magnification"], df["tumor"]).plot.bar(ax=ax[1])
    ax[1].set_title("Images per magnification (B = benign, M = malignant)")
    fig.tight_layout()
    _save(fig, path)
    return fig


def show_batch(images, labels, class_names, mean, std, preds=None, n=16, path=None):
    #Grid of images with true and predicted labels
    images = denormalize(images[:n], mean, std)
    cols = 8
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2.2 * rows))
    for i, ax in enumerate(np.atleast_1d(axes).flat):
        ax.axis("off")
        if i >= len(images):
            continue
        ax.imshow(images[i].permute(1, 2, 0).numpy())
        t = class_names[int(labels[i])]
        if preds is None:
            ax.set_title(t, fontsize=9)
        else:
            p = class_names[int(preds[i])]
            ax.set_title(f"T:{t} P:{p}", fontsize=8, color="green" if p == t else "red")
    fig.tight_layout()
    _save(fig, path)
    return fig


def compare_curves(histories, metric="val_acc", title="", path=None):
    #Overlay one metric for several runs
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, hist in histories.items():
        h = pd.DataFrame(hist)
        ax.plot(h["epoch"], h[metric], label=name)
    ax.set_xlabel("epoch"), ax.set_ylabel(metric), ax.grid(alpha=0.3)
    ax.set_title(title or metric), ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_heatmap(table, title="", fmt="{:.3f}", path=None):
    fig, ax = plt.subplots(figsize=(1.6 * table.shape[1] + 2, 0.8 * table.shape[0] + 1.5))
    im = ax.imshow(table.values.astype(float), cmap="viridis")
    ax.set_xticks(range(table.shape[1]), table.columns)
    ax.set_yticks(range(table.shape[0]), table.index)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.values[i, j]
            ax.text(j, i, "-" if pd.isna(v) else fmt.format(v), ha="center", va="center", color="w")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_data_efficiency(df, x="train_fraction", y="test_bal_acc", hue="family", title="", path=None):
    #Metric vs fraction of training data
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, g in df.groupby(hue):
        g = g.sort_values(x)
        ax.plot(g[x] * 100, g[y], marker="o", label=name)
    ax.set_xlabel("% of training images"), ax.set_ylabel(y), ax.grid(alpha=0.3), ax.legend()
    ax.set_title(title or "Data efficiency")
    fig.tight_layout()
    _save(fig, path)
    return fig


def make_grid_image(images, mean, std, nrow=8):
    return torchvision.utils.make_grid(denormalize(images, mean, std), nrow=nrow)
