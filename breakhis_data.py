#Dataset, transforms  and DataLoaders for the prepared BreaKHis data

import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T
from torchvision.transforms import v2

import config
from utils import load_json

# Dataset
 
class BreakHisDataset(Dataset):
   
    def __init__(self, df, root, label_col, transform=None):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.labels = self.df[label_col].to_numpy()
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        idx = int(idx)  # also accept tensor / numpy indices
        path = os.path.join(self.root, self.df.at[idx, "path"])
        with Image.open(path) as im:
            image = im.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.labels[idx])


def load_metadata(root=None, magnifications=None):
    root = root or config.data_dir()
    df = pd.read_csv(os.path.join(root, "metadata.csv"))
    if magnifications:
        df = df[df["magnification"].isin(magnifications)].reset_index(drop=True)
    return df


def get_norm_stats(pretrained, root=None):
    if pretrained:
        return config.IMAGENET_MEAN, config.IMAGENET_STD
    stats_path = os.path.join(root or config.data_dir(), "stats.json")
    if os.path.exists(stats_path):
        s = load_json(stats_path)
        return tuple(s["mean"]), tuple(s["std"])
    return config.IMAGENET_MEAN, config.IMAGENET_STD

# Transforms
def get_transforms(augmentation="basic", img_size=224, mean=config.IMAGENET_MEAN,
                   std=config.IMAGENET_STD, train=True):
    """
    none   : resize + center crop
    basic  : random crop + horizontal/vertical flips + 90-degree rotations
    strong : basic + random-resized-crop + affine + colour jitter 
    MixUp / CutMix are applied on whole batches in engine.py 
    """
    normalize = [T.ToTensor(), T.Normalize(mean, std)]
    if not train or augmentation == "none":
        return T.Compose([T.Resize(img_size), T.CenterCrop(img_size), *normalize])

    if augmentation == "basic":
        aug = [
            T.Resize(img_size),
            T.RandomCrop(img_size),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomChoice([T.RandomRotation((a, a)) for a in (0, 90, 180, 270)]),
        ]
    elif augmentation == "strong":
        aug = [
            T.RandomResizedCrop(img_size, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomChoice([T.RandomRotation((a, a)) for a in (0, 90, 180, 270)]),
            T.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        ]
    else:
        raise ValueError(f"Unknown augmentation '{augmentation}' (use none / basic / strong)")
    return T.Compose([*aug, *normalize])


def get_batch_mixer(num_classes, mixup=0.0, cutmix=0.0):
    """Batch-level MixUp / CutMix  , it returns None when both are disabled
    alpha > 0 enables the method; if both are enabled one is chosen at random per batch"""
    choices = []
    if mixup > 0:
        choices.append(v2.MixUp(alpha=mixup, num_classes=num_classes))
    if cutmix > 0:
        choices.append(v2.CutMix(alpha=cutmix, num_classes=num_classes))
    if not choices:
        return None
    return choices[0] if len(choices) == 1 else v2.RandomChoice(choices)


# Class balancing

def class_counts(labels, num_classes):
    return np.bincount(np.asarray(labels), minlength=num_classes)


def class_weights(labels, num_classes):
    counts = class_counts(labels, num_classes).astype(np.float64)
    w = 1.0 / np.maximum(counts, 1)
    w = w * num_classes / w.sum()
    return torch.tensor(w, dtype=torch.float32)


def balanced_sampler(labels, num_classes, seed=config.SEED):
    counts = class_counts(labels, num_classes)
    sample_w = 1.0 / counts[np.asarray(labels)]
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.as_tensor(sample_w, dtype=torch.double),
                                 num_samples=len(labels), replacement=True, generator=g)


def subsample_train(df, label_col, fraction, seed=config.SEED):
    if fraction >= 1.0:
        return df
    parts = []
    for _, g in df.groupby(label_col):
        n = max(1, int(round(len(g) * fraction)))
        parts.append(g.sample(n=n, random_state=seed))
    return pd.concat(parts).reset_index(drop=True)

# Loaders

def get_dataloaders(cfg, root=None):
    #Build train / val / test DataLoaders from an experiment config dict.
    
    root = root or config.data_dir()
    task = cfg.get("task", config.get_task())
    label_col = config.get_label_column(task)
    class_names = config.get_class_names(task)
    num_classes = len(class_names)

    df = load_metadata(root, cfg.get("magnifications"))
    train_df = subsample_train(df[df["split"] == "train"], label_col,
                               cfg.get("train_fraction", 1.0), cfg.get("seed", config.SEED))
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    pretrained = cfg.get("model", {}).get("pretrained", False)
    mean, std = get_norm_stats(pretrained, root)
    img_size = cfg.get("img_size", 224)
    train_tf = get_transforms(cfg.get("augmentation", "basic"), img_size, mean, std, train=True)
    eval_tf = get_transforms("none", img_size, mean, std, train=False)

    train_ds = BreakHisDataset(train_df, root, label_col, train_tf)
    val_ds = BreakHisDataset(val_df, root, label_col, eval_tf)
    test_ds = BreakHisDataset(test_df, root, label_col, eval_tf)

    bs = cfg.get("batch_size", 64)
    nw = cfg.get("num_workers", 2)
    pin = torch.cuda.is_available()
    sampler = None
    if cfg.get("balance", "sampler") == "sampler":
        sampler = balanced_sampler(train_ds.labels, num_classes, cfg.get("seed", config.SEED))

    loaders = {
        "train": DataLoader(train_ds, batch_size=bs, shuffle=sampler is None, sampler=sampler,
                            num_workers=nw, pin_memory=pin, drop_last=len(train_ds) > bs,
                            persistent_workers=nw > 0),
        "val": DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin,
                          persistent_workers=nw > 0),
        "test": DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin),
        
        "train_eval": DataLoader(
            BreakHisDataset(train_df.sample(n=min(len(train_df), cfg.get("train_eval_size", 1000)),
                                            random_state=cfg.get("seed", config.SEED)),
                            root, label_col, eval_tf),
            batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin, persistent_workers=nw > 0),
    }
    info = {
        "task": task, "label_col": label_col, "class_names": class_names, "num_classes": num_classes,
        "mean": list(mean), "std": list(std),
        "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
        "train_class_counts": class_counts(train_ds.labels, num_classes).tolist(),
        "class_weights": class_weights(train_ds.labels, num_classes).tolist(),
        "test_df": test_df.reset_index(drop=True),
    }
    return loaders, info


def denormalize(images, mean, std):
    #Undo Normalize for plotting
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    return (images.cpu() * std + mean).clamp(0, 1)
