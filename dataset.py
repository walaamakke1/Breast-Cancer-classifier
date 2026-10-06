from pathlib import Path

import pandas as pd
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import (
    CLASS_NAMES,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
)


def scan_breakhis(root, magnification=None):

    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset folder does not exist: {root}")
    rows = []

    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue

        parts = path.parts

        label_name = None
        for class_name in CLASS_NAMES:
            if class_name in parts:
                label_name = class_name
                break

        if label_name is None:
            continue

        mag = path.parent.name
        if mag not in {"40X", "100X", "200X", "400X"}:
            continue
        if magnification is not None and mag != magnification:
            continue

        
        pieces = path.stem.split("-")
        if len(pieces) != 5 or not pieces[0].startswith("SOB_"):
            raise ValueError(f"Unexpected BreaKHis filename: {path.name}")
        patient_id = "-".join(pieces[1:3])

        rows.append({
            "path": str(path.relative_to(root)),
            "label_name": label_name,
            "label": CLASS_NAMES.index(label_name),
            "patient_id": patient_id,
            "magnification": mag,
        })

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise RuntimeError(
            f"No BreaKHis images were found under: {root}\n"
            "Check DATASET_ROOT in your notebook and the original BreaKHis folder structure."
        )

    return df


def patient_level_split(df, train_size=0.70, val_size=0.15,
                        test_size=0.15, seed=SEED, attempts=1000):
    
    if min(train_size, val_size, test_size) <= 0:
        raise ValueError("All three split fractions must be positive.")
    if not np.isclose(train_size + val_size + test_size, 1):
        raise ValueError("Split fractions must add to one.")
    if set(df.label) != set(range(len(CLASS_NAMES))):
        raise ValueError("The dataset must contain all eight classes.")
    if not isinstance(attempts, (int, np.integer)) or attempts < 1:
        raise ValueError("attempts must be a positive integer.")
    if df.patient_id.isna().any():
        raise ValueError("Every image must have an identifier group.")

  
    counts = pd.crosstab(df.patient_id, df.label)
    counts = counts.reindex(columns=range(len(CLASS_NAMES)), fill_value=0)
    counts = counts.sort_index()
    if (counts.gt(0).sum(axis=0) < 3).any():
        raise ValueError("Each class needs at least 3 distinct identifier groups.")
    ids = counts.index.to_numpy()
    image_counts = counts.to_numpy()
    has_class = image_counts > 0
    class_totals = image_counts.sum(axis=0)
    desired = np.array([train_size, val_size, test_size])
    n = len(ids)
    n_val = max(1, round(n * val_size))
    n_test = max(1, round(n * test_size))
    n_train = n - n_val - n_test
    if min(n_train, n_val, n_test) < 1:
        raise ValueError("Not enough groups for the requested fractions.")

    rng = np.random.default_rng(seed)
    targets = (n_train, n_val, n_test)
    rare_first = np.argsort(has_class.sum(axis=0), kind="stable")
    best_groups = None
    best_score = float("inf")
    for _ in range(attempts):
        groups = [[], [], []]
        available = np.ones(n, dtype=bool)
        failed = False
       
        for label in rare_first:
            for split in rng.permutation(3):
                if has_class[groups[split], label].any():
                    continue
                candidates = np.flatnonzero(available & has_class[:, label])
                if len(candidates) == 0 or len(groups[split]) >= targets[split]:
                    failed = True
                    break
                chosen = rng.choice(candidates)
                groups[split].append(chosen)
                available[chosen] = False
            if failed:
                break
        if failed:
            continue
      
        remaining = rng.permutation(np.flatnonzero(available))
        offset = 0
        for split, target in enumerate(targets):
            count = target - len(groups[split])
            groups[split].extend(remaining[offset:offset + count])
            offset += count
        split_counts = np.array([image_counts[index].sum(axis=0)
                                 for index in groups])
        
        class_fractions = split_counts / class_totals
        overall_fractions = split_counts.sum(axis=1) / class_totals.sum()
        score = np.mean((class_fractions - desired[:, None]) ** 2)
        score += 0.1 * np.mean((overall_fractions - desired) ** 2)
        if score < best_score:
            best_score = score
            best_groups = [index.copy() for index in groups]

    if best_groups is None:
        raise ValueError("Could not form splits containing every class. "
                         "Inspect class/group counts or adjust split fractions.")
    parts = tuple(df[df.patient_id.isin(ids[index])].reset_index(drop=True)
                  for index in best_groups)
    validate_splits(*parts)
    return parts


def validate_splits(train_df, val_df, test_df):
  
    parts = (train_df, val_df, test_df)
    groups = [set(part.patient_id) for part in parts]
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        if groups[i] & groups[j]:
            raise ValueError("Patient leakage: the saved splits overlap.")
    for part in parts:
        if set(part.label) != set(range(len(CLASS_NAMES))):
            raise ValueError("Each split must contain all 8 classes.")


def load_splits(split_dir):
    parts = tuple(pd.read_csv(Path(split_dir) / f"{name}_split.csv")
                  for name in ["train", "val", "test"])
    validate_splits(*parts)
    return parts


def get_transforms(augment=False):
  
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(
                degrees=15,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
            ),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
                hue=0.03,
            ),
            transforms.ToTensor(),
            normalize,
        ])

    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])


class BreakHisDataset(Dataset):
    def __init__(self, dataframe, root, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        with Image.open(self.root / row["path"]) as source:
            image = source.convert("RGB")
        label = int(row["label"])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def make_loaders(train_df, val_df, test_df, root, augment=False,
                 batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    validate_splits(train_df, val_df, test_df)
    train_dataset = BreakHisDataset(
        train_df,
        root=root,
        transform=get_transforms(augment=augment),
    )
    val_dataset = BreakHisDataset(
        val_df,
        root=root,
        transform=get_transforms(augment=False),
    )
    test_dataset = BreakHisDataset(
        test_df,
        root=root,
        transform=get_transforms(augment=False),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader


def print_split_summary(train_df, val_df, test_df):
    for name, part in [
        ("TRAIN", train_df),
        ("VALIDATION", val_df),
        ("TEST", test_df),
    ]:
        print(
            f"{name:10s}: "
            f"{len(part):5d} images | "
            f"{part['patient_id'].nunique():3d} patients"
        )

    print("\nPatients overlap checks:")
    print(
        "train/val:",
        len(set(train_df.patient_id) & set(val_df.patient_id)),
    )
    print(
        "train/test:",
        len(set(train_df.patient_id) & set(test_df.patient_id)),
    )
    print(
        "val/test:",
        len(set(val_df.patient_id) & set(test_df.patient_id)),
    )
