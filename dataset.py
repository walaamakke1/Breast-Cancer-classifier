from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from config import CLASS_NAMES, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, SEED


def scan_breakhis(root, magnification=None):
    """Create a table with each image's path, class, identifier, and magnification."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset folder does not exist: {root}")
    rows = []

    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue

        label_name = None
        for class_name in CLASS_NAMES:
            if class_name in path.parts:
                label_name = class_name
                break

        if label_name is None:
            continue

        mag = path.parent.name
        if mag not in {"40X", "100X", "200X", "400X"}:
            continue
        if magnification is not None and mag != magnification:
            continue

        # Example: SOB_B_A-14-22549AB-40-001.png -> identifier 14-22549AB.
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

    if not rows:
        raise RuntimeError(f"No BreaKHis images found. Check DATASET_ROOT: {root}")
    return pd.DataFrame(rows)


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

    # Each row is an identifier; columns count its images for each label.
    # Keep all its images together even when it has more than one subtype.
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

