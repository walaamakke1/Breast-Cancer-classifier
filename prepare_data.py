"""
1. we Clean first by dropping files whose name does not follow the BreaKHis naming scheme, unreadable / corrupt images ,
the same file found twice, exact pixel duplicates ,files whose benign/malignant letter contradicts their subtype
2. Then we did Patient level split (train / val / test). All images of a patient go to ONE split
3. We Resizes every image
4. Compute the per channel mean/std on the TRAIN split 
5. We write metadata.csv, stats.json, cleaning_report.json and a single .zip
"""

import argparse
import hashlib
import os
import re
import shutil
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image

from config import BENIGN_SUBTYPES, SEED, SUBTYPES
from utils import save_json

# SOB_B_A-14-22549AB-40-001.png -> procedure SOB, class B, subtype A, patient 14-22549AB, 40X, image 001
FILENAME_RE = re.compile(
    r"^SOB_(?P<tumor>[BM])_(?P<subtype>[A-Z]+)-(?P<patient>\d+-\d+[A-Z]*)-(?P<mag>\d+)-(?P<seq>\d+)\.png$",
    re.IGNORECASE,
)
STANDARD_SIZE = (700, 460)

# 1. Discover + parse

def parse_filename(name):
    m = FILENAME_RE.match(name)
    if m is None:
        return None
    d = m.groupdict()
    return {
        "filename": name,
        "tumor": d["tumor"].upper(),
        "subtype": d["subtype"].upper(),
        "patient": d["patient"].upper(),
        "magnification": int(d["mag"]),
        "seq": int(d["seq"]),
    }


def find_images(raw_dir):
    paths = []
    for root, _, files in os.walk(raw_dir):
        for f in files:
            if f.lower().endswith(".png"):
                paths.append(os.path.join(root, f))
    return sorted(paths)


def _inspect(path):
    
    try:
        with open(path, "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
        with Image.open(path) as im:
            im.verify()  
        with Image.open(path) as im:
            im.convert("RGB").load()
            w, h = im.size
        return path, md5, w, h
    except Exception:
        return path, None, None, None


# 2. Clean

def build_clean_table(raw_dir, workers=4):
    report = Counter()
    paths = find_images(raw_dir)
    report["png_files_found"] = len(paths)
    if not paths:
        raise FileNotFoundError(f"No .png files found under {raw_dir}. Did the archive extract correctly?")

    # naming scheme
    rows, bad_names = [], []
    for p in paths:
        rec = parse_filename(os.path.basename(p))
        if rec is None:
            bad_names.append(os.path.basename(p))
            continue
        rec["raw_path"] = p
        rows.append(rec)
    report["dropped_bad_filename"] = len(bad_names)
    df = pd.DataFrame(rows)

    # if the same file name found in several folders so keep the first copy
    before = len(df)
    df = df.drop_duplicates(subset="filename", keep="first").reset_index(drop=True)
    report["dropped_duplicate_filename"] = before - len(df)

    #  corrupt files, size and content hash
    with Pool(workers) as pool:
        info = pool.map(_inspect, df["raw_path"].tolist(), chunksize=32)
    info = pd.DataFrame(info, columns=["raw_path", "md5", "width", "height"])
    df = df.merge(info, on="raw_path")
    corrupt = df["md5"].isna()
    report["dropped_corrupt"] = int(corrupt.sum())
    df = df[~corrupt].reset_index(drop=True)

    # exact pixel duplicates under different names
    before = len(df)
    df = df.drop_duplicates(subset="md5", keep="first").reset_index(drop=True)
    report["dropped_duplicate_content"] = before - len(df)

    # labels must be consistent 
    known = df["subtype"].isin(SUBTYPES)
    expected_tumor = np.where(df["subtype"].isin(BENIGN_SUBTYPES), "B", "M")
    consistent = known & (df["tumor"] == expected_tumor)
    report["dropped_inconsistent_label"] = int((~consistent).sum())
    df = df[consistent].reset_index(drop=True)

    # non-standard sizes are kept (we will do resizing) but reported
    nonstd = (df["width"] != STANDARD_SIZE[0]) | (df["height"] != STANDARD_SIZE[1])
    report["kept_nonstandard_size"] = int(nonstd.sum())

    # patients whose images carry more than one subtype -> reported
    multi = df.groupby("patient")["subtype"].nunique()
    report["patients_with_multiple_subtypes"] = multi[multi > 1].index.tolist()

    report["images_after_cleaning"] = len(df)
    report["patients_after_cleaning"] = int(df["patient"].nunique())
    report["bad_filename_examples"] = bad_names[:20]
    return df, dict(report)


# 3. Patient-level stratified split

def patient_split(df, val_frac=0.15, test_frac=0.15, seed=SEED):
   
    patient_subtype = df.groupby("patient")["subtype"].agg(lambda s: s.value_counts().index[0])
    rng = np.random.RandomState(seed)
    assignment = {}
    for subtype in SUBTYPES:
        patients = sorted(patient_subtype[patient_subtype == subtype].index)
        rng.shuffle(patients)
        n = len(patients)
        if n == 0:
            continue
        n_test = max(1, int(round(n * test_frac))) if n >= 2 else 0
        n_val = max(1, int(round(n * val_frac))) if n >= 3 else 0
        for i, p in enumerate(patients):
            assignment[p] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
    return pd.Series(assignment, name="split")


# 4. Resize and cache
def _resize_one(args):
    src, dst, short_side = args
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = short_side / min(w, h)
        im = im.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        im.save(dst, "JPEG", quality=95)
    return dst


def compute_mean_std(paths, max_images=2000, seed=SEED):
    rng = np.random.RandomState(seed)
    paths = list(paths)
    if len(paths) > max_images:
        paths = list(rng.choice(paths, max_images, replace=False))
    s, s2, n = np.zeros(3), np.zeros(3), 0
    for p in paths:
        with Image.open(p) as im:
            x = np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0
        x = x.reshape(-1, 3)
        s += x.sum(0)
        s2 += (x ** 2).sum(0)
        n += x.shape[0]
    mean = s / n
    std = np.sqrt(s2 / n - mean ** 2)
    return mean.round(4).tolist(), std.round(4).tolist()


def prepare(raw_dir, out_dir, zip_path=None, short_side=256, val_frac=0.15, test_frac=0.15,
            seed=SEED, workers=4):
    print(f"[1/5] Scanning + cleaning {raw_dir} ...")
    df, report = build_clean_table(raw_dir, workers=workers)
    for k, v in report.items():
        if k != "bad_filename_examples":
            print(f"      {k}: {v}")

    print("[2/5] Patient-level split ...")
    split = patient_split(df, val_frac, test_frac, seed)
    df["split"] = df["patient"].map(split)

    # Integer labels for both tasks
    df["label_subtype"] = df["subtype"].map({s: i for i, s in enumerate(SUBTYPES)})
    df["label_binary"] = (df["tumor"] == "M").astype(int)

    print(f"[3/5] Resizing {len(df)} images (short side = {short_side}px) ...")
    df["path"] = [os.path.join("images", st, os.path.splitext(fn)[0] + ".jpg")
                  for st, fn in zip(df["subtype"], df["filename"])]
    jobs = [(src, os.path.join(out_dir, rel), short_side) for src, rel in zip(df["raw_path"], df["path"])]
    with Pool(workers) as pool:
        pool.map(_resize_one, jobs, chunksize=32)

    print("[4/5] Computing train mean/std ...")
    train_paths = [os.path.join(out_dir, p) for p in df.loc[df["split"] == "train", "path"]]
    mean, std = compute_mean_std(train_paths, seed=seed)
    print(f"      mean={mean} std={std}")

    cols = ["path", "filename", "tumor", "subtype", "patient", "magnification", "seq",
            "split", "label_subtype", "label_binary", "width", "height", "md5"]
    df = df[cols].sort_values(["split", "subtype", "patient", "magnification", "seq"]).reset_index(drop=True)
    df.to_csv(os.path.join(out_dir, "metadata.csv"), index=False)

    stats = {"mean": mean, "std": std, "short_side": short_side, "seed": seed,
             "val_frac": val_frac, "test_frac": test_frac}
    save_json(stats, os.path.join(out_dir, "stats.json"))
    report["split_images"] = df["split"].value_counts().to_dict()
    report["split_patients"] = df.groupby("split")["patient"].nunique().to_dict()
    report["split_by_subtype"] = pd.crosstab(df["subtype"], df["split"]).to_dict()
    save_json(report, os.path.join(out_dir, "cleaning_report.json"))
    print(pd.crosstab(df["subtype"], df["split"], margins=True))

    if zip_path:
        print(f"[5/5] Zipping to {zip_path} ...")
        os.makedirs(os.path.dirname(os.path.abspath(zip_path)), exist_ok=True)
        base = os.path.splitext(os.path.abspath(zip_path))[0]
        # the zip contains a top-level folder named like out_dir (e.g. breakhis/)
        shutil.make_archive(base, "zip", root_dir=os.path.dirname(os.path.abspath(out_dir)),
                            base_dir=os.path.basename(os.path.abspath(out_dir)))
    print("Done.")
    return df, report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="folder that contains the extracted BreaKHis archive")
    ap.add_argument("--out", required=True, help="output folder for the prepared dataset")
    ap.add_argument("--zip", default=None, help="optional: write the prepared dataset to this .zip")
    ap.add_argument("--short-side", type=int, default=256)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    a = ap.parse_args()
    prepare(a.raw, a.out, a.zip, a.short_side, a.val_frac, a.test_frac, a.seed, a.workers)


if __name__ == "__main__":
    main()
