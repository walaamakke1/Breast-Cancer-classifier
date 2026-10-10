
import argparse
import os
import shutil
import tempfile

import numpy as np
import torch
from PIL import Image

import config


def _test_models(num_classes=8, img_size=224, pretrained=False):
    from models import build_model, count_parameters
    from experiments import list_experiments
    seen = set()
    for cfg in list_experiments(include_inactive=True):
        key = str(cfg["model"])
        if key in seen:
            continue
        seen.add(key)
        m = build_model(cfg["model"], num_classes, pretrained=pretrained)
        y = m(torch.randn(2, 3, img_size, img_size))
        assert y.shape == (2, num_classes), y.shape
        print(f"  ok {cfg['id']:<28} out={tuple(y.shape)} params={count_parameters(m) / 1e6:6.2f}M "
              f"trainable={count_parameters(m, True) / 1e6:6.2f}M")
    print("_test_models passed")


def _test_data(cfg_overrides=None):
    from breakhis_data import get_dataloaders
    from experiments import get_experiment
    from utils import deep_merge
    cfg = deep_merge(get_experiment("t1_baseline"), cfg_overrides or {})
    loaders, info = get_dataloaders(cfg)
    x, y = next(iter(loaders["train"]))
    print(f"  batch {tuple(x.shape)} labels {y[:8].tolist()} | train/val/test {info['n_train']}/"
          f"{info['n_val']}/{info['n_test']} | train class counts {info['train_class_counts']}")
    assert x.shape[1:] == (3, cfg["img_size"], cfg["img_size"])
    print("_test_data passed")
    return loaders, info


def _test_training(exp_id="t1_baseline", out_root=None, **overrides):
    from experiments import run_experiment
    out_root = out_root or tempfile.mkdtemp(prefix="breakhis_smoke_")
    ov = {"epochs": 1, "train_fraction": 0.05}
    ov.update(overrides)
    res = run_experiment(exp_id, out_root=out_root, overrides=ov, force=True)
    print(f"_test_training passed -> {out_root}")
    return res, out_root


# Fake dataset (for testing the code without the real images)

def make_fake_breakhis(root, patients_per_subtype=3, images_per_mag=3, size=(140, 92)):
    rng = np.random.RandomState(0)
    for k, st in enumerate(config.SUBTYPES):
        tumor = "B" if st in config.BENIGN_SUBTYPES else "M"
        for p in range(patients_per_subtype):
            pid = f"14-{1000 + 10 * k + p}"
            for mag in config.MAGNIFICATIONS:
                folder = os.path.join(root, "BreaKHis_v1", tumor, st, pid, f"{mag}X")
                os.makedirs(folder, exist_ok=True)
                for i in range(images_per_mag):
                    base = np.array([30 * k % 255, 255 - 25 * k, 120], dtype=np.float32)
                    img = np.clip(base + rng.normal(0, 25, (size[1], size[0], 3)), 0, 255).astype(np.uint8)
                    Image.fromarray(img).save(os.path.join(folder, f"SOB_{tumor}_{st}-{pid}-{mag}-{i + 1:03d}.png"))
    Image.new("RGB", size).save(os.path.join(root, "BreaKHis_v1", "readme_image.png"))  # bad name
    with open(os.path.join(root, "BreaKHis_v1", "SOB_B_A-14-9999-40-001.png"), "wb") as f:
        f.write(b"not an image")  # corrupt
    dup_src = os.path.join(root, "BreaKHis_v1", "B", "A", "14-1000", "40X", "SOB_B_A-14-1000-40-001.png")
    os.makedirs(os.path.join(root, "copy"), exist_ok=True)
    shutil.copy(dup_src, os.path.join(root, "copy", os.path.basename(dup_src)))  # duplicate file name


def _fake_end_to_end():
    os.environ.setdefault("MPLBACKEND", "Agg")
    work = tempfile.mkdtemp(prefix="breakhis_fake_")
    raw, data, out = (os.path.join(work, d) for d in ("raw", "data/breakhis", "outputs"))
    make_fake_breakhis(raw)
    os.environ["BREAKHIS_DATA_DIR"], os.environ["BREAKHIS_OUT_DIR"] = data, out

    from prepare_data import prepare
    _, report = prepare(raw, data, zip_path=os.path.join(work, "breakhis_prepared.zip"), short_side=64, workers=2)
    assert report["dropped_bad_filename"] == 1 and report["dropped_corrupt"] == 1
    assert report["dropped_duplicate_filename"] == 1

    small = {"img_size": 48, "batch_size": 16, "num_workers": 0, "epochs": 2, "train_eval_size": 64}
    _test_models(img_size=64)
    _test_data(small)
    from experiments import benchmark_table, run_experiment
    run_experiment("t1_baseline", overrides=small)
    run_experiment("t2aug_mixup", overrides=small)
    run_experiment("t2bal_focal", overrides=small)
    run_experiment("t2s_plateau", overrides=small)
    run_experiment("t3_resnet18_partial", overrides={**small, "model": {"pretrained": False}})
    run_experiment("t3eff_scratch_25", overrides=small)
    print(benchmark_table().iloc[:, :8].to_string())

    from evaluation import load_checkpoint, predict_images
    model, ckpt = load_checkpoint(os.path.join(out, "task3_transfer", "t3_resnet18_partial", "best.pt"))
    import pandas as pd
    test = pd.read_csv(os.path.join(data, "metadata.csv")).query("split == 'test'").head(4)
    names, probs = predict_images(model, [os.path.join(data, p) for p in test["path"]], ckpt)
    print("  predictions:", names, "true:", test["subtype"].tolist())
    print(f"\nFAKE END-TO-END TEST PASSED ({work})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true", help="run everything on a tiny generated dataset")
    if ap.parse_args().fake:
        _fake_end_to_end()
    else:
        _test_models()
        _test_data()
        _test_training()
