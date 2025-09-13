#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Run inference on the test split and save:
# - CSV with per-sample probabilities/predictions
# - JSON with confusion/metrics
# Uses best checkpoint and best_threshold.json if available.

import os, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

from code.config import (
    ROOT, LABELS_DIR, MODELS_DIR,
    RESULTS_METRICS_DIR, RESULTS_PREDS_DIR,
    TRAIN_MODALITY, TRAIN_ARCH, TRAIN_PRETRAINED, TRAIN_FREEZE_BACKBONE,
    TRAIN_CHANNELS_LAST, TRAIN_THRESHOLD,
    EVAL_BATCH_SIZE, EVAL_WORKERS, EVAL_PREFETCH,
)

# ---------- utils ----------
def map_path_for_modality(raw_rel_path: str, modality: str) -> str:
    """Map RAW → {raw,hough,combined} paths with our naming."""
    p = raw_rel_path.replace("\\", "/")
    if modality == "raw" or "/patches/raw/" not in p:
        return p
    if modality == "hough":
        q = p.replace("/patches/raw/", "/patches/hough/")
        q = q.replace("_r.png", "_h.png").replace("_r_", "_h_")
        return q
    if modality == "combined":
        q = p.replace("/patches/raw/", "/patches/combined/")
        q = q.replace("_r.png", "_c.png").replace("_r_", "_c_")
        return q
    return p

def make_resnet18(pretrained=True, grayscale=True, freeze_backbone=False):
    """ResNet18 1-channel in, single logit out."""
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(weights=None)
    if grayscale and model.conv1.in_channels == 3:
        with torch.no_grad():
            w = model.conv1.weight
            new = nn.Conv2d(1, w.shape[0], kernel_size=7, stride=2, padding=3, bias=False)
            new.weight.copy_(w.mean(dim=1, keepdim=True))
        model.conv1 = new
    if freeze_backbone:
        for n, p in model.named_parameters():
            if not n.startswith("fc."):
                p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model

def arch_dirname(arch: str) -> str:
    return "resnet" if arch.startswith("resnet") else arch

def metrics_from_preds(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> dict:
    """Compute confusion and metrics at threshold thr."""
    y_pred = (y_prob >= thr).astype(np.int32)
    TP = int(np.sum((y_pred == 1) & (y_true == 1)))
    TN = int(np.sum((y_pred == 0) & (y_true == 0)))
    FP = int(np.sum((y_pred == 1) & (y_true == 0)))
    FN = int(np.sum((y_pred == 0) & (y_true == 1)))
    acc  = (TP + TN) / max(1, TP + TN + FP + FN)
    prec = TP / max(1, TP + FP)
    rec  = TP / max(1, TP + FN)
    tnr  = TN / max(1, TN + FP)
    f1   = (2 * prec * rec) / max(1e-9, prec + rec)
    f2   = (5 * prec * rec) / max(1e-9, (4 * prec + rec))
    balacc = 0.5 * (rec + tnr)
    return {"threshold": float(thr), "acc": acc, "prec1": prec, "recall1": rec,
            "tnr": tnr, "f1": f1, "f2": f2, "balacc": balacc,
            "tp": TP, "fp": FP, "tn": TN, "fn": FN, "n": int(len(y_true))}

# ---------- dataset ----------
class TestDataset(torch.utils.data.Dataset):
    """Load grayscale PNGs; normalize to mean=0.5,std=0.25."""
    def __init__(self, csv_path: Path, repo_root: Path, modality: str):
        df = pd.read_csv(csv_path)
        self.paths  = [repo_root / map_path_for_modality(p, modality) for p in df["path"].tolist()]
        self.labels = df["label"].astype(np.float32).values
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.25]),
        ])

    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("L")
        x   = self.tf(img)
        y   = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return x, y, str(self.paths[idx])

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Test/evaluate on test split")
    ap.add_argument("--arch", choices=["resnet18", "cnn"], default=None)
    ap.add_argument("--modality", choices=["raw", "hough", "combined"], default=None)
    ap.add_argument("--ckpt", choices=["best", "last"], default="best")
    args = ap.parse_args()

    arch     = args.arch or TRAIN_ARCH
    modality = args.modality or TRAIN_MODALITY

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_csv = LABELS_DIR / "test.csv"
    if not test_csv.exists():
        raise FileNotFoundError("labels/test.csv not found. Run split first.")

    ds = TestDataset(test_csv, ROOT, modality)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=EVAL_BATCH_SIZE, shuffle=False,
        num_workers=EVAL_WORKERS, pin_memory=torch.cuda.is_available(),
        prefetch_factor=EVAL_PREFETCH if EVAL_WORKERS > 0 else None,
        persistent_workers=(EVAL_WORKERS > 0),
    )

    # model + weights
    if arch == "cnn":
        raise SystemExit("CNN not implemented yet. Use --arch resnet18.")
    model = make_resnet18(pretrained=TRAIN_PRETRAINED, grayscale=True, freeze_backbone=TRAIN_FREEZE_BACKBONE)
    if torch.cuda.is_available() and TRAIN_CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last)
    model = model.to(device)

    ckpt_dir = MODELS_DIR / arch_dirname(arch) / modality
    ckpt_path = ckpt_dir / (f"{args.ckpt}.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    # choose threshold: prefer saved best_threshold.json
    best_thr_file = ckpt_dir / "best_threshold.json"
    if best_thr_file.exists():
        best_thr = float(json.loads(best_thr_file.read_text())["threshold"])
    else:
        best_thr = float(TRAIN_THRESHOLD)

    # inference
    probs, labels, relpaths = [], [], []
    with torch.no_grad():
        for xb, yb, pth in loader:
            xb = xb.to(device, non_blocking=True)
            if TRAIN_CHANNELS_LAST and xb.is_cuda:
                xb = xb.to(memory_format=torch.channels_last)
            logits = model(xb)
            pr = torch.sigmoid(logits).view(-1).detach().cpu().numpy()
            lb = yb.view(-1).detach().cpu().numpy()
            probs.append(pr); labels.append(lb); relpaths += pth

    y_prob = np.concatenate(probs, axis=0)
    y_true = np.concatenate(labels, axis=0).astype(np.int32)
    mets   = metrics_from_preds(y_true, y_prob, best_thr)

    # save predictions CSV
    out_pred_dir = RESULTS_PREDS_DIR / arch_dirname(arch) / modality
    out_pred_dir.mkdir(parents=True, exist_ok=True)
    pred_csv = out_pred_dir / "test_preds.csv"
    pd.DataFrame({"path": relpaths, "label": y_true, "prob": y_prob, "pred": (y_prob >= best_thr).astype(int)}).to_csv(pred_csv, index=False)

    # save metrics JSON
    out_met_dir = RESULTS_METRICS_DIR / arch_dirname(arch) / modality
    out_met_dir.mkdir(parents=True, exist_ok=True)
    met_json = out_met_dir / "test_metrics.json"
    with open(met_json, "w", encoding="utf-8") as f:
        json.dump({"arch": arch, "modality": modality, "ckpt": args.ckpt, "threshold": best_thr, "metrics": mets}, f, indent=2)

    # print short summary
    print(f"[OK] Test done | arch={arch} modality={modality} thr={best_thr:.3f}")
    print(f"acc={mets['acc']*100:.2f}%  prec1={mets['prec1']:.4f}  recall1={mets['recall1']:.4f}  "
          f"tnr={mets['tnr']:.4f}  balacc={mets['balacc']:.4f}")
    print(f"TP={mets['tp']} FP={mets['fp']} TN={mets['tn']} FN={mets['fn']}")
    print(f"preds:  {pred_csv}")
    print(f"metrics:{met_json}")

if __name__ == "__main__":
    main()
