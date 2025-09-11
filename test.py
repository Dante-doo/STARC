#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, gzip, time
from datetime import datetime
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.io import read_image, ImageReadMode

# ---------- utils ----------
def map_path_for_modality(raw_path: str, modality: str) -> str:
    p = raw_path.replace("\\", "/")
    if modality == "raw":
        return p
    if "/patches/raw/" not in p:
        return p
    if modality == "hough":
        p2 = p.replace("/patches/raw/", "/patches/hough/")
        stem, ext = os.path.splitext(p2)
        return f"{stem}_hough{ext}"
    if modality == "combined":
        p2 = p.replace("/patches/raw/", "/patches/combined/")
        stem, ext = os.path.splitext(p2)
        return f"{stem}_combined{ext}"
    raise ValueError("modality inválida")

class CsvDataset(Dataset):
    """Carrega paths do CSV e lê PNGs; retorna também o path usado para avaliação."""
    def __init__(self, csv_path: str, repo_root: str, modality: str, grayscale: bool):
        self.df = pd.read_csv(csv_path)
        self.repo_root = Path(repo_root)
        self.modality = modality
        self.grayscale = grayscale

        ops = [transforms.Resize((224, 224), antialias=True),
               transforms.ConvertImageDtype(torch.float32)]
        if grayscale:
            mean, std = [0.5], [0.25]
        else:
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        ops += [transforms.Normalize(mean=mean, std=std)]
        self.tf = transforms.Compose(ops)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        raw_rel = str(row["path"])
        eval_rel = map_path_for_modality(raw_rel, self.modality)
        full = str(self.repo_root / eval_rel)
        mode = ImageReadMode.GRAY if self.grayscale else ImageReadMode.RGB
        img_u8 = read_image(full, mode=mode)  # uint8 [C,H,W]
        x = self.tf(img_u8)
        y = torch.tensor([float(row["label"])], dtype=torch.float32)
        return x, y, raw_rel, eval_rel

# ---------- modelos ----------
def make_resnet18(grayscale: bool):
    model = models.resnet18(weights=None)
    if grayscale and model.conv1.in_channels == 3:
        with torch.no_grad():
            w = model.conv1.weight
            new = nn.Conv2d(1, w.shape[0], kernel_size=7, stride=2, padding=3, bias=False)
            new.weight.copy_(w.mean(dim=1, keepdim=True))
        model.conv1 = new
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model

def build_model(arch: str, grayscale: bool):
    arch = arch.lower()
    if arch == "resnet18":
        return make_resnet18(grayscale)
    if arch == "cnn":
        raise NotImplementedError("CNN personalizada ainda não foi implementada.")
    raise ValueError(f"Arquitetura desconhecida: {arch}")

# ---------- métricas ----------
def bin_metrics(y_true, y_prob, thr=0.5):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= thr).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    eps = 1e-9
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / max(1, tp + fp + eps)
    rec = tp / max(1, tp + fn + eps)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return acc, prec, rec, f1, tp, fp, fn, tn

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["resnet18", "cnn"], default="resnet18")
    ap.add_argument("--modality", choices=["raw", "hough", "combined"], required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--grayscale", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--no-save", action="store_true", help="Não salva métricas/predições em disco")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]  # .../satelite
    csv_path = repo_root / "labels" / "test.csv"

    # checkpoints padrão: models/<arch_or_resnet>/<modality>/last.pt
    arch_dir = "resnet" if args.arch == "resnet18" else args.arch
    ckpt_dir = repo_root / "models" / arch_dir / args.modality
    ckpt_path = ckpt_dir / "last.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device} | arch={args.arch} | modality={args.modality} | grayscale={args.grayscale}")
    print(f"[INFO] CSV: {csv_path}")
    print(f"[INFO] Checkpoint: {ckpt_path}")

    if args.arch == "cnn":
        print("CNN personalizada ainda não foi implementada. Use --arch resnet18.")
        return

    ds = CsvDataset(str(csv_path), repo_root=str(repo_root),
                    modality=args.modality, grayscale=args.grayscale)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=torch.cuda.is_available())

    model = build_model(args.arch, grayscale=args.grayscale).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state"] if (isinstance(ckpt, dict) and "model_state" in ckpt) else ckpt
    model.load_state_dict(state)

    model.eval()
    y_true, y_prob, rows = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for xb, yb, raw_rel, eval_rel in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            prob = torch.sigmoid(model(xb).squeeze(1))
            y_true.extend(yb.squeeze(1).cpu().numpy().tolist())
            y_prob.extend(prob.cpu().numpy().tolist())
            # guarda paths e rótulos para reuso posterior
            for rraw, rev, yy, pp in zip(raw_rel, eval_rel, yb.squeeze(1).cpu().tolist(), prob.cpu().tolist()):
                rows.append((rraw, rev, float(yy), float(pp)))

    acc, prec, rec, f1, tp, fp, fn, tn = bin_metrics(y_true, y_prob, thr=args.threshold)
    print("\n==== RESULTADOS (TEST) ====")
    print(f"Acc: {acc*100:.2f}% | Precision: {prec*100:.2f}% | Recall: {rec*100:.2f}% | F1: {f1*100:.2f}%")
    print(f"Confusion Matrix: tp={tp} fp={fp} fn={fn} tn={tn}")
    print(f"Amostras: {len(ds)} | Tempo: {time.time()-t0:.1f}s")

    if not args.no_save:
        # diretórios de saída
        metrics_dir = repo_root / "results" / "metrics" / arch_dir / args.modality
        preds_dir   = repo_root / "results" / "predictions" / arch_dir / args.modality
        metrics_dir.mkdir(parents=True, exist_ok=True)
        preds_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")

        # salva métricas agregadas (json)
        metrics = {
            "arch": args.arch,
            "modality": args.modality,
            "grayscale": bool(args.grayscale),
            "threshold": float(args.threshold),
            "samples": int(len(ds)),
            "acc": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "checkpoint": str(ckpt_path),
            "timestamp": ts
        }
        metrics_path = metrics_dir / f"test_metrics_{ts}.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        # cópia "latest"
        with open(metrics_dir / "test_metrics_latest.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        # salva predições (csv.gz)
        preds_df = pd.DataFrame(rows, columns=["path_raw", "path_eval", "label", "prob"])
        preds_df["pred"] = (preds_df["prob"] >= args.threshold).astype(int)
        preds_path = preds_dir / f"test_preds_{ts}.csv.gz"
        preds_df.to_csv(preds_path, index=False, compression="gzip")
        preds_df.to_csv(preds_dir / "test_preds_latest.csv.gz", index=False, compression="gzip")

        print(f"[OK] Métricas: {metrics_path}")
        print(f"[OK] Predições: {preds_path}")
        print(f"[OK] Também salvos *_latest.* para consumo rápido pelo results.py")

if __name__ == "__main__":
    main()
