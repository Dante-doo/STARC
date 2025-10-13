#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Testa no split de teste e salva:
# - CSV com (raw_path, mapped_path, label, prob, pred)
# - JSON com métricas e confusão
# Usa o checkpoint "best.pt" (ou "last.pt") e, se existir, best_threshold.json.
# Compatível com as modalidades: raw | hough | combined
# e com as subpastas de checkpoint por política de augment (aug-none / aug-rebalance / aug-all).

import os, json, argparse, re
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

# ---------------- path mapping RAW -> target modality ----------------
# Mesma lógica robusta usada no train.py (zero-padding para hough e rot subdir)
_RE_RAW_STD = re.compile(r".*/patches/raw/standart/(full\d+)/(full\d+)_(\d+)_(\d+)_r\.png$")
_RE_RAW_AUG = re.compile(r".*/patches/raw/augment/(full\d+)/(full\d+)_(\d+)_(\d+)_r_(\d+)\.png$")

def _pad2(x: int) -> str:
    return f"{int(x):02d}"

def map_path_for_modality(raw_rel_path: str, modality: str) -> str:
    p = raw_rel_path.replace("\\", "/")
    if modality == "raw":
        return p

    m_std = _RE_RAW_STD.match(p)
    m_aug = _RE_RAW_AUG.match(p)

    if modality == "hough":
        if m_std:
            full_id, _, l, c = m_std.group(1), m_std.group(2), int(m_std.group(3)), int(m_std.group(4))
            head = p.split("/patches/raw/standart/")[0]
            return f"{head}/patches/hough/accumulator/{full_id}/{full_id}_{_pad2(l)}_{_pad2(c)}_h.png"
        if m_aug:
            full_id, _, l, c, ang = m_aug.group(1), m_aug.group(2), int(m_aug.group(3)), int(m_aug.group(4)), int(m_aug.group(5))
            head = p.split("/patches/raw/augment/")[0]
            hid  = f"{full_id}_rot{ang:03d}"
            return f"{head}/patches/hough/accumulator/{hid}/{hid}_{_pad2(l)}_{_pad2(c)}_h.png"
        return p  # fallback

    if modality == "combined":
        if m_std:
            full_id, _, l, c = m_std.group(1), m_std.group(2), int(m_std.group(3)), int(m_std.group(4))
            head = p.split("/patches/raw/standart/")[0]
            return f"{head}/patches/combined/standart/{full_id}/{full_id}_{l}_{c}_cmb.png"
        if m_aug:
            full_id, _, l, c, ang = m_aug.group(1), m_aug.group(2), int(m_aug.group(3)), int(m_aug.group(4)), int(m_aug.group(5))
            head = p.split("/patches/raw/augment/")[0]
            return f"{head}/patches/combined/augment/{full_id}/{full_id}_{l}_{c}_cmb_{ang}.png"
        return p

    return p

# ---------------- model factories ----------------
def make_resnet18(pretrained=True, in_channels=1, freeze_backbone=False):
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(weights=None)

    if in_channels != 3:
        with torch.no_grad():
            w = model.conv1.weight
            new = nn.Conv2d(in_channels, w.shape[0], kernel_size=7, stride=2, padding=3, bias=False)
            if in_channels == 1 and w.shape[1] == 3:
                new.weight.copy_(w.mean(dim=1, keepdim=True))
            else:
                nn.init.kaiming_normal_(new.weight, mode='fan_out', nonlinearity='relu')
        model.conv1 = new

    if freeze_backbone:
        for n, p in model.named_parameters():
            if not n.startswith("fc."):
                p.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, 1)
    return model

def make_deepnet121(pretrained=True, in_channels=1, freeze_backbone=False):
    """
    DeepNet121 (substitua por sua implementação/arquitetura real).
    Aqui usamos DenseNet121 como 'stand-in' se DeepNet não estiver disponível.
    Ajusta a primeira conv para in_channels e a cabeça para 1 logit.
    """
    # Tente importar sua DeepNet se existir
    model = None
    try:
        # from code.deepnet import deepnet121  # exemplo se você tiver um módulo próprio
        # model = deepnet121(pretrained=pretrained)
        pass
    except Exception:
        model = None

    if model is None:
        # fallback robusto: DenseNet121 do torchvision
        try:
            weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.densenet121(weights=weights)
        except Exception:
            model = models.densenet121(weights=None)

        # adaptar primeira camada para in_channels
        first = model.features.conv0
        if first.in_channels != in_channels:
            new0 = nn.Conv2d(in_channels, first.out_channels, kernel_size=first.kernel_size,
                             stride=first.stride, padding=first.padding, bias=False)
            with torch.no_grad():
                if in_channels == 1 and first.in_channels == 3:
                    new0.weight.copy_(first.weight.mean(dim=1, keepdim=True))
                else:
                    nn.init.kaiming_normal_(new0.weight, mode='fan_out', nonlinearity='relu')
            model.features.conv0 = new0

        if freeze_backbone:
            for n, p in model.named_parameters():
                if not n.startswith("classifier."):
                    p.requires_grad = False

        # cabeça -> 1 logit
        in_f = model.classifier.in_features
        model.classifier = nn.Linear(in_f, 1)
        return model

    # Caso você tenha sua DeepNet real:
    # adaptar primeira camada se necessário
    # (exemplo genérico; ajuste conforme sua implementação)
    try:
        first = model.conv1 if hasattr(model, "conv1") else None
        if first is not None and first.in_channels != in_channels:
            new0 = nn.Conv2d(in_channels, first.out_channels, kernel_size=first.kernel_size,
                             stride=first.stride, padding=first.padding, bias=False)
            with torch.no_grad():
                if in_channels == 1 and first.in_channels == 3:
                    new0.weight.copy_(first.weight.mean(dim=1, keepdim=True))
                else:
                    nn.init.kaiming_normal_(new0.weight, mode='fan_out', nonlinearity='relu')
            if hasattr(model, "conv1"):
                model.conv1 = new0
    except Exception:
        pass

    if freeze_backbone:
        for n, p in model.named_parameters():
            if "fc" not in n and "classifier" not in n:
                p.requires_grad = False

    # garantir cabeça -> 1 logit
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        model.classifier = nn.Linear(model.classifier.in_features, 1)
    else:
        # fallback: adicionar uma head genérica
        model.classifier = nn.Linear(getattr(model, "num_features", 1024), 1)

    return model

def build_model(arch: str, in_channels: int, pretrained: bool, freeze_backbone: bool):
    a = arch.lower()
    if a.startswith("resnet18"):
        return make_resnet18(pretrained=pretrained, in_channels=in_channels, freeze_backbone=freeze_backbone)
    if a in ("deepnet121", "deepnet"):
        return make_deepnet121(pretrained=pretrained, in_channels=in_channels, freeze_backbone=freeze_backbone)
    raise ValueError(f"Arquitetura não suportada: {arch}")

def arch_dirname(arch: str) -> str:
    """
    Nome da pasta onde ficam os checkpoints.
    - 'resnet18' -> 'resnet'
    - 'deepnet121' -> 'deepnet121' (com fallback para 'deepnet' na busca)
    """
    a = arch.lower()
    if a.startswith("resnet"):
        return "resnet"
    if a in ("deepnet121", "deepnet"):
        return "deepnet121"
    return a

# ---------------- metrics ----------------
def metrics_from_preds(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> dict:
    y_pred = (y_prob >= thr).astype(np.int32)
    TP = int(np.sum((y_pred == 1) & (y_true == 1)))
    TN = int(np.sum((y_pred == 0) & (y_true == 0)))
    FP = int(np.sum((y_pred == 1) & (y_true == 0)))
    FN = int(np.sum((y_pred == 0) & (y_true == 1)))
    acc  = (TP + TN) / max(1, TP + TN + FP + FN)
    prec = TP / max(1, TP + FP)
    rec  = TP / max(1, TP + FN)
    tnr  = TN / max(1, TN + FP)
    f1   = (2 * prec * rec) / max(1e-9, (prec + rec))
    f2   = (5 * prec * rec) / max(1e-9, (4 * prec + rec))
    balacc = 0.5 * (rec + tnr)
    return {"threshold": float(thr), "acc": acc, "prec1": prec, "recall1": rec,
            "tnr": tnr, "f1": f1, "f2": f2, "balacc": balacc,
            "tp": TP, "fp": FP, "tn": TN, "fn": FN, "n": int(len(y_true))}

# ---------------- dataset ----------------
class TestDataset(torch.utils.data.Dataset):
    """Carrega PNGs conforme modalidade. combined=RGB; raw/hough=grayscale."""
    def __init__(self, csv_path: Path, repo_root: Path, modality: str):
        df = pd.read_csv(csv_path)
        self.raw_paths = df["path"].tolist()
        self.map_paths = [map_path_for_modality(p, modality) for p in self.raw_paths]
        self.paths  = [repo_root / q for q in self.map_paths]
        self.labels = df["label"].astype(np.float32).values
        if modality == "combined":
            self.tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.25,0.25,0.25]),
            ])
            self.mode_rgb = True
        else:
            self.tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.25]),
            ])
            self.mode_rgb = False

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB" if self.mode_rgb else "L")
        x   = self.tf(img)
        y   = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return x, y, self.raw_paths[idx], self.map_paths[idx]

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate on test split and persist predictions/metrics")
    ap.add_argument("--arch", choices=["resnet18", "deepnet121", "deepnet"], default=None, help="backbone")
    ap.add_argument("--modality", choices=["raw", "hough", "combined"], default=None, help="input modality")
    ap.add_argument("--aug", choices=["none","rebalance","all"], default="rebalance", help="subdir do checkpoint (igual ao treino)")
    ap.add_argument("--ckpt", choices=["best", "last"], default="best", help="qual checkpoint carregar")
    args = ap.parse_args()

    arch     = (args.arch or TRAIN_ARCH)
    modality = (args.modality or TRAIN_MODALITY)
    aug_pol  = (args.aug or "rebalance").lower()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_csv = LABELS_DIR / "test.csv"
    if not test_csv.exists():
        raise FileNotFoundError("labels/test.csv não encontrado. Rode split primeiro.")

    ds = TestDataset(test_csv, ROOT, modality)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=EVAL_BATCH_SIZE, shuffle=False,
        num_workers=EVAL_WORKERS, pin_memory=torch.cuda.is_available(),
        prefetch_factor=EVAL_PREFETCH if EVAL_WORKERS > 0 else None,
        persistent_workers=(EVAL_WORKERS > 0),
    )

    # modelo
    in_ch = 3 if modality == "combined" else 1
    model = build_model(arch, in_channels=in_ch, pretrained=TRAIN_PRETRAINED, freeze_backbone=TRAIN_FREEZE_BACKBONE)
    if torch.cuda.is_available() and TRAIN_CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last)
    model = model.to(device)
    model.eval()

    # localizar checkpoint: tenta pasta "deepnet121" (ou "resnet") e cai para "deepnet" se necessário
    arch_dir = arch_dirname(arch)
    ckpt_dir_primary = MODELS_DIR / arch_dir / modality
    if aug_pol != "rebalance":
        ckpt_dir_primary = ckpt_dir_primary / f"aug-{aug_pol}"
    # fallback p/ compat: se arch_dir == deepnet121 mas existir "deepnet", use-o
    ckpt_dir_fallback = None
    if arch_dir == "deepnet121":
        ckpt_dir_fallback = MODELS_DIR / "deepnet" / modality
        if aug_pol != "rebalance":
            ckpt_dir_fallback = ckpt_dir_fallback / f"aug-{aug_pol}"

    def find_ckpt(ckpt_name: str) -> Path:
        p1 = ckpt_dir_primary / f"{ckpt_name}.pt"
        if p1.exists():
            return p1
        if ckpt_dir_fallback is not None:
            p2 = ckpt_dir_fallback / f"{ckpt_name}.pt"
            if p2.exists():
                return p2
        return p1  # devolve primário (existente ou não) para mensagem clara

    ckpt_path = find_ckpt(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"], strict=True)

    # threshold: preferir best_threshold.json (no mesmo dir do ckpt encontrado)
    thr_dir = ckpt_path.parent
    best_thr_file = thr_dir / "best_threshold.json"
    if best_thr_file.exists():
        best_thr = float(json.loads(best_thr_file.read_text())["threshold"])
    else:
        best_thr = float(TRAIN_THRESHOLD)

    # inferência
    probs, labels, raw_rel, mapped_rel = [], [], [], []
    with torch.no_grad():
        for xb, yb, rp, mp in loader:
            xb = xb.to(device, non_blocking=True)
            if TRAIN_CHANNELS_LAST and xb.is_cuda:
                xb = xb.to(memory_format=torch.channels_last)
            logits = model(xb)
            pr = torch.sigmoid(logits).view(-1).detach().cpu().numpy()
            lb = yb.view(-1).detach().cpu().numpy()
            probs.append(pr); labels.append(lb)
            raw_rel += list(rp); mapped_rel += list(mp)

    y_prob = np.concatenate(probs, axis=0)
    y_true = np.concatenate(labels, axis=0).astype(np.int32)
    mets   = metrics_from_preds(y_true, y_prob, best_thr)

    # salvar CSV de predições
    out_pred_dir = RESULTS_PREDS_DIR / arch_dir / modality
    if aug_pol != "rebalance":
        out_pred_dir = out_pred_dir / f"aug-{aug_pol}"
    # Se usamos fallback deepnet/, espelhar na saída pelo mesmo dir do ckpt real
    if not (ckpt_path.parent / "dummy").exists() and "deepnet" in str(ckpt_path.parent):
        out_pred_dir = RESULTS_PREDS_DIR / "deepnet" / modality / (f"aug-{aug_pol}" if aug_pol != "rebalance" else "")
    out_pred_dir = Path(str(out_pred_dir)).resolve()
    out_pred_dir.mkdir(parents=True, exist_ok=True)
    pred_csv = out_pred_dir / "test_preds.csv"
    pd.DataFrame({
        "raw_path": raw_rel,
        "mapped_path": mapped_rel,
        "label": y_true,
        "prob": y_prob,
        "pred": (y_prob >= best_thr).astype(int)
    }).to_csv(pred_csv, index=False)

    # salvar JSON de métricas
    out_met_dir = RESULTS_METRICS_DIR / arch_dir / modality
    if aug_pol != "rebalance":
        out_met_dir = out_met_dir / f"aug-{aug_pol}"
    if not (ckpt_path.parent / "dummy").exists() and "deepnet" in str(ckpt_path.parent):
        out_met_dir = RESULTS_METRICS_DIR / "deepnet" / modality / (f"aug-{aug_pol}" if aug_pol != "rebalance" else "")
    out_met_dir = Path(str(out_met_dir)).resolve()
    out_met_dir.mkdir(parents=True, exist_ok=True)
    met_json = out_met_dir / "test_metrics.json"
    with open(met_json, "w", encoding="utf-8") as f:
        json.dump({
            "arch": arch,
            "modality": modality,
            "aug_policy": aug_pol,
            "ckpt": args.ckpt,
            "threshold": best_thr,
            "metrics": mets
        }, f, indent=2)

    # resumo
    print(f"[OK] Test finalizado | arch={arch} modality={modality} aug={aug_pol} thr={best_thr:.3f}")
    print(f"acc={mets['acc']*100:.2f}%  prec1={mets['prec1']:.4f}  recall1={mets['recall1']:.4f}  "
          f"tnr={mets['tnr']:.4f}  balacc={mets['balacc']:.4f}")
    print(f"TP={mets['tp']} FP={mets['fp']} TN={mets['tn']} FN={mets['fn']}")
    print(f"preds:   {pred_csv}")
    print(f"metrics: {met_json}")

if __name__ == "__main__":
    main()
