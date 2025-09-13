#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Train a binary classifier (ResNet18 -> 1 logit) on RAW/Hough/Combined.
# Quality-first:
# - Rebalance train by limiting augments so train pos_rate ~= base pos_rate
# - No pos_weight by default (can be enabled)
# - Best checkpoint by Balanced Accuracy (BA)
# - Threshold sweep on val to store the best decision threshold (by BA)

import os, time, math, json, argparse, warnings, re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms

from code.config import (
    ROOT, LABELS_DIR, MODELS_DIR,
    TRAIN_ARCH, TRAIN_MODALITY,
    TRAIN_EPOCHS, TRAIN_BATCH_SIZE, TRAIN_WORKERS, TRAIN_PREFETCH,
    TRAIN_LR, TRAIN_VAL_EVERY, TRAIN_LOG_INTERVAL,
    TRAIN_USE_AUG, TRAIN_PRETRAINED, TRAIN_FREEZE_BACKBONE,
    TRAIN_USE_SAMPLER, TRAIN_USE_POS_WEIGHT, TRAIN_THRESHOLD,
    TRAIN_DDP, TRAIN_CHANNELS_LAST, TRAIN_MIXED_PRECISION, TRAIN_TF32, TRAIN_COMPILE,
    TRAIN_MATCH_BASE_RATE, TRAIN_TARGET_POS_RATE, TRAIN_AUG_MAX_PER_BASE,
    TRAIN_SELECT_METRIC, TRAIN_SWEEP_MIN_T, TRAIN_SWEEP_MAX_T, TRAIN_SWEEP_STEPS,
)

# ----------------- utils -----------------
def set_seed(seed=42):
    """Determinism off for speed; seeded ops for baseline reproducibility."""
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = bool(TRAIN_TF32)
    torch.backends.cudnn.allow_tf32 = bool(TRAIN_TF32)
    try:
        torch.set_float32_matmul_precision("highest" if not TRAIN_TF32 else "high")
    except Exception:
        pass

def now() -> str:
    return f"{datetime.now():%Y-%m-%d %H:%M:%S}"

# Map RAW → other modalities by our naming/layout
def map_path_for_modality(raw_rel_path: str, modality: str) -> str:
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
    """ResNet18 with conv1 adapted to 1 channel and final FC -> 1 logit."""
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

# ----------------- dataset -----------------
class PatchesDataset(Dataset):
    """Load grayscale PNGs, optional flips, normalize to mean=0.5,std=0.25."""
    def __init__(self, data, repo_root: Path, modality: str, use_aug: bool):
        # 'data' can be a Path to CSV or a DataFrame filtered upstream
        if isinstance(data, (str, Path)):
            df = pd.read_csv(data)
        else:
            df = data.copy()
        paths = df["path"].tolist()
        self.labels = df["label"].astype(np.float32).values
        self.paths = [repo_root / map_path_for_modality(p, modality) for p in paths]

        ops = []
        if use_aug:
            ops += [transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5)]
        ops += [transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.25])]
        self.tf = transforms.Compose(ops)

    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("L")
        x = self.tf(img)
        y = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return x, y

# ----------------- train rebalancing (limit augments) -----------------
RE_AUG = re.compile(r"/augment/")
RE_KEY = re.compile(r"(full\d+_\d+_\d+)")

def is_aug(path: str) -> bool:
    return bool(RE_AUG.search(path.replace("\\", "/")))

def base_key(path: str) -> str:
    m = RE_KEY.search(path)
    if not m:
        raise ValueError(f"Cannot extract base key from: {path}")
    return m.group(1)

def rebalance_train_df(train_csv: Path, match_base_rate=True, target_pos_rate=None, max_aug_per_base=1, seed=42) -> pd.DataFrame:
    """
    Keep all base rows. Keep only a subset of augments so that
    final train pos_rate ~= base pos_rate (or a given target).
    Also cap to at most 'max_aug_per_base' per base positive.
    """
    rng = np.random.default_rng(seed)
    df = pd.read_csv(train_csv)
    df["is_aug"] = df["path"].apply(is_aug)

    base = df[~df["is_aug"]].copy()
    aug  = df[df["is_aug"]].copy()

    # base stats
    n_pos_base = int(base["label"].sum())
    n_neg_base = int((1 - base["label"]).sum())
    r_base = n_pos_base / max(1, (n_pos_base + n_neg_base))

    # decide target rate
    r_target = r_base if match_base_rate else (float(target_pos_rate) if target_pos_rate is not None else r_base)

    # If no aug or target already met, return base as-is (no aug needed)
    if aug.empty:
        return base[["path", "label"]]

    # cap aug per base
    aug["key"] = aug["path"].apply(base_key)
    aug = aug.groupby("key", group_keys=False).apply(lambda g: g.sample(n=min(len(g), max_aug_per_base), random_state=seed))

    # compute how many aug to keep to hit r_target approximately
    # Solve for K: (n_pos_base + K) / (n_pos_base + n_neg_base + K) = r_target
    K = int(round((r_target * (n_pos_base + n_neg_base) - n_pos_base) / max(1e-9, (1 - r_target))))
    K = max(0, min(K, len(aug)))  # clamp to available aug count

    # sample K aug uniformly
    if K > 0:
        aug_keep = aug.sample(n=K, random_state=seed)
        out = pd.concat([base[["path","label"]], aug_keep[["path","label"]]], ignore_index=True)
    else:
        out = base[["path","label"]]

    return out

# ----------------- metrics -----------------
def confusion_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int,int,int,int]:
    """Return TP, FP, TN, FN for binary arrays."""
    TP = int(np.sum((y_pred == 1) & (y_true == 1)))
    TN = int(np.sum((y_pred == 0) & (y_true == 0)))
    FP = int(np.sum((y_pred == 1) & (y_true == 0)))
    FN = int(np.sum((y_pred == 0) & (y_true == 1)))
    return TP, FP, TN, FN

def metrics_from_probs(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> dict:
    """Compute metrics at a given threshold."""
    y_pred = (y_prob >= thr).astype(np.int32)
    TP, FP, TN, FN = confusion_from_preds(y_true, y_pred)
    acc  = (TP + TN) / max(1, TP + TN + FP + FN)
    prec = TP / max(1, TP + FP)
    rec  = TP / max(1, TP + FN)
    tnr  = TN / max(1, TN + FP)  # specificity
    f1   = (2 * prec * rec) / max(1e-9, prec + rec)
    f2   = (5 * prec * rec) / max(1e-9, (4 * prec + rec))  # beta=2
    balacc = 0.5 * (rec + tnr)
    return {"acc": acc, "prec1": prec, "recall1": rec, "tnr": tnr, "f1": f1, "f2": f2, "balacc": balacc,
            "tp": TP, "fp": FP, "tn": TN, "fn": FN}

@torch.no_grad()
def infer_probs_and_labels(model, loader, device, ch_last=False) -> tuple[np.ndarray, np.ndarray]:
    """Run model on loader and return (y_true, y_prob) arrays."""
    model.eval()
    probs, labels = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        if ch_last and xb.is_cuda:
            xb = xb.to(memory_format=torch.channels_last)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        p = torch.sigmoid(logits).float().view(-1).detach().cpu().numpy()
        y = yb.view(-1).detach().cpu().numpy()
        probs.append(p); labels.append(y)
    y_prob = np.concatenate(probs, axis=0)
    y_true = np.concatenate(labels, axis=0).astype(np.int32)
    return y_true, y_prob

def sweep_thresholds(y_true: np.ndarray, y_prob: np.ndarray, t_min: float, t_max: float, steps: int, metric: str) -> tuple[float, dict]:
    """Grid-search for best threshold by metric in {'balacc','f1','f2'}."""
    ts = np.linspace(t_min, t_max, num=steps, endpoint=True)
    best_t, best_m = ts[0], None
    best_score = -1.0
    for t in ts:
        m = metrics_from_probs(y_true, y_prob, float(t))
        score = float(m[metric])
        if score > best_score:
            best_score, best_t, best_m = score, float(t), m
    return best_t, best_m

# ----------------- training -----------------
def train_one_epoch(model, loader, device, optimizer, scaler, pos_weight, epoch, total_epochs, log, log_interval=100, ch_last=False, mp=False):
    """Standard train loop; AMP off by default (mp=False)."""
    model.train()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    t0 = time.time()
    seen, loss_sum = 0, 0.0
    for step, (xb, yb) in enumerate(loader, 1):
        xb = xb.to(device, non_blocking=True)
        if ch_last and xb.is_cuda:
            xb = xb.to(memory_format=torch.channels_last)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if mp and scaler is not None and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(xb); loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
        else:
            logits = model(xb); loss = criterion(logits, yb)
            loss.backward(); optimizer.step()

        bs = xb.size(0)
        seen += bs; loss_sum += loss.item() * bs
        if (step % log_interval) == 0 or step == len(loader):
            speed = seen / (time.time() - t0 + 1e-9)
            log(f"[{epoch:02d}/{total_epochs:02d}] step {step:05d}/{len(loader):05d} "
                f"loss={loss_sum/seen:.5f} | {speed:.1f} img/s")
    return loss_sum / max(1, seen)

# ----------------- main -----------------
def main():
    # CLI overrides for arch/modality only
    ap = argparse.ArgumentParser(description="Satellite trail classifier training")
    ap.add_argument("--arch", choices=["resnet18", "cnn"], default=None)
    ap.add_argument("--modality", choices=["raw", "hough", "combined"], default=None)
    args = ap.parse_args()

    arch     = args.arch or TRAIN_ARCH
    modality = args.modality or TRAIN_MODALITY

    set_seed(42)

    # DDP init
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    ddp = bool(TRAIN_DDP and int(os.environ.get("WORLD_SIZE", "1")) > 1)
    if ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank if use_cuda else 0)
        backend = "nccl" if use_cuda else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    else:
        local_rank = 0; world_size = 1; rank = 0

    is_master = (rank == 0)
    if not is_master:
        warnings.filterwarnings("ignore")

    def log(msg: str):
        if is_master:
            s = f"{now()} | {msg}"
            print(s, flush=True)
            f = ROOT / "code" / "train.log"
            f.parent.mkdir(parents=True, exist_ok=True)
            with open(f, "a", encoding="utf-8") as fh: fh.write(s + "\n")

    if is_master:
        log(f"[INFO] device={device} ddp={ddp} world_size={world_size} arch={arch} modality={modality}")
        if use_cuda:
            log(f"[INFO] GPU={torch.cuda.get_device_name(local_rank)} CC={torch.cuda.get_device_capability(local_rank)}")
        log("[INFO] Accuracy-first: no AMP/TF32 by default.")

    # Early exit for CNN
    if arch == "cnn":
        log("[INFO] 'cnn' architecture not implemented yet. Use --arch resnet18.")
        if TRAIN_DDP and dist.is_initialized(): dist.destroy_process_group()
        return

    # Splits
    train_csv = LABELS_DIR / "train.csv"
    val_csv   = LABELS_DIR / "val.csv"
    if is_master and (not train_csv.exists() or not val_csv.exists()):
        raise FileNotFoundError("Missing labels/train.csv or labels/val.csv. Run split first.")

    # ---- Rebalance train by limiting augments ----
    tr_df = rebalance_train_df(
        train_csv,
        match_base_rate=TRAIN_MATCH_BASE_RATE,
        target_pos_rate=TRAIN_TARGET_POS_RATE,
        max_aug_per_base=TRAIN_AUG_MAX_PER_BASE,
        seed=42
    )
    if is_master:
        n = len(tr_df); p = int(tr_df["label"].sum()); pr = p / max(1, n)
        log(f"[INFO] train after rebalance: n={n} pos={p} pos_rate={pr:.6f}")

    # Datasets
    train_ds = PatchesDataset(tr_df, ROOT, modality, use_aug=TRAIN_USE_AUG)
    val_ds   = PatchesDataset(val_csv, ROOT, modality, use_aug=False)

    # Class stats for (optional) pos_weight
    train_labels_np = np.array([lbl for lbl in train_ds.labels], dtype=np.float32)
    n_pos = float(train_labels_np.sum()); n_neg = float(len(train_labels_np) - n_pos)
    pos_weight_val = torch.tensor([n_neg / max(1.0, n_pos)], dtype=torch.float32, device=device)

    # Sampler (avoid under DDP)
    if TRAIN_USE_SAMPLER and not ddp:
        w_pos = 1.0 / max(1.0, n_pos); w_neg = 1.0 / max(1.0, n_neg)
        sample_weights = torch.tensor([w_pos if y > 0.5 else w_neg for y in train_labels_np], dtype=torch.double)
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    else:
        sampler = None

    # Dataloaders
    dl_kwargs = dict(
        num_workers=TRAIN_WORKERS,
        pin_memory=use_cuda,
        drop_last=False,
        prefetch_factor=TRAIN_PREFETCH if TRAIN_WORKERS > 0 else None,
        persistent_workers=(TRAIN_WORKERS > 0),
    )
    train_loader = DataLoader(
        train_ds, batch_size=TRAIN_BATCH_SIZE,
        shuffle=(sampler is None) and (not ddp),
        sampler=sampler, **dl_kwargs
    )
    val_loader = DataLoader(
        val_ds, batch_size=TRAIN_BATCH_SIZE,
        shuffle=False, **dl_kwargs
    )

    # Model/opt
    model = make_resnet18(pretrained=TRAIN_PRETRAINED, grayscale=True, freeze_backbone=TRAIN_FREEZE_BACKBONE)
    if use_cuda and TRAIN_CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last)
    model = model.to(device)

    if TRAIN_COMPILE:
        try:
            model = torch.compile(model, mode="max-autotune")
            log("[INFO] torch.compile enabled.")
        except Exception as e:
            log(f"[WARN] torch.compile disabled: {e}")

    if ddp and use_cuda:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=TRAIN_LR)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(use_cuda and TRAIN_MIXED_PRECISION))

    pos_w = (pos_weight_val if TRAIN_USE_POS_WEIGHT else None)
    ch_last = bool(use_cuda and TRAIN_CHANNELS_LAST)
    mp = bool(use_cuda and TRAIN_MIXED_PRECISION)

    # Checkpoints
    arch_dir = "resnet" if arch.startswith("resnet") else arch
    ckpt_dir = MODELS_DIR / arch_dir / modality
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt = ckpt_dir / "last.pt"
    best_ckpt = ckpt_dir / "best.pt"
    best_thr_file = ckpt_dir / "best_threshold.json"

    # Track best by selected metric (balanced accuracy default)
    best_metric_score = -1.0
    best_metric_name = TRAIN_SELECT_METRIC

    # Train loop
    for epoch in range(1, TRAIN_EPOCHS + 1):
        tr_loss = train_one_epoch(
            model, train_loader, device, optimizer, scaler, pos_w,
            epoch, TRAIN_EPOCHS, log, TRAIN_LOG_INTERVAL, ch_last=ch_last, mp=mp
        )

        do_val = ((epoch % TRAIN_VAL_EVERY) == 0) or (epoch == TRAIN_EPOCHS)
        if do_val:
            # 1) Default threshold (for trace)
            y_true, y_prob = infer_probs_and_labels(model, val_loader, device, ch_last=ch_last)
            m_def = metrics_from_probs(y_true, y_prob, TRAIN_THRESHOLD)
            log(f"[VAL {epoch:02d} thr={TRAIN_THRESHOLD:.2f}] "
                f"loss=NA acc={m_def['acc']*100:.2f}% prec1={m_def['prec1']:.4f} "
                f"recall1={m_def['recall1']:.4f} tnr={m_def['tnr']:.4f} "
                f"f1={m_def['f1']:.4f} f2={m_def['f2']:.4f} balacc={m_def['balacc']:.4f} "
                f"TP={m_def['tp']} FP={m_def['fp']} TN={m_def['tn']} FN={m_def['fn']}")

            # 2) Threshold sweep to find best by selected metric
            best_t, m_best = sweep_thresholds(
                y_true, y_prob, TRAIN_SWEEP_MIN_T, TRAIN_SWEEP_MAX_T, TRAIN_SWEEP_STEPS, TRAIN_SELECT_METRIC
            )
            log(f"[VAL {epoch:02d} best by {TRAIN_SELECT_METRIC}] thr={best_t:.3f} "
                f"acc={m_best['acc']*100:.2f}% prec1={m_best['prec1']:.4f} "
                f"recall1={m_best['recall1']:.4f} tnr={m_best['tnr']:.4f} "
                f"f1={m_best['f1']:.4f} f2={m_best['f2']:.4f} balacc={m_best['balacc']:.4f} "
                f"TP={m_best['tp']} FP={m_best['fp']} TN={m_best['tn']} FN={m_best['fn']}")

            # Save last
            state_dict = model.module.state_dict() if isinstance(model, nn.parallel.DistributedDataParallel) else model.state_dict()
            torch.save({"epoch": epoch, "model": state_dict}, last_ckpt)

            # Save best (by selected metric)
            score = float(m_best[TRAIN_SELECT_METRIC])
            if score > best_metric_score:
                best_metric_score = score
                torch.save({"epoch": epoch, "model": state_dict}, best_ckpt)
                with open(best_thr_file, "w", encoding="utf-8") as f:
                    json.dump({"threshold": best_t, "metric": TRAIN_SELECT_METRIC, "score": score, "metrics": m_best}, f, indent=2)
                log(f"[CKPT] Saved BEST → {best_ckpt} | {TRAIN_SELECT_METRIC}={score:.4f} | thr={best_t:.3f}")

    if TRAIN_DDP and dist.is_initialized():
        dist.destroy_process_group()

    # Final info
    info = {
        "arch": arch,
        "modality": modality,
        "epochs": TRAIN_EPOCHS,
        "batch_size_per_gpu": TRAIN_BATCH_SIZE,
        "lr": TRAIN_LR,
        "use_aug": TRAIN_USE_AUG,
        "use_pos_weight": TRAIN_USE_POS_WEIGHT,
        "rebalance": {"match_base_rate": TRAIN_MATCH_BASE_RATE, "target_pos_rate": TRAIN_TARGET_POS_RATE,
                      "max_aug_per_base": TRAIN_AUG_MAX_PER_BASE},
        "selection_metric": TRAIN_SELECT_METRIC,
        "thr_sweep": {"min": TRAIN_SWEEP_MIN_T, "max": TRAIN_SWEEP_MAX_T, "steps": TRAIN_SWEEP_STEPS},
    }
    with (ckpt_dir / "train_meta.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    log(f"[OK] Training finished. Checkpoints at: {ckpt_dir}")

if __name__ == "__main__":
    main()
