#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, time, argparse, math
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import torchvision
from torchvision import models, transforms
from PIL import Image

# ----------------- util -----------------
def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def map_path_for_modality(raw_path: str, modality: str) -> str:
    """
    Espera caminhos relativos tipo: images/patches/raw/<...>.png
    Para hough/combined, troca a pasta e acrescenta sufixo no stem.
    """
    p = raw_path.replace("\\", "/")
    if modality == "raw":
        return p
    if "/patches/raw/" not in p:
        # mantém como está se já não for 'raw'
        return p
    stem, ext = os.path.splitext(p)
    if modality == "hough":
        q = p.replace("/patches/raw/", "/patches/hough/")
        return f"{os.path.splitext(q)[0]}_hough{ext}"
    if modality == "combined":
        q = p.replace("/patches/raw/", "/patches/combined/")
        return f"{os.path.splitext(q)[0]}_combined{ext}"
    return p

# ----------------- dataset -----------------
class PatchesDataset(Dataset):
    def __init__(self, csv_path: Path, repo_root: Path, modality: str, grayscale: bool = True, aug: bool = True):
        df = pd.read_csv(csv_path)
        self.paths = [repo_root / map_path_for_modality(p, modality) for p in df["path"].tolist()]
        self.labels = df["label"].astype(float).values
        self.grayscale = grayscale

        ops = [transforms.Resize((224, 224), antialias=True)]
        if aug:
            ops += [
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
            ]
        if grayscale:
            ops += [transforms.Grayscale(num_output_channels=1)]
        ops += [
            transforms.ToTensor(),
        ]
        if grayscale:
            mean, std = [0.5], [0.25]
        else:
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        ops += [transforms.Normalize(mean=mean, std=std)]
        self.tf = transforms.Compose(ops)

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        with Image.open(img_path).convert("RGB") as im:
            x = self.tf(im)
        y = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return x, y

# ----------------- modelo -----------------
def make_resnet18(pretrained=True, grayscale=True, freeze_backbone=False):
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(weights=None)

    if grayscale and model.conv1.in_channels == 3:
        # adapta conv1 para 1 canal
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

# ----------------- treino/val -----------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tot, corr = 0, 0
    all_loss, count = 0.0, 0
    criterion = nn.BCEWithLogitsLoss()
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        all_loss += loss.item() * xb.size(0)
        count += xb.size(0)
        pred = (logits.sigmoid() >= 0.5).float()
        corr += (pred == yb).sum().item()
        tot += yb.numel()
    return (all_loss / max(1, count)), (corr / max(1, tot))

def train_one_epoch(model, loader, device, optimizer, scaler, log, epoch, total_epochs, log_interval=100):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    t0 = time.time()
    seen, loss_sum = 0, 0.0
    for step, (xb, yb) in enumerate(loader, 1):
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
        else:
            logits = model(xb); loss = criterion(logits, yb)
            loss.backward(); optimizer.step()

        bs = xb.size(0)
        seen += bs
        loss_sum += loss.item() * bs
        if (step % log_interval) == 0 or step == len(loader):
            speed = seen / (time.time() - t0 + 1e-9)
            log(f"[{epoch:02d}/{total_epochs:02d}] step {step:05d}/{len(loader):05d} "
                f"loss={loss_sum/seen:.5f} | {speed:.1f} img/s")
    return loss_sum / max(1, seen)

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["resnet18", "cnn"], default="resnet18")
    ap.add_argument("--modality", choices=["raw", "hough", "combined"], required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--log-interval", type=int, default=100)
    ap.add_argument("--val-every", type=int, default=1)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--freeze-backbone", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--ddp", action="store_true")
    args = ap.parse_args()

    # --------- logging (rank0) ----------
    repo_root = Path(__file__).resolve().parents[1]  # .../satelite
    log_file = repo_root / "code" / "log.txt"
    ensure_dir(log_file)
    def log(msg: str):
        if (not args.ddp) or (dist.is_available() and dist.is_initialized() and dist.get_rank() == 0):
            s = f"{datetime.now():%Y-%m-%d %H:%M:%S} | {msg}"
            print(s, flush=True)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(s + "\n")

    set_seed(42)

    # --------- DDP ----------
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if args.ddp:
        # torchrun exporta LOCAL_RANK/RANK/WORLD_SIZE
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank if use_cuda else 0)
        backend = "nccl" if use_cuda else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    else:
        local_rank, world_size, rank = 0, 1, 0

    # --------- info ----------
    if rank == 0:
        log(f"[INFO] device: {device} | ddp={args.ddp} | world_size={world_size} | rank={rank} | local_rank={local_rank}")
        if use_cuda:
            log(f"[INFO] GPU: {torch.cuda.get_device_name(local_rank)} | CC: {torch.cuda.get_device_capability(local_rank)}")

    # --------- early exit for cnn ----------
    if args.arch == "cnn":
        if rank == 0:
            log("[INFO] CNN personalizada ainda não implementada. Use --arch resnet18.")
        if args.ddp and dist.is_initialized(): dist.destroy_process_group()
        return

    # --------- paths/splits ----------
    labels_dir = repo_root / "labels"
    train_csv = labels_dir / "train.csv"
    val_csv   = labels_dir / "val.csv"
    if rank == 0:
        if not (train_csv.exists() and val_csv.exists()):
            log("[ERRO] Splits não encontrados em labels/train.csv e labels/val.csv")
            return
        log("[INFO] Usando splits existentes em labels/")

    # --------- datasets/loaders ----------
    train_ds = PatchesDataset(train_csv, repo_root, args.modality, grayscale=True, aug=not args.no_aug)
    val_ds   = PatchesDataset(val_csv,   repo_root, args.modality, grayscale=True, aug=False)

    if args.ddp:
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=False)
        val_sampler   = DistributedSampler(val_ds,   shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.workers, pin_memory=use_cuda, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, sampler=val_sampler,
        num_workers=args.workers, pin_memory=use_cuda, drop_last=False
    )

    # --------- model/opt ----------
    model = make_resnet18(pretrained=not args.no_pretrained, grayscale=True, freeze_backbone=args.freeze_backbone)
    model = model.to(device)
    if args.ddp and use_cuda:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    # loss usa pos_weight para balancear classes implicitamente
    # (aqui já incluído dentro do BCE padrão no train loop — simples e robusto)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

    # --------- treino ----------
    for epoch in range(1, args.epochs + 1):
        if args.ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        tr_loss = train_one_epoch(model, train_loader, device, optimizer, scaler, log, epoch, args.epochs, args.log_interval)

        if ((epoch % args.val_every) == 0) or (epoch == args.epochs):
            vl_loss, vl_acc = evaluate(model, val_loader, device)
            if rank == 0:
                log(f"[VAL  {epoch:02d}] loss={vl_loss:.5f} | acc={vl_acc*100:.2f}%")

    if args.ddp and dist.is_initialized():
        dist.destroy_process_group()

    if rank == 0:
        log("[OK] Treinamento finalizado.")

if __name__ == "__main__":
    main()
