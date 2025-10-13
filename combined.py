#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Gera patches "combined" 3-canais (RAW, ACC-resized, INV).
# Para AUG:
#   1) tenta achar ACC/INV do Hough rotacionados (fullX_rot090 / fullX_90)
#   2) se não existir, usa ACC/INV **base** e ROTACIONA no ato para o ângulo do RAW aug

import re
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np

from code.config import (
    RAW_STD_DIR, RAW_AUG_DIR,
    PATCH_HOUGH_ACC_DIR, PATCH_HOUGH_INV_DIR,
    COMBINED_STD_DIR, COMBINED_AUG_DIR,
    SFX_R, SFX_H, SFX_INV, SFX_C,
    PATCH_SIZE,
    COMBINED_JOBS,
)

RE_STD = re.compile(rf"^(full\d+)_([0-9]+)_([0-9]+)_{re.escape(SFX_R)}\.png$")
RE_AUG = re.compile(rf"^(full\d+)_([0-9]+)_([0-9]+)_{re.escape(SFX_R)}_(\d+)\.png$")

def rotate_k(img: np.ndarray, ang: int) -> np.ndarray:
    k = (ang // 90) % 4
    if k == 0: return img
    return np.ascontiguousarray(np.rot90(img, k=k))

def iter_raw_patches() -> list[tuple[Path, bool]]:
    out = []
    if RAW_STD_DIR.exists():
        for d in sorted(RAW_STD_DIR.glob("full*")):
            if d.is_dir():
                out += [(p, False) for p in sorted(d.glob(f"*_{SFX_R}.png"))]
    if RAW_AUG_DIR.exists():
        for d in sorted(RAW_AUG_DIR.glob("full*")):
            if d.is_dir():
                out += [(p, True) for p in sorted(d.glob(f"*_{SFX_R}_*.png"))]
    return out

def parse_std_name(name: str):
    m = RE_STD.match(name)
    if not m: return None
    return m.group(1), int(m.group(2)), int(m.group(3))

def parse_aug_name(name: str):
    m = RE_AUG.match(name)
    if not m: return None
    return m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))

def load_gray(path: Path, fallback_shape: tuple[int,int] | None = None) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None and fallback_shape is not None:
        return np.zeros(fallback_shape, np.uint8)
    return img

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def first_existing(cands: list[Path]) -> Path | None:
    for p in cands:
        if p.exists(): return p
    return None

def build_paths(raw_path: Path, is_aug: bool):
    n = raw_path.name
    if is_aug:
        parsed = parse_aug_name(n)
        if not parsed: return None
        full_id, l, c, ang = parsed

        hid_rot   = f"{full_id}_rot{int(ang):03d}"   # prioridade
        hid_plain = f"{full_id}_{int(ang)}"

        # caminhos Hough AUG (se existirem)
        acc_rot = [
            PATCH_HOUGH_ACC_DIR / hid_rot   / f"{hid_rot}_{l:02d}_{c:02d}_{SFX_H}.png",
            PATCH_HOUGH_ACC_DIR / hid_plain / f"{hid_plain}_{l:02d}_{c:02d}_{SFX_H}.png",
        ]
        inv_rot = [
            PATCH_HOUGH_INV_DIR / hid_rot   / f"{hid_rot}_{l:02d}_{c:02d}_{SFX_INV}.png",
            PATCH_HOUGH_INV_DIR / hid_plain / f"{hid_plain}_{l:02d}_{c:02d}_{SFX_INV}.png",
        ]

        # caminhos Hough BASE (fallback para rotacionar)
        acc_base = PATCH_HOUGH_ACC_DIR / full_id / f"{full_id}_{l:02d}_{c:02d}_{SFX_H}.png"
        inv_base = PATCH_HOUGH_INV_DIR / full_id / f"{full_id}_{l:02d}_{c:02d}_{SFX_INV}.png"

        out_dir = COMBINED_AUG_DIR / full_id
        out = out_dir / f"{full_id}_{l}_{c}_{SFX_C}_{ang}.png"

        return dict(full_id=full_id, l=l, c=c, ang=ang,
                    acc_rot_opts=acc_rot, inv_rot_opts=inv_rot,
                    acc_base=acc_base, inv_base=inv_base,
                    out_dir=out_dir, out=out, is_aug=True)
    else:
        parsed = parse_std_name(n)
        if not parsed: return None
        full_id, l, c = parsed
        acc = PATCH_HOUGH_ACC_DIR / full_id / f"{full_id}_{l:02d}_{c:02d}_{SFX_H}.png"
        inv = PATCH_HOUGH_INV_DIR / full_id / f"{full_id}_{l:02d}_{c:02d}_{SFX_INV}.png"
        out_dir = COMBINED_STD_DIR / full_id
        out = out_dir / f"{full_id}_{l}_{c}_{SFX_C}.png"
        return dict(full_id=full_id, l=l, c=c, ang=None,
                    acc=acc, inv=inv, out_dir=out_dir, out=out, is_aug=False)

def make_one(raw_path: Path, is_aug: bool, overwrite: bool=False) -> tuple[Path, bool]:
    meta = build_paths(raw_path, is_aug)
    if meta is None:
        return raw_path, False

    out_path: Path = meta["out"]
    if out_path.exists() and not overwrite:
        return out_path, True

    ensure_dir(meta["out_dir"])

    raw = load_gray(raw_path)
    if raw is None:
        return out_path, False

    if not is_aug:
        inv = load_gray(meta["inv"], fallback_shape=(PATCH_SIZE, PATCH_SIZE))
        acc = load_gray(meta["acc"])
        acc_res = np.zeros((PATCH_SIZE, PATCH_SIZE), np.uint8) if acc is None else cv2.resize(acc, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_LINEAR)
    else:
        ang = int(meta["ang"])
        # tenta Hough AUG
        acc_r = first_existing(meta["acc_rot_opts"])
        inv_r = first_existing(meta["inv_rot_opts"])
        if acc_r is not None and inv_r is not None:
            inv = load_gray(inv_r, fallback_shape=(PATCH_SIZE, PATCH_SIZE))
            acc = load_gray(acc_r)
            acc_res = np.zeros((PATCH_SIZE, PATCH_SIZE), np.uint8) if acc is None else cv2.resize(acc, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_LINEAR)
        else:
            # fallback: usa base e rotaciona
            inv_b = load_gray(meta["inv_base"], fallback_shape=(PATCH_SIZE, PATCH_SIZE))
            acc_b = load_gray(meta["acc_base"])
            inv = rotate_k(inv_b, ang)
            if acc_b is None:
                acc_res = np.zeros((PATCH_SIZE, PATCH_SIZE), np.uint8)
            else:
                acc_b = cv2.resize(acc_b, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_LINEAR)
                acc_res = rotate_k(acc_b, ang)

    # normaliza tamanhos
    if raw.shape != (PATCH_SIZE, PATCH_SIZE):
        raw = cv2.resize(raw, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)
    if inv.shape != (PATCH_SIZE, PATCH_SIZE):
        inv = cv2.resize(inv, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_NEAREST)

    combo = np.dstack([raw, acc_res, inv]).astype(np.uint8)
    ok = cv2.imwrite(str(out_path), combo)
    return out_path, ok

def main():
    ap = argparse.ArgumentParser(description="Gerar patches combined (RAW, ACC224, INV) com fallback de rotação do ACC/INV base.")
    ap.add_argument("--jobs", type=int, default=COMBINED_JOBS, help="threads para I/O (default: config.COMBINED_JOBS)")
    ap.add_argument("--overwrite", action="store_true", help="regravar arquivos já existentes")
    args = ap.parse_args()

    raws = iter_raw_patches()
    if not raws:
        raise FileNotFoundError(f"Nenhum RAW patch encontrado em {RAW_STD_DIR} ou {RAW_AUG_DIR}")

    print(f"[combined] total RAW patches: {len(raws)}  | jobs={args.jobs}  | overwrite={args.overwrite}")

    ok = skip = fail = 0
    with ThreadPoolExecutor(max_workers=max(1, int(args.jobs))) as ex:
        futs = [ex.submit(make_one, p, is_aug, args.overwrite) for (p, is_aug) in raws]
        for fu in as_completed(futs):
            out, res = fu.result()
            if res is True: ok += 1
            elif res is False: fail += 1
            else: skip += 1

    print(f"[combined] ok={ok}  skip={skip}  fail={fail}")
    print(f"[combined] out: {COMBINED_STD_DIR}  |  {COMBINED_AUG_DIR}")

if __name__ == "__main__":
    main()
