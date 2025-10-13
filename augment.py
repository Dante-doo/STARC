#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Gera RAW augmentados (apenas para patches com trilha) a partir do que o Hough já produziu.
# Verifica as pastas base (fullX) e rotacionadas (fullX_rot090 / fullX_90) em hough/inverted/.
# Para cada INV com pixels > 0, grava o RAW rotacionado em images/patches/raw/augment/<fullX>/fullX_l_c_r_ang.png
# Agora compatível com RAW base sem zero-padding (fullX_l_c_r.png) e também com zero-padding.

import re
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np

from code.config import (
    RAW_STD_DIR, RAW_AUG_DIR,
    PATCH_HOUGH_INV_DIR,
    SFX_R, SFX_INV,
    PATCH_SIZE,
    AUGMENT_ANGLES,
)

# Pastas possíveis em hough/inverted/
RE_DIR_BASE   = re.compile(r"^(full\d+)$")                 # full12
RE_DIR_ROT    = re.compile(r"^(full\d+)_rot(\d{3})$")      # full12_rot090
RE_DIR_PLAIN  = re.compile(r"^(full\d+)_([0-9]{2,3})$")    # full12_90

# INV filename (aceita base e rot)
# base: full12_03_07_inv.png
# rot : full12_rot090_03_07_inv.png  ou  full12_90_03_07_inv.png
RE_INV_FILE_ANY = re.compile(
    r"^(full\d+(?:_rot\d{3}|_[0-9]{2,3})?)_([0-9]{2})_([0-9]{2})_" + re.escape(SFX_INV) + r"\.png$"
)

def count_nonzero(p: Path) -> int:
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0
    return int(cv2.countNonZero(img))

def parse_dir(d: Path):
    m = RE_DIR_BASE.match(d.name)
    if m:  return "base", m.group(1), None
    m = RE_DIR_ROT.match(d.name)
    if m:  return "rot", m.group(1), int(m.group(2))
    m = RE_DIR_PLAIN.match(d.name)
    if m:  return "plain", m.group(1), int(m.group(2))
    return None

def iter_inv_dirs():
    if not PATCH_HOUGH_INV_DIR.exists():
        return
    for d in sorted(PATCH_HOUGH_INV_DIR.iterdir()):
        if d.is_dir():
            parsed = parse_dir(d)
            if parsed:
                yield d, parsed  # (Path, (kind, full_id, ang_or_none))

def tasks_from_inv():
    tasks = []
    for d, (kind, full_id, ang_dir) in iter_inv_dirs():
        for f in sorted(d.glob(f"*_{SFX_INV}.png")):
            m = RE_INV_FILE_ANY.match(f.name)
            if not m:
                continue
            l = int(m.group(2))
            c = int(m.group(3))
            if count_nonzero(f) <= 0:
                continue  # sem trilha
            if kind == "base":
                # gera todas as rotações configuradas
                for ang in AUGMENT_ANGLES:
                    tasks.append((full_id, l, c, int(ang)))
            else:
                # já é uma pasta rotacionada; usa o ângulo da pasta
                if ang_dir in set(AUGMENT_ANGLES):
                    tasks.append((full_id, l, c, int(ang_dir)))
    # remover duplicatas
    tasks = sorted(set(tasks))
    return tasks

def rotate_k(img: np.ndarray, ang: int) -> np.ndarray:
    k = (ang // 90) % 4
    if k == 0:
        return img
    return np.ascontiguousarray(np.rot90(img, k=k))

def resolve_raw_std_path(full_id: str, l: int, c: int) -> Path | None:
    """
    Resolve o caminho do RAW base tentando sem padding (padrão do slice.py)
    e, em seguida, com zero-padding. Retorna o primeiro existente.
    """
    # sem padding
    p1 = RAW_STD_DIR / full_id / f"{full_id}_{l}_{c}_{SFX_R}.png"
    if p1.exists():
        return p1
    # com padding (compat)
    p2 = RAW_STD_DIR / full_id / f"{full_id}_{l:02d}_{c:02d}_{SFX_R}.png"
    if p2.exists():
        return p2
    return None

def make_one(full_id: str, l: int, c: int, ang: int, overwrite: bool=False) -> tuple[Path, bool]:
    raw_std = resolve_raw_std_path(full_id, l, c)
    out_dir = RAW_AUG_DIR / full_id
    out_dir.mkdir(parents=True, exist_ok=True)
    # Saída SEM padding, como no slice.py
    out = out_dir / f"{full_id}_{l}_{c}_{SFX_R}_{ang}.png"

    if out.exists() and not overwrite:
        return out, True

    if raw_std is None:
        return out, False

    img = cv2.imread(str(raw_std), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return out, False

    if img.shape[:2] != (PATCH_SIZE, PATCH_SIZE):
        img = cv2.resize(img, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)

    rot = rotate_k(img, ang)
    ok = cv2.imwrite(str(out), rot)
    return out, ok

def main():
    ap = argparse.ArgumentParser(description="Gerar RAW augmentados com base nos INV do Hough (base e/ou rotacionados).")
    ap.add_argument("--jobs", type=int, default=8, help="threads de I/O (default=8)")
    ap.add_argument("--overwrite", action="store_true", help="reescrever se já existir")
    args = ap.parse_args()

    todo = tasks_from_inv()
    if not todo:
        print("[augment] Nenhum patch com trilha encontrado nos INV do Hough (base ou rot).")
        return

    print(f"[augment] tarefas únicas (full,l,c,ang) = {len(todo)} | jobs={args.jobs} | overwrite={args.overwrite}")

    ok = skip = fail = 0
    with ThreadPoolExecutor(max_workers=max(1, int(args.jobs))) as ex:
        futs = [ex.submit(make_one, fid, l, c, ang, args.overwrite) for (fid, l, c, ang) in todo]
        for fu in as_completed(futs):
            out, res = fu.result()
            if res is True:
                ok += 1
            elif res is False:
                fail += 1
            else:
                skip += 1

    print(f"[augment] ok={ok}  skip={skip}  fail={fail}")
    print(f"[augment] out: {RAW_AUG_DIR}")

if __name__ == "__main__":
    main()
