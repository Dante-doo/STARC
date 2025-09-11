#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import cv2
import time
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# ===========================
# Parâmetros fixos
# ===========================
TILE_H, TILE_W = 224, 224
N_THETA = 180                  # 1° por coluna
KERNEL_TOPHAT = (15, 15)
KERNEL_OPEN   = (3, 3)

# ===========================
# LUTs globais por processo
# ===========================
_LUT_RIDX = None
_N_RHO    = None

def _init_worker(tile_h=TILE_H, tile_w=TILE_W, n_theta=N_THETA):
    """Inicializador por processo: pré-computa LUT (y,x,theta) -> rho_idx."""
    global _LUT_RIDX, _N_RHO
    cv2.setNumThreads(0)  # evita oversubscription

    thetas = np.deg2rad(np.linspace(-90.0, 89.0, n_theta, endpoint=True)).astype(np.float32)
    cos_t  = np.cos(thetas).astype(np.float32)
    sin_t  = np.sin(thetas).astype(np.float32)

    rho_max = int(np.ceil(np.hypot(tile_h, tile_w)))
    _N_RHO  = 2 * rho_max + 1  # [-rho_max, +rho_max] -> [0, N_RHO)

    ys, xs = np.indices((tile_h, tile_w), dtype=np.float32)
    rhos = ys[..., None] * sin_t[None, None, :] + xs[..., None] * cos_t[None, None, :]
    _LUT_RIDX = np.rint(rhos + rho_max).astype(np.int16)


def _build_hough_image(patch_bgr):
    """Constrói imagem Hough (224×224) de um patch 224×224 usando LUT."""
    global _LUT_RIDX, _N_RHO

    # Gray
    if patch_bgr.ndim == 3:
        gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = patch_bgr

    # Pré-processamento (mesmo da versão base)
    tophat = cv2.morphologyEx(
        gray, cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, KERNEL_TOPHAT)
    )
    eq   = cv2.equalizeHist(tophat)
    blur = cv2.medianBlur(eq, 3)

    # Bordas: Sobel + Otsu + abertura
    sx  = cv2.Sobel(blur, cv2.CV_16S, 1, 0, ksize=3)
    sy  = cv2.Sobel(blur, cv2.CV_16S, 0, 1, ksize=3)
    mag = cv2.convertScaleAbs(cv2.magnitude(sx.astype(np.float32), sy.astype(np.float32)))
    _, bin_edges = cv2.threshold(mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_edges = cv2.morphologyEx(
        bin_edges, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, KERNEL_OPEN),
        iterations=1
    )

    ys, xs = np.nonzero(bin_edges)
    if len(xs) == 0:
        return np.zeros((TILE_H, TILE_W), dtype=np.uint8)

    # Acúmulo com LUT
    ridx = _LUT_RIDX[ys, xs]  # shape: [num_pts, N_THETA]
    acc = np.zeros((_N_RHO, N_THETA), dtype=np.uint32)
    for t in range(N_THETA):
        acc[:, t] = np.bincount(ridx[:, t], minlength=_N_RHO)

    # Normalização robusta (1–99%) + resize para 224×224
    acc = acc.astype(np.float32)
    lo = np.percentile(acc, 1.0); hi = np.percentile(acc, 99.0)
    if hi <= lo:
        hi = acc.max() if acc.max() > 0 else (lo + 1.0)
    acc = np.clip((acc - lo) / (hi - lo), 0.0, 1.0)
    img = (acc * 255.0).astype(np.uint8)
    img = cv2.resize(img, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
    return img


def _process_one(args):
    in_path, out_path = args
    if os.path.exists(out_path):
        return True
    patch = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
    if patch is None:
        return False
    hough_img = _build_hough_image(patch)
    return cv2.imwrite(out_path, hough_img)


def process_all_patches(raw_root="../images/patches/raw",
                        hough_root="../images/patches/hough",
                        workers=None):
    os.makedirs(hough_root, exist_ok=True)

    jobs = []
    for full_dir in sorted(os.listdir(raw_root)):
        in_dir  = os.path.join(raw_root,  full_dir)
        out_dir = os.path.join(hough_root, full_dir)
        if not os.path.isdir(in_dir):
            continue
        os.makedirs(out_dir, exist_ok=True)

        for in_path in glob.iglob(os.path.join(in_dir, "*.png")):
            base, _ = os.path.splitext(os.path.basename(in_path))
            out_path = os.path.join(out_dir, f"{base}_hough.png")
            jobs.append((in_path, out_path))

    total = len(jobs)
    if total == 0:
        print("Nada para processar.")
        return

    print(f"{total} patches encontrados. Iniciando…")
    t0, done, ok = time.time(), 0, 0

    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
        futures = [ex.submit(_process_one, j) for j in jobs]
        for f in as_completed(futures):
            res = f.result()
            done += 1; ok += bool(res)
            if done % 5000 == 0 or done == total:
                elapsed = time.time() - t0
                print(f"{done}/{total} ({ok} salvos) em {elapsed/60:.1f} min")

    print(f"Concluído: {ok}/{total} arquivos gerados.")


def parse_args():
    p = argparse.ArgumentParser(description="Gera patches no domínio de Hough em paralelo (sem perda de fidelidade).")
    p.add_argument("--raw-root", default="../images/patches/raw",
                   help="Pasta de entrada dos patches crus (por fullX).")
    p.add_argument("--hough-root", default="../images/patches/hough",
                   help="Saída dos patches Hough (por fullX).")
    p.add_argument("--workers", type=int, default=None,
                   help="Nº de processos em paralelo (padrão: os.cpu_count()).")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process_all_patches(
        raw_root=args.raw_root,
        hough_root=args.hough_root,
        workers=args.workers
    )
