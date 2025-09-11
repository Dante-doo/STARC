#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import cv2
import time
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

TILE_H, TILE_W = 224, 224  # tamanho alvo

def _to_u8_robust(img):
    """
    Normaliza qualquer imagem para uint8 [0,255] usando percentis 1-99.
    Evita saturação por outliers e mantém contraste comparável entre patches.
    """
    img = img.astype(np.float32)
    lo = np.percentile(img, 1.0)
    hi = np.percentile(img, 99.0)
    if hi <= lo:
        hi = img.max() if img.max() > lo else (lo + 1.0)
    img = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    return (img * 255.0).astype(np.uint8)

def _load_gray_224(path):
    """Lê imagem, converte para cinza e redimensiona para 224×224; retorna uint8 robusto."""
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise IOError(f"Erro ao ler: {path}")
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.shape[:2] != (TILE_H, TILE_W):
        im = cv2.resize(im, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
    return _to_u8_robust(im)

def _process_one(args):
    raw_path, hough_path, out_path = args
    if os.path.exists(out_path):
        return True
    try:
        raw_u8   = _load_gray_224(raw_path)
        hough_u8 = _load_gray_224(hough_path)

        # Combined científico: stack de canais independentes (R=raw, G=hough, B=zeros)
        zeros = np.zeros_like(raw_u8, dtype=np.uint8)
        combined = cv2.merge([raw_u8, hough_u8, zeros])  # 224×224×3

        ok = cv2.imwrite(out_path, combined)
        return bool(ok)
    except Exception as e:
        print(f"  ! Falha em {raw_path}: {e}")
        return False

def process_all(
    raw_root="../images/patches/raw",
    hough_root="../images/patches/hough",
    combined_root="../images/patches/combined",
    workers=None
):
    """
    Gera patches 'combined' 224×224×3 a partir de raw (R) e hough (G), B=zero.
    Mantém subpastas por fullX e nomes sincronizados: fullX_l_c_combined.png
    """
    os.makedirs(combined_root, exist_ok=True)

    jobs = []
    for full_dir in sorted(os.listdir(raw_root)):
        in_raw_dir   = os.path.join(raw_root,   full_dir)
        in_hough_dir = os.path.join(hough_root, full_dir)
        out_dir      = os.path.join(combined_root, full_dir)

        if not os.path.isdir(in_raw_dir):
            continue
        if not os.path.isdir(in_hough_dir):
            print(f"Aviso: hough não encontrado para {full_dir}, pulando.")
            continue
        os.makedirs(out_dir, exist_ok=True)

        for raw_path in glob.iglob(os.path.join(in_raw_dir, "*.png")):
            base = os.path.splitext(os.path.basename(raw_path))[0]
            hough_path = os.path.join(in_hough_dir, f"{base}_hough.png")
            if not os.path.exists(hough_path):
                # Hough ainda não gerado para este patch → pule
                continue
            out_path = os.path.join(out_dir, f"{base}_combined.png")
            jobs.append((raw_path, hough_path, out_path))

    total = len(jobs)
    if total == 0:
        print("Nada para processar (verifique se os patches Hough já existem).")
        return

    print(f"{total} combined patches a gerar. Iniciando…")
    t0, done, ok = time.time(), 0, 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_process_one, j) for j in jobs]
        for f in as_completed(futures):
            res = f.result()
            done += 1; ok += bool(res)
            if done % 5000 == 0 or done == total:
                elapsed = time.time() - t0
                print(f"{done}/{total} ({ok} salvos) em {elapsed/60:.1f} min")
    print(f"Concluído: {ok}/{total} arquivos gerados.")

def parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="Gera patches 'combined' 224×224×3 a partir de RAW (R) e HOUGH (G), com B=zero."
    )
    p.add_argument("--raw-root", default="../images/patches/raw", help="Diretório dos patches RAW.")
    p.add_argument("--hough-root", default="../images/patches/hough", help="Diretório dos patches HOUGH.")
    p.add_argument("--combined-root", default="../images/patches/combined", help="Saída dos patches COMBINED.")
    p.add_argument("--workers", type=int, default=None, help="Número de processos em paralelo.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process_all(
        raw_root=args.raw_root,
        hough_root=args.hough_root,
        combined_root=args.combined_root,
        workers=args.workers
    )
