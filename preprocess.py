#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import argparse
from pathlib import Path

import lmdb
import msgpack
import numpy as np
import pandas as pd
from torchvision.io import read_image, ImageReadMode


def map_path_for_modality(rel_path: str, modality: str) -> str:
    """
    Converte o caminho do CSV (sempre POSIX) para a modalidade desejada.
    - raw      : mantém igual
    - hough    : patches/hough + sufixo _hough
    - combined : patches/combined + sufixo _combined
    """
    p = rel_path.replace("\\", "/")
    if modality == "raw":
        return p
    if "/patches/raw/" not in p:
        raise ValueError(f"Caminho RAW inesperado: {p}")
    stem, ext = os.path.splitext(p)
    if modality == "hough":
        return stem.replace("/patches/raw/", "/patches/hough/") + "_hough" + ext
    if modality == "combined":
        return stem.replace("/patches/raw/", "/patches/combined/") + "_combined" + ext
    raise ValueError("modality inválida")


def write_lmdb(modality: str, repo_root: Path, out_dir: Path, mapsize_gb: int = 200, commit_every: int = 2000):
    """
    Lê all_labels.csv e grava um único LMDB por modalidade em models/lmdb/<modality>
    - chave: path POSIX exatamente como ficará no CSV da modalidade escolhida
    - valor: msgpack com {"img": uint8[C,H,W] em list(), "label": int}
    """
    labels_dir = repo_root / "labels"
    csv_all = labels_dir / "all_labels.csv"
    if not csv_all.exists():
        raise FileNotFoundError(f"{csv_all} não encontrado. Rode label.py antes.")

    df = pd.read_csv(csv_all)
    # normaliza path do CSV
    df["path"] = df["path"].astype(str).str.replace("\\", "/", regex=False)

    out_dir.mkdir(parents=True, exist_ok=True)

    env = lmdb.open(
        str(out_dir),
        map_size=mapsize_gb * (1024**3),
        subdir=True,
        readonly=False,
        lock=True,
        readahead=False,
        meminit=False
    )

    n_written = 0
    txn = env.begin(write=True)
    try:
        for i, row in df.iterrows():
            rel_raw = row["path"]
            rel_mod = map_path_for_modality(rel_raw, modality)
            full_path = (repo_root / rel_mod)

            if not full_path.is_file():
                # avisa e pula, sem abortar tudo
                if i < 20:
                    print(f"[WARN] arquivo ausente: {full_path}")
                continue

            # Tensor uint8 [C,H,W]
            img = read_image(str(full_path), mode=ImageReadMode.RGB)  # uint8
            key = rel_mod.encode("utf-8")

            # serializa em list() para não depender de msgpack_numpy
            payload = {
                "img": img.numpy().tolist(),
                "label": int(row["label"])
            }
            buf = msgpack.dumps(payload, use_bin_type=True)
            txn.put(key, buf)
            n_written += 1

            if (i + 1) % commit_every == 0:
                txn.commit()
                print(f"[{i+1}/{len(df)}] commit parcial | gravados={n_written}")
                txn = env.begin(write=True)

        txn.commit()
    finally:
        env.sync()
        env.close()

    print(f"[OK] LMDB salvo em {out_dir} | amostras escritas: {n_written}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["raw", "hough", "combined"], required=True)
    ap.add_argument("--mapsize-gb", type=int, default=200, help="Tamanho do LMDB (GB).")
    ap.add_argument("--commit-every", type=int, default=2000, help="Itens por transação (commit) para reduzir RAM.")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "models" / "lmdb" / args.modality

    write_lmdb(args.modality, repo_root, out_dir, mapsize_gb=args.mapsize_gb, commit_every=args.commit_every)


if __name__ == "__main__":
    main()
