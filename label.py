#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import glob
import cv2
from collections import defaultdict

def main():
    # Diretórios (relativos à pasta 'code/')
    repo_root  = os.path.dirname(os.path.dirname(__file__))
    mask_root  = os.path.join(repo_root, "images", "patches", "asta_masks")
    raw_root   = os.path.join(repo_root, "images", "patches", "raw")
    labels_dir = os.path.join(repo_root, "labels")
    out_csv    = os.path.join(labels_dir, "all_labels.csv")

    os.makedirs(labels_dir, exist_ok=True)

    # Sanidade básica
    if not os.path.isdir(mask_root):
        raise FileNotFoundError(f"Diretório não encontrado: {mask_root}")
    if not os.path.isdir(raw_root):
        raise FileNotFoundError(f"Diretório não encontrado: {raw_root}")

    totals_by_full = defaultdict(lambda: {"n": 0, "pos": 0, "neg": 0})
    total_n = total_pos = total_neg = 0
    missing_raw = 0

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])  # cabeçalho

        # Varre subpastas por fullX em asta_masks
        for full_dir in sorted(os.listdir(mask_root)):
            mask_subdir = os.path.join(mask_root, full_dir)
            raw_subdir  = os.path.join(raw_root,  full_dir)
            if not os.path.isdir(mask_subdir):
                continue
            if not os.path.isdir(raw_subdir):
                print(f"[AVISO] Pasta raw ausente para {full_dir}, pulando.")
                continue

            # Para cada *_mask.png
            mask_list = sorted(glob.glob(os.path.join(mask_subdir, "*_mask.png")))
            if not mask_list:
                print(f"[AVISO] Sem máscaras em {mask_subdir}")
                continue

            for i, mask_path in enumerate(mask_list, 1):
                base_mask = os.path.basename(mask_path)                 # fullX_l_c_mask.png
                raw_name  = base_mask.replace("_mask", "")              # fullX_l_c.png
                raw_path  = os.path.join(raw_subdir, raw_name)

                if not os.path.isfile(raw_path):
                    missing_raw += 1
                    # Se quiser falhar quando não encontrar o raw, troque para:
                    # raise FileNotFoundError(f"Raw ausente: {raw_path}")
                    continue

                # Lê máscara em gray e decide rótulo
                mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_img is None:
                    print(f"[AVISO] Falha ao ler máscara: {mask_path}")
                    continue

                label = 1 if cv2.countNonZero(mask_img) > 0 else 0

                # Caminho relativo para o CSV (padrão Linux-friendly)
                rel_raw = os.path.relpath(raw_path, repo_root).replace("\\", "/")
                writer.writerow([rel_raw, label])

                # Estatísticas
                totals_by_full[full_dir]["n"] += 1
                if label == 1:
                    totals_by_full[full_dir]["pos"] += 1
                    total_pos += 1
                else:
                    totals_by_full[full_dir]["neg"] += 1
                    total_neg += 1
                total_n += 1

                if i % 10000 == 0:
                    print(f"[{full_dir}] {i} máscaras processadas...")

    # Resumo
    print("\n=== Resumo por imagem full ===")
    for full_dir in sorted(totals_by_full.keys()):
        d = totals_by_full[full_dir]
        print(f"{full_dir}: total={d['n']}, positivos={d['pos']}, negativos={d['neg']}")

    print("\n=== Resumo geral ===")
    print(f"Total linhas CSV: {total_n}")
    print(f"Positivos (label=1): {total_pos}")
    print(f"Negativos (label=0): {total_neg}")
    if missing_raw > 0:
        print(f"[AVISO] {missing_raw} máscaras tinham raw correspondente ausente.")

    print(f"\nCSV salvo em: {out_csv}")

if __name__ == "__main__":
    main()
