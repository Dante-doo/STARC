#!/usr/bin/env python3
import os
import glob
import cv2

def slice_full_images(
    raw_dir="../images/full/raw",
    mask_dir="../images/full/full_asta_masks",
    out_raw="../images/patches/raw",
    out_mask="../images/patches/asta_masks",
    tile_size=224,
    grid_size=47
):
    # garante diretórios base
    os.makedirs(out_raw, exist_ok=True)
    os.makedirs(out_mask, exist_ok=True)

    for raw_path in glob.glob(os.path.join(raw_dir, "*")):
        base = os.path.splitext(os.path.basename(raw_path))[0]
        mask_path = os.path.join(mask_dir, f"{base}_mask.png")
        if not os.path.isfile(mask_path):
            print(f"Aviso: máscara não encontrada para {base}, pulando.")
            continue

        # cria subpastas específicas para esta imagem
        raw_subdir  = os.path.join(out_raw, base)
        mask_subdir = os.path.join(out_mask, base)
        os.makedirs(raw_subdir,  exist_ok=True)
        os.makedirs(mask_subdir, exist_ok=True)

        raw_img  = cv2.imread(raw_path, cv2.IMREAD_UNCHANGED)
        mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        h, w     = raw_img.shape[:2]

        count = 0
        for row in range(grid_size):
            for col in range(grid_size):
                y = row * tile_size
                x = col * tile_size
                if y + tile_size > h or x + tile_size > w:
                    continue

                patch_raw  = raw_img[y:y+tile_size, x:x+tile_size]
                patch_mask = mask_img[y:y+tile_size, x:x+tile_size]

                out_raw_path  = os.path.join(
                    raw_subdir,  f"{base}_{row}_{col}.png")
                out_mask_path = os.path.join(
                    mask_subdir, f"{base}_{row}_{col}_mask.png")

                cv2.imwrite(out_raw_path,  patch_raw)
                cv2.imwrite(out_mask_path, patch_mask)
                count += 1

        print(f"Fatiado {base}: {count} patches gerados em '{base}/'")

if __name__ == "__main__":
    slice_full_images()
