#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Slice full images into a centered 47x47 grid of 224x224 patches.
# Outputs:
#   images/patches/raw/standart/<fullX>/<fullX_l_c_r>.png
#   images/patches/mask/standart/<fullX>/<fullX_l_c_m>.png

from PIL import Image
from code.config import (
    FULL_RAW_DIR, FULL_MASKS_DIR,
    RAW_STD_DIR, MASK_STD_DIR,
    SFX_R, SFX_M, PATCH_SIZE, GRID_ROWS, GRID_COLS
)

def main():
    """Slice all full images + masks into fixed-size grayscale patches."""
    raws = sorted(FULL_RAW_DIR.glob("full*.png"))
    if not raws:
        raise FileNotFoundError(f"No 'full*.png' in {FULL_RAW_DIR}")

    for rp in raws:
        fid = rp.stem  # e.g., 'full12'
        mp = FULL_MASKS_DIR / f"{fid}_mask.png"
        if not mp.exists():  # minimal validation
            print(f"[slice] Missing mask for {fid}, skipping.")
            continue

        # Ensure per-full subfolders exist (raw and mask)
        raw_sub  = (RAW_STD_DIR  / fid); raw_sub.mkdir(parents=True, exist_ok=True)
        mask_sub = (MASK_STD_DIR / fid); mask_sub.mkdir(parents=True, exist_ok=True)

        # Load raw as 8-bit grayscale; binarize mask to {0,255}
        raw = Image.open(rp).convert("L")
        msk = Image.open(mp).convert("L").point(lambda v: 255 if v > 0 else 0, "L")

        # Centered grid offsets
        W, H = raw.size
        TW, TH = GRID_COLS * PATCH_SIZE, GRID_ROWS * PATCH_SIZE
        offx, offy = (W - TW) // 2, (H - TH) // 2

        # Save deterministic, globally unique names
        for l in range(GRID_ROWS):
            y = offy + l * PATCH_SIZE
            for c in range(GRID_COLS):
                x = offx + c * PATCH_SIZE
                box = (x, y, x + PATCH_SIZE, y + PATCH_SIZE)
                raw.crop(box).save(raw_sub  / f"{fid}_{l}_{c}_{SFX_R}.png", "PNG", optimize=True)
                msk.crop(box).save(mask_sub / f"{fid}_{l}_{c}_{SFX_M}.png", "PNG", optimize=True)

        print(f"[slice] {fid}: 47x47 patches OK")

if __name__ == "__main__":
    main()
