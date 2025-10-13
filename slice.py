#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Slice full images into a centered 9x9 grid of 224x224 patches,
# after downscaling each full to a working max side (default: 2048).
#
# Outputs:
#   images/patches/raw/standart/<fullX>/<fullX_r_c_r>.png
#   images/patches/mask/standart/<fullX>/<fullX_r_c_m>.png
#
# Notes:
# - Raw is resized with high-quality filter (LANCZOS/BILINEAR as set in config).
# - Mask is resized with NEAREST to preserve labels.
# - Grid is centered within the downscaled image.
# - File/directory layout matches the rest of the pipeline.

from pathlib import Path
from PIL import Image

from code.config import (
    # inputs/outputs
    FULL_RAW_DIR, FULL_MASKS_DIR,
    RAW_STD_DIR, MASK_STD_DIR,
    # naming
    SFX_R, SFX_M,
    # slicing knobs
    SLICE_TILE, SLICE_GRID_R, SLICE_GRID_C,
    SLICE_DOWNSCALE_MAXSIDE, SLICE_RESAMPLE_RAW, SLICE_RESAMPLE_MASK,
)

# ------------------- helpers -------------------
def _pil_resample(name: str):
    """Map string knob -> PIL resample enum."""
    name = (name or "").lower()
    if name in ("nearest", "nn"):       return Image.NEAREST
    if name in ("bilinear", "linear"):  return Image.BILINEAR
    if name in ("bicubic",):            return Image.BICUBIC
    if name in ("lanczos", "antialias"):return Image.LANCZOS
    return Image.BILINEAR

def _center_offsets(W: int, H: int, TW: int, TH: int):
    """Compute top-left offsets to center a TW×TH region in W×H."""
    offx = max(0, (W - TW) // 2)
    offy = max(0, (H - TH) // 2)
    return offx, offy

def _downscale_keep_aspect(img: Image.Image, maxside: int, resample) -> Image.Image:
    """Resize so that max(height, width) == maxside, keeping aspect ratio."""
    W, H = img.size
    if max(W, H) <= maxside:
        return img
    if W >= H:
        newW = maxside
        newH = int(round(H * (maxside / float(W))))
    else:
        newH = maxside
        newW = int(round(W * (maxside / float(H))))
    return img.resize((newW, newH), resample=resample)

# ------------------- main -------------------
def main():
    raws = sorted(FULL_RAW_DIR.glob("full*.png"))
    if not raws:
        raise FileNotFoundError(f"No 'full*.png' in {FULL_RAW_DIR}")

    # Resolve PIL resample methods from config strings
    res_raw  = _pil_resample(SLICE_RESAMPLE_RAW)
    res_mask = _pil_resample(SLICE_RESAMPLE_MASK)

    P  = int(SLICE_TILE)
    R  = int(SLICE_GRID_R)
    C  = int(SLICE_GRID_C)
    TW = C * P
    TH = R * P

    for rp in raws:
        fid = rp.stem  # e.g., 'full12'
        mp  = FULL_MASKS_DIR / f"{fid}_mask.png"
        if not mp.exists():
            print(f"[slice] Missing mask for {fid}, skipping.")
            continue

        # Ensure per-full subfolders exist (raw and mask)
        raw_sub  = (RAW_STD_DIR  / fid); raw_sub.mkdir(parents=True, exist_ok=True)
        mask_sub = (MASK_STD_DIR / fid); mask_sub.mkdir(parents=True, exist_ok=True)

        # Load raw/mask; convert to L; resize to working max side
        raw0 = Image.open(rp).convert("L")
        msk0 = Image.open(mp).convert("L")

        raw = _downscale_keep_aspect(raw0, int(SLICE_DOWNSCALE_MAXSIDE), res_raw)
        # Resize mask with NEAREST (set in config) then binarize to {0,255}
        msk = _downscale_keep_aspect(msk0, int(SLICE_DOWNSCALE_MAXSIDE), res_mask)
        msk = msk.point(lambda v: 255 if v > 0 else 0, "L")

        W, H = raw.size
        if W < TW or H < TH:
            print(f"[slice][warn] Downscaled size {W}x{H} smaller than grid {TW}x{TH}; "
                  f"reduce grid or maxside. Skipping {fid}.")
            continue

        offx, offy = _center_offsets(W, H, TW, TH)

        # Save deterministic, globally unique names
        for r in range(R):
            y = offy + r * P
            for c in range(C):
                x = offx + c * P
                box = (x, y, x + P, y + P)
                raw.crop(box).save(raw_sub  / f"{fid}_{r}_{c}_{SFX_R}.png", "PNG", optimize=True)
                msk.crop(box).save(mask_sub / f"{fid}_{r}_{c}_{SFX_M}.png", "PNG", optimize=True)

        print(f"[slice] {fid}: {R}x{C} patches @ {P} saved "
              f"(downscaled to maxside={SLICE_DOWNSCALE_MAXSIDE}, canvas={W}x{H}, offsets=({offx},{offy}))")

if __name__ == "__main__":
    main()
