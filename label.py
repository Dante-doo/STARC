#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Build labels/all_labels.csv from the standardized layout.
# - Standard: mask decides label (0/1)
# - Augment (raw): always label=1
# Output CSV: labels/all_labels.csv with columns [path,label]
# 'path' is repo-root-relative and always points to a RAW image.

from pathlib import Path
import re, csv, cv2
from code.config import (
    ROOT, LABELS_DIR,
    RAW_STD_DIR, MASK_STD_DIR, RAW_AUG_DIR,
    SFX_R, SFX_M,
)

# Strict filename patterns
RE_MASK = re.compile(r"^(full\d+)_(\d+)_(\d+)_" + re.escape(SFX_M) + r"\.png$")
RE_AUG  = re.compile(r"^(full\d+)_(\d+)_(\d+)_" + re.escape(SFX_R) + r"_(\d+)\.png$")

def relpath(p: Path) -> str:
    """Project-root relative POSIX path for CSV."""
    return p.resolve().relative_to(ROOT).as_posix()

def any_white(mask_png: Path) -> bool:
    """True if any non-zero pixel exists (fast via OpenCV)."""
    img = cv2.imread(str(mask_png), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read mask: {mask_png}")
    return cv2.countNonZero(img) > 0

def main():
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = LABELS_DIR / "all_labels.csv"

    total_base = total_pos = total_neg = total_aug = 0

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label"])

        # 1) STANDARD (mask -> label)
        for full_dir in sorted(MASK_STD_DIR.glob("full*")):
            if not full_dir.is_dir():
                continue
            full_id = full_dir.name
            raw_sub  = RAW_STD_DIR  / full_id
            mask_sub = MASK_STD_DIR / full_id
            if not raw_sub.exists() or not mask_sub.exists():
                print(f"[label] Missing raw/mask folder for {full_id}, skipping.")
                continue

            for mpath in sorted(mask_sub.glob(f"{full_id}_*_*_{SFX_M}.png")):
                m = RE_MASK.match(mpath.name)
                if not m:
                    continue  # unexpected name
                l, c = int(m.group(2)), int(m.group(3))
                rname = f"{full_id}_{l}_{c}_{SFX_R}.png"
                rpath = raw_sub / rname
                if not rpath.exists():
                    print(f"[label] Missing RAW for mask: {rpath}")
                    continue

                y = 1 if any_white(mpath) else 0
                w.writerow([relpath(rpath), y])

                total_base += 1
                if y == 1: total_pos += 1
                else:      total_neg += 1

        # 2) AUGMENT (raw only; always label=1)
        if RAW_AUG_DIR.exists():
            for full_dir in sorted(RAW_AUG_DIR.glob("full*")):
                if not full_dir.is_dir():
                    continue
                full_id = full_dir.name
                for apath in sorted(full_dir.glob(f"{full_id}_*_*_{SFX_R}_*.png")):
                    if not RE_AUG.match(apath.name):
                        continue  # keep strict naming
                    w.writerow([relpath(apath), 1])
                    total_aug += 1

    # Summary
    print("\n=== labels summary ===")
    print(f"standard: total={total_base}  pos={total_pos}  neg={total_neg}")
    print(f"augment : total={total_aug} (all labeled 1)")
    print(f"csv -> {out_csv}")

if __name__ == "__main__":
    main()
