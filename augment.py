#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Create rotational augments for positive RAW patches only.
# Reads labels/all_labels.csv, filters standard positives (no augment),
# rotates by angles in config.AUGMENT_ANGLES, and writes to:
# images/patches/raw/augment/<fullX>/<fullX_l_c_r_<angle>>.png

import re
import pandas as pd
from pathlib import Path
from PIL import Image
from code.config import (
    ROOT, LABELS_DIR, RAW_AUG_DIR, SFX_R, AUGMENT_ANGLES
)

RE_BASE = re.compile(r"(full\d+)_([0-9]+)_([0-9]+)_" + re.escape(SFX_R) + r"\.png$")

def to_abs(rel: str) -> Path:
    """Resolve repo-root-relative path to absolute Path."""
    return (ROOT / rel).resolve()

def main():
    labels_csv = LABELS_DIR / "all_labels.csv"
    if not labels_csv.exists():
        raise FileNotFoundError(f"Missing {labels_csv}. Run: python -m code.label first.")

    df = pd.read_csv(labels_csv)
    # standard raw positives only (exclude any '/augment/' rows)
    std_pos = df[(df["label"] == 1) & (~df["path"].str.contains("/augment/"))]["path"].tolist()

    created = skipped = 0
    for rel in std_pos:
        abs_path = to_abs(rel)
        m = RE_BASE.search(abs_path.name)
        if not m:
            continue  # unexpected naming; keep strict
        full_id, l, c = m.group(1), int(m.group(2)), int(m.group(3))

        # target: images/patches/raw/augment/<fullX>/
        out_dir = RAW_AUG_DIR / full_id
        out_dir.mkdir(parents=True, exist_ok=True)

        img = Image.open(abs_path).convert("L")  # load once

        for ang in AUGMENT_ANGLES:
            out_name = f"{full_id}_{l}_{c}_{SFX_R}_{int(ang)}.png"
            out_path = out_dir / out_name
            if out_path.exists():
                skipped += 1
                continue
            img.rotate(int(ang), resample=Image.BICUBIC, expand=False).save(out_path, "PNG", optimize=True)
            created += 1

    print(f"[augment] created={created}  skipped={skipped}")
    print("Run `python -m code.label` again to include augments in all_labels.csv.")

if __name__ == "__main__":
    main()
