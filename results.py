#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Aggregate results across arch/modality:
# - Scans results/metrics/**/test_metrics.json
# - Prints a sortable table (by balanced accuracy)
# - Saves a CSV summary in results/summary/summary.csv

import json
from pathlib import Path
import pandas as pd

from code.config import RESULTS_METRICS_DIR, RESULTS_SUMMARY_DIR

def collect():
    rows = []
    for met_file in RESULTS_METRICS_DIR.rglob("test_metrics.json"):
        try:
            d = json.loads(met_file.read_text())
            arch = d.get("arch", "na")
            mod  = d.get("modality", "na")
            thr  = d.get("threshold", None)
            m    = d.get("metrics", {})
            rows.append({
                "arch": arch,
                "modality": mod,
                "threshold": thr,
                "acc": m.get("acc", None),
                "prec1": m.get("prec1", None),
                "recall1": m.get("recall1", None),
                "tnr": m.get("tnr", None),
                "f1": m.get("f1", None),
                "f2": m.get("f2", None),
                "balacc": m.get("balacc", None),
                "tp": m.get("tp", None),
                "fp": m.get("fp", None),
                "tn": m.get("tn", None),
                "fn": m.get("fn", None),
            })
        except Exception:
            continue
    return pd.DataFrame(rows)

def main():
    df = collect()
    if df.empty:
        print("[WARN] No test_metrics.json files found under results/metrics/")
        return

    # sort by balanced accuracy desc, then by acc
    df = df.sort_values(by=["balacc", "acc"], ascending=False).reset_index(drop=True)

    # save summary CSV
    RESULTS_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_SUMMARY_DIR / "summary.csv"
    df.to_csv(out_csv, index=False)

    # pretty print
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(df)

    print(f"\n[OK] Summary saved at: {out_csv}")

if __name__ == "__main__":
    main()
