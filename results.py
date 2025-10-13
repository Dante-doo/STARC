#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aggregate and visualize test results.

Scans results/metrics/**/test_metrics.json and builds:
- all_runs.csv  (todas as execuções, ordenadas)
- best_by_group.csv  (melhor por arch/modality/aug_policy)
- mean_by_group.csv  (média por grupo)
- balacc_best_by_group.png
- balacc_mean_by_group.png
- confusion_<arch>_<modality>_<aug>.png  (uma por melhor run do grupo)

Use --no-filter para não remover execuções patológicas (tnr==0 && recall==1).
"""

from __future__ import annotations
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------- config import + fallbacks ----------------------------
from code.config import (
    RESULTS_DIR, RESULTS_METRICS_DIR, RESULTS_PREDS_DIR,
    RESULTS_SUMMARY_DIR, RESULTS_VIS_DIR,
)

RESULTS_METRICS_DIR = Path(RESULTS_METRICS_DIR) if 'RESULTS_METRICS_DIR' in globals() else (Path(RESULTS_DIR) / "metrics")
RESULTS_PREDS_DIR   = Path(RESULTS_PREDS_DIR)   if 'RESULTS_PREDS_DIR'   in globals() else (Path(RESULTS_DIR) / "predictions")
RESULTS_SUMMARY_DIR = Path(RESULTS_SUMMARY_DIR) if 'RESULTS_SUMMARY_DIR' in globals() else (Path(RESULTS_DIR) / "summary")
RESULTS_VIS_DIR     = Path(RESULTS_VIS_DIR)     if 'RESULTS_VIS_DIR'     in globals() else (Path(RESULTS_DIR) / "vis")

# ------------------------------- IO helpers ----------------------------------
def load_metrics_files(root: Path) -> List[Path]:
    return sorted(root.rglob("test_metrics.json"))

def _infer_triplet_from_path(met_file: Path) -> Tuple[str, str, str]:
    """
    Tenta inferir (arch, modality, aug_policy) a partir do caminho do arquivo.
    Suporta dirs 'resnet', 'deepnet121' e 'deepnet' (compat).
    Ex.: results/metrics/deepnet/raw/aug-rebalance/test_metrics.json
    """
    txt = str(met_file).replace("\\", "/")
    parts = txt.split("/")
    # procurar 'metrics/<arch>/<modality>/(aug-xxx)?/test_metrics.json'
    try:
        m_idx = parts.index("metrics")
        arch = parts[m_idx+1] if m_idx+1 < len(parts) else "na"
        modality = parts[m_idx+2] if m_idx+2 < len(parts) else "na"
        aug = parts[m_idx+3] if (m_idx+3 < len(parts) and parts[m_idx+3].startswith("aug-")) else None
    except ValueError:
        arch, modality, aug = "na", "na", None

    # normalizar arch
    arch_norm = arch.lower()
    if arch_norm == "deepnet":
        arch_norm = "deepnet"          # compat antigo
    elif arch_norm == "deepnet121":
        arch_norm = "deepnet121"
    elif arch_norm.startswith("resnet"):
        arch_norm = "resnet"
    # modality fica como está (raw|hough|combined)
    aug_policy = "rebalance"
    if aug:
        if "aug-all" in aug: aug_policy = "all"
        elif "aug-none" in aug: aug_policy = "none"
        else: aug_policy = "rebalance"
    return arch_norm, modality, aug_policy

def load_one(met_file: Path) -> Dict[str, Any]:
    try:
        d = json.loads(met_file.read_text())
        m = d.get("metrics", {})
        row = {
            "arch": d.get("arch", "na"),
            "modality": d.get("modality", "na"),
            "aug_policy": d.get("aug_policy", d.get("aug", "unknown")),
            "ckpt": d.get("ckpt", "best"),
            "threshold": d.get("threshold", None),
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
            "n": m.get("n", None),
            "metrics_path": str(met_file),
        }
        # fallback: inferir pelo caminho se faltar algo
        a, mod, augp = _infer_triplet_from_path(met_file)
        if row["arch"] in ("na", None, ""): row["arch"] = a
        if row["modality"] in ("na", None, ""): row["modality"] = mod
        if row["aug_policy"] in ("unknown", None, ""): row["aug_policy"] = augp

        # normalizar arch vindo do JSON: permitir 'deepnet' e 'deepnet121'
        arch_l = str(row["arch"]).lower()
        if arch_l in ("deepnet", "deepnet121"):
            row["arch"] = arch_l
        elif arch_l.startswith("resnet"):
            row["arch"] = "resnet"

        return row
    except Exception:
        return {}

def collect_df() -> pd.DataFrame:
    files = load_metrics_files(RESULTS_METRICS_DIR)
    rows = [load_one(p) for p in files]
    rows = [r for r in rows if r]
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for c in ("acc","prec1","recall1","tnr","f1","f2","balacc","threshold"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("tp","fp","tn","fn","n"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    # aug_policy pelo caminho quando ausente/unknown
    if "aug_policy" in df.columns:
        inferred = []
        for mp in df.get("metrics_path", []):
            txt = str(mp).replace("\\", "/")
            if "/aug-all/" in txt:
                inferred.append("all")
            elif "/aug-none/" in txt:
                inferred.append("none")
            elif "/aug-rebalance/" in txt or "/aug-rebalanced/" in txt:
                inferred.append("rebalance")
            else:
                inferred.append(np.nan)
        df["aug_policy"] = df["aug_policy"].replace({"unknown": np.nan})
        df["aug_policy"] = df["aug_policy"].fillna(pd.Series(inferred, index=df.index))
        df["aug_policy"] = df["aug_policy"].fillna("rebalance")
    else:
        df["aug_policy"] = "rebalance"

    # garantir arch normalizada | deepnet121 e deepnet aparecem como estão
    df["arch"] = df["arch"].astype(str).str.lower()
    df.loc[df["arch"].str.startswith("resnet"), "arch"] = "resnet"
    # deepnet e deepnet121 mantidas

    df = df.sort_values(by=["balacc", "acc"], ascending=False, na_position="last").reset_index(drop=True)
    return df

# ------------------------------ outlier filter -------------------------------
def filter_outliers(df: pd.DataFrame, enable: bool = True) -> pd.DataFrame:
    if not enable or df.empty:
        return df
    mask = ~((df["tnr"].fillna(0) <= 1e-9) & (df["recall1"].fillna(0) >= 1.0))
    return df.loc[mask].reset_index(drop=True)

# ------------------------------- save helpers --------------------------------
def ensure_dirs():
    RESULTS_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_VIS_DIR.mkdir(parents=True, exist_ok=True)

def save_csvs(df_all: pd.DataFrame):
    out_all = RESULTS_SUMMARY_DIR / "all_runs.csv"
    df_all.to_csv(out_all, index=False)

    group_cols = ["arch", "modality", "aug_policy"]
    best = (df_all.sort_values(["balacc","acc"], ascending=False)
                  .groupby(group_cols, as_index=False).head(1)
                  .reset_index(drop=True))
    out_best = RESULTS_SUMMARY_DIR / "best_by_group.csv"
    best.to_csv(out_best, index=False)

    mean_cols = ["acc","prec1","recall1","tnr","f1","f2","balacc"]
    mean_df = (df_all.groupby(group_cols, dropna=False)[mean_cols]
                     .mean().reset_index())
    out_mean = RESULTS_SUMMARY_DIR / "mean_by_group.csv"
    mean_df.to_csv(out_mean, index=False)

    return out_all, out_best, out_mean, best, mean_df

# --------------------------------- plotting ----------------------------------
def bar_plot(df: pd.DataFrame, title: str, fname: Path, value_col: str = "balacc"):
    if df.empty:
        return
    labels = df.apply(lambda r: f"{r['arch']}/{r['modality']}/{r['aug_policy']}", axis=1)
    vals = df[value_col].astype(float).values

    plt.figure(figsize=(max(6, len(df)*1.2), 4.5))
    idx = np.arange(len(df))
    plt.bar(idx, vals)
    plt.xticks(idx, labels, rotation=35, ha="right")
    plt.ylabel(value_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=160)
    plt.close()

def plot_confusion_matrix(tp:int, fp:int, tn:int, fn:int, title:str, out_path:Path, normalize:bool=True):
    cm = np.array([[tn, fp],
                   [fn, tp]], dtype=float)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True) + 1e-9
        cm_norm = cm / row_sums
        show = cm_norm
        fmt_vals = np.array([[f"{int(tn)}\n{cm_norm[0,0]*100:.1f}%",
                              f"{int(fp)}\n{cm_norm[0,1]*100:.1f}%"],
                             [f"{int(fn)}\n{cm_norm[1,0]*100:.1f}%",
                              f"{int(tp)}\n{cm_norm[1,1]*100:.1f}%"]], dtype=object)
        subtitle = " (counts + row-normalized %)"
    else:
        show = cm
        fmt_vals = np.array([[f"{int(tn)}", f"{int(fp)}"],
                             [f"{int(fn)}", f"{int(tp)}"]], dtype=object)
        subtitle = " (counts)"

    plt.figure(figsize=(4.2, 4.0))
    im = plt.imshow(show)
    plt.title(title + subtitle)
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.xticks([0,1], ["0", "1"])
    plt.yticks([0,1], ["0", "1"])

    # grid + annotations
    for (i, j), _ in np.ndenumerate(show):
        plt.text(j, i, fmt_vals[i, j], ha="center", va="center", fontsize=10)

    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def save_confusions(best_df: pd.DataFrame):
    if best_df.empty:
        return []
    out_files = []
    for _, r in best_df.iterrows():
        # alguns runs antigos podem não ter todos os campos
        tp = int(r.get("tp,") if "tp," in r else r.get("tp", 0)) if pd.notna(r.get("tp", np.nan)) else 0
        fp = int(r.get("fp,") if "fp," in r else r.get("fp", 0)) if pd.notna(r.get("fp", np.nan)) else 0
        tn = int(r.get("tn,") if "tn," in r else r.get("tn", 0)) if pd.notna(r.get("tn", np.nan)) else 0
        fn = int(r.get("fn,") if "fn," in r else r.get("fn", 0)) if pd.notna(r.get("fn", np.nan)) else 0
        tag = f"{r['arch']}_{r['modality']}_{r['aug_policy']}".replace("/", "-")
        out_png = RESULTS_VIS_DIR / f"confusion_{tag}.png"
        title = f"Confusion: {r['arch']}/{r['modality']}/{r['aug_policy']}"
        plot_confusion_matrix(tp, fp, tn, fn, title, out_png, normalize=True)
        out_files.append(out_png)
    return out_files

# ---------------------------------- main -------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Aggregate and visualize test results")
    ap.add_argument("--no-filter", action="store_true", help="don't remove pathological/outlier runs")
    args = ap.parse_args()

    ensure_dirs()
    df = collect_df()
    if df.empty:
        print(f"[WARN] No test_metrics.json found under: {RESULTS_METRICS_DIR}")
        return

    print(f"[INFO] Loaded {len(df)} runs from {RESULTS_METRICS_DIR}")

    df_f = filter_outliers(df, enable=(not args.no_filter))
    removed = len(df) - len(df_f)
    if removed > 0:
        print(f"[INFO] Removed {removed} pathological run(s) by filter.")

    all_csv, best_csv, mean_csv, best_df, mean_df = save_csvs(df_f)

    bar_plot(best_df, "Balanced Accuracy (best by group)", RESULTS_VIS_DIR / "balacc_best_by_group.png", value_col="balacc")
    bar_plot(mean_df.sort_values(["balacc","acc"], ascending=False),
             "Balanced Accuracy (mean by group)",
             RESULTS_VIS_DIR / "balacc_mean_by_group.png",
             value_col="balacc")

    conf_pngs = save_confusions(best_df)

    with pd.option_context("display.max_columns", None, "display.width", 160):
        print("\n=== TOP 20 RUNS (by balacc, acc) ===")
        print(df_f.sort_values(by=["balacc","acc"], ascending=False).head(20))
        print("\n=== BEST BY GROUP ===")
        print(best_df.sort_values(by=["balacc","acc"], ascending=False))
        print("\n=== MEAN BY GROUP ===")
        print(mean_df.sort_values(by=["balacc","acc"], ascending=False))

    print(f"\n[OK] Saved:\n- {all_csv}\n- {best_csv}\n- {mean_csv}\n"
          f"Plots:\n- {RESULTS_VIS_DIR/'balacc_best_by_group.png'}\n"
          f"- {RESULTS_VIS_DIR/'balacc_mean_by_group.png'}")
    if conf_pngs:
        print("- Confusions:")
        for p in conf_pngs:
            print(f"  - {p}")

if __name__ == "__main__":
    main()
