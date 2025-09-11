#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def bin_metrics(y_true, y_prob, thr=0.5):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= thr).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    eps = 1e-9
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / max(1, tp + fp + eps)
    rec = tp / max(1, tp + fn + eps)
    f1 = 2 * prec * rec / max(eps, (prec + rec))
    return {"acc":acc, "precision":prec, "recall":rec, "f1":f1, "tp":tp,"fp":fp,"fn":fn,"tn":tn}

def load_preds(preds_path):
    df = pd.read_csv(preds_path)
    # garante colunas
    need = {"label","prob"}
    if not need.issubset(set(df.columns)):
        raise ValueError(f"Arquivo de predições não contém colunas {need}.")
    return df

def sweep_thresholds(df, steps=500):
    y_true = df["label"].astype(int).to_numpy()
    y_prob = df["prob"].astype(float).to_numpy()
    thrs = np.linspace(0.0, 1.0, steps)
    rows = []
    for t in thrs:
        m = bin_metrics(y_true, y_prob, thr=t)
        rows.append({"thr": t, **m})
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["resnet18","resnet","cnn"], default="resnet18")
    ap.add_argument("--modality", choices=["raw","hough","combined"], default="raw")
    ap.add_argument("--preds", type=str, default="")
    ap.add_argument("--thr", type=float, default=None, help="Calcula métricas nesse threshold (0-1).")
    ap.add_argument("--sweep", action="store_true", help="Varre thresholds 0..1 e salva resultados.")
    ap.add_argument("--target-recall", type=float, default=None, help="Encontra o menor threshold cujo recall >= alvo (ex 0.99).")
    ap.add_argument("--outdir", type=str, default="", help="Pasta para salvar saídas. Default: results/analysis/<arch>/<modality>")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    arch_dir = "resnet" if args.arch in ("resnet","resnet18") else args.arch

    # caminho padrão para o latest
    if args.preds:
        preds_path = Path(args.preds)
    else:
        preds_path = repo_root / "results" / "predictions" / arch_dir / args.modality / "test_preds_latest.csv.gz"

    outdir = Path(args.outdir) if args.outdir else (repo_root / "results" / "analysis" / arch_dir / args.modality)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Carregando predições: {preds_path}")
    df = load_preds(preds_path)

    # threshold único
    if args.thr is not None:
        m = bin_metrics(df["label"].to_numpy(), df["prob"].to_numpy(), thr=args.thr)
        print(f"\n== Métricas em thr={args.thr:.3f} ==")
        print(f"Acc {m['acc']*100:.2f}% | Precision {m['precision']*100:.2f}% | Recall {m['recall']*100:.2f}% | F1 {m['f1']*100:.2f}%")
        print(f"tp={m['tp']} fp={m['fp']} fn={m['fn']} tn={m['tn']}")
        with open(outdir / f"metrics_thr_{args.thr:.3f}.json", "w") as f:
            json.dump({"thr":args.thr, **m}, f, indent=2)
        print(f"[OK] Salvo {outdir / f'metrics_thr_{args.thr:.3f}.json'}")

    # sweep
    if args.sweep or (args.target_recall is not None):
        swe = sweep_thresholds(df, steps=500)
        swe.to_csv(outdir / "threshold_sweep.csv", index=False)
        print(f"[OK] Sweep salvo em {outdir / 'threshold_sweep.csv'}")

        # melhor F1
        idx = swe["f1"].idxmax()
        best = swe.iloc[int(idx)]
        print(f"\n== Melhor F1 ==")
        print(f"thr={best['thr']:.3f} | F1={best['f1']*100:.2f}% | Precision={best['precision']*100:.2f}% | Recall={best['recall']*100:.2f}%")

        # threshold para recall alvo
        if args.target_recall is not None:
            cand = swe[swe["recall"] >= args.target_recall]
            if len(cand) > 0:
                tstar = cand.sort_values("thr").iloc[0]
                print(f"\n== Menor thr com Recall >= {args.target_recall:.3f} ==")
                print(f"thr={tstar['thr']:.3f} | Recall={tstar['recall']*100:.2f}% | Precision={tstar['precision']*100:.2f}% | F1={tstar['f1']*100:.2f}%")
                with open(outdir / f"threshold_for_recall_{args.target_recall:.3f}.json", "w") as f:
                    json.dump({"target_recall":args.target_recall, **tstar.to_dict()}, f, indent=2)
                print(f"[OK] Salvo {outdir / f'threshold_for_recall_{args.target_recall:.3f}.json'}")
            else:
                print(f"\n[WARN] Nenhum threshold atingiu recall >= {args.target_recall:.3f}.")

        # plot PR
        if not args.no_plots:
            plt.figure()
            plt.plot(swe["recall"], swe["precision"])
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision–Recall (a partir de preds salvas)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(outdir / "pr_curve.png", dpi=160)
            print(f"[OK] PR curve: {outdir / 'pr_curve.png'}")

if __name__ == "__main__":
    main()
