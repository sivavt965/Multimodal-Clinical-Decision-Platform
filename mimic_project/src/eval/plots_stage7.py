#!/usr/bin/env python3
"""
Stage 7: Plots only (NO inference)

Reads saved CSV/JSON/NPY outputs from earlier stages and writes plots:
plots/preTS/*
plots/postTS/*

Designed to be robust: skips a plot if required input file is missing.
"""

import argparse, json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

METHODS = ["entropy", "variance", "mi"]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_csv_rows(path: Path):
    rows = []
    with open(path, "r") as f:
        header = next(f).strip().split(",")
        for line in f:
            if not line.strip():
                continue
            rows.append(line.strip().split(","))
    return header, rows

def save_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")

def masked_flatten(y_true, y_prob, valid_mask):
    m = valid_mask.astype(bool).reshape(-1)
    yt = y_true.reshape(-1)[m]
    yp = y_prob.reshape(-1)[m]
    return yt, yp

def plot_roc_pr_micro(plot_dir: Path, y, p, m):
    # ROC micro (threshold sweep approximation)
    yt, yp = masked_flatten(y, p, m)
    thrs = np.linspace(0.0, 1.0, 200)

    roc_rows = []
    for t in thrs:
        yb = (yp >= t).astype(int)
        TP = np.sum((yb==1)&(yt==1))
        FP = np.sum((yb==1)&(yt==0))
        TN = np.sum((yb==0)&(yt==0))
        FN = np.sum((yb==0)&(yt==1))
        tpr = TP/(TP+FN+1e-9)
        fpr = FP/(FP+TN+1e-9)
        roc_rows.append([t, fpr, tpr])

    save_csv(plot_dir/"roc_curve_micro.csv", ["threshold","fpr","tpr"], roc_rows)
    plt.figure()
    plt.plot([r[1] for r in roc_rows], [r[2] for r in roc_rows])
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC (Micro)")
    plt.savefig(plot_dir/"roc_curve_micro.png", dpi=200, bbox_inches="tight")
    plt.close()

    # PR micro
    pr_rows = []
    for t in thrs:
        yb = (yp >= t).astype(int)
        TP = np.sum((yb==1)&(yt==1))
        FP = np.sum((yb==1)&(yt==0))
        FN = np.sum((yb==0)&(yt==1))
        prec = TP/(TP+FP+1e-9)
        rec  = TP/(TP+FN+1e-9)
        pr_rows.append([t, prec, rec])

    save_csv(plot_dir/"pr_curve_micro.csv", ["threshold","precision","recall"], pr_rows)
    plt.figure()
    plt.plot([r[2] for r in pr_rows], [r[1] for r in pr_rows])
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR (Micro)")
    plt.savefig(plot_dir/"pr_curve_micro.png", dpi=200, bbox_inches="tight")
    plt.close()

def plot_roc_pr_per_label(plot_dir: Path, labels, y, p, m):
    thrs = np.linspace(0.0, 1.0, 200)
    roc_per = []
    pr_per = []

    for c, lab in enumerate(labels):
        mv = m[:, c].astype(bool)
        yt = y[mv, c]
        yp = p[mv, c]
        if yt.size == 0:
            continue

        for t in thrs:
            yb = (yp >= t).astype(int)
            TP = np.sum((yb==1)&(yt==1))
            FP = np.sum((yb==1)&(yt==0))
            TN = np.sum((yb==0)&(yt==0))
            FN = np.sum((yb==0)&(yt==1))
            tpr = TP/(TP+FN+1e-9)
            fpr = FP/(FP+TN+1e-9)
            roc_per.append([lab, t, fpr, tpr])

        for t in thrs:
            yb = (yp >= t).astype(int)
            TP = np.sum((yb==1)&(yt==1))
            FP = np.sum((yb==1)&(yt==0))
            FN = np.sum((yb==0)&(yt==1))
            prec = TP/(TP+FP+1e-9)
            rec  = TP/(TP+FN+1e-9)
            pr_per.append([lab, t, prec, rec])

    save_csv(plot_dir/"roc_curve_per_label.csv", ["label","threshold","fpr","tpr"], roc_per)
    save_csv(plot_dir/"pr_curve_per_label.csv", ["label","threshold","precision","recall"], pr_per)

    plt.figure(figsize=(10,6))
    for lab in labels:
        pts = [r for r in roc_per if r[0]==lab]
        if not pts: continue
        plt.plot([p[2] for p in pts], [p[3] for p in pts], label=lab)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC (Per Label)")
    plt.legend(fontsize=7, ncol=2)
    plt.savefig(plot_dir/"roc_curve_per_label_grid.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10,6))
    for lab in labels:
        pts = [r for r in pr_per if r[0]==lab]
        if not pts: continue
        plt.plot([p[3] for p in pts], [p[2] for p in pts], label=lab)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR (Per Label)")
    plt.legend(fontsize=7, ncol=2)
    plt.savefig(plot_dir/"pr_curve_per_label_grid.png", dpi=200, bbox_inches="tight")
    plt.close()

def plot_calibration(plot_dir: Path, met_dir: Path, labels):
    # overall bins
    overall_path = met_dir/"reliability_bins_overall.csv"
    if overall_path.exists():
        header, rows = read_csv_rows(overall_path)
        pts = []
        for r in rows:
            # expected: bin,lo,hi,count,acc,conf
            if len(r) < 6: 
                continue
            cnt, acc, conf = r[3], r[4], r[5]
            if cnt == "0" or acc == "" or conf == "":
                continue
            pts.append([float(conf), float(acc)])

        save_csv(plot_dir/"calibration_overall.csv", ["conf","acc"], pts)
        plt.figure()
        if pts:
            plt.plot([p[0] for p in pts], [p[1] for p in pts], marker="o")
        plt.plot([0,1],[0,1], linestyle="--")
        plt.xlabel("Confidence"); plt.ylabel("Accuracy"); plt.title("Calibration (Overall)")
        plt.savefig(plot_dir/"calibration_overall.png", dpi=200, bbox_inches="tight")
        plt.close()

    # per-label bins
    per_path = met_dir/"reliability_bins_per_label.csv"
    if per_path.exists():
        header, rows = read_csv_rows(per_path)
        pts = []
        for r in rows:
            # expected: label,bin,lo,hi,count,acc,conf
            if len(r) < 7:
                continue
            lab, cnt, acc, conf = r[0], r[4], r[5], r[6]
            if cnt == "0" or acc == "" or conf == "":
                continue
            pts.append([lab, float(conf), float(acc)])

        save_csv(plot_dir/"calibration_per_label.csv", ["label","conf","acc"], pts)

        plt.figure(figsize=(10,6))
        for lab in labels:
            lp = [p for p in pts if p[0] == lab]
            if not lp: 
                continue
            plt.plot([p[1] for p in lp], [p[2] for p in lp], marker="o", label=lab)
        plt.plot([0,1],[0,1], linestyle="--")
        plt.xlabel("Confidence"); plt.ylabel("Accuracy"); plt.title("Calibration (Per Label)")
        plt.legend(fontsize=7, ncol=2)
        plt.savefig(plot_dir/"calibration_per_label_grid.png", dpi=200, bbox_inches="tight")
        plt.close()

def plot_threshold_sweep(plot_dir: Path, met_dir: Path):
    sweep_path = met_dir/"threshold_sweep_micro.csv"
    if not sweep_path.exists():
        return
    header, rows = read_csv_rows(sweep_path)

    th = []
    sens = []
    spec = []
    ppv = []
    npv = []
    for r in rows:
        # expected columns: threshold,TP,FP,TN,FN,sensitivity,specificity,PPV,NPV,...
        if len(r) < 9:
            continue
        th.append(float(r[0]))
        sens.append(float(r[5]))
        spec.append(float(r[6]))
        ppv.append(float(r[7]))
        npv.append(float(r[8]))

    plt.figure()
    plt.plot(th, sens, label="sens")
    plt.plot(th, spec, label="spec")
    plt.plot(th, ppv, label="ppv")
    plt.plot(th, npv, label="npv")
    plt.xlabel("Threshold"); plt.ylabel("Metric"); plt.title("Threshold sweep (micro)")
    plt.legend()
    plt.savefig(plot_dir/"threshold_sweep_metrics.png", dpi=200, bbox_inches="tight")
    plt.close()

def plot_selective_prediction(plot_dir: Path, sp_dir: Path, uq_dir: Path, mc_passes: int):
    # coverage curves from selective prediction
    for method in METHODS:
        cov_path = sp_dir/"test"/f"coverage_curves_{method}.csv"
        if not cov_path.exists():
            continue

        header, rows = read_csv_rows(cov_path)
        # expected: coverage,retained_n,abstained_n,micro_auc,macro_auc,fn_rate...
        cov = []
        micro_auc = []
        macro_auc = []
        fn_rate = []
        for r in rows:
            if len(r) < 6:
                continue
            cov.append(float(r[0]))
            micro_auc.append(float(r[3]))
            macro_auc.append(float(r[4]))
            fn_rate.append(float(r[5]))

        save_csv(plot_dir/f"coverage_curves_{method}.csv",
                 ["coverage","micro_auc","macro_auc","fn_rate"],
                 list(zip(cov, micro_auc, macro_auc, fn_rate)))

        plt.figure()
        plt.plot(cov, micro_auc, marker="o")
        plt.xlabel("Coverage"); plt.ylabel("Micro AUROC")
        plt.title(f"Coverage vs Micro AUROC ({method})")
        plt.savefig(plot_dir/f"coverage_vs_auroc_micro_{method}.png", dpi=200, bbox_inches="tight")
        plt.close()

        plt.figure()
        plt.plot(cov, macro_auc, marker="o")
        plt.xlabel("Coverage"); plt.ylabel("Macro AUROC")
        plt.title(f"Coverage vs Macro AUROC ({method})")
        plt.savefig(plot_dir/f"coverage_vs_auroc_macro_{method}.png", dpi=200, bbox_inches="tight")
        plt.close()

        plt.figure()
        plt.plot(cov, fn_rate, marker="o")
        plt.xlabel("Coverage"); plt.ylabel("FN rate (retained)")
        plt.title(f"Coverage vs FN rate ({method})")
        plt.savefig(plot_dir/f"coverage_vs_fn_{method}.png", dpi=200, bbox_inches="tight")
        plt.close()

        # coverage vs risk from uq risk_coverage csv
        rc = uq_dir/"test"/f"test_risk_coverage_T{mc_passes}_{method}.csv"
        if rc.exists():
            _, rr = read_csv_rows(rc)
            cc = []
            risk = []
            for r in rr:
                if len(r) < 4:
                    continue
                cc.append(float(r[0]))
                risk.append(float(r[3]))
            plt.figure()
            plt.plot(cc, risk, marker="o")
            plt.xlabel("Coverage"); plt.ylabel("Risk (Brier)")
            plt.title(f"Coverage vs Risk ({method})")
            plt.savefig(plot_dir/f"coverage_vs_risk_{method}.png", dpi=200, bbox_inches="tight")
            plt.close()

        # uncertainty hist (micro scalar)
        npz_path = uq_dir/"test"/f"test_mc_T{mc_passes}.npz"
        if npz_path.exists():
            d = np.load(npz_path)
            vmask = d["valid_mask"]
            uq_mat = d["entropy"] if method == "entropy" else (d["var"] if method == "variance" else d["mi"])
            uq_mat = uq_mat.copy()
            uq_mat[~vmask.astype(bool)] = np.nan
            uq_scalar = np.nanmean(uq_mat, axis=1)
            uq_scalar = uq_scalar[~np.isnan(uq_scalar)]
            plt.figure()
            plt.hist(uq_scalar, bins=40)
            plt.xlabel("Uncertainty (micro scalar)"); plt.ylabel("Count")
            plt.title(f"Uncertainty histogram ({method})")
            plt.savefig(plot_dir/f"uncertainty_hist_{method}.png", dpi=200, bbox_inches="tight")
            plt.close()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", required=True)
    p.add_argument("--mc_passes", type=int, default=60)
    return p.parse_args()

def main():
    args = parse_args()
    outdir = Path(args.outdir)

    labels = None
    # Prefer to read labels from 00_meta/label_columns.txt if exists
    lab_path = outdir/"00_meta"/"label_columns.txt"
    if lab_path.exists():
        labels = [l.strip() for l in lab_path.read_text().splitlines() if l.strip()]
    else:
        labels = list(__import__("src.data.dataloader_cloud", fromlist=["LABEL_COLUMNS"]).LABEL_COLUMNS)

    # -------- preTS --------
    pre_plot = outdir/"plots"/"preTS"
    ensure_dir(pre_plot)

    pre_probs = outdir/"01_predictions"/"test"/"test_probs.npy"
    y = outdir/"01_predictions"/"test"/"test_targets.npy"
    m = outdir/"01_predictions"/"test"/"test_valid_mask.npy"
    if pre_probs.exists() and y.exists() and m.exists():
        p = np.load(pre_probs); yt = np.load(y); vm = np.load(m)
        plot_roc_pr_micro(pre_plot, yt, p, vm)
        plot_roc_pr_per_label(pre_plot, labels, yt, p, vm)

    plot_calibration(pre_plot, outdir/"02_metrics_preTS", labels)
    plot_threshold_sweep(pre_plot, outdir/"02_metrics_preTS")
    plot_selective_prediction(pre_plot, outdir/"04_selpred_preTS", outdir/"03_uq_preTS", args.mc_passes)

    # -------- postTS --------
    post_plot = outdir/"plots"/"postTS"
    ensure_dir(post_plot)

    post_probs = outdir/"05_temp_scaling"/"test_probs_postTS.npy"
    if post_probs.exists() and y.exists() and m.exists():
        p = np.load(post_probs); yt = np.load(y); vm = np.load(m)
        plot_roc_pr_micro(post_plot, yt, p, vm)
        plot_roc_pr_per_label(post_plot, labels, yt, p, vm)

    plot_calibration(post_plot, outdir/"06_metrics_postTS", labels)
    plot_threshold_sweep(post_plot, outdir/"06_metrics_postTS")
    plot_selective_prediction(post_plot, outdir/"08_selpred_postTS", outdir/"07_uq_postTS", args.mc_passes)

    print(f"\n✅ Stage 7 DONE. Plots written to: {outdir/'plots'}\n")

if __name__ == "__main__":
    main()
