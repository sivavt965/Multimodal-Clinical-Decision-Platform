#!/usr/bin/env python3
"""
Stage 5: Post-TS deterministic metrics (spec compliant)

Reads:
- 05_temp_scaling/val_probs_postTS.npy
- 05_temp_scaling/test_probs_postTS.npy
- 01_predictions/val/val_targets.npy
- 01_predictions/val/val_valid_mask.npy
- 01_predictions/test/test_targets.npy
- 01_predictions/test/test_valid_mask.npy

Writes (mirrors Stage 1 naming):
06_metrics_postTS/
- auc_per_label.json
- auc_micro_macro.json
- auprc_per_label.json
- auprc_micro_macro.json
- auc_bootstrap_ci.json
- ece_overall.json
- reliability_bins_overall.csv
- ece_per_label.json
- reliability_bins_per_label.csv
- brier_overall.json
- nll_overall.json
- brier_per_label.json
- nll_per_label.json
- threshold_grid.json
- threshold_sweep_micro.csv
- threshold_sweep_per_label.csv
- clinical_operating_points.json         (selected on VAL postTS ONLY)
- confusion_matrix_spec95sens.json       (applied on TEST)
- confusion_matrix_spec90sens.json
- confusion_matrix_youdenj.json
- confusion_matrix_f1max.json
"""

import argparse, json, math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

from src.data.dataloader_cloud import LABEL_COLUMNS


def json_safe(obj):
    import numpy as _np
    if isinstance(obj, _np.integer):
        return int(obj)
    if isinstance(obj, _np.floating):
        return float(obj)
    if isinstance(obj, _np.bool_):
        return bool(obj)
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    return obj


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=json_safe)


def save_csv(path: Path, header: List[str], rows: List[List]):
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


def per_label_valid(y_true, y_prob, valid_mask, c):
    m = valid_mask[:, c].astype(bool)
    return y_true[m, c], y_prob[m, c]


def compute_auc_auprc(y_true, y_prob, valid_mask, labels: List[str]) -> Dict:
    C = y_true.shape[1]
    auc_per, auprc_per = {}, {}

    for c in range(C):
        yt, yp = per_label_valid(y_true, y_prob, valid_mask, c)
        if yt.size < 2 or len(np.unique(yt)) < 2:
            auc, ap = float("nan"), float("nan")
        else:
            auc = float(roc_auc_score(yt, yp))
            ap = float(average_precision_score(yt, yp))
        auc_per[labels[c]] = auc
        auprc_per[labels[c]] = ap

    yt_micro, yp_micro = masked_flatten(y_true, y_prob, valid_mask)
    if yt_micro.size < 2 or len(np.unique(yt_micro)) < 2:
        auc_micro, ap_micro = float("nan"), float("nan")
    else:
        auc_micro = float(roc_auc_score(yt_micro, yp_micro))
        ap_micro = float(average_precision_score(yt_micro, yp_micro))

    auc_vals = [v for v in auc_per.values() if not (isinstance(v, float) and math.isnan(v))]
    ap_vals = [v for v in auprc_per.values() if not (isinstance(v, float) and math.isnan(v))]
    auc_macro = float(np.mean(auc_vals)) if auc_vals else float("nan")
    ap_macro = float(np.mean(ap_vals)) if ap_vals else float("nan")

    return {
        "auc_per_label": auc_per,
        "auprc_per_label": auprc_per,
        "auc_micro_macro": {"micro": auc_micro, "macro": auc_macro},
        "auprc_micro_macro": {"micro": ap_micro, "macro": ap_macro},
    }


def brier_score(y_true, y_prob, valid_mask):
    yt, yp = masked_flatten(y_true, y_prob, valid_mask)
    return float(np.mean((yp - yt) ** 2)) if yt.size else float("nan")


def nll_score(y_true, y_prob, valid_mask, eps=1e-7):
    yt, yp = masked_flatten(y_true, y_prob, valid_mask)
    if yt.size == 0:
        return float("nan")
    yp = np.clip(yp, eps, 1 - eps)
    return float(np.mean(-(yt * np.log(yp) + (1 - yt) * np.log(1 - yp))))


def ece_bins_overall(y_true, y_prob, valid_mask, n_bins=15):
    yt, yp = masked_flatten(y_true, y_prob, valid_mask)
    if yt.size == 0:
        return float("nan"), []
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    rows = []
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (yp >= lo) & (yp < hi) if i < n_bins - 1 else (yp >= lo) & (yp <= hi)
        if not np.any(m):
            rows.append([i, lo, hi, 0, "", ""])
            continue
        conf = float(np.mean(yp[m]))
        acc = float(np.mean(yt[m]))
        cnt = int(np.sum(m))
        ece += (cnt / yt.size) * abs(acc - conf)
        rows.append([i, lo, hi, cnt, acc, conf])
    return float(ece), rows


def bootstrap_auc_ci(y_true, y_prob, valid_mask, labels: List[str], n_boot=1000, seed=1337):
    rng = np.random.default_rng(seed)
    N = y_true.shape[0]

    micro_list, macro_list = [], []
    per_label_lists = {lab: [] for lab in labels}

    for _ in tqdm(range(n_boot), desc="Bootstrap AUC (TEST)", dynamic_ncols=True):
        idx = rng.integers(0, N, size=N)
        out = compute_auc_auprc(y_true[idx], y_prob[idx], valid_mask[idx], labels)
        micro_list.append(out["auc_micro_macro"]["micro"])
        macro_list.append(out["auc_micro_macro"]["macro"])
        for lab in labels:
            per_label_lists[lab].append(out["auc_per_label"][lab])

    def ci(arr):
        arr = np.array(arr, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return {"mean": float("nan"), "lower": float("nan"), "upper": float("nan")}
        return {
            "mean": float(np.mean(arr)),
            "lower": float(np.percentile(arr, 2.5)),
            "upper": float(np.percentile(arr, 97.5)),
        }

    return {
        "micro": ci(micro_list),
        "macro": ci(macro_list),
        "per_label": {lab: ci(per_label_lists[lab]) for lab in labels},
        "n_bootstrap": int(n_boot),
        "ci": "95%",
    }


def threshold_sweep(y_true, y_prob, valid_mask, labels: List[str], grid: np.ndarray):
    rows_micro = []
    yt_micro, yp_micro = masked_flatten(y_true, y_prob, valid_mask)

    def metrics_counts(yt, yp_bin):
        TP = int(np.sum((yp_bin == 1) & (yt == 1)))
        FP = int(np.sum((yp_bin == 1) & (yt == 0)))
        TN = int(np.sum((yp_bin == 0) & (yt == 0)))
        FN = int(np.sum((yp_bin == 0) & (yt == 1)))
        eps = 1e-9
        sens = TP / (TP + FN + eps)
        spec = TN / (TN + FP + eps)
        ppv = TP / (TP + FP + eps)
        npv = TN / (TN + FN + eps)
        f1 = 2 * ppv * sens / (ppv + sens + eps)
        bal_acc = 0.5 * (sens + spec)
        mcc_den = math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN) + eps)
        mcc = ((TP*TN) - (FP*FN)) / mcc_den
        ppr = (TP + FP) / (TP + FP + TN + FN + eps)
        return TP, FP, TN, FN, sens, spec, ppv, npv, f1, bal_acc, mcc, ppr

    for thr in grid:
        yp_bin = (yp_micro >= thr).astype(int)
        TP,FP,TN,FN,sens,spec,ppv,npv,f1,bal,mcc,ppr = metrics_counts(yt_micro.astype(int), yp_bin)
        rows_micro.append([thr, TP,FP,TN,FN, sens,spec,ppv,npv, f1, bal, mcc, ppr])

    rows_per = []
    for c, lab in enumerate(labels):
        m = valid_mask[:, c].astype(bool)
        yt = y_true[m, c].astype(int)
        yp = y_prob[m, c]
        for thr in grid:
            yp_bin = (yp >= thr).astype(int)
            TP,FP,TN,FN,sens,spec,ppv,npv,f1,bal,mcc,ppr = metrics_counts(yt, yp_bin)
            rows_per.append([lab, thr, TP,FP,TN,FN, sens,spec,ppv,npv, f1, bal, mcc, ppr])

    return rows_micro, rows_per


def pick_operating_points_on_val(y_true_val, y_prob_val, valid_mask_val, labels: List[str]):
    grid = np.linspace(0.01, 0.99, 200)
    out = {lab: {} for lab in labels}
    eps = 1e-9

    for c, lab in enumerate(labels):
        m = valid_mask_val[:, c].astype(bool)
        yt = y_true_val[m, c].astype(int)
        yp = y_prob_val[m, c]
        if yt.size < 2:
            for k in ["spec95sens", "spec90sens", "youdenj", "f1max"]:
                out[lab][k] = 0.5
            continue

        best_spec95 = (None, -1)
        best_spec90 = (None, -1)
        best_yj = (None, -1e9)
        best_f1 = (None, -1)

        for thr in grid:
            yb = (yp >= thr).astype(int)
            TP = np.sum((yb==1)&(yt==1))
            FP = np.sum((yb==1)&(yt==0))
            TN = np.sum((yb==0)&(yt==0))
            FN = np.sum((yb==0)&(yt==1))
            sens = TP / (TP + FN + eps)
            spec = TN / (TN + FP + eps)
            ppv  = TP / (TP + FP + eps)
            f1   = 2*ppv*sens / (ppv+sens+eps)
            yj   = sens + spec - 1.0

            if sens >= 0.95 and spec > best_spec95[1]:
                best_spec95 = (thr, spec)
            if sens >= 0.90 and spec > best_spec90[1]:
                best_spec90 = (thr, spec)
            if yj > best_yj[1]:
                best_yj = (thr, yj)
            if f1 > best_f1[1]:
                best_f1 = (thr, f1)

        out[lab]["spec95sens"] = float(best_spec95[0] if best_spec95[0] is not None else 0.5)
        out[lab]["spec90sens"] = float(best_spec90[0] if best_spec90[0] is not None else 0.5)
        out[lab]["youdenj"] = float(best_yj[0] if best_yj[0] is not None else 0.5)
        out[lab]["f1max"] = float(best_f1[0] if best_f1[0] is not None else 0.5)

    return out


def confusion_at_threshold(y_true, y_prob, valid_mask, thr_per_label: Dict[str, float], labels: List[str]) -> Dict:
    out = {}
    for c, lab in enumerate(labels):
        thr = float(thr_per_label[lab])
        m = valid_mask[:, c].astype(bool)
        yt = y_true[m, c].astype(int)
        yp = (y_prob[m, c] >= thr).astype(int)
        TP = int(np.sum((yp == 1) & (yt == 1)))
        FP = int(np.sum((yp == 1) & (yt == 0)))
        TN = int(np.sum((yp == 0) & (yt == 0)))
        FN = int(np.sum((yp == 0) & (yt == 1)))
        out[lab] = {"threshold": thr, "TP": TP, "FP": FP, "TN": TN, "FN": FN}
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", required=True)
    p.add_argument("--force", action="store_true")
    p.add_argument("--bootstrap", type=int, default=1000)
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outm = outdir / "06_metrics_postTS"
    outm.mkdir(parents=True, exist_ok=True)

    labels = list(LABEL_COLUMNS)

    # Load postTS probs
    val_probs = np.load(outdir / "05_temp_scaling/val_probs_postTS.npy")
    test_probs = np.load(outdir / "05_temp_scaling/test_probs_postTS.npy")

    # Load targets/masks (from Stage 1 deterministic)
    val_t = np.load(outdir / "01_predictions/val/val_targets.npy")
    val_m = np.load(outdir / "01_predictions/val/val_valid_mask.npy")
    test_t = np.load(outdir / "01_predictions/test/test_targets.npy")
    test_m = np.load(outdir / "01_predictions/test/test_valid_mask.npy")

    # Discrimination on TEST
    test_disc = compute_auc_auprc(test_t, test_probs, test_m, labels)
    write_json(outm / "auc_per_label.json", test_disc["auc_per_label"])
    write_json(outm / "auc_micro_macro.json", test_disc["auc_micro_macro"])
    write_json(outm / "auprc_per_label.json", test_disc["auprc_per_label"])
    write_json(outm / "auprc_micro_macro.json", test_disc["auprc_micro_macro"])

    # Bootstrap CI on TEST
    ci = bootstrap_auc_ci(test_t, test_probs, test_m, labels, n_boot=args.bootstrap)
    write_json(outm / "auc_bootstrap_ci.json", ci)

    # Calibration on TEST (overall + per-label)
    ece_overall, bins_overall = ece_bins_overall(test_t, test_probs, test_m, n_bins=15)
    write_json(outm / "ece_overall.json", {"ece": ece_overall, "n_bins": 15})
    save_csv(outm / "reliability_bins_overall.csv",
             ["bin","lo","hi","count","acc","conf"], bins_overall)

    ece_per = {}
    rows_per = []
    bins = np.linspace(0.0, 1.0, 16)
    for c, lab in enumerate(labels):
        m = test_m[:, c].astype(bool)
        yt = test_t[m, c]
        yp = test_probs[m, c]
        if yt.size == 0:
            ece_per[lab] = float("nan")
            continue
        ece = 0.0
        for i in range(15):
            lo, hi = bins[i], bins[i+1]
            mm = (yp >= lo) & (yp < hi) if i < 14 else (yp >= lo) & (yp <= hi)
            if not np.any(mm):
                rows_per.append([lab, i, lo, hi, 0, "", ""])
                continue
            conf = float(np.mean(yp[mm]))
            acc = float(np.mean(yt[mm]))
            cnt = int(np.sum(mm))
            ece += (cnt / yt.size) * abs(acc - conf)
            rows_per.append([lab, i, lo, hi, cnt, acc, conf])
        ece_per[lab] = float(ece)

    write_json(outm / "ece_per_label.json", ece_per)
    save_csv(outm / "reliability_bins_per_label.csv",
             ["label","bin","lo","hi","count","acc","conf"], rows_per)

    # Brier/NLL on TEST (overall + per-label)
    write_json(outm / "brier_overall.json", {"brier": brier_score(test_t, test_probs, test_m)})
    write_json(outm / "nll_overall.json", {"nll": nll_score(test_t, test_probs, test_m)})

    brier_per, nll_per = {}, {}
    for c, lab in enumerate(labels):
        m = test_m[:, c].astype(bool)
        yt = test_t[m, c]
        yp = test_probs[m, c]
        if yt.size == 0:
            brier_per[lab] = float("nan")
            nll_per[lab] = float("nan")
            continue
        brier_per[lab] = float(np.mean((yp - yt) ** 2))
        yp2 = np.clip(yp, 1e-7, 1-1e-7)
        nll_per[lab] = float(np.mean(-(yt*np.log(yp2) + (1-yt)*np.log(1-yp2))))
    write_json(outm / "brier_per_label.json", brier_per)
    write_json(outm / "nll_per_label.json", nll_per)

    # Threshold sweeps on TEST
    grid = np.linspace(0.05, 0.95, 50)
    write_json(outm / "threshold_grid.json", {"thresholds": grid.tolist()})

    rows_micro, rows_per = threshold_sweep(test_t, test_probs, test_m, labels, grid)
    save_csv(outm / "threshold_sweep_micro.csv",
             ["threshold","TP","FP","TN","FN","sensitivity","specificity","PPV","NPV","F1","balanced_accuracy","MCC","predicted_positive_rate"],
             rows_micro)
    save_csv(outm / "threshold_sweep_per_label.csv",
             ["label","threshold","TP","FP","TN","FN","sensitivity","specificity","PPV","NPV","F1","balanced_accuracy","MCC","predicted_positive_rate"],
             rows_per)

    # Clinical operating points: SELECT on VAL postTS, APPLY on TEST
    ops = pick_operating_points_on_val(val_t, val_probs, val_m, labels)
    clinical = {lab: {
        "Spec@95%Sens": ops[lab]["spec95sens"],
        "Spec@90%Sens": ops[lab]["spec90sens"],
        "YoudenJ": ops[lab]["youdenj"],
        "F1max": ops[lab]["f1max"],
    } for lab in labels}
    write_json(outm / "clinical_operating_points.json", clinical)

    thr_spec95 = {lab: ops[lab]["spec95sens"] for lab in labels}
    thr_spec90 = {lab: ops[lab]["spec90sens"] for lab in labels}
    thr_yj = {lab: ops[lab]["youdenj"] for lab in labels}
    thr_f1 = {lab: ops[lab]["f1max"] for lab in labels}

    write_json(outm / "confusion_matrix_spec95sens.json", confusion_at_threshold(test_t, test_probs, test_m, thr_spec95, labels))
    write_json(outm / "confusion_matrix_spec90sens.json", confusion_at_threshold(test_t, test_probs, test_m, thr_spec90, labels))
    write_json(outm / "confusion_matrix_youdenj.json", confusion_at_threshold(test_t, test_probs, test_m, thr_yj, labels))
    write_json(outm / "confusion_matrix_f1max.json", confusion_at_threshold(test_t, test_probs, test_m, thr_f1, labels))

    print(f"\n✅ Stage 5 DONE. Outputs written to: {outm}\n")


if __name__ == "__main__":
    main()
