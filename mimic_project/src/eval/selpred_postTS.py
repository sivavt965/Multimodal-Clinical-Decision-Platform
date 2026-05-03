#!/usr/bin/env python3
"""
Stage 6B: Post-TS Selective Prediction (spec compliant, NO inference)

Reads:
- 07_uq_postTS/val/validate_mc_T60.npz
- 07_uq_postTS/test/test_mc_T60.npz
- 06_metrics_postTS/clinical_operating_points.json

Writes:
08_selpred_postTS/
  val/
    uncertainty_thresholds_entropy.json
    uncertainty_thresholds_variance.json
    uncertainty_thresholds_mi.json
    coverage_curves_entropy.csv
    coverage_curves_variance.csv
    coverage_curves_mi.csv
  test/
    coverage_curves_entropy.csv
    coverage_curves_variance.csv
    coverage_curves_mi.csv
    confusion_matrix_cov90_entropy.json (and cov80)
    confusion_matrix_cov90_variance.json (and cov80)
    confusion_matrix_cov90_mi.json (and cov80)

Rule:
- ALL cutoffs learned on VAL only; TEST is evaluation only.
"""

import argparse, json, math
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

from src.data.dataloader_cloud import LABEL_COLUMNS

COVERAGE_LEVELS = [1.00, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50]


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


def read_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


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


def prevalence_per_label(y_true, valid_mask, labels):
    prev = {}
    for c, lab in enumerate(labels):
        m = valid_mask[:, c].astype(bool)
        yt = y_true[m, c]
        prev[lab] = float(np.mean(yt)) if yt.size else float("nan")
    return prev


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


def micro_uq_scalar(uq_mat: np.ndarray, vmask: np.ndarray) -> np.ndarray:
    uq = uq_mat.copy()
    uq[~vmask.astype(bool)] = np.nan
    return np.nanmean(uq, axis=1)


def coverage_cutoffs_on_val(uq_scalar: np.ndarray, coverage_levels: List[float]) -> Dict[float, float]:
    uq_sorted = np.sort(uq_scalar)
    N = uq_sorted.size
    out = {}
    for cov in coverage_levels:
        k = int(math.ceil(cov * N)) - 1
        k = max(0, min(N - 1, k))
        out[cov] = float(uq_sorted[k])
    return out


def apply_cutoff(uq_scalar: np.ndarray, cutoff: float) -> np.ndarray:
    return uq_scalar <= cutoff


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", required=True)
    p.add_argument("--mc_passes", type=int, default=60)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)

    uq_dir = outdir / "07_uq_postTS"
    met_dir = outdir / "06_metrics_postTS"
    outsp = outdir / "08_selpred_postTS"

    (outsp / "val").mkdir(parents=True, exist_ok=True)
    (outsp / "test").mkdir(parents=True, exist_ok=True)

    labels = list(LABEL_COLUMNS)

    clinical = read_json(met_dir / "clinical_operating_points.json")
    thr95 = {lab: float(clinical[lab]["Spec@95%Sens"]) for lab in labels}

    # ---- Load VAL postTS UQ (choose cutoffs on VAL only)
    val_npz = np.load(uq_dir / "val" / f"validate_mc_T{args.mc_passes}.npz")
    val_probs_mean = val_npz["probs_mean"]
    val_tgts = val_npz["targets"]
    val_vmask = val_npz["valid_mask"]

    val_ent = val_npz["entropy"]
    val_var = val_npz["var"]
    val_mi = val_npz["mi"]

    cut_entropy = coverage_cutoffs_on_val(micro_uq_scalar(val_ent, val_vmask), COVERAGE_LEVELS)
    cut_variance = coverage_cutoffs_on_val(micro_uq_scalar(val_var, val_vmask), COVERAGE_LEVELS)
    cut_mi = coverage_cutoffs_on_val(micro_uq_scalar(val_mi, val_vmask), COVERAGE_LEVELS)

    write_json(outsp / "val" / "uncertainty_thresholds_entropy.json", {str(k): v for k, v in cut_entropy.items()})
    write_json(outsp / "val" / "uncertainty_thresholds_variance.json", {str(k): v for k, v in cut_variance.items()})
    write_json(outsp / "val" / "uncertainty_thresholds_mi.json", {str(k): v for k, v in cut_mi.items()})

    def make_curves(split: str, method: str, cutoffs: Dict[float, float]):
        prefix = "validate" if split == "val" else "test"
        npz = np.load(uq_dir / split / f"{prefix}_mc_T{args.mc_passes}.npz")

        probs_mean = npz["probs_mean"]
        tgts = npz["targets"]
        vmask = npz["valid_mask"]
        uq_mat = npz["entropy"] if method == "entropy" else (npz["var"] if method == "variance" else npz["mi"])

        uq_scalar = micro_uq_scalar(uq_mat, vmask)

        prev_all = prevalence_per_label(tgts, vmask, labels)

        rows = []
        for cov in COVERAGE_LEVELS:
            cutoff = float(cutoffs[cov])
            keep = apply_cutoff(uq_scalar, cutoff)

            retained_n = int(np.sum(keep))
            abstained_n = int(keep.size - retained_n)

            disc = compute_auc_auprc(tgts[keep], probs_mean[keep], vmask[keep], labels)

            cm = confusion_at_threshold(tgts[keep], probs_mean[keep], vmask[keep], thr95, labels)
            FN_sum = sum(v["FN"] for v in cm.values())
            TP_sum = sum(v["TP"] for v in cm.values())
            fn_rate = float(FN_sum / (FN_sum + TP_sum + 1e-9))

            prev_ret = prevalence_per_label(tgts[keep], vmask[keep], labels)
            prev_shift = {lab: (prev_ret[lab] - prev_all[lab]) for lab in labels}

            rows.append([
                cov, retained_n, abstained_n,
                disc["auc_micro_macro"]["micro"],
                disc["auc_micro_macro"]["macro"],
                fn_rate,
                json.dumps(prev_ret),
                json.dumps(prev_shift),
            ])

            # save key confusion matrices on TEST only
            if split == "test" and cov in [0.90, 0.80]:
                write_json(outsp / "test" / f"confusion_matrix_cov{int(cov*100)}_{method}.json", cm)

        save_csv(
            outsp / split / f"coverage_curves_{method}.csv",
            [
                "coverage","retained_n","abstained_n",
                "micro_auc","macro_auc",
                "fn_rate_retained_at_spec95sens_thr",
                "retained_prevalence_per_label_json",
                "prevalence_shift_per_label_json",
            ],
            rows
        )

    for method, cut in [("entropy", cut_entropy), ("variance", cut_variance), ("mi", cut_mi)]:
        make_curves("val", method, cut)
        make_curves("test", method, cut)

    print(f"\n✅ Stage 6B DONE. Outputs written to: {outsp}\n")


if __name__ == "__main__":
    main()

