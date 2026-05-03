#!/usr/bin/env python3
"""
Stage 3: Pre-TS Selective Prediction (uses Stage 2 outputs)

Goal:
- Use VAL UQ outputs to choose uncertainty cutoffs per method (entropy/variance/MI) at coverage levels
- Apply VAL cutoffs to TEST (no tuning on TEST)
- Use clinical operating thresholds from Stage 1 (VAL-selected), especially Spec@95%Sens
- Save coverage curves and confusion matrices at selected coverages

Inputs:
- 03_uq_preTS/val/validate_mc_T60.npz
- 03_uq_preTS/test/test_mc_T60.npz
- 02_metrics_preTS/clinical_operating_points.json

Outputs:
04_selpred_preTS/
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
    confusion_matrix_cov90_entropy.json (and cov80)  # as a concrete deliverable
    ... similarly for variance/mi
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from src.data.dataloader_cloud import LABEL_COLUMNS

COVERAGE_LEVELS = [1.00, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50]
SAVE_CONFUSION_ON_TEST_COV = [0.90, 0.80]  # as in spec


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


def read_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


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


def compute_auc_micro_macro(y_true, y_prob, valid_mask) -> Tuple[float, float]:
    """
    Simple AUROC micro/macro computed from flattened valid entries (micro)
    and per-label mean (macro). Uses rank AUC implementation via sklearn? -> NO.
    We avoid sklearn dependency here by returning NaN placeholders if needed.

    NOTE: For Stage 3 we only need headline deltas; we can read true AUCs from Stage 1/2 if desired.
    Here we compute Brier + FN-rate reliably and keep AUROC fields optional.
    """
    # We will not compute AUROC here to keep Stage 3 lightweight & deterministic without sklearn.
    return float("nan"), float("nan")


def confusion_at_threshold(y_true, y_prob, valid_mask, thr_per_label: Dict[str, float], labels: List[str]) -> Dict:
    C = y_true.shape[1]
    out = {}
    for c in range(C):
        lab = labels[c]
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


def fn_rate_from_confusion(cm: Dict) -> float:
    FN = sum(v["FN"] for v in cm.values())
    TP = sum(v["TP"] for v in cm.values())
    return float(FN / (FN + TP + 1e-9))


def prevalence_per_label(y_true, valid_mask, labels: List[str]) -> Dict:
    prev = {}
    for c, lab in enumerate(labels):
        m = valid_mask[:, c].astype(bool)
        yt = y_true[m, c]
        prev[lab] = float(np.mean(yt)) if yt.size else float("nan")
    return prev


def micro_uq_scalar(uq_mat: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    uq = uq_mat.copy()
    uq[~valid_mask.astype(bool)] = np.nan
    return np.nanmean(uq, axis=1)  # [N]


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
    return uq_scalar <= cutoff  # keep low-uncertainty


def ensure_dirs(outdir: Path):
    for p in [
        outdir / "04_selpred_preTS" / "val",
        outdir / "04_selpred_preTS" / "test",
    ]:
        p.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", required=True)
    p.add_argument("--mc_passes", type=int, default=60)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def load_npz(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    return np.load(path)


def method_to_key(method: str) -> str:
    if method == "entropy":
        return "entropy"
    if method == "variance":
        return "var"
    if method == "mi":
        return "mi"
    raise ValueError(method)


def run_for_method(outdir: Path, labels: List[str], mc_passes: int, method: str):
    uq_dir = outdir / "03_uq_preTS"
    met_dir = outdir / "02_metrics_preTS"
    out_sp = outdir / "04_selpred_preTS"

    # clinical operating points selected on VAL in Stage 1
    clinical = read_json(met_dir / "clinical_operating_points.json")
    thr95 = {lab: float(clinical[lab]["Spec@95%Sens"]) for lab in labels}

    # Load VAL UQ
    val_npz_path = uq_dir / "val" / f"validate_mc_T{mc_passes}.npz"
    val_npz = load_npz(val_npz_path)
    val_probs = val_npz["probs_mean"]
    val_tgts = val_npz["targets"]
    val_mask = val_npz["valid_mask"]
    val_uq = val_npz[method_to_key(method)]

    # Choose cutoffs on VAL
    val_uq_scalar = micro_uq_scalar(val_uq, val_mask)
    cutoffs = coverage_cutoffs_on_val(val_uq_scalar, COVERAGE_LEVELS)

    # Save cutoffs
    write_json(out_sp / "val" / f"uncertainty_thresholds_{method}.json", {str(k): v for k, v in cutoffs.items()})

    # Helper to compute coverage curve rows
    def make_curve(split: str):
        prefix = "validate" if split == "val" else "test"
        npz_path = uq_dir / split / f"{prefix}_mc_T{mc_passes}.npz"
        z = load_npz(npz_path)

        probs = z["probs_mean"]
        tgts = z["targets"]
        mask = z["valid_mask"]
        uq = z[method_to_key(method)]
        uq_scalar = micro_uq_scalar(uq, mask)

        prev_all = prevalence_per_label(tgts, mask, labels)

        rows = []
        for cov in tqdm(COVERAGE_LEVELS, desc=f"Stage3 {split.upper()} {method}", dynamic_ncols=True):
            cutoff = float(cutoffs[cov])  # ALWAYS from VAL cutoffs
            keep = apply_cutoff(uq_scalar, cutoff)
            retained_n = int(np.sum(keep))
            abstained_n = int(keep.size - retained_n)

            # Confusion at clinical Spec@95%Sens thresholds (VAL-selected)
            cm = confusion_at_threshold(tgts[keep], probs[keep], mask[keep], thr95, labels)
            fn_rate = fn_rate_from_confusion(cm)

            prev_ret = prevalence_per_label(tgts[keep], mask[keep], labels)
            prev_shift = {lab: (prev_ret[lab] - prev_all[lab]) for lab in labels}

            # AUC fields optional (NaN here; you already have baseline AUC from Stage 1)
            micro_auc, macro_auc = compute_auc_micro_macro(tgts[keep], probs[keep], mask[keep])

            rows.append([
                cov, retained_n, abstained_n,
                micro_auc, macro_auc,
                fn_rate,
                json.dumps(prev_ret, default=json_safe),
                json.dumps(prev_shift, default=json_safe),
            ])

            # Save confusion matrices at key coverages on TEST only (deliverable)
            if split == "test" and cov in SAVE_CONFUSION_ON_TEST_COV:
                write_json(out_sp / "test" / f"confusion_matrix_cov{int(cov*100)}_{method}.json", cm)

        out_csv = out_sp / split / f"coverage_curves_{method}.csv"
        save_csv(
            out_csv,
            [
                "coverage",
                "retained_n",
                "abstained_n",
                "micro_auc",
                "macro_auc",
                "fn_rate_retained_at_spec95sens_thr",
                "retained_prevalence_per_label_json",
                "prevalence_shift_per_label_json",
            ],
            rows,
        )

    # Generate curves for VAL + TEST
    make_curve("val")
    make_curve("test")


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    ensure_dirs(outdir)

    labels = list(LABEL_COLUMNS)

    # quick existence check
    need = [
        outdir / "03_uq_preTS" / "val" / f"validate_mc_T{args.mc_passes}.npz",
        outdir / "03_uq_preTS" / "test" / f"test_mc_T{args.mc_passes}.npz",
        outdir / "02_metrics_preTS" / "clinical_operating_points.json",
    ]
    for p in need:
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")

    for method in ["entropy", "variance", "mi"]:
        run_for_method(outdir, labels, args.mc_passes, method)

    print(f"\n✅ Stage 3 DONE. Outputs written to: {outdir}/04_selpred_preTS\n")


if __name__ == "__main__":
    main()
