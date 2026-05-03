"""
Stage 1: Pre-TS deterministic evaluation (NO MC)

Outputs (spec-compliant):
01_predictions/{val,test}/
  val_logits.npy, val_probs.npy, val_targets.npy, val_valid_mask.npy, val_ids.jsonl
  test_logits.npy, test_probs.npy, test_targets.npy, test_valid_mask.npy, test_ids.jsonl

02_metrics_preTS/
  auc_per_label.json
  auc_micro_macro.json
  auprc_per_label.json
  auprc_micro_macro.json
  auc_bootstrap_ci.json              (TEST, n_boot=1000)
  ece_overall.json
  reliability_bins_overall.csv
  ece_per_label.json
  reliability_bins_per_label.csv
  brier_overall.json
  nll_overall.json
  brier_per_label.json
  nll_per_label.json
  threshold_grid.json
  threshold_sweep_micro.csv
  threshold_sweep_per_label.csv
  clinical_operating_points.json     (chosen on VAL only)
  confusion_matrix_spec95sens.json   (TEST at VAL-chosen thresholds)
  confusion_matrix_spec90sens.json
  confusion_matrix_youdenj.json
  confusion_matrix_f1max.json

Rules enforced:
- VAL-only selection for operating points
- TEST evaluation only
- For inference stages prints once:
    torch.cuda.is_available()
    next(model.parameters()).device
- tqdm progress bars for VAL/TEST + bootstrap
"""

import argparse
import json
import math
import platform
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from src.models.densenet121 import build_densenet121
from src.data.dataloader_cloud import create_dataloader, LABEL_COLUMNS


# -----------------------
# JSON safety
# -----------------------
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


def write_jsonl(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=json_safe) + "\n")


def save_csv(path: Path, header: List[str], rows: List[List]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")


# -----------------------
# Dirs
# -----------------------
def ensure_stage1_dirs(outdir: Path):
    for p in [
        outdir / "00_meta",
        outdir / "01_predictions" / "val",
        outdir / "01_predictions" / "test",
        outdir / "02_metrics_preTS",
        outdir / "logs",
        outdir / "plots" / "preTS",
    ]:
        p.mkdir(parents=True, exist_ok=True)


# -----------------------
# CSV helpers (ids.jsonl)
# -----------------------
def read_csv_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "split" not in df.columns:
        raise ValueError("CSV must contain 'split' column")
    if "gcs_path" not in df.columns:
        raise ValueError("CSV must contain 'gcs_path' column")
    for col in LABEL_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"CSV missing label column: {col}")
    return df


def build_ids_jsonl_rows(df_split: pd.DataFrame) -> List[dict]:
    cols_keep = ["gcs_path"]
    for c in ["subject_id", "study_id", "dicom_id", "ViewPosition"]:
        if c in df_split.columns:
            cols_keep.append(c)

    rows = []
    for i in range(len(df_split)):
        r = {k: (df_split.iloc[i][k] if k in df_split.columns else None) for k in cols_keep}
        r["index"] = int(i)
        rows.append(r)
    return rows


# -----------------------
# Model loader
# -----------------------
def build_model_and_load_checkpoint(checkpoint_path: str, dropout_p: float, device: torch.device) -> nn.Module:
    model = build_densenet121(num_classes=8, pretrained=False, dropout_p=dropout_p)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state = ckpt["model"]
        else:
            state = ckpt
    else:
        state = ckpt

    cleaned = {}
    for k, v in state.items():
        k2 = k
        if k2.startswith("module."):
            k2 = k2[len("module.") :]
        if k2.startswith("model."):
            k2 = k2[len("model.") :]
        cleaned[k2] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print("[WARN] Missing keys:", missing[:20], ("..." if len(missing) > 20 else ""))
    if unexpected:
        print("[WARN] Unexpected keys:", unexpected[:20], ("..." if len(unexpected) > 20 else ""))

    return model.to(device)


# -----------------------
# Metrics helpers
# -----------------------
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
            auc_per[labels[c]] = float("nan")
            auprc_per[labels[c]] = float("nan")
"""
Stage 1: Pre-TS deterministic evaluation (NO MC)

Outputs (spec-compliant):
01_predictions/{val,test}/
  val_logits.npy, val_probs.npy, val_targets.npy, val_valid_mask.npy, val_ids.jsonl
  test_logits.npy, test_probs.npy, test_targets.npy, test_valid_mask.npy, test_ids.jsonl

02_metrics_preTS/
  auc_per_label.json
  auc_micro_macro.json
  auprc_per_label.json
  auprc_micro_macro.json
  auc_bootstrap_ci.json              (TEST, n_boot=1000)
  ece_overall.json
  reliability_bins_overall.csv
  ece_per_label.json
  reliability_bins_per_label.csv
  brier_overall.json
  nll_overall.json
  brier_per_label.json
  nll_per_label.json
  threshold_grid.json
  threshold_sweep_micro.csv
  threshold_sweep_per_label.csv
  clinical_operating_points.json     (chosen on VAL only)
  confusion_matrix_spec95sens.json   (TEST at VAL-chosen thresholds)
  confusion_matrix_spec90sens.json
  confusion_matrix_youdenj.json
  confusion_matrix_f1max.json

Rules enforced:
- VAL-only selection for operating points
- TEST evaluation only
- For inference stages prints once:
    torch.cuda.is_available()
    next(model.parameters()).device
- tqdm progress bars for VAL/TEST + bootstrap
"""

import argparse
import json
import math
import platform
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from src.models.densenet121 import build_densenet121
from src.data.dataloader_cloud import create_dataloader, LABEL_COLUMNS


# -----------------------
# JSON safety
# -----------------------
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


def write_jsonl(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=json_safe) + "\n")


def save_csv(path: Path, header: List[str], rows: List[List]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")


# -----------------------
# Dirs
# -----------------------
def ensure_stage1_dirs(outdir: Path):
    for p in [
        outdir / "00_meta",
        outdir / "01_predictions" / "val",
        outdir / "01_predictions" / "test",
        outdir / "02_metrics_preTS",
        outdir / "logs",
        outdir / "plots" / "preTS",
    ]:
        p.mkdir(parents=True, exist_ok=True)


# -----------------------
# CSV helpers (ids.jsonl)
# -----------------------
def read_csv_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "split" not in df.columns:
        raise ValueError("CSV must contain 'split' column")
    if "gcs_path" not in df.columns:
        raise ValueError("CSV must contain 'gcs_path' column")
    for col in LABEL_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"CSV missing label column: {col}")
    return df


def build_ids_jsonl_rows(df_split: pd.DataFrame) -> List[dict]:
    cols_keep = ["gcs_path"]
    for c in ["subject_id", "study_id", "dicom_id", "ViewPosition"]:
        if c in df_split.columns:
            cols_keep.append(c)

    rows = []
    for i in range(len(df_split)):
        r = {k: (df_split.iloc[i][k] if k in df_split.columns else None) for k in cols_keep}
        r["index"] = int(i)
        rows.append(r)
    return rows


# -----------------------
# Model loader
# -----------------------
def build_model_and_load_checkpoint(checkpoint_path: str, dropout_p: float, device: torch.device) -> nn.Module:
    model = build_densenet121(num_classes=8, pretrained=False, dropout_p=dropout_p)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state = ckpt["model"]
        else:
            state = ckpt
    else:
        state = ckpt

    cleaned = {}
    for k, v in state.items():
        k2 = k
        if k2.startswith("module."):
            k2 = k2[len("module.") :]
        if k2.startswith("model."):
            k2 = k2[len("model.") :]
        cleaned[k2] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print("[WARN] Missing keys:", missing[:20], ("..." if len(missing) > 20 else ""))
    if unexpected:
        print("[WARN] Unexpected keys:", unexpected[:20], ("..." if len(unexpected) > 20 else ""))

    return model.to(device)


# -----------------------
# Metrics helpers
# -----------------------
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
            auc_per[labels[c]] = float("nan")
            auprc_per[labels[c]] = float("nan")
        else:
            auc_per[labels[c]] = float(roc_auc_score(yt, yp))
            auprc_per[labels[c]] = float(average_precision_score(yt, yp))

    yt_micro, yp_micro = masked_flatten(y_true, y_prob, valid_mask)
    if yt_micro.size < 2 or len(np.unique(yt_micro)) < 2:
        auc_micro = float("nan")
        ap_micro = float("nan")
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

    def aucs_for_indices(idx):
        yt = y_true[idx]
        yp = y_prob[idx]
        vm = valid_mask[idx]
        out = compute_auc_auprc(yt, yp, vm, labels)
        return out["auc_micro_macro"]["micro"], out["auc_micro_macro"]["macro"], out["auc_per_label"]

    micro_list, macro_list = [], []
    per_label_lists = {lab: [] for lab in labels}

    for _ in tqdm(range(n_boot), desc="Bootstrap CI (TEST)", dynamic_ncols=True):
        idx = rng.integers(0, N, size=N)
        micro, macro, per = aucs_for_indices(idx)
        micro_list.append(micro)
        macro_list.append(macro)
        for lab in labels:
            per_label_lists[lab].append(per[lab])

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
        bal = 0.5 * (sens + spec)
        mcc_den = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + eps)
        mcc = ((TP * TN) - (FP * FN)) / mcc_den
        ppr = (TP + FP) / (TP + FP + TN + FN + eps)
        return TP, FP, TN, FN, sens, spec, ppv, npv, f1, bal, mcc, ppr

    for thr in grid:
        yp_bin = (yp_micro >= thr).astype(int)
        TP, FP, TN, FN, sens, spec, ppv, npv, f1, bal, mcc, ppr = metrics_counts(yt_micro.astype(int), yp_bin)
        rows_micro.append([thr, TP, FP, TN, FN, sens, spec, ppv, npv, f1, bal, mcc, ppr])

    rows_per = []
    for c, lab in enumerate(labels):
        m = valid_mask[:, c].astype(bool)
        yt = y_true[m, c].astype(int)
        yp = y_prob[m, c]
        for thr in grid:
            yp_bin = (yp >= thr).astype(int)
            TP, FP, TN, FN, sens, spec, ppv, npv, f1, bal, mcc, ppr = metrics_counts(yt, yp_bin)
            rows_per.append([lab, thr, TP, FP, TN, FN, sens, spec, ppv, npv, f1, bal, mcc, ppr])

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
            TP = np.sum((yb == 1) & (yt == 1))
            FP = np.sum((yb == 1) & (yt == 0))
            TN = np.sum((yb == 0) & (yt == 0))
            FN = np.sum((yb == 0) & (yt == 1))

            sens = TP / (TP + FN + eps)
            spec = TN / (TN + FP + eps)
            ppv = TP / (TP + FP + eps)
            f1 = 2 * ppv * sens / (ppv + sens + eps)
            yj = sens + spec - 1.0

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


# -----------------------
# Prediction (deterministic)
# -----------------------
@torch.no_grad()
def predict_logits_probs(model, loader, device, desc: str):
    model.eval()
    all_logits, all_probs, all_tgts, all_mask = [], [], [], []
    times = []

    pbar = tqdm(loader, desc=desc, dynamic_ncols=True)
    for images, raw_labels in pbar:
        t0 = time.time()

        valid_mask = (raw_labels != -1).float()
        targets = (raw_labels == 1).float()

        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.sigmoid(logits).detach().cpu().numpy()

        all_logits.append(logits.detach().cpu().numpy())
        all_probs.append(probs)
        all_tgts.append(targets.detach().cpu().numpy())
        all_mask.append(valid_mask.detach().cpu().numpy())

        dt = time.time() - t0
        times.append(dt)
        if len(times) % 20 == 0:
            pbar.set_postfix({"sec/batch": f"{np.mean(times[-20:]):.3f}"})

    logits_np = np.concatenate(all_logits, axis=0)
    probs_np = np.concatenate(all_probs, axis=0)
    tgts_np = np.concatenate(all_tgts, axis=0)
    mask_np = np.concatenate(all_mask, axis=0)

    stats = {
        "n_samples": int(tgts_np.shape[0]),
        "mean_sec_per_batch": float(np.mean(times)) if times else 0.0,
        "std_sec_per_batch": float(np.std(times)) if times else 0.0,
        "batch_size_effective": int(getattr(loader, "batch_size", -1)),
    }
    return logits_np, probs_np, tgts_np, mask_np, stats


# -----------------------
# Args / main
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", required=True)
    p.add_argument("--cache_dir", required=True)

    # Cache is ON by default (to avoid silent cache=OFF)
    p.add_argument("--use_cache", action="store_true", default=True)
    p.add_argument("--no_cache", action="store_true")

    p.add_argument("--csv_path", required=True)
    p.add_argument("--checkpoint_path", required=True)

    p.add_argument("--batch_size", type=int, default=60)
    p.add_argument("--num_workers", type=int, default=10)
    p.add_argument("--image_size", type=int, default=512)

    # deterministic eval doesn't need dropout, but keep consistent with your model builder
    p.add_argument("--dropout_p", type=float, default=0.3)

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.no_cache:
        args.use_cache = False

    outdir = Path(args.outdir)
    ensure_stage1_dirs(outdir)

    # ensure cache dir exists (ephemeral ok)
    if args.use_cache:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    labels = list(LABEL_COLUMNS)  # exact order

    # minimal meta
    meta_dir = outdir / "00_meta"
    env_lines = [
        f"python={platform.python_version()}",
        f"torch={torch.__version__}",
        f"cuda_available={torch.cuda.is_available()}",
        f"cuda_version={torch.version.cuda}",
        f"numpy={np.__version__}",
        f"pandas={pd.__version__}",
    ]
    (meta_dir / "env_versions.txt").write_text("\n".join(env_lines) + "\n")

    df = read_csv_df(args.csv_path)

    model = build_model_and_load_checkpoint(args.checkpoint_path, args.dropout_p, device)

    # critical rule prints (once, before inference)
    print("\n[VERIFY] torch.cuda.is_available():", torch.cuda.is_available())
    print("[VERIFY] next(model.parameters()).device:", next(model.parameters()).device, "\n", flush=True)

    pred_val_dir = outdir / "01_predictions" / "val"
    pred_test_dir = outdir / "01_predictions" / "test"

    needed_val = [
        pred_val_dir / "val_logits.npy",
        pred_val_dir / "val_probs.npy",
        pred_val_dir / "val_targets.npy",
        pred_val_dir / "val_valid_mask.npy",
        pred_val_dir / "val_ids.jsonl",
    ]
    needed_test = [
        pred_test_dir / "test_logits.npy",
        pred_test_dir / "test_probs.npy",
        pred_test_dir / "test_targets.npy",
        pred_test_dir / "test_valid_mask.npy",
        pred_test_dir / "test_ids.jsonl",
    ]

    if (not args.force) and all(p.exists() for p in needed_val + needed_test):
        print("[Stage1] Predictions already exist (use --force to overwrite).", flush=True)
    else:
        # VAL
        df_val = df[df["split"] == "validate"].reset_index(drop=True)
        write_jsonl(pred_val_dir / "val_ids.jsonl", build_ids_jsonl_rows(df_val))

        print(f"[VERIFY] use_cache={args.use_cache} cache_dir={args.cache_dir}", flush=True)
        loader_val = create_dataloader(
            csv_path=args.csv_path,
            split="validate",
            batch_size=args.batch_size,
            image_size=args.image_size,
            cache_dir=args.cache_dir,
            use_cache=args.use_cache,
            num_workers=args.num_workers,
        )

        print("[Stage1] Predicting VAL...", flush=True)
        val_logits, val_probs, val_targets, val_mask, val_stats = predict_logits_probs(
            model, loader_val, device, desc="Stage1 VAL"
        )

        np.save(pred_val_dir / "val_logits.npy", val_logits)
        np.save(pred_val_dir / "val_probs.npy", val_probs)
        np.save(pred_val_dir / "val_targets.npy", val_targets)
        np.save(pred_val_dir / "val_valid_mask.npy", val_mask)

        # TEST
        df_test = df[df["split"] == "test"].reset_index(drop=True)
        write_jsonl(pred_test_dir / "test_ids.jsonl", build_ids_jsonl_rows(df_test))

        print(f"[VERIFY] use_cache={args.use_cache} cache_dir={args.cache_dir}", flush=True)
        loader_test = create_dataloader(
            csv_path=args.csv_path,
            split="test",
            batch_size=args.batch_size,
            image_size=args.image_size,
            cache_dir=args.cache_dir,
            use_cache=args.use_cache,
            num_workers=args.num_workers,
        )

        print("[Stage1] Predicting TEST...", flush=True)
        test_logits, test_probs, test_targets, test_mask, test_stats = predict_logits_probs(
            model, loader_test, device, desc="Stage1 TEST"
        )

        np.save(pred_test_dir / "test_logits.npy", test_logits)
        np.save(pred_test_dir / "test_probs.npy", test_probs)
        np.save(pred_test_dir / "test_targets.npy", test_targets)
        np.save(pred_test_dir / "test_valid_mask.npy", test_mask)

        write_json(meta_dir / "inference_stats.json", {"val": val_stats, "test": test_stats, "device": str(device)})

    # -----------------------
    # Metrics (preTS) on TEST; operating points from VAL
    # -----------------------
    outm = outdir / "02_metrics_preTS"

    val_probs = np.load(pred_val_dir / "val_probs.npy")
    val_t = np.load(pred_val_dir / "val_targets.npy")
    val_m = np.load(pred_val_dir / "val_valid_mask.npy")

    test_probs = np.load(pred_test_dir / "test_probs.npy")
    test_t = np.load(pred_test_dir / "test_targets.npy")
    test_m = np.load(pred_test_dir / "test_valid_mask.npy")

    print("[Stage1] Computing discrimination metrics on TEST...", flush=True)
    test_disc = compute_auc_auprc(test_t, test_probs, test_m, labels)
    write_json(outm / "auc_per_label.json", test_disc["auc_per_label"])
    write_json(outm / "auc_micro_macro.json", test_disc["auc_micro_macro"])
    write_json(outm / "auprc_per_label.json", test_disc["auprc_per_label"])
    write_json(outm / "auprc_micro_macro.json", test_disc["auprc_micro_macro"])

    print("[Stage1] Bootstrap CI (TEST, n=1000)...", flush=True)
    ci = bootstrap_auc_ci(test_t, test_probs, test_m, labels, n_boot=1000)
    write_json(outm / "auc_bootstrap_ci.json", ci)

    print("[Stage1] Calibration metrics on TEST...", flush=True)
    ece_overall, bins_overall = ece_bins_overall(test_t, test_probs, test_m, n_bins=15)
    write_json(outm / "ece_overall.json", {"ece": ece_overall, "n_bins": 15})
    save_csv(outm / "reliability_bins_overall.csv", ["bin", "lo", "hi", "count", "acc", "conf"], bins_overall)

    # per-label ECE bins
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
            lo, hi = bins[i], bins[i + 1]
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
    save_csv(
        outm / "reliability_bins_per_label.csv",
        ["label", "bin", "lo", "hi", "count", "acc", "conf"],
        rows_per,
    )

    # Brier/NLL (overall + per-label)
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
        yp2 = np.clip(yp, 1e-7, 1 - 1e-7)
        nll_per[lab] = float(np.mean(-(yt * np.log(yp2) + (1 - yt) * np.log(1 - yp2))))
    write_json(outm / "brier_per_label.json", brier_per)
    write_json(outm / "nll_per_label.json", nll_per)

    # threshold sweeps (TEST)
    grid = np.linspace(0.05, 0.95, 50)
    write_json(outm / "threshold_grid.json", {"thresholds": grid.tolist()})

    rows_micro, rows_pl = threshold_sweep(test_t, test_probs, test_m, labels, grid)
    save_csv(
        outm / "threshold_sweep_micro.csv",
        [
            "threshold",
            "TP",
            "FP",
            "TN",
            "FN",
            "sensitivity",
            "specificity",
            "PPV",
            "NPV",
            "F1",
            "balanced_accuracy",
            "MCC",
            "predicted_positive_rate",
        ],
        rows_micro,
    )
    save_csv(
        outm / "threshold_sweep_per_label.csv",
        [
            "label",
            "threshold",
            "TP",
            "FP",
            "TN",
            "FN",
            "sensitivity",
            "specificity",
            "PPV",
            "NPV",
            "F1",
            "balanced_accuracy",
            "MCC",
            "predicted_positive_rate",
        ],
        rows_pl,
    )

    # clinical operating points: choose on VAL, apply confusion on TEST
    print("[Stage1] Selecting clinical operating points on VAL; confusion on TEST...", flush=True)
    ops = pick_operating_points_on_val(val_t, val_probs, val_m, labels)
    clinical = {
        lab: {
            "Spec@95%Sens": ops[lab]["spec95sens"],
            "Spec@90%Sens": ops[lab]["spec90sens"],
            "YoudenJ": ops[lab]["youdenj"],
            "F1max": ops[lab]["f1max"],
        }
        for lab in labels
    }
    write_json(outm / "clinical_operating_points.json", clinical)

    thr_spec95 = {lab: ops[lab]["spec95sens"] for lab in labels}
    thr_spec90 = {lab: ops[lab]["spec90sens"] for lab in labels}
    thr_yj = {lab: ops[lab]["youdenj"] for lab in labels}
    thr_f1 = {lab: ops[lab]["f1max"] for lab in labels}

    write_json(outm / "confusion_matrix_spec95sens.json", confusion_at_threshold(test_t, test_probs, test_m, thr_spec95, labels))
    write_json(outm / "confusion_matrix_spec90sens.json", confusion_at_threshold(test_t, test_probs, test_m, thr_spec90, labels))
    write_json(outm / "confusion_matrix_youdenj.json", confusion_at_threshold(test_t, test_probs, test_m, thr_yj, labels))
    write_json(outm / "confusion_matrix_f1max.json", confusion_at_threshold(test_t, test_probs, test_m, thr_f1, labels))

    print(f"\n✅ Stage 1 DONE. Outputs written to: {outdir}\n", flush=True)


if __name__ == "__main__":
    main()
