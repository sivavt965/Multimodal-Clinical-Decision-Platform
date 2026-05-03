#!/usr/bin/env python3
"""
Phase-2 Full Evaluation (ALL IN ONE FILE) — spec-compliant

Wired to your repo:
- Model: src/models/densenet121.py (build_densenet121)
- Loader: src/data/dataloader_cloud.py (create_dataloader, LABEL_COLUMNS, splits train/validate/test)
- CSV must contain: split, gcs_path, and LABEL_COLUMNS

Key rules enforced:
- VAL-only selection: operating points, TS fit, UQ cutoffs
- TEST evaluation only

Outputs written ONLY to OUTDIR in exact folder structure.
"""

import os, io, json, time, math, platform, argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
def json_safe(obj):
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj



# --- JSON safety (numpy/pandas types) ---
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

# -----------------------
# Repo imports (your code)
# -----------------------
from src.models.densenet121 import build_densenet121
from src.data.dataloader_cloud import create_dataloader, LABEL_COLUMNS


# -----------------------
# Constants
# -----------------------
COVERAGE_LEVELS = [1.00, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50]


# -----------------------
# Args
# -----------------------
@dataclass
class Args:
    outdir: str
    cache_dir: str
    use_cache: bool
    csv_path: str
    checkpoint_path: str

    batch_size: int
    num_workers: int
    image_size: int

    dropout_p: float
    mc_passes: int

    device: str
    force: bool


# -----------------------
# IO helpers
# -----------------------
def ensure_dirs(outdir: Path):
    for p in [
        outdir/"00_meta",
        outdir/"01_predictions"/"val",
        outdir/"01_predictions"/"test",
        outdir/"02_metrics_preTS",
        outdir/"03_uq_preTS"/"val",
        outdir/"03_uq_preTS"/"test",
        outdir/"04_selpred_preTS"/"val",
        outdir/"04_selpred_preTS"/"test",
        outdir/"05_temp_scaling",
        outdir/"06_metrics_postTS",
        outdir/"07_uq_postTS"/"val",
        outdir/"07_uq_postTS"/"test",
        outdir/"08_selpred_postTS"/"val",
        outdir/"08_selpred_postTS"/"test",
        outdir/"plots"/"preTS",
        outdir/"plots"/"postTS",
        outdir/"logs",
    ]:
        p.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=json_safe)


def read_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


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
# Numerics helpers
# -----------------------
def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def masked_flatten(y_true, y_prob, valid_mask):
    m = valid_mask.astype(bool).reshape(-1)
    yt = y_true.reshape(-1)[m]
    yp = y_prob.reshape(-1)[m]
    return yt, yp


def per_label_valid(y_true, y_prob, valid_mask, c):
    m = valid_mask[:, c].astype(bool)
    return y_true[m, c], y_prob[m, c]


# -----------------------
# Metrics
# -----------------------
def compute_auc_auprc(y_true, y_prob, valid_mask, labels: List[str]) -> Dict:
    C = y_true.shape[1]
    auc_per = {}
    auprc_per = {}
    for c in range(C):
        yt, yp = per_label_valid(y_true, y_prob, valid_mask, c)
        if yt.size < 2 or len(np.unique(yt)) < 2:
            auc = float("nan")
            ap = float("nan")
        else:
            auc = float(roc_auc_score(yt, yp))
            ap = float(average_precision_score(yt, yp))
        auc_per[labels[c]] = auc
        auprc_per[labels[c]] = ap

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

    for _ in range(n_boot):
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
        return {"mean": float(np.mean(arr)),
                "lower": float(np.percentile(arr, 2.5)),
                "upper": float(np.percentile(arr, 97.5))}

    return {
        "micro": ci(micro_list),
        "macro": ci(macro_list),
        "per_label": {lab: ci(per_label_lists[lab]) for lab in labels},
        "n_bootstrap": n_boot,
        "ci": "95%"
    }


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


# -----------------------
# UQ helpers
# -----------------------
def set_dropout_train_bn_eval(model: nn.Module):
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


def uq_entropy_from_mean(p_mean, eps=1e-7):
    p = np.clip(p_mean, eps, 1 - eps)
    return -(p*np.log(p) + (1-p)*np.log(1-p))


def uq_variance(probs_passes):
    return np.var(probs_passes, axis=0)


def uq_mutual_information(probs_passes, eps=1e-7):
    mean_p = np.mean(probs_passes, axis=0)
    H_mean = uq_entropy_from_mean(mean_p, eps=eps)
    H_each = uq_entropy_from_mean(np.clip(probs_passes, eps, 1-eps), eps=eps)  # [T,N,C]
    E_H = np.mean(H_each, axis=0)
    return H_mean - E_H


def prevalence_per_label(y_true, valid_mask, labels):
    prev = {}
    for c, lab in enumerate(labels):
        m = valid_mask[:, c].astype(bool)
        yt = y_true[m, c]
        prev[lab] = float(np.mean(yt)) if yt.size else float("nan")
    return prev


# -----------------------
# CSV / split utilities
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


def split_name_for_dataset(split: str) -> str:
    # pipeline uses val/test; dataset uses validate/test
    if split == "val":
        return "validate"
    return split


def build_ids_jsonl_rows(df_split: pd.DataFrame) -> List[dict]:
    # stable order (dataset order)
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


def class_distribution_from_df(df_split: pd.DataFrame) -> Dict:
    out = {}
    # values are in {-1,0,1} (or NaN); dataloader converts NaN->-1 at fetch
    # Here we do the same conversion to be consistent.
    for lab in LABEL_COLUMNS:
        v = df_split[lab].astype("float32").to_numpy()
        v = np.nan_to_num(v, nan=-1.0)
        pos = int(np.sum(v == 1.0))
        neg = int(np.sum(v == 0.0))
        unc = int(np.sum(v == -1.0))
        denom = pos + neg
        prev = float(pos / denom) if denom > 0 else 0.0
        out[lab] = {"positive": pos, "negative": neg, "uncertain": unc, "prevalence": prev}
    return out


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
            k2 = k2[len("module."):]
        if k2.startswith("model."):
            k2 = k2[len("model."):]
        cleaned[k2] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print("[WARN] Missing keys:", missing[:20], ("..." if len(missing) > 20 else ""))
    if unexpected:
        print("[WARN] Unexpected keys:", unexpected[:20], ("..." if len(unexpected) > 20 else ""))

    return model.to(device)


# -----------------------
# Prediction (deterministic)
# -----------------------
@torch.no_grad()
def predict_logits_probs(model, loader, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    model.eval()
    all_logits, all_probs, all_tgts, all_mask = [], [], [], []
    times = []

    for images, raw_labels in loader:
        t0 = time.time()

        # raw_labels in {-1,0,1} (NaN already handled in dataset)
        valid_mask = (raw_labels != -1).float()
        targets = (raw_labels == 1).float()

        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.sigmoid(logits).detach().cpu().numpy()

        all_logits.append(logits.detach().cpu().numpy())
        all_probs.append(probs)
        all_tgts.append(targets.detach().cpu().numpy())
        all_mask.append(valid_mask.detach().cpu().numpy())

        times.append(time.time() - t0)

    logits_np = np.concatenate(all_logits, axis=0)
    probs_np  = np.concatenate(all_probs, axis=0)
    tgts_np   = np.concatenate(all_tgts, axis=0)
    mask_np   = np.concatenate(all_mask, axis=0)

    stats = {
        "n_samples": int(tgts_np.shape[0]),
        "mean_sec_per_batch": float(np.mean(times)) if times else 0.0,
        "std_sec_per_batch": float(np.std(times)) if times else 0.0,
        "batch_size_effective": int(getattr(loader, "batch_size", -1)),
    }
    return logits_np, probs_np, tgts_np, mask_np, stats


# -----------------------
# MC Dropout (logits passes -> probs passes, with optional TS)
# -----------------------
@torch.no_grad()
def mc_dropout_probs(model, loader, device, T: int, temperature: Optional[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      probs_passes: [T, N, C]
      probs_mean:   [N, C]
      targets:      [N, C]
      valid_mask:   [N, C]

    If temperature is provided, it is applied to logits of each pass before sigmoid:
        probs = sigmoid(logits / temperature)
    This yields "true postTS UQ" passes.
    """
    set_dropout_train_bn_eval(model)

    passes = []
    targets_np = None
    mask_np = None

    for _ in range(T):
        all_probs, all_tgts, all_mask = [], [], []
        for images, raw_labels in loader:
            valid_mask = (raw_labels != -1).float()
            targets = (raw_labels == 1).float()

            images = images.to(device, non_blocking=True)
            logits = model(images)

            if temperature is not None:
                logits = logits / float(temperature)

            probs = torch.sigmoid(logits).detach().cpu().numpy()

            all_probs.append(probs)
            all_tgts.append(targets.detach().cpu().numpy())
            all_mask.append(valid_mask.detach().cpu().numpy())

        probs_np = np.concatenate(all_probs, axis=0)
        tgts = np.concatenate(all_tgts, axis=0)
        msk = np.concatenate(all_mask, axis=0)

        if targets_np is None:
            targets_np = tgts
            mask_np = msk
        else:
            assert tgts.shape == targets_np.shape
            assert msk.shape == mask_np.shape

        passes.append(probs_np)

    probs_passes = np.stack(passes, axis=0)  # [T,N,C]
    probs_mean = np.mean(probs_passes, axis=0)
    return probs_passes, probs_mean, targets_np, mask_np


# -----------------------
# Steps
# -----------------------
def step00_meta(args: Args, outdir: Path, df: pd.DataFrame, labels: List[str], device: torch.device):
    meta_dir = outdir/"00_meta"

    # env_versions.txt
    env_lines = [
        f"python={platform.python_version()}",
        f"torch={torch.__version__}",
        f"cuda_available={torch.cuda.is_available()}",
        f"cuda_version={torch.version.cuda}",
        f"cudnn_version={torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None}",
        f"numpy={np.__version__}",
        f"pandas={pd.__version__}",
    ]
    with open(meta_dir/"env_versions.txt", "w") as f:
        f.write("\n".join(env_lines) + "\n")

    # hardware.txt
    hw = [f"device={args.device}"]
    if torch.cuda.is_available():
        hw.append(f"gpu_name={torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        hw.append(f"gpu_total_vram_bytes={props.total_memory}")
    hw.append(f"cpu={platform.processor()}")
    with open(meta_dir/"hardware.txt", "w") as f:
        f.write("\n".join(hw) + "\n")

    # label_columns.txt (exact order)
    with open(meta_dir/"label_columns.txt", "w") as f:
        for lab in labels:
            f.write(lab + "\n")

    # split_sizes.json
    split_sizes = {
        "train": int((df["split"] == "train").sum()),
        "val": int((df["split"] == "validate").sum()),
        "test": int((df["split"] == "test").sum()),
    }
    write_json(meta_dir/"split_sizes.json", split_sizes)

    # masking_strategy.json
    masking_strategy = {
        "label_encoding_in_csv": "labels in {-1,0,1} (NaN allowed)",
        "nan_handling": "NaN converted to -1 (invalid) at dataset load",
        "invalid_label_value": -1,
        "valid_mask_rule": "valid_mask = (label != -1)",
        "target_binarization": "targets = (label == 1)",
        "random_training_masking": "none in this Phase-2 eval pipeline (eval uses dataset-provided labels only)",
    }
    write_json(meta_dir/"masking_strategy.json", masking_strategy)

    # class_distribution.json (per split)
    class_dist = {
        "train": class_distribution_from_df(df[df["split"] == "train"]),
        "val": class_distribution_from_df(df[df["split"] == "validate"]),
        "test": class_distribution_from_df(df[df["split"] == "test"]),
    }
    write_json(meta_dir/"class_distribution.json", class_dist)

    # run_config.json
    run_cfg = {
        "dropout_p": args.dropout_p,
        "mc_passes": args.mc_passes,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "image_size": args.image_size,
        "use_cache": args.use_cache,
        "cache_dir": args.cache_dir,
        "csv_path": args.csv_path,
        "checkpoint_path": args.checkpoint_path,
        "device": args.device,
    }
    write_json(meta_dir/"run_config.json", run_cfg)

    # checkpoint_info.json
    write_json(meta_dir/"checkpoint_info.json", {
        "checkpoint_path_used": args.checkpoint_path
    })

    # copy baseline_best.pt into 00_meta/
    try:
        dst = meta_dir/"baseline_best.pt"
        if (not dst.exists()) or args.force:
            with open(args.checkpoint_path, "rb") as fsrc, open(dst, "wb") as fdst:
                fdst.write(fsrc.read())
    except Exception as e:
        with open(meta_dir/"checkpoint_copy_error.txt", "w") as f:
            f.write(str(e) + "\n")


def step01_predictions(args: Args, outdir: Path, df: pd.DataFrame, labels: List[str], device: torch.device):
    pred_val_dir = outdir/"01_predictions"/"val"
    pred_test_dir = outdir/"01_predictions"/"test"
    meta_dir = outdir/"00_meta"

    needed_val = [pred_val_dir/"val_logits.npy", pred_val_dir/"val_probs.npy", pred_val_dir/"val_targets.npy", pred_val_dir/"val_label_valid_mask.npy", pred_val_dir/"val_ids.jsonl"]
    needed_test = [pred_test_dir/"test_logits.npy", pred_test_dir/"test_probs.npy", pred_test_dir/"test_targets.npy", pred_test_dir/"test_label_valid_mask.npy", pred_test_dir/"test_ids.jsonl"]
    if (not args.force) and all(p.exists() for p in needed_val+needed_test):
        return

    model = build_model_and_load_checkpoint(args.checkpoint_path, args.dropout_p, device)

    # VAL
    df_val = df[df["split"] == "validate"].reset_index(drop=True)
    ids_val = build_ids_jsonl_rows(df_val)
    loader_val = create_dataloader(
        csv_path=args.csv_path, split="validate",
        batch_size=args.batch_size, image_size=args.image_size,
        cache_dir=args.cache_dir, use_cache=args.use_cache,
        num_workers=args.num_workers
    )
    val_logits, val_probs, val_targets, val_mask, val_stats = predict_logits_probs(model, loader_val, device)
    np.save(pred_val_dir/"val_logits.npy", val_logits)
    np.save(pred_val_dir/"val_probs.npy", val_probs)
    np.save(pred_val_dir/"val_targets.npy", val_targets)
    np.save(pred_val_dir/"val_label_valid_mask.npy", val_mask)
    write_jsonl(pred_val_dir/"val_ids.jsonl", ids_val)

    # TEST
    df_test = df[df["split"] == "test"].reset_index(drop=True)
    ids_test = build_ids_jsonl_rows(df_test)
    loader_test = create_dataloader(
        csv_path=args.csv_path, split="test",
        batch_size=args.batch_size, image_size=args.image_size,
        cache_dir=args.cache_dir, use_cache=args.use_cache,
        num_workers=args.num_workers
    )
    test_logits, test_probs, test_targets, test_mask, test_stats = predict_logits_probs(model, loader_test, device)
    np.save(pred_test_dir/"test_logits.npy", test_logits)
    np.save(pred_test_dir/"test_probs.npy", test_probs)
    np.save(pred_test_dir/"test_targets.npy", test_targets)
    np.save(pred_test_dir/"test_label_valid_mask.npy", test_mask)
    write_jsonl(pred_test_dir/"test_ids.jsonl", ids_test)

    # inference_stats.json
    write_json(meta_dir/"inference_stats.json", {
        "val": val_stats,
        "test": test_stats,
        "device": str(device),
        "pin_memory": True,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    })

    # test_size.txt (actual evaluated N)
    with open(meta_dir/"test_size.txt", "w") as f:
        f.write(f"{test_targets.shape[0]}\n")


def step02_metrics(args: Args, outdir: Path, labels: List[str], stage: str):
    if stage == "preTS":
        outm = outdir/"02_metrics_preTS"
        val_probs = np.load(outdir/"01_predictions/val/val_probs.npy")
        test_probs = np.load(outdir/"01_predictions/test/test_probs.npy")
    else:
        outm = outdir/"06_metrics_postTS"
        val_probs = np.load(outdir/"05_temp_scaling/val_probs_postTS.npy")
        test_probs = np.load(outdir/"05_temp_scaling/test_probs_postTS.npy")

    val_t = np.load(outdir/"01_predictions/val/val_targets.npy")
    val_m = np.load(outdir/"01_predictions/val/val_label_valid_mask.npy")
    test_t = np.load(outdir/"01_predictions/test/test_targets.npy")
    test_m = np.load(outdir/"01_predictions/test/test_label_valid_mask.npy")

    # discrimination on TEST
    test_disc = compute_auc_auprc(test_t, test_probs, test_m, labels)
    write_json(outm/"auc_per_label.json", test_disc["auc_per_label"])
    write_json(outm/"auc_micro_macro.json", test_disc["auc_micro_macro"])
    write_json(outm/"auprc_per_label.json", test_disc["auprc_per_label"])
    write_json(outm/"auprc_micro_macro.json", test_disc["auprc_micro_macro"])

    # bootstrap CI on TEST
    ci = bootstrap_auc_ci(test_t, test_probs, test_m, labels, n_boot=1000)
    write_json(outm/"auc_bootstrap_ci.json", ci)

    # calibration on TEST
    ece_overall, bins_overall = ece_bins_overall(test_t, test_probs, test_m, n_bins=15)
    write_json(outm/"ece_overall.json", {"ece": ece_overall, "n_bins": 15})
    save_csv(outm/"reliability_bins_overall.csv",
             ["bin","lo","hi","count","acc","conf"], bins_overall)

    # per-label ECE (compute bins per label)
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

    write_json(outm/"ece_per_label.json", ece_per)
    save_csv(outm/"reliability_bins_per_label.csv",
             ["label","bin","lo","hi","count","acc","conf"], rows_per)

    # Brier/NLL on TEST (overall + per-label)
    write_json(outm/"brier_overall.json", {"brier": brier_score(test_t, test_probs, test_m)})
    write_json(outm/"nll_overall.json", {"nll": nll_score(test_t, test_probs, test_m)})

    brier_per = {}
    nll_per = {}
    for c, lab in enumerate(labels):
        m = test_m[:, c].astype(bool)
        yt = test_t[m, c]
        yp = test_probs[m, c]
        if yt.size == 0:
            brier_per[lab] = float("nan")
            nll_per[lab] = float("nan")
            continue
        brier_per[lab] = float(np.mean((yp-yt)**2))
        yp2 = np.clip(yp, 1e-7, 1-1e-7)
        nll_per[lab] = float(np.mean(-(yt*np.log(yp2) + (1-yt)*np.log(1-yp2))))
    write_json(outm/"brier_per_label.json", brier_per)
    write_json(outm/"nll_per_label.json", nll_per)

    # threshold sweep on TEST
    grid = np.linspace(0.05, 0.95, 50)
    write_json(outm/"threshold_grid.json", {"thresholds": grid.tolist()})

    rows_micro, rows_per = threshold_sweep(test_t, test_probs, test_m, labels, grid)
    save_csv(outm/"threshold_sweep_micro.csv",
             ["threshold","TP","FP","TN","FN","sensitivity","specificity","PPV","NPV","F1","balanced_accuracy","MCC","predicted_positive_rate"],
             rows_micro)
    save_csv(outm/"threshold_sweep_per_label.csv",
             ["label","threshold","TP","FP","TN","FN","sensitivity","specificity","PPV","NPV","F1","balanced_accuracy","MCC","predicted_positive_rate"],
             rows_per)

    # clinical operating points: select on VAL, apply on TEST
    ops = pick_operating_points_on_val(val_t, val_probs, val_m, labels)
    clinical = {lab: {
        "Spec@95%Sens": ops[lab]["spec95sens"],
        "Spec@90%Sens": ops[lab]["spec90sens"],
        "YoudenJ": ops[lab]["youdenj"],
        "F1max": ops[lab]["f1max"],
    } for lab in labels}
    write_json(outm/"clinical_operating_points.json", clinical)

    thr_spec95 = {lab: ops[lab]["spec95sens"] for lab in labels}
    thr_spec90 = {lab: ops[lab]["spec90sens"] for lab in labels}
    thr_yj = {lab: ops[lab]["youdenj"] for lab in labels}
    thr_f1 = {lab: ops[lab]["f1max"] for lab in labels}

    write_json(outm/"confusion_matrix_spec95sens.json", confusion_at_threshold(test_t, test_probs, test_m, thr_spec95, labels))
    write_json(outm/"confusion_matrix_spec90sens.json", confusion_at_threshold(test_t, test_probs, test_m, thr_spec90, labels))
    write_json(outm/"confusion_matrix_youdenj.json", confusion_at_threshold(test_t, test_probs, test_m, thr_yj, labels))
    write_json(outm/"confusion_matrix_f1max.json", confusion_at_threshold(test_t, test_probs, test_m, thr_f1, labels))


def step05_temp_scaling(args: Args, outdir: Path, device: torch.device):
    ts_dir = outdir/"05_temp_scaling"
    temp_path = ts_dir/"temperature.txt"
    if (not args.force) and temp_path.exists():
        return

    val_logits = np.load(outdir/"01_predictions/val/val_logits.npy")
    val_t = np.load(outdir/"01_predictions/val/val_targets.npy")
    val_m = np.load(outdir/"01_predictions/val/val_label_valid_mask.npy")

    test_logits = np.load(outdir/"01_predictions/test/test_logits.npy")

    # optimize scalar temperature on VAL (masked NLL)
    logits_t = torch.tensor(val_logits, dtype=torch.float32, device=device)
    targets_t = torch.tensor(val_t, dtype=torch.float32, device=device)
    mask_t = torch.tensor(val_m, dtype=torch.float32, device=device)

    logT = torch.zeros((), device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([logT], lr=0.5, max_iter=200)

    def closure():
        optimizer.zero_grad()
        T = torch.exp(logT).clamp(1e-3, 100.0)
        probs = torch.sigmoid(logits_t / T).clamp(1e-7, 1-1e-7)
        nll = -(targets_t*torch.log(probs) + (1-targets_t)*torch.log(1-probs))
        nll = (nll * mask_t).sum() / (mask_t.sum() + 1e-9)
        nll.backward()
        return nll

    optimizer.step(closure)
    T_final = float(torch.exp(logT).detach().cpu().item())

    with open(temp_path, "w") as f:
        f.write(f"{T_final:.6f}\n")

    # pre vs post on VAL
    val_probs_pre = sigmoid_np(val_logits)
    val_probs_post = sigmoid_np(val_logits / T_final)
    pre = {
        "ece": ece_bins_overall(val_t, val_probs_pre, val_m, n_bins=15)[0],
        "nll": nll_score(val_t, val_probs_pre, val_m),
        "brier": brier_score(val_t, val_probs_pre, val_m),
    }
    post = {
        "ece": ece_bins_overall(val_t, val_probs_post, val_m, n_bins=15)[0],
        "nll": nll_score(val_t, val_probs_post, val_m),
        "brier": brier_score(val_t, val_probs_post, val_m),
    }
    write_json(ts_dir/"val_calibration_pre_post.json", {"temperature": T_final, "pre": pre, "post": post})

    np.save(ts_dir/"val_probs_postTS.npy", val_probs_post)
    np.save(ts_dir/"test_probs_postTS.npy", sigmoid_np(test_logits / T_final))


def step03_uq(args: Args, outdir: Path, labels: List[str], device: torch.device, stage: str):
    outuq = outdir/("03_uq_preTS" if stage=="preTS" else "07_uq_postTS")
    model = build_model_and_load_checkpoint(args.checkpoint_path, args.dropout_p, device)

    temperature = None
    if stage == "postTS":
        temperature = float(open(outdir/"05_temp_scaling/temperature.txt","r").read().strip())

    for split in ["val", "test"]:
        ds_split = "validate" if split=="val" else "test"
        split_dir = outuq/split
        prefix = "validate" if split=="val" else "test"

        npz_path = split_dir/f"{prefix}_mc_T{args.mc_passes}.npz"
        json_path = split_dir/f"{prefix}_mc_T{args.mc_passes}.json"
        if (not args.force) and npz_path.exists() and json_path.exists():
            continue

        loader = create_dataloader(
            csv_path=args.csv_path, split=ds_split,
            batch_size=args.batch_size, image_size=args.image_size,
            cache_dir=args.cache_dir, use_cache=args.use_cache,
            num_workers=args.num_workers
        )

        probs_passes, probs_mean, tgts, vmask = mc_dropout_probs(model, loader, device, args.mc_passes, temperature)

        var = uq_variance(probs_passes)
        ent = uq_entropy_from_mean(probs_mean)
        mi = uq_mutual_information(probs_passes)

        np.savez(
            npz_path,
            probs_passes=probs_passes,
            probs_mean=probs_mean,
            var=var,
            entropy=ent,
            mi=mi,
            targets=tgts,
            valid_mask=vmask
        )

        # UQ summary stats per label (explicit)
        uq_summary = {"uq_summary_per_label": {}, "uq_summary_micro_macro": {}}
        for c, lab in enumerate(labels):
            m = vmask[:, c].astype(bool)
            if not np.any(m):
                uq_summary["uq_summary_per_label"][lab] = {
                    "entropy": {"mean": np.nan, "median": np.nan, "std": np.nan, "min": np.nan, "max": np.nan},
                    "variance": {"mean": np.nan, "median": np.nan, "std": np.nan, "min": np.nan, "max": np.nan},
                    "mi": {"mean": np.nan, "median": np.nan, "std": np.nan, "min": np.nan, "max": np.nan},
                }
                continue
            e = ent[m, c]; v = var[m, c]; u = mi[m, c]
            uq_summary["uq_summary_per_label"][lab] = {
                "entropy": {"mean": float(np.mean(e)), "median": float(np.median(e)), "std": float(np.std(e)), "min": float(np.min(e)), "max": float(np.max(e))},
                "variance": {"mean": float(np.mean(v)), "median": float(np.median(v)), "std": float(np.std(v)), "min": float(np.min(v)), "max": float(np.max(v))},
                "mi": {"mean": float(np.mean(u)), "median": float(np.median(u)), "std": float(np.std(u)), "min": float(np.min(u)), "max": float(np.max(u))},
            }

        # Risk/coverage CSVs (micro scalar = nanmean across labels)
        def micro_uq_scalar(uq_mat):
            uq = uq_mat.copy()
            uq[~vmask.astype(bool)] = np.nan
            return np.nanmean(uq, axis=1)

        def risk_coverage_csv(uq_scalar, name):
            order = np.argsort(uq_scalar)
            uq_sorted = uq_scalar[order]
            probs_sorted = probs_mean[order]
            tgts_sorted = tgts[order]
            vmask_sorted = vmask[order]
            N = tgts_sorted.shape[0]

            rows = []
            for cov in COVERAGE_LEVELS:
                k = int(math.ceil(cov * N))
                kept = slice(0, k)
                risk = brier_score(tgts_sorted[kept], probs_sorted[kept], vmask_sorted[kept])
                rows.append([
                    cov,
                    k,
                    N - k,
                    risk,
                    float(uq_sorted[k-1]) if k > 0 else float(uq_sorted[0]),
                    json.dumps(prevalence_per_label(tgts_sorted[kept], vmask_sorted[kept], labels))
                ])

            save_csv(split_dir/f"{prefix}_risk_coverage_T{args.mc_passes}_{name}.csv",
                     ["coverage","retained_n","abstained_n","risk","uq_cutoff","retained_prevalence_per_label_json"],
                     rows)

        risk_coverage_csv(micro_uq_scalar(ent), "entropy")
        risk_coverage_csv(micro_uq_scalar(var), "variance")
        risk_coverage_csv(micro_uq_scalar(mi), "mi")

        disc = compute_auc_auprc(tgts, probs_mean, vmask, labels)
        summ = {
            "n_samples": int(tgts.shape[0]),
            "mc_passes": int(args.mc_passes),
            "dropout_p": float(args.dropout_p),
            "stage": stage,
            "temperature_used": float(temperature) if temperature is not None else None,
            "mean_probs_auc_micro_macro": disc["auc_micro_macro"],
            "mean_probs_auc_per_label": disc["auc_per_label"],
            "mean_probs_brier_overall": brier_score(tgts, probs_mean, vmask),
            "mean_probs_nll_overall": nll_score(tgts, probs_mean, vmask),
        }
        summ.update(uq_summary)
        write_json(json_path, summ)


def coverage_cutoffs_on_val(uq_scalar: np.ndarray, coverage_levels: List[float]) -> Dict[float, float]:
    uq_sorted = np.sort(uq_scalar)
    N = uq_sorted.size
    out = {}
    for cov in coverage_levels:
        k = int(math.ceil(cov * N)) - 1
        k = max(0, min(N-1, k))
        out[cov] = float(uq_sorted[k])
    return out


def apply_cutoff(uq_scalar: np.ndarray, cutoff: float) -> np.ndarray:
    return uq_scalar <= cutoff


def step04_selpred(args: Args, outdir: Path, labels: List[str], stage: str):
    uq_dir = outdir/("03_uq_preTS" if stage=="preTS" else "07_uq_postTS")
    outsp = outdir/("04_selpred_preTS" if stage=="preTS" else "08_selpred_postTS")
    met_dir = outdir/("02_metrics_preTS" if stage=="preTS" else "06_metrics_postTS")

    clinical = read_json(met_dir/"clinical_operating_points.json")
    thr95 = {lab: float(clinical[lab]["Spec@95%Sens"]) for lab in labels}

    # VAL UQ to choose cutoffs
    val_npz = np.load(uq_dir/"val"/f"validate_mc_T{args.mc_passes}.npz")
    val_m = val_npz["valid_mask"]
    val_ent = val_npz["entropy"]
    val_var = val_npz["var"]
    val_mi = val_npz["mi"]

    def micro_uq(uq_mat):
        uq = uq_mat.copy()
        uq[~val_m.astype(bool)] = np.nan
        return np.nanmean(uq, axis=1)

    cut_entropy = coverage_cutoffs_on_val(micro_uq(val_ent), COVERAGE_LEVELS)
    cut_variance = coverage_cutoffs_on_val(micro_uq(val_var), COVERAGE_LEVELS)
    cut_mi = coverage_cutoffs_on_val(micro_uq(val_mi), COVERAGE_LEVELS)

    write_json(outsp/"val"/"uncertainty_thresholds_entropy.json", {str(k): v for k,v in cut_entropy.items()})
    write_json(outsp/"val"/"uncertainty_thresholds_variance.json", {str(k): v for k,v in cut_variance.items()})
    write_json(outsp/"val"/"uncertainty_thresholds_mi.json", {str(k): v for k,v in cut_mi.items()})

    def make_curves(split: str, method: str, cutoffs: Dict[float,float]):
        prefix = "validate" if split=="val" else "test"
        npz = np.load(uq_dir/split/(f"{prefix}_mc_T{args.mc_passes}.npz"))
        probs_mean = npz["probs_mean"]
        tgts = npz["targets"]
        vmask = npz["valid_mask"]
        uq_mat = npz["entropy"] if method=="entropy" else (npz["var"] if method=="variance" else npz["mi"])

        uq_micro = uq_mat.copy()
        uq_micro[~vmask.astype(bool)] = np.nan
        uq_scalar = np.nanmean(uq_micro, axis=1)

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

            prev_all = prevalence_per_label(tgts, vmask, labels)
            prev_ret = prevalence_per_label(tgts[keep], vmask[keep], labels)

            rows.append([
                cov, retained_n, abstained_n,
                disc["auc_micro_macro"]["micro"],
                disc["auc_micro_macro"]["macro"],
                fn_rate,
                json.dumps(prev_ret),
                json.dumps({lab: (prev_ret[lab]-prev_all[lab]) for lab in labels})
            ])

            if split == "test" and cov in [0.90, 0.80]:
                write_json(outsp/"test"/f"confusion_matrix_cov{int(cov*100)}_{method}.json", cm)

        save_csv(outsp/split/f"coverage_curves_{method}.csv",
                 ["coverage","retained_n","abstained_n","micro_auc","macro_auc","fn_rate_retained_at_spec95sens_thr","retained_prevalence_per_label_json","prevalence_shift_per_label_json"],
                 rows)

    for method, cut in [("entropy", cut_entropy), ("variance", cut_variance), ("mi", cut_mi)]:
        make_curves("val", method, cut)
        make_curves("test", method, cut)


def step06_plots(args: Args, outdir: Path, labels: List[str], stage: str):
    plot_dir = outdir/"plots"/stage
    plot_dir.mkdir(parents=True, exist_ok=True)

    if stage == "preTS":
        probs_test = np.load(outdir/"01_predictions/test/test_probs.npy")
    else:
        probs_test = np.load(outdir/"05_temp_scaling/test_probs_postTS.npy")

    y_test = np.load(outdir/"01_predictions/test/test_targets.npy")
    m_test = np.load(outdir/"01_predictions/test/test_label_valid_mask.npy")

    # ROC micro (threshold sweep approximation)
    def roc_curve_micro(y, p, m):
        yt, yp = masked_flatten(y, p, m)
        thrs = np.linspace(0.0, 1.0, 200)
        rows = []
        for t in thrs:
            yb = (yp >= t).astype(int)
            TP = np.sum((yb==1)&(yt==1))
            FP = np.sum((yb==1)&(yt==0))
            TN = np.sum((yb==0)&(yt==0))
            FN = np.sum((yb==0)&(yt==1))
            tpr = TP/(TP+FN+1e-9)
            fpr = FP/(FP+TN+1e-9)
            rows.append([t, fpr, tpr])
        return rows

    roc_rows = roc_curve_micro(y_test, probs_test, m_test)
    save_csv(plot_dir/"roc_curve_micro.csv", ["threshold","fpr","tpr"], roc_rows)
    plt.figure()
    plt.plot([r[1] for r in roc_rows], [r[2] for r in roc_rows])
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC (Micro)")
    plt.savefig(plot_dir/"roc_curve_micro.png", dpi=200, bbox_inches="tight")
    plt.close()

    # PR micro
    def pr_curve_micro(y, p, m):
        yt, yp = masked_flatten(y, p, m)
        thrs = np.linspace(0.0, 1.0, 200)
        rows = []
        for t in thrs:
            yb = (yp >= t).astype(int)
            TP = np.sum((yb==1)&(yt==1))
            FP = np.sum((yb==1)&(yt==0))
            FN = np.sum((yb==0)&(yt==1))
            prec = TP/(TP+FP+1e-9)
            rec  = TP/(TP+FN+1e-9)
            rows.append([t, prec, rec])
        return rows

    pr_rows = pr_curve_micro(y_test, probs_test, m_test)
    save_csv(plot_dir/"pr_curve_micro.csv", ["threshold","precision","recall"], pr_rows)
    plt.figure()
    plt.plot([r[2] for r in pr_rows], [r[1] for r in pr_rows])
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR (Micro)")
    plt.savefig(plot_dir/"pr_curve_micro.png", dpi=200, bbox_inches="tight")
    plt.close()

    # per-label ROC/PR grids
    thrs = np.linspace(0.0, 1.0, 200)
    roc_per = []
    pr_per = []
    for c, lab in enumerate(labels):
        mv = m_test[:, c].astype(bool)
        yt = y_test[mv, c]
        yp = probs_test[mv, c]
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

    # Calibration plots (read bins from metrics folders)
    met_dir = outdir/("02_metrics_preTS" if stage=="preTS" else "06_metrics_postTS")
    bins_overall = []
    with open(met_dir/"reliability_bins_overall.csv","r") as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6: continue
            cnt = parts[3]; acc = parts[4]; conf = parts[5]
            if cnt == "0" or acc=="" or conf=="": continue
            bins_overall.append([float(conf), float(acc)])
    save_csv(plot_dir/"calibration_overall.csv", ["conf","acc"], bins_overall)
    plt.figure()
    if bins_overall:
        plt.plot([b[0] for b in bins_overall], [b[1] for b in bins_overall], marker="o")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("Confidence"); plt.ylabel("Accuracy"); plt.title("Calibration (Overall)")
    plt.savefig(plot_dir/"calibration_overall.png", dpi=200, bbox_inches="tight")
    plt.close()

    per_rows = []
    with open(met_dir/"reliability_bins_per_label.csv","r") as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 7: continue
            lab = parts[0]
            cnt = parts[4]; acc = parts[5]; conf = parts[6]
            if cnt == "0" or acc=="" or conf=="": continue
            per_rows.append([lab, float(conf), float(acc)])
    save_csv(plot_dir/"calibration_per_label.csv", ["label","conf","acc"], per_rows)
    plt.figure(figsize=(10,6))
    for lab in labels:
        pts = [r for r in per_rows if r[0]==lab]
        if not pts: continue
        plt.plot([p[1] for p in pts], [p[2] for p in pts], marker="o", label=lab)
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("Confidence"); plt.ylabel("Accuracy"); plt.title("Calibration (Per Label)")
    plt.legend(fontsize=7, ncol=2)
    plt.savefig(plot_dir/"calibration_per_label_grid.png", dpi=200, bbox_inches="tight")
    plt.close()

    # threshold_sweep_metrics.png from micro sweep
    thr_rows = []
    with open(met_dir/"threshold_sweep_micro.csv","r") as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 13: continue
            thr = float(parts[0])
            sens = float(parts[5]); spec = float(parts[6]); ppv = float(parts[7]); npv = float(parts[8])
            thr_rows.append([thr, sens, spec, ppv, npv])
    plt.figure()
    if thr_rows:
        plt.plot([r[0] for r in thr_rows], [r[1] for r in thr_rows], label="sens")
        plt.plot([r[0] for r in thr_rows], [r[2] for r in thr_rows], label="spec")
        plt.plot([r[0] for r in thr_rows], [r[3] for r in thr_rows], label="ppv")
        plt.plot([r[0] for r in thr_rows], [r[4] for r in thr_rows], label="npv")
    plt.xlabel("Threshold"); plt.ylabel("Metric"); plt.title("Threshold sweep (micro)")
    plt.legend()
    plt.savefig(plot_dir/"threshold_sweep_metrics.png", dpi=200, bbox_inches="tight")
    plt.close()

    # selective prediction plots + csv copies
    sp_dir = outdir/("04_selpred_preTS" if stage=="preTS" else "08_selpred_postTS")
    for method in ["entropy","variance","mi"]:
        csv_path = sp_dir/"test"/f"coverage_curves_{method}.csv"
        if not csv_path.exists():
            continue
        cov_rows = []
        with open(csv_path,"r") as f:
            next(f)
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6: continue
                cov = float(parts[0])
                micro_auc = float(parts[3])
                macro_auc = float(parts[4])
                fn_rate = float(parts[5])
                cov_rows.append([cov, micro_auc, macro_auc, fn_rate])

        save_csv(plot_dir/f"coverage_curves_{method}.csv", ["coverage","micro_auc","macro_auc","fn_rate"], cov_rows)

        plt.figure()
        plt.plot([r[0] for r in cov_rows], [r[1] for r in cov_rows], marker="o")
        plt.xlabel("Coverage"); plt.ylabel("Micro AUROC")
        plt.title(f"Coverage vs Micro AUROC ({method})")
        plt.savefig(plot_dir/f"coverage_vs_auroc_micro_{method}.png", dpi=200, bbox_inches="tight")
        plt.close()

        plt.figure()
        plt.plot([r[0] for r in cov_rows], [r[2] for r in cov_rows], marker="o")
        plt.xlabel("Coverage"); plt.ylabel("Macro AUROC")
        plt.title(f"Coverage vs Macro AUROC ({method})")
        plt.savefig(plot_dir=f"{plot_dir}/coverage_vs_auroc_macro_{method}.png", dpi=200, bbox_inches="tight")
        plt.close()

        plt.figure()
        plt.plot([r[0] for r in cov_rows], [r[3] for r in cov_rows], marker="o")
        plt.xlabel("Coverage"); plt.ylabel("FN rate (retained)")
        plt.title(f"Coverage vs FN rate ({method})")
        plt.savefig(plot_dir/f"coverage_vs_fn_{method}.png", dpi=200, bbox_inches="tight")
        plt.close()

        uq_dir = outdir/("03_uq_preTS" if stage=="preTS" else "07_uq_postTS")
        rc = uq_dir/"test"/f"test_risk_coverage_T{args.mc_passes}_{method}.csv"
        if rc.exists():
            rr=[]
            with open(rc,"r") as f:
                next(f)
                for line in f:
                    p=line.strip().split(",")
                    rr.append([float(p[0]), float(p[3])])
            plt.figure()
            plt.plot([r[0] for r in rr], [r[1] for r in rr], marker="o")
            plt.xlabel("Coverage"); plt.ylabel("Risk (Brier)")
            plt.title(f"Coverage vs Risk ({method})")
            plt.savefig(plot_dir/f"coverage_vs_risk_{method}.png", dpi=200, bbox_inches="tight")
            plt.close()

        npz = np.load(uq_dir/"test"/f"test_mc_T{args.mc_passes}.npz")
        vmask = npz["valid_mask"]
        uq_mat = npz["entropy"] if method=="entropy" else (npz["var"] if method=="variance" else npz["mi"])
        uq_mat = uq_mat.copy()
        uq_mat[~vmask.astype(bool)] = np.nan
        uq_scalar = np.nanmean(uq_mat, axis=1)
        plt.figure()
        plt.hist(uq_scalar[~np.isnan(uq_scalar)], bins=40)
        plt.xlabel("Uncertainty (micro scalar)"); plt.ylabel("Count")
        plt.title(f"Uncertainty histogram ({method})")
        plt.savefig(plot_dir/f"uncertainty_hist_{method}.png", dpi=200, bbox_inches="tight")
        plt.close()


# -----------------------
# Main
# -----------------------
def parse_args() -> Args:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", required=True)
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--use_cache", action="store_true")
    p.add_argument("--csv_path", required=True)
    p.add_argument("--checkpoint_path", required=True)

    p.add_argument("--batch_size", type=int, default=60)
    p.add_argument("--num_workers", type=int, default=10)
    p.add_argument("--image_size", type=int, default=512)

    p.add_argument("--dropout_p", type=float, default=0.3)
    p.add_argument("--mc_passes", type=int, default=60)

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--force", action="store_true")

    a = p.parse_args()
    return Args(**vars(a))


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    ensure_dirs(outdir)

    # ensure cache dir exists (ephemeral ok)
    if args.use_cache:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    labels = list(LABEL_COLUMNS)  # exact order

    df = read_csv_df(args.csv_path)

    # 00 meta
    step00_meta(args, outdir, df, labels, device)

    # 01 predictions
    step01_predictions(args, outdir, df, labels, device)

    # 02 metrics preTS
    step02_metrics(args, outdir, labels, stage="preTS")

    # 03 uq preTS
    step03_uq(args, outdir, labels, device, stage="preTS")

    # 04 selpred preTS
    step04_selpred(args, outdir, labels, stage="preTS")

    # 05 temp scaling (VAL only)
    step05_temp_scaling(args, outdir, device)

    # 06 metrics postTS
    step02_metrics(args, outdir, labels, stage="postTS")

    # 07 uq postTS (applies temperature to EACH MC pass => true postTS UQ)
    step03_uq(args, outdir, labels, device, stage="postTS")

    # 08 selpred postTS
    step04_selpred(args, outdir, labels, stage="postTS")

    # plots mirrored
    step06_plots(args, outdir, labels, stage="preTS")
    step06_plots(args, outdir, labels, stage="postTS")

    print(f"\n✅ DONE. Outputs written to: {outdir}\n")


if __name__ == "__main__":
    main()
