#!/usr/bin/env python3
"""
Stage 2: Pre-TS MC Dropout (UQ only) — spec-compliant

Goal:
- Run MC Dropout on VAL + TEST with T=60, p=0.3
- Save uncertainty arrays + risk/coverage CSVs for:
    entropy, variance, MI

Inputs:
- checkpoint: models/baseline_best.pt
- loaders: validate + test
- caching: use_cache + cache_dir

Outputs:
03_uq_preTS/
  val/validate_mc_T60.npz + validate_mc_T60.json
  test/test_mc_T60.npz     + test_mc_T60.json
  plus risk/coverage CSVs for each method:
    validate_risk_coverage_T60_entropy.csv, variance, mi
    test_risk_coverage_T60_entropy.csv, variance, mi

Critical GPU rule:
- prints once before inference:
    torch.cuda.is_available()
    next(model.parameters()).device
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.models.densenet121 import build_densenet121
from src.data.dataloader_cloud import create_dataloader, LABEL_COLUMNS

COVERAGE_LEVELS = [1.00, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50]


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


def save_csv(path: Path, header: List[str], rows: List[List]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")


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
# MC Dropout behavior:
# dropout ON, batchnorm eval
# -----------------------
def set_dropout_train_bn_eval(model: nn.Module):
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


# -----------------------
# UQ metrics
# -----------------------
def uq_entropy_from_mean(p_mean, eps=1e-7):
    p = np.clip(p_mean, eps, 1 - eps)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def uq_variance(probs_passes):
    # probs_passes: [T,N,C]
    return np.var(probs_passes, axis=0)


def uq_mutual_information(probs_passes, eps=1e-7):
    mean_p = np.mean(probs_passes, axis=0)
    H_mean = uq_entropy_from_mean(mean_p, eps=eps)
    H_each = uq_entropy_from_mean(np.clip(probs_passes, eps, 1 - eps), eps=eps)  # [T,N,C]
    E_H = np.mean(H_each, axis=0)
    return H_mean - E_H


def brier_score(y_true, y_prob, valid_mask):
    m = valid_mask.astype(bool).reshape(-1)
    yt = y_true.reshape(-1)[m]
    yp = y_prob.reshape(-1)[m]
    return float(np.mean((yp - yt) ** 2)) if yt.size else float("nan")


def nll_score(y_true, y_prob, valid_mask, eps=1e-7):
    m = valid_mask.astype(bool).reshape(-1)
    yt = y_true.reshape(-1)[m]
    yp = y_prob.reshape(-1)[m]
    if yt.size == 0:
        return float("nan")
    yp = np.clip(yp, eps, 1 - eps)
    return float(np.mean(-(yt * np.log(yp) + (1 - yt) * np.log(1 - yp))))


def prevalence_per_label(y_true, valid_mask, labels):
    prev = {}
    for c, lab in enumerate(labels):
        m = valid_mask[:, c].astype(bool)
        yt = y_true[m, c]
        prev[lab] = float(np.mean(yt)) if yt.size else float("nan")
    return prev


# -----------------------
# MC Dropout inference
# -----------------------
@torch.no_grad()
def mc_dropout_probs(
    model: nn.Module,
    loader,
    device: torch.device,
    T: int,
    desc_prefix: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      probs_passes: [T, N, C]
      probs_mean:   [N, C]
      targets:      [N, C]
      valid_mask:   [N, C]
    """
    set_dropout_train_bn_eval(model)

    passes = []
    targets_np = None
    mask_np = None

    for t in range(T):
        all_probs, all_tgts, all_mask = [], [], []

        pbar = tqdm(loader, desc=f"{desc_prefix} pass {t+1}/{T}", dynamic_ncols=True)
        for images, raw_labels in pbar:
            valid_mask = (raw_labels != -1).float()
            targets = (raw_labels == 1).float()

            images = images.to(device, non_blocking=True)
            logits = model(images)
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
            # ensure same order/shape across passes
            assert tgts.shape == targets_np.shape
            assert msk.shape == mask_np.shape

        passes.append(probs_np)

    probs_passes = np.stack(passes, axis=0)  # [T,N,C]
    probs_mean = np.mean(probs_passes, axis=0)
    return probs_passes, probs_mean, targets_np, mask_np


def micro_uq_scalar(uq_mat: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    # uq_mat: [N,C], valid_mask: [N,C]
    uq = uq_mat.copy()
    uq[~valid_mask.astype(bool)] = np.nan
    return np.nanmean(uq, axis=1)  # [N]


def risk_coverage_rows(
    probs_mean: np.ndarray,
    targets: np.ndarray,
    valid_mask: np.ndarray,
    uq_scalar: np.ndarray,
    labels: List[str],
) -> List[List]:
    order = np.argsort(uq_scalar)  # low uncertainty first
    uq_sorted = uq_scalar[order]
    probs_sorted = probs_mean[order]
    tgts_sorted = targets[order]
    vmask_sorted = valid_mask[order]
    N = tgts_sorted.shape[0]

    rows = []
    for cov in COVERAGE_LEVELS:
        k = int(math.ceil(cov * N))
        kept = slice(0, k)
        risk = brier_score(tgts_sorted[kept], probs_sorted[kept], vmask_sorted[kept])
        rows.append(
            [
                cov,
                k,
                N - k,
                risk,
                float(uq_sorted[k - 1]) if k > 0 else float(uq_sorted[0]),
                json.dumps(prevalence_per_label(tgts_sorted[kept], vmask_sorted[kept], labels)),
            ]
        )
    return rows


def ensure_stage2_dirs(outdir: Path):
    for p in [
        outdir / "03_uq_preTS" / "val",
        outdir / "03_uq_preTS" / "test",
    ]:
        p.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", required=True)
    p.add_argument("--cache_dir", required=True)

    # cache ON by default
    p.add_argument("--use_cache", action="store_true", default=True)
    p.add_argument("--no_cache", action="store_true")

    p.add_argument("--csv_path", required=True)
    p.add_argument("--checkpoint_path", required=True)

    p.add_argument("--batch_size", type=int, default=60)
    p.add_argument("--num_workers", type=int, default=10)
    p.add_argument("--image_size", type=int, default=512)

    p.add_argument("--dropout_p", type=float, default=0.3)
    p.add_argument("--mc_passes", type=int, default=60)

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.no_cache:
        args.use_cache = False

    outdir = Path(args.outdir)
    ensure_stage2_dirs(outdir)

    if args.use_cache:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    labels = list(LABEL_COLUMNS)

    # IMPORTANT: build model once
    model = build_model_and_load_checkpoint(args.checkpoint_path, args.dropout_p, device)

    # critical GPU verification prints
    print("\n[VERIFY] torch.cuda.is_available():", torch.cuda.is_available())
    print("[VERIFY] next(model.parameters()).device:", next(model.parameters()).device, "\n", flush=True)

    for split in ["val", "test"]:
        ds_split = "validate" if split == "val" else "test"
        prefix = "validate" if split == "val" else "test"
        split_dir = outdir / "03_uq_preTS" / split

        npz_path = split_dir / f"{prefix}_mc_T{args.mc_passes}.npz"
        json_path = split_dir / f"{prefix}_mc_T{args.mc_passes}.json"

        if (not args.force) and npz_path.exists() and json_path.exists():
            print(f"[Stage2] {split} exists, skipping (use --force to overwrite).")
            continue

        print(f"[VERIFY] use_cache={args.use_cache} cache_dir={args.cache_dir}", flush=True)
        loader = create_dataloader(
            csv_path=args.csv_path,
            split=ds_split,
            batch_size=args.batch_size,
            image_size=args.image_size,
            cache_dir=args.cache_dir,
            use_cache=args.use_cache,
            num_workers=args.num_workers,
        )

        print(f"[Stage2] Running MC Dropout on {split.upper()} (T={args.mc_passes}, p={args.dropout_p})...", flush=True)
        t0 = time.time()
        probs_passes, probs_mean, targets, valid_mask = mc_dropout_probs(
            model=model,
            loader=loader,
            device=device,
            T=args.mc_passes,
            desc_prefix=f"Stage2 {split.upper()}",
        )
        dt = time.time() - t0

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
            targets=targets,
            valid_mask=valid_mask,
        )

        # risk/coverage CSVs (micro scalar)
        for name, uq_mat in [("entropy", ent), ("variance", var), ("mi", mi)]:
            uq_scalar = micro_uq_scalar(uq_mat, valid_mask)
            rows = risk_coverage_rows(probs_mean, targets, valid_mask, uq_scalar, labels)
            save_csv(
                split_dir / f"{prefix}_risk_coverage_T{args.mc_passes}_{name}.csv",
                ["coverage", "retained_n", "abstained_n", "risk", "uq_cutoff", "retained_prevalence_per_label_json"],
                rows,
            )

        summ = {
            "split": split,
            "n_samples": int(targets.shape[0]),
            "mc_passes": int(args.mc_passes),
            "dropout_p": float(args.dropout_p),
            "stage": "preTS",
            "device": str(device),
            "seconds_total": float(dt),
            "mean_probs_brier_overall": brier_score(targets, probs_mean, valid_mask),
            "mean_probs_nll_overall": nll_score(targets, probs_mean, valid_mask),
        }
        write_json(json_path, summ)

        print(f"[Stage2] Saved: {npz_path}")
        print(f"[Stage2] Saved: {json_path}\n")

    print(f"\n✅ Stage 2 DONE. Outputs written to: {outdir}/03_uq_preTS\n")


if __name__ == "__main__":
    main()

