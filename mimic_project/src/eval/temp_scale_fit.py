#!/usr/bin/env python3
"""
Stage 4: Temperature scaling (fit on VAL only) — spec compliant

Inputs (from Stage 1):
- 01_predictions/val/val_logits.npy
- 01_predictions/val/val_targets.npy
- 01_predictions/val/val_valid_mask.npy   (ignore uncertain labels where mask=0)
- 01_predictions/test/test_logits.npy     (only for applying T after fit; NOT for fitting)

Outputs:
05_temp_scaling/
- temperature.txt
- val_calibration_pre_post.json
- val_probs_postTS.npy
- test_probs_postTS.npy

Notes:
- Fit is done on VAL ONLY using masked NLL
- Runs on CPU by default; can run on cuda too, but not required
- Prints verification lines once
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


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


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def masked_nll_torch(logits, targets, mask, T, eps=1e-7):
    # logits/targets/mask: [N,C] tensors on same device
    probs = torch.sigmoid(logits / T).clamp(eps, 1 - eps)
    nll = -(targets * torch.log(probs) + (1 - targets) * torch.log(1 - probs))
    nll = (nll * mask).sum() / (mask.sum() + 1e-9)
    return nll


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


def ece_bins_overall(y_true, y_prob, valid_mask, n_bins=15):
    yt = y_true.reshape(-1)[valid_mask.astype(bool).reshape(-1)]
    yp = y_prob.reshape(-1)[valid_mask.astype(bool).reshape(-1)]
    if yt.size == 0:
        return float("nan")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (yp >= lo) & (yp < hi) if i < n_bins - 1 else (yp >= lo) & (yp <= hi)
        if not np.any(m):
            continue
        conf = float(np.mean(yp[m]))
        acc = float(np.mean(yt[m]))
        cnt = int(np.sum(m))
        ece += (cnt / yt.size) * abs(acc - conf)
    return float(ece)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", required=True)
    p.add_argument("--device", default="cpu")  # cpu is fine
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    ts_dir = outdir / "05_temp_scaling"
    ts_dir.mkdir(parents=True, exist_ok=True)

    temp_path = ts_dir / "temperature.txt"
    if temp_path.exists() and not args.force:
        print(f"[Stage4] {temp_path} exists, skipping (use --force to overwrite).")
        return

    # Inputs
    val_logits = np.load(outdir / "01_predictions/val/val_logits.npy")
    val_t = np.load(outdir / "01_predictions/val/val_targets.npy")
    val_m = np.load(outdir / "01_predictions/val/val_valid_mask.npy")

    test_logits = np.load(outdir / "01_predictions/test/test_logits.npy")

    device = torch.device(args.device)

    print("\n[VERIFY] torch.cuda.is_available():", torch.cuda.is_available())
    print("[VERIFY] device used for TS fit:", device, "\n", flush=True)

    logits_t = torch.tensor(val_logits, dtype=torch.float32, device=device)
    targets_t = torch.tensor(val_t, dtype=torch.float32, device=device)
    mask_t = torch.tensor(val_m, dtype=torch.float32, device=device)

    # Optimize scalar temperature (logT) with LBFGS on VAL masked NLL
    logT = torch.zeros((), device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([logT], lr=0.5, max_iter=200)

    def closure():
        optimizer.zero_grad()
        T = torch.exp(logT).clamp(1e-3, 100.0)
        loss = masked_nll_torch(logits_t, targets_t, mask_t, T)
        loss.backward()
        return loss

    optimizer.step(closure)
    T_final = float(torch.exp(logT).detach().cpu().item())

    with open(temp_path, "w") as f:
        f.write(f"{T_final:.6f}\n")

    # Pre vs post calibration on VAL
    val_probs_pre = sigmoid_np(val_logits)
    val_probs_post = sigmoid_np(val_logits / T_final)

    pre = {
        "ece": ece_bins_overall(val_t, val_probs_pre, val_m, n_bins=15),
        "nll": nll_score(val_t, val_probs_pre, val_m),
        "brier": brier_score(val_t, val_probs_pre, val_m),
    }
    post = {
        "ece": ece_bins_overall(val_t, val_probs_post, val_m, n_bins=15),
        "nll": nll_score(val_t, val_probs_post, val_m),
        "brier": brier_score(val_t, val_probs_post, val_m),
    }

    write_json(ts_dir / "val_calibration_pre_post.json", {"temperature": T_final, "pre": pre, "post": post})

    # Save post-TS probs for Stage 5
    np.save(ts_dir / "val_probs_postTS.npy", val_probs_post)
    np.save(ts_dir / "test_probs_postTS.npy", sigmoid_np(test_logits / T_final))

    print(f"\n✅ Stage 4 DONE. Temperature={T_final:.6f}")
    print(f"Saved: {temp_path}")
    print(f"Saved: {ts_dir/'val_probs_postTS.npy'}")
    print(f"Saved: {ts_dir/'test_probs_postTS.npy'}\n")


if __name__ == "__main__":
    main()
