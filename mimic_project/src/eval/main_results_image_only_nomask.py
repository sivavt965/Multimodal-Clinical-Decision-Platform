"""
MAIN RESULTS — IMAGE ONLY (NO LABEL MASKING)
===========================================

Runs ALL baseline evaluations and saves results into one folder:

results/image_only_nomask_main/
  - cache_images/validate , cache_images/test        (raw image cache from GCS)
  - cache_preds/validate.npz , test.npz              (deterministic probs + y_true)
  - auc/{validate,test}_auc.csv + auc_summary.json
  - mcdropout/{validate,test}_mc_T55.npz + stats json + risk_coverage csv
  - calibration/{validate,test}_calibration.json + reliability_*.png

This is the CANONICAL runner for baseline image-only model results.
"""

import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.models.densenet121 import DenseNet121
from src.data.dataloader_cloud import create_dataloader, LABEL_COLUMNS


# ------------------------
# Utils
# ------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def normalize_split(s: str) -> str:
    return "validate" if s == "val" else s

def strip_module_prefix(sd):
    if not any(k.startswith("module.") for k in sd.keys()):
        return sd
    return {k.replace("module.", "", 1): v for k, v in sd.items()}

def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        elif "model" in ckpt:
            sd = ckpt["model"]
        else:
            raise KeyError(f"No model weights found in checkpoint keys: {list(ckpt.keys())}")
    else:
        sd = ckpt
    model.load_state_dict(strip_module_prefix(sd), strict=True)


# ------------------------
# Metrics
# ------------------------

def per_label_auc(y_true, y_prob):
    aucs = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) < 2:
            aucs.append(np.nan)
        else:
            aucs.append(float(roc_auc_score(y_true[:, i], y_prob[:, i])))
    return aucs

def macro_auc(aucs):
    return float(np.nanmean(np.array(aucs, dtype=np.float32)))

def micro_auc(y_true, y_prob):
    if len(np.unique(y_true.ravel())) < 2:
        return float("nan")
    return float(roc_auc_score(y_true.ravel(), y_prob.ravel()))

def brier_score(y_true, y_prob):
    return float(np.mean((y_prob - y_true) ** 2))

def ece_micro(y_true, y_prob, n_bins=15):
    yt = y_true.ravel()
    yp = y_prob.ravel()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(yp)
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        mask = (yp >= lo) & (yp < hi) if b < n_bins - 1 else (yp >= lo) & (yp <= hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(yp[mask]))
        acc = float(np.mean(yt[mask]))
        frac = float(np.sum(mask)) / float(N)
        ece += frac * abs(acc - conf)
    return float(ece)


# ------------------------
# Reliability plot
# ------------------------

def reliability_plot_micro(y_true, y_prob, out_png, n_bins=15, title="Reliability (micro)"):
    yt = y_true.ravel()
    yp = y_prob.ravel()
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    accs, confs = [], []
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        mask = (yp >= lo) & (yp < hi) if b < n_bins - 1 else (yp >= lo) & (yp <= hi)
        if not np.any(mask):
            continue
        confs.append(float(np.mean(yp[mask])))
        accs.append(float(np.mean(yt[mask])))

    plt.figure()
    plt.plot([0, 1], [0, 1])
    plt.plot(confs, accs, marker="o")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ------------------------
# Forward passes
# ------------------------

@torch.no_grad()
def forward_probs(model, loader, device, desc="forward"):
    model.eval()
    probs_list, y_list = [], []
    for batch in tqdm(loader, desc=desc):
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            images, y = batch[0], batch[1]
        elif isinstance(batch, dict):
            images = batch.get("image", batch.get("images"))
            y = batch.get("labels", batch.get("targets", batch.get("target")))
            if images is None or y is None:
                raise KeyError(f"Batch dict keys not recognized: {list(batch.keys())}")
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

        images = images.to(device, non_blocking=True)
        y = y.float()

        logits = model(images)
        probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)

        probs_list.append(probs)
        y_list.append(y.cpu().numpy().astype(np.float32))

    probs = np.concatenate(probs_list, axis=0)
    y_true = np.concatenate(y_list, axis=0)
    return probs, y_true


# ------------------------
# MC Dropout
# ------------------------

def enable_dropout_only(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()

@torch.no_grad()
def mc_dropout_probs(model, loader, device, passes=55):
    probs_passes = []
    y_true = None

    for t in tqdm(range(passes), desc=f"MC Dropout ({passes} passes)"):
        enable_dropout_only(model)

        probs_list = []
        y_list = []

        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                images, y = batch[0], batch[1]
            elif isinstance(batch, dict):
                images = batch.get("image", batch.get("images"))
                y = batch.get("labels", batch.get("targets", batch.get("target")))
                if images is None or y is None:
                    raise KeyError(f"Batch dict keys not recognized: {list(batch.keys())}")
            else:
                raise TypeError(f"Unsupported batch type: {type(batch)}")

            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)

            probs_list.append(probs)
            y_list.append(y.float().cpu().numpy().astype(np.float32))

        probs_t = np.concatenate(probs_list, axis=0)
        probs_passes.append(probs_t)

        if y_true is None:
            y_true = np.concatenate(y_list, axis=0)

    probs_all = np.stack(probs_passes, axis=0)   # [T, N, C]
    probs_mean = probs_all.mean(axis=0)          # [N, C]
    var = probs_all.var(axis=0)                  # [N, C]
    unc_meanvar = var.mean(axis=1)               # [N]
    return probs_all, probs_mean, unc_meanvar, y_true

def risk_coverage_anylabel(y_true, y_prob, unc, thr=0.5, coverages=(0.5,0.6,0.7,0.8,0.9)):
    y_hat = (y_prob >= thr).astype(np.int32)
    yt = y_true.astype(np.int32)
    wrong = (y_hat != yt).any(axis=1).astype(np.float32)

    order = np.argsort(unc)  # low uncertainty first
    N = len(order)
    rows = []
    for c in coverages:
        k = int(round(c * N))
        idx = order[:k]
        risk = float(wrong[idx].mean()) if k > 0 else float("nan")
        rows.append((float(c), risk))
    return rows


# ------------------------
# Main
# ------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="models/baseline_best.pt")
    ap.add_argument("--csv", type=str, default="data/processed/processed_metadata.csv")
    ap.add_argument("--batch_size", type=int, default=60)
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=10)
    ap.add_argument("--splits", type=str, default="validate,test")
    ap.add_argument("--mc_passes", type=int, default=55)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--ece_bins", type=int, default=15)
    ap.add_argument("--out_root", type=str, default="results/image_only_nomask_main")
    ap.add_argument("--use_cache", action="store_true", help="use local image cache_dir per split")
    args = ap.parse_args()

    splits = [normalize_split(s.strip()) for s in args.splits.split(",") if s.strip()]

    # folders
    ensure_dir(args.out_root)
    cache_img_root = os.path.join(args.out_root, "cache_images")
    cache_pred_dir = os.path.join(args.out_root, "cache_preds")
    auc_dir = os.path.join(args.out_root, "auc")
    mc_dir = os.path.join(args.out_root, "mcdropout")
    cal_dir = os.path.join(args.out_root, "calibration")

    for d in [cache_img_root, cache_pred_dir, auc_dir, mc_dir, cal_dir]:
        ensure_dir(d)

    # marker file
    marker = os.path.join(args.out_root, "THIS_IS_MAIN_RESULTS_IMAGE_ONLY_NOMASK.txt")
    if not os.path.exists(marker):
        with open(marker, "w") as f:
            f.write("MAIN RESULTS: image-only baseline, NO label masking.\n")
            f.write(f"Checkpoint: {args.ckpt}\n")
            f.write(f"CSV: {args.csv}\n")
            f.write(f"Labels: {LABEL_COLUMNS}\n")

    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet121(num_classes=len(LABEL_COLUMNS)).to(device)
    load_checkpoint(model, args.ckpt, device)

    # -------- Stage 1: deterministic cache preds (validate/test)
    for split in splits:
        out_npz = os.path.join(cache_pred_dir, f"{split}.npz")
        if os.path.exists(out_npz):
            print(f"[cache_preds] exists, skipping: {out_npz}")
            continue

        split_cache_dir = os.path.join(cache_img_root, split)
        ensure_dir(split_cache_dir)

        loader = create_dataloader(
            csv_path=args.csv,
            split=split,
            batch_size=args.batch_size,
            image_size=args.image_size,
            cache_dir=(split_cache_dir if args.use_cache else None),
            use_cache=True,
            num_workers=args.num_workers,
            shuffle=False
        )

        probs, y_true = forward_probs(model, loader, device, desc=f"{split} deterministic")
        np.savez_compressed(out_npz, probs=probs, y_true=y_true, labels=np.array(LABEL_COLUMNS, dtype=object))
        print(f"[cache_preds] saved: {out_npz}")

    # -------- Stage 2: AUC metrics from cached preds
    auc_summary = {"labels": LABEL_COLUMNS, "splits": {}}
    for split in splits:
        data = np.load(os.path.join(cache_pred_dir, f"{split}.npz"), allow_pickle=True)
        probs = data["probs"]
        y_true = data["y_true"]

        aucs = per_label_auc(y_true, probs)
        macro = macro_auc(aucs)
        micro = micro_auc(y_true, probs)

        auc_summary["splits"][split] = {
            "macro_auc": macro,
            "micro_auc": micro,
            "per_label_auc": {k: v for k, v in zip(LABEL_COLUMNS, aucs)}
        }

        out_csv = os.path.join(auc_dir, f"{split}_auc.csv")
        with open(out_csv, "w") as f:
            f.write("label,auc\n")
            for k, v in zip(LABEL_COLUMNS, aucs):
                f.write(f"{k},{v}\n")
            f.write(f"MACRO,{macro}\n")
            f.write(f"MICRO,{micro}\n")
        print(f"[auc] saved: {out_csv}")

    out_auc_json = os.path.join(auc_dir, "auc_summary.json")
    with open(out_auc_json, "w") as f:
        json.dump(auc_summary, f, indent=2)
    print(f"[auc] saved: {out_auc_json}")

    # -------- Stage 3: MC Dropout (runs model passes; uses image cache if enabled)
    for split in splits:
        out_npz = os.path.join(mc_dir, f"{split}_mc_T{args.mc_passes}.npz")
        out_json = os.path.join(mc_dir, f"{split}_mc_T{args.mc_passes}.json")
        out_rcsv = os.path.join(mc_dir, f"{split}_risk_coverage_T{args.mc_passes}.csv")

        if os.path.exists(out_npz) and os.path.exists(out_json) and os.path.exists(out_rcsv):
            print(f"[mcdropout] exists, skipping: {split}")
            continue

        split_cache_dir = os.path.join(cache_img_root, split)
        ensure_dir(split_cache_dir)

        loader = create_dataloader(
            csv_path=args.csv,
            split=split,
            batch_size=args.batch_size,
            image_size=args.image_size,
            cache_dir=(split_cache_dir if args.use_cache else None),
            use_cache=True,
            num_workers=args.num_workers,
            shuffle=False
        )

        probs_all, probs_mean, unc, y_true = mc_dropout_probs(model, loader, device, passes=args.mc_passes)

        np.savez_compressed(
            out_npz,
            probs_all=probs_all,
            probs_mean=probs_mean,
            unc=unc,
            y_true=y_true,
            labels=np.array(LABEL_COLUMNS, dtype=object),
        )
        print(f"[mcdropout] saved: {out_npz}")

        stats = {
            "split": split,
            "passes": int(args.mc_passes),
            "uncertainty_mean": float(np.mean(unc)),
            "uncertainty_p90": float(np.quantile(unc, 0.90)),
            "uncertainty_p95": float(np.quantile(unc, 0.95)),
            "uncertainty_max": float(np.max(unc)),
        }

        rc = risk_coverage_anylabel(y_true, probs_mean, unc, thr=args.thr)
        stats["risk_coverage_anylabel_error"] = [{"coverage": c, "risk": r} for c, r in rc]

        with open(out_json, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"[mcdropout] saved: {out_json}")

        with open(out_rcsv, "w") as f:
            f.write("coverage,risk_anylabel_error\n")
            for c, r in rc:
                f.write(f"{c},{r}\n")
        print(f"[mcdropout] saved: {out_rcsv}")

    # -------- Stage 4: Calibration from cached deterministic preds
    for split in splits:
        data = np.load(os.path.join(cache_pred_dir, f"{split}.npz"), allow_pickle=True)
        probs = data["probs"]
        y_true = data["y_true"]

        brier = brier_score(y_true, probs)
        ece = ece_micro(y_true, probs, n_bins=args.ece_bins)

        out_json = os.path.join(cal_dir, f"{split}_calibration.json")
        with open(out_json, "w") as f:
            json.dump({"split": split, "brier": brier, "ece_micro": ece, "n_bins": int(args.ece_bins)}, f, indent=2)
        print(f"[calibration] saved: {out_json}")

        out_png = os.path.join(cal_dir, f"reliability_{split}.png")
        reliability_plot_micro(y_true, probs, out_png, n_bins=args.ece_bins, title=f"Reliability (micro) - {split}")
        print(f"[calibration] saved: {out_png}")

    print("\nALL BASELINE RESULTS COMPLETED ✅")
    print("Saved under:", args.out_root)
    if args.use_cache:
        print("Image cache used under:", cache_img_root)
    else:
        print("Image cache disabled (pass --use_cache to cache images locally).")


if __name__ == "__main__":
    main()
