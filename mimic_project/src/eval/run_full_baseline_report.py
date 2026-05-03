import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from src.models.densenet121 import DenseNet121
from src.data.dataloader_cloud import create_dataloader, LABEL_COLUMNS


# -------------------------
# Helpers
# -------------------------

def normalize_split(split: str) -> str:
    # your dataloader wants "validate", but user may pass "val"
    return "validate" if split == "val" else split

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def strip_module_prefix(state_dict):
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}

def load_checkpoint_weights(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            raise KeyError(f"No model weights found in checkpoint keys: {list(ckpt.keys())}")
    else:
        state_dict = ckpt
    state_dict = strip_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=True)

def set_dropout_train_only(model: torch.nn.Module):
    """
    Keep model in eval mode overall, but enable dropout layers for MC Dropout.
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()

@torch.no_grad()
def collect_logits_and_targets(model, loader, device, mc_passes: int = 1, mc_dropout: bool = False):
    """
    Returns:
      logits_mean: [N, C] float32
      probs_mean : [N, C] float32
      probs_all  : [T, N, C] if mc_passes>1 else None
      y_true     : [N, C] float32
    """
    if mc_passes < 1:
        raise ValueError("mc_passes must be >= 1")

    # Collect targets once (single pass)
    ys = []
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            _, targets = batch[0], batch[1]
        elif isinstance(batch, dict):
            targets = batch.get("labels", batch.get("targets", batch.get("target")))
            if targets is None:
                raise KeyError(f"Batch dict keys not recognized: {list(batch.keys())}")
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")
        ys.append(targets.float().cpu())
    y_true = torch.cat(ys, dim=0).numpy().astype(np.float32)

    # Collect probs for each pass
    probs_passes = []
    for t in range(mc_passes):
        if mc_dropout and mc_passes > 1:
            set_dropout_train_only(model)
        else:
            model.eval()

        ps = []
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                images = batch[0]
            elif isinstance(batch, dict):
                images = batch.get("image", batch.get("images"))
                if images is None:
                    raise KeyError(f"Batch dict keys not recognized: {list(batch.keys())}")
            else:
                raise TypeError(f"Unsupported batch type: {type(batch)}")

            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.sigmoid(logits)
            ps.append(probs.detach().cpu())

        probs_passes.append(torch.cat(ps, dim=0).numpy().astype(np.float32))

    probs_all = np.stack(probs_passes, axis=0)  # [T, N, C]
    probs_mean = probs_all.mean(axis=0)         # [N, C]

    # Convert mean probs back to logits_mean (numerically safe)
    eps = 1e-7
    probs_clipped = np.clip(probs_mean, eps, 1 - eps)
    logits_mean = np.log(probs_clipped / (1 - probs_clipped)).astype(np.float32)

    return logits_mean, probs_mean, (probs_all if mc_passes > 1 else None), y_true

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
    # mean squared error over all labels
    return float(np.mean((y_prob - y_true) ** 2))

def expected_calibration_error(y_true, y_prob, n_bins=15):
    """
    ECE over flattened (micro) probabilities.
    """
    y_true_f = y_true.ravel()
    y_prob_f = y_prob.ravel()

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(y_prob_f)

    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        mask = (y_prob_f >= lo) & (y_prob_f < hi) if b < n_bins - 1 else (y_prob_f >= lo) & (y_prob_f <= hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(y_prob_f[mask]))
        acc = float(np.mean(y_true_f[mask]))
        frac = float(np.sum(mask)) / float(N)
        ece += frac * abs(acc - conf)

    return float(ece)

def confusion_metrics_at_threshold(y_true, y_prob, thr=0.5):
    """
    Per-label operational metrics at a fixed threshold.
    Returns dict of per-label metrics + micro-averaged metrics.
    """
    y_hat = (y_prob >= thr).astype(np.int32)
    out = {"threshold": float(thr), "per_label": {}, "micro": {}}

    # per label
    for i, name in enumerate(LABEL_COLUMNS):
        yt = y_true[:, i].astype(np.int32)
        yp = y_hat[:, i].astype(np.int32)

        tp = int(((yp == 1) & (yt == 1)).sum())
        tn = int(((yp == 0) & (yt == 0)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")  # sensitivity
        spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        f1   = (2 * prec * rec) / (prec + rec) if (np.isfinite(prec) and np.isfinite(rec) and (prec + rec) > 0) else float("nan")

        out["per_label"][name] = {
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "precision": float(prec) if np.isfinite(prec) else None,
            "recall_sensitivity": float(rec) if np.isfinite(rec) else None,
            "specificity": float(spec) if np.isfinite(spec) else None,
            "f1": float(f1) if np.isfinite(f1) else None,
        }

    # micro (sum over all labels)
    tp = int(((y_hat == 1) & (y_true == 1)).sum())
    tn = int(((y_hat == 0) & (y_true == 0)).sum())
    fp = int(((y_hat == 1) & (y_true == 0)).sum())
    fn = int(((y_hat == 0) & (y_true == 1)).sum())

    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    f1   = (2 * prec * rec) / (prec + rec) if (np.isfinite(prec) and np.isfinite(rec) and (prec + rec) > 0) else float("nan")

    out["micro"] = {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": float(prec) if np.isfinite(prec) else None,
        "recall_sensitivity": float(rec) if np.isfinite(rec) else None,
        "specificity": float(spec) if np.isfinite(spec) else None,
        "f1": float(f1) if np.isfinite(f1) else None,
    }
    return out

def selective_prediction_metrics(y_true, y_prob_mean, uncertainty, coverages=(0.5, 0.6, 0.7, 0.8, 0.9)):
    """
    Simple operational selective-prediction summary:
    - Sort by uncertainty (low -> high)
    - Keep top k% most certain
    - Report micro error rate on kept set (risk)
    """
    # Use micro-error at threshold 0.5 as "risk" (you can swap later)
    thr = 0.5
    y_hat = (y_prob_mean >= thr).astype(np.int32)
    y_true_i = y_true.astype(np.int32)

    # define micro error per example (across labels)
    # error=1 if any label wrong; alternative definitions exist, but this is a clean start
    per_example_wrong = (y_hat != y_true_i).any(axis=1).astype(np.int32)

    order = np.argsort(uncertainty)  # most certain first
    N = len(order)
    results = []
    for c in coverages:
        k = int(round(c * N))
        idx = order[:k]
        risk = float(per_example_wrong[idx].mean()) if k > 0 else float("nan")
        results.append({"coverage": float(c), "risk_anylabel_error": risk})
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="models/baseline_best.pt")
    p.add_argument("--csv", type=str, default="data/processed/processed_metadata.csv")
    p.add_argument("--batch_size", type=int, default=60)
    p.add_argument("--splits", type=str, default="validate,test", help="comma-separated: train,val,validate,test")
    p.add_argument("--out_dir", type=str, default="results/image_only_nomask")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--mc_passes", type=int, default=1, help=">1 enables MC Dropout passes")
    p.add_argument("--mc_dropout", action="store_true", help="Enable dropout at inference if mc_passes>1")
    args = p.parse_args()

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(args.ckpt)
    if not os.path.exists(args.csv):
        raise FileNotFoundError(args.csv)

    ensure_dir(args.out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DenseNet121(num_classes=len(LABEL_COLUMNS)).to(device)
    load_checkpoint_weights(model, args.ckpt, device)

    report = {
        "run_tag": "image_only_no_label_masking",
        "checkpoint": args.ckpt,
        "csv": args.csv,
        "label_columns_order": LABEL_COLUMNS,
        "threshold": float(args.threshold),
        "mc_passes": int(args.mc_passes),
        "mc_dropout": bool(args.mc_dropout),
        "splits": {}
    }

    splits = [normalize_split(s.strip()) for s in args.splits.split(",") if s.strip()]
    for split in splits:
        loader = create_dataloader(split=split, csv_path=args.csv, batch_size=args.batch_size)

        logits_mean, probs_mean, probs_all, y_true = collect_logits_and_targets(
            model, loader, device,
            mc_passes=args.mc_passes,
            mc_dropout=args.mc_dropout
        )

        # AUCs
        aucs = per_label_auc(y_true, probs_mean)
        macro = macro_auc(aucs)
        micro = micro_auc(y_true, probs_mean)

        # Calibration
        brier = brier_score(y_true, probs_mean)
        ece = expected_calibration_error(y_true, probs_mean, n_bins=15)

        # Operational metrics (thresholded)
        ops = confusion_metrics_at_threshold(y_true, probs_mean, thr=args.threshold)

        split_block = {
            "n_rows": int(y_true.shape[0]),
            "per_label_auc": {name: auc for name, auc in zip(LABEL_COLUMNS, aucs)},
            "macro_auc": macro,
            "micro_auc": micro,
            "brier": brier,
            "ece_15bin_micro": ece,
            "operational_metrics": ops,
        }

        # MC dropout / selective prediction
        if probs_all is not None:
            # predictive uncertainty: mean variance across classes
            var = probs_all.var(axis=0)               # [N, C]
            unc = var.mean(axis=1)                    # [N]
            split_block["mc_uncertainty_meanvar"] = {
                "mean": float(unc.mean()),
                "p90": float(np.quantile(unc, 0.90)),
                "p95": float(np.quantile(unc, 0.95)),
                "max": float(unc.max()),
            }
            split_block["selective_prediction"] = selective_prediction_metrics(
                y_true=y_true,
                y_prob_mean=probs_mean,
                uncertainty=unc,
                coverages=(0.5, 0.6, 0.7, 0.8, 0.9)
            )

        report["splits"][split] = split_block

        # Print quick summary to terminal
        print(f"\n=== {split.upper()} SUMMARY ===")
        print("Macro AUC:", f"{macro:.4f}", "| Micro AUC:", f"{micro:.4f}", "| Brier:", f"{brier:.5f}", "| ECE:", f"{ece:.5f}")

    # Save JSON (main “one file”)
    json_path = os.path.join(args.out_dir, "baseline_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print("\nSaved:", json_path)

    # Also save a compact CSV for AUCs
    csv_path = os.path.join(args.out_dir, "baseline_auc_table.csv")
    with open(csv_path, "w") as f:
        header = ["split", "macro_auc", "micro_auc"] + [f"auc_{c}" for c in LABEL_COLUMNS]
        f.write(",".join(header) + "\n")
        for split, block in report["splits"].items():
            row = [split, str(block["macro_auc"]), str(block["micro_auc"])]
            row += [str(block["per_label_auc"][c]) for c in LABEL_COLUMNS]
            f.write(",".join(row) + "\n")
    print("Saved:", csv_path)


if __name__ == "__main__":
    main()
