import os
import json
import numpy as np
import matplotlib.pyplot as plt

from src.data.dataloader_cloud import LABEL_COLUMNS


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_npz(path: str):
    d = np.load(path, allow_pickle=True)
    probs = d["probs"] if "probs" in d else d["p"]
    y_true = d["y_true"] if "y_true" in d else d["y"]
    return probs.astype(np.float32), y_true.astype(np.float32)

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

def operational_metrics_at_threshold(y_true, y_prob, thr=0.5):
    y_hat = (y_prob >= thr).astype(np.int32)
    y_t = y_true.astype(np.int32)

    out = {"threshold": float(thr), "per_label": {}, "micro": {}}

    for i, name in enumerate(LABEL_COLUMNS):
        yt = y_t[:, i]
        yp = y_hat[:, i]

        tp = int(((yp == 1) & (yt == 1)).sum())
        tn = int(((yp == 0) & (yt == 0)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        f1   = (2 * prec * rec) / (prec + rec) if np.isfinite(prec) and np.isfinite(rec) and (prec + rec) > 0 else float("nan")
        fnr  = fn / (fn + tp) if (fn + tp) > 0 else float("nan")
        fpr  = fp / (fp + tn) if (fp + tn) > 0 else float("nan")

        out["per_label"][name] = {
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "precision": None if not np.isfinite(prec) else float(prec),
            "recall_sensitivity": None if not np.isfinite(rec) else float(rec),
            "specificity": None if not np.isfinite(spec) else float(spec),
            "f1": None if not np.isfinite(f1) else float(f1),
            "fnr": None if not np.isfinite(fnr) else float(fnr),
            "fpr": None if not np.isfinite(fpr) else float(fpr),
        }

    tp = int(((y_hat == 1) & (y_t == 1)).sum())
    tn = int(((y_hat == 0) & (y_t == 0)).sum())
    fp = int(((y_hat == 1) & (y_t == 0)).sum())
    fn = int(((y_hat == 0) & (y_t == 1)).sum())

    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    f1   = (2 * prec * rec) / (prec + rec) if np.isfinite(prec) and np.isfinite(rec) and (prec + rec) > 0 else float("nan")
    fnr  = fn / (fn + tp) if (fn + tp) > 0 else float("nan")
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else float("nan")

    out["micro"] = {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": None if not np.isfinite(prec) else float(prec),
        "recall_sensitivity": None if not np.isfinite(rec) else float(rec),
        "specificity": None if not np.isfinite(spec) else float(spec),
        "f1": None if not np.isfinite(f1) else float(f1),
        "fnr": None if not np.isfinite(fnr) else float(fnr),
        "fpr": None if not np.isfinite(fpr) else float(fpr),
    }
    return out


def main():
    root = "results/image_only_nomask_main"
    in_dir = os.path.join(root, "cache_preds")
    out_root = os.path.join(root, "operational_baseline")
    out_cal = os.path.join(out_root, "calibration")
    out_ops = os.path.join(out_root, "operational")

    for d in [out_root, out_cal, out_ops]:
        ensure_dir(d)

    for split in ["validate", "test"]:
        npz = os.path.join(in_dir, f"{split}.npz")
        if not os.path.exists(npz):
            raise FileNotFoundError(f"Missing cached preds: {npz}")

        probs, y_true = load_npz(npz)

        # calibration baseline
        brier = brier_score(y_true, probs)
        ece = ece_micro(y_true, probs, n_bins=15)
        with open(os.path.join(out_cal, f"{split}_calibration_baseline.json"), "w") as f:
            json.dump({"split": split, "brier": brier, "ece_micro_15bin": ece}, f, indent=2)

        reliability_plot_micro(
            y_true, probs,
            os.path.join(out_cal, f"reliability_{split}_baseline.png"),
            n_bins=15,
            title=f"Reliability (micro) BASELINE - {split}"
        )

        # operational baseline @ 0.5
        ops = operational_metrics_at_threshold(y_true, probs, thr=0.5)
        with open(os.path.join(out_ops, f"{split}_operational_baseline.json"), "w") as f:
            json.dump(ops, f, indent=2)

    print("DONE ✅ Baseline calibration+operational saved to:", out_root)


if __name__ == "__main__":
    main()
