import os
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.data.dataloader_cloud import LABEL_COLUMNS


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_npz(path: str):
    d = np.load(path, allow_pickle=True)
    probs = d["probs"] if "probs" in d else d["p"]
    y_true = d["y_true"] if "y_true" in d else d["y"]
    return probs.astype(np.float32), y_true.astype(np.float32)

def save_npz(path: str, probs: np.ndarray, y_true: np.ndarray):
    np.savez_compressed(
        path,
        probs=probs.astype(np.float32),
        y_true=y_true.astype(np.float32),
        labels=np.array(LABEL_COLUMNS, dtype=object),
    )

def prob_to_logit(p: np.ndarray, eps: float = 1e-7):
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p)).astype(np.float32)

def sigmoid(x: np.ndarray):
    return 1.0 / (1.0 + np.exp(-x))

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


def fit_temperature_on_validate(probs_val: np.ndarray, y_val: np.ndarray, device="cpu"):
    logits = prob_to_logit(probs_val)
    logits_t = torch.from_numpy(logits).to(device)
    y_t = torch.from_numpy(y_val).to(device)

    logT = torch.zeros((), device=device, requires_grad=True)
    criterion = nn.BCEWithLogitsLoss()

    opt = torch.optim.LBFGS([logT], lr=0.5, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        T = torch.exp(logT) + 1e-8
        loss = criterion(logits_t / T, y_t)
        loss.backward()
        return loss

    opt.step(closure)
    return float((torch.exp(logT) + 1e-8).detach().cpu().item())

def apply_temperature(probs: np.ndarray, T: float):
    logits = prob_to_logit(probs)
    return sigmoid(logits / T).astype(np.float32)


def main():
    root = "results/image_only_nomask_main"
    in_dir = os.path.join(root, "cache_preds")
    val_path = os.path.join(in_dir, "validate.npz")
    test_path = os.path.join(in_dir, "test.npz")

    if not os.path.exists(val_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            "Missing cached predictions. Expected:\n"
            f"  {val_path}\n  {test_path}\n"
            "Run main baseline script first to create cache_preds/*.npz"
        )

    out_root = os.path.join(root, "temp_scaling")
    out_preds = os.path.join(out_root, "calibrated_preds")
    out_cal = os.path.join(out_root, "calibration")
    out_ops = os.path.join(out_root, "operational")
    for d in [out_root, out_preds, out_cal, out_ops]:
        ensure_dir(d)

    probs_val, y_val = load_npz(val_path)
    probs_test, y_test = load_npz(test_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    T = fit_temperature_on_validate(probs_val, y_val, device=device)

    with open(os.path.join(out_root, "T.json"), "w") as f:
        json.dump({"temperature_T": T, "fit_split": "validate", "labels": LABEL_COLUMNS}, f, indent=2)

    probs_val_cal = apply_temperature(probs_val, T)
    probs_test_cal = apply_temperature(probs_test, T)

    save_npz(os.path.join(out_preds, "validate_calibrated.npz"), probs_val_cal, y_val)
    save_npz(os.path.join(out_preds, "test_calibrated.npz"), probs_test_cal, y_test)

    # AFTER-TS calibration + plots
    for split, y_true, p_cal in [("validate", y_val, probs_val_cal), ("test", y_test, probs_test_cal)]:
        brier = brier_score(y_true, p_cal)
        ece = ece_micro(y_true, p_cal, n_bins=15)

        with open(os.path.join(out_cal, f"{split}_calibration_after_ts.json"), "w") as f:
            json.dump(
                {"split": split, "brier": brier, "ece_micro_15bin": ece, "temperature_T": T},
                f, indent=2
            )

        reliability_plot_micro(
            y_true, p_cal,
            os.path.join(out_cal, f"reliability_{split}_after_ts.png"),
            n_bins=15,
            title=f"Reliability (micro) AFTER TS - {split}"
        )

    # AFTER-TS operational @ 0.5
    rows = []
    for split, y_true, p_cal in [("validate", y_val, probs_val_cal), ("test", y_test, probs_test_cal)]:
        ops = operational_metrics_at_threshold(y_true, p_cal, thr=0.5)
        with open(os.path.join(out_ops, f"{split}_operational_after_ts.json"), "w") as f:
            json.dump(ops, f, indent=2)

        m = ops["micro"]
        rows.append((split, m["precision"], m["recall_sensitivity"], m["specificity"], m["f1"], m["fnr"], m["fpr"]))

    with open(os.path.join(out_ops, "operational_after_ts_summary.csv"), "w") as f:
        f.write("split,micro_precision,micro_recall,micro_specificity,micro_f1,micro_fnr,micro_fpr\n")
        for r in rows:
            f.write(",".join([str(x) for x in r]) + "\n")

    print("DONE ✅ Temp scaling + AFTER calibration+operational saved to:", out_root)
    print(f"Learned T = {T:.4f}")


if __name__ == "__main__":
    main()
