import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def multilabel_predictive_entropy(p_mean: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """
    p_mean: [N, C] mean sigmoid probabilities across MC passes
    Returns: [N] entropy summed across labels (multilabel Bernoulli entropy)
    """
    p = np.clip(p_mean, eps, 1 - eps)
    ent = -(p * np.log(p) + (1 - p) * np.log(1 - p))  # [N,C]
    return ent.sum(axis=1)  # [N]


def any_label_error(yhat: np.ndarray, ytrue: np.ndarray) -> np.ndarray:
    """Strict error: 1 if ANY label wrong."""
    return (yhat != ytrue).any(axis=1).astype(np.int32)


def hamming_error(yhat: np.ndarray, ytrue: np.ndarray) -> np.ndarray:
    """Hamming loss per sample in [0,1]."""
    return (yhat != ytrue).mean(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to *_mc_T55.npz")
    ap.add_argument("--out_dir", default="results/uncertainty_rescore", help="Output folder")
    ap.add_argument("--threshold", type=float, default=0.5, help="Threshold to binarize mean probs")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.npz, allow_pickle=True)

    # Try common keys (adjust if your npz uses different names)
    # Expect probs_all: [T, N, C] and y_true: [N, C]
    probs_all = None
    y_true = None

    for k in ["probs_all", "probs", "mc_probs", "probs_mc"]:
        if k in data:
            probs_all = data[k]
            break
    for k in ["y_true", "targets", "y", "labels"]:
        if k in data:
            y_true = data[k]
            break

    if probs_all is None or y_true is None:
        print("NPZ keys:", list(data.keys()))
        raise ValueError("Could not find probs_all and y_true in the NPZ. See keys above and adjust names.")

    probs_all = np.asarray(probs_all)  # [T,N,C]
    y_true = np.asarray(y_true).astype(np.int32)  # [N,C]

    # Mean probabilities across MC passes
    p_mean = probs_all.mean(axis=0)  # [N,C]
    y_hat = (p_mean >= args.threshold).astype(np.int32)

    # Uncertainty scores
    var = probs_all.var(axis=0)  # [N,C]
    unc_meanvar = var.mean(axis=1)  # [N]
    unc_maxvar = var.max(axis=1)  # [N]
    unc_entropy = multilabel_predictive_entropy(p_mean)  # [N]

    # Error definitions
    err_any = any_label_error(y_hat, y_true)  # [N] binary
    err_ham = hamming_error(y_hat, y_true)    # [N] float

    # AUROC(uncertainty, error) for binary targets:
    # For Hamming, convert to binary "has any error" OR use a cutoff.
    # We'll do both:
    err_ham_bin = (err_ham > 0).astype(np.int32)

    def safe_auroc(ybin, score):
        if len(np.unique(ybin)) < 2:
            return None
        return float(roc_auc_score(ybin, score))

    results = {
        "npz": args.npz,
        "threshold": args.threshold,
        "shapes": {
            "probs_all": list(probs_all.shape),
            "p_mean": list(p_mean.shape),
            "y_true": list(y_true.shape),
        },
        "uncertainty_summary": {
            "meanvar": {
                "mean": float(np.mean(unc_meanvar)),
                "p90": float(np.percentile(unc_meanvar, 90)),
                "p95": float(np.percentile(unc_meanvar, 95)),
                "max": float(np.max(unc_meanvar)),
            },
            "maxvar": {
                "mean": float(np.mean(unc_maxvar)),
                "p90": float(np.percentile(unc_maxvar, 90)),
                "p95": float(np.percentile(unc_maxvar, 95)),
                "max": float(np.max(unc_maxvar)),
            },
            "entropy": {
                "mean": float(np.mean(unc_entropy)),
                "p90": float(np.percentile(unc_entropy, 90)),
                "p95": float(np.percentile(unc_entropy, 95)),
                "max": float(np.max(unc_entropy)),
            },
        },
        "auroc_uncertainty_vs_error": {
            "any_label_error": {
                "meanvar": safe_auroc(err_any, unc_meanvar),
                "maxvar": safe_auroc(err_any, unc_maxvar),
                "entropy": safe_auroc(err_any, unc_entropy),
            },
            "hamming_error_bin": {
                "meanvar": safe_auroc(err_ham_bin, unc_meanvar),
                "maxvar": safe_auroc(err_ham_bin, unc_maxvar),
                "entropy": safe_auroc(err_ham_bin, unc_entropy),
            }
        }
    }

    # Bucket test (top/bottom 10% uncertainty) for strict any-label error rate
    def bucket_error_rate(unc, err_bin, q=0.10):
        n = len(unc)
        k = max(1, int(n * q))
        idx = np.argsort(unc)
        low = idx[:k]
        high = idx[-k:]
        return float(err_bin[low].mean()), float(err_bin[high].mean())

    low_m, high_m = bucket_error_rate(unc_meanvar, err_any)
    low_x, high_x = bucket_error_rate(unc_maxvar, err_any)
    low_e, high_e = bucket_error_rate(unc_entropy, err_any)

    results["bucket_test_any_error_rate_top_bottom_10pct"] = {
        "meanvar": {"low10%": low_m, "high10%": high_m},
        "maxvar": {"low10%": low_x, "high10%": high_x},
        "entropy": {"low10%": low_e, "high10%": high_e},
    }

    # Save json + arrays
    out_json = out_dir / (Path(args.npz).stem + "_uncertainty_rescore.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    np.savez_compressed(
        out_dir / (Path(args.npz).stem + "_uncertainty_scores.npz"),
        unc_meanvar=unc_meanvar,
        unc_maxvar=unc_maxvar,
        unc_entropy=unc_entropy,
        err_any=err_any,
        err_ham=err_ham,
        p_mean=p_mean,
        y_true=y_true,
    )

    print("Saved:", out_json)
    print(json.dumps(results["auroc_uncertainty_vs_error"], indent=2))
    print("Bucket test (any-label error rate):")
    print(json.dumps(results["bucket_test_any_error_rate_top_bottom_10pct"], indent=2))


if __name__ == "__main__":
    main()
