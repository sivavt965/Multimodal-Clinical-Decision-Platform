import argparse
import os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.models.densenet121 import DenseNet121
from src.data.dataloader_cloud import create_dataloader


LABEL_NAMES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Opacity",
    "Pleural Effusion",
]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probs_list, targets_list = [], []

    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            images, targets = batch[0], batch[1]
        elif isinstance(batch, dict):
            images = batch.get("image", batch.get("images"))
            targets = batch.get("labels", batch.get("targets", batch.get("target")))
            if images is None or targets is None:
                raise KeyError(f"Batch dict keys: {list(batch.keys())}")
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).float()

        logits = model(images)
        probs = torch.sigmoid(logits)

        probs_list.append(probs.detach().cpu())
        targets_list.append(targets.detach().cpu())

    y_true = torch.cat(targets_list, dim=0).numpy()
    y_prob = torch.cat(probs_list, dim=0).numpy()

    per_label_auc = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) < 2:
            per_label_auc.append(np.nan)
        else:
            per_label_auc.append(roc_auc_score(y_true[:, i], y_prob[:, i]))

    macro_auc = float(np.nanmean(per_label_auc))

    if len(np.unique(y_true.ravel())) < 2:
        micro_auc = float("nan")
    else:
        micro_auc = float(roc_auc_score(y_true.ravel(), y_prob.ravel()))

    return per_label_auc, macro_auc, micro_auc


def _strip_module_prefix(state_dict):
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def load_best_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict):
        # your checkpoint uses model_state_dict (confirmed earlier)
        for key in ["model_state_dict", "state_dict", "model", "net"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                state_dict = ckpt[key]
                break
        else:
            raise KeyError(f"No model weights found in checkpoint keys: {list(ckpt.keys())}")
    else:
        state_dict = ckpt

    state_dict = _strip_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="models/baseline_best.pt")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--csv", type=str, default=None, help="Path to the split CSV (test/val/train)")
    parser.add_argument("--batch_size", type=int, default=60)
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    # If user didn't pass --csv, try common defaults
    if args.csv is None:
        candidates = [
            f"data/{args.split}.csv",
            f"data/{args.split}_split.csv",
            f"data/{args.split}_data.csv",
            f"data/mimic_{args.split}.csv",
            f"data/physionet_{args.split}.csv",
        ]
        found = None
        for p in candidates:
            if os.path.exists(p):
                found = p
                break
        if found is None:
            raise FileNotFoundError(
                "Couldn't auto-find split CSV. Please pass it explicitly, e.g.\n"
                "python src/eval/test_auc.py --csv data/test.csv --batch_size 60\n"
                f"Tried: {candidates}"
            )
        args.csv = found

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DenseNet121(num_classes=8).to(device)
    load_best_checkpoint(model, args.ckpt, device)

    # ✅ Your dataloader requires csv_path and batch_size
    test_loader = create_dataloader(
        split=args.split,
        csv_path=args.csv,
        batch_size=args.batch_size,
    )

    per_label_auc, macro_auc, micro_auc = evaluate(model, test_loader, device)

    print("\n=== TEST ROC-AUC (baseline_best.pt) ===")
    print(f"CSV: {args.csv}")
    for name, auc in zip(LABEL_NAMES, per_label_auc):
        if np.isnan(auc):
            print(f"{name:28s}: NaN (single-class)")
        else:
            print(f"{name:28s}: {auc:.4f}")

    print(f"\nMacro AUC: {macro_auc:.4f}")
    print(f"Micro AUC: {micro_auc:.4f}")


if __name__ == "__main__":
    main()

