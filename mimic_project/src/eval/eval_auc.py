import argparse
import os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.models.densenet121 import DenseNet121
from src.data.dataloader_cloud import create_dataloader, LABEL_COLUMNS


@torch.no_grad()
def evaluate_auc(model, loader, device):
    model.eval()
    probs_list, targets_list = [], []

    for batch in loader:
        # supports (images, labels) or dict batches
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            images, targets = batch[0], batch[1]
        elif isinstance(batch, dict):
            images = batch.get("image", batch.get("images"))
            targets = batch.get("labels", batch.get("targets", batch.get("target")))
            if images is None or targets is None:
                raise KeyError(f"Batch dict keys not recognized: {list(batch.keys())}")
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).float()

        logits = model(images)
        probs = torch.sigmoid(logits)

        probs_list.append(probs.detach().cpu())
        targets_list.append(targets.detach().cpu())

    y_true = torch.cat(targets_list, dim=0).numpy()  # [N, C]
    y_prob = torch.cat(probs_list, dim=0).numpy()    # [N, C]

    # Per-label AUC (safe against single-class labels)
    per_label_auc = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) < 2:
            per_label_auc.append(np.nan)
        else:
            per_label_auc.append(roc_auc_score(y_true[:, i], y_prob[:, i]))

    macro_auc = float(np.nanmean(per_label_auc))

    # Micro AUC (flatten all labels)
    if len(np.unique(y_true.ravel())) < 2:
        micro_auc = float("nan")
    else:
        micro_auc = float(roc_auc_score(y_true.ravel(), y_prob.ravel()))

    return per_label_auc, macro_auc, micro_auc


def _strip_module_prefix(state_dict):
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def load_checkpoint_weights(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    # Your checkpoints store weights under model_state_dict (seen earlier)
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

    state_dict = _strip_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=True)


def normalize_split(split: str) -> str:
    # accept both "val" and "validate"
    if split == "val":
        return "validate"
    return split


def run_one_split(model, csv_path, split, batch_size, device):
    split = normalize_split(split)

    loader = create_dataloader(
        split=split,
        csv_path=csv_path,
        batch_size=batch_size,
    )

    per_label_auc, macro_auc, micro_auc = evaluate_auc(model, loader, device)

    print(f"\n=== {split.upper()} ROC-AUC ({os.path.basename(args.ckpt)}) ===")
    print("LABEL_COLUMNS order:", LABEL_COLUMNS)
    for name, auc in zip(LABEL_COLUMNS, per_label_auc):
        if np.isnan(auc):
            print(f"{name:20s}: NaN (single-class)")
        else:
            print(f"{name:20s}: {auc:.4f}")

    print(f"\nMacro AUC: {macro_auc:.4f}")
    print(f"Micro AUC: {micro_auc:.4f}")


def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="models/baseline_best.pt")
    parser.add_argument("--csv", type=str, default="data/processed/processed_metadata.csv")
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "validate", "test", "both"],
        default="both",
        help='Choose one split or "both" to run validate + test in one run.'
    )
    parser.add_argument("--batch_size", type=int, default=60)
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model with correct output size
    model = DenseNet121(num_classes=len(LABEL_COLUMNS)).to(device)

    # Load weights once
    load_checkpoint_weights(model, args.ckpt, device)

    # Run one or both splits
    if args.split == "both":
        run_one_split(model, args.csv, "validate", args.batch_size, device)
        run_one_split(model, args.csv, "test", args.batch_size, device)
    else:
        run_one_split(model, args.csv, args.split, args.batch_size, device)


if __name__ == "__main__":
    main()
