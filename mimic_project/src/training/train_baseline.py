import argparse
import os
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.data.dataloader_cloud import create_dataloader, LABEL_COLUMNS
from src.models.densenet121 import build_densenet121


def parse_args():
    p = argparse.ArgumentParser("MIMIC-CXR image-only training (cache + resume + masking + cosine)")

    p.add_argument("--csv_path", type=str, default="data/processed/processed_metadata.csv")
    p.add_argument("--batch_size", type=int, default=60)
    p.add_argument("--epochs", type=int, default=14)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=10)

    p.add_argument("--cache_dir", type=str, default="/ephemeral/mimic_cache")
    p.add_argument("--use_cache", action="store_true")
    p.add_argument("--no_cache", action="store_true")

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    # ✅ keep as requested
    p.add_argument("--t_max", type=int, default=100)
    p.add_argument("--eta_min", type=float, default=1e-6)

    p.add_argument("--model_out", type=str, default="models/baseline_best.pt")
    p.add_argument("--log_path", type=str, default="training-log.csv")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--eval_test_at_end", action="store_true")

    # Label masking (TRAIN only)
    p.add_argument("--enable_label_masking", action="store_true")
    p.add_argument("--mask_frac", type=float, default=0.30)
    p.add_argument("--mask_neg_share", type=float, default=0.70)
    p.add_argument("--mask_pos_share", type=float, default=0.30)

    return p.parse_args()


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_prob, average="macro"))
    except ValueError:
        return float("nan")


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt, "dict"
    if isinstance(ckpt, dict):
        return {"model_state_dict": ckpt}, "raw_dict"
    raise ValueError("Unsupported checkpoint format.")


def make_preferential_mask(
    targets: torch.Tensor,
    mask_frac: float,
    neg_share: float,
    pos_share: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    targets: [B,C] with values ~0/1 (float)
    Returns:
      masked_targets: masked entries set to -1.0 (sentinel)
      keep_mask: float mask 1=use in loss, 0=ignore
    """
    if targets.ndim != 2:
        raise ValueError(f"Expected targets shape [B,C], got {targets.shape}")

    B, C = targets.shape
    total = B * C

    if mask_frac <= 0:
        keep_mask = torch.ones_like(targets, dtype=torch.float32)
        return targets, keep_mask

    k = int(round(mask_frac * total))
    k = max(0, min(k, total - 1))  # keep at least 1 unmasked

    flat = targets.view(-1)
    pos_idx = (flat > 0.5).nonzero(as_tuple=False).squeeze(1)
    neg_idx = (flat <= 0.5).nonzero(as_tuple=False).squeeze(1)

    k_neg = int(round(k * neg_share))
    k_pos = k - k_neg

    k_neg = min(k_neg, neg_idx.numel())
    k_pos = min(k_pos, pos_idx.numel())

    remaining = k - (k_neg + k_pos)
    if remaining > 0:
        neg_left = neg_idx.numel() - k_neg
        pos_left = pos_idx.numel() - k_pos
        if neg_left >= pos_left and neg_left > 0:
            add = min(remaining, neg_left)
            k_neg += add
            remaining -= add
        if remaining > 0 and pos_left > 0:
            add = min(remaining, pos_left)
            k_pos += add
            remaining -= add

    masked_flat = torch.zeros(total, device=targets.device, dtype=torch.bool)

    if k_neg > 0:
        perm = torch.randperm(neg_idx.numel(), device=targets.device)[:k_neg]
        masked_flat[neg_idx[perm]] = True

    if k_pos > 0:
        perm = torch.randperm(pos_idx.numel(), device=targets.device)[:k_pos]
        masked_flat[pos_idx[perm]] = True

    masked = masked_flat.view(B, C)
    keep_mask = (~masked).float()

    masked_targets = targets.clone()
    masked_targets[masked] = -1.0

    return masked_targets, keep_mask


def masked_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
    # replace -1 sentinel to valid range; masked positions are multiplied by 0 anyway
    safe_targets = targets.clamp(0.0, 1.0)
    per_elem = F.binary_cross_entropy_with_logits(logits, safe_targets, reduction="none")
    denom = keep_mask.sum().clamp_min(1.0)
    return (per_elem * keep_mask).sum() / denom


def run_one_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
    train: bool,
    optimizer=None,
    scaler: Optional[GradScaler] = None,
    enable_label_masking: bool = False,
    mask_frac: float = 0.30,
    mask_neg_share: float = 0.70,
    mask_pos_share: float = 0.30,
) -> Tuple[float, np.ndarray, np.ndarray]:

    model.train() if train else model.eval()

    losses = []
    all_targets = []
    all_probs = []

    pbar = tqdm(loader, desc=("train" if train else "eval"), leave=False)

    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if train and enable_label_masking and mask_frac > 0:
            masked_targets, keep_mask = make_preferential_mask(
                targets=targets,
                mask_frac=mask_frac,
                neg_share=mask_neg_share,
                pos_share=mask_pos_share,
            )
        else:
            masked_targets = targets
            keep_mask = torch.ones_like(targets, dtype=torch.float32, device=targets.device)

        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(images)
            loss = masked_bce_with_logits(logits, masked_targets, keep_mask)

        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is None:
                raise ValueError("GradScaler is required for training when using AMP.")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        li = float(loss.detach().item())
        losses.append(li)
        pbar.set_postfix(loss=li)

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)

        # AUC should use original true targets (unmasked)
        all_targets.append(targets.detach().cpu().numpy())

    mean_loss = float(np.mean(losses))
    y_true = np.concatenate(all_targets, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)
    return mean_loss, y_true, y_prob


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_baseline] device={device}")

    use_cache = bool(args.use_cache) and (not bool(args.no_cache))
    cache_dir = args.cache_dir if use_cache else None

    train_loader = create_dataloader(
        csv_path=args.csv_path,
        split="train",
        batch_size=args.batch_size,
        image_size=args.image_size,
        cache_dir=cache_dir,
        use_cache=use_cache,
        num_workers=args.num_workers,
    )
    val_loader = create_dataloader(
        csv_path=args.csv_path,
        split="validate",
        batch_size=args.batch_size,
        image_size=args.image_size,
        cache_dir=cache_dir,
        use_cache=use_cache,
        num_workers=args.num_workers,
    )
    test_loader = create_dataloader(
        csv_path=args.csv_path,
        split="test",
        batch_size=args.batch_size,
        image_size=args.image_size,
        cache_dir=cache_dir,
        use_cache=use_cache,
        num_workers=args.num_workers,
    )

    model = build_densenet121(num_classes=len(LABEL_COLUMNS), pretrained=True, dropout_p=0.3)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.t_max, eta_min=args.eta_min)

    scaler = GradScaler(enabled=(device.type == "cuda"))

    start_epoch = 1
    best_val_auc = -1.0

    if args.resume and os.path.exists(args.model_out):
        print(f"[train_baseline] RESUME enabled. Loading {args.model_out}")
        ckpt, _ = load_checkpoint(args.model_out, device=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)

        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"] is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])

        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1
        if "best_val_auc" in ckpt:
            best_val_auc = float(ckpt["best_val_auc"])
        elif "val_auc" in ckpt:
            best_val_auc = float(ckpt["val_auc"])

        print(f"[train_baseline] Resuming from epoch {start_epoch}, best_val_auc={best_val_auc:.4f}")
    else:
        print("[train_baseline] Starting fresh training.")

    if not os.path.exists(args.log_path):
        with open(args.log_path, "w") as f:
            f.write("epoch,train_loss,val_loss,val_auc\n")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")

        train_loss, _, _ = run_one_epoch(
            model=model,
            loader=train_loader,
            device=device,
            train=True,
            optimizer=optimizer,
            scaler=scaler,
            enable_label_masking=args.enable_label_masking,
            mask_frac=args.mask_frac,
            mask_neg_share=args.mask_neg_share,
            mask_pos_share=args.mask_pos_share,
        )

        val_loss, y_true_val, y_prob_val = run_one_epoch(
            model=model,
            loader=val_loader,
            device=device,
            train=False,
            optimizer=None,
            scaler=None,
        )
        val_auc = compute_auc(y_true_val, y_prob_val)

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")

        with open(args.log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.4f},{val_loss:.4f},{val_auc:.4f}\n")

        scheduler.step()

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                    "best_val_auc": best_val_auc,
                    "label_columns": LABEL_COLUMNS,
                    "args": vars(args),
                },
                args.model_out,
            )
            print(f"✅ Saved new best model -> {args.model_out}")

    if args.eval_test_at_end:
        print("\n=== Final Test Evaluation (best checkpoint) ===")
        ckpt, _ = load_checkpoint(args.model_out, device=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)

        test_loss, y_true_test, y_prob_test = run_one_epoch(
            model=model,
            loader=test_loader,
            device=device,
            train=False,
            optimizer=None,
            scaler=None,
        )
        test_auc = compute_auc(y_true_test, y_prob_test)
        print(f"Test Loss: {test_loss:.4f} | Test AUC: {test_auc:.4f}")


if __name__ == "__main__":
    main()
