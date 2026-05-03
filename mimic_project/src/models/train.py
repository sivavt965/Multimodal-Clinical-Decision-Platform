# src/training/train_mm.py
#
# Multimodal trainer (FiLM/Gated) — FULL CODE
# Essentials implemented:
# - batch_size = 10
# - unfreeze at epoch 3 (epochs 0,1,2 frozen)
# - DenseNet LR = 3e-5 (gentle finetune), heads/meta LR = 3e-4
# - ImageNet pretrained DenseNet (3-channel) assumed (matches baseline)
# - pos_weight computed from TRAIN valid labels only (clamped)
# - Masked loss (ignores -1/NaN etc via mask)
# - Sanity print: n_valid / pos / neg per label during val
# - Saves best checkpoint by val_loss
#
# Env vars expected:
#   TRAIN_CSV, VAL_PERF_CSV
# Optional:
#   OUTDIR, CACHE_DIR, BASELINE_CKPT
#   SPLIT_COL (default: "split")
#   TRAIN_SPLIT (default: "train")
#   VAL_SPLIT (default: "validate")
#
import os
import json
import time
import csv
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from src.data.dataloader_cloud_mm import create_dataloader_mm, LABEL_COLUMNS
from src.models.model_mm_film_gated import MultiModalFiLMGated


# -----------------------------
# Config
# -----------------------------
@dataclass
class TrainCfg:
    img_size: int = 512
    batch_size: int = 10
    num_workers: int = 10

    max_epochs: int = 30
    amp: bool = True
    grad_clip: float = 1.0

    # Unfreeze at epoch 3 => epochs 0,1,2 frozen
    freeze_backbone_epochs: int = 3

    # Gentle finetune
    lr_densenet: float = 3e-5
    lr_heads: float = 3e-4
    wd_densenet: float = 1e-4
    wd_heads: float = 1e-5
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    # Optional warmup for first epochs
    warmup_factors: Tuple[float, ...] = (0.33, 0.66)

    # Optional gate regularization (keeps gates from collapsing early)
    gate_lambda_init: float = 0.01
    gate_lambda_epochs: int = 5


# -----------------------------
# Utils
# -----------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def warmup_factor(epoch: int, cfg: TrainCfg) -> float:
    return float(cfg.warmup_factors[epoch]) if epoch < len(cfg.warmup_factors) else 1.0

def apply_warmup(opt: torch.optim.Optimizer, epoch: int, cfg: TrainCfg):
    f = warmup_factor(epoch, cfg)
    for pg in opt.param_groups:
        pg["lr"] = pg["base_lr"] * f

def gate_lambda(epoch: int, cfg: TrainCfg) -> float:
    L = max(1, int(cfg.gate_lambda_epochs))
    return float(cfg.gate_lambda_init) * max(0.0, (L - epoch) / L)

def set_backbone_trainable(model: MultiModalFiLMGated, trainable: bool):
    for p in model.image_encoder.parameters():
        p.requires_grad = trainable

def set_backbone_lr(opt: torch.optim.Optimizer, lr: float):
    # backbone is param_group[0] by construction below
    opt.param_groups[0]["lr"] = float(lr)

def compute_pos_weight_from_train(
    train_csv: str,
    split_col: str,
    train_split: str,
    clamp_max: float = 30.0,
) -> np.ndarray:
    """
    pos_weight[k] = neg/pos computed only on valid labels {0,1} in TRAIN split.
    Clamped for stability.
    """
    df = pd.read_csv(train_csv)
    df = df[df[split_col] == train_split].reset_index(drop=True)

    pos_w = []
    for col in LABEL_COLUMNS:
        v = df[col].values
        valid = np.isin(v, [0, 1])
        y = v[valid].astype(np.float32)
        pos = float((y == 1).sum())
        neg = float((y == 0).sum())
        w = (neg / max(pos, 1.0))
        w = float(np.clip(w, 1.0, clamp_max))
        pos_w.append(w)

    return np.array(pos_w, dtype=np.float32)

def masked_bce_with_logits_posweight(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    BCEWithLogits (pos_weight) + mask for ignoring missing/uncertain labels.
    """
    denom = mask.sum()
    if float(denom.item()) < eps:
        return torch.zeros((), device=logits.device, requires_grad=True)

    t = targets.clamp(0.0, 1.0)
    per = F.binary_cross_entropy_with_logits(
        logits, t, reduction="none", pos_weight=pos_weight
    )
    per = per * mask
    return per.sum() / denom.clamp_min(eps)

def compute_auc_metrics(probs: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "per_label_auc": [float("nan")] * probs.shape[1],
        "macro_auc": float("nan"),
        "micro_auc": float("nan"),
    }
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        out["note"] = "sklearn not available; AUC not computed"
        return out

    K = probs.shape[1]
    per = []
    for k in range(K):
        v = mask[:, k] > 0.5
        if v.sum() < 2:
            per.append(float("nan"))
            continue
        yk = targets[v, k]
        pk = probs[v, k]
        if np.all(yk == 0) or np.all(yk == 1):
            per.append(float("nan"))
            continue
        try:
            per.append(float(roc_auc_score(yk, pk)))
        except Exception:
            per.append(float("nan"))

    out["per_label_auc"] = per
    valid = [x for x in per if np.isfinite(x)]
    if len(valid) > 0:
        out["macro_auc"] = float(np.mean(valid))

    vflat = mask.reshape(-1) > 0.5
    if vflat.sum() >= 2:
        yflat = targets.reshape(-1)[vflat]
        pflat = probs.reshape(-1)[vflat]
        if not (np.all(yflat == 0) or np.all(yflat == 1)):
            try:
                out["micro_auc"] = float(roc_auc_score(yflat, pflat))
            except Exception:
                out["micro_auc"] = float("nan")

    return out

def sanity_label_counts_np(y_np: np.ndarray, m_np: np.ndarray):
    print("\n=== SANITY: valid/pos/neg per label ===")
    for k, name in enumerate(LABEL_COLUMNS):
        v = m_np[:, k] > 0.5
        yk = y_np[v, k]
        n_valid = int(v.sum())
        pos = int((yk == 1).sum())
        neg = int((yk == 0).sum())
        print(f"{name:18s} n_valid={n_valid:7d} pos={pos:7d} neg={neg:7d}")
    print("======================================\n")

def load_baseline_into_mm_image_encoder(model: MultiModalFiLMGated, ckpt_path: str) -> Dict[str, Any]:
    """
    Optional: initialize MM image encoder from baseline ckpt.
    Supports:
      - baseline ckpt containing 'model_state_dict'
      - same shapes load directly
      - if baseline conv0 is 1ch and MM is 3ch, replicate 1->3
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model_state_dict" not in ckpt:
        raise KeyError(f"Expected 'model_state_dict' in baseline ckpt. Keys: {list(ckpt.keys())}")
    sd = ckpt["model_state_dict"]

    mm_state = model.image_encoder.state_dict()
    mm_keys = set(mm_state.keys())
    load_dict = {}

    for bk, bv in sd.items():
        candidates = [bk, "m." + bk]  # some codebases prefix with m.
        for mk in candidates:
            if mk not in mm_keys:
                continue

            # direct match
            if mm_state[mk].shape == bv.shape:
                load_dict[mk] = bv
                break

            # conv0 1ch -> 3ch replication
            if mk.endswith("features.conv0.weight"):
                mm_w = mm_state[mk]
                if tuple(bv.shape) == (64, 1, 7, 7) and tuple(mm_w.shape) == (64, 3, 7, 7):
                    load_dict[mk] = bv.repeat(1, 3, 1, 1) / 3.0
                    break

    missing, unexpected = model.image_encoder.load_state_dict(load_dict, strict=False)
    return {"loaded": len(load_dict), "missing": missing, "unexpected": unexpected}


# -----------------------------
# Eval
# -----------------------------
@torch.no_grad()
def eval_val(
    model: MultiModalFiLMGated,
    loader,
    device: torch.device,
    pos_weight_t: torch.Tensor,
) -> Dict[str, Any]:
    model.eval()
    losses = []
    all_probs, all_y, all_m = [], [], []

    for x, meta, y, m in tqdm(loader, desc="val", leave=False, dynamic_ncols=True):
        x = x.to(device, non_blocking=True)
        meta = meta.to(device, non_blocking=True, dtype=torch.float32)
        y = y.to(device, non_blocking=True, dtype=torch.float32)
        m = m.to(device, non_blocking=True, dtype=torch.float32)

        logits, gates = model(x, meta, mc_dropout=False)
        loss = masked_bce_with_logits_posweight(logits, y, m, pos_weight_t)
        losses.append(float(loss.item()))

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)
        all_y.append(y.detach().cpu().numpy())
        all_m.append(m.detach().cpu().numpy())

    val_loss = float(np.mean(losses)) if len(losses) else float("nan")

    if len(all_probs):
        probs_np = np.concatenate(all_probs, axis=0)
        y_np = np.concatenate(all_y, axis=0)
        m_np = np.concatenate(all_m, axis=0)
        sanity_label_counts_np(y_np, m_np)
        auc = compute_auc_metrics(probs_np, y_np, m_np)
    else:
        auc = {"per_label_auc": [float("nan")] * len(LABEL_COLUMNS),
               "macro_auc": float("nan"), "micro_auc": float("nan")}

    return {"val_loss": val_loss, "auc": auc}


# -----------------------------
# Main
# -----------------------------
def main():
    cfg = TrainCfg()

    train_csv = os.environ["TRAIN_CSV"]
    val_csv = os.environ["VAL_PERF_CSV"]

    cache_dir = os.environ.get("CACHE_DIR", "/ephemeral/ubuntu/mimic_cache")
    outdir = os.environ.get("OUTDIR", f"/ephemeral/ubuntu/results_mm/mm_run_{int(time.time())}")

    split_col = os.environ.get("SPLIT_COL", "split")
    train_split = os.environ.get("TRAIN_SPLIT", "train")
    val_split = os.environ.get("VAL_SPLIT", "validate")

    ensure_dir(outdir)
    ensure_dir(cache_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # dataloaders
    train_loader = create_dataloader_mm(
        csv_path=train_csv,
        split=train_split,
        batch_size=cfg.batch_size,
        shuffle=True,
        image_size=cfg.img_size,
        cache_dir=cache_dir,
        num_workers=cfg.num_workers,
        split_col=split_col,
    )
    val_loader = create_dataloader_mm(
        csv_path=val_csv,
        split=val_split,
        batch_size=cfg.batch_size,
        shuffle=False,
        image_size=cfg.img_size,
        cache_dir=cache_dir,
        num_workers=cfg.num_workers,
        split_col=split_col,
    )

    # pos_weight
    pos_w_np = compute_pos_weight_from_train(train_csv, split_col=split_col, train_split=train_split, clamp_max=30.0)
    pos_weight_t = torch.tensor(pos_w_np, device=device, dtype=torch.float32)
    print("✅ pos_weight:", pos_w_np.tolist())

    # model (meta dim must match dataloader: 11)
    model = MultiModalFiLMGated(num_labels=len(LABEL_COLUMNS)).to(device)

    # optional baseline init
    baseline_ckpt = os.environ.get("BASELINE_CKPT", "")
    if baseline_ckpt and os.path.exists(baseline_ckpt):
        info = load_baseline_into_mm_image_encoder(model, baseline_ckpt)
        print(f"✅ baseline init loaded from {baseline_ckpt} -> {info}")
    else:
        print("ℹ️ BASELINE_CKPT not provided/found. Using ImageNet init only.")

    # optimizer param groups
    densenet_params = model.image_encoder.parameters()
    head_params = (
        list(model.meta_encoder.parameters())
        + list(getattr(model, "film", []).parameters()) if hasattr(model, "film") else []
    )
    # If your model file does NOT include FiLM module explicitly (simplified),
    # just gather all non-backbone params like this:
    head_params = [p for n, p in model.named_parameters() if not n.startswith("image_encoder.")]

    opt = AdamW(
        [
            {"params": densenet_params, "lr": cfg.lr_densenet, "weight_decay": cfg.wd_densenet, "base_lr": cfg.lr_densenet},
            {"params": head_params, "lr": cfg.lr_heads, "weight_decay": cfg.wd_heads, "base_lr": cfg.lr_heads},
        ],
        betas=cfg.betas,
        eps=cfg.eps,
    )

    scaler = GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    # logs/checkpoints
    ckpt_path = os.path.join(outdir, "mm_film_best.pt")
    log_jsonl = os.path.join(outdir, "train_log.jsonl")
    log_csv = os.path.join(outdir, "train_log.csv")

    write_header = not os.path.exists(log_csv)
    best_val = float("inf")

    with open(log_jsonl, "a") as fj, open(log_csv, "a", newline="") as fc:
        w = csv.writer(fc)
        if write_header:
            w.writerow([
                "epoch","backbone_frozen",
                "train_loss","val_loss",
                "val_auc_macro","val_auc_micro",
                "lr_densenet","lr_heads",
                "lambda_gate",
            ])

        for epoch in range(cfg.max_epochs):
            backbone_frozen = bool(epoch < cfg.freeze_backbone_epochs)
            set_backbone_trainable(model, trainable=(not backbone_frozen))

            # warmup
            apply_warmup(opt, epoch, cfg)
            if backbone_frozen:
                set_backbone_lr(opt, 0.0)

            lam_gate = gate_lambda(epoch, cfg)

            model.train()
            losses = []

            pbar = tqdm(train_loader, desc=f"train ep{epoch}", leave=True, dynamic_ncols=True)
            for x, meta, y, m in pbar:
                x = x.to(device, non_blocking=True)
                meta = meta.to(device, non_blocking=True, dtype=torch.float32)
                y = y.to(device, non_blocking=True, dtype=torch.float32)
                m = m.to(device, non_blocking=True, dtype=torch.float32)

                opt.zero_grad(set_to_none=True)

                with autocast(device_type="cuda", enabled=(cfg.amp and device.type == "cuda")):
                    logits, gates = model(x, meta, mc_dropout=False)

                    loss_cls = masked_bce_with_logits_posweight(logits, y, m, pos_weight_t)
                    # keep gate reg minimal (optional)
                    loss_gate = ((gates - 0.5) ** 2).mean() if gates is not None else 0.0
                    loss = loss_cls + float(lam_gate) * loss_gate

                if not torch.isfinite(loss):
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()

                losses.append(float(loss.item()))
                pbar.set_postfix(
                    frozen=int(backbone_frozen),
                    loss=f"{loss.item():.4f}",
                    lr0=f"{opt.param_groups[0]['lr']:.1e}",
                    lr1=f"{opt.param_groups[1]['lr']:.1e}",
                )

            train_loss = float(np.mean(losses)) if len(losses) else float("nan")

            # val
            val_out = eval_val(model, val_loader, device, pos_weight_t)
            val_loss = float(val_out["val_loss"])
            auc = val_out["auc"]
            macro = float(auc.get("macro_auc", float("nan")))
            micro = float(auc.get("micro_auc", float("nan")))

            # save best by val_loss
            if np.isfinite(val_loss) and val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optim": opt.state_dict(),
                        "scaler": scaler.state_dict() if scaler is not None else None,
                        "epoch": int(epoch),
                        "val_loss": float(val_loss),
                        "cfg": cfg.__dict__,
                        "pos_weight": pos_w_np.tolist(),
                        "auc": auc,
                    },
                    ckpt_path,
                )

            log = {
                "epoch": int(epoch),
                "backbone_frozen": bool(backbone_frozen),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_auc_macro": macro,
                "val_auc_micro": micro,
                "lr_densenet": float(opt.param_groups[0]["lr"]),
                "lr_heads": float(opt.param_groups[1]["lr"]),
                "lambda_gate": float(lam_gate),
                "best_val_loss": float(best_val),
            }
            fj.write(json.dumps(log) + "\n")
            fj.flush()

            w.writerow([
                int(epoch), int(backbone_frozen),
                float(train_loss), float(val_loss),
                macro, micro,
                float(opt.param_groups[0]["lr"]), float(opt.param_groups[1]["lr"]),
                float(lam_gate),
            ])
            fc.flush()

            print(json.dumps(log))

    print(f"✅ Done. Best checkpoint: {ckpt_path}  (best val_loss={best_val:.4f})")


if __name__ == "__main__":
    main()
