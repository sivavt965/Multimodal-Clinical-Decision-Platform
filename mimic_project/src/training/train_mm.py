# src/training/train_mm.py
#
# Multimodal FiLM + label-wise gates trainer (with tqdm batch progress)
# + Loads baseline image encoder weights (from baseline_best.pt)
# + Freezes image encoder for first N epochs (here: 5) then unfreezes
# + ✅ Resume support (continues epochs instead of starting from 0)
#
# ENV REQUIRED:
#   TRAIN_CSV     (path to csv)
#   VAL_PERF_CSV  (path to csv)
#
# ENV OPTIONAL:
#   BASELINE_CKPT (path to baseline ckpt to init image encoder)
#   RESUME_CKPT   (path to resume MM ckpt; default: OUTDIR/mm_film_best.pt if exists)
#   CACHE_DIR     (default /ephemeral/ubuntu/mimic_cache)
#   OUTDIR        (default /ephemeral/ubuntu/results_mm/mm_run_<timestamp>)
#   SPLIT_COL     (default "split")
#   TRAIN_SPLIT   (default "train")
#   VAL_SPLIT     (default "validate")
#
import os
import csv
import json
import time
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List, Optional

import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_

from tqdm import tqdm

from src.data.dataloader_cloud_mm import create_dataloader_mm, LABEL_COLUMNS
from src.models.model_mm_film_gated import MultiModalFiLMGated


# ----------------------------
# CONFIG
# ----------------------------
@dataclass
class TrainCfg:
    # Data
    img_size: int = 512
    batch_size: int = 9         # ✅ set to 8 to avoid OOM
    num_workers: int = 10

    # Training
    max_epochs: int = 30
    amp: bool = True
    grad_clip: float = 1.0

    # ✅ Freeze image backbone for first N epochs
    freeze_backbone_epochs: int = 5

    # Optimizer groups
    lr_densenet: float = 1e-4
    lr_heads: float = 3e-4
    wd_densenet: float = 1e-4
    wd_heads: float = 1e-5
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    # Warmup (epoch 0 and 1), applies to BOTH groups (base_lr * factor)
    warmup_factors: Tuple[float, ...] = (0.33, 0.66)

    # Gate regularization schedule
    gate_lambda_init: float = 0.01
    gate_lambda_recover: float = 0.005
    gate_lambda_epochs: int = 5  # linear decay over first N epochs

    # Collapse detection + recovery
    collapse_hi: float = 0.95
    collapse_lo: float = 0.05
    collapse_std: float = 0.01
    collapse_min_label_samples: int = 100

    recover_epochs: int = 3
    recover_lr_factor: float = 0.3  # DenseNet base_lr *= factor when collapse detected

    # Optional: preferential masking over currently-valid labels (train only)
    prefer_mask_frac: float = 0.0  # set to 0.30 if you want; keep 0.0 if not desired


# ----------------------------
# LOSS
# ----------------------------
def masked_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    denom = valid_mask.sum()
    if float(denom.item()) < eps:
        return torch.zeros((), device=logits.device, requires_grad=True)
    safe_targets = targets.clamp(0.0, 1.0)
    per = F.binary_cross_entropy_with_logits(logits, safe_targets, reduction="none") * valid_mask
    return per.sum() / denom.clamp_min(eps)


# ----------------------------
# WARMUP
# ----------------------------
def warmup_factor(epoch: int, cfg: TrainCfg) -> float:
    return float(cfg.warmup_factors[epoch]) if epoch < len(cfg.warmup_factors) else 1.0


def apply_warmup(opt: torch.optim.Optimizer, epoch: int, cfg: TrainCfg):
    f = warmup_factor(epoch, cfg)
    for pg in opt.param_groups:
        pg["lr"] = pg["base_lr"] * f


# ----------------------------
# GATE LAMBDA
# ----------------------------
def gate_lambda(epoch: int, cfg: TrainCfg) -> float:
    L = max(1, int(cfg.gate_lambda_epochs))
    return cfg.gate_lambda_init * max(0.0, (L - epoch) / L)


# ----------------------------
# GATE STATS (Welford on batch means)
# ----------------------------
class GateStats:
    def __init__(self, K: int, device: torch.device):
        self.n = 0
        self.mean = torch.zeros(K, device=device)
        self.M2 = torch.zeros(K, device=device)

    def update(self, gates_bk: torch.Tensor):
        x = gates_bk.mean(dim=0)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)

    def finalize(self):
        var = self.M2 / max(self.n - 1, 1)
        return self.mean.detach().cpu(), var.sqrt().detach().cpu()


# ----------------------------
# CHANNEL SAFETY (defensive)
# ----------------------------
def maybe_match_channels_to_backbone(model: MultiModalFiLMGated, x: torch.Tensor) -> torch.Tensor:
    try:
        conv0 = model.image_encoder.features.conv0
        expected = int(conv0.in_channels)
    except Exception:
        return x

    if x.ndim != 4:
        return x
    c = int(x.shape[1])

    if expected == c:
        return x
    if expected == 3 and c == 1:
        return x.repeat(1, 3, 1, 1)
    if expected == 1 and c == 3:
        return x.mean(dim=1, keepdim=True)
    return x


# ----------------------------
# OPTIONAL PREFERENTIAL MASK (train-only)
# ----------------------------
def apply_preferential_mask(
    y: torch.Tensor, m: torch.Tensor, frac: float
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    if frac <= 0.0:
        return y, m, 0, int(m.sum().item())

    with torch.no_grad():
        eligible = (m > 0.5)
        eligible_count = int(eligible.sum().item())
        if eligible_count == 0:
            return y, m, 0, 0

        r = torch.rand_like(m)
        to_mask = eligible & (r < float(frac))

        m2 = m.clone()
        m2[to_mask] = 0.0

        masked_count = int(to_mask.sum().item())
        return y, m2, masked_count, eligible_count


# ----------------------------
# AUC (val)
# ----------------------------
def compute_auc_metrics(
    probs: np.ndarray, targets: np.ndarray, mask: np.ndarray
) -> Dict[str, Any]:
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
    per: List[float] = []
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


# ----------------------------
# EVAL (val_perf)
# ----------------------------
@torch.no_grad()
def eval_val_perf(model: MultiModalFiLMGated, loader, device: torch.device):
    model.eval()
    total_loss, n = 0.0, 0
    label_valid = torch.zeros(len(LABEL_COLUMNS), device=device)
    gs = GateStats(len(LABEL_COLUMNS), device=device)

    all_probs = []
    all_y = []
    all_m = []

    for x, meta, y, m in tqdm(loader, desc="val", leave=False, dynamic_ncols=True):
        x = x.to(device, non_blocking=True)
        meta = meta.to(device, non_blocking=True, dtype=torch.float32)
        y = y.to(device, non_blocking=True, dtype=torch.float32)
        m = m.to(device, non_blocking=True, dtype=torch.float32)

        x = maybe_match_channels_to_backbone(model, x)

        logits, gates = model(x, meta, mc_dropout=False)
        loss = masked_bce_with_logits(logits, y, m)

        total_loss += float(loss.item())
        n += 1
        label_valid += m.sum(dim=0)
        gs.update(gates)

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)
        all_y.append(y.detach().cpu().numpy())
        all_m.append(m.detach().cpu().numpy())

    gate_mean, gate_std = gs.finalize()

    if len(all_probs) > 0:
        probs_np = np.concatenate(all_probs, axis=0)
        y_np = np.concatenate(all_y, axis=0)
        m_np = np.concatenate(all_m, axis=0)
        auc = compute_auc_metrics(probs_np, y_np, m_np)
    else:
        auc = {
            "per_label_auc": [float("nan")] * len(LABEL_COLUMNS),
            "macro_auc": float("nan"),
            "micro_auc": float("nan"),
        }

    return (total_loss / max(n, 1)), gate_mean, gate_std, label_valid.cpu(), auc


def check_collapse(gate_mean, gate_std, label_valid, cfg: TrainCfg) -> bool:
    valid = label_valid > cfg.collapse_min_label_samples
    if int(valid.sum().item()) == 0:
        return False
    gm = gate_mean[valid]
    gs = gate_std[valid]
    all_high = bool((gm > cfg.collapse_hi).all())
    all_low = bool((gm < cfg.collapse_lo).all())
    all_flat = bool((gs < cfg.collapse_std).all())
    return all_high or all_low or all_flat


# ----------------------------
# FREEZE / UNFREEZE helpers
# ----------------------------
def set_backbone_trainable(model: MultiModalFiLMGated, trainable: bool):
    for p in model.image_encoder.parameters():
        p.requires_grad = trainable


def set_backbone_lr(opt: torch.optim.Optimizer, lr: float):
    # group 0 is image encoder
    opt.param_groups[0]["lr"] = float(lr)


# ----------------------------
# BASELINE LOADING (image encoder init)
# ----------------------------
def load_baseline_into_mm_image_encoder(model: MultiModalFiLMGated, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model_state_dict" not in ckpt:
        raise KeyError(f"Expected 'model_state_dict' in baseline ckpt, got keys: {list(ckpt.keys())}")
    sd = ckpt["model_state_dict"]

    mm_state = model.image_encoder.state_dict()
    mm_keys = set(mm_state.keys())

    load_dict = {}
    for bk, bv in sd.items():
        candidates = [bk, "m." + bk]  # handles features.* or m.features.*
        for mk in candidates:
            if mk in mm_keys and mm_state[mk].shape == bv.shape:
                load_dict[mk] = bv
                break

    missing, unexpected = model.image_encoder.load_state_dict(load_dict, strict=False)
    return {"loaded": len(load_dict), "missing": missing, "unexpected": unexpected}


# ----------------------------
# RESUME SUPPORT
# ----------------------------
def maybe_resume(
    model: MultiModalFiLMGated,
    opt: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    resume_path: str,
    device: torch.device,
):
    if (not resume_path) or (not os.path.exists(resume_path)):
        return 0, float("inf"), 0  # start_epoch, best_val, recovering

    ckpt = torch.load(resume_path, map_location=device)
    if "model" not in ckpt or "optim" not in ckpt:
        raise KeyError(f"Bad resume ckpt (need keys 'model','optim'), got: {list(ckpt.keys())}")

    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optim"])

    if scaler is not None and ckpt.get("scaler", None) is not None:
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception:
            pass

    start_epoch = int(ckpt.get("epoch", -1)) + 1
    best_val = float(ckpt.get("val_loss", float("inf")))
    recovering = int(ckpt.get("recovering_left", 0))

    print(f"✅ Resuming from {resume_path} at epoch {start_epoch} (best_val={best_val:.4f})")
    return start_epoch, best_val, recovering


# ----------------------------
# MAIN
# ----------------------------
def main():
    train_csv = os.environ["TRAIN_CSV"]
    val_csv = os.environ["VAL_PERF_CSV"]

    cache_dir = os.environ.get("CACHE_DIR", "/ephemeral/ubuntu/mimic_cache")
    outdir = os.environ.get("OUTDIR", "")
    if not outdir:
        outdir = f"/ephemeral/ubuntu/results_mm/mm_run_{int(time.time())}"

    split_col = os.environ.get("SPLIT_COL", "split")
    train_split = os.environ.get("TRAIN_SPLIT", "train")
    val_split = os.environ.get("VAL_SPLIT", "validate")

    os.makedirs(outdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    cfg = TrainCfg()

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

    assert len(train_loader) > 0, "Train loader is empty (check split_col/TRAIN_SPLIT)"
    assert len(val_loader) > 0, "Val loader is empty (check split_col/VAL_SPLIT)"

    model = MultiModalFiLMGated(num_labels=len(LABEL_COLUMNS)).to(device)

    baseline_ckpt = os.environ.get("BASELINE_CKPT", "")
    baseline_info = None
    if baseline_ckpt and os.path.exists(baseline_ckpt):
        baseline_info = load_baseline_into_mm_image_encoder(model, baseline_ckpt)
        print(f"✅ Loaded baseline image encoder init from: {baseline_ckpt} | {baseline_info}")
    else:
        print("⚠️ BASELINE_CKPT not set or not found; training MM from scratch/ImageNet init.")

    densenet_params = model.image_encoder.parameters()
    head_params = (
        list(model.meta_encoder.parameters())
        + list(model.film.parameters())
        + list(model.meta_to_img.parameters())
        + list(model.gates.parameters())
        + list(model.heads.parameters())
    )

    opt = AdamW(
        [
            {
                "params": densenet_params,
                "lr": cfg.lr_densenet,
                "weight_decay": cfg.wd_densenet,
                "base_lr": cfg.lr_densenet,
            },
            {
                "params": head_params,
                "lr": cfg.lr_heads,
                "weight_decay": cfg.wd_heads,
                "base_lr": cfg.lr_heads,
            },
        ],
        betas=cfg.betas,
        eps=cfg.eps,
    )

    scaler = GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    ckpt_path = os.path.join(outdir, "mm_film_best.pt")
    log_path = os.path.join(outdir, "train_log.jsonl")
    csv_log_path = os.path.join(outdir, "train_log.csv")

    # ✅ Resume (default: OUTDIR/mm_film_best.pt if exists)
    resume_path = os.environ.get("RESUME_CKPT", "")
    if not resume_path:
        resume_path = ckpt_path
    start_epoch, best_val, recovering = maybe_resume(model, opt, scaler, resume_path, device)

    write_header = not os.path.exists(csv_log_path)
    with open(csv_log_path, "a", newline="") as f_csv, open(log_path, "a") as f_jsonl:
        w = csv.writer(f_csv)
        if write_header:
            w.writerow([
                "epoch",
                "backbone_frozen",
                "train_loss_cls", "train_loss_total",
                "val_loss",
                "val_auc_macro", "val_auc_micro",
                "lr_densenet", "lr_heads",
                "base_lr_densenet", "base_lr_heads",
                "lambda_gate",
                "prefer_mask_frac", "train_masked_frac_eligible",
                "collapsed", "recovering_left",
            ])

        for epoch in range(start_epoch, cfg.max_epochs):
            backbone_frozen = bool(epoch < cfg.freeze_backbone_epochs)
            set_backbone_trainable(model, trainable=(not backbone_frozen))

            apply_warmup(opt, epoch, cfg)

            if backbone_frozen:
                set_backbone_lr(opt, 0.0)

            lam = cfg.gate_lambda_recover if recovering > 0 else gate_lambda(epoch, cfg)

            model.train()
            t0 = time.time()

            running_cls, running_total, n_batches = 0.0, 0.0, 0
            masked_total, eligible_total = 0, 0

            pbar = tqdm(train_loader, desc=f"train ep{epoch}", leave=True, dynamic_ncols=True)
            for x, meta, y, m in pbar:
                x = x.to(device, non_blocking=True)
                meta = meta.to(device, non_blocking=True, dtype=torch.float32)
                y = y.to(device, non_blocking=True, dtype=torch.float32)
                m = m.to(device, non_blocking=True, dtype=torch.float32)

                x = maybe_match_channels_to_backbone(model, x)

                y2, m2, masked_ct, elig_ct = apply_preferential_mask(y, m, cfg.prefer_mask_frac)
                masked_total += masked_ct
                eligible_total += elig_ct

                opt.zero_grad(set_to_none=True)

                with autocast(device_type="cuda", enabled=(cfg.amp and device.type == "cuda")):
                    logits, gates = model(x, meta, mc_dropout=False)
                    loss_cls = masked_bce_with_logits(logits, y2, m2)
                    loss_gate = ((gates - 0.5) ** 2).mean()
                    loss_total = loss_cls + float(lam) * loss_gate

                # ✅ guard against NaNs/Infs
                if not torch.isfinite(loss_total):
                    opt.zero_grad(set_to_none=True)
                    continue

                scaler.scale(loss_total).backward()
                scaler.unscale_(opt)
                clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()

                running_cls += float(loss_cls.item())
                running_total += float(loss_total.item())
                n_batches += 1

                pbar.set_postfix(
                    frozen=int(backbone_frozen),
                    cls=f"{loss_cls.item():.4f}",
                    total=f"{loss_total.item():.4f}",
                    lr0=f"{opt.param_groups[0]['lr']:.1e}",
                    lr1=f"{opt.param_groups[1]['lr']:.1e}",
                )

            train_loss_cls = running_cls / max(n_batches, 1)
            train_loss_total = running_total / max(n_batches, 1)
            train_masked_frac_eligible = float(masked_total / max(1, eligible_total))

            val_loss, gate_mean, gate_std, label_valid, auc = eval_val_perf(model, val_loader, device)
            collapsed = check_collapse(gate_mean, gate_std, label_valid, cfg)

            if (not backbone_frozen) and collapsed and recovering == 0:
                opt.param_groups[0]["base_lr"] *= cfg.recover_lr_factor
                f = warmup_factor(epoch, cfg)
                opt.param_groups[0]["lr"] = opt.param_groups[0]["base_lr"] * f
                opt.param_groups[1]["lr"] = opt.param_groups[1]["base_lr"] * f
                recovering = cfg.recover_epochs

            if recovering > 0:
                recovering -= 1

            if val_loss < best_val:
                best_val = float(val_loss)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optim": opt.state_dict(),
                        "scaler": scaler.state_dict() if scaler is not None else None,
                        "epoch": int(epoch),
                        "val_loss": float(val_loss),
                        "cfg": cfg.__dict__,
                        "gate_mean": gate_mean.tolist(),
                        "gate_std": gate_std.tolist(),
                        "label_valid_counts": label_valid.tolist(),
                        "auc": auc,
                        "baseline_init": baseline_info,
                        "recovering_left": int(recovering),
                    },
                    ckpt_path,
                )

            log = {
                "epoch": int(epoch),
                "backbone_frozen": bool(backbone_frozen),
                "train_loss_cls": float(train_loss_cls),
                "train_loss_total": float(train_loss_total),
                "val_loss": float(val_loss),
                "val_auc_macro": float(auc.get("macro_auc", float("nan"))),
                "val_auc_micro": float(auc.get("micro_auc", float("nan"))),
                "seconds": round(time.time() - t0, 1),
                "lr_densenet": float(opt.param_groups[0]["lr"]),
                "lr_heads": float(opt.param_groups[1]["lr"]),
                "base_lr_densenet": float(opt.param_groups[0]["base_lr"]),
                "base_lr_heads": float(opt.param_groups[1]["base_lr"]),
                "lambda_gate": float(lam),
                "prefer_mask_frac": float(cfg.prefer_mask_frac),
                "train_masked_frac_eligible": float(train_masked_frac_eligible),
                "collapsed": bool(collapsed),
                "recovering_left": int(recovering),
                "gate_mean": gate_mean.tolist(),
                "gate_std": gate_std.tolist(),
                "label_valid_counts": label_valid.tolist(),
                "per_label_auc": auc.get("per_label_auc", [float("nan")] * len(LABEL_COLUMNS)),
                "baseline_init": baseline_info,
                "paths": {
                    "train_csv": train_csv,
                    "val_csv": val_csv,
                    "cache_dir": cache_dir,
                    "outdir": outdir,
                    "baseline_ckpt": baseline_ckpt,
                    "resume_ckpt": resume_path,
                },
            }

            f_jsonl.write(json.dumps(log) + "\n")
# --- SAVE LAST CHECKPOINT EVERY EPOCH ---
last_path = os.path.join(outdir, "mm_film_last.pt")
torch.save(
    {
        "model": model.state_dict(),
        "optim": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "val_loss": float(val_loss),
        "auc": auc,
        "cfg": cfg.__dict__,
        "baseline_init": baseline_info,
    },
    last_path,
)
# ---------------------------------------
                       
            f_jsonl.flush()

            w.writerow([
                int(epoch),
                int(backbone_frozen),
                float(train_loss_cls), float(train_loss_total),
                float(val_loss),
                float(auc.get("macro_auc", float("nan"))),
                float(auc.get("micro_auc", float("nan"))),
                float(opt.param_groups[0]["lr"]), float(opt.param_groups[1]["lr"]),
                float(opt.param_groups[0]["base_lr"]), float(opt.param_groups[1]["base_lr"]),
                float(lam),
                float(cfg.prefer_mask_frac), float(train_masked_frac_eligible),
                int(collapsed), int(recovering),
            ])
            f_csv.flush()

            print(json.dumps({
                "epoch": int(epoch),
                "backbone_frozen": bool(backbone_frozen),
                "train_loss_cls": float(train_loss_cls),
                "train_loss_total": float(train_loss_total),
                "val_loss": float(val_loss),
                "val_auc_macro": float(auc.get("macro_auc", float("nan"))),
                "val_auc_micro": float(auc.get("micro_auc", float("nan"))),
                "seconds": round(time.time() - t0, 1),
                "collapsed": bool(collapsed),
                "recovering_left": int(recovering),
            }))

    print(f"✅ Best checkpoint saved: {ckpt_path} (best val_loss={best_val:.4f})")


if __name__ == "__main__":
    main()

