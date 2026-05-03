#!/usr/bin/env python3
"""
mc_dropout.py — Uncertainty Engine via Monte Carlo Dropout
===========================================================
Implements MC Dropout (Gal & Ghahramani, 2016) to quantify predictive
uncertainty for a single CXR image.  The model is kept in eval() mode so
that BatchNorm layers use population statistics, but Dropout is explicitly
enabled via the mc_dropout=True flag supported by the project's DenseNet
implementation.

Usage (CLI):
    python mc_dropout.py --image path/to/image.png
    python mc_dropout.py --image path/to/image.png --n_passes 20 --ckpt models/baseline_best.pt

Usage (import):
    from inference_engine.mc_dropout import quantify_uncertainty
    result = quantify_uncertainty(image_path="path/to/cxr.png")

MC Dropout — Mathematical Background
--------------------------------------
Classic Dropout (Srivastava et al., 2014) zeros activations at random
during training. At test time, all neurons are typically active and weights
are scaled. MC Dropout (Gal & Ghahramani, 2016) keeps Dropout ACTIVE at
test time and runs T stochastic forward passes to approximate the posterior
predictive distribution:

    p(y | x, D) ≈ (1/T) Σ_{t=1}^{T}  p(y | x, ω_t)
                                          ̂ω_t ~ q(ω)   [dropout distribution]

This gives us two useful statistics per class k:

    Mean probability:   μ_k = (1/T) Σ_t σ(z_t,k)
    Variance:           σ²_k = (1/T) Σ_t [σ(z_t,k) - μ_k]²

High variance → high uncertainty → the model is "confused" about this
example.  This is particularly useful in clinical settings to flag cases
that need expert review.

Categorical uncertainty thresholds (empirical, tuned for 20 passes):
    • Mean variance < 0.005   → "Low Uncertainty"
    • Mean variance < 0.015   → "Moderate Uncertainty"
    • Mean variance ≥ 0.015   → "High Uncertainty"

These can be recalibrated on your validation set.

Notes on the DenseNet used here:
    - The project has two DenseNet variants:
        a) `densenet121.py`  — uses nn.Dropout layers (classic)
        b) `model_mm_film_gated.py` — uses F.dropout with mc_dropout flag

    - For the image-only baseline (baseline_best.pt), we use (a) and
      enable dropout by iterating through nn.Dropout modules and setting
      `training=True` via a context manager, so BN remains in eval mode.

    - For the multimodal model (mm_film_best.pt), we use (b) and call
      model.forward(x_img, x_meta, mc_dropout=True) directly.
"""

import argparse
import json
import os
import sys
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# ──────────────────────────────────────────────
# Project-level imports
# ──────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.models.densenet121 import build_densenet121  # noqa: E402

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
LABEL_COLUMNS: List[str] = [
    "Cardiomegaly",
    "Pleural Effusion",
    "Edema",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Consolidation",
    "Support Devices",
]

DEFAULT_CKPT: str = os.path.join(_PROJECT_ROOT, "models", "baseline_best.pt")
IMG_SIZE: int = 512
DEFAULT_N_PASSES: int = 20
DEFAULT_SEED: Optional[int] = 42   # Set None for true stochastic behaviour

# Variance thresholds for uncertainty categorisation
# Derived empirically; recalibrate on your val set if needed.
_VAR_MODERATE_THRESHOLD: float = 0.005
_VAR_HIGH_THRESHOLD: float = 0.015


# ──────────────────────────────────────────────
# Context Manager: Enable Dropout in Eval Mode
# ──────────────────────────────────────────────
@contextmanager
def enable_dropout(model: nn.Module) -> Generator[None, None, None]:
    """
    Context manager that temporarily switches all nn.Dropout (and
    nn.Dropout2d) layers to training mode while keeping the rest of the
    model in eval mode.

    This is the canonical way to implement MC Dropout when the model uses
    nn.Dropout rather than F.dropout(training=True):

        with enable_dropout(model):
            output = model(x)   # Dropout is active; BN uses pop. stats.

    Why not just call model.train()?
        model.train() would ALSO switch BatchNorm to training mode, making
        it use batch statistics instead of population statistics — this
        would give unreliable outputs for single-sample inference.

    Implementation detail:
        We collect references to Dropout modules BEFORE the context, set
        their .training attribute directly, then restore it on exit.
        This is thread-safe for single-GPU use.

    Args:
        model: The nn.Module to operate on.

    Yields:
        None (acts as a with-block).
    """
    # Collect all Dropout modules
    dropout_modules: List[nn.Module] = [
        m for m in model.modules()
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.AlphaDropout))
    ]

    # Enable dropout
    for m in dropout_modules:
        m.training = True

    try:
        yield
    finally:
        # Restore dropout to eval mode
        for m in dropout_modules:
            m.training = False


# ──────────────────────────────────────────────
# Image loading & preprocessing
# ──────────────────────────────────────────────
def _load_and_preprocess(image_path: str, device: torch.device) -> torch.Tensor:
    """
    Load image from disk and return a preprocessed tensor on `device`.

    Handles:
        - DICOM (.dcm): normalises pixel_array to [0, 1], converts to RGB
        - Standard rasters: PIL open → RGB

    Args:
        image_path: Path to the image.
        device: Target compute device.

    Returns:
        Float tensor of shape [1, 3, 512, 512] on `device`.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"[mc_dropout] Image not found: {image_path}")

    ext = os.path.splitext(image_path)[1].lower()

    if ext == ".dcm":
        try:
            import pydicom  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "pydicom required for DICOM. Install: pip install pydicom"
            ) from exc
        ds = pydicom.dcmread(image_path)
        arr = ds.pixel_array.astype(np.float32)
        if arr.max() > arr.min():
            arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255.0
        img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    else:
        img = Image.open(image_path).convert("RGB")

    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    return tensor.to(device)


# ──────────────────────────────────────────────
# Model loading (reusable; mirrors cxr_inference.py)
# ──────────────────────────────────────────────
def _load_model(ckpt_path: str, device: torch.device) -> nn.Module:
    """
    Load DenseNet121 from checkpoint (baseline_best.pt format).

    Args:
        ckpt_path: Path to .pt file.
        device: Target device.

    Returns:
        eval-mode DenseNet121 nn.Module.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"[mc_dropout] Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        sd = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    else:
        sd = ckpt

    cleaned: Dict[str, Any] = {}
    for k, v in sd.items():
        k2 = k
        for prefix in ("module.", "model."):
            if k2.startswith(prefix):
                k2 = k2[len(prefix):]
        cleaned[k2] = v

    model = build_densenet121(num_classes=8, pretrained=False, dropout_p=0.3)
    model.load_state_dict(cleaned, strict=False)
    model.to(device)
    model.eval()  # BN in eval mode — DO NOT call model.train()
    return model


# ──────────────────────────────────────────────
# Uncertainty categorisation
# ──────────────────────────────────────────────
def _categorise_uncertainty(mean_variance: float) -> str:
    """
    Map a scalar mean-variance value to a clinical uncertainty category.

    Thresholds (calibrated for T=20 passes, dropout_p=0.3):
        < 0.005  → "Low Uncertainty"       (safe to act on prediction)
        < 0.015  → "Moderate Uncertainty"  (consider second opinion)
        ≥ 0.015  → "High Uncertainty"      (flag for expert review)

    Args:
        mean_variance: Mean per-class variance averaged over all 8 labels.

    Returns:
        Categorical string: one of "Low", "Moderate", or "High" Uncertainty.
    """
    if mean_variance < _VAR_MODERATE_THRESHOLD:
        return "Low Uncertainty"
    elif mean_variance < _VAR_HIGH_THRESHOLD:
        return "Moderate Uncertainty"
    else:
        return "High Uncertainty"


# ──────────────────────────────────────────────
# Core function
# ──────────────────────────────────────────────
def quantify_uncertainty(
    image_path: str,
    ckpt_path: str = DEFAULT_CKPT,
    n_passes: int = DEFAULT_N_PASSES,
    device_str: Optional[str] = None,
    seed: Optional[int] = DEFAULT_SEED,
) -> Dict[str, Any]:
    """
    Run MC Dropout to estimate predictive uncertainty for a CXR image.

    Algorithm:
        1. Load model and set to eval() mode (BN uses population stats).
        2. Use `enable_dropout()` context manager to activate Dropout.
        3. Run `n_passes` stochastic forward passes on the same input.
        4. Collect sigmoid probabilities from each pass: shape [T, 8].
        5. Compute per-class mean (μ_k) and variance (σ²_k) across T passes.
        6. Mean variance over all classes = scalar uncertainty score.
        7. Categorise: Low / Moderate / High.

    Args:
        image_path: Path to CXR image.
        ckpt_path: Path to DenseNet121 checkpoint.
        n_passes: Number of stochastic forward passes (T). Default: 20.
        device_str: "cpu" or "cuda". Auto-detected if None.
        seed: Integer random seed for reproducibility. Set None for true
              stochastic behaviour (production). Default: 42 (debug-safe).
              Seeds both torch and numpy RNGs so variance scores are
              identical across runs on the same hardware.

    Returns:
        JSON-serialisable dict:
        {
            "image_path": str,
            "n_passes": int,
            "uncertainty_level": "Low Uncertainty" | "Moderate Uncertainty" | "High Uncertainty",
            "mean_variance": float,           # averaged over all 8 classes
            "per_class_variance": {           # σ² for each finding
                "Cardiomegaly": float, ...
            },
            "per_class_mean_prob": {          # μ for each finding (%)
                "Cardiomegaly": float, ...
            },
            "std_dev": float,                 # √(mean_variance) for interpretability
            "status": "ok" | "error"
        }
    """
    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    # ── Reproducibility seed ────────────────────────────────────────────
    # MC Dropout is stochastic by design; seeding makes variance scores
    # identical across debug runs without changing the statistical meaning.
    # In production, pass seed=None to get genuine stochastic uncertainty.
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)   # safe no-op if CUDA unavailable
        np.random.seed(seed)

    result: Dict[str, Any] = {
        "image_path": str(image_path),
        "n_passes": n_passes,
        "device": str(device),
        "seed": seed,
        "status": "ok",
    }

    try:
        # ── Load inputs ─────────────────────────────────────────────────
        x = _load_and_preprocess(image_path, device)   # [1, 3, 512, 512]
        model = _load_model(ckpt_path, device)

        # ── MC Dropout loop ─────────────────────────────────────────────
        # probs_stack will be [T, 8] after the loop
        probs_stack: List[np.ndarray] = []

        # We use `enable_dropout` so BN stays in eval mode while
        # Dropout is stochastically active — this is the correct
        # implementation of MC Dropout at inference time.
        with enable_dropout(model):
            for pass_idx in range(n_passes):
                try:
                    # We need gradients to be OFF but Dropout ON.
                    # torch.no_grad() is correct here (NOT inference_mode,
                    # which can block dropout's random sampling in some
                    # PyTorch versions if tensor views are involved).
                    with torch.no_grad():
                        logits = model(x)                 # [1, 8]
                        probs_t = torch.sigmoid(logits)   # [1, 8]
                    probs_stack.append(
                        probs_t.squeeze(0).cpu().numpy()  # [8]
                    )
                except torch.cuda.OutOfMemoryError as oom:
                    torch.cuda.empty_cache()
                    raise RuntimeError(
                        f"[mc_dropout] CUDA OOM on pass {pass_idx + 1}/{n_passes}. "
                        f"Try --device cpu. Details: {oom}"
                    ) from oom

        # ── Statistics ──────────────────────────────────────────────────
        # probs_all: shape [T, 8]
        probs_all = np.stack(probs_stack, axis=0)   # [T, 8]

        mu: np.ndarray = probs_all.mean(axis=0)      # [8]
        var: np.ndarray = probs_all.var(axis=0)      # [8]  σ²

        mean_variance: float = float(var.mean())     # scalar
        std_dev: float = float(np.sqrt(mean_variance))

        uncertainty_level: str = _categorise_uncertainty(mean_variance)

        # ── Package output ──────────────────────────────────────────────
        result["uncertainty_level"] = uncertainty_level
        result["mean_variance"] = round(mean_variance, 6)
        result["std_dev"] = round(std_dev, 6)
        result["per_class_variance"] = {
            label: round(float(v), 6)
            for label, v in zip(LABEL_COLUMNS, var)
        }
        result["per_class_mean_prob"] = {
            label: round(float(m) * 100, 2)
            for label, m in zip(LABEL_COLUMNS, mu)
        }

    except FileNotFoundError as exc:
        result["status"] = "error"
        result["error"] = str(exc)
    except RuntimeError as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        torch.cuda.empty_cache()
    finally:
        try:
            del model
        except NameError:
            pass
        torch.cuda.empty_cache()

    return result


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MC Dropout Uncertainty Engine for CXR predictions"
    )
    p.add_argument("--image", required=True, help="Path to CXR image")
    p.add_argument("--ckpt", default=DEFAULT_CKPT, help="Path to model checkpoint")
    p.add_argument(
        "--n_passes", type=int, default=DEFAULT_N_PASSES,
        help=f"Number of stochastic forward passes (default: {DEFAULT_N_PASSES})"
    )
    p.add_argument("--device", default=None, help="'cpu' or 'cuda'")
    p.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=(
            f"Random seed for reproducibility (default: {DEFAULT_SEED}). "
            "Pass -1 to disable seeding (true stochastic mode)."
        )
    )
    return p


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    seed = None if args.seed == -1 else args.seed
    output = quantify_uncertainty(
        image_path=args.image,
        ckpt_path=args.ckpt,
        n_passes=args.n_passes,
        device_str=args.device,
        seed=seed,
    )

    print(json.dumps(output, indent=2))
