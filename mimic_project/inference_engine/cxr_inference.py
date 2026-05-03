#!/usr/bin/env python3
"""
cxr_inference.py — Phase B Core: CXR Inference Engine
=======================================================
Loads a DenseNet121-based CXR model (baseline_best.pt or any compatible
checkpoint), runs a forward pass on a single image (DICOM or PNG/JPG),
applies Temperature Scaling to convert raw logits into calibrated
probabilities, and returns the top-5 findings as a JSON-serialisable dict.

Usage (CLI):
    python cxr_inference.py --image path/to/image.png [--ckpt path/to/model.pt]
    python cxr_inference.py --image path/to/image.dcm --temperature 1.5

Usage (import):
    from inference_engine.cxr_inference import predict
    result = predict(image_path="path/to/cxr.png")

Temperature Scaling — Mathematical Background
---------------------------------------------
After training, neural networks are often over-confident: the raw sigmoid
output (sigmoid(logit)) is not a reliable probability estimate.

Temperature Scaling is the simplest post-hoc calibration method:

    p_calibrated = sigmoid(logit / T)

where T > 1 shrinks the logits, softening the distribution (less extreme
probabilities), and T < 1 sharpens it. T is learned on a held-out
validation set by minimising NLL:

    T* = argmin_T  1/N Σ BCE(sigmoid(logit_i / T), y_i)

Since the model checkpoint may not include a saved T, we default to T=1.0
(no-op, raw sigmoid) and allow the caller to pass a calibrated T.

Model weights:
    models/baseline_best.pt     — DenseNet121, 8-class, image-only
    models/baseline_best_old_state_dict.pt — older state-dict format

Notes:
    - DICOM support requires `pydicom` (`pip install pydicom`).
    - For production, use a per-class T vector learned on val set.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ──────────────────────────────────────────────
# Project-level imports (resolve from any CWD)
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

# Default checkpoint (baseline DenseNet121, image-only, 8 classes)
DEFAULT_CKPT: str = os.path.join(_PROJECT_ROOT, "models", "baseline_best.pt")

# Image size expected by the model during training
IMG_SIZE: int = 512


# ──────────────────────────────────────────────
# Image loading (PNG / JPG / DICOM)
# ──────────────────────────────────────────────
def _load_image_pil(image_path: str) -> Image.Image:
    """
    Load an image from disk as a PIL Image (RGB mode).

    Handles:
        - Standard raster formats (PNG, JPG, BMP, TIFF) — via Pillow
        - DICOM (.dcm) — via pydicom; pixel array is normalised to [0, 255]

    Args:
        image_path: Absolute or relative path to the image file.

    Returns:
        PIL.Image.Image in RGB mode.

    Raises:
        FileNotFoundError: If the path does not exist.
        ImportError: If pydicom is required but not installed.
        RuntimeError: If the file cannot be decoded.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"[cxr_inference] Image not found: {image_path}")

    ext = os.path.splitext(image_path)[1].lower()

    if ext == ".dcm":
        # ── DICOM branch ──────────────────────────────────────────────────
        try:
            import pydicom  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "pydicom is required for DICOM images. "
                "Install it with: pip install pydicom"
            ) from exc

        ds = pydicom.dcmread(image_path)
        arr = ds.pixel_array.astype(np.float32)

        # Rescale to [0, 255]
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
        else:
            arr = np.zeros_like(arr)

        img = Image.fromarray(arr.astype(np.uint8))
        # DICOM CXRs are usually grayscale — convert to RGB by replication
        img = img.convert("RGB")

    else:
        # ── Standard PIL branch ───────────────────────────────────────────
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(
                f"[cxr_inference] Failed to open image '{image_path}': {exc}"
            ) from exc

    return img


def _preprocess(img: Image.Image, img_size: int = IMG_SIZE) -> torch.Tensor:
    """
    Apply the same preprocessing pipeline used during training:
        1. Resize to (img_size × img_size)
        2. Convert to float32 tensor in [0, 1]  (divide by 255)
        3. Add batch dimension → [1, 3, H, W]

    Note: We deliberately do NOT apply ImageNet mean/std normalisation
    because the training dataloader used mean=(0,0,0), std=(1,1,1)
    (i.e., simple [0,1] scaling). See dataloader_cloud_mm.py line 191.

    Args:
        img: RGB PIL Image.
        img_size: Target square resolution.

    Returns:
        Float tensor of shape [1, 3, img_size, img_size].
    """
    img = img.resize((img_size, img_size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0          # [H, W, 3] in [0,1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1)         # [3, H, W]
    return tensor.unsqueeze(0)                               # [1, 3, H, W]


# ──────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────
def _load_model(
    ckpt_path: str,
    device: torch.device,
    num_classes: int = 8,
) -> nn.Module:
    """
    Load a DenseNet121 model from a checkpoint file.

    Supports both checkpoint formats found in this project:
        • ``{"model_state_dict": state_dict, ...}``  — baseline_best.pt
        • Raw state_dict without a wrapping key

    Prefix stripping removes "module." and "model." prefixes that arise
    from DataParallel / custom wrapper training.

    Args:
        ckpt_path: Path to the .pt checkpoint file.
        device: Target device (cpu / cuda).
        num_classes: Number of output classes (default 8).

    Returns:
        Loaded, eval-mode nn.Module.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"[cxr_inference] Checkpoint not found: {ckpt_path}")

    print(f"[cxr_inference] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Unwrap checkpoint dict if necessary
    if isinstance(ckpt, dict):
        sd = ckpt.get(
            "model_state_dict",
            ckpt.get("model", ckpt),  # "model" key is used by mm trainer
        )
    else:
        sd = ckpt

    # Strip common prefixes from DataParallel / wrapper training
    cleaned: Dict[str, Any] = {}
    for k, v in sd.items():
        k2 = k
        for prefix in ("module.", "model."):
            if k2.startswith(prefix):
                k2 = k2[len(prefix):]
        cleaned[k2] = v

    model = build_densenet121(num_classes=num_classes, pretrained=False, dropout_p=0.3)
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(
        f"[cxr_inference] State dict loaded — "
        f"missing: {len(missing)}, unexpected: {len(unexpected)}"
    )
    model.to(device)
    model.eval()
    return model


# ──────────────────────────────────────────────
# Temperature Scaling
# ──────────────────────────────────────────────
def temperature_scale(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Apply Temperature Scaling calibration to raw logits.

    Mathematical derivation:
    ------------------------
    Standard sigmoid:
        p = σ(z) = 1 / (1 + exp(-z))

    Temperature-scaled sigmoid:
        p_cal = σ(z / T) = 1 / (1 + exp(-z/T))

    Effect of T on the probability distribution:
        • T = 1.0  → no-op, raw model output
        • T > 1.0  → sharpens confidence toward 0.5 (less extreme)
                      useful when model is over-confident
        • T < 1.0  → pushes probabilities toward 0 or 1 (more extreme)
                      rarely used; only if model is under-confident

    T is typically optimised on a validation set by minimising NLL loss.
    For this project the training logs show the model used BCE without
    temperature, so T=1.3 is a commonly cited default for DenseNet CXR
    models. Callers can pass the exact T found from calibration curves.

    Args:
        logits: Raw logit tensor of shape [B, C].
        temperature: Positive float. T=1.0 → identity.

    Returns:
        Calibrated probability tensor of shape [B, C] in (0, 1).
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    return torch.sigmoid(logits / temperature)


# ──────────────────────────────────────────────
# Core inference function
# ──────────────────────────────────────────────
def predict(
    image_path: str,
    ckpt_path: str = DEFAULT_CKPT,
    temperature: float = 1.0,
    top_k: int = 5,
    device_str: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run CXR inference on a single image and return calibrated findings.

    Pipeline:
        1. Load image (DICOM or raster)
        2. Preprocess (resize, normalise)
        3. Load DenseNet121 checkpoint
        4. Forward pass under torch.inference_mode()
        5. Apply Temperature Scaling to logits
        6. Sort by probability, return top-k findings

    Args:
        image_path: Path to CXR image (.dcm / .png / .jpg).
        ckpt_path: Path to model checkpoint (.pt).
        temperature: Temperature T for calibration (default 1.0 = no-op).
        top_k: Number of top findings to return (default 5).
        device_str: "cpu" or "cuda". Auto-detected if None.

    Returns:
        JSON-serialisable dict:
        {
            "image_path": str,
            "temperature": float,
            "device": str,
            "top_findings": [
                {"rank": 1, "label": "Pleural Effusion", "probability_pct": 78.3},
                ...
            ],
            "all_findings": {"Cardiomegaly": 23.1, ...},
            "status": "ok"
        }

    Raises:
        FileNotFoundError: If image or checkpoint is missing.
        RuntimeError: On CUDA OOM or model-load failure.
    """
    # ── Device selection ────────────────────────────────────────────────
    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    result: Dict[str, Any] = {
        "image_path": str(image_path),
        "temperature": float(temperature),
        "device": str(device),
        "status": "ok",
    }

    try:
        # ── 1. Load image ───────────────────────────────────────────────
        img = _load_image_pil(image_path)
        x = _preprocess(img).to(device)   # [1, 3, 512, 512]

        # ── 2. Load model ───────────────────────────────────────────────
        model = _load_model(ckpt_path, device=device)

        # ── 3. Forward pass (no gradient tracking) ──────────────────────
        # torch.inference_mode() is faster than no_grad(): it disables
        # both grad tracking AND the view tracking of autograd.
        try:
            with torch.inference_mode():
                logits: torch.Tensor = model(x)   # [1, 8]
        except torch.cuda.OutOfMemoryError as oom:
            torch.cuda.empty_cache()
            raise RuntimeError(
                f"[cxr_inference] CUDA Out-of-Memory during forward pass. "
                f"Try running on CPU with --device cpu. Details: {oom}"
            ) from oom

        # ── 4. Temperature Scaling ──────────────────────────────────────
        probs: np.ndarray = (
            temperature_scale(logits, temperature)
            .squeeze(0)          # [8]
            .cpu()
            .numpy()
        )

        # ── 5. Build output ─────────────────────────────────────────────
        all_findings: Dict[str, float] = {
            label: round(float(prob) * 100, 2)
            for label, prob in zip(LABEL_COLUMNS, probs)
        }

        # Sort descending by probability, take top_k
        sorted_items: List[Tuple[str, float]] = sorted(
            all_findings.items(), key=lambda kv: kv[1], reverse=True
        )
        top_findings = [
            {"rank": i + 1, "label": label, "probability_pct": pct}
            for i, (label, pct) in enumerate(sorted_items[:top_k])
        ]

        result["top_findings"] = top_findings
        result["all_findings"] = all_findings

    except FileNotFoundError as exc:
        result["status"] = "error"
        result["error"] = str(exc)
    except RuntimeError as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        torch.cuda.empty_cache()
    finally:
        # Aggressive memory cleanup: free model and CUDA cache
        try:
            del model
        except NameError:
            pass
        torch.cuda.empty_cache()

    return result


# ──────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CXR Inference — calibrated multi-label CXR prediction"
    )
    p.add_argument("--image", required=True, help="Path to CXR image (.dcm/.png/.jpg)")
    p.add_argument("--ckpt", default=DEFAULT_CKPT, help="Path to model checkpoint (.pt)")
    p.add_argument(
        "--temperature", type=float, default=1.0,
        help="Temperature scaling factor T (default: 1.0 = no-op)"
    )
    p.add_argument("--top_k", type=int, default=5, help="Number of top findings to display")
    p.add_argument(
        "--device", default=None,
        help="Compute device: 'cpu' or 'cuda' (auto-detected if omitted)"
    )
    return p


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    output = predict(
        image_path=args.image,
        ckpt_path=args.ckpt,
        temperature=args.temperature,
        top_k=args.top_k,
        device_str=args.device,
    )

    print(json.dumps(output, indent=2))
