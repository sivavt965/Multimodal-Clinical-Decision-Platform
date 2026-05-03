#!/usr/bin/env python3
"""
gradcam_gen.py — Visual Explanation via Grad-CAM
=================================================
Generates a Grad-CAM saliency overlay for any target class from the
DenseNet121 CXR model, and blends it with the original grayscale image
so the underlying anatomy remains fully visible.

Usage (CLI):
    python gradcam_gen.py --image path/to/image.png --class_idx 7
    python gradcam_gen.py --image path/to/image.dcm --class_idx 1 --out my_overlay.png

    Class indices (8-label model):
        0: Cardiomegaly        4: Atelectasis
        1: Pleural Effusion    5: Pneumothorax
        2: Edema               6: Consolidation
        3: Pneumonia           7: Support Devices

Usage (import):
    from inference_engine.gradcam_gen import generate_gradcam
    result = generate_gradcam(image_path="cxr.png", class_idx=7)

Grad-CAM — Mathematical Background
------------------------------------
Selvaraju et al. (2017), "Grad-CAM: Visual Explanations from Deep
Networks via Gradient-based Localisation"

For a target class c and a convolutional feature map A^k of shape [H, W]
at the final conv layer (denseblock4 in DenseNet121):

1. Compute gradients of the class score S^c w.r.t. each feature map A^k:
       ∂S^c / ∂A^k_{ij}

2. Global Average Pool the gradients (importance weight):
       α^c_k = (1/Z) Σ_{i,j} ∂S^c / ∂A^k_{ij}
   where Z = H × W.

3. Compute the weighted sum of feature maps, ReLU to keep only
   positive-influence activations:
       L^c_{Grad-CAM} = ReLU( Σ_k  α^c_k · A^k )

4. Upsample L^c to the input resolution (512×512).

5. Normalise to [0, 1], apply COLORMAP_JET via OpenCV.

6. Alpha-blend the heatmap with the original grayscale image using the
   "mix-blend" approach so anatomy remains visible:

       overlay = α · heatmap_rgb + (1 - α) · original_rgb

   where α = 0.45 (heatmap weight). The value 0.45 was chosen so that
   the anatomical structures remain clear while the activation region is
   distinguishable even on a printed black-and-white printout.

Notes on hook placement:
    The hook is registered on `model.features.denseblock4` — the last
    dense block before the global average pooling. This is the correct
    layer for DenseNet121 Grad-CAM (hooking norm5/relu would give the
    same spatial maps but would require working around inplace ReLU
    gradient issues; denseblock4 output avoids this cleanly).
"""

import argparse
import os
import sys
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
LABEL_COLUMNS = [
    "Cardiomegaly", "Pleural Effusion", "Edema", "Pneumonia",
    "Atelectasis", "Pneumothorax", "Consolidation", "Support Devices",
]
DEFAULT_CKPT: str = os.path.join(_PROJECT_ROOT, "models", "baseline_best.pt")
IMG_SIZE: int = 512

# Alpha blend weight: heatmap contributes 45%, original image 55%
# Keeping anatomy visible is critical for clinical use.
HEATMAP_ALPHA: float = 0.45

# Default output path (relative to CWD)
DEFAULT_OUTPUT: str = "gradcam_output.png"


# ──────────────────────────────────────────────
# Grad-CAM hook class
# ──────────────────────────────────────────────
class GradCAMHook:
    """
    Registers forward and backward hooks on a target layer to capture:
        • `activations` — feature maps output by the layer (forward pass)
        • `gradients`   — gradients of the loss w.r.t. those feature maps
                          (backward pass)

    These are used to compute the Grad-CAM importance weights:
        α^c_k = GAP(∂S^c / ∂A^k)

    Usage:
        hook = GradCAMHook(target_layer)
        output = model(x)                   # triggers forward hook
        output[0, class_idx].backward()     # triggers backward hook
        cam = hook.compute_cam()            # builds the heatmap
        hook.remove()                       # clean up hooks
    """

    def __init__(self, target_layer: nn.Module) -> None:
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(
        self,
        module: nn.Module,
        inp: Tuple,
        out: torch.Tensor,
    ) -> None:
        """Called automatically after each forward pass through the target layer."""
        # Detach to avoid accumulating computation graph in memory
        self.activations = out.detach()

    def _save_gradient(
        self,
        module: nn.Module,
        grad_input: Tuple,
        grad_output: Tuple,
    ) -> None:
        """Called automatically during backprop through the target layer."""
        self.gradients = grad_output[0].detach()

    def compute_cam(self) -> np.ndarray:
        """
        Compute the Grad-CAM heatmap from captured activations and gradients.

        Steps:
            1. Global-average-pool the gradients → per-channel weight α^c_k
               Shape: [1, K, H, W] → GAP → [1, K, 1, 1]

            2. Weighted sum of feature maps A^k:
               L = Σ_k  α_k · A^k   (element-wise broadcast)
               Squeeze spatial dims: shape → [H, W]

            3. ReLU: keep only positively-activating features
               L_relu = max(0, L)

            4. Normalise to [0, 1]

        Returns:
            np.ndarray of shape [H, W], dtype float32, values in [0, 1].
        """
        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "[GradCAMHook] No activations/gradients captured. "
                "Did you run a forward+backward pass?"
            )

        # activations: [1, K, H, W]
        # gradients:   [1, K, H, W]
        acts = self.activations          # [1, K, H, W]
        grads = self.gradients           # [1, K, H, W]

        # Step 1: Global Average Pooling of gradients → α^c_k
        # α shape: [1, K, 1, 1] — one scalar weight per feature channel
        alpha = grads.mean(dim=(2, 3), keepdim=True)   # [1, K, 1, 1]

        # Step 2: Weighted combination of feature maps
        # Broadcast alpha over spatial dims, sum over channel dim K
        cam = (alpha * acts).sum(dim=1, keepdim=False)  # [1, H, W]
        cam = cam.squeeze(0)                              # [H, W]

        # Step 3: ReLU — only keep features that contribute positively
        cam = F.relu(cam)

        # Step 4: Normalise to [0, 1]
        cam_np = cam.cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np.astype(np.float32)

    def remove(self) -> None:
        """Remove hooks to prevent memory leaks."""
        self._fwd_handle.remove()
        self._bwd_handle.remove()


# ──────────────────────────────────────────────
# Image loading & preprocessing
# ──────────────────────────────────────────────
def _load_image(image_path: str) -> Tuple[np.ndarray, torch.Tensor]:
    """
    Load image from disk, returning:
        - `orig_rgb`: uint8 numpy array [H, W, 3] at ORIGINAL resolution
                      (used as the base for the blend overlay)
        - `tensor`:   float32 tensor [1, 3, 512, 512] for model input

    Args:
        image_path: Path to image (.dcm / .png / .jpg).

    Returns:
        (orig_rgb, tensor)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"[gradcam_gen] Image not found: {image_path}")

    ext = os.path.splitext(image_path)[1].lower()

    if ext == ".dcm":
        try:
            import pydicom  # type: ignore
        except ImportError as exc:
            raise ImportError("pip install pydicom required for DICOM") from exc
        ds = pydicom.dcmread(image_path)
        arr = ds.pixel_array.astype(np.float32)
        if arr.max() > arr.min():
            arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255.0
        img_pil = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    else:
        img_pil = Image.open(image_path).convert("RGB")

    # Preserve original for overlay
    orig_rgb = np.array(img_pil, dtype=np.uint8)   # [H_orig, W_orig, 3]

    # Resize and normalise for model input
    img_resized = img_pil.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr_norm = np.array(img_resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr_norm).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]

    return orig_rgb, tensor


# ──────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────
def _load_model_for_gradcam(ckpt_path: str) -> nn.Module:
    """
    Load DenseNet121 for Grad-CAM.

    Unlike pure inference, Grad-CAM REQUIRES gradients to flow, so:
        - torch.inference_mode() is NOT used (it disables grad tracking)
        - The model is kept in .eval() for BN stability
        - Input tensor will have requires_grad=True enabled by the hook
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"[gradcam_gen] Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        sd = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    else:
        sd = ckpt

    cleaned = {}
    for k, v in sd.items():
        k2 = k
        for pfx in ("module.", "model."):
            if k2.startswith(pfx):
                k2 = k2[len(pfx):]
        cleaned[k2] = v

    model = build_densenet121(num_classes=8, pretrained=False, dropout_p=0.0)
    model.load_state_dict(cleaned, strict=False)
    model.eval()   # BN uses population statistics
    return model


# ──────────────────────────────────────────────
# Heatmap overlay blending
# ──────────────────────────────────────────────
def _blend_heatmap(
    cam_np: np.ndarray,
    orig_rgb: np.ndarray,
    alpha: float = HEATMAP_ALPHA,
) -> np.ndarray:
    """
    Apply COLORMAP_JET to the Grad-CAM map and alpha-blend with the
    original image so anatomical detail is preserved.

    Mix-blend equation:
        overlay = α · heatmap_rgb + (1 - α) · original_rgb

    The approach differs from a simple additive blend in that the sum
    of coefficients equals 1.0, so pixel values remain in [0, 255]
    without clipping artefacts.

    Args:
        cam_np: Normalised Grad-CAM map [H_cam, W_cam], float32 in [0,1].
        orig_rgb: Original image [H_orig, W_orig, 3], uint8.
        alpha: Heatmap blend weight. 0 → invisible, 1 → heatmap only.

    Returns:
        Blended overlay as uint8 numpy array [H_orig, W_orig, 3].
    """
    H_orig, W_orig = orig_rgb.shape[:2]

    # Resize cam to original image resolution for pixel-perfect overlay
    cam_resized = cv2.resize(cam_np, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)

    # Convert to uint8 for OpenCV colormap (expects [0, 255])
    cam_uint8 = np.uint8(255 * cam_resized)

    # Apply COLORMAP_JET: hot spots → red, cold regions → blue
    # cv2 returns BGR; convert to RGB for consistency
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)  # [H, W, 3] uint8

    # Mix-blend: weighted linear interpolation
    heatmap_f = heatmap_rgb.astype(np.float32)
    orig_f = orig_rgb.astype(np.float32)

    overlay_f = alpha * heatmap_f + (1.0 - alpha) * orig_f
    overlay = np.clip(overlay_f, 0, 255).astype(np.uint8)

    return overlay


# ──────────────────────────────────────────────
# Core function
# ──────────────────────────────────────────────
def generate_gradcam(
    image_path: str,
    class_idx: int,
    ckpt_path: str = DEFAULT_CKPT,
    output_path: str = DEFAULT_OUTPUT,
    alpha: float = HEATMAP_ALPHA,
    device_str: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a Grad-CAM heatmap overlay for a target CXR finding.

    Args:
        image_path: Path to CXR image (.dcm / .png / .jpg).
        class_idx: Target class index (0–7). See LABEL_COLUMNS.
        ckpt_path: Path to DenseNet121 checkpoint.
        output_path: Where to save the output PNG overlay.
        alpha: Heatmap blend weight [0, 1]. Default 0.45.
        device_str: "cpu" or "cuda". Auto-detected if None.

    Returns:
        JSON-serialisable dict:
        {
            "image_path": str,
            "class_idx": int,
            "class_label": str,
            "predicted_probability_pct": float,
            "output_path": str,
            "alpha": float,
            "status": "ok" | "error"
        }
    """
    if class_idx < 0 or class_idx >= len(LABEL_COLUMNS):
        raise ValueError(
            f"class_idx must be 0–{len(LABEL_COLUMNS)-1}, got {class_idx}"
        )

    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    result: Dict[str, Any] = {
        "image_path": str(image_path),
        "class_idx": class_idx,
        "class_label": LABEL_COLUMNS[class_idx],
        "output_path": str(output_path),
        "alpha": alpha,
        "status": "ok",
    }

    model: Optional[nn.Module] = None
    hook: Optional[GradCAMHook] = None

    try:
        # ── 1. Load image ───────────────────────────────────────────────
        orig_rgb, tensor = _load_image(image_path)
        tensor = tensor.to(device)

        # ── 2. Load model ───────────────────────────────────────────────
        model = _load_model_for_gradcam(ckpt_path)
        model.to(device)

        # ── 3. Register hooks on the last convolutional block ───────────
        # `features.denseblock4` is the last dense block (before norm5).
        # It outputs [B, 1024, H_feat, W_feat] where H_feat ≈ 16 for 512px input.
        # Using this layer avoids inplace ReLU issues that arise with norm5.
        target_layer: nn.Module = model.features.denseblock4
        hook = GradCAMHook(target_layer)

        # ── 4. Forward pass (gradient tracking enabled) ─────────────────
        # We must NOT use torch.inference_mode() or torch.no_grad() here
        # because we need gradients to flow backward.
        model.zero_grad()
        tensor.requires_grad_(False)  # gradient flows through model params, not input
        try:
            logits = model(tensor)                  # [1, 8]
        except torch.cuda.OutOfMemoryError as oom:
            torch.cuda.empty_cache()
            raise RuntimeError(
                f"[gradcam_gen] CUDA OOM during forward pass. "
                f"Try --device cpu. Details: {oom}"
            ) from oom

        prob = float(torch.sigmoid(logits[0, class_idx]).item())

        # ── 5. Backward pass for target class ───────────────────────────
        # score = raw logit (before sigmoid) — this is standard Grad-CAM
        score = logits[0, class_idx]
        score.backward()

        # ── 6. Compute Grad-CAM map ─────────────────────────────────────
        cam_np = hook.compute_cam()   # [H_feat, W_feat], float32, [0,1]

        # ── 7. Blend heatmap with original image ────────────────────────
        overlay = _blend_heatmap(cam_np, orig_rgb, alpha=alpha)

        # ── 8. Save output ──────────────────────────────────────────────
        out_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(out_dir, exist_ok=True)

        Image.fromarray(overlay).save(output_path)
        print(f"[gradcam_gen] Saved overlay to: {output_path}")

        result["predicted_probability_pct"] = round(prob * 100, 2)
        result["output_path"] = os.path.abspath(output_path)

    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        result["status"] = "error"
        result["error"] = str(exc)
    finally:
        if hook is not None:
            hook.remove()
        if model is not None:
            model.zero_grad(set_to_none=True)
            del model
        torch.cuda.empty_cache()

    return result


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Grad-CAM visual explanation generator for CXR DenseNet121. "
            "Saves a colour-overlay PNG showing where the model looks."
        )
    )
    p.add_argument("--image", required=True, help="Path to CXR image")
    p.add_argument(
        "--class_idx", type=int, required=True,
        help=(
            "Target class index (0=Cardiomegaly, 1=Pleural Effusion, "
            "2=Edema, 3=Pneumonia, 4=Atelectasis, 5=Pneumothorax, "
            "6=Consolidation, 7=Support Devices)"
        )
    )
    p.add_argument("--ckpt", default=DEFAULT_CKPT, help="Path to checkpoint")
    p.add_argument(
        "--out", default=DEFAULT_OUTPUT,
        help="Output PNG path (default: gradcam_output.png)"
    )
    p.add_argument(
        "--alpha", type=float, default=HEATMAP_ALPHA,
        help=f"Heatmap blend weight [0-1] (default: {HEATMAP_ALPHA})"
    )
    p.add_argument("--device", default=None, help="'cpu' or 'cuda'")
    return p


if __name__ == "__main__":
    import json
    parser = _build_arg_parser()
    args = parser.parse_args()

    output = generate_gradcam(
        image_path=args.image,
        class_idx=args.class_idx,
        ckpt_path=args.ckpt,
        output_path=args.out,
        alpha=args.alpha,
        device_str=args.device,
    )

    print(json.dumps(output, indent=2))
