# =============================================================================
# engine/inference.py
# Real PyTorch Grad-CAM inference for Chest X-Ray classification.
# =============================================================================
"""
Public surface
--------------
    run_cxr_inference(image_path, target_label, case_id) -> InferenceResult

Preprocessing must match training exactly:
  - Resize to 512×512
  - ToTensor: converts PIL uint8 [0,255] → float [0,1]
  - Normalize with ImageNet mean/std:
      mean = (0.485, 0.456, 0.406)
      std  = (0.229, 0.224, 0.225)
    This matches the training dataloader in
    mimic_project/src/data/dataloader_cloud.py (lines 72-73).
  - 3-channel RGB input

MC Dropout:
  - The custom CXRDenseNet121 has 4 nn.Dropout(p=0.3) layers.
  - model.train() enables dropout → stochastic forward passes.
  - We run N=10 passes, compute mean/variance per class.
  - Capped at 10 passes for real-time performance.
  - Timeout protection: aborts if passes exceed MC_DROPOUT_TIMEOUT_SEC.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from engine.model_loader import CHEXPERT_LABELS, get_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output directory — relative to the repo root so Next.js can serve them
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
HEATMAP_DIR = _REPO_ROOT / "frontend" / "public" / "mock-data" / "heatmaps"
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Image pre-processing — MUST match training pipeline exactly
# ---------------------------------------------------------------------------
# Training used (dataloader_cloud.py, the baseline image-only model):
#   A.Resize(512, 512) → A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD) → ToTensorV2()
# Confirmed in code/constants.py: IMAGENET_MEAN = [0.485, 0.456, 0.406], IMAGENET_STD = [0.229, 0.224, 0.225]
#
# CRITICAL: The previous version omitted ImageNet normalization, causing all
# probabilities to be ~0. This is now fixed.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

_PREPROCESS = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),           # PIL uint8 [0,255] → float [0,1]
        transforms.Normalize(
            mean=_IMAGENET_MEAN,
            std=_IMAGENET_STD,
        ),
    ]
)

# MC Dropout configuration
MC_DROPOUT_PASSES: int = 10    # Capped at 10 for real-time inference
MC_DROPOUT_TIMEOUT_SEC: float = 30.0  # Abort MC dropout after this many seconds

# The model singleton is shared across all threads (FastAPI BackgroundTasks land
# in a thread pool). MC Dropout flips it into train() mode mid-call, so a
# parallel "deterministic" forward pass on the same model would see stochastic
# outputs. Serialise every model-touching section with this lock.
_INFERENCE_LOCK: threading.Lock = threading.Lock()


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------
@dataclass
class InferenceResult:
    """Structured output from a single CXR inference run."""

    probabilities: dict[str, float] = field(default_factory=dict)
    """Maps each CheXpert label → sigmoid probability in [0, 1]."""

    predictions: list[dict] = field(default_factory=list)
    """List of {label, probability, risk_badge, ...} dicts."""

    heatmap_url: Optional[str] = None
    """Public URL path served by Next.js."""

    heatmap_label: Optional[str] = None
    """The CheXpert label the Grad-CAM was computed for."""

    embedding: Optional[np.ndarray] = None
    """L2-normalised 1024-d GAP vector from DenseNet121 features block."""

    # MC Dropout uncertainty
    mc_mean_probs: Optional[dict[str, float]] = None
    mc_std_devs: Optional[dict[str, float]] = None
    mc_uncertainty_level: Optional[str] = None
    mc_mean_variance: Optional[float] = None

    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Grad-CAM hooks
# ---------------------------------------------------------------------------

class _GradCAMHook:
    """Captures activations and gradients from the target layer."""

    def __init__(self, layer: torch.nn.Module) -> None:
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._fwd_handle = layer.register_forward_hook(self._save_activation)
        self._bwd_handle = layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inp, output):
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()


class _GAPHook:
    """Captures the GAP embedding (1024-d) from the features block."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.gap_vector: Optional[torch.Tensor] = None
        # Hook on denseblock4 output (before drop4 and norm5)
        self._handle = model.features.denseblock4.register_forward_hook(self._capture)

    def _capture(self, _module, _inp, output: torch.Tensor) -> None:
        activated = F.relu(output, inplace=False)
        pooled = F.adaptive_avg_pool2d(activated, (1, 1))
        self.gap_vector = pooled.squeeze(-1).squeeze(-1).detach()

    def remove(self) -> None:
        self._handle.remove()


def _compute_gradcam(
    model: torch.nn.Module,
    device: torch.device,
    input_tensor: torch.Tensor,
    class_idx: int,
) -> np.ndarray:
    """Return a (512, 512) float32 Grad-CAM array in [0, 1]."""
    # Target: denseblock4 (same as training gradcam code)
    target_layer = model.features.denseblock4
    hook = _GradCAMHook(target_layer)

    try:
        model.zero_grad()
        with torch.set_grad_enabled(True):
            x = input_tensor.to(device).requires_grad_(True)
            logits = model(x)
            score = logits[0, class_idx]
            score.backward()

        grads = hook.gradients        # (1, C, H, W)
        acts  = hook.activations      # (1, C, H, W)
    finally:
        # Always remove the forward+backward hooks — leaving them registered
        # on a shared singleton model leaks state and slows every later pass.
        hook.remove()

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1, keepdim=True)
    cam = F.relu(cam)

    cam = cam.squeeze().cpu().numpy()
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()

    cam = cv2.resize(cam, (512, 512))
    return cam.astype(np.float32)


# ---------------------------------------------------------------------------
# Heatmap rendering
# ---------------------------------------------------------------------------

def _overlay_heatmap(
    original_img: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend a Jet-colourmap Grad-CAM over the original CXR image."""
    orig_np = np.array(original_img.convert("RGB").resize((512, 512)))
    orig_bgr = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)

    heatmap_uint8 = np.uint8(255 * cam)
    jet = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(orig_bgr, 1 - alpha, jet, alpha, 0)
    return overlay


def _save_heatmap(overlay_bgr: np.ndarray, stem: str) -> str:
    fname = f"{stem}.png"
    out_path = HEATMAP_DIR / fname
    cv2.imwrite(str(out_path), overlay_bgr)
    logger.info("[Inference] Heatmap saved → %s", out_path)
    return f"/mock-data/heatmaps/{fname}"


# ---------------------------------------------------------------------------
# Risk badge + uncertainty helpers
# ---------------------------------------------------------------------------

def _risk_badge(prob: float) -> str:
    if prob >= 0.15:
        return "Elevated Risk"
    if prob >= 0.05:
        return "Monitor"
    return "Unlikely"


def _uncertainty_level(mean_variance: float) -> str:
    """Classify overall MC Dropout variance into a human-readable level.

    Thresholds calibrated for DenseNet121 with 10 dropout passes — observed
    mean variance typically falls in 0.0001–0.001 for the trained labels.
    """
    if mean_variance < 0.0002:
        return "Low Uncertainty"
    if mean_variance < 0.0006:
        return "Moderate Uncertainty"
    return "High Uncertainty"


def _per_finding_uncertainty(std_dev: float) -> str:
    """Classify per-finding uncertainty from its own MC Dropout std deviation.

    std_dev is the std of the sigmoid probability across MC passes, so it
    directly represents the spread of estimated probability for that finding.
    """
    if std_dev < 0.015:
        return "Low Uncertainty"
    if std_dev < 0.04:
        return "Moderate Uncertainty"
    return "High Uncertainty"


# ---------------------------------------------------------------------------
# MC Dropout
# ---------------------------------------------------------------------------

def _run_mc_dropout(
    model: torch.nn.Module,
    device: torch.device,
    input_tensor: torch.Tensor,
    n_passes: int = MC_DROPOUT_PASSES,
    timeout_sec: float = MC_DROPOUT_TIMEOUT_SEC,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run MC Dropout: n_passes stochastic forward passes with dropout enabled.

    Includes timeout protection — aborts early if wall-clock time exceeds
    timeout_sec, returning results from however many passes completed.

    Returns
    -------
    mean_probs : (num_classes,) — mean sigmoid probability per class
    std_devs   : (num_classes,) — std deviation per class
    """
    # Enable dropout layers ONLY — keep batchnorm in eval mode
    model.train()
    # Force batchnorm back to eval (we only want dropout stochasticity)
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            module.eval()

    all_probs = []
    t_start = time.monotonic()
    completed_passes = 0

    try:
        with torch.no_grad():
            x = input_tensor.to(device)
            for i in range(n_passes):
                # Timeout check
                elapsed = time.monotonic() - t_start
                if elapsed > timeout_sec:
                    logger.warning(
                        "[MC Dropout] Timeout after %.1fs (%d/%d passes). Returning partial results.",
                        elapsed, completed_passes, n_passes,
                    )
                    break

                logits = model(x)
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()
                all_probs.append(probs)
                completed_passes += 1

                # Free intermediate tensors
                del logits
    finally:
        # ALWAYS restore eval mode, even if an exception occurs
        model.eval()
        # Free GPU memory if available
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if completed_passes == 0:
        raise RuntimeError("MC Dropout completed 0 passes (timeout too aggressive or model failure).")

    logger.info(
        "[MC Dropout] Completed %d/%d passes in %.2fs",
        completed_passes, n_passes, time.monotonic() - t_start,
    )

    stacked = np.stack(all_probs, axis=0)  # (completed_passes, num_classes)
    mean_probs = stacked.mean(axis=0)
    std_devs   = stacked.std(axis=0)

    return mean_probs, std_devs


# ---------------------------------------------------------------------------
# GAP embedding extraction
# ---------------------------------------------------------------------------

def _extract_gap_embedding(
    model: torch.nn.Module,
    device: torch.device,
    input_tensor: torch.Tensor,
) -> np.ndarray:
    """Return L2-normalised 1024-d embedding as (1, 1024) float32."""
    from sklearn.preprocessing import normalize as sk_normalize

    gap_hook = _GAPHook(model)
    try:
        with torch.no_grad():
            _ = model(input_tensor.to(device))
    finally:
        gap_hook.remove()

    if gap_hook.gap_vector is None:
        raise RuntimeError("GAP hook did not capture any output.")

    vec: np.ndarray = gap_hook.gap_vector.cpu().numpy()
    vec = sk_normalize(vec, norm="l2")
    return vec


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def run_cxr_inference(
    image_path: str | Path,
    target_label: Optional[str] = None,
    case_id: Optional[str] = None,
) -> InferenceResult:
    """
    Run DenseNet121 inference + Grad-CAM + MC Dropout + GAP embedding.

    Parameters
    ----------
    target_label : Optional[str]
        Which CheXpert label to compute Grad-CAM for. Pass None (the default)
        to auto-pick the highest-probability label after the forward pass —
        this is what the initial-upload path wants. Pass an explicit label
        (e.g. from the regenerate-heatmap endpoint) to force that class.
    """
    from engine.vector_store import get_vector_store

    image_path = Path(image_path)
    result = InferenceResult()

    loader = get_model()
    model  = loader.model
    device = loader.device
    model_labels = loader.model_labels

    # ------------------------------------------------------------------
    # 0. Validate target label (if explicitly requested)
    # ------------------------------------------------------------------
    if target_label is not None and target_label not in model_labels:
        logger.warning(
            "[Inference] Label '%s' not in model's %d classes; falling back to "
            "auto-pick top label.",
            target_label, len(model_labels),
        )
        target_label = None

    # ------------------------------------------------------------------
    # 1. Load image
    # ------------------------------------------------------------------
    try:
        original_img = Image.open(image_path).convert("RGB")
    except Exception as exc:
        result.error = f"Cannot open image '{image_path}': {exc}"
        logger.error("[Inference] %s", result.error)
        return result

    # ------------------------------------------------------------------
    # 2. Pre-process (512×512, ImageNet-normalised)
    # ------------------------------------------------------------------
    input_tensor: torch.Tensor = _PREPROCESS(original_img).unsqueeze(0)
    logger.info(
        "[Inference] Tensor shape: %s, range: [%.3f, %.3f] (ImageNet-normalised)",
        tuple(input_tensor.shape),
        input_tensor.min().item(),
        input_tensor.max().item(),
    )

    # ------------------------------------------------------------------
    # Lock the singleton model for the rest of this call. Steps 3-6 all
    # mutate or depend on global module state (eval/train, hooks, autograd).
    # ------------------------------------------------------------------
    with _INFERENCE_LOCK:
        # ------------------------------------------------------------------
        # 3. Forward pass — deterministic sigmoid probabilities
        # ------------------------------------------------------------------
        try:
            model.eval()
            with torch.no_grad():
                logits = model(input_tensor.to(device))
                probs  = torch.sigmoid(logits).squeeze().cpu()
        except Exception as exc:
            result.error = f"Forward pass failed: {exc}"
            logger.error("[Inference] %s", result.error)
            return result

        # Only use the labels the model was actually trained on
        prob_dict: dict[str, float] = {}
        for i, label in enumerate(model_labels):
            prob_dict[label] = float(probs[i])

        result.probabilities = prob_dict

        # Honour caller's target_label if given; otherwise auto-pick top class.
        if target_label is None:
            target_label = max(prob_dict, key=prob_dict.get)  # type: ignore
        class_idx = model_labels.index(target_label)
        result.heatmap_label = target_label

        # ------------------------------------------------------------------
        # 4. MC Dropout uncertainty estimation
        # ------------------------------------------------------------------
        try:
            mc_mean, mc_std = _run_mc_dropout(model, device, input_tensor.clone())

            mc_mean_dict: dict[str, float] = {}
            mc_std_dict: dict[str, float] = {}
            for i, label in enumerate(model_labels):
                mc_mean_dict[label] = float(mc_mean[i])
                mc_std_dict[label] = float(mc_std[i])

            # Per-class variance → mean variance
            variances = [mc_std[model_labels.index(l)] ** 2 for l in model_labels]
            mean_var = float(np.mean(variances))

            result.mc_mean_probs = mc_mean_dict
            result.mc_std_devs = mc_std_dict
            result.mc_mean_variance = mean_var
            result.mc_uncertainty_level = _uncertainty_level(mean_var)

            logger.info(
                "[Inference] MC Dropout (%d passes): mean_var=%.5f → %s",
                MC_DROPOUT_PASSES, mean_var, result.mc_uncertainty_level,
            )
        except Exception as exc:
            logger.error("[Inference] MC Dropout failed: %s", exc, exc_info=True)

        # ── Build predictions list (only the 8 trained labels) ─────────────
        # Per-finding uncertainty derived from each label's own std_dev — far more
        # informative than the global mean_variance which is the same for all rows.
        result.predictions = [
            {
                "label": label,
                "probability": prob_dict[label],
                "risk_badge": _risk_badge(prob_dict[label]),
                "uncertainty_level": (
                    _per_finding_uncertainty((result.mc_std_devs or {}).get(label, 0.0))
                    if result.mc_std_devs is not None
                    else result.mc_uncertainty_level
                ),
                "mean_variance": result.mc_mean_variance,
                "std_dev": (result.mc_std_devs or {}).get(label, None),
                "mc_passes": MC_DROPOUT_PASSES,
            }
            for label in model_labels
        ]

        # ------------------------------------------------------------------
        # 5. GAP embedding extraction
        # ------------------------------------------------------------------
        try:
            model.eval()
            embedding = _extract_gap_embedding(model, device, input_tensor.clone())
            result.embedding = embedding

            if case_id:
                vs = get_vector_store()
                vs.add_to_index(case_id=case_id, embedding=embedding)
                # Debounced save — disk write at most once per interval, and the
                # lifespan shutdown hook does a forced flush so nothing is lost.
                vs.save_if_needed()
                logger.info("[Inference] Embedding indexed for case %s", case_id)
        except Exception as exc:
            logger.error("[Inference] GAP embedding failed: %s", exc, exc_info=True)

        # ------------------------------------------------------------------
        # 6. Grad-CAM
        # ------------------------------------------------------------------
        try:
            model.eval()
            cam = _compute_gradcam(model, device, input_tensor.clone(), class_idx)
            overlay = _overlay_heatmap(original_img, cam)
            stem = f"heatmap_{image_path.stem}_{uuid.uuid4().hex[:8]}"
            result.heatmap_url = _save_heatmap(overlay, stem)
        except Exception as exc:
            logger.error("[Inference] Grad-CAM failed: %s", exc, exc_info=True)
            result.heatmap_url = None

    # Log top finding (outside the lock — no model access here)
    modeled = {k: v for k, v in prob_dict.items() if v > 0}
    if modeled:
        top_label = max(modeled, key=modeled.get)  # type: ignore
        logger.info(
            "[Inference] Done — top: %s (%.3f), uncertainty: %s",
            top_label, modeled[top_label],
            result.mc_uncertainty_level or "N/A",
        )

    return result
