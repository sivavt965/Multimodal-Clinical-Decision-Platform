# =============================================================================
# engine/model_loader.py
# Singleton loader for the pre-trained DenseNet121 CXR classification model.
# =============================================================================
"""
Weight file configuration
--------------------------
Set the environment variable  CXR_WEIGHTS_PATH  **or** edit the constant below
to point at your .pth checkpoint before starting the server.

The checkpoint is baseline_best.pt from the MIMIC project — an 8-class model
using a **custom DenseNet121** with 4 dropout layers (p=0.3) after each
transition block.  Input resolution: 512×512.  Normalisation: simple 0→1.

The loader uses the custom architecture from the training code to ensure
weight compatibility.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121, DenseNet121_Weights

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ⚙️  Configuration
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS_PATH: str = os.getenv("CXR_WEIGHTS_PATH", "")

# Full CheXpert 14-label taxonomy (used for display across the UI)
CHEXPERT_LABELS: list[str] = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

NUM_CLASSES_FULL: int = len(CHEXPERT_LABELS)

# The 8 labels the trained baseline_best.pt model actually predicts
MODEL_LABELS: list[str] = [
    "Cardiomegaly",
    "Pleural Effusion",
    "Edema",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Consolidation",
    "Support Devices",
]


# ---------------------------------------------------------------------------
# Custom DenseNet121 — MUST match the training architecture exactly
# ---------------------------------------------------------------------------
class CXRDenseNet121(nn.Module):
    """
    DenseNet121 with 4 dropout layers (after each transition/denseblock),
    matching the architecture in mimic_project/src/models/densenet121.py.
    """

    def __init__(self, num_classes: int, pretrained: bool = False, dropout_p: float = 0.3):
        super().__init__()

        if pretrained:
            backbone = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        else:
            backbone = densenet121(weights=None)

        self.features = backbone.features   # conv0 → norm5
        in_features = backbone.classifier.in_features  # 1024

        # Dropout layers after transition1/2/3 and denseblock4
        self.drop1 = nn.Dropout(p=dropout_p)
        self.drop2 = nn.Dropout(p=dropout_p)
        self.drop3 = nn.Dropout(p=dropout_p)
        self.drop4 = nn.Dropout(p=dropout_p)

        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features

        x = f.conv0(x)
        x = f.norm0(x)
        x = f.relu0(x)
        x = f.pool0(x)

        x = f.denseblock1(x)
        x = f.transition1(x)
        x = self.drop1(x)

        x = f.denseblock2(x)
        x = f.transition2(x)
        x = self.drop2(x)

        x = f.denseblock3(x)
        x = f.transition3(x)
        x = self.drop3(x)

        x = f.denseblock4(x)
        x = self.drop4(x)

        x = f.norm5(x)
        x = F.relu(x, inplace=True)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Device selection — prefer GPU/DirectML over CPU
# ---------------------------------------------------------------------------

def _select_device() -> torch.device:
    """Pick the best available device: CUDA > DirectML > CPU."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        logger.info("[ModelLoader] Using CUDA GPU: %s", torch.cuda.get_device_name(0))
        return dev

    # Try DirectML for AMD/Intel GPUs on Windows
    try:
        import torch_directml  # type: ignore
        dev = torch_directml.device()
        logger.info("[ModelLoader] Using DirectML GPU")
        return dev
    except ImportError:
        pass

    logger.info("[ModelLoader] Using CPU (install torch-directml or CUDA-enabled PyTorch for GPU)")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

class _ModelSingleton:
    """Thread-safe singleton that loads the model exactly once."""

    _instance: "_ModelSingleton | None" = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "_ModelSingleton":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._initialized = False
                    cls._instance = inst
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:          # type: ignore[attr-defined]
            return
        with _ModelSingleton._lock:
            if self._initialized:      # double-checked locking
                return

            self._device = _select_device()

            # Defaults — will be updated after loading
            self._model_labels: list[str] = CHEXPERT_LABELS
            self._num_model_classes: int = NUM_CLASSES_FULL

            weights_path = Path(DEFAULT_WEIGHTS_PATH) if DEFAULT_WEIGHTS_PATH else None

            if weights_path and weights_path.exists():
                try:
                    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)

                    # ── Unwrap checkpoint ───────────────────────────────────
                    state_dict = checkpoint
                    if isinstance(checkpoint, dict):
                        if "model_state_dict" in checkpoint:
                            state_dict = checkpoint["model_state_dict"]
                            if "label_columns" in checkpoint:
                                self._model_labels = checkpoint["label_columns"]
                        elif "model" in checkpoint:
                            state_dict = checkpoint["model"]
                        elif "state_dict" in checkpoint:
                            state_dict = checkpoint["state_dict"]

                    # ── Clean state dict keys (strip module./model. prefix) ─
                    cleaned_sd: dict = {}
                    for k, v in state_dict.items():
                        k2 = k
                        for prefix in ("module.", "model."):
                            if k2.startswith(prefix):
                                k2 = k2[len(prefix):]
                        cleaned_sd[k2] = v

                    # ── Detect num_classes from classifier weight ───────────
                    if "classifier.weight" in cleaned_sd:
                        self._num_model_classes = cleaned_sd["classifier.weight"].shape[0]
                    else:
                        self._num_model_classes = len(self._model_labels)

                    logger.info(
                        "[ModelLoader] Checkpoint: %d classes, labels=%s",
                        self._num_model_classes, self._model_labels,
                    )

                    # ── Build CUSTOM DenseNet121 with dropout layers ────────
                    self._model = CXRDenseNet121(
                        num_classes=self._num_model_classes,
                        pretrained=False,
                        dropout_p=0.3,
                    )
                    missing, unexpected = self._model.load_state_dict(cleaned_sd, strict=False)
                    if missing:
                        logger.warning("[ModelLoader] Missing keys: %s", missing)
                    if unexpected:
                        logger.warning("[ModelLoader] Unexpected keys: %s", unexpected)

                    logger.info("[ModelLoader] ✓ Loaded weights from %s", weights_path)

                    if "best_val_auc" in checkpoint:
                        logger.info(
                            "[ModelLoader]   Best val AUC: %.4f (epoch %s)",
                            checkpoint["best_val_auc"],
                            checkpoint.get("epoch", "?"),
                        )

                except Exception as exc:
                    logger.error(
                        "[ModelLoader] Failed to load weights from %s — %s. "
                        "Falling back to ImageNet weights.",
                        weights_path, exc,
                    )
                    self._model = CXRDenseNet121(NUM_CLASSES_FULL, pretrained=True)
                    self._num_model_classes = NUM_CLASSES_FULL
                    self._model_labels = CHEXPERT_LABELS
            else:
                if DEFAULT_WEIGHTS_PATH:
                    logger.warning(
                        "[ModelLoader] Weights file not found at '%s'. "
                        "Using ImageNet pre-trained backbone (dev mode).",
                        DEFAULT_WEIGHTS_PATH,
                    )
                else:
                    logger.warning(
                        "[ModelLoader] CXR_WEIGHTS_PATH not set. "
                        "Using ImageNet pre-trained backbone (dev mode only)."
                    )
                self._model = CXRDenseNet121(NUM_CLASSES_FULL, pretrained=True)
                self._num_model_classes = NUM_CLASSES_FULL
                self._model_labels = CHEXPERT_LABELS

            self._model.to(self._device)
            self._model.eval()
            self._initialized = True
            logger.info(
                "[ModelLoader] Model ready — %d classes on %s.",
                self._num_model_classes, self._device,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def labels(self) -> list[str]:
        """Return the full 14-label CheXpert taxonomy for display."""
        return CHEXPERT_LABELS

    @property
    def model_labels(self) -> list[str]:
        """Return the labels the loaded model actually predicts."""
        return self._model_labels

    @property
    def num_model_classes(self) -> int:
        return self._num_model_classes


# Convenience accessor used by inference.py
def get_model() -> _ModelSingleton:
    """Return the loaded singleton model wrapper (lazy init on first call)."""
    return _ModelSingleton()
