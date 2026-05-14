# =============================================================================
# engine/symile_encoder.py
# Singleton loader for the trained Symile-MIMIC multimodal model + the public
# run_symile_inference() entry-point that produces the 24576-d embedding
# (concatenation of the three 8192-d per-modality representations).
# =============================================================================
"""
The trained checkpoint lives at  symile_mimic_model.ckpt  in the repo root.
Hyper-parameters discovered from that file:
  d                  = 8192   (per-modality projection dim)
  loss_fn            = "symile"
  pretrained         = False  (architecture is the un-pretrained ResNet variants)
  freeze_logit_scale = False

Forward inputs (batch_sz=1 for inference):
  cxr               : (1, 3, 320, 320) float32
  ecg               : (1, 1, 5000, 12) float32
  labs_percentiles  : (1, 50)          float32
  labs_missingness  : (1, 50)          float32
  hadm_id           : (1,)             int   — unused at inference time

Forward outputs:
  r_c, r_e, r_l : each (1, 8192) float32
  logit_scale   : scalar (unused for retrieval)

The combined embedding indexed in FAISS is  concat([r_c, r_e, r_l])  →  (1, 24576).

Loading the checkpoint on Windows requires patching pathlib.PosixPath because the
checkpoint was pickled on Linux.

This module is intentionally side-effect-free at import time. Call get_symile_model()
only when Symile inference is actually needed.
"""

from __future__ import annotations

import logging
import os
import pathlib
import sys
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import normalize as sk_normalize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path / config
# ---------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT   = _BACKEND_DIR.parent

DEFAULT_SYMILE_CHECKPOINT: Path = Path(
    os.getenv("SYMILE_CHECKPOINT_PATH",
              str(_REPO_ROOT / "symile_mimic_model.ckpt"))
)

# The Symile model module imports `datasets`, `losses`, `utils` from the
# experiments/ folder — add those to sys.path so import works.
_SYMILE_EXPERIMENTS = _REPO_ROOT / "symile-main" / "symile-main" / "experiments"
_SYMILE_MODELS      = _SYMILE_EXPERIMENTS / "models"


def _ensure_symile_imports_on_path() -> None:
    for p in (_SYMILE_EXPERIMENTS, _SYMILE_MODELS):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml  # type: ignore
        return torch_directml.device()
    except ImportError:
        pass
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

class _SymileSingleton:
    """Loads the SymileMIMICModel checkpoint exactly once."""

    _instance: "_SymileSingleton | None" = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "_SymileSingleton":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._initialized = False
                    cls._instance = inst
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:                # type: ignore[attr-defined]
            return
        with _SymileSingleton._lock:
            if self._initialized:
                return

            ckpt_path = DEFAULT_SYMILE_CHECKPOINT
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"Symile checkpoint not found at {ckpt_path}. "
                    f"Set SYMILE_CHECKPOINT_PATH or place the .ckpt file there."
                )

            _ensure_symile_imports_on_path()

            # The checkpoint was pickled on Linux; pathlib.PosixPath instances
            # in hyper_parameters would crash unpickling on Windows.
            _orig_posix = pathlib.PosixPath
            if os.name == "nt":
                pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[misc]

            try:
                from symile_mimic_model import SymileMIMICModel  # type: ignore

                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                hparams = ckpt.get("hyper_parameters", {})
                self._d: int = int(hparams.get("d", 8192))

                model = SymileMIMICModel(**hparams)
                missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
                if missing:
                    logger.warning("[SymileLoader] Missing keys: %s", missing[:5])
                if unexpected:
                    logger.warning("[SymileLoader] Unexpected keys: %s", unexpected[:5])
            finally:
                if os.name == "nt":
                    pathlib.PosixPath = _orig_posix  # type: ignore[misc]

            self._device = _select_device()
            model.to(self._device)
            model.eval()
            self._model = model
            self._initialized = True
            logger.info(
                "[SymileLoader] ✓ Loaded checkpoint from %s (d=%d) on %s",
                ckpt_path, self._d, self._device,
            )

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def d(self) -> int:
        return self._d


def get_symile_model() -> _SymileSingleton:
    """Lazy accessor — instantiates on first call, returns the cached singleton."""
    return _SymileSingleton()


# ---------------------------------------------------------------------------
# Inference entry-point
# ---------------------------------------------------------------------------

# Thread safety: like the DenseNet path, the model is global and not
# thread-safe. Serialise calls into it.
_SYMILE_INFERENCE_LOCK: threading.Lock = threading.Lock()


def run_symile_inference(
    cxr: np.ndarray | torch.Tensor,
    ecg: np.ndarray | torch.Tensor,
    labs_percentiles: np.ndarray | torch.Tensor,
    labs_missingness: np.ndarray | torch.Tensor,
    hadm_id: Optional[int] = None,
) -> np.ndarray:
    """
    Encode one multimodal sample with the Symile-MIMIC model.

    Parameters
    ----------
    cxr               : (3, 320, 320) or (1, 3, 320, 320)
    ecg               : (1, 5000, 12) or (1, 1, 5000, 12)
    labs_percentiles  : (50,) or (1, 50)
    labs_missingness  : (50,) or (1, 50)
    hadm_id           : ignored at inference time; kept for signature parity

    Returns
    -------
    np.ndarray of shape (1, 3*d) — L2-normalised concatenation of the three
    per-modality representations (r_c, r_e, r_l). With the trained checkpoint
    that's (1, 24576).
    """
    loader = get_symile_model()
    model  = loader.model
    device = loader.device

    def _to_tensor(x, expected_ndim: int) -> torch.Tensor:
        t = torch.as_tensor(x, dtype=torch.float32)
        # Insert a leading batch dim if missing.
        while t.ndim < expected_ndim:
            t = t.unsqueeze(0)
        return t.to(device)

    cxr_t = _to_tensor(cxr, expected_ndim=4)              # (1, 3, 320, 320)
    ecg_t = _to_tensor(ecg, expected_ndim=4)              # (1, 1, 5000, 12)
    lp_t  = _to_tensor(labs_percentiles, expected_ndim=2) # (1, 50)
    lm_t  = _to_tensor(labs_missingness, expected_ndim=2) # (1, 50)
    hid_t = torch.as_tensor([hadm_id or 0], dtype=torch.long, device=device)

    with _SYMILE_INFERENCE_LOCK:
        with torch.no_grad():
            r_c, r_e, r_l, _logit_scale = model([cxr_t, ecg_t, lp_t, lm_t, hid_t])

    # Concatenate the three per-modality reps into one vector. L2-normalise so
    # FAISS L2 distance behaves like cosine distance (same convention as the
    # DenseNet path).
    combined = torch.cat([r_c, r_e, r_l], dim=1).cpu().numpy().astype(np.float32)
    combined = sk_normalize(combined, norm="l2")
    return combined


# ---------------------------------------------------------------------------
# Convenience: encode a case from the precomputed data_npy bundle
# ---------------------------------------------------------------------------

_DATA_NPY_DIR = _REPO_ROOT / "data_npy"


def encode_case_by_hadm_id(
    hadm_id: int,
    split: str = "val",
) -> np.ndarray:
    """
    Look up a hadm_id inside data_npy/{split}/ and run Symile inference on it.

    Useful for bulk-indexing the symile FAISS store from the existing precomputed
    tensors without going through DICOM/ECG re-preprocessing.
    """
    split_dir = _DATA_NPY_DIR / split
    if not split_dir.exists():
        raise FileNotFoundError(f"data_npy split dir not found: {split_dir}")

    hadm_ids = np.load(split_dir / f"hadm_id_{split}.npy")
    matches  = np.where(hadm_ids == hadm_id)[0]
    if len(matches) == 0:
        raise KeyError(f"hadm_id {hadm_id} not found in {split} split")
    row = int(matches[0])

    cxr = np.load(split_dir / f"cxr_{split}.npy", mmap_mode="r")[row]
    ecg = np.load(split_dir / f"ecg_{split}.npy", mmap_mode="r")[row]
    lp  = np.load(split_dir / f"labs_percentiles_{split}.npy", mmap_mode="r")[row]
    lm  = np.load(split_dir / f"labs_missingness_{split}.npy", mmap_mode="r")[row]

    return run_symile_inference(cxr, ecg, lp, lm, hadm_id=int(hadm_id))
