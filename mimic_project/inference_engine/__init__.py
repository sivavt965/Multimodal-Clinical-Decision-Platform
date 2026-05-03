"""
inference_engine — Multimodal Clinical Decision Support Platform
================================================================
Standalone, modular inference scripts for the MIMIC-CXR project.
Each module can be run independently via CLI or imported into a backend.

Modules:
    cxr_inference      — DenseNet121 CXR inference + Temperature Scaling
    mc_dropout         — MC Dropout uncertainty quantification
    gradcam_gen        — Grad-CAM visual explanation overlay
    early_risk_inference — ECG/lab tabular risk stratification (MLP)

Model files expected at:
    ../models/baseline_best.pt          (DenseNet121, 8-class, image-only)
    ../models/baseline_best_old_state_dict.pt   (older format, same arch)
"""

from .cxr_inference import predict
from .mc_dropout import quantify_uncertainty
from .gradcam_gen import generate_gradcam
from .early_risk_inference import assess_early_risk

__all__ = [
    "predict",
    "quantify_uncertainty",
    "generate_gradcam",
    "assess_early_risk",
]
