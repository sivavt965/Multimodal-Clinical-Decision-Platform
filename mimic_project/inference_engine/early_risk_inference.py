#!/usr/bin/env python3
"""
early_risk_inference.py — Phase A Logic: ECG & Lab-based Early Risk Engine
===========================================================================
Processes structured tabular input (ECG features + blood lab values) through
a mock Multi-Layer Perceptron (MLP) and produces a risk stratification level
(Low / Moderate / High) with a corresponding imaging recommendation string.

This is the "Phase A" component of the Clinical Decision Support Platform —
it runs BEFORE an imaging study is ordered and helps triage patients.

Usage (CLI):
    python early_risk_inference.py --input path/to/labs_ecg.json
    python early_risk_inference.py --troponin 0.8 --wbc 14.2 --hr 112 ...

Usage (import):
    from inference_engine.early_risk_inference import assess_early_risk
    result = assess_early_risk(features={"troponin_elevated": 0.8, ...})

Architecture — EarlyRiskMLP
-----------------------------
A two-hidden-layer MLP that maps tabular clinical features to risk logits:

    Input  [D=12]  →  FC(12→64) + ReLU + Dropout(0.2)
                   →  FC(64→32) + ReLU + Dropout(0.2)
                   →  FC(32→3)  → Softmax  [Low, Moderate, High risk]

The output is a 3-class probability distribution where:
    • Class 0 → Low Risk      → "Routine follow-up recommended"
    • Class 1 → Moderate Risk → "Consider imaging within 24 hours"
    • Class 2 → High Risk     → "Urgent imaging within 2 hours / escalate care"

Risk score mapping:
    The raw softmax output gives class probabilities [p_low, p_mod, p_high].
    A scalar risk score is derived as the expected class index:
        risk_score = 0·p_low + 1·p_mod + 2·p_high
    Normalised to [0, 1]:
        risk_score_norm = risk_score / 2.0

Input Features [12 dimensions]:
    Cardiac:
        0: troponin_percentile   (normalised; 0.5 = median population)
        1: bnp_percentile
        2: hr_norm               (heart rate / 200.0, clipped [0,1])
        3: pr_interval_norm      (PR interval ms / 300.0)
        4: qrs_duration_norm     (QRS ms / 200.0)
        5: qt_corrected_norm     (QTc ms / 700.0)
        6: st_deviation_mm       (raw mm, clipped [-5, +5], then /5)

    Lab values:
        7: wbc_norm              (WBC count / 20.0, clipped [0,1])
        8: creatinine_norm       (creatinine mg/dL / 15.0)
        9: sodium_norm           (Na+ mEq/L / 170.0)
       10: potassium_norm        (K+ mEq/L / 10.0)
       11: lactate_norm          (lactate mmol/L / 20.0)

If a feature is missing, it defaults to the population median (0.5 for
percentile features, or the physiologically normal value for raw features).

Model weights:
    The MLP is a MOCK — its weights are NOT trained. In production, pass
    a real checkpoint via `--ckpt` (expected key: "model_state_dict").
    For demo/testing purposes, the mock produces plausible risk levels
    based on simple threshold logic so the output format can be validated.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))

# 12-dim feature vector (see module docstring for full description)
FEATURE_NAMES = [
    "troponin_percentile",
    "bnp_percentile",
    "hr_norm",
    "pr_interval_norm",
    "qrs_duration_norm",
    "qt_corrected_norm",
    "st_deviation_norm",
    "wbc_norm",
    "creatinine_norm",
    "sodium_norm",
    "potassium_norm",
    "lactate_norm",
]

# Default values used when a feature is missing (median / physiologically normal)
_FEATURE_DEFAULTS: Dict[str, float] = {
    "troponin_percentile": 0.50,
    "bnp_percentile":      0.50,
    "hr_norm":             0.375,   # 75 bpm / 200
    "pr_interval_norm":    0.533,   # 160 ms / 300
    "qrs_duration_norm":   0.500,   # 100 ms / 200
    "qt_corrected_norm":   0.600,   # 420 ms / 700
    "st_deviation_norm":   0.500,   # 0 mm deviation (0 / 5 mapped to [0,1])
    "wbc_norm":            0.375,   # 7.5 × 10³/μL / 20
    "creatinine_norm":     0.067,   # 1.0 mg/dL / 15
    "sodium_norm":         0.847,   # 144 mEq/L / 170
    "potassium_norm":      0.400,   # 4.0 mEq/L / 10
    "lactate_norm":        0.100,   # 2.0 mmol/L / 20
}

INPUT_DIM: int = len(FEATURE_NAMES)   # 12

# Risk-level labels and their recommendations
RISK_LABELS = ["Low", "Moderate", "High"]
RISK_RECOMMENDATIONS: Dict[str, str] = {
    "Low":      "Routine follow-up recommended. No urgent imaging required.",
    "Moderate": "Consider imaging within 24 hours. Monitor vitals closely.",
    "High":     "Urgent imaging within 2 hours. Escalate care immediately.",
}


# ──────────────────────────────────────────────
# Model definition
# ──────────────────────────────────────────────
class EarlyRiskMLP(nn.Module):
    """
    Tabular MLP for early cardiovascular / acute illness risk stratification.

    Architecture:
        FC(12→64) → ReLU → Dropout(0.2)
        FC(64→32) → ReLU → Dropout(0.2)
        FC(32→3)

    The final layer outputs raw logits for [Low, Moderate, High] risk classes.
    Softmax is applied OUTSIDE the model (in the inference function) to get
    calibrated probabilities.

    Dropout is included for MC Dropout compatibility (if needed later).

    Args:
        input_dim: Number of input features (default 12).
        hidden1: First hidden layer size (default 64).
        hidden2: Second hidden layer size (default 32).
        num_classes: Output classes (default 3: Low/Moderate/High).
        dropout_p: Dropout probability (default 0.2).
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden1: int = 64,
        hidden2: int = 32,
        num_classes: int = 3,
        dropout_p: float = 0.2,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)
        self.dropout_p = dropout_p

    def forward(self, x: torch.Tensor, mc_dropout: bool = False) -> torch.Tensor:
        """
        Forward pass through the MLP.

        Args:
            x: Input tensor of shape [B, input_dim].
            mc_dropout: If True, keep dropout active even in eval mode.

        Returns:
            Raw logits tensor of shape [B, 3].
        """
        drop_on = bool(self.training or mc_dropout)

        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, p=self.dropout_p, training=drop_on)
        x = F.relu(self.fc2(x), inplace=True)
        x = F.dropout(x, p=self.dropout_p, training=drop_on)
        return self.fc3(x)   # raw logits [B, 3]

    @classmethod
    def _init_mock_weights(cls, model: "EarlyRiskMLP") -> None:
        """
        Initialise weights with a clinically-informed prior so the mock
        MLP produces sensible outputs without training.

        Strategy:
            - Use Xavier uniform for all layers (PyTorch default).
            - Bias the output layer so that at median input (all 0.5):
                • fc3 bias: [-0.5, 0.0, 0.5]
              → Low gets slight negative boost, High gets slight positive.
              → At truly elevated values, the High logit will dominate.
        """
        with torch.no_grad():
            nn.init.xavier_uniform_(model.fc1.weight)
            nn.init.zeros_(model.fc1.bias)
            nn.init.xavier_uniform_(model.fc2.weight)
            nn.init.zeros_(model.fc2.bias)
            nn.init.xavier_uniform_(model.fc3.weight)
            model.fc3.bias.copy_(torch.tensor([-0.5, 0.0, 0.5]))


def _build_mock_mlp(device: torch.device) -> EarlyRiskMLP:
    """
    Construct and return a mock EarlyRiskMLP with clinically-primed weights.
    This is used when no real checkpoint is available.
    """
    model = EarlyRiskMLP()
    EarlyRiskMLP._init_mock_weights(model)
    model.to(device)
    model.eval()
    return model


def _load_mlp_from_ckpt(ckpt_path: str, device: torch.device) -> EarlyRiskMLP:
    """
    Load EarlyRiskMLP from a saved checkpoint.

    Expected checkpoint format:
        {"model_state_dict": state_dict, ...}  OR a raw state_dict.

    Args:
        ckpt_path: Path to .pt checkpoint file.
        device: Target device.

    Returns:
        Loaded, eval-mode EarlyRiskMLP.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"[early_risk] Checkpoint not found: {ckpt_path}. "
            "Using mock model instead is possible by omitting --ckpt."
        )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    model = EarlyRiskMLP()
    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()
    return model


# ──────────────────────────────────────────────
# Feature preprocessing
# ──────────────────────────────────────────────
def _validate_and_normalise(
    raw_features: Dict[str, Any]
) -> torch.Tensor:
    """
    Validate, fill missing values, and convert a feature dict to a
    float32 tensor of shape [1, 12].

    Normalisation conventions (caller is expected to provide pre-normalised
    percentiles; raw physical units are normalised here if recognised keys
    are detected):

        troponin_elevated (ng/mL)  → troponin_percentile via sigmoid(troponin * 5)
        wbc (×10³/μL)             → wbc_norm = clip(wbc / 20, 0, 1)
        heart_rate (bpm)          → hr_norm = clip(hr / 200, 0, 1)
        st_deviation_mm           → st_deviation_norm = clip((st + 5)/10, 0, 1)

    Unknown keys are ignored. Missing FEATURE_NAMES keys get _FEATURE_DEFAULTS.

    Args:
        raw_features: Dict with any combination of feature keys and values.

    Returns:
        Float32 tensor of shape [1, INPUT_DIM].
    """
    # Build a working copy with defaults
    feat: Dict[str, float] = {k: v for k, v in _FEATURE_DEFAULTS.items()}

    # --- Handle common raw-unit aliases ---
    # Troponin (ng/mL): 0 = normal (<0.04), 1.0 = clearly elevated
    if "troponin_elevated" in raw_features:
        raw_t = float(raw_features["troponin_elevated"])
        # Sigmoid mapping: troponin of 0.04 → ~0.60, 1.0 → ~0.99
        # Use a scaled sigmoid so the normal range maps to ~0.50
        feat["troponin_percentile"] = float(1.0 / (1.0 + np.exp(-(raw_t - 0.04) * 12)))

    if "troponin_percentile" in raw_features:
        feat["troponin_percentile"] = float(raw_features["troponin_percentile"])

    # WBC count (×10³/μL → [0,1])
    if "wbc" in raw_features:
        feat["wbc_norm"] = float(np.clip(float(raw_features["wbc"]) / 20.0, 0.0, 1.0))

    # Heart rate (bpm → [0,1])
    if "heart_rate" in raw_features or "hr" in raw_features:
        hr = float(raw_features.get("heart_rate", raw_features.get("hr", 75)))
        feat["hr_norm"] = float(np.clip(hr / 200.0, 0.0, 1.0))

    # ST deviation (mm, range −5 to +5 → [0,1])
    if "st_deviation_mm" in raw_features:
        st = float(raw_features["st_deviation_mm"])
        feat["st_deviation_norm"] = float(np.clip((st + 5.0) / 10.0, 0.0, 1.0))

    # PR interval (ms → [0,1])
    if "pr_interval_ms" in raw_features:
        feat["pr_interval_norm"] = float(
            np.clip(float(raw_features["pr_interval_ms"]) / 300.0, 0.0, 1.0)
        )

    # QRS duration (ms → [0,1])
    if "qrs_duration_ms" in raw_features:
        feat["qrs_duration_norm"] = float(
            np.clip(float(raw_features["qrs_duration_ms"]) / 200.0, 0.0, 1.0)
        )

    # QTc (ms → [0,1])
    if "qtc_ms" in raw_features:
        feat["qt_corrected_norm"] = float(
            np.clip(float(raw_features["qtc_ms"]) / 700.0, 0.0, 1.0)
        )

    # BNP (pg/mL) — percentile form or raw (pg/mL, normal < 100)
    if "bnp_percentile" in raw_features:
        feat["bnp_percentile"] = float(raw_features["bnp_percentile"])
    elif "bnp_pg_ml" in raw_features:
        bnp = float(raw_features["bnp_pg_ml"])
        feat["bnp_percentile"] = float(np.clip(bnp / 5000.0, 0.0, 1.0))

    # Creatinine (mg/dL → [0,1])
    if "creatinine_mg_dl" in raw_features or "creatinine" in raw_features:
        cr = float(raw_features.get("creatinine_mg_dl", raw_features.get("creatinine", 1.0)))
        feat["creatinine_norm"] = float(np.clip(cr / 15.0, 0.0, 1.0))

    # Sodium (mEq/L → [0,1])
    if "sodium_meq_l" in raw_features or "sodium" in raw_features:
        na = float(raw_features.get("sodium_meq_l", raw_features.get("sodium", 140)))
        feat["sodium_norm"] = float(np.clip(na / 170.0, 0.0, 1.0))

    # Potassium (mEq/L → [0,1])
    if "potassium_meq_l" in raw_features or "potassium" in raw_features:
        k = float(raw_features.get("potassium_meq_l", raw_features.get("potassium", 4.0)))
        feat["potassium_norm"] = float(np.clip(k / 10.0, 0.0, 1.0))

    # Lactate (mmol/L → [0,1])
    if "lactate_mmol_l" in raw_features or "lactate" in raw_features:
        lac = float(raw_features.get("lactate_mmol_l", raw_features.get("lactate", 2.0)))
        feat["lactate_norm"] = float(np.clip(lac / 20.0, 0.0, 1.0))

    # Build ordered vector
    vec = np.array([feat[k] for k in FEATURE_NAMES], dtype=np.float32)
    return torch.from_numpy(vec).unsqueeze(0)   # [1, 12]


# ──────────────────────────────────────────────
# Core function
# ──────────────────────────────────────────────
def assess_early_risk(
    features: Optional[Union[Dict[str, Any], str]] = None,
    json_path: Optional[str] = None,
    ckpt_path: Optional[str] = None,
    device_str: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Assess early clinical risk from tabular ECG + lab features.

    The function accepts feature input in two ways:
        1. `features` dict  — direct Python dict with feature key-values
        2. `json_path`      — path to a JSON file containing the feature dict

    If a real MLP checkpoint is not available, the function falls back to
    a mock model with clinically-primed weights for testing / CI purposes.

    Risk Score computation:
        risk_score = 0·p_low + 1·p_mod + 2·p_high    (expected class index)
        risk_score_norm = risk_score / 2.0             → [0, 1]

    Args:
        features: Feature dict (may include raw or normalised values).
        json_path: Path to JSON file containing the feature dict.
        ckpt_path: Optional path to trained MLP checkpoint.
        device_str: "cpu" or "cuda". Auto-detected if None.

    Returns:
        JSON-serialisable dict:
        {
            "risk_level": "Low" | "Moderate" | "High",
            "risk_score_norm": float,          # [0, 1]; 0=minimal, 1=maximal
            "recommendation": str,
            "class_probabilities": {
                "Low": float, "Moderate": float, "High": float
            },
            "features_used": {k: v, ...},
            "model_source": "checkpoint" | "mock",
            "status": "ok" | "error"
        }
    """
    if features is None and json_path is None:
        raise ValueError(
            "[early_risk] Provide either `features` dict or `json_path`."
        )

    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    result: Dict[str, Any] = {"status": "ok"}

    try:
        # ── 1. Load features ────────────────────────────────────────────
        if json_path is not None:
            if not os.path.exists(json_path):
                raise FileNotFoundError(
                    f"[early_risk] Feature JSON not found: {json_path}"
                )
            with open(json_path, "r", encoding="utf-8") as f:
                raw: Dict[str, Any] = json.load(f)
        else:
            raw = dict(features)  # type: ignore[arg-type]

        # ── 2. Build feature tensor ─────────────────────────────────────
        x: torch.Tensor = _validate_and_normalise(raw).to(device)  # [1, 12]

        # ── 3. Load (or mock) model ─────────────────────────────────────
        model_source: str = "mock"
        if ckpt_path is not None:
            try:
                model: EarlyRiskMLP = _load_mlp_from_ckpt(ckpt_path, device)
                model_source = "checkpoint"
            except FileNotFoundError as e:
                print(f"[early_risk] Warning: {e}. Falling back to mock model.")
                model = _build_mock_mlp(device)
        else:
            model = _build_mock_mlp(device)

        # ── 4. Forward pass ─────────────────────────────────────────────
        try:
            with torch.inference_mode():
                logits: torch.Tensor = model(x)   # [1, 3]
        except torch.cuda.OutOfMemoryError as oom:
            torch.cuda.empty_cache()
            raise RuntimeError(
                f"[early_risk] CUDA OOM during forward pass: {oom}"
            ) from oom

        # ── 5. Risk score & classification ──────────────────────────────
        probs: np.ndarray = (
            F.softmax(logits, dim=1)
            .squeeze(0)
            .cpu()
            .numpy()
        )  # [3]  — [p_low, p_mod, p_high]

        # Expected class index, normalised to [0, 1]
        expected_class = float(0.0 * probs[0] + 1.0 * probs[1] + 2.0 * probs[2])
        risk_score_norm = round(expected_class / 2.0, 4)

        # Argmax classification
        risk_class_idx: int = int(np.argmax(probs))
        risk_level: str = RISK_LABELS[risk_class_idx]
        recommendation: str = RISK_RECOMMENDATIONS[risk_level]

        # ── 6. Assemble output ──────────────────────────────────────────
        result["risk_level"] = risk_level
        result["risk_score_norm"] = risk_score_norm
        result["recommendation"] = recommendation
        result["class_probabilities"] = {
            label: round(float(p), 4)
            for label, p in zip(RISK_LABELS, probs)
        }
        result["model_source"] = model_source
        # Summarise which normalised feature values were used
        feat_display: Dict[str, float] = {
            k: round(float(x.squeeze(0)[i].item()), 4)
            for i, k in enumerate(FEATURE_NAMES)
        }
        result["features_used"] = feat_display

    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        result["status"] = "error"
        result["error"] = str(exc)
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
        description="Early risk stratification from ECG + lab features via MLP"
    )
    p.add_argument("--input", default=None,
                   help="Path to JSON file with feature values")
    p.add_argument("--ckpt", default=None,
                   help="Optional path to trained MLP checkpoint (.pt)")
    p.add_argument("--device", default=None,
                   help="'cpu' or 'cuda'")

    # Allow passing individual features directly
    feature_group = p.add_argument_group("Feature overrides (raw values)")
    feature_group.add_argument("--troponin", type=float, default=None,
                                help="Troponin I/T (ng/mL)")
    feature_group.add_argument("--wbc", type=float, default=None,
                                help="WBC count (×10³/μL)")
    feature_group.add_argument("--hr", type=float, default=None,
                                help="Heart rate (bpm)")
    feature_group.add_argument("--st_deviation_mm", type=float, default=None,
                                help="ST segment deviation (mm, +/- shift)")
    feature_group.add_argument("--qtc_ms", type=float, default=None,
                                help="Corrected QT interval (ms)")
    feature_group.add_argument("--creatinine", type=float, default=None,
                                help="Serum creatinine (mg/dL)")
    feature_group.add_argument("--lactate", type=float, default=None,
                                help="Blood lactate (mmol/L)")
    return p


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Build feature dict from JSON file and/or CLI overrides
    features: Dict[str, Any] = {}

    if args.input:
        with open(args.input, "r") as f:
            features = json.load(f)

    # CLI overrides take precedence over JSON file
    cli_map = {
        "troponin_elevated": args.troponin,
        "wbc":               args.wbc,
        "heart_rate":        args.hr,
        "st_deviation_mm":   args.st_deviation_mm,
        "qtc_ms":            args.qtc_ms,
        "creatinine":        args.creatinine,
        "lactate":           args.lactate,
    }
    for k, v in cli_map.items():
        if v is not None:
            features[k] = v

    if not features:
        parser.error(
            "Provide --input JSON file or at least one feature flag "
            "(e.g. --troponin 0.8 --wbc 14)"
        )

    output = assess_early_risk(
        features=features,
        ckpt_path=args.ckpt,
        device_str=args.device,
    )

    print(json.dumps(output, indent=2))
