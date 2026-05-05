# =============================================================================
# engine/early_risk_inference.py
# Phase A early-risk inference for the Clinical Decision Support Platform.
#
# Entry-point: assess_early_risk(ecg_data, lab_data, labs_raw=None) -> dict
#
# Two paths:
#   1. MLP checkpoint (early_risk_model.pt in backend/) — loaded once at first
#      call via _get_model(); produces soft class probabilities over
#      {Low, Moderate, High} from 7 ECG + 50 lab percentile + 50 missingness
#      features (107-dim input).
#   2. Rules-based fallback — used when no checkpoint is present. Combines the
#      five numeric ECG features with the seven key lab biomarkers. Deterministic
#      and always available, so the front-end is never blocked on training.
#
# Return schema mirrors PhaseAResult in schemas.py:
#   status, risk_level, risk_score_norm, recommendation,
#   class_probabilities, model_source, features_used, error
# =============================================================================
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "heart_rate",
    "pr_interval_ms",
    "qrs_duration_ms",
    "qtc_ms",
    "st_deviation_mm",
    "rhythm_interpretation",
    "acquired_at",
]

_ECG_NUMERIC = FEATURE_NAMES[:5]  # continuous features fed into the MLP

# 50 MIMIC-IV lab itemids used by the Symile-MIMIC dataset (ordered list used
# to build the percentile vector at inference time — must match the order in
# code/constants.py and data_npy/).
LAB_ITEMIDS: list[str] = [
    "50868", "50882", "50893", "50902", "50912", "50931", "50960",
    "50970", "50971", "50983", "51006", "51221", "51222", "51237",
    "51248", "51249", "51250", "51265", "51274", "51275", "51277",
    "51279", "51301", "51491", "51498", "50809", "50813", "50818",
    "50820", "50821", "50822", "50824", "50861", "50863", "50878",
    "50885", "50916", "50953", "51003", "51200", "51214", "51240",
    "51244", "51256", "51254", "51255", "51257", "51478", "51484",
    "51487",
]

_CHECKPOINT_PATH = Path(
    os.getenv("EARLY_RISK_CHECKPOINT_PATH",
              str(Path(__file__).resolve().parent.parent / "early_risk_model.pt"))
)

_INPUT_DIM = len(_ECG_NUMERIC) + 2 * len(LAB_ITEMIDS)  # 5 + 100 = 105

# ---------------------------------------------------------------------------
# MLP model definition (used when checkpoint is present)
# ---------------------------------------------------------------------------
def _build_mlp():
    """Return an nn.Module for the Phase A risk classifier."""
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(_INPUT_DIM, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 3),   # logits for Low / Moderate / High
    )


# ---------------------------------------------------------------------------
# Singleton loader
# ---------------------------------------------------------------------------
_model = None
_model_lock = threading.Lock()


def _get_model():
    """Return the loaded MLP, or None if the checkpoint is not present."""
    global _model
    with _model_lock:
        if _model is not None:
            return _model
        if not _CHECKPOINT_PATH.exists():
            logger.info("[early_risk] No checkpoint at %s — using rules-based fallback.", _CHECKPOINT_PATH)
            return None
        try:
            import torch
            net = _build_mlp()
            state = torch.load(_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
            net.load_state_dict(state)
            net.eval()
            _model = net
            logger.info("[early_risk] MLP checkpoint loaded from %s.", _CHECKPOINT_PATH)
            return _model
        except Exception as exc:
            logger.warning("[early_risk] Checkpoint load failed (%s) — falling back to rules.", exc)
            return None


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------
def _ecg_numeric_vector(ecg_data: dict) -> list[float]:
    """Return the five numeric ECG features in FEATURE_NAMES order."""
    return [float(ecg_data.get(k) or 0.0) for k in _ECG_NUMERIC]


def _labs_percentile_vector(labs_raw: dict) -> tuple[list[float], list[float]]:
    """Return (percentile_values, missingness) vectors over the 50 ordered itemids.

    labs_raw keys are MIMIC-IV itemid strings; values are raw clinical units.
    Missing values are imputed with 0.5 (training-mean percentile placeholder)
    and flagged in the missingness vector.
    """
    vals, miss = [], []
    for iid in LAB_ITEMIDS:
        v = labs_raw.get(str(iid)) if labs_raw else None
        if v is None:
            vals.append(0.5)
            miss.append(1.0)
        else:
            vals.append(float(v))
            miss.append(0.0)
    return vals, miss


# ---------------------------------------------------------------------------
# Rules-based fallback scorer
# ---------------------------------------------------------------------------
_RECOMMENDATIONS = {
    "High": (
        "Elevated cardiac and metabolic markers. "
        "Recommend urgent imaging and cardiology review."
    ),
    "Moderate": (
        "Moderate risk indicators detected. "
        "Consider early follow-up imaging and repeat labs."
    ),
    "Low": (
        "Low-risk profile based on initial labs and ECG. "
        "Continue standard monitoring."
    ),
}


def _rules_based_risk(ecg_data: dict, lab_data: dict) -> tuple[str, float]:
    """Deterministic risk scorer combining ECG + lab heuristics.

    Returns (risk_level, risk_score_norm) where risk_score_norm ∈ [0, 1].
    """
    def _lv(d: dict, key: str) -> float:
        v = d.get(key)
        return float(v) if v is not None else 0.0

    score = 0.0

    # -- Lab rules (ordered by clinical weight) --------------------------------
    if _lv(lab_data, "troponin_ng_ml") > 0.04:
        score += 0.40   # acute myocardial injury marker
    if _lv(lab_data, "lactate_mmol_l") > 2.0:
        score += 0.20   # tissue hypoperfusion
    if _lv(lab_data, "creatinine_mg_dl") > 1.3:
        score += 0.15   # renal dysfunction
    k = _lv(lab_data, "potassium_meq_l")
    if k > 5.0 or (0 < k < 3.5):
        score += 0.10   # electrolyte imbalance
    na = _lv(lab_data, "sodium_meq_l")
    if 0 < na < 136 or na > 145:
        score += 0.10   # dysnatraemia

    # -- ECG rules -------------------------------------------------------------
    hr     = _lv(ecg_data, "heart_rate")
    qtc    = _lv(ecg_data, "qtc_ms")
    qrs    = _lv(ecg_data, "qrs_duration_ms")
    st_dev = _lv(ecg_data, "st_deviation_mm")

    if hr > 0:
        if hr > 130 or hr < 40:
            score += 0.20   # severe tachycardia / bradycardia
        elif hr > 100 or hr < 50:
            score += 0.10   # mild tachycardia / bradycardia

    if abs(st_dev) > 1.0:
        score += 0.20   # significant ST deviation

    if qtc > 480:
        score += 0.10   # prolonged QTc
    if qrs > 120:
        score += 0.05   # wide QRS

    score = min(score, 1.0)
    level = "High" if score >= 0.4 else "Moderate" if score >= 0.2 else "Low"
    return level, score


def _score_to_probs(score: float) -> Dict[str, float]:
    """Convert a continuous risk score to soft class probabilities."""
    # Logits designed so Low wins at score→0, High wins at score→1
    logits = np.array([
        -score * 4.0,          # Low
        (score - 0.3) * 2.0,   # Moderate peaks around 0.3
        (score - 0.4) * 4.0,   # High
    ])
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    return {
        "Low":      float(probs[0]),
        "Moderate": float(probs[1]),
        "High":     float(probs[2]),
    }


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------
def assess_early_risk(
    ecg_data: dict,
    lab_data: dict,
    labs_raw: Optional[dict] = None,
) -> Dict[str, Any]:
    """Run Phase A risk inference and return a PhaseAResult-compatible dict.

    Args:
        ecg_data:  ECGData fields (heart_rate, pr_interval_ms, …).
        lab_data:  LabData fields (troponin_ng_ml, lactate_mmol_l, …).
        labs_raw:  Optional raw MIMIC-IV itemid→value dict for MLP path.

    Returns a dict with keys:
        status, risk_level, risk_score_norm, recommendation,
        class_probabilities, model_source, features_used, error
    """
    try:
        model = _get_model()

        if model is not None and labs_raw is not None:
            # ── MLP path ────────────────────────────────────────────────────
            import torch
            ecg_vec          = _ecg_numeric_vector(ecg_data)
            labs_pct, miss   = _labs_percentile_vector(labs_raw)
            x = torch.tensor(
                [ecg_vec + labs_pct + miss],
                dtype=torch.float32,
            )
            with torch.no_grad():
                logits = model(x)[0]
                probs  = torch.softmax(logits, dim=0).tolist()

            class_probs = {
                "Low":      probs[0],
                "Moderate": probs[1],
                "High":     probs[2],
            }
            idx   = int(np.argmax(probs))
            level = ["Low", "Moderate", "High"][idx]
            score = probs[2] + 0.5 * probs[1]   # weighted score for display
            source = "checkpoint"

        else:
            # ── Rules-based fallback ─────────────────────────────────────────
            level, score = _rules_based_risk(ecg_data, lab_data)
            class_probs  = _score_to_probs(score)
            source       = "mock"

        ecg_num = dict(zip(_ECG_NUMERIC, _ecg_numeric_vector(ecg_data)))
        features_used = {
            **ecg_num,
            "troponin_ng_ml":   float(lab_data.get("troponin_ng_ml") or 0),
            "creatinine_mg_dl": float(lab_data.get("creatinine_mg_dl") or 0),
            "lactate_mmol_l":   float(lab_data.get("lactate_mmol_l") or 0),
            "potassium_meq_l":  float(lab_data.get("potassium_meq_l") or 0),
            "sodium_meq_l":     float(lab_data.get("sodium_meq_l") or 0),
        }

        return {
            "status":             "ok",
            "risk_level":         level,
            "risk_score_norm":    round(min(score, 1.0), 4),
            "recommendation":     _RECOMMENDATIONS[level],
            "class_probabilities": class_probs,
            "model_source":       source,
            "features_used":      features_used,
            "error":              None,
        }

    except Exception as exc:
        logger.exception("[early_risk] assess_early_risk failed: %s", exc)
        return {
            "status":             "error",
            "risk_level":         "Low",
            "risk_score_norm":    0.0,
            "recommendation":     "Risk assessment unavailable. Please review manually.",
            "class_probabilities": {"Low": 1.0, "Moderate": 0.0, "High": 0.0},
            "model_source":       "mock",
            "features_used":      {},
            "error":              str(exc),
        }
