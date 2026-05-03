# =============================================================================
# schemas.py — Pydantic models for the Clinical Decision Support Platform
# Mirrors frontend/src/lib/types.ts exactly.
# =============================================================================
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Union

# ---------------------------------------------------------------------------
# Primitive enums (mirror Postgres ENUM types / TypeScript literal unions)
# ---------------------------------------------------------------------------
RiskBadge = Literal["Unlikely", "Monitor", "Elevated Risk"]
UncertaintyLevel = Literal["Low Uncertainty", "Moderate Uncertainty", "High Uncertainty"]
PhaseARisk = Literal["Low", "Moderate", "High"]
Sex = Literal["M", "F", "Other"]
CXRLabel = Literal[
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
ConsultationRole = Literal["ward_doctor", "radiologist"]
MessageType = Literal["text", "annotation", "viewport_sync", "finding_flag"]


# ---------------------------------------------------------------------------
# Phase A — ECG & Lab data structures
# ---------------------------------------------------------------------------
class ECGData(BaseModel):
    """7 cardiac features from early_risk_inference.py FEATURE_NAMES."""

    heart_rate: float
    pr_interval_ms: float
    qrs_duration_ms: float
    qtc_ms: float
    st_deviation_mm: float
    rhythm_interpretation: str
    acquired_at: str  # ISO 8601


class LabData(BaseModel):
    """Lab values (raw clinical units, pre-normalisation)."""

    troponin_ng_ml: float
    bnp_pg_ml: float
    wbc_count: float
    creatinine_mg_dl: float
    sodium_meq_l: float
    potassium_meq_l: float
    lactate_mmol_l: float
    collected_at: str  # ISO 8601


# ---------------------------------------------------------------------------
# Viewport / Annotation structures
# ---------------------------------------------------------------------------
class AnnotationMark(BaseModel):
    id: str
    x: float
    y: float
    width: float
    height: float
    label: str
    color: str
    author_role: ConsultationRole
    created_at: str


class ViewportState(BaseModel):
    zoom: float
    contrast: float
    brightness: float
    window_center: float
    window_width: float
    gradcam_alpha: Optional[float] = None
    annotations: List[AnnotationMark] = []


# ---------------------------------------------------------------------------
# Consultation messages
# ---------------------------------------------------------------------------
class FindingFlag(BaseModel):
    label: CXRLabel
    risk_badge: RiskBadge
    probability: float
    note: Optional[str] = None


class ConsultationMessage(BaseModel):
    id: str
    role: ConsultationRole
    type: MessageType
    content: Union[str, AnnotationMark, ViewportState, FindingFlag]
    sent_at: str  # ISO 8601
    read: bool


# ---------------------------------------------------------------------------
# Core domain models
# ---------------------------------------------------------------------------
class Patient(BaseModel):
    id: str
    mrn: str
    first_name: str
    last_name: str
    date_of_birth: str
    sex: Sex
    age_at_admission: int
    mimic_subject_id: Optional[int] = None
    created_at: str
    updated_at: str


class Case(BaseModel):
    id: str
    patient_id: str
    admitted_at: str
    discharged_at: Optional[str] = None

    ecg_data: ECGData
    lab_data: LabData

    # Full 50-lab dictionary keyed by MIMIC-IV itemid (e.g. {"50912": 1.2, ...})
    # Populated from the ingestion wizard; missing labs are simply absent.
    labs_raw: Optional[Dict[str, float]] = None

    # Phase A outputs
    phase_a_risk_level: Optional[PhaseARisk] = None
    phase_a_risk_score: Optional[float] = None
    phase_a_recommendation: Optional[str] = None
    phase_a_run_at: Optional[str] = None

    # CXR image
    cxr_dicom_url: Optional[str] = None
    cxr_acquired_at: Optional[str] = None

    # Grad-CAM heatmap (populated by background inference task)
    cxr_heatmap_url: Optional[str] = None
    cxr_heatmap_label: Optional[str] = None

    mimic_study_id: Optional[int] = None
    created_at: str
    updated_at: str


class Prediction(BaseModel):
    id: str
    case_id: str
    model_checkpoint: str
    temperature: float = 1.0
    inference_run_at: str

    label: CXRLabel
    probability: float
    risk_badge: RiskBadge

    # MC Dropout uncertainty
    uncertainty_level: Optional[UncertaintyLevel] = None
    mean_variance: Optional[float] = None
    std_dev: Optional[float] = None
    mc_passes: int = 20

    gradcam_url: Optional[str] = None
    gradcam_alpha: float = 0.45


class Consultation(BaseModel):
    id: str
    case_id: str
    ward_doctor_id: str = "00000000-0000-0000-0000-000000000000"
    radiologist_id: Optional[str] = None
    is_open: bool = True
    opened_at: Optional[str] = None
    closed_at: Optional[str] = None
    urgency_flag: bool = False
    messages: List[ConsultationMessage] = []
    viewport_state: Optional[ViewportState] = None
    ward_doctor_last_view: Optional[str] = None
    radiologist_last_view: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


# ---------------------------------------------------------------------------
# API response wrappers
# ---------------------------------------------------------------------------
class CaseDetail(BaseModel):
    """Full case payload — GET /api/cases/{case_id}"""

    patient: Patient
    case: Case
    predictions: List[Prediction]
    consultation: Optional[Consultation] = None


class CaseSummary(BaseModel):
    """Summary row for the dashboard table — GET /api/cases"""

    case_id: str
    patient_name: str
    mrn: str
    admitted_at: str
    phase_a_risk_level: Optional[PhaseARisk] = None
    top_finding_label: Optional[CXRLabel] = None
    top_finding_badge: Optional[RiskBadge] = None
    top_finding_probability: Optional[float] = None
    consultation_open: bool = False
    urgency_flag: bool = False
    similarity_score: Optional[float] = None
    cxr_dicom_url: Optional[str] = None


# ---------------------------------------------------------------------------
# Inference engine result schemas (Phase A / Phase B / MC Dropout)
# These mirror the raw Python dict outputs from the inference scripts.
# ---------------------------------------------------------------------------
class PhaseAResult(BaseModel):
    """Mirrors assess_early_risk() return dict from early_risk_inference.py."""

    status: Literal["ok", "error"]
    risk_level: PhaseARisk
    risk_score_norm: float = Field(..., ge=0, le=1)
    recommendation: str
    class_probabilities: Dict[PhaseARisk, float]
    model_source: Literal["checkpoint", "mock"]
    features_used: Dict[str, float]
    error: Optional[str] = None


class PhaseBFinding(BaseModel):
    rank: int
    label: CXRLabel
    probability_pct: float = Field(..., ge=0, le=100)


class PhaseBResult(BaseModel):
    """Mirrors predict() return dict from cxr_inference.py."""

    status: Literal["ok", "error"]
    image_path: str
    temperature: float
    device: str
    top_findings: List[PhaseBFinding]
    all_findings: Dict[CXRLabel, float]  # label → probability_pct
    error: Optional[str] = None


class MCDropoutResult(BaseModel):
    """Mirrors quantify_uncertainty() return dict from mc_dropout.py."""

    status: Literal["ok", "error"]
    image_path: str
    n_passes: int
    seed: Optional[int] = None
    device: str
    uncertainty_level: UncertaintyLevel
    mean_variance: float
    std_dev: float
    per_class_variance: Dict[CXRLabel, float]
    per_class_mean_prob: Dict[CXRLabel, float]
    error: Optional[str] = None
