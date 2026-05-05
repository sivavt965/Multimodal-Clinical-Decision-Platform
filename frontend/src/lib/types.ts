// =============================================================================
// types.ts — TypeScript Interfaces for the Clinical Decision Support Platform
// Mirrors schema.sql exactly. Import into Next.js pages / API routes.
// =============================================================================

// ---------------------------------------------------------------------------
// Primitive enums (mirror Postgres ENUM types)
// ---------------------------------------------------------------------------
/** Derived categorical risk badge — thresholded from raw probability.
 *  probability < 0.05          → 'Unlikely'      (Gray)
 *  0.05 ≤ probability ≤ 0.15  → 'Monitor'       (Yellow)
 *  probability > 0.15          → 'Elevated Risk' (Red)
 */
export type RiskBadge = 'Unlikely' | 'Monitor' | 'Elevated Risk';

/** MC Dropout uncertainty category from mc_dropout.py */
export type UncertaintyLevel =
  | 'Low Uncertainty'
  | 'Moderate Uncertainty'
  | 'High Uncertainty';

/** Phase A (tabular) risk stratification from early_risk_inference.py */
export type PhaseARisk = 'Low' | 'Moderate' | 'High';

export type Sex = 'M' | 'F' | 'Other';

/** 8 CXR finding labels (matching LABEL_COLUMNS in all inference scripts) */
export type CXRLabel =
  | 'No Finding'
  | 'Enlarged Cardiomediastinum'
  | 'Cardiomegaly'
  | 'Lung Opacity'
  | 'Lung Lesion'
  | 'Edema'
  | 'Consolidation'
  | 'Pneumonia'
  | 'Atelectasis'
  | 'Pneumothorax'
  | 'Pleural Effusion'
  | 'Pleural Other'
  | 'Fracture'
  | 'Support Devices';

export type ConsultationRole = 'ward_doctor' | 'radiologist';

export type MessageType = 'text' | 'annotation' | 'viewport_sync' | 'finding_flag';

/** RBAC roles for the platform (matches user_role_enum in 002_users_audit_outcome.sql) */
export type UserRole = 'radiologist' | 'ward_doctor' | 'clinical_admin' | 'system_admin';

export type UserStatus = 'active' | 'inactive' | 'suspended';

/** Final disposition for a completed case */
export type CaseOutcome =
  | 'General Ward'
  | 'ICU Transfer'
  | 'Step-Down Unit'
  | 'Discharged'
  | 'Deceased'
  | 'Other';

export interface PlatformUser {
  id: string;
  email: string;
  full_name: string;
  role: UserRole;
  status: UserStatus;
  last_active_at: string | null;
  created_at: string;
  updated_at: string;
}

export interface AuditLogEntry {
  id: number;
  user_id: string | null;
  user_role: UserRole | null;
  action: string;
  target_type: string | null;
  target_id: string | null;
  metadata: Record<string, unknown>;
  created_at: string;
}

// ---------------------------------------------------------------------------
// Phase A — ECG data structure (raw clinical units, pre-normalisation)
// Maps to the 7 cardiac features in early_risk_inference.py FEATURE_NAMES.
// Stored as JSONB in cases.ecg_data
// ---------------------------------------------------------------------------
export interface ECGData {
  /** Heart rate in beats per minute */
  heart_rate: number;

  /** PR interval in milliseconds (normal: 120–200 ms) */
  pr_interval_ms: number;

  /** QRS complex duration in milliseconds (normal: 70–120 ms) */
  qrs_duration_ms: number;

  /** Corrected QT interval in milliseconds (normal: 350–450 ms) */
  qtc_ms: number;

  /** ST segment deviation in mm — positive = elevation, negative = depression */
  st_deviation_mm: number;

  /** Free-text rhythm interpretation (e.g., "Normal Sinus Rhythm", "AFib") */
  rhythm_interpretation: string;

  /** ISO 8601 timestamp when the ECG was acquired */
  acquired_at: string;
}

// ---------------------------------------------------------------------------
// Phase A — Lab values (raw clinical units, pre-normalisation)
// Maps to the 5 lab features in early_risk_inference.py FEATURE_NAMES.
// Stored as JSONB in cases.lab_data
// ---------------------------------------------------------------------------
export interface LabData {
  /** Troponin I or T in ng/mL. Normal < 0.04 ng/mL */
  troponin_ng_ml: number;

  /** B-type Natriuretic Peptide in pg/mL. Normal < 100 pg/mL */
  bnp_pg_ml: number;

  /** White Blood Cell count in ×10³/μL (thousands per microlitre). Normal: 4.5–11 */
  wbc_count: number;

  /** Serum creatinine in mg/dL. Normal: 0.7–1.3 mg/dL */
  creatinine_mg_dl: number;

  /** Serum sodium in mEq/L. Normal: 136–145 mEq/L */
  sodium_meq_l: number;

  /** Serum potassium in mEq/L. Normal: 3.5–5.0 mEq/L */
  potassium_meq_l: number;

  /** Blood lactate in mmol/L. Normal: 0.5–2.0 mmol/L */
  lactate_mmol_l: number;

  /** ISO 8601 timestamp when samples were collected */
  collected_at: string;
}

// ---------------------------------------------------------------------------
// Viewport state — shared DICOM viewer memory between both users
// Stored as JSONB in consultations.viewport_state
// ---------------------------------------------------------------------------
export interface AnnotationMark {
  id: string;                   // UUID
  x: number;                   // % from left (0–100)
  y: number;                   // % from top  (0–100)
  width: number;               // % of image width
  height: number;              // % of image height
  label: string;               // e.g., "Effusion margin", "Opacity"
  color: string;               // Hex color string
  author_role: ConsultationRole;
  created_at: string;          // ISO 8601
}

export interface ViewportState {
  zoom: number;                // 1.0 = 100%, 2.0 = 200%
  contrast: number;            // 0–200; 100 = normal
  brightness: number;          // 0–200; 100 = normal
  window_center: number;       // DICOM windowing: HU value at center
  window_width: number;        // DICOM windowing: range of HU values displayed
  gradcam_alpha?: number;      // Opacity of Grad-CAM overlay
  annotations: AnnotationMark[];
}

// ---------------------------------------------------------------------------
// Message — one entry in consultations.messages JSONB array
// ---------------------------------------------------------------------------
export interface ConsultationMessage {
  id: string;                     // UUID
  role: ConsultationRole;
  type: MessageType;
  /** For type='text': plain string.
   *  For type='annotation': AnnotationMark object.
   *  For type='viewport_sync': ViewportState object.
   *  For type='finding_flag': { label: CXRLabel, risk_badge: RiskBadge } */
  content: string | AnnotationMark | ViewportState | FindingFlag;
  sent_at: string;                // ISO 8601
  read: boolean;
}

export interface FindingFlag {
  label: CXRLabel;
  risk_badge: RiskBadge;
  probability: number;
  note?: string;
}

// ---------------------------------------------------------------------------
// TABLE: patients
// ---------------------------------------------------------------------------
export interface Patient {
  id: string;                     // UUID
  mrn: string;
  first_name: string;
  last_name: string;
  date_of_birth: string;          // ISO 8601 date string
  sex: Sex;
  age_at_admission: number;
  mimic_subject_id: number | null;
  created_at: string;
  updated_at: string;
}

// ---------------------------------------------------------------------------
// TABLE: cases
// ---------------------------------------------------------------------------
export interface Case {
  id: string;                     // UUID
  patient_id: string;             // FK → patients.id

  admitted_at: string;            // ISO 8601
  discharged_at: string | null;

  ecg_data: ECGData;
  lab_data: LabData;

  // Phase A outputs
  phase_a_risk_level: PhaseARisk | null;
  phase_a_risk_score: number | null;   // [0, 1]
  phase_a_recommendation: string | null;
  phase_a_run_at: string | null;       // ISO 8601

  // CXR image
  cxr_dicom_url: string | null;
  cxr_acquired_at: string | null;      // ISO 8601

  // Grad-CAM heatmap (populated by background inference task)
  cxr_heatmap_url: string | null;
  cxr_heatmap_label: string | null;

  mimic_study_id: number | null;

  /** Full 50-lab dictionary keyed by MIMIC-IV itemid, populated from wizard.
   *  Values are raw clinical units (e.g., mg/dL, mmol/L).
   *  Missing labs are omitted (not null). */
  labs_raw?: Record<string, number>;

  created_at: string;
  updated_at: string;
}

// ---------------------------------------------------------------------------
// TABLE: predictions
// One row per CXR label per inference run.
// ---------------------------------------------------------------------------
export interface Prediction {
  id: string;                        // UUID
  case_id: string;                   // FK → cases.id

  // Inference configuration
  model_checkpoint: string;          // e.g., 'baseline_best.pt'
  temperature: number;               // e.g., 1.3
  inference_run_at: string;          // ISO 8601

  // Per-label CXR output
  label: CXRLabel;

  /** Raw calibrated probability in [0, 1] from temperature_scale(logits, T).
   *  NOT a percentage — multiply by 100 for display. */
  probability: number;

  /** DERIVED field — computed from probability via threshold mapping.
   *  Stored in DB as a GENERATED ALWAYS AS column.
   *  probability < 0.05          → 'Unlikely'      (render Gray)
   *  0.05 ≤ probability ≤ 0.15  → 'Monitor'       (render Yellow)
   *  probability > 0.15          → 'Elevated Risk' (render Red)
   */
  risk_badge: RiskBadge;

  // MC Dropout uncertainty
  uncertainty_level: UncertaintyLevel | null;
  mean_variance: number | null;
  std_dev: number | null;
  mc_passes: number;                 // default 20

  /** Absolute URL or local path to the Grad-CAM overlay PNG.
   *  Local paths follow convention: /mock-data/heatmaps/case_{n}_{label}.png
   *  Production: GCS signed URL or /api/gradcam/{case_id}/{label} */
  gradcam_url: string | null;
  gradcam_alpha: number;             // default 0.45
}

// ---------------------------------------------------------------------------
// TABLE: consultations
// ---------------------------------------------------------------------------
export interface Consultation {
  id: string;                        // UUID
  case_id: string;                   // FK → cases.id (UNIQUE)

  ward_doctor_id: string;            // Supabase Auth UID
  radiologist_id: string | null;     // NULL until accepted

  is_open: boolean;
  opened_at: string;                 // ISO 8601
  closed_at: string | null;
  urgency_flag: boolean;

  messages: ConsultationMessage[];
  viewport_state: ViewportState;

  ward_doctor_last_view: string | null;     // ISO 8601
  radiologist_last_view: string | null;     // ISO 8601

  created_at: string;
  updated_at: string;
}

// ---------------------------------------------------------------------------
// API Response wrappers — used by Next.js API routes
// ---------------------------------------------------------------------------
/** Full case payload returned by GET /api/cases/[id] */
export interface CaseDetail {
  patient: Patient;
  case: Case;
  predictions: Prediction[];         // 8 rows, one per CXR label
  consultation: Consultation | null;
}

/** Alias used by BeforeAfterTab */
export type CaseResponse = CaseDetail;

/** Summary row for the case list / dashboard table */
export interface CaseSummary {
  case_id: string;
  patient_name: string;             // first_name + last_name
  mrn: string;
  admitted_at: string;
  phase_a_risk_level: PhaseARisk | null;
  top_finding_label: CXRLabel | null;
  top_finding_badge: RiskBadge | null;
  top_finding_probability: number | null;
  consultation_open: boolean;
  urgency_flag: boolean;
  reanalysis_requested?: boolean;
  similarity_score?: number | null;  // [0, 100] — populated by /similar endpoint only
  cxr_dicom_url?: string | null;
  /**
   * Expert-annotated CheXpert findings from the Symile-MIMIC ground-truth CSV.
   * A real chest X-ray case usually has several positive findings; the
   * similar-cases card displays these instead of just the model's argmax.
   */
  ground_truth_findings?: string[];
}

// ---------------------------------------------------------------------------
// Phase A engine raw output — mirrors assess_early_risk() return dict
// Used when calling the Python inference engine directly from Next.js API route
// ---------------------------------------------------------------------------
export interface PhaseAResult {
  status: 'ok' | 'error';
  risk_level: PhaseARisk;
  risk_score_norm: number;           // [0, 1]
  recommendation: string;
  class_probabilities: {
    Low: number;
    Moderate: number;
    High: number;
  };
  model_source: 'checkpoint' | 'mock';
  features_used: Record<string, number>;
  error?: string;
}

// ---------------------------------------------------------------------------
// Phase B engine raw output — mirrors predict() return dict from cxr_inference.py
// ---------------------------------------------------------------------------
export interface PhaseBFinding {
  rank: number;
  label: CXRLabel;
  probability_pct: number;          // 0–100
}

export interface PhaseBResult {
  status: 'ok' | 'error';
  image_path: string;
  temperature: number;
  device: string;
  top_findings: PhaseBFinding[];
  all_findings: Record<CXRLabel, number>;  // label → probability_pct
  error?: string;
}

// ---------------------------------------------------------------------------
// MC Dropout raw output — mirrors quantify_uncertainty() return dict
// ---------------------------------------------------------------------------
export interface MCDropoutResult {
  status: 'ok' | 'error';
  image_path: string;
  n_passes: number;
  seed: number | null;
  device: string;
  uncertainty_level: UncertaintyLevel;
  mean_variance: number;
  std_dev: number;
  per_class_variance: Record<CXRLabel, number>;
  per_class_mean_prob: Record<CXRLabel, number>;
  error?: string;
}

// ---------------------------------------------------------------------------
// Utility: risk badge colour map — for use in UI badge components
// ---------------------------------------------------------------------------
export const RISK_BADGE_COLORS: Record<RiskBadge, { bg: string; text: string; border: string }> = {
  'Unlikely':      { bg: 'bg-gray-100',   text: 'text-gray-700',  border: 'border-gray-300'  },
  'Monitor':       { bg: 'bg-amber-100',  text: 'text-amber-900', border: 'border-amber-400' },
  'Elevated Risk': { bg: 'bg-red-100',    text: 'text-red-900',   border: 'border-red-500'   },
} as const;

/** Compute risk_badge from raw probability [0, 1] — mirrors SQL GENERATED ALWAYS AS logic */
export function computeRiskBadge(probability: number): RiskBadge {
  if (probability < 0.05)  return 'Unlikely';
  if (probability <= 0.15) return 'Monitor';
  return 'Elevated Risk';
}

export const CLINICAL_NORMAL_RANGES = {
  troponin: '<0.04',
  bnp: '<100',
  wbc: '4.5-11.0',
  creatinine: '0.6-1.2',
  sodium: '135-145',
  potassium: '3.5-5.0',
  lactate: '0.5-2.2',
  pr_interval: '120-200',
  qrs: '<120',
  qtc: '<440'
} as const;
