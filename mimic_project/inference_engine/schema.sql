-- =============================================================================
-- schema.sql — Supabase Postgres
-- Multimodal Clinical Decision Support Platform
-- =============================================================================
-- Tables:
--   patients       — demographic/identity record
--   cases          — one admission; carries Phase-A ECG+Lab JSON
--   predictions    — Phase-B CXR inference outputs per label per case
--   consultations  — dual-user thread: Ward Doctor ↔ Radiologist
--
-- Risk badge thresholds (enforced via CHECK + application logic):
--   probability < 0.05          → 'Unlikely'     (Gray)
--   0.05 ≤ probability ≤ 0.15   → 'Monitor'      (Yellow)
--   probability > 0.15          → 'Elevated Risk' (Red)
--
-- Uncertainty levels from mc_dropout.py:
--   'Low Uncertainty' | 'Moderate Uncertainty' | 'High Uncertainty'
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Extensions
-- ---------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS "pgcrypto";   -- gen_random_uuid()

-- ---------------------------------------------------------------------------
-- ENUM types
-- ---------------------------------------------------------------------------
CREATE TYPE risk_badge_enum AS ENUM (
    'Unlikely',       -- probability < 0.05  → Gray
    'Monitor',        -- 0.05 ≤ p ≤ 0.15    → Yellow
    'Elevated Risk'   -- probability > 0.15  → Red
);

CREATE TYPE uncertainty_level_enum AS ENUM (
    'Low Uncertainty',
    'Moderate Uncertainty',
    'High Uncertainty'
);

CREATE TYPE phase_a_risk_enum AS ENUM (
    'Low',
    'Moderate',
    'High'
);

CREATE TYPE sex_enum AS ENUM ('M', 'F', 'Other');

CREATE TYPE cxr_label_enum AS ENUM (
    'Cardiomegaly',
    'Pleural Effusion',
    'Edema',
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',
    'Consolidation',
    'Support Devices'
);

CREATE TYPE consultation_role_enum AS ENUM (
    'ward_doctor',
    'radiologist'
);

CREATE TYPE message_type_enum AS ENUM (
    'text',
    'annotation',
    'viewport_sync',
    'finding_flag'
);

-- ---------------------------------------------------------------------------
-- TABLE: patients
-- One row per unique patient identity.
-- ---------------------------------------------------------------------------
CREATE TABLE patients (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Identity
    mrn                 VARCHAR(20)     NOT NULL UNIQUE,   -- Medical Record Number
    first_name          VARCHAR(100)    NOT NULL,
    last_name           VARCHAR(100)    NOT NULL,
    date_of_birth       DATE            NOT NULL,
    sex                 sex_enum        NOT NULL,

    -- Derived / computed at insert (no PII risk — stored for UI display)
    age_at_admission    SMALLINT        NOT NULL CHECK (age_at_admission BETWEEN 0 AND 130),

    -- MIMIC linkage (nullable for non-MIMIC patients)
    mimic_subject_id    BIGINT          UNIQUE,

    -- Audit
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_patients_mrn ON patients (mrn);
CREATE INDEX idx_patients_mimic ON patients (mimic_subject_id);

-- ---------------------------------------------------------------------------
-- TABLE: cases
-- One row per hospital admission/encounter. Carries the Phase-A tabular data.
--
-- ecg_data  JSONB — mirrors ECGData TypeScript interface (see types.ts)
-- lab_data  JSONB — mirrors LabData TypeScript interface (see types.ts)
-- ---------------------------------------------------------------------------
CREATE TABLE cases (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id              UUID            NOT NULL REFERENCES patients (id) ON DELETE CASCADE,

    -- Admission window
    admitted_at             TIMESTAMPTZ     NOT NULL,
    discharged_at           TIMESTAMPTZ,               -- NULL = still admitted

    -- Phase A — ECG measurements
    -- Schema: { heart_rate, pr_interval_ms, qrs_duration_ms, qtc_ms,
    --           st_deviation_mm, rhythm_interpretation }
    ecg_data                JSONB           NOT NULL DEFAULT '{}',

    -- Phase A — Blood lab values (raw clinical units, not normalised)
    -- Schema: { troponin_ng_ml, bnp_pg_ml, wbc_count, creatinine_mg_dl,
    --           sodium_meq_l, potassium_meq_l, lactate_mmol_l, collected_at }
    lab_data                JSONB           NOT NULL DEFAULT '{}',

    -- Phase A — Early risk engine output (from early_risk_inference.py)
    phase_a_risk_level      phase_a_risk_enum,          -- 'Low' | 'Moderate' | 'High'
    phase_a_risk_score      NUMERIC(5, 4)   CHECK (phase_a_risk_score BETWEEN 0 AND 1),
    phase_a_recommendation  TEXT,
    phase_a_run_at          TIMESTAMPTZ,

    -- CXR image reference
    cxr_dicom_path          TEXT,                       -- GCS / local path to DICOM
    cxr_acquired_at         TIMESTAMPTZ,

    -- MIMIC linkage
    mimic_study_id          BIGINT,

    -- Audit
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_cases_patient ON cases (patient_id);
CREATE INDEX idx_cases_admitted ON cases (admitted_at DESC);
CREATE INDEX idx_cases_phase_a_risk ON cases (phase_a_risk_level);

-- JSONB GIN indexes for fast lab/ECG queries
CREATE INDEX idx_cases_ecg_gin   ON cases USING GIN (ecg_data);
CREATE INDEX idx_cases_lab_gin   ON cases USING GIN (lab_data);

-- ---------------------------------------------------------------------------
-- TABLE: predictions
-- One row per CXR label per inference run (8 rows per case run).
-- Stores raw probability + thresholded risk_badge + uncertainty from mc_dropout.
--
-- risk_badge is DERIVED from probability:
--   probability < 0.05            → 'Unlikely'
--   0.05 ≤ probability ≤ 0.15    → 'Monitor'
--   probability > 0.15            → 'Elevated Risk'
--
-- Generated column enforces the mapping in SQL:
-- ---------------------------------------------------------------------------
CREATE TABLE predictions (
    id                      UUID                PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id                 UUID                NOT NULL REFERENCES cases (id) ON DELETE CASCADE,

    -- Inference configuration
    model_checkpoint        VARCHAR(255)        NOT NULL DEFAULT 'baseline_best.pt',
    temperature             NUMERIC(4, 2)       NOT NULL DEFAULT 1.3
                                                CHECK (temperature > 0),
    inference_run_at        TIMESTAMPTZ         NOT NULL DEFAULT NOW(),

    -- Per-label output
    label                   cxr_label_enum      NOT NULL,
    probability             NUMERIC(6, 4)       NOT NULL
                                                CHECK (probability BETWEEN 0 AND 1),

    -- Derived categorical badge — computed by application layer and stored
    -- for fast frontend queries without re-computation.
    -- Enforced by CHECK to keep SQL and app logic in sync.
    risk_badge              risk_badge_enum     NOT NULL
                                                GENERATED ALWAYS AS (
                                                    CASE
                                                        WHEN probability < 0.05  THEN 'Unlikely'::risk_badge_enum
                                                        WHEN probability <= 0.15 THEN 'Monitor'::risk_badge_enum
                                                        ELSE                          'Elevated Risk'::risk_badge_enum
                                                    END
                                                ) STORED,

    -- MC Dropout uncertainty (from mc_dropout.py)
    uncertainty_level       uncertainty_level_enum,
    mean_variance           NUMERIC(10, 6),     -- scalar mean across all passes
    std_dev                 NUMERIC(10, 6),
    mc_passes               SMALLINT            DEFAULT 20,

    -- Grad-CAM visual explanation
    gradcam_url             TEXT,               -- URL / local path to overlay PNG
    gradcam_alpha           NUMERIC(3, 2)       DEFAULT 0.45
                                                CHECK (gradcam_alpha BETWEEN 0 AND 1),

    -- Unique: one prediction per label per inference run per case
    CONSTRAINT uq_prediction_case_label_run UNIQUE (case_id, label, inference_run_at)
);

CREATE INDEX idx_predictions_case   ON predictions (case_id);
CREATE INDEX idx_predictions_badge  ON predictions (risk_badge);
CREATE INDEX idx_predictions_label  ON predictions (label);
CREATE INDEX idx_predictions_run    ON predictions (inference_run_at DESC);

-- ---------------------------------------------------------------------------
-- TABLE: consultations
-- One per case — the shared thread between Ward Doctor and Radiologist.
-- Messages and viewport state are stored as JSONB arrays for flexibility.
--
-- viewport_state JSONB — schema: { zoom, contrast, brightness,
--                                   window_center, window_width, annotations }
-- messages       JSONB — array of { id, role, type, content, sent_at, read }
-- ---------------------------------------------------------------------------
CREATE TABLE consultations (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id                 UUID            NOT NULL UNIQUE REFERENCES cases (id) ON DELETE CASCADE,

    -- Participants
    ward_doctor_id          UUID            NOT NULL,   -- FK → auth.users (Supabase Auth)
    radiologist_id          UUID,                       -- NULL until radiologist accepts

    -- Status
    is_open                 BOOLEAN         NOT NULL DEFAULT TRUE,
    opened_at               TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    closed_at               TIMESTAMPTZ,
    urgency_flag            BOOLEAN         NOT NULL DEFAULT FALSE,

    -- Message thread
    -- Array of: { id: uuid, role: 'ward_doctor'|'radiologist',
    --             type: 'text'|'annotation'|'viewport_sync'|'finding_flag',
    --             content: string | AnnotationPayload | ViewportPayload,
    --             sent_at: timestamptz, read: boolean }
    messages                JSONB           NOT NULL DEFAULT '[]',

    -- Shared DICOM viewport memory (last known state synced between both users)
    -- { zoom: float, contrast: int, brightness: int,
    --   window_center: float, window_width: float,
    --   annotations: [{ id, x, y, width, height, label, color, author_role }] }
    viewport_state          JSONB           NOT NULL DEFAULT '{
        "zoom": 1.0,
        "contrast": 100,
        "brightness": 100,
        "window_center": 40.0,
        "window_width": 400.0,
        "annotations": []
    }',

    -- Last viewport sync timestamps per role
    ward_doctor_last_view   TIMESTAMPTZ,
    radiologist_last_view   TIMESTAMPTZ,

    -- Audit
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_consultations_case     ON consultations (case_id);
CREATE INDEX idx_consultations_doctor   ON consultations (ward_doctor_id);
CREATE INDEX idx_consultations_rad      ON consultations (radiologist_id);
CREATE INDEX idx_consultations_open     ON consultations (is_open) WHERE is_open = TRUE;
CREATE INDEX idx_consultations_messages ON consultations USING GIN (messages);
CREATE INDEX idx_consultations_viewport ON consultations USING GIN (viewport_state);

-- ---------------------------------------------------------------------------
-- Row-Level Security (RLS) — enable for Supabase Auth integration
-- ---------------------------------------------------------------------------
ALTER TABLE patients      ENABLE ROW LEVEL SECURITY;
ALTER TABLE cases         ENABLE ROW LEVEL SECURITY;
ALTER TABLE predictions   ENABLE ROW LEVEL SECURITY;
ALTER TABLE consultations ENABLE ROW LEVEL SECURITY;

-- Example policy: authenticated users can read all records (tighten per role later)
CREATE POLICY "authenticated_read_patients"   ON patients      FOR SELECT USING (auth.role() = 'authenticated');
CREATE POLICY "authenticated_read_cases"      ON cases         FOR SELECT USING (auth.role() = 'authenticated');
CREATE POLICY "authenticated_read_predictions"ON predictions   FOR SELECT USING (auth.role() = 'authenticated');
CREATE POLICY "authenticated_read_consults"   ON consultations FOR SELECT USING (auth.role() = 'authenticated');

-- Ward doctor can only update consultations they own
CREATE POLICY "doctor_update_consultation" ON consultations
    FOR UPDATE USING (auth.uid() = ward_doctor_id);

-- Radiologist can update once assigned
CREATE POLICY "radiologist_update_consultation" ON consultations
    FOR UPDATE USING (auth.uid() = radiologist_id);

-- ---------------------------------------------------------------------------
-- Auto-update updated_at triggers
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_patients_updated_at
    BEFORE UPDATE ON patients
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TRIGGER trg_cases_updated_at
    BEFORE UPDATE ON cases
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TRIGGER trg_consultations_updated_at
    BEFORE UPDATE ON consultations
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();
