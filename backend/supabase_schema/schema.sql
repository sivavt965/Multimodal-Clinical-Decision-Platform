-- =============================================================================
-- schema.sql — Supabase Postgres
-- Multimodal Clinical Decision Support Platform
-- =============================================================================
-- Tables:
--   patients       — patient metadata
--   cases          — admission details, linked to patients, containing temporal ECG/Lab tabular JSON
--   predictions    — Phase B CXR outputs, linked to cases, containing raw probability AND the thresholded `risk_badge`, plus uncertainty levels
--   consultations  — The dual‑user thread connecting the Ward Doctor and Radiologist, including viewport memory like zoom/contrast states.
--
-- Risk badge thresholds (enforced via CHECK + application logic):
--   < 0.05          → "Unlikely" (Gray)
--   0.05‑0.15       → "Monitor" (Yellow)
--   > 0.15          → "Elevated Risk" (Red)
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
    'Cardiomegaly','Pleural Effusion','Edema','Pneumonia',
    'Atelectasis','Pneumothorax','Consolidation','Support Devices',
    'No Finding','Enlarged Cardiomediastinum','Lung Opacity',
    'Lung Lesion','Pleural Other','Fracture'
);

CREATE TYPE consultation_role_enum AS ENUM ('ward_doctor','radiologist');
CREATE TYPE message_type_enum AS ENUM ('text','annotation','viewport_sync','finding_flag');

-- ---------------------------------------------------------------------------
-- TABLE: patients
-- ---------------------------------------------------------------------------
CREATE TABLE patients (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    mrn             VARCHAR(20) NOT NULL UNIQUE,
    first_name      VARCHAR(100) NOT NULL,
    last_name       VARCHAR(100) NOT NULL,
    date_of_birth   DATE NOT NULL,
    sex             sex_enum NOT NULL,
    age_at_admission SMALLINT NOT NULL CHECK (age_at_admission BETWEEN 0 AND 130),
    mimic_subject_id BIGINT UNIQUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_patients_mrn ON patients(mrn);
CREATE INDEX idx_patients_mimic ON patients(mimic_subject_id);

-- ---------------------------------------------------------------------------
-- TABLE: cases
-- ---------------------------------------------------------------------------
CREATE TABLE cases (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id       UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    admitted_at      TIMESTAMPTZ NOT NULL,
    discharged_at    TIMESTAMPTZ,
    ecg_data         JSONB NOT NULL DEFAULT '{}',
    lab_data         JSONB NOT NULL DEFAULT '{}',
    phase_a_risk_level phase_a_risk_enum,
    phase_a_risk_score NUMERIC(5,4) CHECK (phase_a_risk_score BETWEEN 0 AND 1),
    phase_a_recommendation TEXT,
    phase_a_run_at   TIMESTAMPTZ,
    cxr_dicom_url    TEXT,
    cxr_heatmap_url  TEXT,
    cxr_heatmap_label cxr_label_enum,
    cxr_acquired_at  TIMESTAMPTZ,
    mimic_study_id   BIGINT,
    labs_raw         JSONB NOT NULL DEFAULT '{}',
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_cases_patient ON cases(patient_id);
CREATE INDEX idx_cases_admitted ON cases(admitted_at DESC);
CREATE INDEX idx_cases_phase_a_risk ON cases(phase_a_risk_level);
CREATE INDEX idx_cases_ecg_gin ON cases USING GIN (ecg_data);
CREATE INDEX idx_cases_lab_gin ON cases USING GIN (lab_data);

-- ---------------------------------------------------------------------------
-- TABLE: predictions
-- ---------------------------------------------------------------------------
CREATE TABLE predictions (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id           UUID NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    model_checkpoint  VARCHAR(255) NOT NULL DEFAULT 'baseline_best.pt',
    temperature       NUMERIC(4,2) NOT NULL DEFAULT 1.30 CHECK (temperature > 0),
    inference_run_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    label             cxr_label_enum NOT NULL,
    probability       NUMERIC(6,4) NOT NULL CHECK (probability BETWEEN 0 AND 1),
    risk_badge        risk_badge_enum NOT NULL GENERATED ALWAYS AS (
                        CASE
                          WHEN probability < 0.05 THEN 'Unlikely'::risk_badge_enum
                          WHEN probability <= 0.15 THEN 'Monitor'::risk_badge_enum
                          ELSE 'Elevated Risk'::risk_badge_enum
                        END) STORED,
    uncertainty_level uncertainty_level_enum,
    mean_variance     NUMERIC(10,6),
    std_dev           NUMERIC(10,6),
    mc_passes         SMALLINT DEFAULT 20,
    gradcam_url       TEXT,
    gradcam_alpha     NUMERIC(3,2) DEFAULT 0.45 CHECK (gradcam_alpha BETWEEN 0 AND 1),
    CONSTRAINT uq_prediction_case_label_run UNIQUE (case_id, label, inference_run_at)
);
CREATE INDEX idx_predictions_case   ON predictions(case_id);
CREATE INDEX idx_predictions_badge  ON predictions(risk_badge);
CREATE INDEX idx_predictions_label  ON predictions(label);
CREATE INDEX idx_predictions_run    ON predictions(inference_run_at DESC);

-- ---------------------------------------------------------------------------
-- TABLE: consultations
-- ---------------------------------------------------------------------------
CREATE TABLE consultations (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id            UUID NOT NULL UNIQUE REFERENCES cases(id) ON DELETE CASCADE,
    ward_doctor_id     UUID NOT NULL,
    radiologist_id     UUID,
    is_open            BOOLEAN NOT NULL DEFAULT TRUE,
    opened_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    closed_at          TIMESTAMPTZ,
    urgency_flag       BOOLEAN NOT NULL DEFAULT FALSE,
    messages           JSONB NOT NULL DEFAULT '[]',
    viewport_state     JSONB NOT NULL DEFAULT '{"zoom":1.0,"contrast":100,"brightness":100,"window_center":40.0,"window_width":400.0,"annotations":[]}',
    ward_doctor_last_view TIMESTAMPTZ,
    radiologist_last_view TIMESTAMPTZ,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_consultations_case   ON consultations(case_id);
CREATE INDEX idx_consultations_doctor ON consultations(ward_doctor_id);
CREATE INDEX idx_consultations_rad    ON consultations(radiologist_id);
CREATE INDEX idx_consultations_open   ON consultations(is_open) WHERE is_open = TRUE;
CREATE INDEX idx_consultations_messages ON consultations USING GIN (messages);
CREATE INDEX idx_consultations_viewport ON consultations USING GIN (viewport_state);

-- ---------------------------------------------------------------------------
-- Row‑level security (Supabase Auth integration)
-- ---------------------------------------------------------------------------
ALTER TABLE patients      ENABLE ROW LEVEL SECURITY;
ALTER TABLE cases         ENABLE ROW LEVEL SECURITY;
ALTER TABLE predictions   ENABLE ROW LEVEL SECURITY;
ALTER TABLE consultations ENABLE ROW LEVEL SECURITY;

CREATE POLICY "authenticated_read_patients"      ON patients      FOR SELECT USING (auth.role() = 'authenticated');
CREATE POLICY "authenticated_read_cases"         ON cases         FOR SELECT USING (auth.role() = 'authenticated');
CREATE POLICY "authenticated_read_predictions"   ON predictions   FOR SELECT USING (auth.role() = 'authenticated');
CREATE POLICY "authenticated_read_consultations" ON consultations FOR SELECT USING (auth.role() = 'authenticated');

CREATE POLICY "doctor_update_consultation" ON consultations
    FOR UPDATE USING (auth.uid() = ward_doctor_id);
CREATE POLICY "radiologist_update_consultation" ON consultations
    FOR UPDATE USING (auth.uid() = radiologist_id);

-- ---------------------------------------------------------------------------
-- Auto‑update updated_at triggers
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_patients_updated_at BEFORE UPDATE ON patients FOR EACH ROW EXECUTE FUNCTION set_updated_at();
CREATE TRIGGER trg_cases_updated_at    BEFORE UPDATE ON cases    FOR EACH ROW EXECUTE FUNCTION set_updated_at();
CREATE TRIGGER trg_consultations_updated_at BEFORE UPDATE ON consultations FOR EACH ROW EXECUTE FUNCTION set_updated_at();
