-- =============================================================================
-- 002_users_audit_outcome.sql — Phase 1 schema additions
-- Adds:
--   * user_role_enum + users table        (RBAC foundation)
--   * audit_log table                     (HIPAA-style action trail)
--   * case_outcome_enum + cases.outcome   (case completion outcome)
-- Idempotent: safe to re-run.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- ENUMs
-- ---------------------------------------------------------------------------
DO $$ BEGIN
    CREATE TYPE user_role_enum AS ENUM (
        'radiologist',
        'ward_doctor',
        'clinical_admin',
        'system_admin'
    );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE user_status_enum AS ENUM ('active', 'inactive', 'suspended');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE case_outcome_enum AS ENUM (
        'General Ward',
        'ICU Transfer',
        'Step-Down Unit',
        'Discharged',
        'Deceased',
        'Other'
    );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- ---------------------------------------------------------------------------
-- TABLE: users
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email           VARCHAR(255) NOT NULL UNIQUE,
    full_name       VARCHAR(150) NOT NULL,
    role            user_role_enum NOT NULL,
    status          user_status_enum NOT NULL DEFAULT 'active',
    last_active_at  TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_users_role   ON users(role);
CREATE INDEX IF NOT EXISTS idx_users_email  ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_status ON users(status);

-- updated_at trigger reuses the existing set_updated_at() function from schema.sql
DO $$ BEGIN
    CREATE TRIGGER trg_users_updated_at
        BEFORE UPDATE ON users
        FOR EACH ROW EXECUTE FUNCTION set_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- ---------------------------------------------------------------------------
-- TABLE: audit_log
-- ---------------------------------------------------------------------------
-- One row per user-initiated action. target_type names the table the action
-- pertains to (cases, predictions, users, system); target_id is its UUID.
-- metadata captures action-specific fields (latency, outcome, finding, etc).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS audit_log (
    id           BIGSERIAL PRIMARY KEY,
    user_id      UUID REFERENCES users(id) ON DELETE SET NULL,
    user_role    user_role_enum,
    action       VARCHAR(64) NOT NULL,
    target_type  VARCHAR(32),
    target_id    UUID,
    metadata     JSONB NOT NULL DEFAULT '{}',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_audit_user        ON audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_target      ON audit_log(target_type, target_id);
CREATE INDEX IF NOT EXISTS idx_audit_created_at  ON audit_log(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_action      ON audit_log(action);

-- ---------------------------------------------------------------------------
-- COLUMNS: cases.outcome + cases.outcome_note + cases.completed_at
-- ---------------------------------------------------------------------------
ALTER TABLE cases ADD COLUMN IF NOT EXISTS outcome       case_outcome_enum;
ALTER TABLE cases ADD COLUMN IF NOT EXISTS outcome_note  TEXT;
ALTER TABLE cases ADD COLUMN IF NOT EXISTS completed_at  TIMESTAMPTZ;
ALTER TABLE cases ADD COLUMN IF NOT EXISTS completed_by  UUID REFERENCES users(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_cases_outcome      ON cases(outcome);
CREATE INDEX IF NOT EXISTS idx_cases_completed_at ON cases(completed_at DESC) WHERE completed_at IS NOT NULL;

-- ---------------------------------------------------------------------------
-- RLS — enable but defer policies until real auth is wired (Phase 5)
-- ---------------------------------------------------------------------------
ALTER TABLE users     ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_log ENABLE ROW LEVEL SECURITY;

-- Service-role key bypasses RLS, so backend continues to work.
-- Until real auth lands, restrict anon reads explicitly.
DO $$ BEGIN
    CREATE POLICY "service_only_users"     ON users     FOR ALL  USING (false);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE POLICY "service_only_audit_log" ON audit_log FOR ALL  USING (false);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;
