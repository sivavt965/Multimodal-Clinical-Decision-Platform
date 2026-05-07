-- =============================================================================
-- 004_atomic_consultation_messages.sql — fix lost-update race on chat
-- =============================================================================
-- Problem
-- -------
-- The Python helper database.append_consultation_message() did a
-- read-modify-write on consultations.messages (JSONB array): SELECT the
-- existing array, .append() in Python, UPDATE with the new array. Under
-- concurrent appends (e.g. ward doctor and radiologist both posting at
-- the same moment), one message would be lost:
--
--   A: SELECT [m1,m2]
--   B: SELECT [m1,m2]
--   A: UPDATE → [m1,m2,A]
--   B: UPDATE → [m1,m2,B]   -- A gone
--
-- Fix
-- ---
-- Push the read-modify-write into a single Postgres function. The UPDATE
-- statement takes a row-level lock so concurrent appends serialise; the
-- jsonb || operator concatenates atomically server-side. No retries, no
-- explicit transactions needed — Postgres handles it via MVCC.
--
-- Backend code (database.py append_consultation_message) is updated to
-- call db.rpc('append_consultation_message', ...) instead of doing the
-- two-step read+write. Frontend is unchanged.
--
-- Idempotent: CREATE OR REPLACE FUNCTION. Safe to re-run.
-- =============================================================================

CREATE OR REPLACE FUNCTION public.append_consultation_message(
    p_case_id UUID,
    p_message JSONB
)
    RETURNS consultations
    LANGUAGE plpgsql
AS $$
DECLARE
    result_row consultations;
BEGIN
    -- Try the common path first: append to an existing consultation.
    -- The UPDATE acquires a row-level exclusive lock; concurrent callers
    -- with the same case_id queue here and apply their || atomically in
    -- order. No message can be lost.
    UPDATE consultations
    SET    messages   = messages || p_message,
           updated_at = NOW()
    WHERE  case_id    = p_case_id
    RETURNING * INTO result_row;

    IF FOUND THEN
        RETURN result_row;
    END IF;

    -- No consultation row yet — create one with this message as the seed.
    -- The UNIQUE (case_id) constraint on consultations.case_id makes this
    -- safe under a concurrent first-message race: only one INSERT wins,
    -- the other raises unique_violation and the caller can retry. In
    -- practice the frontend always opens the consultation before sending,
    -- so this branch is the cold-start path.
    INSERT INTO consultations (
        case_id,
        ward_doctor_id,
        is_open,
        opened_at,
        messages
    )
    VALUES (
        p_case_id,
        '00000000-0000-0000-0000-000000000000'::uuid,  -- placeholder until real assignment
        TRUE,
        NOW(),
        jsonb_build_array(p_message)
    )
    RETURNING * INTO result_row;

    RETURN result_row;
END;
$$;

-- The backend runs with the service-role key which bypasses RLS, but we
-- grant EXECUTE explicitly so the intent is documented.
GRANT EXECUTE ON FUNCTION public.append_consultation_message(UUID, JSONB)
    TO service_role;

-- ---------------------------------------------------------------------------
-- Smoke test (run manually after applying):
--
--   SELECT public.append_consultation_message(
--       '<some-existing-case-uuid>'::uuid,
--       '{"id":"test","role":"radiologist","content":"hi","sent_at":"2026-05-07T10:00:00Z","read":false}'::jsonb
--   );
--
--   -- Then verify:
--   SELECT jsonb_array_length(messages) AS n_messages
--   FROM   consultations
--   WHERE  case_id = '<same-uuid>';
-- ---------------------------------------------------------------------------
