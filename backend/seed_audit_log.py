"""
seed_audit_log.py
Inserts ~6 plausible audit_log rows so the System Admin → Audit Log table
has visible content on first render. Real entries will accrue once
audit-logging middleware is wired (Phase 6).

Idempotent: deletes any rows where metadata->>'seed' = 'demo' before inserting.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from supabase import create_client

_BACKEND = Path(__file__).resolve().parent
load_dotenv(_BACKEND / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")


def main():
    if not SUPABASE_URL or not SERVICE_KEY:
        sys.exit("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in backend/.env")

    client = create_client(SUPABASE_URL, SERVICE_KEY)

    users = {u["role"]: u for u in client.table("users").select("id,role").execute().data}
    cases = client.table("cases").select("id").limit(2).execute().data
    if not users or not cases:
        sys.exit("Seed users + cases first (seed_demo_users.py + migrate_to_supabase.py)")

    case_id = cases[0]["id"]
    now = datetime.now(timezone.utc)

    # Wipe any prior demo seed rows to keep this idempotent
    client.table("audit_log").delete().eq("metadata->>seed", "demo").execute()

    rows = [
        {
            "user_id": users["clinical_admin"]["id"],
            "user_role": "clinical_admin",
            "action": "case.create",
            "target_type": "cases",
            "target_id": case_id,
            "metadata": {"seed": "demo", "via": "registration_wizard"},
            "created_at": (now - timedelta(minutes=35)).isoformat(),
        },
        {
            "user_id": users["radiologist"]["id"],
            "user_role": "radiologist",
            "action": "cxr.analyze",
            "target_type": "cases",
            "target_id": case_id,
            "metadata": {"seed": "demo", "latency_ms": 2143, "mc_passes": 10},
            "created_at": (now - timedelta(minutes=22)).isoformat(),
        },
        {
            "user_id": users["radiologist"]["id"],
            "user_role": "radiologist",
            "action": "case.flag_critical",
            "target_type": "cases",
            "target_id": case_id,
            "metadata": {"seed": "demo", "finding": "Pneumonia"},
            "created_at": (now - timedelta(minutes=21)).isoformat(),
        },
        {
            "user_id": users["ward_doctor"]["id"],
            "user_role": "ward_doctor",
            "action": "consultation.message",
            "target_type": "cases",
            "target_id": case_id,
            "metadata": {"seed": "demo", "message_len": 142},
            "created_at": (now - timedelta(minutes=12)).isoformat(),
        },
        {
            "user_id": users["ward_doctor"]["id"],
            "user_role": "ward_doctor",
            "action": "case.complete",
            "target_type": "cases",
            "target_id": case_id,
            "metadata": {"seed": "demo", "outcome": "ICU Transfer"},
            "created_at": (now - timedelta(minutes=4)).isoformat(),
        },
        {
            "user_id": users["system_admin"]["id"],
            "user_role": "system_admin",
            "action": "faiss.reload",
            "target_type": "system",
            "target_id": None,
            "metadata": {"seed": "demo", "vectors_loaded": 38, "duration_ms": 312},
            "created_at": (now - timedelta(minutes=1)).isoformat(),
        },
    ]

    client.table("audit_log").insert(rows).execute()
    print(f"Inserted {len(rows)} demo audit entries.")

    resp = client.table("audit_log").select("id", count="exact").execute()
    print(f"Total audit_log rows: {resp.count}")


if __name__ == "__main__":
    main()
