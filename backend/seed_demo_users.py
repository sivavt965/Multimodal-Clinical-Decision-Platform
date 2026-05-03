"""
seed_demo_users.py
Inserts one demo user per role (radiologist, ward_doctor, clinical_admin,
system_admin) into the Supabase users table. Idempotent — uses upsert by email.

Run AFTER applying 002_users_audit_outcome.sql.

Usage:
    py -V:3.14 seed_demo_users.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

_BACKEND = Path(__file__).resolve().parent
load_dotenv(_BACKEND / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

DEMO_USERS = [
    {"email": "dr.smith@hospital.org",   "full_name": "Dr. Alice Smith",   "role": "radiologist"},
    {"email": "dr.johnson@hospital.org", "full_name": "Dr. Ben Johnson",   "role": "ward_doctor"},
    {"email": "sarah.lee@hospital.org",  "full_name": "Sarah Lee",         "role": "clinical_admin"},
    {"email": "ops@hospital.org",        "full_name": "System Operator",   "role": "system_admin"},
]


def main():
    if not SUPABASE_URL or not SERVICE_KEY:
        sys.exit("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in backend/.env")

    client = create_client(SUPABASE_URL, SERVICE_KEY)

    for u in DEMO_USERS:
        u["status"] = "active"
        client.table("users").upsert(u, on_conflict="email").execute()
        print(f"  upserted {u['role']:<15} -> {u['email']}")

    resp = client.table("users").select("id,email,full_name,role").order("role").execute()
    print(f"\nTotal users in Supabase: {len(resp.data)}")
    for row in resp.data:
        print(f"  {row['role']:<15} {row['full_name']:<22} {row['email']}  ({row['id']})")


if __name__ == "__main__":
    main()
