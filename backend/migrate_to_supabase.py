"""
migrate_to_supabase.py
One-time migration of local_db.json -> Supabase Postgres.

Prereqs:
  1. backend/.env has SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY
  2. schema.sql has been applied via Supabase Dashboard -> SQL Editor

Usage:
    py -V:3.14 migrate_to_supabase.py            # do the migration
    py -V:3.14 migrate_to_supabase.py --dry-run  # validate only
    py -V:3.14 migrate_to_supabase.py --wipe     # delete all rows first (DANGER)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from supabase import Client, create_client

_BACKEND = Path(__file__).resolve().parent
load_dotenv(_BACKEND / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
DB_PATH = _BACKEND / "local_db.json"

# Predictions: risk_badge is GENERATED ALWAYS in Postgres -> exclude from insert
PREDICTION_DROP_FIELDS = {"risk_badge"}


def make_client() -> Client:
    if not SUPABASE_URL or not SERVICE_KEY:
        sys.exit("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in backend/.env")
    return create_client(SUPABASE_URL, SERVICE_KEY)


def chunked(items, size=100):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def wipe(client: Client):
    # Delete in FK-safe order
    print("Wiping existing rows...")
    for table in ("predictions", "consultations", "cases", "patients"):
        client.table(table).delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
        print(f"  cleared {table}")


def migrate(dry_run: bool, do_wipe: bool):
    db = json.load(open(DB_PATH, encoding="utf-8"))

    patients = db.get("patients", [])
    cases = db.get("cases", [])
    predictions = db.get("predictions", [])
    consultations = db.get("consultations", [])

    # Dedupe patients by mimic_subject_id (schema enforces UNIQUE).
    # When duplicates exist, keep the first row and remap dropped patient_ids
    # on cases so FK references stay valid.
    seen_mimic: dict[int, str] = {}
    id_remap: dict[str, str] = {}
    deduped_patients = []
    for p in patients:
        mid = p.get("mimic_subject_id")
        if mid is not None and mid in seen_mimic:
            id_remap[p["id"]] = seen_mimic[mid]
            continue
        if mid is not None:
            seen_mimic[mid] = p["id"]
        deduped_patients.append(p)
    if id_remap:
        for c in cases:
            if c["patient_id"] in id_remap:
                c["patient_id"] = id_remap[c["patient_id"]]
        print(f"  deduped patients: {len(patients)} -> {len(deduped_patients)} "
              f"(remapped {len(id_remap)} duplicate FKs)")
    patients = deduped_patients

    # Strip generated columns from prediction rows
    predictions_clean = [
        {k: v for k, v in p.items() if k not in PREDICTION_DROP_FIELDS}
        for p in predictions
    ]

    print(f"Loaded from JSON:")
    print(f"  patients      : {len(patients)}")
    print(f"  cases         : {len(cases)}")
    print(f"  predictions   : {len(predictions_clean)}")
    print(f"  consultations : {len(consultations)}")

    if dry_run:
        print("\n[DRY RUN] No data inserted.")
        return

    client = make_client()

    if do_wipe:
        wipe(client)

    # Insert in FK order: patients -> cases -> predictions -> consultations
    for table, rows in [
        ("patients", patients),
        ("cases", cases),
        ("predictions", predictions_clean),
        ("consultations", consultations),
    ]:
        if not rows:
            print(f"  {table}: 0 rows, skipping")
            continue
        total = 0
        for chunk in chunked(rows, 100):
            client.table(table).upsert(chunk, on_conflict="id").execute()
            total += len(chunk)
        print(f"  {table}: upserted {total}")

    # Verify counts
    print("\nVerifying row counts in Supabase:")
    for table in ("patients", "cases", "predictions", "consultations"):
        resp = client.table(table).select("id", count="exact").execute()
        print(f"  {table}: {resp.count}")

    print("\nMigration complete.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--wipe", action="store_true", help="delete all rows before inserting")
    args = p.parse_args()
    migrate(dry_run=args.dry_run, do_wipe=args.wipe)


if __name__ == "__main__":
    main()
