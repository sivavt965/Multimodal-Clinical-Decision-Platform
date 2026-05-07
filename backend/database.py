# =============================================================================
# database.py — Supabase persistence layer
# All MOCK_DB access in main.py is replaced by calls to these functions.
# =============================================================================
"""
Tables used
-----------
  patients      — one row per patient
  cases         — one row per clinical case (FK → patients.id)
  predictions   — one row per CheXpert label per case (FK → cases.id)
  consultations — one row per consultation thread (FK → cases.id)

Storage buckets (create via Supabase Dashboard → Storage)
-----------
  cxr-images   — original uploaded CXR files
  heatmaps     — Grad-CAM overlay PNGs
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from supabase import Client, create_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Client initialisation (singleton with local fallback)
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).resolve().parent / ".env")

_SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
# Prefer service_role on the server: it bypasses RLS and lets the FastAPI
# backend read/write any row. The anon key would be blocked by the
# `auth.role() = 'authenticated'` SELECT policy.
_SUPABASE_KEY: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY", "")

CXR_BUCKET: str     = os.getenv("SUPABASE_CXR_BUCKET", "cxr-images")
HEATMAP_BUCKET: str = os.getenv("SUPABASE_HEATMAP_BUCKET", "heatmaps")

_client: Optional[Client] = None
_LOCAL_DB_PATH = Path(__file__).resolve().parent / "local_db.json"

def _load_local_db() -> Dict[str, Any]:
    """Load local JSON database with all relational tables."""
    _EMPTY_DB = {
        "patients": [],
        "cases": [],
        "predictions": [],
        "consultations": [],
        "lab_results": [],
        "ecg_records": [],
    }
    if not _LOCAL_DB_PATH.exists():
        return _EMPTY_DB.copy()
    import json
    try:
        with open(_LOCAL_DB_PATH, "r") as f:
            data = json.load(f)
        # Ensure new tables exist in legacy DBs
        for key in _EMPTY_DB:
            if key not in data:
                data[key] = []
        return data
    except Exception:
        return _EMPTY_DB.copy()

def _save_local_db(data: Dict[str, Any]):
    import json
    with open(_LOCAL_DB_PATH, "w") as f:
        json.dump(data, f, indent=2)

def get_db() -> Client:
    """Return the Supabase client, initialising it on the first call."""
    global _client
    if _client is None:
        try:
            if not _SUPABASE_URL or not _SUPABASE_KEY:
                raise ValueError("Missing Supabase credentials")
            _client = create_client(_SUPABASE_URL, _SUPABASE_KEY)
            # Connectivity check
            _client.table("cases").select("id").limit(1).execute()
            logger.info("[DB] Supabase client initialised → %s", _SUPABASE_URL)
        except Exception as exc:
            logger.warning("[DB] Supabase connection failed, falling back to local JSON: %s", exc)
            _client = "LOCAL_MOCK" # Sentinel for mock mode
    return _client


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def upload_file_to_bucket(
    bucket: str,
    dest_path: str,
    file_bytes: bytes,
    content_type: str = "application/octet-stream",
) -> str:
    """
    Upload *file_bytes* to *bucket* at *dest_path*.

    Returns the public URL for the uploaded object.
    """
    db = get_db()
    if db == "LOCAL_MOCK":
        # In mock mode, save the file to the public directory so Next.js can serve it.
        # Route by bucket: CXR images → dicoms/, heatmaps → heatmaps/
        _repo_root = Path(__file__).resolve().parent.parent
        file_name = Path(dest_path).name  # e.g. "case_8371a1da.jpg"

        if bucket == HEATMAP_BUCKET:
            local_dir = _repo_root / "frontend" / "public" / "mock-data" / "heatmaps"
            url_prefix = "/mock-data/heatmaps"
        else:
            local_dir = _repo_root / "frontend" / "public" / "mock-data" / "dicoms"
            url_prefix = "/mock-data/dicoms"

        local_dir.mkdir(parents=True, exist_ok=True)
        local_file = local_dir / file_name
        local_file.write_bytes(file_bytes)
        mock_url = f"{url_prefix}/{file_name}"
        logger.info("[Storage-MOCK] Saved %d bytes → %s (URL: %s)", len(file_bytes), local_file, mock_url)
        return mock_url
        
    db.storage.from_(bucket).upload(
        path=dest_path,
        file=file_bytes,
        file_options={"content-type": content_type, "upsert": "true"},
    )
    public_url: str = db.storage.from_(bucket).get_public_url(dest_path)
    logger.info("[Storage] Uploaded to %s/%s → %s", bucket, dest_path, public_url)
    return public_url


def upload_local_file_to_bucket(
    bucket: str,
    dest_path: str,
    local_path: str | Path,
    content_type: str = "application/octet-stream",
) -> str:
    """Convenience wrapper — reads a local file then calls upload_file_to_bucket."""
    with open(local_path, "rb") as fh:
        data = fh.read()
    return upload_file_to_bucket(bucket, dest_path, data, content_type)


# ---------------------------------------------------------------------------
# READ helpers
# ---------------------------------------------------------------------------

def get_all_cases() -> List[Dict[str, Any]]:
    """
    Return all cases joined with patient, predictions, and consultation data,
    ordered newest-first.
    """
    db = get_db()
    if db == "LOCAL_MOCK":
        data = _load_local_db()
        cases = data.get("cases", [])
        patients = {p["id"]: p for p in data.get("patients", [])}
        predictions = data.get("predictions", [])
        consultations = {c["case_id"]: c for c in data.get("consultations", [])}
        
        rows = []
        for c in cases:
            row = c.copy()
            row["patients"] = patients.get(c["patient_id"], {})
            row["predictions"] = [p for p in predictions if p["case_id"] == c["id"]]
            row["consultations"] = [consultations.get(c["id"])] if c["id"] in consultations else []
            rows.append(_normalise_joined_row(row))
            
        rows.sort(key=lambda x: x["case"].get("admitted_at", ""), reverse=True)
        return rows

    cases_resp = (
        db.table("cases")
        .select(
            "*, "
            "patients(*), "
            "predictions(*), "
            "consultations(*)"
        )
        .order("admitted_at", desc=True)
        .execute()
    )

    rows: List[Dict[str, Any]] = []
    for row in cases_resp.data:
        rows.append(_normalise_joined_row(row))

    return rows


def get_case_by_id(case_id: str) -> Optional[Dict[str, Any]]:
    """
    Return the full CaseDetail payload for *case_id*, or None if not found.
    """
    db = get_db()
    if db == "LOCAL_MOCK":
        rows = get_all_cases()
        for r in rows:
            if r["case"]["id"] == case_id:
                return r
        return None

    # maybe_single() returns None instead of raising on 0 rows, so callers
    # can distinguish "not found" (404) from "DB unavailable" (503).
    resp = (
        db.table("cases")
        .select(
            "*, "
            "patients(*), "
            "predictions(*), "
            "consultations(*)"
        )
        .eq("id", case_id)
        .maybe_single()
        .execute()
    )

    if not resp or not resp.data:
        return None

    return _normalise_joined_row(resp.data)


# ---------------------------------------------------------------------------
# WRITE helpers
# ---------------------------------------------------------------------------

def create_new_case(
    patient_payload: Dict[str, Any],
    case_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Persist a new patient row and case row.
    """
    db = get_db()
    if db == "LOCAL_MOCK":
        data = _load_local_db()

        # Upsert patient — MRN is the natural key. When a row already exists,
        # PRESERVE id and created_at so prior cases referencing this patient
        # stay valid; only refresh mutable demographic fields.
        existing_patient = next((p for p in data["patients"] if p["mrn"] == patient_payload["mrn"]), None)
        if existing_patient:
            preserved_id = existing_patient["id"]
            preserved_created = existing_patient.get("created_at")
            existing_patient.update(patient_payload)
            existing_patient["id"] = preserved_id
            if preserved_created:
                existing_patient["created_at"] = preserved_created
            patient_row = existing_patient
        else:
            patient_row = patient_payload.copy()
            if "id" not in patient_row: patient_row["id"] = str(uuid.uuid4())
            data["patients"].append(patient_row)

        # Insert case
        case_row = case_payload.copy()
        if "id" not in case_row: case_row["id"] = str(uuid.uuid4())
        case_row["patient_id"] = patient_row["id"]
        data["cases"].append(case_row)

        _save_local_db(data)
        logger.info("[DB-MOCK] Created case %s for patient %s", case_row["id"], patient_row["id"])

        return {
            "patient": patient_row,
            "case": case_row,
            "predictions": [],
            "consultation": None,
        }

    # 1. Look up patient by MRN; preserve id + created_at on collision so the
    #    upsert doesn't reassign the primary key (which would orphan prior
    #    cases via the FK on cases.patient_id).
    existing_resp = (
        db.table("patients")
        .select("id,created_at")
        .eq("mrn", patient_payload["mrn"])
        .limit(1)
        .execute()
    )
    if existing_resp.data:
        existing = existing_resp.data[0]
        update_payload = {k: v for k, v in patient_payload.items() if k not in ("id", "created_at")}
        upd = (
            db.table("patients")
            .update(update_payload)
            .eq("id", existing["id"])
            .execute()
        )
        patient_row = upd.data[0] if upd.data else {**existing, **update_payload}
    else:
        patient_resp = db.table("patients").insert(patient_payload).execute()
        patient_row = patient_resp.data[0]

    # 2. Insert case
    case_payload["patient_id"] = patient_row["id"]
    case_resp = db.table("cases").insert(case_payload).execute()
    case_row = case_resp.data[0]

    logger.info("[DB] Created case %s for patient %s", case_row["id"], patient_row["id"])

    return {
        "patient": patient_row,
        "case": case_row,
        "predictions": [],
        "consultation": None,
    }


def update_case_inference(
    case_id: str,
    predictions: List[Dict[str, Any]],
    heatmap_url: Optional[str],
    heatmap_label: Optional[str],
) -> None:
    """
    Called from the BackgroundTask after Grad-CAM inference completes.
    """
    db = get_db()
    now = _utcnow()

    # Each call to this function (re-)generates a heatmap for ONE label. Other
    # findings should retain whatever heatmap was generated for them previously
    # so the Grad-CAM toggle stays meaningful as the user clicks between
    # findings (otherwise non-matching labels get gradcam_url=None and the
    # toggle silently does nothing).
    def _resolve_gradcam_url(label: str, prior_by_label: Dict[str, Optional[str]]) -> Optional[str]:
        if label == heatmap_label:
            return heatmap_url
        return prior_by_label.get(label)

    if db == "LOCAL_MOCK":
        data = _load_local_db()

        # Snapshot existing per-label gradcam_urls before we delete the rows.
        prior_gradcam: Dict[str, Optional[str]] = {
            p["label"]: p.get("gradcam_url")
            for p in data["predictions"]
            if p["case_id"] == case_id
        }

        # Predictions
        if predictions:
            # Remove existing for this case
            data["predictions"] = [p for p in data["predictions"] if p["case_id"] != case_id]
            for p in predictions:
                data["predictions"].append({
                    "id": str(uuid.uuid4()),
                    "case_id": case_id,
                    "model_checkpoint": "densenet121-chexpert",
                    "temperature": 1.0,
                    "inference_run_at": now,
                    "label": p["label"],
                    "probability": p["probability"],
                    "risk_badge": p["risk_badge"],
                    "uncertainty_level": p.get("uncertainty_level"),
                    "mean_variance": p.get("mean_variance"),
                    "std_dev": p.get("std_dev"),
                    "mc_passes": p.get("mc_passes", 0),
                    "gradcam_url": _resolve_gradcam_url(p["label"], prior_gradcam),
                    "gradcam_alpha": 0.45,
                })

        # Case update
        for c in data["cases"]:
            if c["id"] == case_id:
                c["cxr_heatmap_url"] = heatmap_url
                c["cxr_heatmap_label"] = heatmap_label
                c["updated_at"] = now

        _save_local_db(data)
        return

    # ── Predictions ──────────────────────────────────────────────────────────
    if predictions:
        # Snapshot prior per-label gradcam_urls from Supabase before upserting.
        try:
            existing = (
                db.table("predictions")
                .select("label,gradcam_url")
                .eq("case_id", case_id)
                .execute()
                .data
                or []
            )
            prior_gradcam = {row["label"]: row.get("gradcam_url") for row in existing}
        except Exception:
            prior_gradcam = {}

        pred_rows = [
            {
                "id": str(uuid.uuid4()),
                "case_id": case_id,
                "model_checkpoint": "densenet121-chexpert",
                "temperature": 1.0,
                "inference_run_at": now,
                "label": p["label"],
                "probability": p["probability"],
                "risk_badge": p["risk_badge"],
                "uncertainty_level": p.get("uncertainty_level"),
                "mean_variance": p.get("mean_variance"),
                "std_dev": p.get("std_dev"),
                "mc_passes": p.get("mc_passes", 0),
                "gradcam_url": _resolve_gradcam_url(p["label"], prior_gradcam),
                "gradcam_alpha": 0.45,
            }
            for p in predictions
        ]
        db.table("predictions").upsert(pred_rows, on_conflict="case_id,label").execute()

    # ── Case metadata ─────────────────────────────────────────────────────────
    db.table("cases").update(
        {
            "cxr_heatmap_url": heatmap_url,
            "cxr_heatmap_label": heatmap_label,
            "updated_at": now,
        }
    ).eq("id", case_id).execute()


def append_consultation_message(
    case_id: str,
    message: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Append a consultation message.
    """
    db = get_db()
    now = _utcnow()

    if db == "LOCAL_MOCK":
        data = _load_local_db()
        cons = next((c for c in data["consultations"] if c["case_id"] == case_id), None)
        if cons:
            cons["messages"].append(message)
            cons["updated_at"] = now
        else:
            cons = {
                "id": f"cons-{uuid.uuid4().hex[:12]}",
                "case_id": case_id,
                "messages": [message],
                "created_at": now,
                "updated_at": now,
                "is_open": True,
            }
            data["consultations"].append(cons)
        _save_local_db(data)
        return cons

    # Atomic append via Postgres RPC (see supabase_schema/004_*.sql).
    # Replaces the previous read-modify-write which lost messages under
    # concurrent appends from the ward doctor + radiologist.
    resp = db.rpc(
        "append_consultation_message",
        {"p_case_id": case_id, "p_message": message},
    ).execute()

    # rpc() returning a composite row gives back a single dict (not a list).
    data = resp.data
    if isinstance(data, list):
        return data[0] if data else {}
    return data or {}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _utcnow() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------
def log_audit(
    action: str,
    *,
    user_id: Optional[str] = None,
    user_role: Optional[str] = None,
    target_type: Optional[str] = None,
    target_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Best-effort audit-log write. Never raises — logging is observational and
    must not break the request flow if Supabase is unreachable or in mock mode.
    """
    db = get_db()
    if db == "LOCAL_MOCK":
        return  # nothing to write to in mock mode
    try:
        db.table("audit_log").insert({
            "user_id":     user_id,
            "user_role":   user_role,
            "action":      action,
            "target_type": target_type,
            "target_id":   target_id,
            "metadata":    metadata or {},
        }).execute()
    except Exception as exc:
        logger.warning("[Audit] Failed to record %s: %s", action, exc)

    # Bump the actor's last_active_at so System Admin sees a live presence
    # signal in User Management. Best-effort — never raises.
    if user_id:
        try:
            db.table("users").update({"last_active_at": _utcnow()}).eq("id", user_id).execute()
        except Exception:
            pass


def _normalise_joined_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reshape a joined row.
    """
    patient = row.pop("patients", {}) or {}
    predictions = row.pop("predictions", []) or []
    consultations = row.pop("consultations", None)
    
    if isinstance(consultations, list):
        consultation = consultations[0] if consultations else None
    else:
        consultation = consultations

    # Keep labs_raw in the case dict — the frontend Early Risk tab and
    # inference engine both need access to the full 50-lab dictionary.
    # (Previously this was stripped; we now preserve it.)

    return {
        "patient": patient,
        "case": row,
        "predictions": predictions,
        "consultation": consultation,
    }
