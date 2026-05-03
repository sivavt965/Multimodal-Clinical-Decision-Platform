# =============================================================================
# main.py — FastAPI backend for the Multimodal Clinical Decision Support Platform
# Persistence: Supabase (PostgreSQL + Storage) via database.py
# =============================================================================
from dotenv import load_dotenv
load_dotenv()  # Must be called before any os.getenv() reads

import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np

from fastapi import BackgroundTasks, FastAPI, HTTPException, Body, File, Form, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware

from database import (
    append_consultation_message,
    create_new_case,
    get_all_cases,
    get_case_by_id,
    get_db,
    log_audit,
    update_case_inference,
    upload_file_to_bucket,
    upload_local_file_to_bucket,
    CXR_BUCKET,
    HEATMAP_BUCKET,
)


def _actor(request: Request) -> dict:
    """Extract user identity from request headers (set by frontend useUserRole).
    Pre-auth shim: returns {} if headers missing — log entry just has no user."""
    return {
        "user_id":   request.headers.get("X-User-Id") or None,
        "user_role": request.headers.get("X-User-Role") or None,
    }
# Lazy imports — PyTorch / FAISS may not be available in all environments
_run_cxr_inference = None
_vector_store_fn = None

def _get_inference_fn():
    global _run_cxr_inference
    if _run_cxr_inference is None:
        from engine.inference import run_cxr_inference
        _run_cxr_inference = run_cxr_inference
    return _run_cxr_inference

def get_vector_store():
    global _vector_store_fn
    if _vector_store_fn is None:
        from engine.vector_store import get_vector_store as _gvs
        _vector_store_fn = _gvs
    return _vector_store_fn()

from schemas import (
    CaseDetail,
    CaseSummary,
    ConsultationMessage,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Normalise legacy badge strings to the Pydantic RiskBadge enum values
_BADGE_MAP = {
    "High": "Elevated Risk",
    "Moderate": "Monitor",
    "Low": "Unlikely",
    # Already-valid values map to themselves
    "Elevated Risk": "Elevated Risk",
    "Monitor": "Monitor",
    "Unlikely": "Unlikely",
}

def _normalise_badge(raw: str | None) -> str | None:
    """Map legacy or non-conformant risk badges to valid RiskBadge values."""
    if raw is None:
        return None
    return _BADGE_MAP.get(raw, "Monitor")  # default to Monitor for unknowns


# ---------------------------------------------------------------------------
# Lifespan — verify Supabase connectivity on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Verify the Supabase connection at startup, then run."""
    try:
        db = get_db()
        if db == "LOCAL_MOCK":
            logger.info("[OK] Running in local mock mode (Supabase unavailable).")
        else:
            # Lightweight connectivity check — fetch first case row (no data needed)
            db.table("cases").select("id").limit(1).execute()
            logger.info("[OK] Supabase connection verified.")
            
        # Restore persisted FAISS index from disk (populated by real inference runs)
        vs = get_vector_store()
        vs.load()
        logger.info("[OK] FAISS index loaded (%d vectors).", vs.size)
    except Exception as exc:
        logger.error("[WARN] Database connectivity check failed: %s", exc)
        logger.warning("API will start but DB calls may fail until credentials are set.")
    yield
    logger.info("Shutting down Clinical Decision Support API.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Clinical Decision Support API",
    description="Backend for the Symile-MIMIC Multimodal Clinical Workstation",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# GET /api/cases — Dashboard summary list
# ---------------------------------------------------------------------------
@app.get("/api/cases", response_model=List[CaseSummary])
async def get_cases():
    """Return a summary of every active case for the dashboard table."""
    try:
        entries = get_all_cases()
    except Exception as exc:
        logger.error("[GET /api/cases] DB error: %s", exc, exc_info=True)
        raise HTTPException(status_code=503, detail="Database unavailable")

    summaries: List[CaseSummary] = []
    for entry in entries:
        patient      = entry.get("patient", {}) or {}
        case_data    = entry.get("case", {}) or {}
        consultation = entry.get("consultation") or {}
        predictions  = entry.get("predictions", []) or []

        # Filter out zero-probability labels (unmodeled in 8-class checkpoint)
        modeled_preds = [p for p in predictions if p.get("probability", 0) > 0]
        top_pred = (
            max(modeled_preds, key=lambda p: p.get("probability", 0))
            if modeled_preds
            else None
        )

        summaries.append(
            CaseSummary(
                case_id=case_data.get("id", ""),
                patient_name=(
                    f"{patient.get('first_name', '')} {patient.get('last_name', '')}"
                ),
                mrn=patient.get("mrn", ""),
                admitted_at=case_data.get("admitted_at", ""),
                phase_a_risk_level=case_data.get("phase_a_risk_level"),
                top_finding_label=(top_pred.get("label") if top_pred else None),
                top_finding_badge=(_normalise_badge(top_pred.get("risk_badge")) if top_pred else None),
                top_finding_probability=(
                    top_pred.get("probability") if top_pred else None
                ),
                consultation_open=bool(consultation.get("is_open", False)),
                urgency_flag=bool(consultation.get("urgency_flag", False)),
                cxr_dicom_url=case_data.get("cxr_dicom_url"),
            )
        )

    return summaries


# ---------------------------------------------------------------------------
# Background inference + Supabase update task
# ---------------------------------------------------------------------------

def _run_inference_task(
    case_id: str,
    local_image_path: str,
    storage_dest: str,
    target_label: str = "Pleural Effusion",
) -> None:
    """
    Runs in a thread-pool worker after the HTTP response has been sent.

    Steps:
      1. Run DenseNet121 + Grad-CAM on the saved image.
      2. Upload the heatmap PNG to the Supabase 'heatmaps' bucket.
      3. Persist predictions + heatmap URL in Supabase via update_case_inference().
    """
    logger.info("[BG] Inference starting for case %s …", case_id)

    try:
        run_cxr_inference = _get_inference_fn()
        result = run_cxr_inference(
            image_path=local_image_path,
            target_label=target_label,
            case_id=case_id,          # triggers FAISS indexing inside inference
        )
    except Exception as exc:
        logger.error("[BG] Inference crashed for case %s: %s", case_id, exc, exc_info=True)
        return

    if result.error:
        logger.warning("[BG] Inference error for case %s: %s", case_id, result.error)
        return

    # ── Upload heatmap PNG to Supabase Storage ────────────────────────────────
    heatmap_storage_url: str | None = None
    if result.heatmap_url:
        # result.heatmap_url is a local public URL like /mock-data/heatmaps/xxx.png
        # Resolve the local file path from the project root
        _repo_root = Path(__file__).resolve().parent.parent
        local_heatmap = (
            _repo_root / "frontend" / "public" / result.heatmap_url.lstrip("/")
        )
        if local_heatmap.exists():
            try:
                dest = f"{case_id}/{local_heatmap.name}"
                heatmap_storage_url = upload_local_file_to_bucket(
                    HEATMAP_BUCKET, dest, local_heatmap, content_type="image/png"
                )
            except Exception as exc:
                logger.error("[BG] Heatmap upload failed for case %s: %s", case_id, exc)
                heatmap_storage_url = result.heatmap_url  # fall back to local path
        else:
            logger.warning("[BG] Heatmap file not found on disk: %s", local_heatmap)

    # ── Persist to Supabase ───────────────────────────────────────────────────
    try:
        update_case_inference(
            case_id=case_id,
            predictions=result.predictions,
            heatmap_url=heatmap_storage_url,
            heatmap_label=result.heatmap_label,
        )
    except Exception as exc:
        logger.error("[BG] DB update failed for case %s: %s", case_id, exc, exc_info=True)
        return

    logger.info(
        "[BG] Case %s updated — top: %s (%.3f)",
        case_id,
        max(result.probabilities, key=result.probabilities.get),  # type: ignore
        max(result.probabilities.values()),
    )


# ---------------------------------------------------------------------------
# POST /api/cases — Register new case (Ingestion Wizard)
# ---------------------------------------------------------------------------
@app.post("/api/cases", response_model=CaseSummary)
async def create_case(
    background_tasks: BackgroundTasks,
    request: Request,
    case_data: str = Form(...),
    image: UploadFile = File(None),
):
    """Handle multipart form data to register a new clinical case."""
    try:
        data = json.loads(case_data)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON in case_data")

    case_id    = str(uuid.uuid4())
    patient_id = str(uuid.uuid4())
    now        = datetime.now(timezone.utc).isoformat()

    # ── Extract full lab dictionary from the wizard ─────────────────────────────
    # The wizard sends labs as { itemid: value | null }, e.g. { "50912": 1.2, ... }
    labs_raw: dict = data.get("labs", {}) or {}
    logger.info("[POST /api/cases] Received %d lab values from wizard", len(labs_raw))

    # Map well-known itemids → LabData schema fields for backward compat
    def _lab(itemid: str) -> float:
        """Extract a lab value by MIMIC-IV itemid, default 0."""
        v = labs_raw.get(itemid)
        return float(v) if v is not None else 0.0

    troponin_val    = _lab("50947")   # Troponin I
    bnp_val         = _lab("51006")   # Urea Nitrogen (BNP proxy — no direct BNP in MIMIC)
    wbc_val         = _lab("51301")   # White Blood Cells
    creatinine_val  = _lab("50912")   # Creatinine
    sodium_val      = _lab("50983")   # Sodium
    potassium_val   = _lab("50971")   # Potassium
    lactate_val     = _lab("50813")   # Lactate

    # ── Phase A risk heuristic ────────────────────────────────────────────────
    # Simple multi-marker rule: elevations in troponin, creatinine, or lactate
    risk_score = 0.0
    if troponin_val > 0.04:
        risk_score += 0.4
    if creatinine_val > 1.3:
        risk_score += 0.2
    if lactate_val > 2.0:
        risk_score += 0.2
    if potassium_val > 5.0 or (potassium_val > 0 and potassium_val < 3.5):
        risk_score += 0.1
    if sodium_val > 0 and (sodium_val < 136 or sodium_val > 145):
        risk_score += 0.1

    if risk_score >= 0.4:
        draft_risk_level = "High"
    elif risk_score >= 0.2:
        draft_risk_level = "Moderate"
    else:
        draft_risk_level = "Low"

    # ── Upload CXR image to Supabase Storage ──────────────────────────────────
    cxr_storage_url: str | None  = None
    local_image_path: str | None = None
    storage_dest: str | None     = None

    if image:
        file_bytes = await image.read()
        file_ext   = Path(image.filename).suffix if image.filename else ".png"
        file_name  = f"case_{case_id[:8]}{file_ext}"
        storage_dest = f"{case_id}/{file_name}"

        # ── Save locally (needed by PyTorch inference engine) ─────────────────
        _repo_root = Path(__file__).resolve().parent.parent
        local_dir  = _repo_root / "frontend" / "public" / "mock-data" / "dicoms"
        local_dir.mkdir(parents=True, exist_ok=True)
        local_file = local_dir / file_name
        local_file.write_bytes(file_bytes)
        local_image_path = str(local_file)

        # ── Push to Supabase Storage ──────────────────────────────────────────
        try:
            cxr_storage_url = upload_file_to_bucket(
                CXR_BUCKET,
                storage_dest,
                file_bytes,
                content_type=image.content_type or "image/png",
            )
        except Exception as exc:
            logger.error("[POST /api/cases] CXR upload failed: %s", exc)
            # Non-fatal — inference can still run from the local copy
            cxr_storage_url = f"/mock-data/dicoms/{file_name}"

    # ── Build patient payload ─────────────────────────────────────────────────
    patient_payload = {
        "id":                patient_id,
        "mrn":               data.get("mrn", f"MRN-{case_id[:5]}"),
        "first_name":        data.get("firstName", "Unknown"),
        "last_name":         data.get("lastName", "Patient"),
        "date_of_birth":     "1900-01-01",   # wizard doesn't collect DOB yet
        "sex":               data.get("sex", "M"),
        "age_at_admission":  int(data.get("age") or 0),
        "mimic_subject_id":  None,
        "created_at":        now,
        "updated_at":        now,
    }

    # ── Build case payload ────────────────────────────────────────────────────
    case_payload = {
        "id":                     case_id,
        "patient_id":             patient_id,
        "admitted_at":            now,
        "discharged_at":          None,
        "ecg_data": {
            "heart_rate": 0, "pr_interval_ms": 0, "qrs_duration_ms": 0,
            "qtc_ms": 0, "st_deviation_mm": 0,
            "rhythm_interpretation": data.get("rhythm", "Unknown"),
            "acquired_at": now,
        },
        "lab_data": {
            "troponin_ng_ml":   troponin_val,
            "bnp_pg_ml":        bnp_val,
            "wbc_count":        wbc_val,
            "creatinine_mg_dl": creatinine_val,
            "sodium_meq_l":     sodium_val,
            "potassium_meq_l":  potassium_val,
            "lactate_mmol_l":   lactate_val,
            "collected_at":     now,
        },
        # Store the full 50-lab dictionary for inference engine use
        "labs_raw":                labs_raw,
        "phase_a_risk_level":      draft_risk_level,
        "phase_a_risk_score":      min(risk_score, 1.0),
        "phase_a_recommendation":  (
            "Elevated cardiac and metabolic markers. Recommend urgent imaging and monitoring."
            if draft_risk_level == "High"
            else "Moderate risk indicators detected. Consider follow-up imaging."
            if draft_risk_level == "Moderate"
            else "Low-risk profile based on initial labs. Continue standard monitoring."
        ),
        "phase_a_run_at":          now,
        "cxr_dicom_url":           cxr_storage_url,
        "cxr_acquired_at":         now if cxr_storage_url else None,
        "cxr_heatmap_url":         None,
        "cxr_heatmap_label":       None,
        "mimic_study_id":          None,
        "created_at":              now,
        "updated_at":              now,
    }

    # ── Persist to Supabase ───────────────────────────────────────────────────
    try:
        entry = create_new_case(patient_payload, case_payload)
    except Exception as exc:
        logger.error("[POST /api/cases] DB insert failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=503, detail="Database unavailable")

    patient  = entry["patient"]
    case_obj = entry["case"]

    log_audit(
        "case.create",
        **_actor(request),
        target_type="cases",
        target_id=case_id,
        metadata={
            "phase_a_risk_level": case_obj.get("phase_a_risk_level"),
            "has_cxr": bool(local_image_path),
            "labs_count": len(labs_raw),
        },
    )

    # ── Enqueue Grad-CAM inference (non-blocking) ─────────────────────────────
    if local_image_path:
        background_tasks.add_task(
            _run_inference_task,
            case_id=case_id,
            local_image_path=local_image_path,
            storage_dest=storage_dest or "",
            target_label="Pleural Effusion",
        )

    # ── Return summary immediately ─────────────────────────────────────────────
    return CaseSummary(
        case_id=case_id,
        patient_name=f"{patient['first_name']} {patient['last_name']}",
        mrn=patient["mrn"],
        admitted_at=now,
        phase_a_risk_level=case_obj.get("phase_a_risk_level"),
        top_finding_label=None,
        top_finding_badge=None,
        top_finding_probability=None,
        consultation_open=False,
        urgency_flag=False,
    )


# ---------------------------------------------------------------------------
# GET /api/cases/{case_id} — Full multimodal detail for one patient
# ---------------------------------------------------------------------------
@app.get("/api/cases/{case_id}", response_model=CaseDetail)
async def get_case(case_id: str):
    """Return the full multimodal payload for a specific case ID."""
    try:
        entry = get_case_by_id(case_id)
    except Exception as exc:
        logger.error("[GET /api/cases/%s] DB error: %s", case_id, exc, exc_info=True)
        raise HTTPException(status_code=503, detail="Database unavailable")

    if entry is None:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found")

    return entry


# ---------------------------------------------------------------------------
# POST /api/consultation/{case_id} — Save a chat message
# ---------------------------------------------------------------------------
@app.post("/api/consultation/{case_id}")
async def save_consultation_message(
    case_id: str,
    message: ConsultationMessage = Body(...),
):
    """Append a consultation message and persist the thread to Supabase."""
    try:
        cons = append_consultation_message(
            case_id=case_id,
            message=message.model_dump(),
        )
    except Exception as exc:
        logger.error(
            "[POST /api/consultation/%s] DB error: %s", case_id, exc, exc_info=True
        )
        raise HTTPException(status_code=503, detail="Database unavailable")

    messages = cons.get("messages", []) or []
    return {
        "status": "ok",
        "message_id": message.id,
        "thread_length": len(messages),
    }


# ---------------------------------------------------------------------------
# GET /api/cases/{case_id}/similar — FAISS similarity search
# ---------------------------------------------------------------------------
@app.get("/api/cases/{case_id}/similar", response_model=List[CaseSummary])
async def get_similar_cases(
    case_id: str,
    top_k: int = 3,
):
    """
    Return the *top_k* most visually similar historical cases using
    FAISS L2 nearest-neighbour search on 1024-d DenseNet121 GAP embeddings.

    The query case must already have been processed by the inference
    background task (i.e. its embedding is in the in-process FAISS index).

    Query parameters
    ----------------
    top_k : int  — number of similar cases to return (default 3, max 20)
    """
    top_k = max(1, min(top_k, 20))   # clamp to sensible bounds

    vs = get_vector_store()

    # ── Retrieve the query embedding from the FAISS index ─────────────────────
    logger.info(
        "[/similar] Querying FAISS for case %s (top_k=%d, index_size=%d)",
        case_id, top_k, vs.size,
    )

    query_vec = vs.get_embedding(case_id)
    if query_vec is None:
        logger.warning("[/similar] No embedding found for case %s", case_id)
        raise HTTPException(
            status_code=404,
            detail=(
                f"No embedding found for case '{case_id}'. "
                "The inference background task may not have completed yet."
            ),
        )

    logger.info(
        "[/similar] Query vector shape=%s, norm=%.4f",
        query_vec.shape, float(np.linalg.norm(query_vec)),
    )

    # ── Run FAISS search ──────────────────────────────────────────────────────
    hits = vs.search_similar(
        query_embedding=query_vec,
        top_k=top_k,
        exclude_case_id=case_id,
    )

    logger.info("[/similar] FAISS returned %d hits for case %s", len(hits), case_id)

    if not hits:
        return []

    # Build relative similarity scores so results show meaningful spread.
    # Raw L2 distances on an uncalibrated backbone cluster tightly (e.g. 0.037–0.044),
    # making absolute conversion (1 - d/2)*100 output ~98% for every result.
    # Instead we normalise within the returned result set:
    #   best match  → 95%
    #   worst match → 60%  (linear interpolation)
    # This honestly represents relative ranking while avoiding misleading absolutes.
    distances = [h["distance"] for h in hits]
    d_min, d_max = min(distances), max(distances)
    def _relative_score(d: float) -> float:
        if d_max == d_min:
            return 95.0
        return 95.0 - (d - d_min) / (d_max - d_min) * 35.0  # 95→60 range
    hit_scores = {h["case_id"]: round(_relative_score(h["distance"]), 1) for h in hits}

    # ── Resolve case UUIDs → full CaseSummary objects from Supabase ──────────
    summaries: List[CaseSummary] = []

    for cid in [h["case_id"] for h in hits]:
        try:
            entry = get_case_by_id(cid)
        except Exception as exc:
            logger.warning("[/similar] Could not fetch case %s: %s", cid, exc)
            continue

        if entry is None:
            continue

        patient      = entry.get("patient", {}) or {}
        case_data    = entry.get("case", {}) or {}
        consultation = entry.get("consultation") or {}
        predictions  = entry.get("predictions", []) or []

        modeled_preds = [p for p in predictions if p.get("probability", 0) > 0]
        top_pred = (
            max(modeled_preds, key=lambda p: p.get("probability", 0))
            if modeled_preds
            else None
        )

        summaries.append(
            CaseSummary(
                case_id=case_data.get("id", cid),
                patient_name=(
                    f"{patient.get('first_name', '')} {patient.get('last_name', '')}"
                ),
                mrn=patient.get("mrn", ""),
                admitted_at=case_data.get("admitted_at", ""),
                phase_a_risk_level=case_data.get("phase_a_risk_level"),
                top_finding_label=(top_pred.get("label") if top_pred else None),
                top_finding_badge=(_normalise_badge(top_pred.get("risk_badge")) if top_pred else None),
                top_finding_probability=(
                    top_pred.get("probability") if top_pred else None
                ),
                consultation_open=bool(consultation.get("is_open", False)),
                urgency_flag=bool(consultation.get("urgency_flag", False)),
                similarity_score=hit_scores.get(cid),
                cxr_dicom_url=case_data.get("cxr_dicom_url"),
            )
        )

    return summaries


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# POST /api/cases/{case_id}/flag — Radiologist flags a critical finding
# Sets urgency_flag=True on the consultation row (creates one if none exists)
# so the ward doctor sees it in their dashboard.
# ---------------------------------------------------------------------------
@app.post("/api/cases/{case_id}/flag")
async def flag_case_critical(case_id: str, request: Request, payload: dict = Body(default={})):
    finding = (payload or {}).get("finding") or "Unspecified"
    note    = (payload or {}).get("note") or ""

    db = get_db()
    now = datetime.now(timezone.utc).isoformat()

    if db == "LOCAL_MOCK":
        from database import _load_local_db, _save_local_db
        data = _load_local_db()
        cons = next((c for c in data.get("consultations", []) if c["case_id"] == case_id), None)
        if cons:
            cons["urgency_flag"] = True
            cons["updated_at"]   = now
        else:
            data.setdefault("consultations", []).append({
                "id":           f"cons-{uuid.uuid4().hex[:12]}",
                "case_id":      case_id,
                "is_open":      True,
                "urgency_flag": True,
                "messages":     [],
                "created_at":   now,
                "updated_at":   now,
            })
        _save_local_db(data)
    else:
        # Try update first; if no row, insert one
        existing = (
            db.table("consultations").select("id").eq("case_id", case_id).limit(1).execute()
        )
        if existing.data:
            db.table("consultations").update(
                {"urgency_flag": True, "is_open": True, "updated_at": now}
            ).eq("case_id", case_id).execute()
        else:
            actor = _actor(request)
            db.table("consultations").insert({
                "case_id":        case_id,
                "ward_doctor_id": "00000000-0000-0000-0000-000000000000",
                "radiologist_id": actor.get("user_id"),
                "is_open":        True,
                "urgency_flag":   True,
                "messages":       [],
            }).execute()

    log_audit(
        "case.flag_critical",
        **_actor(request),
        target_type="cases",
        target_id=case_id,
        metadata={"finding": finding, "note": note},
    )

    return {"status": "ok", "case_id": case_id, "urgency_flag": True, "finding": finding}


@app.post("/api/admin/reload-faiss")
async def admin_reload_faiss(request: Request):
    """Force-reload the FAISS index from disk. Admin use only."""
    vs = get_vector_store()
    vs.load()
    log_audit(
        "faiss.reload",
        **_actor(request),
        target_type="system",
        metadata={"vectors_loaded": vs.size},
    )
    return {"status": "ok", "faiss_index_size": vs.size}


# ---------------------------------------------------------------------------
# GET /api/admin/users — list all platform users
# Used by System Admin → User Management. Auth gating (system_admin only) is
# applied client-side until Phase 5 wires Supabase Auth + middleware.
# ---------------------------------------------------------------------------
@app.get("/api/admin/users")
async def admin_list_users():
    db = get_db()
    if db == "LOCAL_MOCK":
        return []
    resp = (
        db.table("users")
        .select("id,email,full_name,role,status,last_active_at,created_at,updated_at")
        .order("created_at", desc=True)
        .execute()
    )
    return resp.data or []


# ---------------------------------------------------------------------------
# POST /api/admin/users — create a new platform user
# ---------------------------------------------------------------------------
@app.post("/api/admin/users")
async def admin_create_user(request: Request, payload: dict = Body(...)):
    email     = (payload.get("email") or "").strip().lower()
    full_name = (payload.get("full_name") or "").strip()
    role      = payload.get("role")
    if not email or not full_name or not role:
        raise HTTPException(status_code=400, detail="email, full_name and role are required")
    if role not in ("radiologist", "ward_doctor", "clinical_admin", "system_admin"):
        raise HTTPException(status_code=400, detail=f"Invalid role: {role}")

    db = get_db()
    if db == "LOCAL_MOCK":
        raise HTTPException(status_code=503, detail="User management requires Supabase")

    try:
        resp = db.table("users").insert({
            "email": email, "full_name": full_name, "role": role, "status": "active",
        }).execute()
    except Exception as exc:
        raise HTTPException(status_code=409, detail=f"Could not create user: {exc}")

    new_user = resp.data[0] if resp.data else None
    log_audit(
        "user.create",
        **_actor(request),
        target_type="users",
        target_id=new_user["id"] if new_user else None,
        metadata={"email": email, "role": role},
    )
    return new_user


# ---------------------------------------------------------------------------
# PATCH /api/admin/users/{user_id} — update role and/or status
# ---------------------------------------------------------------------------
@app.patch("/api/admin/users/{user_id}")
async def admin_update_user(user_id: str, request: Request, payload: dict = Body(...)):
    updates: dict = {}
    if "role" in payload:
        if payload["role"] not in ("radiologist", "ward_doctor", "clinical_admin", "system_admin"):
            raise HTTPException(status_code=400, detail=f"Invalid role: {payload['role']}")
        updates["role"] = payload["role"]
    if "status" in payload:
        if payload["status"] not in ("active", "inactive", "suspended"):
            raise HTTPException(status_code=400, detail=f"Invalid status: {payload['status']}")
        updates["status"] = payload["status"]
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    db = get_db()
    if db == "LOCAL_MOCK":
        raise HTTPException(status_code=503, detail="User management requires Supabase")

    resp = db.table("users").update(updates).eq("id", user_id).execute()
    if not resp.data:
        raise HTTPException(status_code=404, detail="User not found")

    log_audit(
        "user.update",
        **_actor(request),
        target_type="users",
        target_id=user_id,
        metadata=updates,
    )
    return resp.data[0]


# ---------------------------------------------------------------------------
# GET /api/admin/audit-log — paginated audit trail
# Optional query params: limit (default 50, max 500), action, user_id
# ---------------------------------------------------------------------------
@app.get("/api/admin/audit-log")
async def admin_audit_log(
    limit: int = 50,
    offset: int = 0,
    action: Optional[str] = None,
    user_id: Optional[str] = None,
):
    limit = max(1, min(limit, 500))
    db = get_db()
    if db == "LOCAL_MOCK":
        return {"total": 0, "items": []}

    query = (
        db.table("audit_log")
        .select("*", count="exact")
        .order("created_at", desc=True)
        .range(offset, offset + limit - 1)
    )
    if action:
        query = query.eq("action", action)
    if user_id:
        query = query.eq("user_id", user_id)
    resp = query.execute()
    return {"total": resp.count or 0, "items": resp.data or []}


@app.get("/api/health")
async def health():
    try:
        db = get_db()
        if db == "LOCAL_MOCK":
            db_status = "ok (mock mode)"
            cases_count = len(get_all_cases())
        else:
            resp = db.table("cases").select("id", count="exact").limit(0).execute()
            cases_count = resp.count if resp.count is not None else "unknown"
            db_status = "ok"
    except Exception as exc:
        cases_count = 0
        db_status = f"error: {exc}"

    vs = get_vector_store()
    return {
        "status": "ok",
        "db_status": db_status,
        "cases_in_db": cases_count,
        "faiss_index_size": vs.size,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# PATCH /api/cases/{case_id}/complete — Archive a case
# ---------------------------------------------------------------------------
@app.patch("/api/cases/{case_id}/complete")
async def complete_case(case_id: str, request: Request):
    """Mark a case as discharged / completed."""
    try:
        db = get_db()
        now = datetime.now(timezone.utc).isoformat()

        if db == "LOCAL_MOCK":
            from database import _load_local_db, _save_local_db

            data = _load_local_db()
            case_row = next((c for c in data["cases"] if c["id"] == case_id), None)
            if case_row is None:
                raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found")
            case_row["discharged_at"] = now
            case_row["updated_at"] = now

            # Also close the consultation if open
            for cons in data.get("consultations", []):
                if cons["case_id"] == case_id and cons.get("is_open"):
                    cons["is_open"] = False
                    cons["closed_at"] = now

            _save_local_db(data)
        else:
            db.table("cases").update({
                "discharged_at": now,
                "updated_at": now,
            }).eq("id", case_id).execute()

        log_audit(
            "case.complete",
            **_actor(request),
            target_type="cases",
            target_id=case_id,
            metadata={"discharged_at": now},
        )
        return {"status": "ok", "case_id": case_id, "discharged_at": now}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[PATCH /complete] Error: %s", exc, exc_info=True)
        raise HTTPException(status_code=503, detail="Database unavailable")


# ---------------------------------------------------------------------------
# DELETE /api/cases/{case_id} — Remove a case entirely
# ---------------------------------------------------------------------------
@app.delete("/api/cases/{case_id}")
async def delete_case(case_id: str):
    """Remove a case and all associated data (predictions, consultations)."""
    try:
        db = get_db()

        if db == "LOCAL_MOCK":
            from database import _load_local_db, _save_local_db

            data = _load_local_db()
            case_row = next((c for c in data["cases"] if c["id"] == case_id), None)
            if case_row is None:
                raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found")

            patient_id = case_row["patient_id"]

            # Remove the case
            data["cases"] = [c for c in data["cases"] if c["id"] != case_id]

            # Remove predictions
            data["predictions"] = [p for p in data["predictions"] if p["case_id"] != case_id]

            # Remove consultations
            data["consultations"] = [c for c in data.get("consultations", []) if c["case_id"] != case_id]

            # Remove patient if no other cases reference them
            other_cases_for_patient = [c for c in data["cases"] if c["patient_id"] == patient_id]
            if not other_cases_for_patient:
                data["patients"] = [p for p in data["patients"] if p["id"] != patient_id]

            _save_local_db(data)
        else:
            db.table("predictions").delete().eq("case_id", case_id).execute()
            db.table("consultations").delete().eq("case_id", case_id).execute()
            db.table("cases").delete().eq("id", case_id).execute()

        return {"status": "ok", "case_id": case_id, "deleted": True}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[DELETE /cases] Error: %s", exc, exc_info=True)
        raise HTTPException(status_code=503, detail="Database unavailable")

# ---------------------------------------------------------------------------
# POST /api/labs/parse — CSV / JSON lab parsing utility
# ---------------------------------------------------------------------------
# Pre-load the labs_means.json for percentile imputation
_REPO_ROOT = Path(__file__).resolve().parent.parent
_LABS_MEANS_PATH = _REPO_ROOT / "labs_means.json"
_LABS_MEANS: dict | None = None

def _get_labs_means() -> dict:
    global _LABS_MEANS
    if _LABS_MEANS is None:
        if _LABS_MEANS_PATH.exists():
            _LABS_MEANS = json.loads(_LABS_MEANS_PATH.read_text())
        else:
            _LABS_MEANS = {}
            logger.warning("[labs] labs_means.json not found at %s", _LABS_MEANS_PATH)
    return _LABS_MEANS


# MIMIC-IV lab itemid → human-readable name mapping
_LAB_NAMES: dict[str, str] = {
    "51221": "Hematocrit", "51265": "Platelet Count", "50912": "Creatinine",
    "50971": "Potassium", "51222": "Hemoglobin", "51301": "White Blood Cells",
    "51249": "MCHC", "51279": "Red Blood Cells", "51250": "MCV", "51248": "MCH",
    "51277": "RDW", "51006": "Urea Nitrogen", "50983": "Sodium", "50902": "Chloride",
    "50882": "Bicarbonate", "50868": "Anion Gap", "50931": "Glucose",
    "50960": "Magnesium", "50893": "Calcium, Total", "50970": "Phosphate",
    "51237": "INR(PT)", "51274": "PT", "51275": "PTT", "51146": "Basophils",
    "51256": "Neutrophils", "51254": "Monocytes", "51200": "Eosinophils",
    "51244": "Lymphocytes", "52172": "RDW-SD", "50934": "H", "51678": "L",
    "50947": "I", "50861": "ALT", "50878": "AST", "50813": "Lactate",
    "50863": "Alkaline Phosphatase", "50885": "Bilirubin, Total", "50820": "pH",
    "50862": "Albumin", "50802": "Base Excess", "50821": "pO2",
    "50804": "Calculated Total CO2", "50818": "pCO2",
    "52075": "Absolute Neutrophil Count", "52073": "Absolute Eosinophil Count",
    "52074": "Absolute Monocyte Count", "52069": "Absolute Basophil Count",
    "51133": "Absolute Lymphocyte Count", "50910": "Creatine Kinase (CK)",
    "52135": "Immature Granulocytes",
}

# Reverse map: lowercase name → itemid
_LAB_NAME_TO_ID: dict[str, str] = {v.lower(): k for k, v in _LAB_NAMES.items()}


@app.post("/api/labs/parse")
async def parse_labs(
    file: UploadFile = File(...),
):
    """
    Parse a CSV or JSON file containing lab values into the structured JSON
    format expected by the inference engine.

    Returns
    -------
    {
      "status": "ok",
      "lab_count": int,
      "labs": { itemid: value, ... },                   // raw values
      "labs_percentile_vector": { "XXXXX_percentile": float, ... }  // 50-d vector
    }

    CSV format — header row with itemids or lab names, value row:
        51221, 50912, 51301
        38.2,  1.1,   8.5

    JSON format — object keyed by itemid or lab name:
        {"51221": 38.2, "Creatinine": 1.1}
    """
    raw_text = (await file.read()).decode("utf-8", errors="replace").strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Empty file")

    parsed: dict[str, float] = {}

    # ── Detect format ────────────────────────────────────────────────────
    if file.filename and file.filename.lower().endswith(".json"):
        try:
            obj = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}")
        for key, val in obj.items():
            _parse_lab_entry(key, val, parsed)

    else:
        # Assume CSV
        lines = raw_text.split("\n")
        if len(lines) < 2:
            raise HTTPException(status_code=400, detail="CSV must have at least a header and one value row")
        headers = [h.strip().strip('"') for h in lines[0].split(",")]
        values  = [v.strip().strip('"') for v in lines[1].split(",")]
        for h, v in zip(headers, values):
            try:
                num = float(v)
            except (ValueError, TypeError):
                continue
            _parse_lab_entry(h, num, parsed)

    # ── Build the 50-d percentile vector ─────────────────────────────────
    labs_means = _get_labs_means()
    percentile_vector: dict[str, float] = {}
    for pkey in sorted(labs_means.keys()):
        itemid = pkey.replace("_percentile", "")
        if itemid in parsed:
            # User provided a raw value — we use the mean percentile as a
            # placeholder because converting raw → percentile requires the
            # full CDF.  In production you'd apply the empirical CDF here.
            percentile_vector[pkey] = labs_means[pkey]
        else:
            # Missing → impute with training mean
            percentile_vector[pkey] = labs_means[pkey]

    return {
        "status": "ok",
        "lab_count": len(parsed),
        "labs": parsed,
        "labs_percentile_vector": percentile_vector,
    }


def _compute_risk_from_lab_data(lab_data: dict) -> tuple[str, float]:
    """Return (risk_level, risk_score) from a lab_data dict."""
    def lv(k): return float(lab_data.get(k) or 0)
    score = 0.0
    if lv("troponin_ng_ml")   > 0.04: score += 0.4
    if lv("creatinine_mg_dl") > 1.3:  score += 0.2
    if lv("lactate_mmol_l")   > 2.0:  score += 0.2
    k = lv("potassium_meq_l")
    if k > 5.0 or (0 < k < 3.5):      score += 0.1
    na = lv("sodium_meq_l")
    if 0 < na < 136 or na > 145:       score += 0.1
    score = min(score, 1.0)
    level = "High" if score >= 0.4 else "Moderate" if score >= 0.2 else "Low"
    return level, score


# ---------------------------------------------------------------------------
# POST /api/cases/{case_id}/upload/cxr — Upload CXR image for an existing case
# ---------------------------------------------------------------------------
@app.post("/api/cases/{case_id}/upload/cxr", response_model=CaseDetail)
async def upload_case_cxr(
    case_id: str,
    background_tasks: BackgroundTasks,
    request: Request,
    image: UploadFile = File(...),
):
    """Upload a CXR image for a case and enqueue Phase B inference."""
    # Validate
    allowed = {".png", ".jpg", ".jpeg", ".dcm"}
    file_ext = Path(image.filename).suffix.lower() if image.filename else ".png"
    if file_ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported format. Accepted: PNG, JPEG, DICOM")

    file_bytes = await image.read()
    if len(file_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large — maximum 10 MB")

    # Save locally — include a timestamp so each upload gets a unique URL.
    # Without this, replacing a CXR keeps the same filename, the browser
    # caches the old image, and the UI never visually updates.
    _repo_root_local = Path(__file__).resolve().parent.parent
    local_dir  = _repo_root_local / "frontend" / "public" / "mock-data" / "dicoms"
    local_dir.mkdir(parents=True, exist_ok=True)
    ts_suffix  = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    file_name  = f"case_{case_id[:8]}_{ts_suffix}{file_ext}"
    local_file = local_dir / file_name
    local_file.write_bytes(file_bytes)

    # Try Supabase upload
    try:
        cxr_url = upload_file_to_bucket(
            CXR_BUCKET, f"{case_id}/{file_name}", file_bytes,
            content_type=image.content_type or "image/png",
        )
    except Exception:
        cxr_url = f"/mock-data/dicoms/{file_name}"

    # Update DB
    now = datetime.now(timezone.utc).isoformat()
    db  = get_db()
    if db == "LOCAL_MOCK":
        from database import _load_local_db, _save_local_db
        data = _load_local_db()
        row = next((c for c in data["cases"] if c["id"] == case_id), None)
        if row is None:
            raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found")
        row["cxr_dicom_url"]  = cxr_url
        row["cxr_acquired_at"] = now
        row["updated_at"]      = now
        _save_local_db(data)
    else:
        db.table("cases").update({"cxr_dicom_url": cxr_url, "cxr_acquired_at": now, "updated_at": now}).eq("id", case_id).execute()

    # Enqueue inference
    background_tasks.add_task(
        _run_inference_task,
        case_id=case_id,
        local_image_path=str(local_file),
        storage_dest=f"{case_id}/{file_name}",
        target_label="Pleural Effusion",
    )

    log_audit(
        "cxr.upload",
        **_actor(request),
        target_type="cases",
        target_id=case_id,
        metadata={"filename": file_name, "size_bytes": len(file_bytes)},
    )

    entry = get_case_by_id(case_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found")
    return entry


# ---------------------------------------------------------------------------
# POST /api/cases/{case_id}/upload/ecg — Update ECG data and recalculate Phase A
# ---------------------------------------------------------------------------
@app.post("/api/cases/{case_id}/upload/ecg", response_model=CaseDetail)
async def upload_case_ecg(case_id: str, request: Request, ecg_data: dict = Body(...)):
    """Save structured ECG data for a case and recalculate Phase A risk."""
    now = datetime.now(timezone.utc).isoformat()
    ecg_payload = {
        "heart_rate":           float(ecg_data.get("heart_rate", 0)),
        "pr_interval_ms":       float(ecg_data.get("pr_interval_ms", 0)),
        "qrs_duration_ms":      float(ecg_data.get("qrs_duration_ms", 0)),
        "qtc_ms":               float(ecg_data.get("qtc_ms", 0)),
        "st_deviation_mm":      float(ecg_data.get("st_deviation_mm", 0)),
        "rhythm_interpretation": str(ecg_data.get("rhythm_interpretation", "Unknown")),
        "acquired_at":          ecg_data.get("acquired_at", now),
    }

    db = get_db()
    if db == "LOCAL_MOCK":
        from database import _load_local_db, _save_local_db
        data = _load_local_db()
        row = next((c for c in data["cases"] if c["id"] == case_id), None)
        if row is None:
            raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found")
        row["ecg_data"]   = ecg_payload
        row["updated_at"] = now
        # Recalculate Phase A using existing lab values
        lab_data = row.get("lab_data") or {}
        level, score = _compute_risk_from_lab_data(lab_data)
        row["phase_a_risk_level"] = level
        row["phase_a_risk_score"] = score
        row["phase_a_run_at"]     = now
        _save_local_db(data)
    else:
        db.table("cases").update({"ecg_data": ecg_payload, "updated_at": now}).eq("id", case_id).execute()

    log_audit(
        "ecg.upload",
        **_actor(request),
        target_type="cases",
        target_id=case_id,
        metadata={"rhythm": ecg_payload["rhythm_interpretation"], "hr": ecg_payload["heart_rate"]},
    )

    entry = get_case_by_id(case_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found")
    return entry


# ---------------------------------------------------------------------------
# POST /api/cases/{case_id}/upload/labs — Parse lab file and update Phase A
# ---------------------------------------------------------------------------
@app.post("/api/cases/{case_id}/upload/labs", response_model=CaseDetail)
async def upload_case_labs(case_id: str, request: Request, file: UploadFile = File(...)):
    """Parse a JSON/CSV lab file, save it to the case, and recalculate Phase A."""
    raw_text = (await file.read()).decode("utf-8", errors="replace").strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Empty file")

    parsed: dict[str, float] = {}

    if file.filename and file.filename.lower().endswith(".json"):
        try:
            obj = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}")
        for key, val in obj.items():
            _parse_lab_entry(key, val, parsed)
    else:
        lines = raw_text.split("\n")
        if len(lines) < 2:
            raise HTTPException(status_code=400, detail="CSV must have at least a header and value row")
        headers = [h.strip().strip('"') for h in lines[0].split(",")]
        values  = [v.strip().strip('"') for v in lines[1].split(",")]
        for h, v in zip(headers, values):
            try:
                _parse_lab_entry(h, float(v), parsed)
            except (ValueError, TypeError):
                continue

    # Build structured lab_data from parsed values
    def _lab(itemid): return parsed.get(itemid, 0.0)
    lab_data = {
        "troponin_ng_ml":   _lab("50947"),
        "bnp_pg_ml":        _lab("51006"),
        "wbc_count":        _lab("51301"),
        "creatinine_mg_dl": _lab("50912"),
        "sodium_meq_l":     _lab("50983"),
        "potassium_meq_l":  _lab("50971"),
        "lactate_mmol_l":   _lab("50813"),
        "collected_at":     datetime.now(timezone.utc).isoformat(),
    }
    level, score = _compute_risk_from_lab_data(lab_data)
    now = datetime.now(timezone.utc).isoformat()

    db = get_db()
    if db == "LOCAL_MOCK":
        from database import _load_local_db, _save_local_db
        data = _load_local_db()
        row = next((c for c in data["cases"] if c["id"] == case_id), None)
        if row is None:
            raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found")
        row["lab_data"]          = lab_data
        row["labs_raw"]          = parsed
        row["phase_a_risk_level"] = level
        row["phase_a_risk_score"] = score
        row["phase_a_run_at"]     = now
        row["updated_at"]         = now
        _save_local_db(data)
    else:
        db.table("cases").update({
            "lab_data": lab_data, "labs_raw": parsed,
            "phase_a_risk_level": level, "phase_a_risk_score": score,
            "phase_a_run_at": now, "updated_at": now,
        }).eq("id", case_id).execute()

    log_audit(
        "labs.upload",
        **_actor(request),
        target_type="cases",
        target_id=case_id,
        metadata={"labs_count": len(parsed), "phase_a_risk_level": level, "phase_a_risk_score": score},
    )

    entry = get_case_by_id(case_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found")
    return entry


def _parse_lab_entry(key: str, val, out: dict[str, float]) -> None:
    """Resolve a single lab entry (by itemid or name) into out[itemid] = float."""
    try:
        num = float(val)
    except (ValueError, TypeError):
        return

    # Direct itemid match
    if key in _LAB_NAMES:
        out[key] = num
        return

    # Name match (case-insensitive)
    match = _LAB_NAME_TO_ID.get(key.lower())
    if match:
        out[match] = num


# ---------------------------------------------------------------------------
# POST /api/cases/{case_id}/ecg — Save / update ECG data for a case
# ---------------------------------------------------------------------------
@app.post("/api/cases/{case_id}/ecg")
async def update_ecg_data(
    case_id: str,
    ecg_data: dict = Body(...),
):
    """
    Save or update structured ECG parameters for a case.

    Accepts a JSON body with clinical ECG measurements:
    {
      "heart_rate": 78,
      "pr_interval_ms": 160,
      "qrs_duration_ms": 88,
      "qtc_ms": 420,
      "st_deviation_mm": 0.5,
      "rhythm_interpretation": "Normal Sinus Rhythm"
    }

    The current baseline model (baseline_best.pt) is CXR-only and does not
    consume raw ECG signals.  This endpoint stores structured ECG metadata
    for clinical display and future multimodal inference.
    """
    try:
        db = get_db()
        now = datetime.now(timezone.utc).isoformat()

        # Build the ECG JSONB payload
        ecg_payload = {
            "heart_rate": float(ecg_data.get("heart_rate", 0)),
            "pr_interval_ms": float(ecg_data.get("pr_interval_ms", 0)),
            "qrs_duration_ms": float(ecg_data.get("qrs_duration_ms", 0)),
            "qtc_ms": float(ecg_data.get("qtc_ms", 0)),
            "st_deviation_mm": float(ecg_data.get("st_deviation_mm", 0)),
            "rhythm_interpretation": str(ecg_data.get("rhythm_interpretation", "Unknown")),
            "acquired_at": ecg_data.get("acquired_at", now),
        }

        if db == "LOCAL_MOCK":
            from database import _load_local_db, _save_local_db

            data = _load_local_db()
            case_row = next((c for c in data["cases"] if c["id"] == case_id), None)
            if case_row is None:
                raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found")
            case_row["ecg_data"] = ecg_payload
            case_row["updated_at"] = now
            _save_local_db(data)
        else:
            db.table("cases").update({
                "ecg_data": ecg_payload,
                "updated_at": now,
            }).eq("id", case_id).execute()

        return {"status": "ok", "case_id": case_id, "ecg_data": ecg_payload}

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[POST /ecg] Error: %s", exc, exc_info=True)
        raise HTTPException(status_code=503, detail="Database unavailable")


# ---------------------------------------------------------------------------
# POST /api/cases/{case_id}/reinfer — Re-run inference with timeout
# ---------------------------------------------------------------------------
@app.post("/api/cases/{case_id}/reinfer")
async def reinfer_case(
    case_id: str,
    background_tasks: BackgroundTasks,
    request: Request,
    target_label: str = "Pleural Effusion",
):
    """
    Re-run CXR inference for an existing case.  Returns immediately with
    status 202 and runs inference in the background.

    Use GET /api/cases/{case_id} to poll for updated predictions.
    """
    try:
        entry = get_case_by_id(case_id)
    except Exception as exc:
        logger.error("[POST /reinfer] DB error: %s", exc, exc_info=True)
        raise HTTPException(status_code=503, detail="Database unavailable")

    if entry is None:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found")

    case_obj = entry.get("case", {})
    cxr_url = case_obj.get("cxr_dicom_url")

    if not cxr_url:
        raise HTTPException(status_code=400, detail="No CXR image available for this case")

    # Resolve local file path from the CXR URL
    local_image_path: str | None = None
    if cxr_url.startswith("/mock-data/"):
        local_path = _REPO_ROOT / "frontend" / "public" / cxr_url.lstrip("/")
        if local_path.exists():
            local_image_path = str(local_path)
    elif Path(cxr_url).exists():
        local_image_path = cxr_url

    if not local_image_path:
        raise HTTPException(
            status_code=400,
            detail=f"CXR image file not found on disk for URL: {cxr_url}",
        )

    background_tasks.add_task(
        _run_inference_task,
        case_id=case_id,
        local_image_path=local_image_path,
        storage_dest="",
        target_label=target_label,
    )

    log_audit(
        "cxr.reinfer",
        **_actor(request),
        target_type="cases",
        target_id=case_id,
        metadata={"target_label": target_label},
    )

    return {
        "status": "accepted",
        "case_id": case_id,
        "message": "Inference re-queued. Poll GET /api/cases/{case_id} for results.",
    }


# ---------------------------------------------------------------------------
# POST /api/cases/{case_id}/gradcam/regenerate — Regenerate Grad-CAM heatmap
# ---------------------------------------------------------------------------
@app.post("/api/cases/{case_id}/gradcam/regenerate")
async def regenerate_gradcam(
    case_id: str,
    background_tasks: BackgroundTasks,
    request: Request,
    target_label: str = "Pleural Effusion",
):
    """
    Regenerate the Grad-CAM heatmap for an existing case.
    Semantically equivalent to reinfer — runs the full DenseNet121 forward pass
    and re-generates the heatmap for the specified target label.
    """
    try:
        entry = get_case_by_id(case_id)
    except Exception as exc:
        logger.error("[POST /gradcam/regenerate] DB error: %s", exc, exc_info=True)
        raise HTTPException(status_code=503, detail="Database unavailable")

    if entry is None:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found")

    case_obj = entry.get("case", {})
    cxr_url = case_obj.get("cxr_dicom_url")

    if not cxr_url:
        raise HTTPException(status_code=400, detail="No CXR image available for this case")

    local_image_path: str | None = None
    if cxr_url.startswith("/mock-data/"):
        local_path = _REPO_ROOT / "frontend" / "public" / cxr_url.lstrip("/")
        if local_path.exists():
            local_image_path = str(local_path)
    elif Path(cxr_url).exists():
        local_image_path = cxr_url

    if not local_image_path:
        raise HTTPException(
            status_code=400,
            detail=f"CXR image file not found on disk for URL: {cxr_url}",
        )

    background_tasks.add_task(
        _run_inference_task,
        case_id=case_id,
        local_image_path=local_image_path,
        storage_dest="",
        target_label=target_label,
    )

    log_audit(
        "gradcam.regenerate",
        **_actor(request),
        target_type="cases",
        target_id=case_id,
        metadata={"target_label": target_label},
    )

    return {
        "status": "accepted",
        "case_id": case_id,
        "message": f"Grad-CAM regeneration queued for label '{target_label}'. Poll GET /api/cases/{case_id} for results.",
    }

