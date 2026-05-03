"""
batch_infer.py — Run DenseNet121 inference on all cases in local_db.json.

What this does per case:
  1. Resolves the local CXR image path from cxr_dicom_url
  2. Runs full DenseNet121 forward pass (14 labels + probabilities)
  3. Runs MC Dropout (10 stochastic passes → uncertainty level)
  4. Computes Grad-CAM overlay → saves PNG to frontend/public/mock-data/heatmaps/
  5. Extracts 1024-d GAP embedding → adds to FAISS index
  6. Persists all predictions + heatmap URL to local_db.json

After this script finishes:
  - Every case will have real Grad-CAM heatmaps (not null)
  - FAISS will be populated with real DenseNet embeddings (not mock/random)
  - Similarity search will return clinically meaningful results

Usage (from the backend/ directory):
    py -3 batch_infer.py
    py -3 batch_infer.py --start 50 --end 100    # process a slice
    py -3 batch_infer.py --dry-run               # validate paths only, no inference
    py -3 batch_infer.py --resume                # skip cases that already have predictions
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

# ── Make sure backend/ is on sys.path so relative imports work ───────────────
_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

_REPO_ROOT = _BACKEND_DIR.parent
_FRONTEND_PUBLIC = _REPO_ROOT / "frontend" / "public"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("batch_infer")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_image_path(cxr_url: str) -> Path | None:
    """Convert a cxr_dicom_url like /mock-data/dicoms/case_xxx.png to an absolute Path."""
    if not cxr_url:
        return None
    if cxr_url.startswith("/mock-data/"):
        p = _FRONTEND_PUBLIC / cxr_url.lstrip("/")
        return p if p.exists() else None
    p = Path(cxr_url)
    return p if p.exists() else None


def load_cases() -> list[dict]:
    db_path = _BACKEND_DIR / "local_db.json"
    with open(db_path, "r") as fh:
        db = json.load(fh)
    return db.get("cases", [])


def cases_with_predictions(db_path: Path) -> set[str]:
    """Return case IDs that already have at least one prediction row."""
    with open(db_path, "r") as fh:
        db = json.load(fh)
    return {p["case_id"] for p in db.get("predictions", [])}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Batch DenseNet121 inference for all cases")
    parser.add_argument("--start",    type=int, default=0,     help="First case index (0-based)")
    parser.add_argument("--end",      type=int, default=None,  help="Last case index (exclusive)")
    parser.add_argument("--dry-run",  action="store_true",     help="Validate image paths only — no inference")
    parser.add_argument("--resume",   action="store_true",     help="Skip cases that already have prediction rows")
    parser.add_argument("--workers",  type=int, default=1,     help="Reserved for future parallel mode (keep 1 for GPU)")
    args = parser.parse_args()

    cases = load_cases()
    total_cases = len(cases)
    logger.info("Loaded %d cases from local_db.json", total_cases)

    # Slice
    end = args.end if args.end is not None else total_cases
    cases = cases[args.start:end]
    logger.info("Processing cases [%d:%d] = %d cases", args.start, end, len(cases))

    # Resume: skip already-inferred cases
    already_done: set[str] = set()
    if args.resume:
        already_done = cases_with_predictions(_BACKEND_DIR / "local_db.json")
        skippable = sum(1 for c in cases if c["id"] in already_done)
        logger.info("--resume: %d/%d cases already have predictions → will skip", skippable, len(cases))

    if args.dry_run:
        _dry_run(cases, already_done)
        return

    # Import heavy dependencies only when actually running inference
    logger.info("Loading DenseNet121 model …")
    t0 = time.monotonic()
    from engine.inference import run_cxr_inference
    from database import update_case_inference
    from engine.vector_store import get_vector_store
    logger.info("Model loaded in %.1fs", time.monotonic() - t0)

    succeeded = 0
    failed = 0
    skipped = 0
    wall_start = time.monotonic()

    for i, case in enumerate(cases, start=1):
        case_id  = case["id"]
        cxr_url  = case.get("cxr_dicom_url", "")
        short_id = case_id[:8]

        # ── Resume check ────────────────────────────────────────────────
        if args.resume and case_id in already_done:
            logger.info("[%3d/%d]  SKIP  %s (already inferred)", i, len(cases), short_id)
            skipped += 1
            continue

        # ── Resolve image path ───────────────────────────────────────────
        img_path = resolve_image_path(cxr_url)
        if img_path is None:
            logger.warning("[%3d/%d]  MISS  %s — image not found: %s", i, len(cases), short_id, cxr_url)
            failed += 1
            continue

        # ── Run inference ────────────────────────────────────────────────
        logger.info("[%3d/%d]  RUN   %s  (%s)", i, len(cases), short_id, img_path.name)
        t_case = time.monotonic()

        try:
            result = run_cxr_inference(
                image_path=str(img_path),
                case_id=case_id,
            )
        except Exception as exc:
            logger.error("[%3d/%d]  ERR   %s — inference crashed: %s", i, len(cases), short_id, exc)
            failed += 1
            continue

        if result.error:
            logger.warning("[%3d/%d]  ERR   %s — %s", i, len(cases), short_id, result.error)
            failed += 1
            continue

        # ── Persist to local_db.json ─────────────────────────────────────
        try:
            update_case_inference(
                case_id=case_id,
                predictions=result.predictions,
                heatmap_url=result.heatmap_url,
                heatmap_label=result.heatmap_label,
            )
        except Exception as exc:
            logger.error("[%3d/%d]  ERR   %s — DB write failed: %s", i, len(cases), short_id, exc)
            failed += 1
            continue

        elapsed = time.monotonic() - t_case
        top_label = max(result.probabilities, key=result.probabilities.get) if result.probabilities else "?"
        top_prob  = result.probabilities.get(top_label, 0.0)
        logger.info(
            "[%3d/%d]  OK    %s  top=%s (%.0f%%)  uncertainty=%s  heatmap=%s  %.1fs",
            i, len(cases), short_id,
            top_label, top_prob * 100,
            result.mc_uncertainty_level or "N/A",
            "✓" if result.heatmap_url else "✗",
            elapsed,
        )
        succeeded += 1

        # ETA
        elapsed_total = time.monotonic() - wall_start
        rate = succeeded / elapsed_total if elapsed_total > 0 else 0
        remaining = len(cases) - i
        eta_sec = remaining / rate if rate > 0 else float("inf")
        if eta_sec < float("inf"):
            eta_min = eta_sec / 60
            logger.info("       ETA: %.0f min remaining (%d left, %.2f cases/min)", eta_min, remaining, rate * 60)

    # ── Save FAISS index ─────────────────────────────────────────────────────
    if succeeded > 0:
        try:
            vs = get_vector_store()
            vs.save()
            logger.info("FAISS index saved (%d vectors total)", vs.size)
        except Exception as exc:
            logger.error("FAISS save failed: %s", exc)

    # ── Summary ──────────────────────────────────────────────────────────────
    total_wall = time.monotonic() - wall_start
    logger.info("")
    logger.info("=" * 60)
    logger.info("Batch inference complete in %.1fs", total_wall)
    logger.info("  Succeeded : %d", succeeded)
    logger.info("  Skipped   : %d", skipped)
    logger.info("  Failed    : %d", failed)
    logger.info("  Total     : %d", len(cases))
    logger.info("=" * 60)
    if failed > 0:
        logger.warning("Some cases failed. Re-run with --resume to retry only failures.")


def _dry_run(cases: list[dict], already_done: set[str]) -> None:
    """Print path resolution results without running inference."""
    missing = []
    found = []
    for case in cases:
        case_id = case["id"]
        cxr_url = case.get("cxr_dicom_url", "")
        img_path = resolve_image_path(cxr_url)
        status = "FOUND" if img_path else "MISSING"
        done   = " [already inferred]" if case_id in already_done else ""
        line   = f"  {status}  {case_id[:8]}  {cxr_url}{done}"
        (found if img_path else missing).append(line)

    for l in found:
        print(l)
    if missing:
        print(f"\n--- MISSING ({len(missing)}) ---")
        for l in missing:
            print(l)
    print(f"\nSummary: {len(found)} found, {len(missing)} missing out of {len(cases)} cases")


if __name__ == "__main__":
    main()
