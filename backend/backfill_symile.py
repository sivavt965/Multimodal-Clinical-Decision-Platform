"""
backfill_symile.py — Populate the Symile FAISS index for every demo case.

For each case in local_db.json:
  1. Map case short_id → MIMIC subject_id  (find_study_urls.ALL_CASES)
  2. Map subject_id → first test-split row (symile_mimic_data.csv +
                                            data_npy/test/hadm_id_test.npy)
  3. Run Symile inference on the real (CXR + ECG + Labs) tensors
  4. Add the resulting 24576-d vector to VectorStore.for_model("symile"),
     keyed by the case_id UUID
  5. Persist the index to disk

After running, the platform's similar-cases retrieval can hit a real multimodal
index (CXR + ECG + Labs) instead of the CXR-only DenseNet path.

Usage:
    py -V:3.14 backfill_symile.py             # encode all cases
    py -V:3.14 backfill_symile.py --limit 5   # smoke test first 5
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

_BACKEND = Path(__file__).resolve().parent
_REPO    = _BACKEND.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("backfill_symile")


def _load_hadm_to_subject() -> dict[int, int]:
    """hadm_id -> subject_id (from symile_mimic_data.csv)."""
    csv_path = _REPO / "symile_mimic_data.csv"
    out: dict[int, int] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                out[int(row["hadm_id"])] = int(row["subject_id"])
            except (KeyError, TypeError, ValueError):
                continue
    return out


def _build_subject_to_row(split: str = "test") -> dict[int, int]:
    """subject_id -> first row index in data_npy/{split}/."""
    hadm_arr = np.load(_REPO / "data_npy" / split / f"hadm_id_{split}.npy")
    hadm_to_subj = _load_hadm_to_subject()
    subj_to_row: dict[int, int] = {}
    for i, h in enumerate(hadm_arr.tolist()):
        s = hadm_to_subj.get(int(h))
        if s is not None and s not in subj_to_row:
            subj_to_row[s] = i
    return subj_to_row


def _short_id(case_id: str) -> str:
    return case_id.replace("-", "")[:8]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Encode at most N cases (smoke test)")
    parser.add_argument("--split", default="test", help="data_npy split to draw tensors from")
    parser.add_argument("--force", action="store_true", help="Re-encode cases that are already in the index")
    args = parser.parse_args()

    # ── Load case list ──────────────────────────────────────────────────────
    local_db = json.loads((_BACKEND / "local_db.json").read_text())
    cases    = local_db.get("cases", [])
    logger.info("Loaded %d cases from local_db.json", len(cases))

    # ── Build subject → row map ─────────────────────────────────────────────
    subj_to_row = _build_subject_to_row(args.split)
    logger.info("Indexed %d subjects in data_npy/%s/", len(subj_to_row), args.split)

    # ── Load case → subject mapping ────────────────────────────────────────
    from find_study_urls import ALL_CASES   # type: ignore

    # ── Heavy imports last ─────────────────────────────────────────────────
    logger.info("Loading Symile-MIMIC model (this can take ~30s on first call)...")
    t0 = time.monotonic()
    from engine.symile_encoder import run_symile_inference, get_symile_model
    get_symile_model()        # forces load, surfaces errors early
    logger.info("Symile model ready in %.1fs", time.monotonic() - t0)

    from engine.vector_store import VectorStore
    vs = VectorStore.for_model("symile")
    vs.load()
    logger.info("Symile FAISS index opened (size=%d, dim=%d)", vs.size, vs._dim)

    # ── Pre-load the data_npy tensors with mmap so we can slice cheaply ─────
    npy_dir = _REPO / "data_npy" / args.split
    cxr_arr = np.load(npy_dir / f"cxr_{args.split}.npy", mmap_mode="r")
    ecg_arr = np.load(npy_dir / f"ecg_{args.split}.npy", mmap_mode="r")
    lp_arr  = np.load(npy_dir / f"labs_percentiles_{args.split}.npy", mmap_mode="r")
    lm_arr  = np.load(npy_dir / f"labs_missingness_{args.split}.npy", mmap_mode="r")
    hadm_arr = np.load(npy_dir / f"hadm_id_{args.split}.npy")

    # ── Encode each case ────────────────────────────────────────────────────
    succeeded, skipped, missing, failed = 0, 0, 0, 0
    wall = time.monotonic()
    targets = cases[: args.limit] if args.limit else cases

    for i, case in enumerate(targets, start=1):
        case_id  = case["id"]
        short    = _short_id(case_id)
        subj_id  = ALL_CASES.get(short)

        if subj_id is None:
            logger.warning("[%3d/%d] %s — no MIMIC mapping", i, len(targets), short)
            missing += 1
            continue

        if not args.force and case_id in vs._id_map:
            logger.info("[%3d/%d] %s — already indexed, skip", i, len(targets), short)
            skipped += 1
            continue

        row = subj_to_row.get(subj_id)
        if row is None:
            logger.warning("[%3d/%d] %s — subject %d not in %s split", i, len(targets), short, subj_id, args.split)
            missing += 1
            continue

        try:
            t_case = time.monotonic()
            emb = run_symile_inference(
                cxr=cxr_arr[row],
                ecg=ecg_arr[row],
                labs_percentiles=lp_arr[row],
                labs_missingness=lm_arr[row],
                hadm_id=int(hadm_arr[row]),
            )
            vs.add_to_index(case_id=case_id, embedding=emb)
            logger.info(
                "[%3d/%d] OK %s  subj=%d row=%d  %.1fs",
                i, len(targets), short, subj_id, row, time.monotonic() - t_case,
            )
            succeeded += 1
        except Exception as exc:
            logger.error("[%3d/%d] FAIL %s — %s", i, len(targets), short, exc, exc_info=True)
            failed += 1

    if succeeded > 0:
        vs.save()
        logger.info("Symile FAISS index saved (size=%d)", vs.size)

    logger.info("=" * 60)
    logger.info("Backfill complete in %.1fs", time.monotonic() - wall)
    logger.info("  succeeded : %d", succeeded)
    logger.info("  skipped   : %d", skipped)
    logger.info("  missing   : %d (no MIMIC mapping or wrong split)", missing)
    logger.info("  failed    : %d", failed)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
