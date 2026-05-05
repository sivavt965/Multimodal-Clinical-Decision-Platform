# =============================================================================
# engine/ground_truth.py
# Ground-truth CheXpert labels for the demo cases, looked up from the bundled
# Symile-MIMIC CSV (symile_mimic_data.csv).
# =============================================================================
"""
Why this exists
---------------
The case rows in the platform are synthetic UUIDs that don't carry MIMIC
identifiers. find_study_urls.ALL_CASES holds the curated mapping
  case short_id (first 8 chars of UUID) → MIMIC subject_id
that was used to seed the demo dataset.

For each case we look the subject_id up in symile_mimic_data.csv and return
the list of CheXpert labels marked positive (value == 1.0) — i.e. the
expert-annotated ground truth, NOT the model's predictions. This is what the
similar-cases card should display, since real CXRs typically carry several
simultaneous findings and showing only argmax is clinically misleading.

Caching
-------
The CSV is ~12 k rows × ~100 cols. We load only the 14 label columns plus
subject_id once at first call and keep an in-memory  subject_id → [labels]
dict thereafter. If the CSV is missing the helper returns an empty list
silently — the rest of the app keeps working.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# All 14 CheXpert label columns present in symile_mimic_data.csv.
_CHEXPERT_LABELS: List[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT   = _BACKEND_DIR.parent
_CSV_PATH    = _REPO_ROOT / "symile_mimic_data.csv"

# subject_id → [positive labels]. Built lazily on first call.
_subject_to_labels: Optional[Dict[int, List[str]]] = None
_load_lock = threading.Lock()


def _load_csv_index() -> Dict[int, List[str]]:
    """
    Load and index the CSV with stdlib csv (no pandas dependency).
    Returns {} if the file is missing or unreadable.
    """
    if not _CSV_PATH.exists():
        logger.warning("[ground_truth] CSV not found at %s — returning empty mapping", _CSV_PATH)
        return {}
    try:
        import csv
        index: Dict[int, List[str]] = {}
        with open(_CSV_PATH, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    subj = int(row["subject_id"])
                except (KeyError, TypeError, ValueError):
                    continue
                # Some subjects appear multiple times (multiple admissions);
                # keep the first row's labels.
                if subj in index:
                    continue
                positives: List[str] = []
                for lbl in _CHEXPERT_LABELS:
                    raw = row.get(lbl, "")
                    if raw == "" or raw is None:
                        continue
                    try:
                        if float(raw) == 1.0:
                            positives.append(lbl)
                    except (TypeError, ValueError):
                        continue
                index[subj] = positives
    except Exception as exc:
        logger.error("[ground_truth] Could not read %s: %s", _CSV_PATH, exc)
        return {}

    logger.info("[ground_truth] Indexed %d subjects from %s", len(index), _CSV_PATH.name)
    return index


def _ensure_loaded() -> Dict[int, List[str]]:
    global _subject_to_labels
    if _subject_to_labels is None:
        with _load_lock:
            if _subject_to_labels is None:
                _subject_to_labels = _load_csv_index()
    return _subject_to_labels


def _short_id(case_id: str) -> str:
    """First 8 hex chars of a UUID, matching find_study_urls.ALL_CASES keys."""
    return case_id.replace("-", "")[:8]


def get_ground_truth_findings(case_id: str) -> List[str]:
    """
    Return the list of CheXpert labels marked positive for the MIMIC subject
    that this demo case maps to. Returns [] if the case doesn't have a mapping
    or the subject isn't in the CSV.

    Filters out 'No Finding' so the UI only shows actionable findings.
    """
    try:
        from find_study_urls import ALL_CASES  # type: ignore
    except Exception as exc:
        logger.warning("[ground_truth] find_study_urls import failed: %s", exc)
        return []

    subj = ALL_CASES.get(_short_id(case_id))
    if subj is None:
        return []
    labels = _ensure_loaded().get(subj, [])
    return [l for l in labels if l != "No Finding"]
