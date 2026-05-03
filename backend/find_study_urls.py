"""
find_study_urls.py
Reads mimic-cxr-2.0.0-metadata.csv.gz and prints exact download URLs
for all cases that need real MIMIC-CXR images.
"""

import gzip
import csv
import json
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent
_META_GZ = _BACKEND / "mimic-cxr-2.0.0-metadata.csv.gz"

# All 40 selected cases: short_id → subject_id
# 6 already have real images (marked), 34 need download
ALL_CASES = {
    "f1e3d197": 13472341,   # already downloaded
    "ad58402a": 14726060,   # already downloaded
    "2f54d7f5": 11536384,   # already downloaded
    "813d610a": 14076508,   # already downloaded
    "b5113ad5": 18264374,   # already downloaded
    "238ecb55": 13055847,   # already downloaded
    "3b1950dc": 11952902,
    "2d15ff35": 15880947,
    "38186792": 18830363,
    "f7148544": 11474034,
    "e427a8fb": 19722227,
    "8343b48c": 18208434,
    "ec4ee9ea": 13505755,
    "60b43b71": 13160874,
    "f0d86a7b": 11496131,
    "d71eef46": 12692062,
    "4abed709": 14188888,
    "a67377ff": 10976602,
    "9c6b22f7": 10216556,
    "36fcd761": 18589881,
    "7d8ce05a": 13958446,
    "dbb0f050": 12514563,
    "7e31a8b0": 17901320,
    "bfbd4450": 13045537,
    "27c45cfd": 15906743,
    "e4300324": 15557817,
    "df7bea55": 19850525,
    "4026462e": 10817445,
    "07afb995": 13505755,
    "e96d0a66": 11531307,
    "aa25e1b2": 17562503,
    "88008c93": 16472049,
    "7eb0348b": 11054411,
    "f3597bf0": 17388366,
    "6c247d24": 10119514,
    "7e5f562d": 19471350,
    "1ad15ac4": 14726060,
    "1845969f": 11496131,
    "d9dfe7a3": 10594290,
    "7037c41f": 14443991,
}

ALREADY_DOWNLOADED = {"f1e3d197", "ad58402a", "2f54d7f5", "813d610a", "b5113ad5", "238ecb55"}

BASE_URL = "https://physionet.org/files/mimic-cxr-jpg/2.1.0/files"

def main():
    if not _META_GZ.exists():
        print(f"ERROR: metadata file not found at {_META_GZ}")
        print("Download mimic-cxr-2.0.0-metadata.csv.gz from PhysioNet first.")
        return

    # subject_id → list of {study_id, dicom_id, view_position}
    needed_subjects = set(ALL_CASES.values())
    by_subject: dict[int, list] = {v: [] for v in needed_subjects}

    print("Reading metadata (this takes ~30s)...")
    with gzip.open(_META_GZ, "rt") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            subj = int(row.get("subject_id", 0))
            if subj in by_subject:
                by_subject[subj].append({
                    "study_id":      row.get("study_id", ""),
                    "dicom_id":      row.get("dicom_id", ""),
                    "view_position": row.get("ViewPosition", ""),
                })

    dest_dir = _BACKEND.parent / "frontend" / "public" / "mock-data" / "dicoms"

    print()
    print("=" * 80)
    print("DOWNLOAD THESE 34 IMAGES MANUALLY FROM YOUR BROWSER")
    print("Log into PhysioNet first: https://physionet.org/login/")
    print("=" * 80)
    print()

    missing = []
    count = 0
    for short_id, subject_id in ALL_CASES.items():
        if short_id in ALREADY_DOWNLOADED:
            continue

        records = by_subject.get(subject_id, [])
        if not records:
            missing.append(f"  WARNING: No studies found for subject {subject_id} (case {short_id})")
            continue

        pa_records = [r for r in records if r["view_position"] == "PA"]
        chosen = pa_records[0] if pa_records else records[0]

        study_id = chosen["study_id"]
        dicom_id = chosen["dicom_id"]
        subj_str = str(subject_id)
        subj_prefix = subj_str[:2]

        url = f"{BASE_URL}/p{subj_prefix}/p{subj_str}/s{study_id}/{dicom_id}.jpg"
        out_file = dest_dir / f"case_{short_id}.png"
        count += 1

        print(f"[{count:2d}/34] Save as: case_{short_id}.png")
        print(f"       URL: {url}")
        print()

    if missing:
        print("--- MISSING SUBJECTS ---")
        for m in missing:
            print(m)

    print("=" * 80)
    print(f"Save all files to:")
    print(f"  {dest_dir}")
    print()
    print("After downloading all 34, come back and run:")
    print("  py -3.14 trim_to_40.py")
    print("to delete the 110 unused cases and run inference on the 40 real ones.")
    print("=" * 80)


if __name__ == "__main__":
    main()
