# src/data/preprocess.py

import os
import pandas as pd

# 8 target labels we care about
TARGET_LABELS = [
    "Cardiomegaly",
    "Pleural Effusion",
    "Edema",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Consolidation",
    "Support Devices",
]

# frontal views only
FRONTAL_VIEWS = {"PA", "AP"}

# official MIMIC-CXR-JPG GCS bucket for JPEGs
IMAGE_GCS_BASE = "gs://mimic-cxr-jpg-2.1.0.physionet.org"


def main():
    # project root = folder that contains src/, data/, etc.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    raw_dir = os.path.join(project_root, "data", "raw")
    out_path = os.path.join(project_root, "data", "processed", "processed_metadata.csv")

    print("[INFO] Project root :", project_root)
    print("[INFO] Raw dir      :", raw_dir)
    print("[INFO] Output CSV   :", out_path)

    # paths to gz files
    meta_path = os.path.join(raw_dir, "mimic-cxr-2.0.0-metadata.csv.gz")
    chex_path = os.path.join(raw_dir, "mimic-cxr-2.0.0-chexpert.csv.gz")
    split_path = os.path.join(raw_dir, "mimic-cxr-2.0.0-split.csv.gz")

    # basic existence check
    for p in [meta_path, chex_path, split_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")

    print("[INFO] Reading metadata    :", meta_path)
    print("[INFO] Reading CheXpert    :", chex_path)
    print("[INFO] Reading split (gz)  :", split_path)

    # pandas can read gz directly
    meta = pd.read_csv(meta_path, compression="gzip")
    chex = pd.read_csv(chex_path, compression="gzip")
    split = pd.read_csv(split_path, compression="gzip")

    print(
        "[INFO] Loaded rows:",
        "metadata =", len(meta),
        "| chexpert =", len(chex),
        "| split =", len(split),
    )

    # sanity checks
    if not {"subject_id", "study_id", "dicom_id"}.issubset(meta.columns):
        raise ValueError("metadata must contain subject_id, study_id, dicom_id")

    if not {"subject_id", "study_id"}.issubset(chex.columns):
        raise ValueError("chexpert must contain subject_id, study_id")

    if not {"dicom_id", "split"}.issubset(split.columns):
        raise ValueError("split must contain dicom_id and split columns")

    # merge: metadata (image-level) + chexpert (study-level per study_id)
    df = meta.merge(
        chex,
        on=["subject_id", "study_id"],
        how="left",
        validate="many_to_one",  # many images per study, one chexpert row per study
    )

    # merge split info PER IMAGE via dicom_id
    df = df.merge(
        split[["dicom_id", "split"]],
        on="dicom_id",
        how="left",
        validate="one_to_one",  # each dicom_id should have exactly one split row
    )

    print("[INFO] After merge rows:", len(df))

    # filter frontal (PA/AP)
    if "ViewPosition" not in df.columns:
        raise ValueError("Expected 'ViewPosition' column in metadata.")
    before = len(df)
    df = df[df["ViewPosition"].isin(FRONTAL_VIEWS)].copy()
    after = len(df)
    print(f"[INFO] Filtered to frontal views: {before} -> {after} rows")

    # clean labels: NaN -> 0, -1 -> 0, clip to [0,1]
    missing_labels = [c for c in TARGET_LABELS if c not in df.columns]
    if missing_labels:
        raise ValueError(f"Missing label columns: {missing_labels}")

    labels = df[TARGET_LABELS].copy()
    labels = labels.fillna(0.0)
    labels = labels.replace(-1.0, 0.0)
    labels = labels.astype(float).clip(0.0, 1.0)
    df[TARGET_LABELS] = labels

    print("[INFO] Label positives (sum over dataset):")
    print(df[TARGET_LABELS].sum())

      # build gcs_path to JPEG in MIMIC-CXR-JPG bucket
    # MIMIC-CXR-JPG layout is:
    # files/p{first2digits_of_subject_id}/p{subject_id}/s{study_id}/{dicom_id}.jpg
    #
    # Example:
    #   subject_id = 10000032 -> p10/p10000032
    #   study_id   = 50414267 -> s50414267
    #   dicom_id   = 4a0397d2-... -> 4a0397d2-....jpg
    #
    # Final path:
    #   gs://mimic-cxr-jpg-2.1.0.physionet.org/files/p10/p10000032/s50414267/4a0397d2-....jpg

    subj_str = df["subject_id"].astype(str)
    study_str = df["study_id"].astype(str)
    dicom_str = df["dicom_id"].astype(str)

    df["gcs_path"] = (
        IMAGE_GCS_BASE.rstrip("/")
        + "/files/p"
        + subj_str.str.slice(0, 2)
        + "/p"
        + subj_str
        + "/s"
        + study_str
        + "/"
        + dicom_str
        + ".jpg"
    )

    # drop rows without split or IDs
    before = len(df)
    df = df.dropna(subset=["split", "subject_id", "study_id", "dicom_id"])
    after = len(df)
    print(f"[INFO] Dropped unusable rows: {before} -> {after}")

    # sort and save
    df = df.sort_values(["subject_id", "study_id", "dicom_id"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

    print("[INFO] Saved processed CSV to:", out_path)
    print("[INFO] Final row count:", len(df))
    print("[INFO] Split counts:")
    print(df["split"].value_counts())


if __name__ == "__main__":
    main()
