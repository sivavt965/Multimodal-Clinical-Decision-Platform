#!/usr/bin/env python3
"""
create_demo_dataset.py
======================
Builds a clean 150-case demo dataset for the dashboard.

Steps
-----
1.  Resets local_db.json to empty (Task 1 — clean dashboard)
2.  Loads test.csv + symile_mimic_data.csv (demographics)
3.  Selects 150 balanced cases across 4 common labels
4.  De-normalises + saves CXR images as PNG (320×320)
5.  Generates realistic ECG JSON per case
6.  Extracts real lab values from test.csv for each case
7.  Computes Phase A risk via backend heuristic
8.  Synthesises per-label predictions from known CheXpert labels
9.  Runs DenseNet121 batch GAP extraction → persists FAISS index
10. Writes all records to local_db.json

Run from project root:
    python create_demo_dataset.py
"""

import json
import os
import random
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent
BACKEND      = ROOT / "backend"
FRONTEND     = ROOT / "frontend"
DATA_NPY     = ROOT / "data_npy" / "test"
DICOMS_DIR   = FRONTEND / "public" / "mock-data" / "dicoms"
LOCAL_DB     = BACKEND / "local_db.json"
FAISS_IDX    = BACKEND / "faiss_index.bin"
FAISS_MAP    = BACKEND / "faiss_id_map.npy"

sys.path.insert(0, str(BACKEND))
DICOMS_DIR.mkdir(parents=True, exist_ok=True)

# ── Label constants ────────────────────────────────────────────────────────────
# Labels shared between Symile test.csv and DenseNet121 (8-class model)
COMMON_LABELS = ["Atelectasis", "Cardiomegaly", "Edema", "Pleural Effusion"]

# DenseNet121 model output labels (8 classes)
MODEL_LABELS = [
    "Cardiomegaly", "Pleural Effusion", "Edema", "Pneumonia",
    "Atelectasis", "Pneumothorax", "Consolidation", "Support Devices",
]

# Full 14-class CheXpert taxonomy used in the predictions schema
CHEXPERT_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices",
]

# ImageNet normalisation constants (used when the NPY arrays were created)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

# Lab itemid → LabData schema field
LAB_FIELD_MAP = {
    "50947": "troponin_ng_ml",
    "51006": "bnp_pg_ml",
    "51301": "wbc_count",
    "50912": "creatinine_mg_dl",
    "50983": "sodium_meq_l",
    "50971": "potassium_meq_l",
    "50813": "lactate_mmol_l",
}

# All 50 MIMIC-IV lab itemids (same order as labs_percentiles_*.npy columns)
ALL_LAB_IDS = [
    "51221","51265","50912","50971","51222","51301","51249","51279","51250","51248",
    "51277","51006","50983","50902","50882","50868","50931","50960","50893","50970",
    "51237","51274","51275","51146","51256","51254","51200","51244","52172","50934",
    "51678","50947","50861","50878","50813","50863","50885","50820","50862","50802",
    "50821","50804","50818","52075","52073","52074","52069","51133","50910","52135",
]

# Background prevalence for labels NOT in test.csv (from class_distribution.json test set)
_BACKGROUND_PROBS = {
    "Pneumonia":       0.05,
    "Pneumothorax":    0.03,
    "Consolidation":   0.06,
    "Support Devices": 0.36,
}

FIRST_NAMES_F = [
    "Sarah","Emily","Jennifer","Jessica","Amanda","Ashley","Rebecca","Laura",
    "Nicole","Stephanie","Patricia","Linda","Barbara","Susan","Karen","Nancy",
    "Betty","Helen","Sandra","Donna","Carol","Ruth","Sharon","Michelle","Dorothy",
    "Lisa","Maria","Anna","Margaret","Elizabeth","Catherine","Christine","Janet",
    "Frances","Virginia","Mary","Alice","Joan","Judith","Rose","Diana","Evelyn",
]
FIRST_NAMES_M = [
    "James","John","Robert","Michael","William","David","Richard","Joseph",
    "Thomas","Charles","Christopher","Daniel","Matthew","Anthony","Donald",
    "Mark","Paul","Steven","Andrew","Kenneth","George","Joshua","Kevin",
    "Brian","Edward","Ronald","Timothy","Jason","Jeffrey","Ryan","Gary",
    "Larry","Jeffrey","Frank","Scott","Eric","Stephen","Raymond","Gregory",
    "Harold","Dennis","Jerry","Tyler","Aaron","Jose","Henry","Douglas","Adam",
]
LAST_NAMES = [
    "Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis",
    "Rodriguez","Martinez","Hernandez","Lopez","Gonzalez","Wilson","Anderson",
    "Thomas","Taylor","Moore","Jackson","Martin","Lee","Perez","Thompson",
    "White","Harris","Sanchez","Clark","Ramirez","Lewis","Robinson","Walker",
    "Young","Hall","Allen","King","Wright","Scott","Torres","Nguyen","Hill",
    "Flores","Green","Adams","Nelson","Baker","Mitchell","Carter","Roberts",
]

RHYTHM_LABELS = [
    "Normal Sinus Rhythm", "Normal Sinus Rhythm", "Normal Sinus Rhythm",
    "Sinus Tachycardia", "Sinus Bradycardia",
    "Atrial Fibrillation", "First-degree AV Block",
    "Left Bundle Branch Block", "Sinus Rhythm with PVCs",
    "Sinus Rhythm with Frequent PACs",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_offset_days(days: int) -> str:
    dt = datetime.now(timezone.utc) - timedelta(days=days)
    return dt.isoformat()


def _risk_badge(prob: float) -> str:
    if prob >= 0.15:
        return "Elevated Risk"
    if prob >= 0.05:
        return "Monitor"
    return "Unlikely"


def denorm_to_pil(img_chw: np.ndarray) -> Image.Image:
    """Convert ImageNet-normalised (3,H,W) float32 → RGB PIL image."""
    img = img_chw * _STD + _MEAN
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img.transpose(1, 2, 0))  # CHW → HWC


def build_labs_raw(row: pd.Series) -> dict:
    """Extract non-NaN raw lab values from a test.csv row keyed by itemid."""
    labs = {}
    for iid in ALL_LAB_IDS:
        val = row.get(iid)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            labs[iid] = float(val)
    return labs


def build_lab_data(labs_raw: dict, collected_at: str) -> dict:
    """Map 7 key labs → LabData schema fields; default to 0 if missing."""
    def _g(iid):
        return float(labs_raw.get(iid, 0.0))
    return {
        "troponin_ng_ml":   _g("50947"),
        "bnp_pg_ml":        _g("51006"),
        "wbc_count":        _g("51301"),
        "creatinine_mg_dl": _g("50912"),
        "sodium_meq_l":     _g("50983"),
        "potassium_meq_l":  _g("50971"),
        "lactate_mmol_l":   _g("50813"),
        "collected_at":     collected_at,
    }


def compute_phase_a(lab_data: dict) -> tuple[str, float]:
    """Exact same heuristic used in main.py POST /api/cases."""
    score = 0.0
    if lab_data["troponin_ng_ml"] > 0.04:
        score += 0.4
    if lab_data["creatinine_mg_dl"] > 1.3:
        score += 0.2
    if lab_data["lactate_mmol_l"] > 2.0:
        score += 0.2
    k = lab_data["potassium_meq_l"]
    if k > 5.0 or (0 < k < 3.5):
        score += 0.1
    na = lab_data["sodium_meq_l"]
    if na > 0 and (na < 136 or na > 145):
        score += 0.1
    if score >= 0.4:
        return "High", min(score, 1.0)
    if score >= 0.2:
        return "Moderate", min(score, 1.0)
    return "Low", min(score, 1.0)


def build_ecg(labels: dict, acquired_at: str) -> dict:
    """Synthesise plausible ECG values conditioned on CXR findings."""
    rng = random.Random()

    # Base heart rate — higher for cardiac/respiratory diagnoses
    hr_mean = 75
    if labels.get("Edema") == 1.0 or labels.get("Cardiomegaly") == 1.0:
        hr_mean = 90
    if labels.get("Pneumonia") == 1.0:
        hr_mean = 95

    hr  = max(45, min(130, rng.gauss(hr_mean, 12)))
    pr  = max(110, min(280, rng.gauss(162, 22)))
    qrs = max(60,  min(160, rng.gauss(92,  14)))
    qtc = max(340, min(520, rng.gauss(425, 28)))

    # ST elevation: slightly positive for cardiomegaly/edema, near zero otherwise
    st_base = 0.1 if labels.get("Cardiomegaly") == 1.0 else 0.0
    st  = max(-2.5, min(3.0, rng.gauss(st_base, 0.35)))

    rhythm = rng.choice(RHYTHM_LABELS)
    if hr > 100:
        rhythm = "Sinus Tachycardia"
    elif hr < 55:
        rhythm = "Sinus Bradycardia"

    return {
        "heart_rate":           round(hr, 1),
        "pr_interval_ms":       round(pr, 1),
        "qrs_duration_ms":      round(qrs, 1),
        "qtc_ms":               round(qtc, 1),
        "st_deviation_mm":      round(st, 2),
        "rhythm_interpretation": rhythm,
        "acquired_at":          acquired_at,
    }


def synthesise_predictions(
    row: pd.Series,
    case_id: str,
    inference_run_at: str,
) -> list:
    """
    Build 14-row prediction list from known CheXpert labels in test.csv.

    Mapping:
      label == 1.0  (positive)   → probability in [0.70, 0.92]
      label == 0.0  (negative)   → probability in [0.02, 0.07]
      label == -1.0 (uncertain)  → probability in [0.06, 0.22]
      NaN / missing              → probability in [0.04, 0.14]
      label not in model         → probability = 0.0
    """
    rng = random.Random(hash(case_id) & 0xFFFFFFFF)

    # Map test.csv column names to CheXpert taxonomy
    csv_label_map = {
        "Atelectasis":    row.get("Atelectasis"),
        "Cardiomegaly":   row.get("Cardiomegaly"),
        "Edema":          row.get("Edema"),
        "Pleural Effusion": row.get("Pleural Effusion"),
        "No Finding":     row.get("No Finding"),
        "Lung Opacity":   row.get("Lung Opacity"),
    }

    preds = []
    for label in CHEXPERT_LABELS:
        if label not in MODEL_LABELS:
            # Not modelled by DenseNet121 — always 0
            preds.append({
                "id":               str(uuid.uuid4()),
                "case_id":          case_id,
                "model_checkpoint": "baseline_best.pt",
                "temperature":      1.2518,
                "inference_run_at": inference_run_at,
                "label":            label,
                "probability":      0.0,
                "risk_badge":       "Unlikely",
                "uncertainty_level": None,
                "mean_variance":    None,
                "std_dev":          None,
                "mc_passes":        0,
                "gradcam_url":      None,
                "gradcam_alpha":    0.45,
            })
            continue

        csv_val = csv_label_map.get(label)

        if label in _BACKGROUND_PROBS:
            # Labels not in test.csv — use background prevalence + noise
            base = _BACKGROUND_PROBS[label]
            prob = max(0.01, min(0.99, rng.gauss(base, base * 0.4)))
        elif csv_val == 1.0:
            prob = rng.uniform(0.70, 0.92)
        elif csv_val == 0.0:
            prob = rng.uniform(0.02, 0.07)
        elif csv_val == -1.0:
            prob = rng.uniform(0.06, 0.22)
        else:
            # NaN / not labelled
            prob = rng.uniform(0.04, 0.14)

        mc_std = rng.uniform(0.01, 0.06)
        mc_var = mc_std ** 2

        preds.append({
            "id":               str(uuid.uuid4()),
            "case_id":          case_id,
            "model_checkpoint": "baseline_best.pt",
            "temperature":      1.2518,
            "inference_run_at": inference_run_at,
            "label":            label,
            "probability":      round(prob, 4),
            "risk_badge":       _risk_badge(prob),
            "uncertainty_level": (
                "Low Uncertainty" if mc_var < 0.005
                else "Moderate Uncertainty" if mc_var < 0.02
                else "High Uncertainty"
            ),
            "mean_variance":    round(mc_var, 6),
            "std_dev":          round(mc_std, 4),
            "mc_passes":        10,
            "gradcam_url":      None,
            "gradcam_alpha":    0.45,
        })

    return preds


# ── FAISS batch embedding ─────────────────────────────────────────────────────

def extract_all_embeddings(image_paths: list[Path]) -> dict[str, np.ndarray]:
    """
    Load DenseNet121 once, run one forward pass per image, extract 1024-d GAP.
    Returns {case_id: (1, 1024) float32}.
    """
    import torch
    import torch.nn.functional as F
    from torchvision import transforms

    print("  Loading DenseNet121 model …")
    from engine.model_loader import get_model
    loader = get_model()
    model  = loader.model
    device = loader.device
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    class _GAPHook:
        def __init__(self):
            self.vec = None
            self._h = model.features.denseblock4.register_forward_hook(self._cap)
        def _cap(self, _m, _i, out):
            act = F.relu(out, inplace=False)
            self.vec = F.adaptive_avg_pool2d(act, (1, 1)).squeeze(-1).squeeze(-1).detach()
        def remove(self):
            self._h.remove()

    from sklearn.preprocessing import normalize as sk_norm

    results = {}
    n = len(image_paths)
    for i, (case_id, img_path) in enumerate(image_paths):
        if (i + 1) % 25 == 0:
            print(f"  Embedding {i+1}/{n} …")
        try:
            img = Image.open(img_path).convert("RGB")
            t   = preprocess(img).unsqueeze(0).to(device)
            hook = _GAPHook()
            with torch.no_grad():
                _ = model(t)
            hook.remove()
            vec = hook.vec.cpu().numpy()          # (1, 1024)
            vec = sk_norm(vec, norm="l2")
            results[case_id] = vec
        except Exception as exc:
            print(f"  [WARN] Embedding failed for {case_id}: {exc}")

    return results


def save_faiss_index(embeddings: dict[str, np.ndarray]) -> None:
    import faiss

    dim   = 1024
    index = faiss.IndexFlatL2(dim)
    id_map = []

    for case_id, vec in embeddings.items():
        index.add(vec.astype(np.float32))
        id_map.append(case_id)

    faiss.write_index(index, str(FAISS_IDX))
    np.save(str(FAISS_MAP), np.array(id_map))
    print(f"  FAISS index saved: {len(id_map)} vectors -> {FAISS_IDX.name}")


# ── Case selection ────────────────────────────────────────────────────────────

def select_150_cases(df_test: pd.DataFrame) -> pd.DataFrame:
    """
    Return 150 rows from test.csv with balanced coverage of COMMON_LABELS.
    Strategy:
      - 38 rows where Atelectasis = 1.0
      - 38 rows where Cardiomegaly = 1.0 (no overlap with Atelectasis batch)
      - 37 rows where Edema = 1.0
      - 37 rows where Pleural Effusion = 1.0 (remaining)
    Total = 150 unique rows.
    """
    rng = random.Random(42)
    chosen_idx = set()
    target_per_label = {"Atelectasis": 38, "Cardiomegaly": 38, "Edema": 37, "Pleural Effusion": 37}

    for label, n in target_per_label.items():
        pool = df_test[df_test[label] == 1.0].index.tolist()
        pool = [i for i in pool if i not in chosen_idx]
        rng.shuffle(pool)
        chosen_idx.update(pool[:n])

    # If we're short (shouldn't happen), fill from remaining clean-label cases
    if len(chosen_idx) < 150:
        all_clean = df_test[
            df_test[COMMON_LABELS].isin([0.0, 1.0]).any(axis=1)
        ].index.tolist()
        extra = [i for i in all_clean if i not in chosen_idx]
        rng.shuffle(extra)
        chosen_idx.update(extra[:150 - len(chosen_idx)])

    selected = df_test.loc[sorted(chosen_idx)].reset_index(drop=False)
    selected = selected.rename(columns={"index": "npy_row"})
    print(f"  Selected {len(selected)} cases.")
    return selected


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    random.seed(42)

    # ── Task 1: Reset local_db.json ───────────────────────────────────────────
    print("\n[1/5] Resetting local_db.json …")
    empty_db = {
        "patients": [],
        "cases": [],
        "predictions": [],
        "consultations": [],
        "lab_results": [],
        "ecg_records": [],
    }
    with open(LOCAL_DB, "w") as f:
        json.dump(empty_db, f, indent=2)

    # Remove stale FAISS index so it gets rebuilt fresh
    for p in [FAISS_IDX, FAISS_MAP]:
        if p.exists():
            p.unlink()

    print("  Done — dashboard is empty.")

    # ── Task 2: Load data ────────────────────────────────────────────────────
    print("\n[2/5] Loading data …")

    df_test = pd.read_csv(ROOT / "test.csv")
    df_demo = pd.read_csv(
        ROOT / "symile_mimic_data.csv",
        usecols=["subject_id", "hadm_id", "gender", "anchor_age", "admittime"],
    )
    # Join demographics by hadm_id
    df_test = df_test.merge(
        df_demo.drop_duplicates("hadm_id"),
        on="hadm_id",
        how="left",
        suffixes=("", "_demo"),
    )

    # Memory-map cxr_test.npy to avoid loading 5 GB into RAM
    cxr_arr = np.load(str(DATA_NPY / "cxr_test.npy"), mmap_mode="r")
    print(f"  test.csv: {len(df_test)} rows | CXR array: {cxr_arr.shape}")

    # ── Task 3: Select 150 balanced cases ────────────────────────────────────
    print("\n[3/5] Selecting 150 balanced cases …")
    selected = select_150_cases(df_test)

    print("  Label distribution among selected cases:")
    for lbl in COMMON_LABELS:
        pos = (selected[lbl] == 1.0).sum()
        print(f"    {lbl:20s}: {pos} positive")

    # ── Task 4: Build DB records + save images ────────────────────────────────
    print("\n[4/5] Building records and saving images …")

    patients    = []
    cases       = []
    predictions = []
    saved_img_paths = []   # list of (case_id, Path) for FAISS

    # Spread admitted_at over the last 6 months
    base_date = datetime.now(timezone.utc)

    for i, row in selected.iterrows():
        case_id    = str(uuid.uuid4())
        patient_id = str(uuid.uuid4())
        npy_row    = int(row["npy_row"])

        # ── Demographics ──────────────────────────────────────────────────────
        gender = str(row.get("gender", "")).upper()
        if gender == "F":
            first = random.choice(FIRST_NAMES_F)
        else:
            first = random.choice(FIRST_NAMES_M)
            gender = "M"
        last   = random.choice(LAST_NAMES)
        age    = int(row["anchor_age"]) if not pd.isna(row.get("anchor_age")) else random.randint(52, 85)
        mrn    = f"MRN-{random.randint(10000, 99999)}"
        sex    = "F" if gender == "F" else "M"

        # Birth year derived from anchor_age
        dob_year = base_date.year - age
        dob = f"{dob_year}-{random.randint(1,12):02d}-{random.randint(1,28):02d}"

        # Spread admissions across the past 6 months
        days_ago = int((150 - i) * 1.2) + random.randint(0, 5)
        admitted_at = (base_date - timedelta(days=days_ago)).isoformat()
        now_str     = _utcnow()

        # ── Save CXR image ─────────────────────────────────────────────────────
        img_chw = cxr_arr[npy_row]                 # (3, 320, 320) float32, ImageNet-normed
        pil_img = denorm_to_pil(img_chw)            # RGB PIL 320×320
        img_filename = f"case_{case_id[:8]}.png"
        img_path = DICOMS_DIR / img_filename
        pil_img.save(str(img_path))
        cxr_url = f"/mock-data/dicoms/{img_filename}"

        saved_img_paths.append((case_id, img_path))

        # ── Labs ──────────────────────────────────────────────────────────────
        labs_raw  = build_labs_raw(row)
        lab_data  = build_lab_data(labs_raw, admitted_at)

        # ── Phase A ───────────────────────────────────────────────────────────
        risk_level, risk_score = compute_phase_a(lab_data)
        if risk_level == "High":
            reco = "Elevated cardiac and metabolic markers. Recommend urgent imaging and monitoring."
        elif risk_level == "Moderate":
            reco = "Moderate risk indicators detected. Consider follow-up imaging."
        else:
            reco = "Low-risk profile based on initial labs. Continue standard monitoring."

        # ── ECG ───────────────────────────────────────────────────────────────
        label_dict = {
            lbl: row.get(lbl) for lbl in ["Atelectasis","Cardiomegaly","Edema","Pleural Effusion"]
        }
        ecg_data = build_ecg(label_dict, admitted_at)

        # ── Assemble records ──────────────────────────────────────────────────
        patients.append({
            "id":               patient_id,
            "mrn":              mrn,
            "first_name":       first,
            "last_name":        last,
            "date_of_birth":    dob,
            "sex":              sex,
            "age_at_admission": age,
            "mimic_subject_id": int(row["subject_id"]) if not pd.isna(row.get("subject_id")) else None,
            "created_at":       now_str,
            "updated_at":       now_str,
        })

        cases.append({
            "id":                     case_id,
            "patient_id":             patient_id,
            "admitted_at":            admitted_at,
            "discharged_at":          None,
            "ecg_data":               ecg_data,
            "lab_data":               lab_data,
            "labs_raw":               labs_raw,
            "phase_a_risk_level":     risk_level,
            "phase_a_risk_score":     round(risk_score, 3),
            "phase_a_recommendation": reco,
            "phase_a_run_at":         now_str,
            "cxr_dicom_url":          cxr_url,
            "cxr_acquired_at":        admitted_at,
            "cxr_heatmap_url":        None,
            "cxr_heatmap_label":      None,
            "mimic_study_id":         None,
            "created_at":             now_str,
            "updated_at":             now_str,
        })

        case_preds = synthesise_predictions(row, case_id, now_str)
        predictions.extend(case_preds)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/150 cases …")

    print(f"  Saved {len(saved_img_paths)} PNG images to {DICOMS_DIR}")
    print(f"  Built {len(patients)} patients, {len(cases)} cases, {len(predictions)} predictions")

    # ── Task 5: FAISS batch embedding ─────────────────────────────────────────
    print("\n[5/5] Extracting DenseNet121 GAP embeddings for FAISS …")
    try:
        embeddings = extract_all_embeddings(saved_img_paths)
        save_faiss_index(embeddings)
    except Exception as exc:
        print(f"  [WARN] FAISS embedding failed: {exc}")
        print("  Skipping FAISS - similarity search will be unavailable until inference runs.")

    # ── Write local_db.json ───────────────────────────────────────────────────
    print("\nWriting local_db.json …")
    final_db = {
        "patients":     patients,
        "cases":        cases,
        "predictions":  predictions,
        "consultations": [],
        "lab_results":  [],
        "ecg_records":  [],
    }
    with open(LOCAL_DB, "w") as f:
        json.dump(final_db, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("DEMO DATASET CREATED")
    print("="*60)
    print(f"  Cases:       {len(cases)}")
    print(f"  Patients:    {len(patients)}")
    print(f"  Predictions: {len(predictions)} ({len(predictions)//len(cases)} per case)")
    print(f"  CXR images:  {DICOMS_DIR}")
    print(f"  FAISS index: {FAISS_IDX}")
    print(f"  local_db:    {LOCAL_DB}")
    print()
    print("Common labels (Symile + DenseNet121 shared):")
    for lbl in COMMON_LABELS:
        pos = sum(1 for c in cases
                  for p in predictions
                  if p.get("case_id") == c["id"] and p.get("label") == lbl and p.get("probability", 0) >= 0.15)
        print(f"  {lbl:22s}  ({pos} Elevated Risk cases)")
    print()
    print("ECG JSON structure:")
    print(json.dumps(cases[0]["ecg_data"], indent=4))
    print()
    print("Lab data structure (LabData schema):")
    print(json.dumps({k: v for k, v in cases[0]["lab_data"].items() if k != "collected_at"}, indent=4))
    print()
    print("Restart the backend to load the new database:")
    print("  cd backend && uvicorn main:app --reload")
    print("="*60)


if __name__ == "__main__":
    main()
