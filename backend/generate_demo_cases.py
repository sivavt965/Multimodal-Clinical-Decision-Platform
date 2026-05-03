#!/usr/bin/env python3
"""
generate_demo_cases.py — Synthetic data generator for the Symile-MIMIC demo.

Produces 150 synthetic cases with:
  - Realistic patient demographics
  - All 50 MIMIC-IV lab itemids with clinically coherent value ranges
  - Structured ECG parameters
  - Phase A risk stratification (heuristic)
  - CXR predictions across the 8 modeled labels (Cardiomegaly, Pleural Effusion,
    Edema, Pneumonia, Atelectasis, Pneumothorax, Consolidation, Support Devices)
  - Mock Grad-CAM and MC Dropout data

Output: Overwrites backend/local_db.json with the synthetic dataset.
"""

import json
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────────────────────
NUM_CASES = 150
OUTPUT_PATH = Path(__file__).parent / "local_db.json"

# The 8 labels that baseline_best.pt actually predicts
MODELED_LABELS = [
    "Cardiomegaly", "Pleural Effusion", "Edema", "Pneumonia",
    "Atelectasis", "Pneumothorax", "Consolidation", "Support Devices",
]

# Full 14-label CheXpert taxonomy (non-modeled labels get probability=0)
ALL_CXR_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices",
]

# ─── Name pools ──────────────────────────────────────────────────────────────
FIRST_NAMES_M = [
    "James", "Robert", "Michael", "David", "William", "Richard", "Joseph",
    "Thomas", "Charles", "Christopher", "Daniel", "Matthew", "Anthony",
    "Mark", "Steven", "Paul", "Andrew", "Joshua", "Kenneth", "Kevin",
    "Brian", "George", "Timothy", "Ronald", "Edward", "Jason", "Jeffrey",
    "Ryan", "Jacob", "Gary", "Nicholas", "Eric", "Jonathan", "Stephen",
    "Larry", "Justin", "Scott", "Brandon", "Benjamin", "Samuel",
]
FIRST_NAMES_F = [
    "Mary", "Patricia", "Jennifer", "Linda", "Barbara", "Elizabeth",
    "Susan", "Jessica", "Sarah", "Karen", "Lisa", "Nancy", "Betty",
    "Margaret", "Sandra", "Ashley", "Dorothy", "Kimberly", "Emily",
    "Donna", "Michelle", "Carol", "Amanda", "Melissa", "Deborah",
    "Stephanie", "Rebecca", "Sharon", "Laura", "Cynthia", "Kathleen",
    "Amy", "Angela", "Shirley", "Anna", "Brenda", "Pamela", "Emma",
    "Nicole", "Helen",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
    "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez",
    "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore",
    "Jackson", "Martin", "Lee", "Perez", "Thompson", "White",
    "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson",
    "Walker", "Young", "Allen", "King", "Wright", "Scott", "Torres",
    "Nguyen", "Hill", "Flores", "Green", "Adams", "Nelson", "Baker",
    "Hall", "Rivera", "Campbell", "Mitchell", "Carter", "Roberts",
]

RHYTHMS = [
    "Normal Sinus Rhythm", "Sinus Tachycardia", "Sinus Bradycardia",
    "Atrial Fibrillation", "Atrial Flutter", "First Degree AV Block",
    "Left Bundle Branch Block", "Right Bundle Branch Block",
]

# ─── Lab ranges by clinical profile ─────────────────────────────────────────
# Keyed by MIMIC-IV itemid → (low_normal, high_normal) for realistic draws
# When risk is "High", some markers will be pushed outside normal range

LAB_NORMAL_RANGES = {
    "51221": (36.0, 46.0),    # Hematocrit (%)
    "51265": (150.0, 400.0),  # Platelet Count (K/uL)
    "50912": (0.6, 1.2),      # Creatinine (mg/dL)
    "50971": (3.5, 5.0),      # Potassium (mEq/L)
    "51222": (12.0, 16.0),    # Hemoglobin (g/dL)
    "51301": (4.5, 11.0),     # White Blood Cells (K/uL)
    "51249": (31.0, 36.0),    # MCHC (g/dL)
    "51279": (4.0, 5.5),      # Red Blood Cells (M/uL)
    "51250": (80.0, 100.0),   # MCV (fL)
    "51248": (27.0, 33.0),    # MCH (pg)
    "51277": (11.5, 14.5),    # RDW (%)
    "51006": (7.0, 20.0),     # Urea Nitrogen (mg/dL)
    "50983": (136.0, 145.0),  # Sodium (mEq/L)
    "50902": (98.0, 106.0),   # Chloride (mEq/L)
    "50882": (22.0, 29.0),    # Bicarbonate (mEq/L)
    "50868": (8.0, 12.0),     # Anion Gap (mEq/L)
    "50931": (70.0, 100.0),   # Glucose (mg/dL)
    "50960": (1.7, 2.2),      # Magnesium (mg/dL)
    "50893": (8.5, 10.5),     # Calcium, Total (mg/dL)
    "50970": (2.5, 4.5),      # Phosphate (mg/dL)
    "51237": (0.9, 1.1),      # INR(PT)
    "51274": (11.0, 14.0),    # PT (sec)
    "51275": (25.0, 35.0),    # PTT (sec)
    "51146": (0.0, 1.0),      # Basophils (%)
    "51256": (40.0, 70.0),    # Neutrophils (%)
    "51254": (2.0, 8.0),      # Monocytes (%)
    "51200": (1.0, 4.0),      # Eosinophils (%)
    "51244": (20.0, 40.0),    # Lymphocytes (%)
    "52172": (35.0, 46.0),    # RDW-SD (fL)
    "50934": (0.0, 0.5),      # H (placeholder)
    "51678": (0.0, 0.5),      # L (placeholder)
    "50947": (0.0, 0.04),     # Troponin I (ng/mL)
    "50861": (7.0, 56.0),     # ALT (U/L)
    "50878": (10.0, 40.0),    # AST (U/L)
    "50813": (0.5, 2.0),      # Lactate (mmol/L)
    "50863": (44.0, 147.0),   # Alkaline Phosphatase (U/L)
    "50885": (0.1, 1.2),      # Bilirubin, Total (mg/dL)
    "50820": (7.35, 7.45),    # pH
    "50862": (3.5, 5.0),      # Albumin (g/dL)
    "50802": (-2.0, 2.0),     # Base Excess (mEq/L)
    "50821": (80.0, 100.0),   # pO2 (mmHg)
    "50804": (22.0, 28.0),    # Calculated Total CO2 (mEq/L)
    "50818": (35.0, 45.0),    # pCO2 (mmHg)
    "52075": (1.8, 7.7),      # Absolute Neutrophil Count (K/uL)
    "52073": (0.04, 0.5),     # Absolute Eosinophil Count (K/uL)
    "52074": (0.1, 0.9),      # Absolute Monocyte Count (K/uL)
    "52069": (0.0, 0.1),      # Absolute Basophil Count (K/uL)
    "51133": (1.0, 4.8),      # Absolute Lymphocyte Count (K/uL)
    "50910": (22.0, 198.0),   # Creatine Kinase CK (U/L)
    "52135": (0.0, 1.0),      # Immature Granulocytes (%)
}


def _draw_lab_value(itemid: str, risk_profile: str) -> float:
    """Draw a lab value — normal for Low, borderline for Moderate, abnormal for High."""
    lo, hi = LAB_NORMAL_RANGES.get(itemid, (0, 1))
    mid = (lo + hi) / 2
    spread = hi - lo

    if risk_profile == "Low":
        # Within the normal range
        return round(random.uniform(lo, hi), 2)
    elif risk_profile == "Moderate":
        # Borderline — occasionally just outside
        if random.random() < 0.3:
            # Push slightly out
            if random.random() < 0.5:
                return round(lo - random.uniform(0, spread * 0.15), 2)
            else:
                return round(hi + random.uniform(0, spread * 0.15), 2)
        return round(random.uniform(lo, hi), 2)
    else:  # High
        # For clinically significant markers, push them out of range
        critical_high = {"50947", "50912", "50813", "51301", "50971", "50885", "50931", "51006"}
        if itemid in critical_high and random.random() < 0.6:
            return round(hi + random.uniform(spread * 0.3, spread * 1.2), 2)
        if random.random() < 0.4:
            if random.random() < 0.5:
                return round(lo - random.uniform(0, spread * 0.3), 2)
            else:
                return round(hi + random.uniform(0, spread * 0.3), 2)
        return round(random.uniform(lo, hi), 2)


def _generate_predictions(risk_profile: str, case_id: str, now_str: str) -> list:
    """Generate 14 prediction rows — 8 modeled + 6 unmodeled (p=0)."""
    predictions = []

    # Decide dominant findings based on risk profile
    if risk_profile == "High":
        # 2-4 findings with elevated probability
        num_elevated = random.randint(2, 4)
        elevated_labels = random.sample(MODELED_LABELS, num_elevated)
    elif risk_profile == "Moderate":
        num_elevated = random.randint(1, 2)
        elevated_labels = random.sample(MODELED_LABELS, num_elevated)
    else:
        num_elevated = random.randint(0, 1)
        elevated_labels = random.sample(MODELED_LABELS, num_elevated)

    heatmap_label = elevated_labels[0] if elevated_labels else "Pleural Effusion"
    heatmap_uuid = uuid.uuid4().hex[:8]
    heatmap_url = f"/mock-data/heatmaps/heatmap_case_{case_id[:8]}_{heatmap_uuid}.png"

    for label in ALL_CXR_LABELS:
        pred_id = str(uuid.uuid4())

        if label not in MODELED_LABELS:
            # Non-modeled label
            predictions.append({
                "id": pred_id,
                "case_id": case_id,
                "model_checkpoint": "densenet121-chexpert",
                "temperature": 1.0,
                "inference_run_at": now_str,
                "label": label,
                "probability": 0.0,
                "risk_badge": "Unlikely",
                "uncertainty_level": None,
                "mean_variance": None,
                "std_dev": 0.0,
                "mc_passes": 0,
                "gradcam_url": None,
                "gradcam_alpha": 0.45,
            })
        else:
            # Modeled label — draw probability
            if label in elevated_labels:
                if risk_profile == "High":
                    prob = round(random.uniform(0.25, 0.85), 4)
                elif risk_profile == "Moderate":
                    prob = round(random.uniform(0.10, 0.50), 4)
                else:
                    prob = round(random.uniform(0.05, 0.25), 4)
            else:
                prob = round(random.uniform(0.001, 0.12), 4)

            # Risk badge thresholds
            if prob < 0.05:
                badge = "Unlikely"
            elif prob <= 0.15:
                badge = "Monitor"
            else:
                badge = "Elevated Risk"

            # MC Dropout data
            mc_passes = 10
            std_dev = round(random.uniform(0.001, 0.05) * prob, 6)
            mean_var = round(std_dev ** 2, 8)
            if std_dev < 0.01:
                unc = "Low Uncertainty"
            elif std_dev < 0.03:
                unc = "Moderate Uncertainty"
            else:
                unc = "High Uncertainty"

            predictions.append({
                "id": pred_id,
                "case_id": case_id,
                "model_checkpoint": "densenet121-chexpert",
                "temperature": 1.0,
                "inference_run_at": now_str,
                "label": label,
                "probability": prob,
                "risk_badge": badge,
                "uncertainty_level": unc,
                "mean_variance": mean_var,
                "std_dev": std_dev,
                "mc_passes": mc_passes,
                "gradcam_url": heatmap_url if label == heatmap_label else None,
                "gradcam_alpha": 0.45,
            })

    return predictions, heatmap_url, heatmap_label


def _risk_score(labs_raw: dict) -> tuple:
    """Compute Phase A risk from raw lab values."""
    troponin = labs_raw.get("50947", 0)
    creatinine = labs_raw.get("50912", 0)
    lactate = labs_raw.get("50813", 0)
    potassium = labs_raw.get("50971", 0)
    sodium = labs_raw.get("50983", 0)
    wbc = labs_raw.get("51301", 0)
    bnp = labs_raw.get("51006", 0)

    score = 0.0
    if troponin > 0.04:
        score += 0.4
    if creatinine > 1.3:
        score += 0.2
    if lactate > 2.0:
        score += 0.2
    if potassium > 5.0 or (potassium > 0 and potassium < 3.5):
        score += 0.1
    if sodium > 0 and (sodium < 136 or sodium > 145):
        score += 0.1

    if score >= 0.4:
        level = "High"
    elif score >= 0.2:
        level = "Moderate"
    else:
        level = "Low"

    return level, min(score, 1.0)


def generate():
    """Generate the full demo dataset."""
    random.seed(42)

    patients = []
    cases = []
    predictions = []
    consultations = []

    # Stratify: 40% Low, 35% Moderate, 25% High
    profiles = (
        ["Low"] * 60 +
        ["Moderate"] * 53 +
        ["High"] * 37
    )
    random.shuffle(profiles)

    base_time = datetime(2026, 4, 20, 8, 0, 0, tzinfo=timezone.utc)

    for i, target_risk in enumerate(profiles):
        case_id = str(uuid.uuid4())
        patient_id = str(uuid.uuid4())

        # Time progression: cases admitted over ~3 days
        offset_hours = random.uniform(0, 72)
        admit_time = base_time + timedelta(hours=offset_hours)
        now_str = admit_time.isoformat()

        # Demographics
        sex = random.choice(["M", "F"])
        if sex == "M":
            first_name = random.choice(FIRST_NAMES_M)
        else:
            first_name = random.choice(FIRST_NAMES_F)
        last_name = random.choice(LAST_NAMES)
        age = random.randint(25, 92)
        mrn = f"MRN-{random.randint(10000, 99999)}"

        # ── Lab values ───────────────────────────────────────────────────
        labs_raw = {}
        for itemid in LAB_NORMAL_RANGES:
            # 80% chance of having each lab (some missing is realistic)
            if random.random() < 0.80:
                labs_raw[itemid] = _draw_lab_value(itemid, target_risk)

        # Ensure critical labs are always present
        for crit_id in ["50912", "50971", "50983", "51301", "51221", "51222"]:
            if crit_id not in labs_raw:
                labs_raw[crit_id] = _draw_lab_value(crit_id, target_risk)

        # ── Force risk markers to match target profile ───────────────────
        # The Phase A heuristic uses troponin(>0.04→+0.4), creatinine(>1.3→+0.2),
        # lactate(>2.0→+0.2), K+, Na+. We force markers so the heuristic output
        # matches the intended target_risk, not just random chance.
        if target_risk == "High":
            # Need score >= 0.4  → force troponin high (guaranteed +0.4)
            labs_raw["50947"] = round(random.uniform(0.06, 0.90), 3)   # Troponin > 0.04
            # Also push creatinine or lactate for variety
            if random.random() < 0.6:
                labs_raw["50912"] = round(random.uniform(1.5, 4.0), 2)  # Creatinine > 1.3
            if random.random() < 0.5:
                labs_raw["50813"] = round(random.uniform(2.5, 8.0), 1)  # Lactate > 2.0
        elif target_risk == "Moderate":
            # Need 0.2 <= score < 0.4  → force creatinine OR lactate high, but NOT troponin
            labs_raw["50947"] = round(random.uniform(0.0, 0.03), 3)     # Troponin normal
            # Pick one of creatinine or lactate to be elevated
            if random.random() < 0.5:
                labs_raw["50912"] = round(random.uniform(1.4, 2.5), 2)  # Creatinine > 1.3
                labs_raw["50813"] = round(random.uniform(0.5, 1.8), 1)  # Lactate normal
            else:
                labs_raw["50912"] = round(random.uniform(0.6, 1.2), 2)  # Creatinine normal
                labs_raw["50813"] = round(random.uniform(2.2, 5.0), 1)  # Lactate > 2.0
        else:  # Low
            # Need score < 0.2  → all critical markers normal
            labs_raw["50947"] = round(random.uniform(0.0, 0.03), 3)     # Troponin normal
            labs_raw["50912"] = round(random.uniform(0.6, 1.2), 2)      # Creatinine normal
            labs_raw["50813"] = round(random.uniform(0.5, 1.8), 1)      # Lactate normal
            labs_raw["50971"] = round(random.uniform(3.6, 4.8), 1)      # Potassium normal
            labs_raw["50983"] = round(random.uniform(137.0, 144.0), 0)  # Sodium normal

        # ── Phase A risk ─────────────────────────────────────────────────
        risk_level, risk_score_val = _risk_score(labs_raw)

        if risk_level == "High":
            recommendation = "Elevated cardiac and metabolic markers. Recommend urgent imaging and ICU monitoring."
        elif risk_level == "Moderate":
            recommendation = "Moderate risk indicators detected. Consider follow-up imaging and serial labs."
        else:
            recommendation = "Low-risk profile based on initial labs. Continue standard monitoring."

        # ── ECG data ─────────────────────────────────────────────────────
        rhythm = random.choice(RHYTHMS)
        if target_risk == "High":
            hr = random.randint(90, 140)
            pr = random.randint(140, 240)
            qrs = random.randint(80, 160)
            qtc = random.randint(400, 550)
            st_dev = round(random.uniform(-2.0, 3.0), 1)
        elif target_risk == "Moderate":
            hr = random.randint(70, 110)
            pr = random.randint(130, 210)
            qrs = random.randint(80, 130)
            qtc = random.randint(380, 480)
            st_dev = round(random.uniform(-1.0, 1.5), 1)
        else:
            hr = random.randint(55, 95)
            pr = random.randint(120, 200)
            qrs = random.randint(70, 110)
            qtc = random.randint(350, 440)
            st_dev = round(random.uniform(-0.5, 0.5), 1)

        ecg_data = {
            "heart_rate": hr,
            "pr_interval_ms": pr,
            "qrs_duration_ms": qrs,
            "qtc_ms": qtc,
            "st_deviation_mm": st_dev,
            "rhythm_interpretation": rhythm,
            "acquired_at": now_str,
        }

        # ── CXR image path (mock) ────────────────────────────────────────
        cxr_url = f"/mock-data/dicoms/case_{case_id[:8]}.jpg"

        # ── Patient ──────────────────────────────────────────────────────
        patients.append({
            "id": patient_id,
            "mrn": mrn,
            "first_name": first_name,
            "last_name": last_name,
            "date_of_birth": f"{random.randint(1935, 2001)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
            "sex": sex,
            "age_at_admission": age,
            "mimic_subject_id": None,
            "created_at": now_str,
            "updated_at": now_str,
        })

        # ── Map labs_raw → legacy lab_data ────────────────────────────────
        lab_data = {
            "troponin_ng_ml": labs_raw.get("50947", 0.0),
            "bnp_pg_ml": labs_raw.get("51006", 0.0),
            "wbc_count": labs_raw.get("51301", 0.0),
            "creatinine_mg_dl": labs_raw.get("50912", 0.0),
            "sodium_meq_l": labs_raw.get("50983", 0.0),
            "potassium_meq_l": labs_raw.get("50971", 0.0),
            "lactate_mmol_l": labs_raw.get("50813", 0.0),
            "collected_at": now_str,
        }

        # ── CXR predictions ──────────────────────────────────────────────
        preds, heatmap_url, heatmap_label = _generate_predictions(
            target_risk, case_id, now_str
        )
        predictions.extend(preds)

        # ── Case ─────────────────────────────────────────────────────────
        # Some cases are already discharged (for "completed" status demo)
        discharged = None
        if random.random() < 0.15:
            discharged = (admit_time + timedelta(hours=random.uniform(24, 120))).isoformat()

        cases.append({
            "id": case_id,
            "patient_id": patient_id,
            "admitted_at": now_str,
            "discharged_at": discharged,
            "ecg_data": ecg_data,
            "lab_data": lab_data,
            "labs_raw": labs_raw,
            "phase_a_risk_level": risk_level,
            "phase_a_risk_score": round(risk_score_val, 2),
            "phase_a_recommendation": recommendation,
            "phase_a_run_at": now_str,
            "cxr_dicom_url": cxr_url,
            "cxr_acquired_at": now_str,
            "cxr_heatmap_url": heatmap_url,
            "cxr_heatmap_label": heatmap_label,
            "mimic_study_id": None,
            "created_at": now_str,
            "updated_at": now_str,
        })

        # ── Consultation (some cases have open threads) ───────────────────
        if random.random() < 0.25:
            num_msgs = random.randint(1, 4)
            messages = []
            for m in range(num_msgs):
                msg_time = admit_time + timedelta(hours=random.uniform(1, 48))
                role = random.choice(["ward_doctor", "radiologist"])
                texts = [
                    "Please review CXR findings — possible effusion.",
                    "Agree with elevated risk assessment. Recommend thoracentesis.",
                    "Labs trending upward. Will re-assess in 6 hours.",
                    "ECG shows new-onset atrial fibrillation. Cardiology consult ordered.",
                    "Imaging confirms consolidation in right lower lobe.",
                    "Patient stable. Continue current management.",
                    "Grad-CAM overlay correlates well with clinical findings.",
                    "MC Dropout shows high uncertainty — consider repeat imaging.",
                ]
                messages.append({
                    "id": uuid.uuid4().hex[:8],
                    "role": role,
                    "type": "text",
                    "content": random.choice(texts),
                    "sent_at": msg_time.isoformat(),
                    "read": random.choice([True, False]),
                })

            is_open = random.choice([True, False])
            consultations.append({
                "id": f"cons-{uuid.uuid4().hex[:12]}",
                "case_id": case_id,
                "messages": messages,
                "created_at": now_str,
                "updated_at": messages[-1]["sent_at"],
                "is_open": is_open,
                "closed_at": None if is_open else (admit_time + timedelta(hours=72)).isoformat(),
            })

    # ── Write output ─────────────────────────────────────────────────────────
    db = {
        "patients": patients,
        "cases": cases,
        "predictions": predictions,
        "consultations": consultations,
    }

    OUTPUT_PATH.write_text(json.dumps(db, indent=2), encoding="utf-8")

    # Stats
    risk_counts = {"Low": 0, "Moderate": 0, "High": 0}
    for c in cases:
        risk_counts[c["phase_a_risk_level"]] += 1

    print(f"[OK] Generated {len(patients)} patients, {len(cases)} cases, "
          f"{len(predictions)} predictions, {len(consultations)} consultations")
    print(f"   Risk distribution: {risk_counts}")
    print(f"   Discharged: {sum(1 for c in cases if c['discharged_at'])}")
    print(f"   Open consultations: {sum(1 for c in consultations if c['is_open'])}")
    print(f"   Written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    generate()
