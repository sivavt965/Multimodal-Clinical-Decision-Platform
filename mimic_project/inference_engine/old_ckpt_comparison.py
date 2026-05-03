# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

"""
old_ckpt_comparison.py
Loads baseline_best_old_state_dict.pt, handles the 'backbone.' prefix
difference, runs inference on the same 3 test images and prints
a side-by-side GT comparison against baseline_best.pt results.
"""

import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image

from src.models.densenet121 import build_densenet121

LABEL_COLUMNS = [
    "Cardiomegaly", "Pleural Effusion", "Edema", "Pneumonia",
    "Atelectasis", "Pneumothorax", "Consolidation", "Support Devices",
]

THRESHOLDS = {
    "Cardiomegaly":    20.0,
    "Pleural Effusion": 15.0,
    "Edema":            10.0,
    "Pneumonia":         8.0,
    "Atelectasis":      15.0,
    "Pneumothorax":      5.0,
    "Consolidation":     8.0,
    "Support Devices":  15.0,
}

OLD_CKPT   = "models/baseline_best_old_state_dict.pt"
NEW_CKPT   = "models/baseline_best.pt"
TEMPERATURE = 1.3
IMG_SIZE    = 512

IMAGES = [
    {
        "dicom_id": "abea5eb9-b7c32823-3a14c5ca-77868030-69c83139",
        "path": r"results\image_only_nomask_main\cache_images\test\files\p10\p10046166\s50051329\abea5eb9-b7c32823-3a14c5ca-77868030-69c83139.jpg",
        "desc": "0-positive (clean CXR)",
    },
    {
        "dicom_id": "4c3c1335-0fce9b11-027c582b-a0ed8d89-ca614d90",
        "path": r"results\image_only_nomask_main\cache_images\test\files\p10\p10268877\s50042142\4c3c1335-0fce9b11-027c582b-a0ed8d89-ca614d90.jpg",
        "desc": "2-positive (Cardiomegaly + Support Devices)",
    },
    {
        "dicom_id": "aeb77932-e37cc2ed-c6a8425e-955a35be-387a1d3e",
        "path": r"results\image_only_nomask_main\cache_images\test\files\p10\p10268877\s51051449\aeb77932-e37cc2ed-c6a8425e-955a35be-387a1d3e.jpg",
        "desc": "5-positive (Cardiomegaly, Pleural Effusion, Edema, Atelectasis, Support Devices)",
    },
]

# ── Previously recorded baseline_best.pt results (T=1.3) ──────────────────
NEW_RESULTS = {
    "abea5eb9-b7c32823-3a14c5ca-77868030-69c83139": {
        "Cardiomegaly": 2.89, "Pleural Effusion": 4.89, "Edema": 0.21,
        "Pneumonia": 4.39, "Atelectasis": 24.14, "Pneumothorax": 0.63,
        "Consolidation": 1.84, "Support Devices": 3.05,
    },
    "4c3c1335-0fce9b11-027c582b-a0ed8d89-ca614d90": {
        "Cardiomegaly": 34.18, "Pleural Effusion": 16.88, "Edema": 23.20,
        "Pneumonia": 8.81, "Atelectasis": 26.86, "Pneumothorax": 1.14,
        "Consolidation": 13.28, "Support Devices": 45.55,
    },
    "aeb77932-e37cc2ed-c6a8425e-955a35be-387a1d3e": {
        "Cardiomegaly": 15.75, "Pleural Effusion": 20.38, "Edema": 5.26,
        "Pneumonia": 5.63, "Atelectasis": 42.58, "Pneumothorax": 1.08,
        "Consolidation": 7.36, "Support Devices": 34.05,
    },
}


# ── Model loader for old checkpoint (backbone. prefix) ─────────────────────
def load_old_model(ckpt_path: str, device: torch.device) -> nn.Module:
    """
    The old state_dict uses keys prefixed with 'backbone.' and 'classifier.'
    The current DenseNet121 model uses 'features.' and 'classifier.'
    
    Remapping:
        backbone.features.* -> features.*
        classifier.*        -> classifier.*   (no change)
    """
    sd_raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    remapped = {}
    skipped = []
    for k, v in sd_raw.items():
        if k.startswith("backbone.features."):
            new_k = k.replace("backbone.features.", "features.", 1)
            remapped[new_k] = v
        elif k.startswith("backbone."):
            new_k = k.replace("backbone.", "", 1)
            remapped[new_k] = v
        elif k.startswith("classifier."):
            remapped[k] = v
        else:
            skipped.append(k)

    model = build_densenet121(num_classes=8, pretrained=False, dropout_p=0.3)
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    print(f"  [old ckpt] Remapped {len(remapped)} keys")
    print(f"  [old ckpt] Missing : {len(missing)} | Unexpected: {len(unexpected)} | Skipped: {len(skipped)}")
    if missing:
        print(f"            Missing sample: {missing[:3]}")
    model.to(device)
    model.eval()
    return model


# ── Preprocessing ──────────────────────────────────────────────────────────
def preprocess(img_path: str, device: torch.device) -> torch.Tensor:
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t.to(device)


def temperature_scale(logits: torch.Tensor, T: float) -> np.ndarray:
    return torch.sigmoid(logits / T).squeeze(0).cpu().numpy()


# ── Main ───────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df_meta = pd.read_csv("data/processed/processed_metadata.csv", low_memory=False)

print(f"\n{'='*78}")
print("  OLD vs NEW CHECKPOINT COMPARISON")
print(f"  OLD: baseline_best_old_state_dict.pt  (backbone.* prefix, raw state_dict)")
print(f"  NEW: baseline_best.pt                 (features.* prefix, full checkpoint)")
print(f"  Temperature: {TEMPERATURE}  |  Device: {device}")
print(f"{'='*78}")

print("\n>> Loading OLD checkpoint...")
old_model = load_old_model(OLD_CKPT, device)

overall_old_c, overall_new_c, overall_total = 0, 0, 0

for img_info in IMAGES:
    dicom_id = img_info["dicom_id"]
    img_path = img_info["path"]
    desc     = img_info["desc"]

    gt_row = df_meta[df_meta["dicom_id"] == dicom_id].iloc[0]

    print(f"\n{'-'*78}")
    print(f"  Image : {dicom_id}")
    print(f"  Case  : {desc}")
    print(f"  View  : {gt_row.get('ViewPosition','?')}   Split: {gt_row.get('split','?')}")
    print()

    # Run old model
    x = preprocess(img_path, device)
    with torch.inference_mode():
        logits_old = old_model(x)
    probs_old = temperature_scale(logits_old, TEMPERATURE)

    # Pull new model cached results
    new_probs_dict = NEW_RESULTS[dicom_id]

    # Header
    print(f"  {'Label':<22} {'GT':>10} | "
          f"{'OLD%':>8} {'O-Pred':>7} | "
          f"{'NEW%':>8} {'N-Pred':>7} | "
          f"{'Thr%':>5}  {'Win?':>6}")
    print(f"  {'-'*22} {'-'*10}-+-{'-'*8}-{'-'*7}-+-{'-'*8}-{'-'*7}-+-{'-'*5}--{'-'*6}")

    case_old_c, case_new_c, case_total = 0, 0, 0

    for i, col in enumerate(LABEL_COLUMNS):
        val      = gt_row.get(col, float("nan"))
        thr      = THRESHOLDS[col]
        old_pct  = float(probs_old[i]) * 100
        new_pct  = new_probs_dict[col]
        old_pred = 1 if old_pct >= thr else 0
        new_pred = 1 if new_pct >= thr else 0

        if pd.isna(val) or float(val) == -1.0:
            gt_str   = "uncertain"
            gt_int   = None
        elif float(val) == 1.0:
            gt_str = "POSITIVE"
            gt_int = 1
        else:
            gt_str = "negative"
            gt_int = 0

        if gt_int is not None:
            old_match = old_pred == gt_int
            new_match = new_pred == gt_int
            old_m_str = "[YES]" if old_match else "[NO] "
            new_m_str = "[YES]" if new_match else "[NO] "
            case_total += 1
            if old_match: case_old_c += 1
            if new_match: case_new_c += 1

            # Who wins this label?
            if old_match and not new_match:
                win = "OLD"
            elif new_match and not old_match:
                win = "NEW"
            elif old_match and new_match:
                win = "BOTH"
            else:
                win = "NONE"
        else:
            old_m_str = "  --- "
            new_m_str = "  --- "
            win       = " --- "

        # Mark GT-positive rows
        mark = " <--" if gt_int == 1 else ""

        print(f"  {col:<22} {gt_str:>10} | "
              f"{old_pct:>7.2f}% {old_pred:>3} {old_m_str} | "
              f"{new_pct:>7.2f}% {new_pred:>3} {new_m_str} | "
              f"{thr:>4.0f}%  {win:>6}{mark}")

    print()
    if case_total > 0:
        oa = case_old_c / case_total * 100
        na = case_new_c / case_total * 100
        print(f"  Case accuracy -- OLD: {case_old_c}/{case_total} ({oa:.1f}%)  |  "
              f"NEW: {case_new_c}/{case_total} ({na:.1f}%)")
        overall_old_c += case_old_c
        overall_new_c += case_new_c
        overall_total += case_total

print(f"\n{'='*78}")
if overall_total:
    oa = overall_old_c / overall_total * 100
    na = overall_new_c / overall_total * 100
    print(f"  OVERALL (24 labels, 3 images)")
    print(f"    OLD checkpoint : {overall_old_c}/{overall_total}  ({oa:.1f}%)")
    print(f"    NEW checkpoint : {overall_new_c}/{overall_total}  ({na:.1f}%)")
    diff = na - oa
    winner = "NEW" if diff > 0 else ("OLD" if diff < 0 else "TIE")
    print(f"    Delta NEW-OLD  : {diff:+.1f} pp  -> {winner} checkpoint is better overall")
print(f"{'='*78}\n")

del old_model
torch.cuda.empty_cache()
