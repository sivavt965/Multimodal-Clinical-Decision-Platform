# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
gt_comparison.py --- Ground Truth vs Inference Comparison
Runs cxr_inference on 3 test images (0, 2, and 5 positive labels)
and prints a side-by-side table vs ground truth labels.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from inference_engine.cxr_inference import predict

LABEL_COLUMNS = [
    "Cardiomegaly", "Pleural Effusion", "Edema", "Pneumonia",
    "Atelectasis", "Pneumothorax", "Consolidation", "Support Devices",
]

# Label-specific thresholds (approximate operating points from ROC, typical for MIMIC-CXR DenseNet)
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

CKPT = "models/baseline_best.pt"
TEMPERATURE = 1.3

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

# Load processed metadata for GT lookup
df_meta = pd.read_csv("data/processed/processed_metadata.csv", low_memory=False)

print("\n" + "=" * 75)
print("  INFERENCE ENGINE --- GROUND TRUTH COMPARISON REPORT")
print("  Checkpoint: baseline_best.pt  |  Temperature: 1.3  |  Device: CUDA")
print("=" * 75)

overall_correct = 0
overall_total = 0

for img_info in IMAGES:
    dicom_id = img_info["dicom_id"]
    img_path = img_info["path"]
    desc = img_info["desc"]

    print(f"\n{'-'*75}")
    print(f"  Image : {dicom_id}")
    print(f"  Case  : {desc}")

    # Ground truth lookup
    row = df_meta[df_meta["dicom_id"] == dicom_id]
    if len(row) == 0:
        print("  [WARN] DICOM ID not found in metadata CSV")
        continue
    gt_row = row.iloc[0]
    print(f"  View  : {gt_row.get('ViewPosition','?')}   Split: {gt_row.get('split','?')}")
    print()

    # Run inference
    result = predict(image_path=img_path, ckpt_path=CKPT, temperature=TEMPERATURE, top_k=8)

    if result["status"] != "ok":
        print(f"  [ERROR] {result.get('error')}")
        continue

    all_probs = result["all_findings"]  # dict label -> pct

    print(f"  {'Label':<22} {'GT':>10}  {'Prob%':>8}  {'Thr%':>6}  {'Pred':>5}  {'Match?':>8}")
    print(f"  {'-'*68}")

    case_correct = 0
    case_total = 0

    for col in LABEL_COLUMNS:
        val = gt_row.get(col, float("nan"))
        prob_pct = all_probs.get(col, 0.0)
        thr = THRESHOLDS[col]
        pred = 1 if prob_pct >= thr else 0

        if pd.isna(val):
            gt_str = "uncertain"
            match_str = "   ---"
        elif float(val) == -1.0:
            gt_str = "uncertain"
            match_str = "   ---"
        elif float(val) == 1.0:
            gt = 1
            gt_str = "POSITIVE"
            match_str = "  [YES]" if pred == gt else "  [NO] "
            case_total += 1
            if pred == gt: case_correct += 1
        elif float(val) == 0.0:
            gt = 0
            gt_str = "negative"
            match_str = "  [YES]" if pred == gt else "  [NO] "
            case_total += 1
            if pred == gt: case_correct += 1
        else:
            gt_str = f"unc({val})"
            match_str = "   ---"

        # Flag interesting rows (positives or wrong predictions)
        flag = " <--" if (float(val) == 1.0 if not pd.isna(val) else False) else ""
        print(f"  {col:<22} {gt_str:>10}  {prob_pct:>7.2f}%  {thr:>5.0f}%  {pred:>5}  {match_str}{flag}")

    print()
    if case_total > 0:
        acc = case_correct / case_total * 100
        print(f"  Case accuracy: {case_correct}/{case_total} ({acc:.1f}%)")
        overall_correct += case_correct
        overall_total += case_total
    print()

print("=" * 75)
if overall_total > 0:
    oa = overall_correct / overall_total * 100
    print(f"  OVERALL ACCURACY (all 3 images, definitive labels): "
          f"{overall_correct}/{overall_total}  ({oa:.1f}%)")
print("=" * 75)
print()
print("Threshold note: Using label-specific operating-point thresholds")
print("(not the naive 50% cutoff). These approximate the Youden-J optimal")
print("threshold typical for MIMIC-CXR DenseNet models.")
