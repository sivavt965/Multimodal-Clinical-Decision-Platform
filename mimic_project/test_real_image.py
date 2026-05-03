#!/usr/bin/env python3
"""
Real MIMIC-CXR image test:
- Picks a locally cached test-split .jpg image
- Looks up its ground truth labels from the processed_metadata.csv
- Runs baseline_best.pt model inference on it
- Compares prediction vs ground truth
"""

import sys, os
sys.path.insert(0, r"b:\mimic_project_full_backup\mimic_project")

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---- Config ----
IMG_PATH  = r"b:\mimic_project_full_backup\mimic_project\results\image_only_nomask_main\cache_images\test\files\p10\p10046166\s50051329\abea5eb9-b7c32823-3a14c5ca-77868030-69c83139.jpg"
CKPT_PATH = r"b:\mimic_project_full_backup\mimic_project\models\baseline_best.pt"
CSV_PATH  = r"b:\mimic_project_full_backup\mimic_project\data\processed\processed_metadata.csv"

LABEL_COLUMNS = [
    "Cardiomegaly",
    "Pleural Effusion",
    "Edema",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Consolidation",
    "Support Devices",
]

THRESHOLD = 0.5

# ---- Extract dicom_id from path ----
dicom_id = os.path.splitext(os.path.basename(IMG_PATH))[0]
print("=" * 65)
print(f"Image    : {os.path.basename(IMG_PATH)}")
print(f"DICOM ID : {dicom_id}")

# ---- Look up ground truth in CSV ----
print("\nLooking up ground-truth labels from CSV...")
df = pd.read_csv(CSV_PATH)
row = df[df["dicom_id"] == dicom_id]

if len(row) == 0:
    print("  [WARN] DICOM ID not found in CSV — proceeding without ground truth")
    ground_truth = None
    split = "unknown"
else:
    row = row.iloc[0]
    split = row.get("split", "unknown")
    ground_truth = {}
    for lab in LABEL_COLUMNS:
        val = row.get(lab, float("nan"))
        if pd.isna(val):
            ground_truth[lab] = "uncertain/missing"
        elif float(val) == 1.0:
            ground_truth[lab] = 1
        elif float(val) == 0.0:
            ground_truth[lab] = 0
        else:
            ground_truth[lab] = f"uncertain ({val})"
    print(f"  Split   : {split}")
    print(f"  ViewPos : {row.get('ViewPosition','?')}")

# ---- Load and preprocess image ----
print("\n" + "=" * 65)
print("Loading image & preprocessing...")
img = Image.open(IMG_PATH).convert("RGB")
img_np = np.array(img, dtype=np.uint8)
print(f"  Original size : {img.size}  mode={img.mode}")

tf = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0., 0., 0.), std=(1., 1., 1.), max_pixel_value=255.0),
    ToTensorV2(),
])
x = tf(image=img_np)["image"].float().unsqueeze(0)  # [1, 3, 512, 512]
print(f"  Tensor shape  : {tuple(x.shape)}")

# ---- Load model ----
print("\n" + "=" * 65)
print("Loading model checkpoint...")
from src.models.densenet121 import build_densenet121

ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
model = build_densenet121(num_classes=8, pretrained=False, dropout_p=0.3)
sd = ckpt.get("model_state_dict", ckpt)
cleaned = {}
for k, v in sd.items():
    k2 = k
    for p in ("module.", "model."):
        if k2.startswith(p): k2 = k2[len(p):]
    cleaned[k2] = v
missing, unexpected = model.load_state_dict(cleaned, strict=False)
print(f"  Missing keys  : {len(missing)}  |  Unexpected keys : {len(unexpected)}")
model.eval()
print(f"  Checkpoint epoch  : {ckpt.get('epoch', '?')}")
print(f"  Best val AUC      : {ckpt.get('best_val_auc', '?'):.4f}")

# ---- Run inference ----
print("\n" + "=" * 65)
print("Running inference on real image...")
with torch.no_grad():
    logits = model(x)           # [1, 8]
    probs  = torch.sigmoid(logits).squeeze(0).numpy()  # [8]

# ---- Print results ----
print("\n" + "=" * 65)
print(f"{'Label':<22} {'Prob':>8}  {'Pred(0.5)':>10}  {'GroundTruth':>12}  {'Match?':>8}")
print("  " + "-" * 63)

correct = 0
total_definitive = 0

for i, lab in enumerate(LABEL_COLUMNS):
    prob  = float(probs[i])
    pred  = 1 if prob >= THRESHOLD else 0
    gt    = ground_truth[lab] if ground_truth else "?"

    if isinstance(gt, int):
        match = "YES" if pred == gt else "NO "
        if pred == gt:
            correct += 1
        total_definitive += 1
    else:
        match = "—"

    print(f"  {lab:<22} {prob:>8.4f}  {pred:>10}  {str(gt):>12}  {match:>8}")

print()
if total_definitive > 0:
    acc = correct / total_definitive
    print(f"  Accuracy on definitive labels: {correct}/{total_definitive} ({acc*100:.1f}%)")

print()
print("✅ Real image inference COMPLETE!")
