#!/usr/bin/env python3
import numpy as np
import json
import random

# File paths
prob_path = r"b:\mimic_project_full_backup\mimic_project\01_predictions\test\test_probs.npy"
tgt_path = r"b:\mimic_project_full_backup\mimic_project\01_predictions\test\test_targets.npy"
mask_path = r"b:\mimic_project_full_backup\mimic_project\01_predictions\test\test_label_valid_mask.npy"
thresh_path = r"b:\mimic_project_full_backup\mimic_project\02_metrics_preTS\clinical_operating_points.json"

try:
    probs = np.load(prob_path)
    tgts = np.load(tgt_path)
    masks = np.load(mask_path)
    with open(thresh_path, "r") as f:
        thresholds_data = json.load(f)
except Exception as e:
    print(f"Error loading files: {e}")
    exit(1)

LABEL_COLUMNS = [
    "Cardiomegaly", "Pleural Effusion", "Edema", "Pneumonia",
    "Atelectasis", "Pneumothorax", "Consolidation", "Support Devices"
]

# Extract F1max threshold for each label
f1_thresh = []
for label in LABEL_COLUMNS:
    f1_thresh.append(thresholds_data[label]["F1max"])

print("============================================================")
print("Evaluating 10 Test Cases using Optimized (F1max) Thresholds")
print("============================================================\n")

# To keep it completely random, pick 10 random indices (seed for consistency if needed)
random.seed(42)
indices = random.sample(range(len(tgts)), 10)

total_correct_all = 0
total_valid_all = 0

for i, idx in enumerate(indices):
    print(f"--- Test Case #{i+1} (Index {idx}) ---")
    correct_count = 0
    valid_count = 0
    
    for c, lab in enumerate(LABEL_COLUMNS):
        if masks[idx, c] == 1:
            valid_count += 1
            t = int(tgts[idx, c])
            p = probs[idx, c]
            
            # Using custom optimized threshold instead of 0.50
            thresh = f1_thresh[c]
            pred_binary = 1 if p >= thresh else 0
            
            if pred_binary == t:
                correct_count += 1
                
    if valid_count > 0:
        acc = correct_count / valid_count * 100
        total_correct_all += correct_count
        total_valid_all += valid_count
        print(f"  Got {correct_count} out of {valid_count} labels right! ({acc:.1f}%)")
    else:
        print(f"  No valid ground truth labels for this patient.")

print("\n============================================================")
avg_acc = (total_correct_all / total_valid_all) * 100
print(f"OVERALL PERFORMANCE ON THESE 10 CASES: {total_correct_all}/{total_valid_all} correct -> {avg_acc:.2f}% Accuracy")
print("============================================================")

