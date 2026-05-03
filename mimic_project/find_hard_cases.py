#!/usr/bin/env python3
import numpy as np

# Load pre-computed test results
prob_path = r"b:\mimic_project_full_backup\mimic_project\01_predictions\test\test_probs.npy"
tgt_path = r"b:\mimic_project_full_backup\mimic_project\01_predictions\test\test_targets.npy"
mask_path = r"b:\mimic_project_full_backup\mimic_project\01_predictions\test\test_label_valid_mask.npy"

try:
    probs = np.load(prob_path)
    tgts = np.load(tgt_path)
    masks = np.load(mask_path)
except Exception as e:
    print(f"Error loading numpy files: {e}")
    exit(1)

LABEL_COLUMNS = [
    "Cardiomegaly", "Pleural Effusion", "Edema", "Pneumonia",
    "Atelectasis", "Pneumothorax", "Consolidation", "Support Devices"
]

def print_case(idx, title):
    print("="*60)
    print(f">>> {title} (Index {idx}) <<<")
    print("="*60)
    print(f"{'Label':<20} {'Prob':>8}   {'GroundTruth':>12}")
    print("-" * 45)
    for c, l in enumerate(LABEL_COLUMNS):
        p = probs[idx, c]
        t = int(tgts[idx, c]) if masks[idx, c] == 1 else "Unknown"
        marker = "--> YES" if p >= 0.50 else "    NO"
        
        # Highlight errors
        if isinstance(t, int):
            if t == 1 and p < 0.50:
                marker += " (False Negative)"
            elif t == 0 and p >= 0.50:
                marker += " (False Positive)"
                
        print(f"{l:<20} {p:>8.4f}    {str(t):>12}  {marker}")
    print()

# 1. Find a case with exactly two 1s in ground truth
found_2_1s = False
for i in range(len(tgts)):
    # count valid 1s
    gt = []
    for c in range(len(LABEL_COLUMNS)):
        if masks[i, c] == 1:
            gt.append(int(tgts[i, c]))
    
    if sum(gt) == 2:
        print_case(i, "Case with exactly two positive (1) labels")
        found_2_1s = True
        break

if not found_2_1s:
    print("Could not find a case with exactly two 1s.")

# 2. Find a hard case (False Negative): Ground truth is 1, but prediction is < 0.10
found_fn = False
for i in range(len(tgts)):
    for c in range(len(LABEL_COLUMNS)):
        if masks[i, c] == 1 and tgts[i, c] == 1 and probs[i, c] < 0.10:
            print_case(i, f"Hard Case: False Negative on {LABEL_COLUMNS[c]}")
            found_fn = True
            break
    if found_fn:
        break

# 3. Find a hard case (False Positive): Ground truth is 0, but prediction is > 0.90
found_fp = False
for i in range(len(tgts)):
    for c in range(len(LABEL_COLUMNS)):
        if masks[i, c] == 1 and tgts[i, c] == 0 and probs[i, c] > 0.90:
            print_case(i, f"Hard Case: False Positive on {LABEL_COLUMNS[c]}")
            found_fp = True
            break
    if found_fp:
        break

