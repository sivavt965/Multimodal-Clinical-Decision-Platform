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

# Find an index where the model predicts at least one pathology with > 80% confidence
# and the ground truth also agrees (True Positive)
found = False
for i in range(len(probs)):
    for c, lab in enumerate(LABEL_COLUMNS):
        if masks[i, c] == 1 and tgts[i, c] == 1 and probs[i, c] > 0.80:
            print("="*60)
            print(f">>> Found a strong POSITIVE prediction at index {i}! <<<")
            print("="*60)
            print(f"{'Label':<20} {'Prob':>8}   {'GroundTruth':>12}")
            print("-" * 45)
            for j, l in enumerate(LABEL_COLUMNS):
                p = probs[i, j]
                t = int(tgts[i, j]) if masks[i, j] == 1 else "Unknown"
                
                # Highlight the predicted YES
                marker = "--> YES" if p >= 0.50 else "    NO"
                print(f"{l:<20} {p:>8.4f}    {str(t):>12}  {marker}")
            
            print("="*60)
            found = True
            break
    if found:
        break

if not found:
    print("Could not find a strong positive prediction in the first sweep.")
