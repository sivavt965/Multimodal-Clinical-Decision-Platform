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
    
    correct_count = 0
    total_valid = 0
    
    for c, l in enumerate(LABEL_COLUMNS):
        if masks[idx, c] == 1:
            total_valid += 1
            p = probs[idx, c]
            t = int(tgts[idx, c])
            
            pred_binary = 1 if p >= 0.50 else 0
            
            marker = "--> YES" if pred_binary == 1 else "    NO"
            
            if pred_binary == t:
                correct_count += 1
                marker += " (Correct)"
            else:
                if t == 1:
                    marker += " (False Negative)"
                else:
                    marker += " (False Positive)"
                    
            print(f"{l:<20} {p:>8.4f}    {str(t):>12}  {marker}")
            
    print("-" * 45)
    if total_valid > 0:
        print(f"Total Correct: {correct_count}/{total_valid} ({correct_count/total_valid*100:.1f}%)")
    print()

# Look for cases with 80% correct (6/8 or 7/8). 
# Also ensure the patient has at least one positive ground truth label (1), 
# so it's a non-trivial case.

found = 0
for i in range(len(tgts)):
    total_valid = np.sum(masks[i] == 1)
    if total_valid >= 4:  # At least half the labels should be valid
        # How many did the model get right?
        correct = 0
        positive_gt_count = 0
        for c in range(8):
            if masks[i, c] == 1:
                t = int(tgts[i, c])
                p_bin = 1 if probs[i, c] >= 0.50 else 0
                if t == p_bin:
                    correct += 1
                if t == 1:
                    positive_gt_count += 1
        
        # Calculate accuracy for this patient
        acc = correct / total_valid
        
        # Check for ~80% accuracy (e.g., 6/8=75%, 7/8=87.5%)
        # and at least one real finding
        if 0.70 < acc < 0.90 and positive_gt_count >= 1:
            print_case(i, f"Case where model is ~{acc*100:.0f}% correct")
            found += 1
            if found >= 2:  # Found 2 examples, that's enough
                break

if found == 0:
    print("Could not find any ~80% cases meeting the criteria.")
