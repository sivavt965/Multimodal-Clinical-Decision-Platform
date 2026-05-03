#!/usr/bin/env python3
"""
Quick test run for baseline_best.pt
- Loads the checkpoint
- Inspects checkpoint keys
- Loads weights into DenseNet121 (8 classes)
- Runs a forward pass with a dummy batch (random tensors)
- Prints output logits and probabilities
"""

import sys, os
sys.path.insert(0, r"b:\mimic_project_full_backup\mimic_project")

import torch
import torch.nn.functional as F
import numpy as np

CKPT_PATH = r"b:\mimic_project_full_backup\mimic_project\models\baseline_best.pt"

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

print("=" * 60)
print("STEP 1: Inspecting checkpoint")
print("=" * 60)
ckpt = torch.load(CKPT_PATH, map_location="cpu")
print(f"  Checkpoint type : {type(ckpt)}")
if isinstance(ckpt, dict):
    print(f"  Top-level keys  : {list(ckpt.keys())}")
    for k, v in ckpt.items():
        if isinstance(v, dict):
            print(f"    '{k}' is a dict with {len(v)} entries")
        elif isinstance(v, torch.Tensor):
            print(f"    '{k}' is a Tensor of shape {tuple(v.shape)}")
        else:
            print(f"    '{k}' = {v}")

print()
print("=" * 60)
print("STEP 2: Loading model weights")
print("=" * 60)

from src.models.densenet121 import build_densenet121

model = build_densenet121(num_classes=8, pretrained=False, dropout_p=0.3)

if isinstance(ckpt, dict):
    if "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif "model" in ckpt and isinstance(ckpt["model"], dict):
        sd = ckpt["model"]
    else:
        sd = ckpt
else:
    sd = ckpt

# Strip common prefixes
cleaned = {}
for k, v in sd.items():
    k2 = k
    for prefix in ("module.", "model."):
        if k2.startswith(prefix):
            k2 = k2[len(prefix):]
    cleaned[k2] = v

missing, unexpected = model.load_state_dict(cleaned, strict=False)
print(f"  Missing keys    : {len(missing)}")
print(f"  Unexpected keys : {len(unexpected)}")
if missing:
    print(f"    [WARN] First 5 missing: {missing[:5]}")
if unexpected:
    print(f"    [WARN] First 5 unexpected: {unexpected[:5]}")
print("  Weights loaded successfully")

print()
print("=" * 60)
print("STEP 3: Forward pass (dummy batch of 4)")
print("=" * 60)

model.eval()
device = torch.device("cpu")
model = model.to(device)

batch_size = 4
x = torch.randn(batch_size, 3, 512, 512)  # [B, 3, H, W]

with torch.no_grad():
    logits = model(x)  # [B, 8]
    probs = torch.sigmoid(logits)

print(f"  Input shape     : {tuple(x.shape)}")
print(f"  Logits shape    : {tuple(logits.shape)}")
print(f"  Probs shape     : {tuple(probs.shape)}")

print()
print("=" * 60)
print("STEP 4: Per-label probability output (sample 0)")
print("=" * 60)

probs_np = probs.numpy()
print(f"  {'Label':<22} {'Prob':>8}")
print("  " + "-" * 32)
for i, lab in enumerate(LABEL_COLUMNS):
    print(f"  {lab:<22} {probs_np[0, i]:.4f}")

print()
print("=" * 60)
print("STEP 5: Stats across all samples in batch")
print("=" * 60)
print(f"  Mean prob : {probs_np.mean():.4f}")
print(f"  Std  prob : {probs_np.std():.4f}")
print(f"  Min  prob : {probs_np.min():.4f}")
print(f"  Max  prob : {probs_np.max():.4f}")
print()
print("TEST RUN COMPLETE - baseline_best.pt is valid and usable!")
