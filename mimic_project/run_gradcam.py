#!/usr/bin/env python3
import sys, os, json, hashlib
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

# Try importing google.cloud.storage to download if missing
try:
    from google.cloud import storage
    HAS_GCP = True
except ImportError:
    HAS_GCP = False

import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, r"b:\mimic_project_full_backup\mimic_project")
from src.models.densenet121 import build_densenet121

# ---- Configurations ----
CKPT_PATH = r"b:\mimic_project_full_backup\mimic_project\models\baseline_best.pt"
JSONL_PATH = r"b:\mimic_project_full_backup\mimic_project\01_predictions\test\test_ids.jsonl"
CACHE_DIR = r"b:\mimic_project_full_backup\mimic_project\results\image_only_nomask_main\cache_images\test"

LABEL_COLUMNS = [
    "Cardiomegaly", "Pleural Effusion", "Edema", "Pneumonia",
    "Atelectasis", "Pneumothorax", "Consolidation", "Support Devices"
]

SUPPORT_DEVICES_IDX = 7  # Index of "Support Devices" in LABEL_COLUMNS
TEST_IDX = 16            # The index we found earlier that had 93% confidence

# ---- 1) Find the Image Path from JSONL ----
local_img_path = r"b:\mimic_project_full_backup\mimic_project\results\image_only_nomask_main\cache_images\test\files\p10\p10268877\s50042142\4c3c1335-0fce9b11-027c582b-a0ed8d89-ca614d90.jpg"

if not os.path.exists(local_img_path):
    print("Could not find the local image.")
    sys.exit(1)

# ---- 2) Load Image & Preprocess ----
img_pil = Image.open(local_img_path).convert("RGB")
img_np = np.array(img_pil, dtype=np.uint8)

tf = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0,0,0), std=(1,1,1), max_pixel_value=255.0),
    ToTensorV2(),
])
x = tf(image=img_np)["image"].unsqueeze(0).float() # [1, 3, 512, 512]

# ---- 3) Load Model & Checkpoint ----
model = build_densenet121(num_classes=8, pretrained=False, dropout_p=0.3)
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)
cleaned = {}
for k, v in sd.items():
    k2 = k
    for p in ("module.", "model."):
        if k2.startswith(p): k2 = k2[len(p):]
    cleaned[k2] = v
model.load_state_dict(cleaned, strict=False)
model.eval()

# ---- 4) Custom Grad-CAM Implementation ----
class SimpleGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, inp, out):
        self.activations = out

    def save_gradient(self, module, g_inp, g_out):
        self.gradients = g_out[0]

    def __call__(self, x, class_idx):
        self.model.zero_grad()
        # Enable gradients for input to track backwards through the network
        x.requires_grad_(True)
        
        out = self.model(x)
        score = out[0, class_idx]
        score.backward()
        
        # Compute weights as mean gradients
        weights = torch.mean(self.gradients, dim=(2,3), keepdim=True)
        # Weighted sum of activations
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = F.relu(cam) # ReLU to keep only features that have positive influence
        
        # Normalize
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        
        return cam.detach().cpu().numpy(), torch.sigmoid(out).detach().cpu().numpy()

# The target layer is usually the last convolutional layer. 
# For DenseNet121, it is `model.features.denseblock4` to avoid inplace ReLU issues with `norm5`.
target_layer = model.features.denseblock4
grad_cam = SimpleGradCAM(model, target_layer)

print("Running Grad-CAM...")
cam_map, probs = grad_cam(x, class_idx=SUPPORT_DEVICES_IDX)

pred_prob = probs[0, SUPPORT_DEVICES_IDX]
print(f"Probability for Support Devices: {pred_prob:.4f}")

# ---- 5) Resize & Overlay Heatmap ----
import cv2
cam_resized = cv2.resize(cam_map, (img_np.shape[1], img_np.shape[0]))
heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

overlay = np.uint8(0.4 * heatmap + 0.6 * img_np)

# ---- 6) Save Result ----
out_path = r"b:\mimic_project_full_backup\mimic_project\gradcam_support_devices.jpg"
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(img_np)
ax[0].set_title("Original Image")
ax[0].axis('off')

ax[1].imshow(cam_map, cmap='jet')
ax[1].set_title(f"Grad-CAM Heatmap\n(Low Res 16x16)")
ax[1].axis('off')

ax[2].imshow(overlay)
ax[2].set_title(f"Overlay\nSupport Devices: {pred_prob*100:.1f}%")
ax[2].axis('off')

plt.tight_layout()
plt.savefig(out_path, dpi=150)
print(f"Saved Grad-CAM visualization to: {out_path}")
