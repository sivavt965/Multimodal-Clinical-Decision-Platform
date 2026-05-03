#!/usr/bin/env python3
import os
import pandas as pd

# We need a local image that has Support Devices = 1
CACHE_DIR = r"b:\mimic_project_full_backup\mimic_project\results\image_only_nomask_main\cache_images\test"
CSV_PATH = r"b:\mimic_project_full_backup\mimic_project\data\processed\processed_metadata.csv"

print("Loading CSV...")
df = pd.read_csv(CSV_PATH)

print("Scanning local cache for images...")
local_images = []
for root, dirs, files in os.walk(CACHE_DIR):
    for f in files:
        if f.endswith(".jpg"):
            local_images.append(os.path.join(root, f))

print(f"Found {len(local_images)} local images in cache.")

# Check if any of these local images have Support Devices == 1
found = False
for img_path in local_images:
    dicom_id = os.path.splitext(os.path.basename(img_path))[0]
    row = df[df["dicom_id"] == dicom_id]
    if len(row) > 0:
        sd_val = row.iloc[0].get("Support Devices", float("nan"))
        if not pd.isna(sd_val) and float(sd_val) == 1.0:
            print(f">>> Found local positive Support Devices case!")
            print(f"DICOM ID : {dicom_id}")
            print(f"Path     : {img_path}")
            found = True
            break

if not found:
    print("Could not find any local images with Support Devices = 1.")
