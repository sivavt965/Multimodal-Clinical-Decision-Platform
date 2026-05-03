import os
import pandas as pd
from google.cloud import storage
from PIL import Image

PROJECT_ROOT = r"e:\mimic_project"
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "processed_metadata.csv")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# 1. Read metadata and take one example row
df = pd.read_csv(CSV_PATH)
print("Total rows in CSV:", len(df))

row = df.iloc[0]
gcs_uri = row["gcs_path"]
print("Example GCS URI:", gcs_uri)

# 2. Prepare local cache path
local_path = os.path.join(CACHE_DIR, "single_test.jpg")

# 3. Parse GCS URI: gs://bucket/path
if not gcs_uri.startswith("gs://"):
    raise ValueError(f"Not a GCS URI: {gcs_uri}")

no_scheme = gcs_uri[5:]
bucket_name, blob_path = no_scheme.split("/", 1)

print("Bucket   :", bucket_name)
print("Blob path:", blob_path)
print("Downloading to:", local_path)

# 4. Download from GCS using google-cloud-storage
client = storage.Client()
bucket = client.bucket(bucket_name)
blob = bucket.blob(blob_path)
blob.download_to_filename(local_path)

print("Download done.")

# 5. Open image with PIL and print size
img = Image.open(local_path).convert("RGB")
print("Image mode:", img.mode)
print("Image size:", img.size)  # (width, height)

# (Optional) save a resized copy to check easily
thumb_path = os.path.join(CACHE_DIR, "single_test_512.png")
img_resized = img.resize((512, 512))
img_resized.save(thumb_path)
print("Saved resized copy to:", thumb_path)
