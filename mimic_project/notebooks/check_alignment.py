import pandas as pd

df = pd.read_csv("data/processed/processed_metadata.csv")

# dicom_id should be unique
assert df["dicom_id"].is_unique

# each dicom_id must belong to exactly one split
assert df.groupby("dicom_id")["split"].nunique().max() == 1

print("✅ No image-level leakage detected")
