import pandas as pd
import json
import os

def main():
    val_csv = "val.csv"
    meta_csv = os.path.join("mimic_project", "data", "processed", "processed_metadata.csv")
    
    df_val = pd.read_csv(val_csv)
    df_meta = pd.read_csv(meta_csv, low_memory=False)
    
    # Extract dicom_id from cxr_path to merge
    # cxr_path example: files/p15/p15509168/s53396022/b715591b-1d9ca5a8-11b2f7d5-70c2734b-674972a3.jpg
    def extract_dicom(path):
        if pd.isna(path): return ""
        basename = os.path.basename(path)
        return basename.split(".")[0]
        
    df_val["dicom_id"] = df_val["cxr_path"].apply(extract_dicom)
    
    # Merge on dicom_id
    df_merged = pd.merge(df_val, df_meta, on="dicom_id", suffixes=("_val", "_meta"))
    
    print(f"Found {len(df_merged)} overlapping records.")
    
    # Take a subset
    subset = df_merged.head(10).to_dict(orient="records")
    
    # Format for JSON
    demo_json = []
    for row in subset:
        # Extract lab values that are in LAB_NORMAL_RANGES
        labs = {k: row[k] for k in row.keys() if k.isdigit() and pd.notna(row[k])}
        
        demo_json.append({
            "subject_id": str(row["subject_id_val"]),
            "hadm_id": str(row["hadm_id"]),
            "study_id": str(row["study_id"]),
            "dicom_id": row["dicom_id"],
            "cxr_path": row["cxr_path"],
            "ecg_path": row["ecg_path"],
            "labs": labs,
            "labels": {
                "Atelectasis": row.get("Atelectasis_val", None),
                "Cardiomegaly": row.get("Cardiomegaly_val", None),
                "Edema": row.get("Edema_val", None),
                "Pleural Effusion": row.get("Pleural Effusion_val", None),
            }
        })
        
    with open("demo_form_values.json", "w") as f:
        json.dump(demo_json, f, indent=2)
        
    # Also save to demo_cases.csv
    df_merged.head(50).to_csv("demo_cases.csv", index=False)
    print("Saved demo_form_values.json and demo_cases.csv")

if __name__ == "__main__":
    main()
