import os, time, hashlib
from io import BytesIO
from typing import List

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from google.cloud import storage


LABEL_COLUMNS = [
    "Cardiomegaly","Pleural Effusion","Edema","Pneumonia",
    "Atelectasis","Pneumothorax","Consolidation","Support Devices",
]


def _is_missing(x):
    return x is None or (isinstance(x,float) and np.isnan(x)) or (isinstance(x,str) and x.strip()=="")

def _safe_upper(x):
    return "" if _is_missing(x) else str(x).upper().strip()

def _studytime_to_hour(t):
    try:
        s = str(int(float(t))).zfill(6)
        return int(s[:2])
    except:
        return 12


def build_meta_vector(row: pd.Series) -> np.ndarray:
    vp  = _safe_upper(row.get("ViewPosition"))
    ori = _safe_upper(row.get("PatientOrientation"))

    blob = " ".join([
        _safe_upper(row.get("ViewCode")),
        _safe_upper(row.get("ProcedureCode")),
        _safe_upper(row.get("PerformedProcedureStepDescription"))
    ])

    is_pa = 1.0 if vp=="PA" else 0.0
    is_ap = 1.0 if vp=="AP" else 0.0
    is_erect  = 1.0 if "ERECT"  in ori else 0.0
    is_supine = 1.0 if "SUPINE" in ori else 0.0
    is_portable = 1.0 if ("PORT" in blob or "BEDSIDE" in blob) else 0.0

    hour = _studytime_to_hour(row.get("StudyTime"))
    is_night = 1.0 if (hour>=20 or hour<=5) else 0.0

    rows = row.get("Rows"); cols = row.get("Columns")
    rows_norm = 0.0 if _is_missing(rows) else np.clip(float(rows)/4000,0,1)
    cols_norm = 0.0 if _is_missing(cols) else np.clip(float(cols)/4000,0,1)
    aspect = np.clip((float(rows)/float(cols)) if (not _is_missing(rows) and not _is_missing(cols) and cols>0) else 0.0,0.5,2.0)

    vp_missing  = 1.0 if _is_missing(row.get("ViewPosition")) else 0.0
    ori_missing = 1.0 if _is_missing(row.get("PatientOrientation")) else 0.0

    return np.array([
        is_pa,is_ap,is_erect,is_supine,is_portable,is_night,
        rows_norm,cols_norm,aspect,vp_missing,ori_missing
    ],dtype=np.float32)


def build_targets_and_mask(row):
    y = np.zeros(8,np.float32); m = np.zeros(8,np.float32)
    for i,col in enumerate(LABEL_COLUMNS):
        v = row.get(col,np.nan)
        if not _is_missing(v) and float(v) in (0,1):
            y[i]=float(v); m[i]=1.0
    return y,m


class MimicCXRCloudDatasetMM(Dataset):
    def __init__(self,csv_path,split,image_size=512,cache_dir="/ephemeral/ubuntu/mimic_cache",split_col="split"):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df[split_col]==split].reset_index(drop=True)
        self.cache_dir = cache_dir; os.makedirs(cache_dir,exist_ok=True)
        self._client=None

        self.tf = A.Compose([
            A.Resize(image_size,image_size),
            A.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225),max_pixel_value=255),
            ToTensorV2(),
        ])

    def __len__(self): return len(self.df)

    def _cache_path(self,uri):
        return os.path.join(self.cache_dir,hashlib.sha1(uri.encode()).hexdigest()+".jpg")

    def __getitem__(self,idx):
        row = self.df.iloc[idx]
        uri = row["gcs_path"]
        cpath = self._cache_path(uri)

        if not os.path.exists(cpath):
            if self._client is None: self._client=storage.Client()
            b,n = uri[5:].split("/",1)
            data = self._client.bucket(b).blob(n).download_as_bytes()
            with open(cpath,"wb") as f: f.write(data)

        img = Image.open(cpath).convert("RGB")
        x = self.tf(image=np.array(img))["image"]
        meta = build_meta_vector(row)
        y,m = build_targets_and_mask(row)
        return x,torch.tensor(meta),torch.tensor(y),torch.tensor(m)


def create_dataloader_mm(csv_path,split,batch_size,shuffle,image_size=512,cache_dir="/ephemeral/ubuntu/mimic_cache",num_workers=10,split_col="split"):
    ds = MimicCXRCloudDatasetMM(csv_path,split,image_size,cache_dir,split_col)
    return DataLoader(ds,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=True,drop_last=(split=="train"))
