# src/data/dataloader_cloud_mm.py
#
# Multimodal dataloader for MIMIC-CXR on GCS with hashed cache + per-worker GCS client.
#
# Objectives:
# - GCS streaming (gs://...) with retry on transient errors
# - Local hashed cache (default /ephemeral/ubuntu/mimic_cache or /tmp/ubuntu/mimic_cache)
# - Safe for num_workers>0: each worker initializes its own storage.Client()
# - Safe for num_workers=0: lazy init storage.Client() inside __getitem__
# - Returns: (image [3,H,W], meta7 [7], targets8 [8], mask8 [8])
#
# Key fix:
# - Cache write is collision-safe across workers (unique tmp per PID + safe replace)

import os
import time
import hashlib
from io import BytesIO
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from google.cloud import storage
from google.api_core.exceptions import ServiceUnavailable, DeadlineExceeded


# ----------------------------
# LABELS
# ----------------------------
LABEL_COLUMNS: List[str] = [
    "Cardiomegaly",
    "Pleural Effusion",
    "Edema",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Consolidation",
    "Support Devices",
]


# ----------------------------
# HELPERS
# ----------------------------
def _is_missing(x) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and np.isnan(x):
        return True
    if isinstance(x, str) and x.strip() == "":
        return True
    return False


def build_meta_vector(
    view_position: Optional[str],
    rows: Optional[float],
    cols: Optional[float],
) -> np.ndarray:
    """
    Meta vector [7]:
      [is_PA, is_AP, rows_norm, cols_norm, vp_missing, rows_missing, cols_missing]
    rows_norm/cols_norm: clipped to [0,1] via /4000.
    """
    vp_missing = 1.0 if _is_missing(view_position) else 0.0
    rows_missing = 1.0 if _is_missing(rows) else 0.0
    cols_missing = 1.0 if _is_missing(cols) else 0.0

    is_pa = 0.0
    is_ap = 0.0
    if not vp_missing:
        vp = str(view_position).upper().strip()
        if vp == "PA":
            is_pa = 1.0
        elif vp == "AP":
            is_ap = 1.0

    rows_norm = 0.0
    cols_norm = 0.0
    if not rows_missing:
        rows_norm = float(np.clip(float(rows) / 4000.0, 0.0, 1.0))
    if not cols_missing:
        cols_norm = float(np.clip(float(cols) / 4000.0, 0.0, 1.0))

    return np.array(
        [is_pa, is_ap, rows_norm, cols_norm, vp_missing, rows_missing, cols_missing],
        dtype=np.float32,
    )


def build_targets_and_mask(row: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    Masking policy (matches your baseline masking style):
      - valid labels in {0,1} => mask=1
      - missing/uncertain/other (e.g., -1) => mask=0 (ignored)
    """
    y = np.zeros((len(LABEL_COLUMNS),), dtype=np.float32)
    m = np.zeros((len(LABEL_COLUMNS),), dtype=np.float32)

    for i, col in enumerate(LABEL_COLUMNS):
        val = row.get(col, np.nan)
        if _is_missing(val):
            y[i] = 0.0
            m[i] = 0.0
        else:
            v = float(val)
            if v in (0.0, 1.0):
                y[i] = v
                m[i] = 1.0
            else:
                # -1 / uncertain / anything else => ignore
                y[i] = 0.0
                m[i] = 0.0

    return y, m


# ----------------------------
# GCS RETRY
# ----------------------------
def _download_with_retry(blob, retries: int = 3) -> bytes:
    for i in range(int(retries)):
        try:
            return blob.download_as_bytes()
        except (ServiceUnavailable, DeadlineExceeded) as e:
            if i == retries - 1:
                raise RuntimeError(f"GCS download failed after {retries} retries: {e}")
            time.sleep(2 ** i)
    raise RuntimeError("Unreachable retry loop end")


# ----------------------------
# DATASET
# ----------------------------
class MimicCXRCloudDatasetMM(Dataset):
    """
    CSV must contain:
      - split_col (default "split")
      - gcs_uri_col (default "gcs_path") containing "gs://..."
      - label columns in LABEL_COLUMNS

    Meta columns are optional; missing handled safely.
    """
    def __init__(
        self,
        csv_path: str,
        split: str,
        image_size: int = 512,
        cache_dir: str = "/ephemeral/ubuntu/mimic_cache",
        gcs_uri_col: str = "gcs_path",
        viewpos_col: str = "ViewPosition",
        rows_col: str = "Rows",
        cols_col: str = "Columns",
        split_col: str = "split",
    ):
        df = pd.read_csv(csv_path)

        required = [split_col, gcs_uri_col] + LABEL_COLUMNS
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"[MimicCXRCloudDatasetMM] Missing required columns: {missing}")

        df = df[df[split_col] == split].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"[MimicCXRCloudDatasetMM] No samples found for split='{split}'")

        self.df = df
        self.image_size = int(image_size)
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.gcs_uri_col = gcs_uri_col
        self.viewpos_col = viewpos_col
        self.rows_col = rows_col
        self.cols_col = cols_col
        self.split_col = split_col

        self._client = None  # set per-worker via worker_init_fn OR lazily for num_workers=0

        # Normalize(mean=0,std=1,max_pixel_value=255) => output approx img/255 in [0,1]
        self.tf = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
            ToTensorV2(),
        ])

    def __len__(self) -> int:
        return len(self.df)

    def _parse_gs_uri(self, uri: str) -> Tuple[str, str]:
        assert uri.startswith("gs://"), f"Invalid GCS URI: {uri}"
        uri = uri[5:]
        bucket, blob = uri.split("/", 1)
        return bucket, blob

    def _cache_path(self, gcs_uri: str) -> str:
        h = hashlib.sha1(gcs_uri.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{h}.jpg")

    def _load_image_rgb(self, gcs_uri: str) -> Image.Image:
        cpath = self._cache_path(gcs_uri)

        # Fast path: cached
        if os.path.exists(cpath):
            with Image.open(cpath) as im:
                return im.convert("RGB").copy()

        # init client
        if self._client is None:
            self._client = storage.Client()

        bucket_name, blob_name = self._parse_gs_uri(gcs_uri)
        blob = self._client.bucket(bucket_name).blob(blob_name)
        data = _download_with_retry(blob)

        # collision-safe atomic write across workers
        tmp = cpath + f".{os.getpid()}.tmp"
        try:
            os.makedirs(self.cache_dir, exist_ok=True)

            with open(tmp, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())

            # If another worker already wrote final, do not crash
            if not os.path.exists(cpath):
                os.replace(tmp, cpath)
            else:
                try:
                    os.remove(tmp)
                except FileNotFoundError:
                    pass

        except Exception:
            # best-effort cleanup
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
            raise

        # Decode from bytes (safe even if cache write lost a race)
        with Image.open(BytesIO(data)) as im:
            return im.convert("RGB").copy()

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        gcs_uri = row[self.gcs_uri_col]

        img = self._load_image_rgb(gcs_uri)
        img_np = np.array(img, dtype=np.uint8)  # [H,W,3]

        # Defensive: handle grayscale slips
        if img_np.ndim == 2:
            img_np = np.stack([img_np, img_np, img_np], axis=-1)
        elif img_np.ndim == 3 and img_np.shape[2] == 1:
            img_np = np.repeat(img_np, 3, axis=2)

        x = self.tf(image=img_np)["image"].float()  # [3,H,W]

        meta = build_meta_vector(
            row.get(self.viewpos_col, None),
            row.get(self.rows_col, None),
            row.get(self.cols_col, None),
        )

        y, m = build_targets_and_mask(row)

        return x, torch.from_numpy(meta).float(), torch.from_numpy(y), torch.from_numpy(m)


# ----------------------------
# DATALOADER
# ----------------------------
def worker_init_fn_mm(worker_id: int):
    """
    Per-worker GCS client init (no cross-process sharing).
    Safe with persistent_workers=True (client persists for worker lifetime).
    """
    info = torch.utils.data.get_worker_info()
    if info is not None:
        ds = info.dataset
        ds._client = storage.Client()


def create_dataloader_mm(
    csv_path: str,
    split: str,
    batch_size: int,
    shuffle: bool,
    image_size: int = 512,
    cache_dir: str = "/ephemeral/ubuntu/mimic_cache",
    num_workers: int = 10,
    split_col: str = "split",
    pin_memory: bool = True,
) -> DataLoader:
    ds = MimicCXRCloudDatasetMM(
        csv_path=csv_path,
        split=split,
        image_size=image_size,
        cache_dir=cache_dir,
        split_col=split_col,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(split == "train"),
        persistent_workers=(num_workers > 0),
        worker_init_fn=worker_init_fn_mm,
    )
