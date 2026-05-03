import os
import io
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from google.cloud import storage

import albumentations as A
from albumentations.pytorch import ToTensorV2


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


def parse_gs_uri(gs_uri: str) -> Tuple[str, str]:
    if not gs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI (must start with gs://): {gs_uri}")
    parts = gs_uri[5:].split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid GCS URI: {gs_uri}")
    return parts[0], parts[1]


class MimicCXRCloudDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        split: str,
        image_size: int = 512,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
    ) -> None:
        df = pd.read_csv(csv_path)

        if "split" not in df.columns:
            raise ValueError("CSV must contain 'split' column")
        if "gcs_path" not in df.columns:
            raise ValueError("CSV must contain 'gcs_path' column")

        if split not in {"train", "validate", "test"}:
            raise ValueError(f"Invalid split: {split}")

        df = df[df["split"] == split].reset_index(drop=True)

        self.df = df
        self.labels = LABEL_COLUMNS
        self.split = split
        self.image_size = image_size

        self.use_cache = bool(use_cache)
        self.cache_dir = cache_dir if (cache_dir and self.use_cache) else None
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        # GCS client must be created per worker
        self._client = None

        # ✅ ImageNet normalization (because your model uses IMAGENET1K_V1 weights)
        norm_mean = (0.485, 0.456, 0.406)
        norm_std = (0.229, 0.224, 0.225)

        if split == "train":
            self.transform = A.Compose(
                [
                    A.Resize(image_size, image_size),
                    A.Rotate(limit=3, p=0.5),
                    A.Affine(
                        translate_percent=(0.0, 0.02),
                        scale=(0.98, 1.02),
                        rotate=0,
                        p=0.5,
                    ),
                    A.HorizontalFlip(p=0.5),
                    A.Normalize(mean=norm_mean, std=norm_std),
                    ToTensorV2(),
                ]
            )
        else:
            self.transform = A.Compose(
                [
                    A.Resize(image_size, image_size),
                    A.Normalize(mean=norm_mean, std=norm_std),
                    ToTensorV2(),
                ]
            )

        print(
            f"[MimicCXRCloudDataset] split={split}, rows={len(self.df)}, "
            f"cache={'ON' if self.cache_dir else 'OFF'}"
        )

    def _get_client(self):
        if self._client is None:
            self._client = storage.Client()
        return self._client

    def _local_cache_path(self, gcs_uri: str) -> str:
        _, blob_path = parse_gs_uri(gcs_uri)
        return os.path.join(self.cache_dir, blob_path)

    def _load_image_from_local(self, path: str) -> Image.Image:
        with Image.open(path) as img:
            return img.convert("RGB")

    def _download_image_from_gcs(self, gcs_uri: str) -> Image.Image:
        bucket_name, blob_path = parse_gs_uri(gcs_uri)
        client = self._get_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        image_bytes = blob.download_as_bytes()
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    def _load_image(self, gcs_uri: str) -> Image.Image:
        if self.cache_dir:
            local_path = self._local_cache_path(gcs_uri)
            if os.path.exists(local_path):
                return self._load_image_from_local(local_path)

            img = self._download_image_from_gcs(gcs_uri)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            tmp_path = local_path + ".tmp"
            try:
                img.save(tmp_path, format="JPEG", quality=95)
                os.replace(tmp_path, local_path)
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
            return img

        return self._download_image_from_gcs(gcs_uri)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        img = self._load_image(row["gcs_path"])
        img_np = np.asarray(img)

        transformed = self.transform(image=img_np)
        img_tensor = transformed["image"]

        label_values = row[self.labels].values.astype("float32")
        # optional safety: replace NaN with -1 (masked loss can ignore if you later choose)
        if np.isnan(label_values).any():
            label_values = np.nan_to_num(label_values, nan=-1.0)

        labels = torch.tensor(label_values, dtype=torch.float32)
        return img_tensor, labels


def create_dataloader(
    csv_path: str,
    split: str,
    batch_size: int,
    image_size: int = 512,
    cache_dir: Optional[str] = None,
    use_cache: bool = True,
    num_workers: int = 10,
) -> DataLoader:
    dataset = MimicCXRCloudDataset(
        csv_path=csv_path,
        split=split,
        image_size=image_size,
        cache_dir=cache_dir,
        use_cache=use_cache,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
    )
    return loader
