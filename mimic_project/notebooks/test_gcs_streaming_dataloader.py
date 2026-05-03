# notebooks/test_gcs_streaming_dataloader.py

import os
import torch

from src.data.dataloader_cloud import create_dataloader, LABEL_COLUMNS


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    csv_path = os.path.join(project_root, "data", "processed", "processed_metadata.csv")

    print("[TEST] CSV path:", csv_path)

    loader = create_dataloader(
        csv_path=csv_path,
        split="train",
        batch_size=4,
        shuffle=False,
        image_size=512,
    )

    print("[TEST] Iterating over 2 batches...")

    for i, (images, labels) in enumerate(loader):
        print(f"[TEST] Batch {i}")
        print("  images shape:", images.shape)  # [B, 3, 512, 512]
        print("  labels shape:", labels.shape)  # [B, 8]
        print("  first labels:", labels[0])

        if i >= 1:
            break

    print("[TEST] Done.")


if __name__ == "__main__":
    main()
