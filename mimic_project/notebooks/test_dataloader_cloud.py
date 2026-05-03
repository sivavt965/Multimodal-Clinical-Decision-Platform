import os
import sys

# ---------------------------------------------------------
# Make sure Python can see the src/ package
# ---------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.dataloader_cloud import create_dataloader  # noqa: E402

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "processed_metadata.csv")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")


def main():
    # Create a small train DataLoader just to test
    train_loader = create_dataloader(
        csv_path=CSV_PATH,
        split="train",
        cache_dir=CACHE_DIR,
        batch_size=4,
        shuffle=True,
        num_workers=0,   # important on Windows
        limit=8,         # only first 8 samples for a quick test
    )

    for batch_idx, (images, labels) in enumerate(train_loader):
        print("Batch idx:", batch_idx)
        print("Images shape:", images.shape)   # [B, 3, 512, 512]
        print("Labels shape:", labels.shape)   # [B, 8]
        print("First labels:", labels[0])
        if batch_idx >= 1:
            break


if __name__ == "__main__":
    main()
