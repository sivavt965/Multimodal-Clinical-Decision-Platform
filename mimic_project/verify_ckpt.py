"""Smoke-test a DenseNet121 baseline checkpoint.

This script intentionally targets the maintained image-only baseline. It checks
that a checkpoint can be loaded into the model class used by training and
inference, then runs a tiny forward pass on random input.
"""

from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.densenet121 import build_densenet121


DEFAULT_CKPT = PROJECT_ROOT / "models" / "baseline_best.pt"


def _state_dict_from_checkpoint(checkpoint: object) -> dict[str, torch.Tensor]:
    """Handle both wrapped training checkpoints and raw state-dict exports."""
    if isinstance(checkpoint, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint  # type: ignore[return-value]
    raise ValueError("Unsupported checkpoint format")


def main() -> None:
    ckpt_path = DEFAULT_CKPT
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. "
            "Pass a baseline checkpoint into mimic_project/models first."
        )

    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        print("Keys:", sorted(checkpoint.keys()))
        print("Epoch:", checkpoint.get("epoch"))
        print("Best AUC:", checkpoint.get("best_auc"))

    model = build_densenet121(num_classes=8, pretrained=False, dropout=0.3)
    model.load_state_dict(_state_dict_from_checkpoint(checkpoint), strict=True)
    model.eval()

    x_img = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        y = model(x_img)

    print("Forward pass OK. Output shape:", tuple(y.shape))
    print("Checkpoint verified.")


if __name__ == "__main__":
    main()
