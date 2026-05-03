import torch
from src.models.model_mm_film_gated import MultiModalFiLMGated

ckpt_path = r"C:\Users\siva\mimic_project\models\mm_film_best.pt"

print("Loading checkpoint...")
ckpt = torch.load(ckpt_path, map_location="cpu")

print("Keys:", ckpt.keys())
print("Epoch:", ckpt.get("epoch"))
print("Best AUC:", ckpt.get("best_auc"))

print("\nLoading model...")
model = MultiModalFiLMGated(num_labels=8)
model.load_state_dict(ckpt["model"], strict=True)
model.eval()

x_img = torch.randn(2, 1, 512, 512)
x_meta = torch.randn(2, 7)

with torch.no_grad():
    y = model(x_img, x_meta)

print("Forward pass OK. Output shape:", y.shape)
print("\n✅ CHECKPOINT VERIFIED — NOT CORRUPTED")
