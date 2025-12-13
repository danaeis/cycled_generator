import torch
from pathlib import Path

ckpt_path = Path("autoencoder_pure_mse_run_lr6/checkpoint_epoch_35.pth")
print("File exists:", ckpt_path.exists())
print("File size:", ckpt_path.stat().st_size / 1024 / 1024, "MB")

# Try to load and inspect
try:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    print("Keys in checkpoint:", list(ckpt.keys()))
    print("Full content preview:")
    for k, v in ckpt.items():
        if isinstance(v, dict):
            print(f"  {k}: dict with keys {list(v.keys())[:5]}...")
        else:
            print(f"  {k}: {type(v)}")
except Exception as e:
    print("FAILED TO LOAD:", e)