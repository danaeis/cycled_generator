# training_autoencoder_final.py
# Pure autoencoder training — no GAN, no conditionals, no garbage.
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast
from torch.amp import GradScaler
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import os
import json
import yaml
import random
from datetime import datetime
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim_sk

import nibabel as nib

# Import your existing optimized dataset and data splitter
from training_phase_gen_slice import CTPhaseDataset, VolumeCache
from dataloader_train_slice import create_data_pairs, run_full_training
from config import train_config  # your config file
from training_phase_gen_slice import Generator2D as Generator

# dataloader_autoencoder.py
from sklearn.model_selection import train_test_split
from typing import Dict, List


logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# =============================================
# 1. CONFIG (override only what we need)
# =============================================
cfg = train_config.copy()
cfg.update({
    'use_autoencoder': True,
    'use_discriminator': False,
    'use_cycle_consistency': False,
    'use_phase_conditioning': False,
    'batch_size': 8,
    'epochs': 100,
    # 'lr': 1e-6,
    'lr': 6e-7,
    # 'lr': 1e-4,
    'save_samples_interval': 1,   # save every epoch
    # 'output_dir': Path("./autoencoder_pure_mse_run_lr6"),
    # 'output_dir': Path("./autoencoder_pure_mse_run_lr5"),
    # 'output_dir': Path("./autoencoder_pure_mse_run_lr4"),
    # 'output_dir': Path("./autoencoder_pure_mse_run_lr4_aug"),
    'keep_last_n_checkpoints': 3,
    'use_augmentation': False,

    'use_cosine': True,
    'cosine_t0': 10,
    'cosine_tmult': 2,
    'cosine_eta_min': 5e-7,
})
cfg['output_dir'].mkdir(parents=True, exist_ok=True)
samples_dir = cfg['output_dir'] / "val_samples"
samples_dir.mkdir(exist_ok=True)

# Save config (both YAML and JSON for maximum compatibility)
config_yaml_path = cfg['output_dir'] / "config.yaml"
config_json_path = cfg['output_dir'] / "config.json"

# Convert Path objects to strings for serialization
cfg_serializable = {k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()}

with open(config_yaml_path, 'w') as f:
    yaml.dump(cfg_serializable, f, default_flow_style=False, indent=2)
with open(config_json_path, 'w') as f:
    json.dump(cfg_serializable, f, indent=2)

logger.info(f"Training config saved to {config_yaml_path} and {config_json_path}")

# Optional: also save a timestamped copy
run_info = {
    "run_started": datetime.now().isoformat(),
    "hostname": __import__('socket').gethostname(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
}
with open(cfg['output_dir'] / "run_info.json", 'w') as f:
    json.dump(run_info, f, indent=2)

history_path = cfg['output_dir'] / "training_history.json"
plot_loss_path = cfg['output_dir'] / "loss_curve.png"
plot_metrics_path = cfg['output_dir'] / "val_metrics.png"
plot_lr_path = cfg['output_dir'] / "lr_schedule.png"  # NEW

# Load existing history if resuming
if history_path.exists():
    import json
    with open(history_path, 'r') as f:
        history = json.load(f)
    logger.info(f"Loaded training history: {len(history['epoch'])} epochs")
else:
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_psnr': [],
        'val_ssim': [],
        'lr': []
    }

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, val_loss, path):
    torch.save({
        'epoch': epoch + 1,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'val_loss': val_loss,
    }, path)
    logger.info(f"Checkpoint saved: {path.name}")

def load_latest_checkpoint(model, optimizer, scheduler, scaler, output_dir):
    ckpts = list(output_dir.glob("checkpoint_epoch_*.pth"))
    if not ckpts:
        return 0, float('inf')
    
    ckpts.sort(key=lambda x: int(x.stem.split('_')[-1]))
    latest = ckpts[-1]
    
    try:
        checkpoint = torch.load(latest, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        # === CRITICAL FIX: Scheduler state is OPTIONAL ===
        if 'scheduler' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler'])
                logger.info(f"Loaded scheduler state from checkpoint")
            except Exception as e:
                logger.warning(f"Scheduler state exists but failed to load: {e}")
                logger.info("→ Starting fresh CosineAnnealingWarmRestarts scheduler")
        else:
            logger.info(f"No scheduler state in checkpoint — initializing fresh CosineAnnealingWarmRestarts")
            # Do nothing — we'll just start a new cosine schedule from current epoch

        start_epoch = checkpoint['epoch']
        best_val = checkpoint['val_loss']
        logger.info(f"Resumed from {latest.name} (epoch {start_epoch}) | Val loss: {best_val:.6f}")
        return start_epoch, best_val
    except Exception as e:
        logger.warning(f"Failed to load checkpoint {latest}: {e}")
        return 0, float('inf')

def cleanup_old_checkpoints(output_dir, keep=3):
    ckpts = sorted(output_dir.glob("checkpoint_epoch_*.pth"))
    for old in ckpts[:-keep]:
        old.unlink()
        logger.info(f"Removed old checkpoint: {old.name}")


def create_data_pairs_autoencoder(
    config: dict,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
    ) -> Dict[str, List[Dict]]:
    """
    Create train/val/test splits for pure autoencoder training.
    Every volume is used as both source and target (self-reconstruction).
    No phase logic. No pairing. No filtering by target_phase.
    """
    data_dir = Path(config['data_dir'])
    all_volumes = []

    logger.info("Scanning for registered volumes (*_registered.nii.gz)...")
    
    for case_dir in sorted(data_dir.iterdir()):
        if not case_dir.is_dir():
            continue
            
        case_id = case_dir.name
        
        # Find all registered-registered image volumes (skip seg)
        for nii_file in case_dir.glob("*_registered.nii.gz"):
            if "_seg" in nii_file.name:
                continue
                
            # Optional: extract original phase from filename for logging only
            phase_hint = "unknown"
            name = nii_file.stem.replace("_registered", "")
            if "non" in name.lower() or "nc" in name.lower():
                phase_hint = "non-contrast"
            elif "art" in name.lower():
                phase_hint = "arterial"
            elif "port" in name.lower() or "pv" in name.lower():
                phase_hint = "portal"
            elif "ven" in name.lower() or "delay" in name.lower():
                phase_hint = "delayed"

            all_volumes.append({
                'path': str(nii_file),
                'case_id': case_id,
                'phase_hint': phase_hint,
                'case_dir': str(case_dir)
            })

    logger.info(f"Found {len(all_volumes)} registered volumes across all phases")

    if len(all_volumes) == 0:
        raise ValueError("No *_registered.nii.gz files found! Check your data_dir.")

    # Extract unique case IDs to split by patient (avoid leakage)
    case_to_volumes = {}
    for vol in all_volumes:
        case_to_volumes.setdefault(vol['case_id'], []).append(vol)

    case_ids = list(case_to_volumes.keys())
    
    # Split cases (not volumes) → prevents data leakage
    train_cases, temp_cases = train_test_split(
        case_ids, test_size=test_size + val_size, random_state=random_state, shuffle=True
    )
    val_cases, test_cases = train_test_split(
        temp_cases, test_size=test_size / (test_size + val_size),
        random_state=random_state, shuffle=True
    )

    def gather_volumes(case_list):
        vols = []
        for case in case_list:
            vols.extend(case_to_volumes[case])
        return vols

    train_vols = gather_volumes(train_cases)
    val_vols   = gather_volumes(val_cases)
    test_vols  = gather_volumes(test_cases)

    # Convert to the format expected by CTPhaseDataset
    def make_split(vol_list, split_name):
        split = []
        for vol in vol_list:
            split.append({
                'source_path': vol['path'],
                'target_path': vol['path'],       # same → autoencoder
                'source_phase': 'any',
                'target_phase': 'any',
                'case_id': vol['case_id'],
                'source_series': 'autoencoder',
                'target_series': 'autoencoder',
            })
        logger.info(f"  {split_name:5s}: {len(case_ids)} cases → {len(vol_list)} volumes")
        return split

    splits = {
        'train': make_split(train_vols, 'train'),
        'val':   make_split(val_vols,   'val'),
        'test':  make_split(test_vols,  'test')
    }

    logger.info(f"\nAutoencoder data splits created successfully!")
    logger.info(f"  Train: {len(train_cases)} cases → {len(splits['train'])} volumes")
    logger.info(f"  Val:   {len(val_cases)} cases → {len(splits['val'])} volumes")
    logger.info(f"  Test:  {len(test_cases)} cases → {len(splits['test'])} volumes")

    return splits


# =============================================
# 2. PATCH NORMALIZATION (mean/std per patch)
# =============================================
def normalize_patch(patch: torch.Tensor) -> torch.Tensor:
    """Normalize patch to zero mean and unit variance."""
    patch = patch.clone()
    mean = patch.mean()
    std = patch.std()
    if std < 1e-8:
        std = 1.0
    return (patch - mean) / std

# =============================================
# 3. YOUR EXACT DATASET (autoencoder mode)
# =============================================
def get_dataloaders():
    # This uses your full pipeline from dataloader_train_slice.py
    data_splits = create_data_pairs_autoencoder(cfg)

    train_dataset = CTPhaseDataset(
        data_pairs=data_splits['train'],
        patch_size=cfg['patch_size'],
        slice_range=cfg['slice_range'],
        overlap_ratio=cfg['overlap_ratio'],
        augment=cfg.get('use_augmentation', False),
        cache_size=cfg.get('cache_size', 12),
        use_memmap=True,
        use_autoencoder=True,
        num_augment_copies=4
    )

    val_dataset = CTPhaseDataset(
        data_pairs=data_splits['val'],
        patch_size=cfg['patch_size'],
        slice_range=cfg['slice_range'],
        overlap_ratio=0.5,
        augment=False,
        cache_size=cfg.get('cache_size', 12),
        use_memmap=True,
        use_autoencoder=True,
        num_augment_copies=1
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'],
                              shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'],
                            shuffle=False, num_workers=4, pin_memory=True)

    logger.info(f"Train patches: {len(train_dataset)} | Val patches: {len(val_dataset)}")
    return train_loader, val_loader, train_dataset, val_dataset


# =============================================
# 5. VALIDATION VISUALIZATION
# =============================================

# def save_val_samples(model, val_dataset, epoch, device, samples_dir="val_samples", num_samples=8, use_amp=True):
#     model.eval()
#     samples_dir = Path(samples_dir) / f"epoch_{epoch+1:03d}"
#     samples_dir.mkdir(parents=True, exist_ok=True)

#     # Randomly select different cases
#     case_indices = list(range(len(val_dataset.data_pairs)))
#     random.shuffle(case_indices)
#     selected_cases = case_indices[:num_samples]

#     fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5*num_samples))
#     if num_samples == 1:
#         axes = axes[None, :]

#     overall_psnr = []
#     overall_ssim = []

#     with torch.no_grad():
#         for idx, pair_idx in enumerate(selected_cases):
#             pair = val_dataset.data_pairs[pair_idx]
#             source_path = Path(pair['source_path'])

#             # Load full volume once
#             vol_nii = nib.load(source_path)
#             vol = vol_nii.get_fdata()
#             vol = np.transpose(vol, (2, 1, 0))  # (D, H, W)
#             vol = np.clip(vol, -1000, 2000)     # safe HU window for display

#             D, H, W = vol.shape
#             z = random.randint(int(D * 0.3), int(D * 0.7))  # middle 40%

#             # Extract full slice
#             # slice_hu = vol[z].copy().astype(np.float32)
            
#             # Extract full slice
#             slice_hu = vol[z].astype(np.float32)
#             H, W = slice_hu.shape
            
#             # Resize to exact training patch size (128, 192)
#             slice_resized = cv2.resize(slice_hu, (192, 128), interpolation=cv2.INTER_LINEAR)

#             # === Normalize exactly like training ===
#             # Normalize exactly like training
#             mean = slice_resized.mean()
#             std = slice_resized.std() + 1e-8
#             slice_norm = (slice_resized - mean) / std
#             input_tensor = torch.from_numpy(slice_norm).unsqueeze(0).unsqueeze(0).to(device)

#             with autocast('cuda', enabled=use_amp):
#                 recon_norm = model(input_tensor).cpu().numpy().squeeze(0)

            
#             # # Denormalize reconstruction
#             # recon_hu = recon_norm * std + mean

#             # # Compute metrics on original HU scale
#             # psnr_val = psnr(slice_hu, recon_hu, data_range=slice_hu.max() - slice_hu.min())
#             # ssim_val = ssim_sk(slice_hu, recon_hu, data_range=slice_hu.max() - slice_hu.min())

#             # overall_psnr.append(psnr_val)
#             # overall_ssim.append(ssim_val)

#                 # Denormalize the 128×192 reconstruction
#                 recon_small = recon_norm * std + mean  # shape: (128, 192)

#                 # Resize back to original full slice size for fair comparison
#                 recon_hu = cv2.resize(recon_small, (W, H), interpolation=cv2.INTER_LINEAR)

#                 # Now shapes match: both (H, W)
#                 data_range = slice_hu.max() - slice_hu.min()
#                 if data_range < 1e-3:
#                     data_range = 1.0  # prevent divide by zero

#                 psnr_val = psnr(slice_hu, recon_hu, data_range=data_range)
#                 ssim_val = ssim_sk(slice_hu, recon_hu, data_range=data_range)

#                 overall_psnr.append(psnr_val)
#                 overall_ssim.append(ssim_val)

#                 # Plot
#                 axes[idx, 0].imshow(slice_hu, cmap='gray', vmin=-200, vmax=800)
#                 axes[idx, 0].set_title("Ground Truth")

#                 im = axes[idx, 1].imshow(recon_hu, cmap='gray', vmin=-200, vmax=800)
#                 axes[idx, 1].set_title(f"Reconstruction\nPSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}")

#                 diff = np.abs(slice_hu - recon_hu)
#                 im2 = axes[idx, 2].imshow(diff, cmap='hot', vmin=0, vmax=200)
#                 axes[idx, 2].set_title(f"Abs Error (max {diff.max():.1f} HU)")

#                 # Residual histogram
#                 axes[idx, 3].hist((slice_hu - recon_hu).ravel(), bins=100, range=(-200, 200), color='purple', alpha=0.7)
#                 axes[idx, 3].set_xlim(-200, 200)
#                 axes[idx, 3].set_title(f"Error Hist\nStd: {(slice_hu - recon_hu).std():.1f} HU")

#                 for ax in axes[idx]:
#                     ax.axis('off')

#         # Overall stats
#         mean_psnr = np.mean(overall_psnr)
#         mean_ssim = np.mean(overall_ssim)
#         plt.suptitle(f"Validation Reconstructions - Epoch {epoch+1} | "
#                      f"Avg PSNR: {mean_psnr:.2f} dB | Avg SSIM: {mean_ssim:.4f}", 
#                      fontsize=16, y=0.98)

#         plt.tight_layout()
#         plt.savefig(samples_dir / "val_summary.png", dpi=150, bbox_inches='tight')
#         plt.close()

#         logger.info(f"Validation samples saved | PSNR: {mean_psnr:.2f} dB | SSIM: {mean_ssim:.4f}")

def save_val_samples(model, val_dataset, epoch, device, samples_dir="val_samples", num_samples=8, use_amp=True):
    model.eval()
    samples_dir = Path(samples_dir) / f"epoch_{epoch+1:03d}"
    samples_dir.mkdir(parents=True, exist_ok=True)

    case_indices = list(range(len(val_dataset.data_pairs)))
    random.shuffle(case_indices)
    selected_cases = case_indices[:num_samples]

    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))
    if num_samples == 1:
        axes = axes[None, :]

    overall_psnr = []
    overall_ssim = []

    patch_h, patch_w = 128, 192  # training patch size: (H, W)

    with torch.no_grad():
        for idx, pair_idx in enumerate(selected_cases):
            pair = val_dataset.data_pairs[pair_idx]
            vol_nii = nib.load(pair['source_path'])
            vol = vol_nii.get_fdata()
            vol = np.transpose(vol, (2, 1, 0))  # (D, H, W)
            vol = np.clip(vol, -1000, 2000)

            D, orig_H, orig_W = vol.shape
            z = random.randint(int(D * 0.3), int(D * 0.7))
            slice_hu = vol[z].astype(np.float32)  # (orig_H, orig_W)

            # === Downsample to training patch size ===
            patch = cv2.resize(slice_hu, (patch_w, patch_h), interpolation=cv2.INTER_LINEAR)  # → (128, 192)

            # === Normalize ===
            mean = patch.mean()
            std = patch.std()
            if std < 1e-8:
                std = 1.0
            patch_norm = (patch - mean) / std
            input_tensor = torch.from_numpy(patch_norm).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,128,192)

            # === Forward ===
            with autocast('cuda', enabled=use_amp):
                recon_norm = model(input_tensor).squeeze(0).squeeze(0).cpu().numpy()  # (128, 192)

            # === Denormalize ===
            recon_small = recon_norm * std + mean  # (128, 192)

            # === Upsample back to original resolution ===
            recon_hu = cv2.resize(recon_small, (orig_W, orig_H), interpolation=cv2.INTER_LINEAR)  # → (orig_H, orig_W)

            # === Safety check ===
            assert slice_hu.shape == recon_hu.shape, f"Shape mismatch: {slice_hu.shape} vs {recon_hu.shape}"

            # === Metrics ===
            data_range = float(slice_hu.max() - slice_hu.min()) or 1.0
            psnr_val = psnr(slice_hu, recon_hu, data_range=data_range)
            ssim_val = ssim_sk(slice_hu, recon_hu, data_range=data_range)

            overall_psnr.append(psnr_val)
            overall_ssim.append(ssim_val)

            # === Plot ===
            axes[idx, 0].imshow(slice_hu, cmap='gray', vmin=-200, vmax=800)
            axes[idx, 0].set_title("Ground Truth")

            axes[idx, 1].imshow(recon_hu, cmap='gray', vmin=-200, vmax=800)
            axes[idx, 1].set_title(f"Reconstruction\nPSNR: {psnr_val:.2f} dB\nSSIM: {ssim_val:.4f}")

            diff = np.abs(slice_hu - recon_hu)
            axes[idx, 2].imshow(diff, cmap='hot', vmin=0, vmax=200)
            axes[idx, 2].set_title(f"Error\nMax: {diff.max():.1f} HU")

            axes[idx, 3].hist(diff.ravel(), bins=80, range=(0, 200), color='red', alpha=0.7)
            axes[idx, 3].set_title(f"Error Dist\nσ = {diff.std():.1f} HU")

            for ax in axes[idx]:
                ax.axis('off')

        # Summary
        mean_psnr = np.mean(overall_psnr)
        mean_ssim = np.mean(overall_ssim)
        plt.suptitle(f"Validation Samples — Epoch {epoch+1}\n"
                     f"Avg PSNR: {mean_psnr:.2f} dB | Avg SSIM: {mean_ssim:.4f}",
                     fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(samples_dir / "val_summary.png", dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Validation samples saved | PSNR: {mean_psnr:.2f} dB | SSIM: {mean_ssim:.4f}")

def update_history(epoch, train_loss, val_loss, val_psnr, val_ssim, lr):
    history['epoch'].append(epoch + 1)
    history['train_loss'].append(float(train_loss))
    history['val_loss'].append(float(val_loss))
    history['val_psnr'].append(float(val_psnr))
    history['val_ssim'].append(float(val_ssim))
    history['lr'].append(float(lr))

    # Save JSON
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

def plot_training_curves():
    if len(history['epoch']) < 1:
        return

    epochs = history['epoch']

    plt.figure(figsize=(15, 10))

    # Loss plot
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (log scale)')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # PSNR
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['val_psnr'], 'g-', label='Val PSNR', linewidth=2.5)
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.title('Validation PSNR ↑')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # SSIM
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['val_ssim'], 'm-', label='Val SSIM', linewidth=2.5)
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('Validation SSIM ↑')
    plt.ylim(0.5, 1.0)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Learning rate
    plt.subplot(2, 2, 4)
    plt.plot(epochs, history['lr'], 'orange', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(cfg['output_dir'] / "training_curves_full.png", dpi=200, bbox_inches='tight')
    plt.savefig(plot_loss_path, dpi=150, bbox_inches='tight')
    plt.savefig(plot_metrics_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Curves updated | Epoch {epochs[-1]} | Val PSNR: {history['val_psnr'][-1]:.2f} dB | SSIM: {history['val_ssim'][-1]:.4f}")

# NEW: Plot the cosine LR schedule
def plot_lr_schedule(cfg, output_path):
    """Generate and save LR curve for cosine schedule."""
    # Dummy optimizer for plotting
    dummy_optimizer = optim.AdamW([torch.tensor(0.0)], lr=cfg['lr'])  # CRITIQUE FIX: Real optimizer
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        dummy_optimizer, T_0=cfg['cosine_t0'], T_mult=cfg['cosine_tmult'], eta_min=cfg['cosine_eta_min']
    )
    
    epochs = np.arange(1, 201)  # Plot for 200 epochs
    lrs = []
    for _ in epochs:
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])
    
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, lrs, label='Cosine LR Schedule', linewidth=2)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title(f'Cosine Annealing LR Schedule\n(T0={cfg["cosine_t0"]}, T_mult={cfg["cosine_tmult"]}, min={cfg["cosine_eta_min"]})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"LR schedule plot saved: {output_path}")
    
# =============================================
# 6. MAIN TRAINING LOOP (pure MSE autoencoder)
# =============================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Generator().to(device)  # your generator
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-5)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    # UPDATED: Cosine scheduler
    if cfg.get('use_cosine', False):
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=cfg['cosine_t0'], T_mult=cfg['cosine_tmult'], eta_min=cfg['cosine_eta_min']
        )
        logger.info(f"Using CosineAnnealingWarmRestarts (T0={cfg['cosine_t0']}, min={cfg['cosine_eta_min']})")
    else:
        # Fallback (but why would you? This is what killed your runs)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        logger.warning("Using ReduceLROnPlateau — expect early plateauing")

    use_amp = cfg.get('use_mixed_precision', True)
    scaler = GradScaler('cuda', enabled=use_amp)
    
    best_val_loss = float('inf')
    samples_dir = cfg['output_dir'] / "val_samples"
    samples_dir.mkdir(exist_ok=True)
    
    # NEW: Plot LR schedule upfront
    if cfg.get('use_cosine', False):
        plot_lr_schedule(cfg, plot_lr_path)
    
    # Resume if possible
    start_epoch, best_val_loss = load_latest_checkpoint(model, optimizer, scheduler, scaler, cfg['output_dir'])

    train_loader, val_loader, train_dataset, val_dataset = get_dataloaders()


    for epoch in range(start_epoch, cfg['epochs']):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            x = batch['source'].to(device)

            # Per-patch mean/std normalization
            x_norm = torch.stack([normalize_patch(patch) for patch in x])

            optimizer.zero_grad()
            with autocast('cuda', enabled=use_amp):
                pred = model(x_norm)
                loss = F.mse_loss(pred, x_norm)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        # Validation
        # model.eval()
        # val_loss = 0.0
        # with torch.no_grad():
        #     for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False):
        #         x = batch['source'].to(device)
        #         x_norm = torch.stack([normalize_patch(p) for p in x])

        #         with autocast('cuda', enabled=use_amp):
        #             pred = model(x_norm)
        #             loss = F.mse_loss(pred, x_norm)
        #         val_loss += loss.item()

        # train_loss /= len(train_loader)
        # val_loss /= len(val_loader)
        # scheduler.step(val_loss)

        # logger.info(f"EPOCH {epoch+1:3d} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f} | "
        #          f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Validation + PSNR/SSIM
        model.eval()
        val_loss = 0.0
        val_psnr_total = 0.0
        val_ssim_total = 0.0
        val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch['source'].to(device)
                x_norm = torch.stack([(p - p.mean()) / (p.std() + 1e-8) for p in x])

                with autocast('cuda', enabled=cfg['use_mixed_precision']):
                    pred = model(x_norm)
                    loss = F.mse_loss(pred, x_norm)
                val_loss += loss.item()

                # Denormalize for metrics
                pred_np = pred.cpu().numpy()
                x_np = x_norm.cpu().numpy()
                for i in range(x_np.shape[0]):
                    orig = x_np[i, 0]
                    recon = pred_np[i, 0]
                    # Reverse normalization
                    orig_hu = orig * orig.std() + orig.mean()
                    recon_hu = recon * orig.std() + orig.mean()

                    range_val = orig_hu.max() - orig_hu.min()
                    if range_val > 1e-3:
                        val_psnr_total += psnr(orig_hu, recon_hu, data_range=range_val)
                        val_ssim_total += ssim_sk(orig_hu, recon_hu, data_range=range_val)
                        val_samples += 1

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        avg_psnr = val_psnr_total / max(val_samples, 1)
        avg_ssim = val_ssim_total / max(val_samples, 1)

        # UPDATED: Step scheduler (cosine every epoch, plateau on val_loss)
        if cfg.get('use_cosine', False):
            scheduler.step()  # Cosine doesn't need val_loss
        else:
            scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']

        # Log + update history
        logger.info(f"EPOCH {epoch+1:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                 f"PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        update_history(epoch, train_loss, val_loss, avg_psnr, avg_ssim, optimizer.param_groups[0]['lr'])
        plot_training_curves()  # ← updates plots every epoch


        # Save samples every epoch
        save_val_samples(model, val_dataset, epoch, device, samples_dir, 3, use_amp)
        save_val_samples(model, train_dataset, epoch, device, samples_dir, 3, use_amp)

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), cfg['output_dir'] / "best_autoencoder.pth")

        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': val_loss,
            }, cfg['output_dir'] / f"checkpoint_epoch_{epoch+1}.pth")

    logger.info("Pure autoencoder training completed.")

if __name__ == "__main__":
    main()