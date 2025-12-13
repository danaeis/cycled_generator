# training_autoencoder_final.py
# Pure autoencoder training â€” no GAN, no conditionals, no garbage.
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast
from torch.amp import GradScaler
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from typing import Optional

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

# from skimage.metrics import structural_similarity as ssim
from skimage.metrics import structural_similarity as ssim_sk

import nibabel as nib

# Import your existing optimized dataset and data splitter
from training_phase_gen_slice import CTPhaseDataset, VolumeCache
from dataloader_train_slice import create_data_pairs, run_full_training
from config import train_config  # your config file
# from training_phase_gen_slice import Generator2D as Generator

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
    'lr': 1e-5,
    'save_samples_interval': 1,
    'keep_last_n_checkpoints': 3,
    'use_augmentation': False,

    # Architecture selection: 'identity', 'shallow', 'medium', 'deep', 'unet'
    'model_mode': 'deep',  # START HERE: identity -> shallow -> medium -> deep
    'output_dir': Path("./autoencoder_deep_lr5"),  # Change for each experiment
    
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
                logger.info("â†’ Starting fresh CosineAnnealingWarmRestarts scheduler")
        else:
            logger.info(f"No scheduler state in checkpoint â€” initializing fresh CosineAnnealingWarmRestarts")
            # Do nothing â€” we'll just start a new cosine schedule from current epoch

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

    logger.info("Scanning for registered_norm volumes (*_registered_norm.nii.gz)...")
    
    for case_dir in sorted(data_dir.iterdir()):
        if not case_dir.is_dir():
            continue
            
        case_id = case_dir.name
        
        # Find all registered_norm-registered_norm image volumes (skip seg)
        for nii_file in case_dir.glob("*_registered_norm.nii.gz"):
            if "_seg" in nii_file.name:
                continue
                
            # Optional: extract original phase from filename for logging only
            phase_hint = "unknown"
            name = nii_file.stem.replace("_registered_norm", "")
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

    logger.info(f"Found {len(all_volumes)} registered_norm volumes across all phases")

    if len(all_volumes) == 0:
        raise ValueError("No *_registered_norm.nii.gz files found! Check your data_dir.")

    # Extract unique case IDs to split by patient (avoid leakage)
    case_to_volumes = {}
    for vol in all_volumes:
        case_to_volumes.setdefault(vol['case_id'], []).append(vol)

    case_ids = list(case_to_volumes.keys())
    
    # Split cases (not volumes) â†’ prevents data leakage
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
                'target_path': vol['path'],       # same â†’ autoencoder
                'source_phase': 'any',
                'target_phase': 'any',
                'case_id': vol['case_id'],
                'source_series': 'autoencoder',
                'target_series': 'autoencoder',
            })
        logger.info(f"  {split_name:5s}: {len(case_ids)} cases â†’ {len(vol_list)} volumes")
        return split

    splits = {
        'train': make_split(train_vols, 'train'),
        'val':   make_split(val_vols,   'val'),
        'test':  make_split(test_vols,  'test')
    }

    logger.info(f"\nAutoencoder data splits created successfully!")
    logger.info(f"  Train: {len(train_cases)} cases â†’ {len(splits['train'])} volumes")
    logger.info(f"  Val:   {len(val_cases)} cases â†’ {len(splits['val'])} volumes")
    logger.info(f"  Test:  {len(test_cases)} cases â†’ {len(splits['test'])} volumes")

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
# 4. Generator Model
# =============================================


class Generator2D(nn.Module):
    """
    Flexible autoencoder with multiple architecture modes:
    - 'identity': Pass-through (tests data pipeline)
    - 'shallow': 2-layer encoder/decoder, minimal compression
    - 'medium': 3-layer, moderate compression
    - 'deep': 4-layer, strong compression (original U-Net depth)
    - 'unet': U-Net with skip connections (not recommended for autoencoding)
    
    Key difference: TRUE autoencoders have NO skip connections (forces learning compressed representations)
    """
    def __init__(
        self,
        base_channels: int = 64,
        dropout: float = 0.3,
        mode: str = 'shallow',  # 'identity', 'shallow', 'medium', 'deep', 'unet'
        latent_spatial_size: tuple = (8, 12),  # For 128x192 input with 4 pooling: 8x12
    ):
        super().__init__()
        
        self.mode = mode
        self.base_channels = base_channels
        self.dropout_rate = dropout
        
        if mode == 'identity':
            # Just pass through - tests if data pipeline works
            self.identity = nn.Identity()
            # Add dummy parameter so optimizer doesn't complain
            self.dummy = nn.Parameter(torch.zeros(1), requires_grad=True)
            logger.info(f"Generator2D: IDENTITY mode (~0 params) - pure passthrough")
           
        elif mode == 'shallow':
            # 2 layers: 128x192 -> 64x96 -> 32x48 (bottleneck) -> 64x96 -> 128x192
            ch1 = base_channels
            ch2 = base_channels * 2
            
            self.encoder = nn.Sequential(
                # Layer 1: (1, 128, 192) -> (ch1, 64, 96)
                nn.Conv2d(1, ch1, 3, padding=1),
                nn.InstanceNorm2d(ch1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(ch1, ch1, 3, padding=1),
                nn.InstanceNorm2d(ch1),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(2),
                
                # Layer 2: (ch1, 64, 96) -> (ch2, 32, 48)
                nn.Conv2d(ch1, ch2, 3, padding=1),
                nn.InstanceNorm2d(ch2),
                nn.LeakyReLU(0.2),
                nn.Conv2d(ch2, ch2, 3, padding=1),
                nn.InstanceNorm2d(ch2),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(2),
            )
            
            self.bottleneck = nn.Sequential(
                nn.Conv2d(ch2, ch2, 3, padding=1),
                nn.InstanceNorm2d(ch2),
                nn.LeakyReLU(0.2),
                nn.Dropout2d(dropout),
            )
            
            self.decoder = nn.Sequential(
                # Layer 2: (ch2, 32, 48) -> (ch1, 64, 96)
                nn.ConvTranspose2d(ch2, ch1, 2, stride=2),
                nn.Conv2d(ch1, ch1, 3, padding=1),
                nn.InstanceNorm2d(ch1),
                nn.LeakyReLU(0.2),
                nn.Dropout2d(dropout),
                
                # Layer 1: (ch1, 64, 96) -> (1, 128, 192)
                nn.ConvTranspose2d(ch1, ch1, 2, stride=2),
                nn.Conv2d(ch1, ch1, 3, padding=1),
                nn.InstanceNorm2d(ch1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(ch1, 1, 1),
                nn.Tanh()
            )
            
        elif mode == 'medium':
            # 3 layers: More compression
            ch1 = base_channels
            ch2 = base_channels * 2
            ch3 = base_channels * 4
            
            self.encoder = nn.Sequential(
                # (1, 128, 192) -> (ch1, 64, 96)
                nn.Conv2d(1, ch1, 3, padding=1),
                nn.InstanceNorm2d(ch1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(ch1, ch1, 3, padding=1),
                nn.InstanceNorm2d(ch1),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(2),
                
                # (ch1, 64, 96) -> (ch2, 32, 48)
                nn.Conv2d(ch1, ch2, 3, padding=1),
                nn.InstanceNorm2d(ch2),
                nn.LeakyReLU(0.2),
                nn.Conv2d(ch2, ch2, 3, padding=1),
                nn.InstanceNorm2d(ch2),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(2),
                
                # (ch2, 32, 48) -> (ch3, 16, 24)
                nn.Conv2d(ch2, ch3, 3, padding=1),
                nn.InstanceNorm2d(ch3),
                nn.LeakyReLU(0.2),
                nn.Conv2d(ch3, ch3, 3, padding=1),
                nn.InstanceNorm2d(ch3),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(2),
            )
            
            self.bottleneck = nn.Sequential(
                nn.Conv2d(ch3, ch3, 3, padding=1),
                nn.InstanceNorm2d(ch3),
                nn.LeakyReLU(0.2),
                nn.Dropout2d(dropout),
            )
            
            self.decoder = nn.Sequential(
                # (ch3, 16, 24) -> (ch2, 32, 48)
                nn.ConvTranspose2d(ch3, ch2, 2, stride=2),
                nn.Conv2d(ch2, ch2, 3, padding=1),
                nn.InstanceNorm2d(ch2),
                nn.LeakyReLU(0.2),
                nn.Dropout2d(dropout),
                
                # (ch2, 32, 48) -> (ch1, 64, 96)
                nn.ConvTranspose2d(ch2, ch1, 2, stride=2),
                nn.Conv2d(ch1, ch1, 3, padding=1),
                nn.InstanceNorm2d(ch1),
                nn.LeakyReLU(0.2),
                nn.Dropout2d(dropout),
                
                # (ch1, 64, 96) -> (1, 128, 192)
                nn.ConvTranspose2d(ch1, ch1, 2, stride=2),
                nn.Conv2d(ch1, ch1, 3, padding=1),
                nn.InstanceNorm2d(ch1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(ch1, 1, 1),
                nn.Tanh()
            )
            
        elif mode == 'deep':
            # 4 layers: Maximum compression (8x12 bottleneck for 128x192 input)
            ch1 = base_channels
            ch2 = base_channels * 2
            ch3 = base_channels * 4
            ch4 = base_channels * 8
            
            self.encoder = nn.Sequential(
                # (1, 128, 192) -> (ch1, 64, 96)
                nn.Conv2d(1, ch1, 3, padding=1),
                nn.InstanceNorm2d(ch1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(ch1, ch1, 3, padding=1),
                nn.InstanceNorm2d(ch1),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(2),
                
                # (ch1, 64, 96) -> (ch2, 32, 48)
                nn.Conv2d(ch1, ch2, 3, padding=1),
                nn.InstanceNorm2d(ch2),
                nn.LeakyReLU(0.2),
                nn.Conv2d(ch2, ch2, 3, padding=1),
                nn.InstanceNorm2d(ch2),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(2),
                
                # (ch2, 32, 48) -> (ch3, 16, 24)
                nn.Conv2d(ch2, ch3, 3, padding=1),
                nn.InstanceNorm2d(ch3),
                nn.LeakyReLU(0.2),
                nn.Conv2d(ch3, ch3, 3, padding=1),
                nn.InstanceNorm2d(ch3),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(2),
                
                # (ch3, 16, 24) -> (ch4, 8, 12)
                nn.Conv2d(ch3, ch4, 3, padding=1),
                nn.InstanceNorm2d(ch4),
                nn.LeakyReLU(0.2),
                nn.Conv2d(ch4, ch4, 3, padding=1),
                nn.InstanceNorm2d(ch4),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(2),
            )
            
            self.bottleneck = nn.Sequential(
                nn.Conv2d(ch4, ch4, 3, padding=1),
                nn.InstanceNorm2d(ch4),
                nn.LeakyReLU(0.2),
                nn.Dropout2d(dropout),
                nn.Conv2d(ch4, ch4, 3, padding=1),
                nn.InstanceNorm2d(ch4),
                nn.LeakyReLU(0.2),
                nn.Dropout2d(dropout),
            )
            
            self.decoder = nn.Sequential(
                # (ch4, 8, 12) -> (ch3, 16, 24)
                nn.ConvTranspose2d(ch4, ch3, 2, stride=2),
                nn.Conv2d(ch3, ch3, 3, padding=1),
                nn.InstanceNorm2d(ch3),
                nn.LeakyReLU(0.2),
                nn.Dropout2d(dropout),
                
                # (ch3, 16, 24) -> (ch2, 32, 48)
                nn.ConvTranspose2d(ch3, ch2, 2, stride=2),
                nn.Conv2d(ch2, ch2, 3, padding=1),
                nn.InstanceNorm2d(ch2),
                nn.LeakyReLU(0.2),
                nn.Dropout2d(dropout),
                
                # (ch2, 32, 48) -> (ch1, 64, 96)
                nn.ConvTranspose2d(ch2, ch1, 2, stride=2),
                nn.Conv2d(ch1, ch1, 3, padding=1),
                nn.InstanceNorm2d(ch1),
                nn.LeakyReLU(0.2),
                nn.Dropout2d(dropout),
                
                # (ch1, 64, 96) -> (1, 128, 192)
                nn.ConvTranspose2d(ch1, ch1, 2, stride=2),
                nn.Conv2d(ch1, ch1, 3, padding=1),
                nn.InstanceNorm2d(ch1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(ch1, 1, 1),
                nn.Tanh()
            )
            
        elif mode == 'unet':
            # Original U-Net with skip connections (NOT recommended for autoencoding)
            ch1 = base_channels
            ch2 = base_channels * 2
            ch3 = base_channels * 4
            ch4 = base_channels * 8
            
            self.enc1 = self._make_enc_block(1, ch1)
            self.enc2 = self._make_enc_block(ch1, ch2)
            self.enc3 = self._make_enc_block(ch2, ch3)
            self.enc4 = self._make_enc_block(ch3, ch4)
            self.pool = nn.MaxPool2d(2)
            
            self.bottleneck = nn.Sequential(
                nn.Conv2d(ch4, ch4, 3, padding=1),
                nn.InstanceNorm2d(ch4),
                nn.LeakyReLU(0.2),
                nn.Dropout2d(dropout),
                nn.Conv2d(ch4, ch4, 3, padding=1),
                nn.InstanceNorm2d(ch4),
                nn.LeakyReLU(0.2),
                nn.Dropout2d(dropout),
            )
            
            self.up4 = nn.ConvTranspose2d(ch4, ch3, 2, stride=2)
            self.dec4 = self._make_dec_block(ch3 + ch4, ch3, dropout)
            
            self.up3 = nn.ConvTranspose2d(ch3, ch2, 2, stride=2)
            self.dec3 = self._make_dec_block(ch2 + ch3, ch2, dropout)
            
            self.up2 = nn.ConvTranspose2d(ch2, ch1, 2, stride=2)
            self.dec2 = self._make_dec_block(ch1 + ch2, ch1, dropout)
            
            self.up1 = nn.ConvTranspose2d(ch1, ch1, 2, stride=2)
            self.dec1 = self._make_dec_block(ch1 + ch1, ch1, dropout)
            
            self.out_conv = nn.Sequential(
                nn.Conv2d(ch1, 1, 1),
                nn.Tanh()
            )
            
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'identity', 'shallow', 'medium', 'deep', or 'unet'")
        
        total_params = sum(p.numel() for p in self.parameters()) / 1e6
        logger.info(f"Generator2D: mode={mode} | base_ch={base_channels} | {total_params:.2f}M params | dropout={dropout}")
    
    def _make_enc_block(self, in_ch: int, out_ch: int):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2),
        )
    
    def _make_dec_block(self, in_ch: int, out_ch: int, dropout: float):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == 'identity':
            return x + self.dummy * 0

        elif self.mode in ['shallow', 'medium', 'deep']:
            z = self.encoder(x)
            z = self.bottleneck(z)
            return self.decoder(z)
        
        elif self.mode == 'unet':
            # U-Net with skip connections
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            e4 = self.enc4(self.pool(e3))
            
            b = self.bottleneck(self.pool(e4))
            
            d = self.up4(b)
            d = torch.cat([d, e4], dim=1)
            d = self.dec4(d)
            
            d = self.up3(d)
            d = torch.cat([d, e3], dim=1)
            d = self.dec3(d)
            
            d = self.up2(d)
            d = torch.cat([d, e2], dim=1)
            d = self.dec2(d)
            
            d = self.up1(d)
            d = torch.cat([d, e1], dim=1)
            d = self.dec1(d)
            
            return self.out_conv(d)



# =============================================
# 5. VALIDATION VISUALIZATION
# =============================================

# def save_val_samples(
#         model,
#         val_dataset,
#         epoch,
#         device,
#         samples_dir="val_samples",
#         num_samples=8,
#         use_amp=True
#     ):
#     model.eval()
#     samples_dir = Path(samples_dir) / f"epoch_{epoch+1:03d}"
#     samples_dir.mkdir(parents=True, exist_ok=True)

#     patches_per_volume = {}
#     for idx, coord in enumerate(val_dataset.patch_coords):
#         pair_idx = coord['pair_idx']
#         if pair_idx not in patches_per_volume:
#             patches_per_volume[pair_idx] = []
#         patches_per_volume[pair_idx].append((idx, coord))

#     if len(patches_per_volume) == 0:
#         logger.warning("No patches available for visualization!")
#         return

#     all_psnr = []
#     all_ssim = []

#     # Randomly sample different volumes
#     selected_volumes = random.sample(list(patches_per_volume.keys()), 
#                                   min(num_samples, len(patches_per_volume)))

#     saved = 0 
#     with torch.no_grad():
#         for pair_idx in enumerate(selected_volumes):
#             if saved >= num_samples:
#                 break

#             patch_list = patches_per_volume[pair_idx]
#             # Pick a random patch from this volume
#             patch_global_idx, coord = random.choice(patch_list)

#             # Extract using dataset logic (reuse your cache!)
#             item = val_dataset[patch_global_idx]  # This uses __getitem__ → super fast thanks to cache

#             source = item['source'].unsqueeze(0).to(device)   # (1,1,H,W)
#             target = item['target'].unsqueeze(0).to(device)

#             with autocast('cuda', enabled=use_amp):
#                 recon = generator(source)

#             # To numpy
#             inp = source[0,0].cpu().numpy()
#             gen = recon[0,0].cpu().numpy()
#             tgt = target[0,0].cpu().numpy()

#             # === Reverse [0,1] → realistic HU-like range for visualization ===
#             def denorm_01(x):
#                 return np.clip(x, 0, 1) * 4.0 - 2.0   # → ~[-2, 2], looks good

#             inp_viz = denorm_01(inp)
#             gen_viz = denorm_01(gen)
#             tgt_viz = denorm_01(tgt)

#             # === Metrics (on normalized data) ===
#             mse = np.mean((gen - tgt) ** 2)
#             psnr_val = 20 * np.log10(4.0 / (np.sqrt(mse) + 1e-8))
#             ssim_val = ssim(tgt, gen, data_range=4.0)

#             all_psnr.append(psnr_val)
#             all_ssim.append(ssim_val)

#             # === Plot ===
#             fig, axes = plt.subplots(1, 5, figsize=(25, 6))

#             axes[0].imshow(inp_viz, cmap='gray', vmin=-1.8, vmax=2.2)
#             axes[0].set_title(f"Input Patch\nCase {val_dataset.data_pairs[pair_idx]['case_id']}\nSlice z={coord['z']}", fontsize=12)
#             axes[0].axis('off')

#             im1 = axes[1].imshow(gen_viz, cmap='gray', vmin=-1.8, vmax=2.2)
#             axes[1].set_title(f"Reconstruction\nPSNR: {psnr_val:.2f} dB\nSSIM: {ssim_val:.4f}", fontsize=12)
#             axes[1].axis('off')

#             axes[2].imshow(tgt_viz, cmap='gray', vmin=-1.8, vmax=2.2)
#             axes[2].set_title("Ground Truth", fontsize=12)
#             axes[2].axis('off')

#             diff = np.abs(gen_viz - tgt_viz)
#             im3 = axes[3].imshow(diff, cmap='hot', vmin=0, vmax=1.0)
#             axes[3].set_title(f"Abs Error Map\nMax: {diff.max():.3f}", fontsize=12)
#             axes[3].axis('off')
#             plt.colorbar(im3, ax=axes[3], fraction=0.046)

#             axes[4].hist(diff.ravel(), bins=100, range=(0, 1.0), color='red', alpha=0.8)
#             axes[4].set_title(f"Error Distribution\nσ = {diff.std():.3f}")
#             axes[4].set_xlabel("Error")
#             axes[4].grid(True, alpha=0.3)

#             plt.suptitle(f"Epoch {epoch+1} | Random Validation Sample {saved+1}/{num_samples}", fontsize=16)
#             plt.tight_layout()
#             plt.savefig(samples_dir / f"sample_{saved:02d}_case{val_dataset.data_pairs[pair_idx]['case_id']}_z{coord['z']}.png",
#                         dpi=150, bbox_inches='tight')
#             plt.close()

#             saved += 1

#     # Summary
#     if all_psnr:
#         mean_psnr = np.mean(all_psnr)
#         mean_ssim = np.mean(all_ssim)
#         logger.info(f"Validation Random Samples | Avg PSNR: {mean_psnr:.2f} dB | Avg SSIM: {mean_ssim:.4f}")
#         # Save summary
#         fig, ax = plt.subplots(figsize=(8, 3))
#         ax.text(0.5, 0.7, f"Epoch {epoch+1} - Random Validation Summary", fontsize=16, ha='center', weight='bold')
#         ax.text(0.5, 0.5, f"Average PSNR: {mean_psnr:.2f} dB", fontsize=18, ha='center', color='blue')
#         ax.text(0.5, 0.3, f"Average SSIM: {mean_ssim:.4f}", fontsize=18, ha='center', color='green')
#         ax.axis('off')
#         plt.savefig(samples_dir / "SUMMARY.png", dpi=200, bbox_inches='tight')
#         plt.close()

#     generator.train()

#     # Final summary plot
#     mean_psnr = np.mean(all_psnr)
#     mean_ssim = np.mean(all_ssim)
#     fig, ax = plt.subplots(1, 1, figsize=(8, 2))
#     ax.text(0.5, 0.6, f"Epoch {epoch+1} Validation Summary", fontsize=16, ha='center')
#     ax.text(0.5, 0.4, f"Avg PSNR: {mean_psnr:.2f} dB", fontsize=20, ha='center', color='blue')
#     ax.text(0.5, 0.2, f"Avg SSIM: {mean_ssim:.4f}", fontsize=20, ha='center', color='green')
#     ax.axis('off')
#     plt.savefig(samples_dir / "SUMMARY.png", dpi=200, bbox_inches='tight')
#     plt.close()

#     logger.info(f"Validation samples saved → PSNR: {mean_psnr:.2f} dB | SSIM: {mean_ssim:.4f}")
#     model.train()



def save_val_samples(
        generator,
        val_dataset,
        epoch,
        device,
        samples_dir="val_samples",
        num_samples=8,
        use_amp=True
    ):
    generator.eval()
    samples_dir = Path(samples_dir) / f"epoch_{epoch+1:03d}"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Build list of valid global patch indices (fast & safe)
    # ------------------------------------------------------------------
    total_patches = len(val_dataset)
    if total_patches == 0:
        logger.warning("Validation dataset is empty! Skipping sample saving.")
        generator.train()
        return

    # Randomly sample patch indices (handles augmentation perfectly)
    selected_indices = random.sample(range(total_patches), min(num_samples, total_patches))

    all_psnr = []
    all_ssim = []

    logger.info(f"Saving {len(selected_indices)} random validation reconstructions (epoch {epoch+1})...")

    with torch.no_grad():
        for plot_idx, global_idx in enumerate(selected_indices):
            try:
                item = val_dataset[global_idx]                # ← uses your cache + augmentation logic
                if 'source' not in item or 'target' not in item:
                    continue

                source = item['source'].unsqueeze(0).to(device)   # (1,1,H,W)
                target = item['target'].unsqueeze(0).to(device)

                with autocast('cuda', enabled=use_amp):
                    recon = generator(source)

                # To numpy
                inp = source[0,0].cpu().numpy()
                gen = recon[0,0].cpu().numpy()
                tgt = target[0,0].cpu().numpy()

                # Reverse [0,1] → realistic range for visualization
                def denorm(x):
                    return np.clip(x, 0, 1) * 4.0 - 2.0   # → [-2, 2]

                inp_viz = denorm(inp)
                gen_viz = denorm(gen)
                tgt_viz = denorm(tgt)

                # Metrics
                mse = np.mean((gen - tgt) ** 2)
                psnr_val = 20 * np.log10(4.0 / (np.sqrt(mse) + 1e-8))
                ssim_val = ssim_sk(tgt, gen, data_range=4.0)

                all_psnr.append(psnr_val)
                all_ssim.append(ssim_val)

                # Try to get case ID and slice (optional – won’t crash if missing)
                try:
                    coord = val_dataset.patch_coords[global_idx // val_dataset.num_augment_copies]
                    case_id = val_dataset.data_pairs[coord['pair_idx']]['case_id']
                    slice_z = coord['z']
                    title_info = f"Case: {case_id} | z={slice_z}"
                except:
                    title_info = f"Patch idx {global_idx}"

                # Plot
                fig, axes = plt.subplots(1, 5, figsize=(25, 6))

                axes[0].imshow(inp_viz, cmap='gray', vmin=-1.8, vmax=2.2)
                axes[0].set_title(f"Input\n{title_info}")
                axes[0].axis('off')

                axes[1].imshow(gen_viz, cmap='gray', vmin=-1.8, vmax=2.2)
                axes[1].set_title(f"Reconstruction\nPSNR: {psnr_val:.2f} dB\nSSIM: {ssim_val:.4f}")
                axes[1].axis('off')

                axes[2].imshow(tgt_viz, cmap='gray', vmin=-1.8, vmax=2.2)
                axes[2].set_title("Ground Truth")
                axes[2].axis('off')

                diff = np.abs(gen_viz - tgt_viz)
                im = axes[3].imshow(diff, cmap='hot', vmin=0, vmax=1.0)
                axes[3].set_title(f"Error Map\nMax: {diff.max():.3f}")
                axes[3].axis('off')
                plt.colorbar(im, ax=axes[3], fraction=0.046)

                axes[4].hist(diff.ravel(), bins=100, range=(0, 1.0), color='red', alpha=0.8)
                axes[4].set_title(f"Error Dist\nσ = {diff.std():.3f}")
                axes[4].grid(True, alpha=0.3)

                plt.suptitle(f"Epoch {epoch+1} | Random Val Sample {plot_idx+1}/{len(selected_indices)}", fontsize=16)
                plt.tight_layout()
                plt.savefig(samples_dir / f"sample_{plot_idx:02d}.png", dpi=150, bbox_inches='tight')
                plt.close()

            except Exception as e:
                logger.error(f"Failed to process patch {global_idx}: {e}")
                continue

    # Summary
    if all_psnr:
        mean_psnr = np.mean(all_psnr)
        mean_ssim = np.mean(all_ssim)
        logger.info(f"RANDOM VAL SAMPLES → PSNR: {mean_psnr:.2f} dB | SSIM: {mean_ssim:.4f}")

        fig, ax = plt.subplots(figsize=(9, 3))
        ax.text(0.5, 0.7, f"Epoch {epoch+1} - Random Validation Summary", ha='center', fontsize=16, weight='bold')
        ax.text(0.5, 0.5, f"Avg PSNR: {mean_psnr:.2f} dB", ha='center', fontsize=18, color='blue')
        ax.text(0.5, 0.3, f"Avg SSIM: {mean_ssim:.4f}", ha='center', fontsize=18, color='green')
        ax.axis('off')
        plt.savefig(samples_dir / "SUMMARY.png", dpi=200, bbox_inches='tight')
        plt.close()

    generator.train()

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
    plt.title('Validation PSNR â†‘')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # SSIM
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['val_ssim'], 'm-', label='Val SSIM', linewidth=2.5)
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('Validation SSIM â†‘')
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
    
    # Initialize model with selected architecture mode
    model = Generator2D(
        base_channels=cfg.get('generator_base_channels', 64),
        dropout=cfg.get('generator_dropout', 0.3),
        mode=cfg.get('model_mode', 'shallow')
    ).to(device)
    
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
        logger.warning("Using ReduceLROnPlateau â€” expect early plateauing")

    use_amp = cfg.get('use_mixed_precision', True)
    scaler = GradScaler('cuda', enabled=use_amp)
    
    best_val_loss = float('inf')
    samples_dir = cfg['output_dir'] / "val_samples"
    samples_dir_train = cfg['output_dir'] / "train_samples"
    samples_dir.mkdir(exist_ok=True)
    samples_dir_train.mkdir(exist_ok=True)
    
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
            x_norm = x #torch.stack([normalize_patch(patch) for patch in x])

            optimizer.zero_grad()
            with autocast('cuda', enabled=use_amp):
                pred = model(x_norm)
                loss = F.mse_loss(pred, x_norm)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        # Validation

        model.eval()
        val_loss = 0.0
        val_psnr_total = 0.0
        val_ssim_total = 0.0
        val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch['source'].to(device)
                x_norm = x #torch.stack([(p - p.mean()) / (p.std() + 1e-8) for p in x])

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
        plot_training_curves()  # â† updates plots every epoch


        # Save samples every epoch
        save_val_samples(model, val_dataset, epoch, device, samples_dir, 3, use_amp)
        save_val_samples(model, train_dataset, epoch, device, samples_dir_train, 3, use_amp)

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), cfg['output_dir'] / "best_autoencoder.pth")

        if (epoch) % 3 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': val_loss,
            }, cfg['output_dir'] / f"checkpoint_epoch_{epoch+1}.pth")

    logger.info("Pure autoencoder training completed.")

if __name__ == "__main__":
    main()






# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # Initialize model with selected architecture mode
#     model = Generator2D(
#         base_channels=cfg.get('generator_base_channels', 64),
#         dropout=cfg.get('generator_dropout', 0.3),
#         mode=cfg.get('model_mode', 'shallow')
#     ).to(device)
    
#     optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-5)
#     # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
#     # UPDATED: Cosine scheduler
#     if cfg.get('use_cosine', False):
#         scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
#             optimizer, T_0=cfg['cosine_t0'], T_mult=cfg['cosine_tmult'], eta_min=cfg['cosine_eta_min']
#         )
#         logger.info(f"Using CosineAnnealingWarmRestarts (T0={cfg['cosine_t0']}, min={cfg['cosine_eta_min']})")
#     else:
#         # Fallback (but why would you? This is what killed your runs)
#         scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
#         logger.warning("Using ReduceLROnPlateau â€” expect early plateauing")

#     use_amp = cfg.get('use_mixed_precision', True)
#     scaler = GradScaler('cuda', enabled=use_amp)
    
#     best_val_loss = float('inf')
#     samples_dir = cfg['output_dir'] / "val_samples"
#     samples_dir.mkdir(exist_ok=True)
    
#     # NEW: Plot LR schedule upfront
#     if cfg.get('use_cosine', False):
#         plot_lr_schedule(cfg, plot_lr_path)
    
#     # Resume if possible
#     start_epoch, best_val_loss = load_latest_checkpoint(model, optimizer, scheduler, scaler, cfg['output_dir'])

#     train_loader, val_loader, train_dataset, val_dataset = get_dataloaders()


#     for epoch in range(start_epoch, cfg['epochs']):
#         model.train()
#         train_loss = 0.0

#         for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
#             x = batch['source'].to(device)

#             # Per-patch mean/std normalization
#             x_norm = torch.stack([normalize_patch(patch) for patch in x])

#             optimizer.zero_grad()
#             with autocast('cuda', enabled=use_amp):
#                 pred = model(x_norm)
#                 loss = F.mse_loss(pred, x_norm)

#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()

#             train_loss += loss.item()

#         # Validation

#         model.eval()
#         val_loss = 0.0
#         val_psnr_total = 0.0
#         val_ssim_total = 0.0
#         val_samples = 0

#         with torch.no_grad():
#             for batch in val_loader:
#                 x = batch['source'].to(device)
#                 x_norm = torch.stack([(p - p.mean()) / (p.std() + 1e-8) for p in x])

#                 with autocast('cuda', enabled=cfg['use_mixed_precision']):
#                     pred = model(x_norm)
#                     loss = F.mse_loss(pred, x_norm)
#                 val_loss += loss.item()

#                 # Denormalize for metrics
#                 pred_np = pred.cpu().numpy()
#                 x_np = x_norm.cpu().numpy()
#                 for i in range(x_np.shape[0]):
#                     orig = x_np[i, 0]
#                     recon = pred_np[i, 0]
#                     # Reverse normalization
#                     orig_hu = orig * orig.std() + orig.mean()
#                     recon_hu = recon * orig.std() + orig.mean()

#                     range_val = orig_hu.max() - orig_hu.min()
#                     if range_val > 1e-3:
#                         val_psnr_total += psnr(orig_hu, recon_hu, data_range=range_val)
#                         val_ssim_total += ssim_sk(orig_hu, recon_hu, data_range=range_val)
#                         val_samples += 1

#         train_loss /= len(train_loader)
#         val_loss /= len(val_loader)
#         avg_psnr = val_psnr_total / max(val_samples, 1)
#         avg_ssim = val_ssim_total / max(val_samples, 1)

#         # UPDATED: Step scheduler (cosine every epoch, plateau on val_loss)
#         if cfg.get('use_cosine', False):
#             scheduler.step()  # Cosine doesn't need val_loss
#         else:
#             scheduler.step(val_loss)

#         current_lr = optimizer.param_groups[0]['lr']

#         # Log + update history
#         logger.info(f"EPOCH {epoch+1:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
#                  f"PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

#         update_history(epoch, train_loss, val_loss, avg_psnr, avg_ssim, optimizer.param_groups[0]['lr'])
#         plot_training_curves()  # â† updates plots every epoch


#         # Save samples every epoch
#         save_val_samples(model, val_dataset, epoch, device, samples_dir, 3, use_amp)

#         # Save checkpoint
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             torch.save(model.state_dict(), cfg['output_dir'] / "best_autoencoder.pth")

#         if (epoch + 1) % 5 == 0:
#             torch.save({
#                 'epoch': epoch + 1,
#                 'model': model.state_dict(),
#                 'optimizer': optimizer.state_dict(),
#                 'val_loss': val_loss,
#             }, cfg['output_dir'] / f"checkpoint_epoch_{epoch+1}.pth")

#     logger.info("Pure autoencoder training completed.")

# if __name__ == "__main__":
#     main()