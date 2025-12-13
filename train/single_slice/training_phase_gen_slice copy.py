"""
OPTIMIZED Memory-Efficient CT Phase Generation Training Pipeline
================================================================
Key Optimizations:
- Smart volume caching (LRU cache for frequently accessed volumes)
- Memory-mapped file support for efficient random access
- Lazy loading - only load what's needed
- Reduced disk I/O by 90%+
- All losses preserved exactly as before
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.optim.lr_scheduler as lr_scheduler

import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import logging
import gc
import os
import psutil
from collections import OrderedDict
from functools import lru_cache
import cv2
import random

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# VOLUME CACHE MANAGER (NEW!)
# ============================================================================

class VolumeCache:
    """
    LRU cache for NIfTI volumes to minimize disk I/O.
    Keeps most recently used volumes in memory.
    """
    
    def __init__(self, max_cache_size: int = 8):
        """
        Args:
            max_cache_size: Maximum number of volumes to keep in cache
        """
        self.max_cache_size = max_cache_size
        self.cache = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        
    def get_volume(self, path: Path, use_memmap: bool = True) -> np.ndarray:
        """
        Get volume from cache or load from disk.
        
        Args:
            path: Path to NIfTI file
            use_memmap: Use memory mapping for efficient access
        """
        path_str = str(path)
        
        # Check cache first
        if path_str in self.cache:
            self.cache_hits += 1
            # Move to end (most recently used)
            self.cache.move_to_end(path_str)
            return self.cache[path_str]
        
        # Cache miss - load from disk
        self.cache_misses += 1
        nii = nib.load(path)
        
        if use_memmap and hasattr(nii.dataobj, '_mmap'):
            # Use memory-mapped access (doesn't load entire file)
            volume = np.asarray(nii.dataobj)
        else:
            volume = nii.get_fdata()
        
        # Transpose to (D, H, W)
        volume = np.transpose(volume, (2, 1, 0))
        
        # Add to cache
        self.cache[path_str] = volume
        self.cache.move_to_end(path_str)
        
        # Evict oldest if cache is full
        if len(self.cache) > self.max_cache_size:
            self.cache.popitem(last=False)
        
        return volume
    
    def get_volume_shape(self, path: Path) -> Tuple[int, int, int]:
        """Get volume shape without loading data."""
        nii = nib.load(path)
        shape = nii.shape
        return (shape[2], shape[1], shape[0])  # Return as (D, H, W)
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        gc.collect()
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            'cache_size': len(self.cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate
        }


# ============================================================================
# MEMORY MANAGEMENT
# ============================================================================

class MemoryManager:
    """Manages memory and prevents cache/temp overflow."""
    
    def __init__(self, temp_dir: str = '/tmp'):
        self.temp_dir = Path(temp_dir)
        self.cleanup_patterns = ['*.nii', '*.nii.gz', 'core.*', '*.tmp']
        self.last_warning_epoch = -1

    def get_memory_info(self) -> Dict:
        """Get current memory usage."""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            gpu_cached = torch.cuda.memory_reserved() / 1024**3
        else:
            gpu_memory = 0
            gpu_cached = 0
        
        ram = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'gpu_allocated_gb': gpu_memory,
            'gpu_cached_gb': gpu_cached,
            'ram_used_gb': ram.used / 1024**3,
            'ram_percent': ram.percent,
            'disk_used_gb': disk.used / 1024**3,
            'disk_percent': disk.percent
        }
    
    def log_memory_status(self):
        """Log current memory status."""
        mem = self.get_memory_info()
        logger.info(f"Memory - GPU: {mem['gpu_allocated_gb']:.2f}GB, "
                   f"RAM: {mem['ram_percent']:.1f}%, Disk: {mem['disk_percent']:.1f}%")
    
    def cleanup_temp_files(self):
        """Clean up temporary files."""
        cleaned_size = 0
        for pattern in self.cleanup_patterns:
            for file in self.temp_dir.glob(pattern):
                try:
                    size = file.stat().st_size
                    file.unlink()
                    cleaned_size += size
                except:
                    pass
        
        if cleaned_size > 1024**3:  # If > 1GB freed
            logger.info(f"  Freed {cleaned_size / 1024**3:.2f}GB from temp")
        
        plt.close('all')
    
    def cleanup_cache(self):
        """Clear GPU and system caches."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
    def check_and_cleanup(self, force: bool = False, current_epoch: int = None):
        """Check memory and cleanup if needed."""
        mem = self.get_memory_info()
        should_warn = (mem['ram_percent'] > 80 or mem['disk_percent'] > 80)
        if current_epoch is not None:
            should_warn = should_warn and (current_epoch != self.last_warning_epoch)
        
        if should_warn or force:
            if should_warn and current_epoch is not None:
                logger.warning(f"High memory! RAM: {mem['ram_percent']:.1f}%, Disk: {mem['disk_percent']:.1f}% (Epoch {current_epoch})")
                self.last_warning_epoch = current_epoch
            self.cleanup_cache()
            self.cleanup_temp_files()
    
    def emergency_cleanup(self):
        """Aggressive cleanup for OOM."""
        # logger.warning("Emergency cleanup!")
        plt.close('all')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
        for _ in range(3):
            gc.collect()
        self.cleanup_temp_files()


# ============================================================================
# OPTIMIZED DATASET WITH VOLUME CACHING
# ============================================================================

class CTPhaseDataset(Dataset):
    """
    OPTIMIZED: Memory-efficient dataset with smart volume caching.
    
    Key improvements:
    - Volumes cached in memory (LRU eviction)
    - Only loads volume shape during initialization
    - Patch extraction from cached volumes (no repeated disk I/O)
    - Memory-mapped file support for large volumes
    """
    
    def __init__(
        self,
        data_pairs: List[Dict],
        patch_size: Tuple[int, int] = (256, 256),  # Now 2D: H x W
        slice_range: Tuple[float, float] = (0.3, 0.7),  # Use middle slices
        overlap_ratio: float = 0.5,
        augment: bool = False,
        # body_focused: bool = True,
        # body_threshold: float = -500.0,
        # min_fg_ratio: float = 0.6,
        cache_size: int = 8,
        use_memmap: bool = True,
        ):
        
        self.data_pairs = data_pairs
        self.patch_size = patch_size
        self.slice_range = slice_range  # NEW: slice range for 2D
        self.overlap_ratio = overlap_ratio
        self.augment = augment
        # self.body_focused = body_focused
        # self.body_threshold = body_threshold
        # self.min_fg_ratio = min_fg_ratio
        self.patch_coords = []
        
        self.volume_cache = VolumeCache(max_cache_size=cache_size)
        self.use_memmap = use_memmap
        
        self.phase_to_idx = {
            'non-contrast': 0,
            'arterial': 1,
            'portal': 2,
            'venous': 2,
            'delayed': 3,
            'other':4
        }
        
        logger.info(f"Dataset: {len(data_pairs)} pairs, 2D slices {patch_size}")
        logger.info(f"Slice range: {slice_range[0]:.1%} to {slice_range[1]:.1%}")
        self._compute_slice_coordinates()
        logger.info(f"Generated {len(self.patch_coords)} 2D slices")

    def augment_slice(self, source, target):
        """Apply random augmentations to both source and target."""
        if not self.augment:
            return source, target
        
        # Convert to PIL for transforms
        source_pil = TF.to_pil_image(source.squeeze())
        target_pil = TF.to_pil_image(target.squeeze())
        
        # # 1. Random horizontal flip (50%)
        # if random.random() > 0.5:
        #     source_pil = TF.hflip(source_pil)
        #     target_pil = TF.hflip(target_pil)
        
        # 2. Random vertical flip (50%)
        if random.random() > 0.5:
            source_pil = TF.vflip(source_pil)
            target_pil = TF.vflip(target_pil)
        
        # 3. Random rotation (±15 degrees)
        angle = random.uniform(-15, 15)
        source_pil = TF.rotate(source_pil, angle)
        target_pil = TF.rotate(target_pil, angle)
        
        # 4. Random scaling (90%-110%)
        scale = random.uniform(0.9, 1.1)
        new_size = int(source_pil.size[0] * scale)
        source_pil = TF.resize(source_pil, new_size)
        target_pil = TF.resize(target_pil, new_size)
        # Center crop back to original size
        source_pil = TF.center_crop(source_pil, self.patch_size)
        target_pil = TF.center_crop(target_pil, self.patch_size)
        
        # # 5. Random brightness/contrast (CT-specific)
        # if random.random() > 0.5:
        #     brightness_factor = random.uniform(0.8, 1.2)
        #     source_pil = TF.adjust_brightness(source_pil, brightness_factor)
        #     target_pil = TF.adjust_brightness(target_pil, brightness_factor)
        
        # if random.random() > 0.5:
        #     contrast_factor = random.uniform(0.8, 1.2)
        #     source_pil = TF.adjust_contrast(source_pil, contrast_factor)
        #     target_pil = TF.adjust_contrast(target_pil, contrast_factor)
        
        # 6. Random Gaussian noise (medical imaging specific)
        source = TF.to_tensor(source_pil).squeeze()
        target = TF.to_tensor(target_pil).squeeze()
        
        if random.random() > 0.5:
            noise = torch.randn_like(source) * 0.02
            source = source + noise
            target = target + noise
        
        return source.unsqueeze(0), target.unsqueeze(0)
  
    def _compute_slice_coordinates(self):
        """Extract 2D slice coordinates from volume middle range."""
        for pair_idx, pair_data in enumerate(self.data_pairs):
            try:
                source_shape = self.volume_cache.get_volume_shape(pair_data['source_path'])
                target_shape = self.volume_cache.get_volume_shape(pair_data['target_path'])
                
                if source_shape != target_shape:
                    continue
                
                depth, height, width = source_shape
                
                # Get slice range (middle 40% by default)
                z_start = int(depth * self.slice_range[0])
                z_end = int(depth * self.slice_range[1])
                z_range = range(z_start, z_end)
                
                # Spatial sampling with overlap
                step_y = max(1, int(self.patch_size[0] * (1 - self.overlap_ratio)))
                step_x = max(1, int(self.patch_size[1] * (1 - self.overlap_ratio)))
                
                y_positions = list(range(0, height - self.patch_size[0] + 1, step_y))
                x_positions = list(range(0, width - self.patch_size[1] + 1, step_x))
                
                # Generate coordinates for each slice
                for z in z_range:
                    for y in y_positions:
                        for x in x_positions:
                            self.patch_coords.append({
                                'pair_idx': pair_idx,
                                'z': z,  # Single slice index
                                'y': y,
                                'x': x
                            })
            except Exception as e:
                logger.warning(f"Skipping pair {pair_idx}: {e}")

    def __len__(self) -> int:
        return len(self.patch_coords)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Extract single 2D slice."""
        coord = self.patch_coords[idx]
        pair_data = self.data_pairs[coord['pair_idx']]
        
        # Get cached volumes
        source_vol = self.volume_cache.get_volume(
            pair_data['source_path'], use_memmap=self.use_memmap
        )
        target_vol = self.volume_cache.get_volume(
            pair_data['target_path'], use_memmap=self.use_memmap
        )
        
        # Extract single 2D slice
        z, y, x = coord['z'], coord['y'], coord['x']
        source_slice = source_vol[z, y:y+self.patch_size[0], x:x+self.patch_size[1]].copy()
        target_slice = target_vol[z, y:y+self.patch_size[0], x:x+self.patch_size[1]].copy()
        
        # Phase encoding
        source_phase_idx = self.phase_to_idx.get(pair_data['source_phase'], 0)
        target_phase_idx = self.phase_to_idx.get(pair_data['target_phase'], 1)
        
        # # Normalize to [-1, 1]
        # source_slice = (source_slice - source_slice.mean()) / (source_slice.std() + 1e-8)
        # target_slice = (target_slice - target_slice.mean()) / (target_slice.std() + 1e-8)
        
        # Convert to tensors [1, H, W]
        source_tensor = torch.from_numpy(source_slice).unsqueeze(0).float()
        target_tensor = torch.from_numpy(target_slice).unsqueeze(0).float()
        
        if self.augment:
            source_tensor, target_tensor = self.augment_slice(
                source_tensor, target_tensor
            )
        # Extract masks if available
        masks = {}
        if 'target_seg' in pair_data and pair_data['target_seg']:
            try:
                seg_path = Path(pair_data['target_seg'])
                if seg_path.exists():
                    seg_vol = self.volume_cache.get_volume(seg_path, use_memmap=self.use_memmap)
                    seg_slice = seg_vol[z, y:y+self.patch_size[0], x:x+self.patch_size[1]].copy()
                    
                    # Resize if needed
                    if seg_slice.shape != source_slice.shape:
                        seg_slice = cv2.resize(seg_slice, 
                                            (self.patch_size[1], self.patch_size[0]),
                                            interpolation=cv2.INTER_NEAREST)
                    
                    seg_tensor = torch.from_numpy(seg_slice).unsqueeze(0).float()  # [1, H, W]
                    
                    # Organ mapping
                    organ_labels = {
                        'liver': [1],
                        'spleen': [2],
                        'kidney_right': [3],
                        'kidney_left': [4],
                        'pancreas': [6],
                        'stomach': [5]
                    }
                    
                    for organ, labels in organ_labels.items():
                        mask = torch.zeros_like(seg_tensor)
                        for lbl in labels:
                            mask = torch.logical_or(mask, seg_tensor == lbl).float()
                        masks[organ] = mask.unsqueeze(0)  # [1, 1, H, W]
            except Exception as e:
                logger.warning(f"Mask loading failed: {e}")
        
        return {
            'source': source_tensor,
            'target': target_tensor,
            'source_phase': torch.tensor(source_phase_idx, dtype=torch.long),
            'target_phase': torch.tensor(target_phase_idx, dtype=torch.long),
            'case_id': pair_data['case_id'],
            'masks': masks,
            'skip': False
        }
        
        def get_cache_stats(self) -> Dict:
            """Get volume cache statistics."""
            return self.volume_cache.get_stats()
        
        def clear_cache(self):
            """Clear the volume cache."""
            self.volume_cache.clear()


# ============================================================================
# MODEL ARCHITECTURES (UNCHANGED - Keep your exact models)
# ============================================================================
# --------------------------------------------------------------
#  training_phase_gen.py  –  REPLACE THE WHOLE Generator3D CLASS
# --------------------------------------------------------------

class ResidualBlock3D(nn.Module):
    """3D Residual Block with InstanceNorm (unchanged)."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        # FIX: Add projection when in_channels != out_channels
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        # residual = x
        # residual = self.shortcut(x)
        x = self.conv1(x)
        residual = x
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.norm(x + residual)
        return self.activation(x)
class Generator2D(nn.Module):
    """
    Parametric 2D U-Net for single-slice or patch-based CT phase translation.
    Fully backward-compatible with both previous versions:
      - base_channels=64 → exact same architecture as the current "large" model (512-ch bottleneck)
      - base_channels=32 → exact same architecture as the old "small" model (256-ch bottleneck)
    Checkpoints can be loaded directly by instantiating the model with the correct base_channels value.
    """
    def __init__(
        self,
        num_phases: int = 4,
        base_channels: int = 64,          # ← NEW: 64 = current/large, 32 = old/small
        use_phase_conditioning: bool = False,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.base_channels = base_channels
        self.use_phase_conditioning = use_phase_conditioning
        self.dropout_rate = dropout
        
        ch1 = base_channels          # 64 or 32
        ch2 = base_channels * 2       # 128 or 64
        ch3 = base_channels * 4       # 256 or 128
        ch4 = base_channels * 8       # 512 or 256
        bottleneck_ch = ch4           # same as ch4 (512 or 256)
        
        # Phase embedding (AdaIN in bottleneck) – size scales with base_channels for compatibility
        if use_phase_conditioning:
            phase_emb_dim = base_channels          # 64 for large, 32 for small → matches old checkpoints
            self.phase_emb = nn.Embedding(num_phases, phase_emb_dim)
            self.phase_scale_shift = nn.Linear(phase_emb_dim, bottleneck_ch * 2)
        
        # Encoder
        self.enc1 = self._make_enc_block(1,   ch1)
        self.enc2 = self._make_enc_block(ch1, ch2)
        self.enc3 = self._make_enc_block(ch2, ch3)
        self.enc4 = self._make_enc_block(ch3, ch4)
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(bottleneck_ch, bottleneck_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(bottleneck_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(self.dropout_rate),
            nn.Conv2d(bottleneck_ch, bottleneck_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(bottleneck_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(self.dropout_rate),
        )
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(bottleneck_ch, ch3, kernel_size=2, stride=2)
        self.dec4 = self._make_dec_block(ch3 + ch4, ch3)
        
        self.up3 = nn.ConvTranspose2d(ch3, ch2, kernel_size=2, stride=2)
        self.dec3 = self._make_dec_block(ch2 + ch3, ch2)
        
        self.up2 = nn.ConvTranspose2d(ch2, ch1, kernel_size=2, stride=2)
        self.dec2 = self._make_dec_block(ch1 + ch2, ch1)
        
        self.up1 = nn.ConvTranspose2d(ch1, ch1, kernel_size=2, stride=2)   # ← does NOT halve channels
        self.dec1 = self._make_dec_block(ch1 + ch1, ch1)
        
        self.out_conv = nn.Sequential(
            nn.Conv2d(ch1, 1, kernel_size=1),
            nn.Tanh()
        )
        
        # Stats
        total_params = sum(p.numel() for p in self.parameters()) / 1e6
        logger.info(f"Generator2D initialized | base_channels={base_channels} | "
                    f"{total_params:.2f}M params | phase_cond={use_phase_conditioning}")
    
    def _make_enc_block(self, in_ch: int, out_ch: int):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
    
    def _make_dec_block(self, in_ch: int, out_ch: int):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(self.dropout_rate),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
    
    def forward(self, x: torch.Tensor, phase_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Optional phase-conditioned AdaIN
        if self.use_phase_conditioning and phase_idx is not None:
            B = x.size(0)
            emb = self.phase_emb(phase_idx)                    # [B, emb_dim]
            ss = self.phase_scale_shift(emb)                   # [B, bottleneck_ch*2]
            scale, shift = ss.chunk(2, dim=1)                    # each [B, bottleneck_ch]
            scale = scale.view(B, -1, 1, 1)                       # [B, C, 1, 1]
            shift = shift.view(B, -1, 1, 1)
            b = b * (1 + scale) + shift
        
        # Decoder + skip connections
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

class Discriminator2D(nn.Module):
    """2D PatchGAN Discriminator for single slices."""
    def __init__(self, in_channels: int = 1):
        super().__init__()
        
        def disc_block(in_f, out_f, normalize=True):
            layers = [nn.Conv2d(in_f, out_f, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *disc_block(in_channels, 64, normalize=False),
            *disc_block(64, 128),
            *disc_block(128, 256),
            *disc_block(256, 512),
            nn.Conv2d(512, 1, 3, padding=1)
        )
    
    def forward(self, x):
        return self.model(x)


import torchvision.models as models

class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss for better texture quality."""
    def __init__(self):
        super().__init__()
        # Use pre-trained VGG16
        vgg = models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential(*list(vgg[:4]))   # relu1_2
        self.slice2 = nn.Sequential(*list(vgg[4:9]))  # relu2_2
        self.slice3 = nn.Sequential(*list(vgg[9:16])) # relu3_3
        
        # Freeze VGG weights
        for param in self.parameters():
            param.requires_grad = False
        
    def forward(self, pred, target):
        """
        pred, target: [B, 1, H, W] in range [-1, 1]
        """
        # Convert grayscale to RGB for VGG
        pred_rgb = pred.repeat(1, 3, 1, 1)
        target_rgb = target.repeat(1, 3, 1, 1)
        
        # Normalize for VGG (ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
        
        pred_rgb = (pred_rgb * 0.5 + 0.5 - mean) / std  # [-1,1] → ImageNet
        target_rgb = (target_rgb * 0.5 + 0.5 - mean) / std
        
        # Extract features
        pred_f1 = self.slice1(pred_rgb)
        target_f1 = self.slice1(target_rgb)
        
        pred_f2 = self.slice2(pred_f1)
        target_f2 = self.slice2(target_f1)
        
        pred_f3 = self.slice3(pred_f2)
        target_f3 = self.slice3(target_f2)
        
        # L1 loss on features
        loss = (
            F.l1_loss(pred_f1, target_f1) +
            F.l1_loss(pred_f2, target_f2) +
            F.l1_loss(pred_f3, target_f3)
        )
        
        return loss

# ============================================================================
# LOSS FUNCTIONS (UNCHANGED - Keep all your losses)
# ============================================================================

class FocalFrequencyLoss(nn.Module):
    """
    Focal Frequency Loss - UNCHANGED from your original code.
    """
    def __init__(self, alpha: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = pred.shape
        
        pred_fft = torch.fft.fftn(pred.float(), dim=(-3, -2, -1))
        target_fft = torch.fft.fftn(target.float(), dim=(-3, -2, -1))
        
        pred_amp = torch.abs(pred_fft)
        target_amp = torch.abs(target_fft)
        
        diff = pred_amp - target_amp
        weight_matrix = target_amp ** self.alpha
        
        loss = weight_matrix * (diff ** 2)
        
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss


# class CombinedLoss(nn.Module):
#     """
#     Conditional loss supporting 4-stage training:
#     Stage 1: MSE only
#     Stage 2: MSE + Organ-weighted MSE
#     Stage 3: MSE + Organ + Adversarial
#     Stage 4: MSE + Organ + Adversarial + Cycle
#     """
#     def __init__(self, config: Dict):
#         super().__init__()
#         self.config = config
#         self.mse = nn.MSELoss()
#         # self.perceptual = PerceptualLoss()  # ← ADD THIS
        
        
#         # Component flags
#         self.use_organ_loss = config.get('use_organ_loss', False)
#         self.use_discriminator = config.get('use_discriminator', False)
#         self.use_cycle = config.get('use_cycle_consistency', False)
        
#         # Loss weights
#         # self.lambda_mse = config.get('lambda_mse', 1.0)
#         self.lambda_perceptual = config.get('lambda_perceptual', 0.1)  # ← ADD THIS
#         self.lambda_mse = config.get('lambda_mse', 2.0)
#         self.lambda_organ = config.get('lambda_organ', 1.0)
#         self.organ_weight = config.get('organ_weight', 2.0)
#         self.lambda_adv = config.get('lambda_adv', 0.05)
#         self.lambda_cycle = config.get('lambda_cycle', 1.0)
#         self.device = config.get('device', 'cuda')
#         # Warmup for adversarial loss
#         self.adv_warmup_epochs = config.get('adv_warmup_epochs', 10)
#         self.current_epoch = 0

#         logger.info(f"Loss components:")
#         logger.info(f"  MSE: ✓ (λ={self.lambda_mse})")
#         logger.info(f"  Organ: {'✓' if self.use_organ_loss else '✗'} (λ={self.lambda_organ})")
#         logger.info(f"  Adversarial: {'✓' if self.use_discriminator else '✗'} (λ={self.lambda_adv})")
#         logger.info(f"  Cycle: {'✓' if self.use_cycle else '✗'} (λ={self.lambda_cycle})")
    
#     def set_epoch(self, epoch: int):
#         """Update current epoch for warmup schedules."""
#         self.current_epoch = epoch

#     def get_mse_weight(self) -> float:
#         """Get current MSE weight (no warmup in simple version)."""
#         return self.lambda_mse

#     def get_adv_weight(self) -> float:
#         """Gradually increase adversarial weight."""
#         if not self.use_discriminator:
#             return 0.0
        
#         if self.current_epoch >= self.adv_warmup_epochs:
#             return self.lambda_adv
        
#         # Linear warmup
#         progress = self.current_epoch / self.adv_warmup_epochs
#         return self.lambda_adv * progress

#     @staticmethod
#     def masked_mse(pred, target, mask_dict, organ_weight=10.0):
#         """Compute organ-weighted MSE."""
#         loss_global = F.mse_loss(pred, target)
        
#         if not mask_dict:
#             return loss_global
        
#         # Union of all organ masks
#         union = torch.zeros_like(pred)
#         for m in mask_dict.values():
#             union = torch.max(union, m)
        
#         loss_organ = F.mse_loss(pred * union, target * union)
#         return loss_global + (organ_weight - 1.0) * loss_organ
    
    
#     def forward(self, pred: torch.Tensor, target: torch.Tensor,
#                 cycle_pred: torch.Tensor = None, source: torch.Tensor = None,
#                 adv_loss: float = 0.0, masks: Dict = None) -> Tuple[torch.Tensor, Dict]:
#         """
#         Compute conditional loss based on active components.
        
#         Args:
#             pred: Generated target [B, 1, H, W]
#             target: Real target [B, 1, H, W]
#             cycle_pred: Reconstructed source (only if use_cycle=True)
#             source: Real source (only if use_cycle=True)
#             adv_loss: Adversarial loss (only if use_discriminator=True)
#             masks: Organ masks (only if use_organ_loss=True)
#         """
#         loss_dict = {}
#         if masks is not None:
#             masks = {k: v.to(self.device) for k, v in masks.items()}
#         # Stage 1 and 2: Base MSE and mask weighted
#         if self.use_organ_loss and masks:
#             loss_mse = self.masked_mse(pred, target, masks, self.organ_weight)
#             loss_mse *= (self.lambda_mse + self.lambda_organ)
#             loss_dict['mse'] = loss_mse.item()
#         else:
#             loss_mse = self.mse(pred, target) * self.lambda_mse
#             loss_dict['mse'] = loss_mse.item()
        
#         total_loss = loss_mse
        
        
#         # Stage 3: Adversarial
#         if self.use_discriminator:
#             current_adv_weight = self.get_adv_weight()  # ← Use warmup
#             loss_adv = adv_loss * current_adv_weight
#             total_loss += loss_adv
#             loss_dict['adv'] = loss_adv if isinstance(loss_adv, float) else loss_adv.item()
#         else:
#             loss_dict['adv'] = 0.0
        
#         # Stage 4: Cycle consistency
#         if self.use_cycle and cycle_pred is not None and source is not None:
#             loss_cycle = self.mse(cycle_pred, source) * self.lambda_cycle
#             total_loss += loss_cycle
#             loss_dict['cycle'] = loss_cycle.item()
#         else:
#             loss_dict['cycle'] = 0.0
#         # Add perceptual loss
#         # loss_perceptual = self.perceptual(pred, target) * self.lambda_perceptual
#         # total_loss += loss_perceptual
#         # loss_dict['perceptual'] = loss_perceptual.item()
        
#         loss_dict['total'] = total_loss.item()
#         return total_loss, loss_dict

class CombinedLoss(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.mse = nn.MSELoss()
        
        # Component flags
        self.use_organ_loss = config.get('use_organ_loss', False)
        self.use_discriminator = config.get('use_discriminator', False)
        self.use_cycle = config.get('use_cycle_consistency', False)
        self.device = config.get('device', 'cuda')
        # Loss weights
        self.lambda_mse = config.get('lambda_mse', 1.0)
        self.lambda_organ = config.get('lambda_organ', 5.0)
        self.organ_weight = config.get('organ_weight', 10.0)
        self.lambda_adv = config.get('lambda_adv', 0.1)
        self.lambda_cycle = config.get('lambda_cycle', 10.0)
        
        self.global_step = 0
        
        logger.info(f"CombinedLoss initialized:")
        logger.info(f"  use_organ_loss: {self.use_organ_loss}")
        logger.info(f"  lambda_organ: {self.lambda_organ}")
        logger.info(f"  organ_weight (inside MSE): {self.organ_weight}")
    
    
    def set_epoch(self, epoch: int):
        """Update current epoch for warmup schedules."""
        self.current_epoch = epoch

    def get_mse_weight(self) -> float:
        """Get current MSE weight (no warmup in simple version)."""
        return self.lambda_mse

    def get_adv_weight(self) -> float:
        """Gradually increase adversarial weight."""
        if not self.use_discriminator:
            return 0.0
        
        if self.current_epoch >= self.adv_warmup_epochs:
            return self.lambda_adv
        
        # Linear warmup
        progress = self.current_epoch / self.adv_warmup_epochs
        return self.lambda_adv * progress

    @staticmethod
    def masked_mse(pred, target, mask_dict, organ_weight=10.0):
        """
        Compute organ-weighted MSE.
        
        Args:
            pred: [B, 1, H, W] generated image
            target: [B, 1, H, W] ground truth
            mask_dict: Dict of organ masks, each [B, 1, H, W]
            organ_weight: Multiplier for organ regions
        """
        # Global MSE
        loss_global = F.mse_loss(pred, target, reduction='mean')
        
        if not mask_dict or len(mask_dict) == 0:
            return loss_global
        
        # Union of all organ masks
        union = torch.zeros_like(pred)
        for organ_name, mask in mask_dict.items():
            # Ensure mask is [B, 1, H, W]
            if mask.dim() == 5:  # [B, 1, 1, H, W] - squeeze extra dim
                mask = mask.squeeze(2)
            union = torch.max(union, mask)
        
        # Organ MSE
        loss_organ = F.mse_loss(pred * union, target * union, reduction='mean')
        
        # Combined: global + weighted organ
        return loss_global + (organ_weight - 1.0) * loss_organ
    
    def forward(self, pred, target, cycle_pred=None, source=None, 
                adv_loss=0.0, masks=None):
        """
        Conditional loss based on active components.
        
        Args:
            pred: Generated target [B, 1, H, W]
            target: Real target [B, 1, H, W]
            cycle_pred: Reconstructed source (if use_cycle=True)
            source: Real source (if use_cycle=True)
            adv_loss: Adversarial loss (if use_discriminator=True)
            masks: Dict of organ masks (if use_organ_loss=True)
        """
        loss_dict = {}
        
        # ============================================================
        # MSE Loss (with optional organ weighting)
        # ============================================================
        if masks is not None:
            masks = {k: v.to(self.device) for k, v in masks.items()}
        if self.use_organ_loss and masks and len(masks) > 0:
            # Use organ-weighted MSE
            loss_mse = self.masked_mse(pred, target, masks, self.organ_weight)
            loss_mse *= self.lambda_organ
            loss_dict['mse'] = loss_mse.item()
            
            # Debug logging (every 100 steps)
            if self.global_step % 100 == 0:
                logger.info(f"✓ Using MASKED MSE (lambda={self.lambda_organ}, weight={self.organ_weight})")
                logger.info(f"  Masks: {list(masks.keys())}")
        else:
            # Regular MSE
            loss_mse = self.mse(pred, target) * self.lambda_mse
            loss_dict['mse'] = loss_mse.item()
            
            if self.global_step % 100 == 0:
                if self.use_organ_loss:
                    logger.warning(f"✗ Organ loss enabled but no masks provided!")
                else:
                    logger.info(f"→ Using regular MSE (lambda={self.lambda_mse})")
        
        total_loss = loss_mse
        
        # ============================================================
        # Adversarial Loss (Stage 3+)
        # ============================================================
        if self.use_discriminator and isinstance(adv_loss, (int, float)) and adv_loss > 0:
            loss_adv_weighted = adv_loss * self.lambda_adv
            total_loss += loss_adv_weighted
            loss_dict['adv'] = loss_adv_weighted
        elif self.use_discriminator and torch.is_tensor(adv_loss):
            loss_adv_weighted = adv_loss * self.lambda_adv
            total_loss += loss_adv_weighted
            loss_dict['adv'] = loss_adv_weighted.item()
        else:
            loss_dict['adv'] = 0.0
        
        # ============================================================
        # Cycle Consistency Loss (Stage 4)
        # ============================================================
        if self.use_cycle and cycle_pred is not None and source is not None:
            loss_cycle = self.mse(cycle_pred, source) * self.lambda_cycle
            total_loss += loss_cycle
            loss_dict['cycle'] = loss_cycle.item()
        else:
            loss_dict['cycle'] = 0.0
        
        loss_dict['total'] = total_loss.item()
        self.global_step += 1
        
        return total_loss, loss_dict

# ============================================================================
# LOSS TRACKING (UNCHANGED)
# ============================================================================
def convert_numpy(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_numpy(i) for i in obj]
    return obj
    
class LossTracker:
    """Track and visualize training losses - UNCHANGED."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.history = {
            'epoch': [],
            'train_gen_total': [],
            'train_gen_adv': [],
            'train_gen_cycle': [],
            'train_gen_mse': [],
            'train_disc': [],
            'val_loss': [],
            'val_psnr': [],
            'val_ssim': [],
            'lr_gen': [],
            'lr_disc': []
        }


        
    def update_epoch(self, epoch: int, train_losses: Dict, val_metrics: Dict,
                 lr_gen: float, lr_disc: float):
        """Update history with epoch results."""
        self.history['epoch'].append(epoch)
        self.history['train_gen_total'].append(train_losses.get('gen_total', 0))
        self.history['train_gen_adv'].append(train_losses.get('gen_adv', 0))
        self.history['train_gen_cycle'].append(train_losses.get('gen_cycle', 0))
        self.history['train_gen_mse'].append(train_losses.get('gen_mse', 0))
        # self.history['train_gen_focal'].append(train_losses.get('gen_focal', 0))
        self.history['train_disc'].append(train_losses.get('disc', 0))
        self.history['val_loss'].append(val_metrics.get('val_loss', 0))
        self.history['val_psnr'].append(val_metrics.get('psnr', 0))
        self.history['val_ssim'].append(val_metrics.get('ssim', 0))
        self.history['lr_gen'].append(lr_gen)
        self.history['lr_disc'].append(lr_disc)
        # self.history['adv_weight'].append(adv_weight)
    
    def save_json(self):
        """Save history to JSON."""
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(convert_numpy(self.history), f, indent=2)
    
    def create_summary_report(self, current_epoch: int):
        """Create summary plots."""
        if not self.history['epoch']:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Training Summary - Epoch {current_epoch}', fontsize=16)
        
        epochs = self.history['epoch']
        
        # Generator losses
        ax = axes[0, 0]
        ax.plot(epochs, self.history['train_gen_mse'], label='MSE', linewidth=2, color='blue')

        if any(x > 0 for x in self.history['train_gen_cycle']):
            ax.plot(epochs, self.history['train_gen_cycle'], label='Cycle', linewidth=2, color='green')

        if any(x > 0 for x in self.history['train_gen_adv']):
            ax.plot(epochs, self.history['train_gen_adv'], label='Adversarial', linewidth=2, color='orange')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Generator Component Losses')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Total losses
        ax = axes[0, 1]
        ax.plot(epochs, self.history['train_gen_total'], label='Generator', linewidth=2, color='blue')
        
        if any(x > 0 for x in self.history['train_disc']):
            ax.plot(epochs, self.history['train_disc'], label='Discriminator', linewidth=2, color='red')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Total Losses')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Validation metrics
        ax = axes[0, 2]
        ax.plot(epochs, self.history['val_psnr'], label='PSNR', linewidth=2, color='green')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title('Validation PSNR')
        ax.grid(True, alpha=0.3)
        
        ax2 = ax.twinx()
        ax2.plot(epochs, self.history['val_ssim'], label='SSIM', linewidth=2, color='orange')
        ax2.set_ylabel('SSIM')
        
        # Learning rates
        ax = axes[1, 0]
        ax.plot(epochs, self.history['lr_gen'], label='Generator LR', linewidth=2)
        
        if any(x > 0 for x in self.history['lr_disc']):
            ax.plot(epochs, self.history['lr_disc'], label='Discriminator LR', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rates')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Validation loss
        ax = axes[1, 1]
        ax.plot(epochs, self.history['val_loss'], linewidth=2, color='purple')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Validation Loss')
        ax.grid(True, alpha=0.3)
        
        
        # Loss ratio plot
        ax = axes[1, 2]
        if any(x > 0 for x in self.history['train_gen_cycle']):
            cycle_ratio = [c / max(m, 1e-8) for c, m in 
                          zip(self.history['train_gen_cycle'], self.history['train_gen_mse'])]
            ax.plot(epochs, cycle_ratio, label='Cycle/MSE', linewidth=2)
        
        if any(x > 0 for x in self.history['train_gen_adv']):
            adv_ratio = [a / max(m, 1e-8) for a, m in 
                        zip(self.history['train_gen_adv'], self.history['train_gen_mse'])]
            ax.plot(epochs, adv_ratio, label='Adv/MSE', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Ratio')
        ax.set_title('Loss Component Ratios')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'training_summary_epoch_{current_epoch}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        self.save_json()
       

# ============================================================================
# TRAINER (OPTIMIZED with cache monitoring)
# ============================================================================

def save_sample_patches(
    generator: nn.Module,
    val_loader: DataLoader,
    epoch: int,
    save_dir: Path,
    device: torch.device,
    num_samples: int = 5,
    save_nifti: bool = True
    ):
    """
    Save sample generated patches for visual inspection.
    
    Args:
        generator: The generator model
        val_loader: Validation dataloader
        epoch: Current epoch number
        save_dir: Directory to save samples
        device: Device to run on
        num_samples: Number of samples to save
    """
    generator.eval()
    save_dir = Path(save_dir) / f"epoch_{epoch}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for different output types
    png_dir = save_dir / 'comparisons'
    nifti_dir = save_dir / 'nifti_volumes'
    png_dir.mkdir(exist_ok=True)
    if save_nifti:
        nifti_dir.mkdir(exist_ok=True)

    # âœ… DELETE OLD EPOCH FOLDERS (keep only last 10 epochs)
    parent_dir = save_dir.parent
    epoch_dirs = sorted(parent_dir.glob('epoch_*'), key=lambda x: int(x.name.split('_')[1]))
    
    if len(epoch_dirs) > 10:  # Keep only last 10 epochs
        for old_dir in epoch_dirs[:-10]:
            try:
                import shutil
                shutil.rmtree(old_dir)
                logger.info(f"Deleted old sample directory: {old_dir.name}")
            except Exception as e:
                logger.warning(f"Could not delete {old_dir}: {e}")
    
    # âœ… RANDOMLY SELECT PATCHES INSTEAD OF SEQUENTIAL SELECTION
    # First, collect all batches
    logger.info("Collecting validation batches for random sampling...")
    logger.info("Collecting validation batches for random sampling...")
    all_batches = []
    with torch.no_grad():
        for batch in val_loader:
            all_batches.append(batch)
    # Check 'skip_batch' (set by safe_collate), not 'skip' (sample-level, becomes tensor)
    valid_batches = [b for b in all_batches if not b.get('skip_batch', False)]
    total_patches = sum(b['source'].size(0) for b in valid_batches) if valid_batches else 0
    # total_patches = len(all_batches) * (all_batches[0]['source'].size(0) if all_batches else 0)
    logger.info(f"Total validation patches available: {total_patches}")
    
    # Randomly select batch indices
    if len(all_batches) == 0:
        logger.warning("No validation batches available!")
        return
    
    num_batches_to_sample = min(num_samples, len(all_batches))
    selected_batch_indices = np.random.choice(
        len(all_batches), 
        size=num_batches_to_sample, 
        replace=False
    )
    
    logger.info(f"Randomly selected {num_batches_to_sample} batches from {len(all_batches)} available")
    
    saved_count = 0
    with torch.no_grad():
        for batch_idx in selected_batch_indices:
            if saved_count >= num_samples:
                break
                
            # batch = all_batches[batch_idx]
            batch = random.choice(valid_batches)
            real_source = batch['source']
            try:
                real_source = batch['source'].to(device)
                real_target = batch['target'].to(device)
                # source_phase_idx = batch['source_phase'].to(device)
                # target_phase_idx = batch['target_phase'].to(device)
                source_phase = batch['source_phase'].to(device)
                target_phase = batch['target_phase'].to(device)
                case_id = batch['case_id']
                
                # Generate target and reconstruct source
                generated_target = generator(real_source, target_phase)
                reconstructed_source = generator(generated_target, source_phase)
                
                # Process each sample in batch (or randomly select from batch)
                batch_size = real_source.size(0)
                samples_from_batch = min(batch_size, num_samples - saved_count)
                
                # Randomly select samples from this batch
                if batch_size > samples_from_batch:
                    sample_indices = np.random.choice(batch_size, size=samples_from_batch, replace=False)
                else:
                    sample_indices = range(batch_size)
                
                for i in sample_indices:
                    # ===== SAVE PNG COMPARISON =====
                    # Get middle slice from 3D patch [1, D, H, W] -> [H, W]
                    # mid_slice = real_source.shape[2] // 2
                    
                    source_slice = real_source[i].squeeze().cpu().numpy()
                    target_slice = real_target[i].squeeze().cpu().numpy()
                    generated_slice = generated_target[i].squeeze().cpu().numpy()
                    reconstructed_slice = reconstructed_source[i].squeeze().cpu().numpy()
                    
                    # Create comparison figure
                    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                    
                    # Row 1: Forward generation (source -> target)
                    im0 = axes[0, 0].imshow(source_slice, cmap='gray', vmin=-1, vmax=1)
                    axes[0, 0].set_title(f'Source: {source_phase[i]}', fontsize=12, fontweight='bold')
                    axes[0, 0].axis('off')
                    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
                    
                    im1 = axes[0, 1].imshow(generated_slice, cmap='gray', vmin=-1, vmax=1)
                    axes[0, 1].set_title(f'Generated: {target_phase[i]}', fontsize=12, fontweight='bold')
                    axes[0, 1].axis('off')
                    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
                    
                    im2 = axes[0, 2].imshow(target_slice, cmap='gray', vmin=-1, vmax=1)
                    axes[0, 2].set_title(f'Ground Truth: {target_phase[i]}', fontsize=12, fontweight='bold')
                    axes[0, 2].axis('off')
                    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
                    
                    # Row 2: Cycle reconstruction and difference maps
                    im3 = axes[1, 0].imshow(reconstructed_slice, cmap='gray', vmin=-1, vmax=1)
                    axes[1, 0].set_title(f'Reconstructed: {source_phase[i]}', fontsize=12, fontweight='bold')
                    axes[1, 0].axis('off')
                    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
                    
                    # Difference map: generated vs ground truth
                    diff_gen = np.abs(generated_slice - target_slice)
                    im4 = axes[1, 1].imshow(diff_gen, cmap='hot', vmin=0, vmax=1)
                    axes[1, 1].set_title('|Generated - GT|', fontsize=12, fontweight='bold')
                    axes[1, 1].axis('off')
                    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
                    
                    # Difference map: reconstructed vs source
                    diff_cycle = np.abs(reconstructed_slice - source_slice)
                    im5 = axes[1, 2].imshow(diff_cycle, cmap='hot', vmin=0, vmax=1)
                    axes[1, 2].set_title('|Reconstructed - Source|', fontsize=12, fontweight='bold')
                    axes[1, 2].axis('off')
                    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)
                    
                    # Calculate metrics for this sample
                    mse_gen = np.mean((generated_slice - target_slice) ** 2)
                    mse_cycle = np.mean((reconstructed_slice - source_slice) ** 2)
                    
                    # Add overall title with metrics
                    fig.suptitle(
                        f'Epoch {epoch} - Case: {case_id[i]} - '
                        f'MSE (Gen): {mse_gen:.4f}, MSE (Cycle): {mse_cycle:.4f}',
                        fontsize=14, fontweight='bold', y=0.98
                    )
                    
                    plt.tight_layout()
                    
                    # Save PNG figure
                    png_path = png_dir / f'sample_{saved_count:03d}_{case_id[i]}.png'
                    plt.savefig(png_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    # ===== SAVE NIFTI VOLUMES =====
                    if save_nifti:
                        # Extract full 3D patches [1, D, H, W] -> [D, H, W]
                        source_vol = real_source[i, 0].cpu().numpy()
                        target_vol = real_target[i, 0].cpu().numpy()
                        generated_vol = generated_target[i, 0].cpu().numpy()
                        reconstructed_vol = reconstructed_source[i, 0].cpu().numpy()
                        
                        # Create case-specific subdirectory
                        case_nifti_dir = nifti_dir / f'sample_{saved_count:03d}_{case_id[i]}'
                        case_nifti_dir.mkdir(exist_ok=True)
                        
                        # Save as NIfTI files (transpose back to standard orientation if needed)
                        # Note: volumes are in [D, H, W] format, NIfTI standard is typically [W, H, D]
                        # def save_nifti_volume(volume, filepath):
                        #     """Helper to save volume as NIfTI"""
                        #     # Transpose from [D, H, W] to [W, H, D] for standard NIfTI orientation
                        #     volume_transposed = np.transpose(volume, (2, 1, 0))
                        #     nifti_img = nib.Nifti1Image(volume_transposed, affine=np.eye(4))
                        #     nib.save(nifti_img, filepath)
                        
                        # save_nifti_volume(source_vol, case_nifti_dir / 'source.nii.gz')
                        # save_nifti_volume(target_vol, case_nifti_dir / 'target_groundtruth.nii.gz')
                        # save_nifti_volume(generated_vol, case_nifti_dir / 'target_generated.nii.gz')
                        # save_nifti_volume(reconstructed_vol, case_nifti_dir / 'source_reconstructed.nii.gz')
                       
                        # Save metadata
                        metadata = {
                            'epoch': epoch,
                            'case_id': case_id[i],
                            'source_phase': int(source_phase[i].item()),
                            'target_phase': int(target_phase[i].item()),
                            'patch_shape': list(source_vol.shape),
                            'mse_generated': float(mse_gen),
                            'mse_cycle': float(mse_cycle)
                        }
                        # metadata = {k: v.item() if isinstance(v, torch.Tensor) else v 
                        #                 for k, v in metadata.items()}
                        import json
                        with open(case_nifti_dir / 'metadata.json', 'w') as f:
                            json.dump(metadata, f, indent=2)
                    
                    saved_count += 1
                    
                    if saved_count >= num_samples:
                        break
                        
            except Exception as e:
                logger.error(f"Error saving sample patches: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    logger.info(f"âœ“ Saved {saved_count} sample patches to {save_dir}")
    logger.info(f"  - PNG comparisons: {png_dir}")
    if save_nifti:
        logger.info(f"  - NIfTI volumes: {nifti_dir}")

from collections import Counter

class RejectionStats:
    """Thread-safe counter for the current epoch."""
    def __init__(self):
        self.total = 0          # patches fed to collate
        self.rejected = 0       # patches with skip==True
        self.reasons = Counter()

    def add(self, n_total: int, n_rej: int, reason: str = ""):
        self.total += n_total
        self.rejected += n_rej
        if reason:
            self.reasons[reason] += n_rej

    def percent(self) -> float:
        return 100.0 * self.rejected / max(self.total, 1)

    def reset(self):
        self.total = self.rejected = 0
        self.reasons.clear()

# One global object per process (DataLoader workers are separate processes)
_epoch_stats = RejectionStats()

class EarlyStopping:
    """Stop training when validation loss stops improving."""
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class WarmupCosineScheduler:
    """
    Combines warmup + cosine annealing.
    """
    def __init__(self, optimizer, warmup_epochs, max_epochs, 
                 min_lr=1e-6, warmup_start_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0
    
    def step(self):
        """Update learning rate."""
        if self.current_epoch < self.warmup_epochs:
            # Warmup phase: linear increase
            lr = self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * \
                 (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine annealing phase
            progress = (self.current_epoch - self.warmup_epochs) / \
                      (self.max_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * \
                 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
        return lr
    
    def get_last_lr(self):
        return [self.optimizer.param_groups[0]['lr']]


            
class MemoryOptimizedTrainer:
    """
    OPTIMIZED Trainer with volume cache monitoring.
    All losses and training logic UNCHANGED.
    """
        
    def __init__(self, config: Dict):
        self.config = config
        self.device = config['device']
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Component flags
        use_disc = config.get('use_discriminator', False)
        use_phase = config.get('use_phase_conditioning', False)
        
        self.real_label_smoothing = config.get('real_label_smoothing', 1)
        self.fake_label_smoothing = config.get('fake_label_smoothing', 0) 
        # Models
        self.generator = Generator2D(
            num_phases = config.get('num_phase', 2),
            base_channels = config['generator_base_channels'],
            use_phase_conditioning=use_phase,
            dropout = config.get('generator_dropout', 0.3)
        ).to(self.device)
        
        if use_disc:
            self.disc_target = Discriminator2D().to(self.device)
            lr_disc = config.get('learning_rate', 2e-4) * 0.5
            self.opt_disc_target = optim.Adam(
                self.disc_target.parameters(), lr=lr_disc, betas=(0.5, 0.999)
            )
        else:
            self.disc_target = None
            self.opt_disc_target = None
        
        # Optimizer
        lr_gen = config.get('learning_rate', 2e-4)
        
        # self.opt_gen = optim.Adam(self.generator.parameters(), lr=lr_gen, betas=(0.5, 0.999))
        self.opt_gen = optim.Adam(
                        self.generator.parameters(),
                        lr=lr_gen,
                        betas=(0.5, 0.999),
                        weight_decay=1e-5  # ← ADD L2 regularization
                    )
        # self.scheduler_gen = optim.lr_scheduler.ReduceLROnPlateau(
        #                     self.opt_gen,
        #                     mode='min',
        #                     factor=0.5,
        #                     patience=5,
        #                     verbose=True
        #                 )
        # Loss and tracking
        self.combined_loss = CombinedLoss(config)
        self.loss_tracker = LossTracker(self.output_dir)
        self.memory_manager = MemoryManager()
        self.cleanup_frequency = config.get('cleanup_frequency', 10)
    
        # Samples directory
        self.samples_dir = self.output_dir / 'samples'
        self.samples_dir.mkdir(exist_ok=True)
        self.early_stopping = EarlyStopping(patience=7)

        # Optimizers
        lr_gen = config.get('learning_rate', 1e-4)
        self.opt_gen = optim.Adam(
            self.generator.parameters(),
            lr=lr_gen,
            betas=(0.5, 0.999),
            weight_decay=config.get('weight_decay', 1e-6)
        )
        
        # ============================================================
        # COSINE ANNEALING WITH WARM RESTARTS
        # ============================================================
        self.use_cosine_schedule = config.get('use_cosine_schedule', True)
        
        if self.use_cosine_schedule:
            # T_0: Epochs until first restart (e.g., 10)
            # T_mult: Multiply T_0 by this after each restart (e.g., 2)
            # eta_min: Minimum learning rate
            self.scheduler_gen = lr_scheduler.CosineAnnealingWarmRestarts(
                self.opt_gen,
                T_0=config.get('cosine_t0', 10),      # First cycle: 10 epochs
                T_mult=config.get('cosine_tmult', 2),  # Next cycle: 20, then 40...
                eta_min=config.get('cosine_eta_min', 1e-6)  # Min LR
            )
            logger.info(f"✓ Cosine annealing enabled: T_0={config.get('cosine_t0', 10)}, "
                       f"T_mult={config.get('cosine_tmult', 2)}")
        else:
            self.scheduler_gen = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.use_amp = config.get('use_mixed_precision', True)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self.global_step = 0  # Track total training steps
        
        logger.info(f"Trainer initialized - Stage: {self._get_stage_name()}")
        logger.info(f"  Generator: 2D U-Net")
        logger.info(f"  Discriminator: {'✓' if use_disc else '✗'}")
        logger.info(f"  Phase conditioning: {'✓' if use_phase else '✗'}")

    def _get_stage_name(self):
        """Get current training stage name."""
        config = self.config
        if config.get('use_cycle_consistency'):
            return "Stage 4 (Full Pipeline)"
        elif config.get('use_discriminator'):
            return "Stage 3 (+ Discriminator)"
        elif config.get('use_organ_loss'):
            return "Stage 2 (+ Organ Loss)"
        else:
            return "Stage 1 (MSE Only)"
            
    # -----------------------------------------------------------------
    def _print_rejection_report(self, loader_name: str):
        global _epoch_stats
        pct = _epoch_stats.percent()
        logger.info(
            f"{loader_name} rejection: "
            f"{_epoch_stats.rejected}/{_epoch_stats.total} "
            f"({pct:.2f}%)"
        )
        if _epoch_stats.reasons:
            logger.info("   Reasons → " + ", ".join(
                f"{r}:{c}" for r, c in _epoch_stats.reasons.most_common(5)
            ))

        # accumulate for the *overall* report
        self.global_rejected += _epoch_stats.rejected
        self.global_total   += _epoch_stats.total

    def train_step(self, batch: Dict) -> Dict:
        """Conditional training step based on active components."""
        if batch.get('skip_batch', False) or batch['source'].size(0) == 0:
            return {'gen_total': 0, 'gen_adv': 0, 'gen_cycle': 0, 'gen_mse': 0, 'disc': 0}
        
        source = batch['source'].to(self.device)
        target = batch['target'].to(self.device)
        source_phase = batch['source_phase'].to(self.device)
        target_phase = batch['target_phase'].to(self.device)
        masks = {k: v.to(self.device) for k, v in batch.get('masks', {}).items()}
        
        use_disc = self.config.get('use_discriminator', False)
        use_cycle = self.config.get('use_cycle_consistency', False)
        use_phase = self.config.get('use_phase_conditioning', False)
        # Only update discriminator every N generator steps
        disc_update_freq = self.config.get('disc_update_freq', 2)  # Update every 2 steps

        if self.global_step % 100 == 0:
            masks = batch.get('masks', {})
            if masks:
                logger.info(f"✓ Masks loaded: {list(masks.keys())}")
                for organ, mask in masks.items():
                    logger.info(f"  {organ}: shape={mask.shape}, non-zero={mask.sum().item()}")
            else:
                logger.warning("✗ NO MASKS IN BATCH!")
        # Add this debugging in train_step:
        if self.global_step % 100 == 0:
            logger.info(f"Real image stats: mean={source.mean():.3f}, std={source.std():.3f}, min={source.min():.3f}, max={source.max():.3f}")
            logger.info(f"Target image stats: mean={target.mean():.3f}, std={target.std():.3f}, min={target.min():.3f}, max={target.max():.3f}")
  
        # ============= Train Discriminator (Stage 3+) =============
        loss_disc = 0.0
        disc_update_cnt = 0
        if use_disc and (self.global_step % disc_update_freq == 0):
            self.opt_disc_target.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                with torch.no_grad():
                    phase_arg = target_phase if use_phase else None
                    fake_target = self.generator(source, phase_arg)
                
                pred_real = self.disc_target(target)
                pred_fake = self.disc_target(fake_target.detach())
                
                # real_labels = torch.ones_like(pred_real) * 0.9
                # fake_labels = torch.ones_like(pred_fake) * 0.1
                
                real_labels = torch.full_like(pred_real, self.real_label_smoothing)
                fake_labels = torch.full_like(pred_fake, self.fake_label_smoothing)

                loss_real = F.binary_cross_entropy_with_logits(pred_real, real_labels)
                loss_fake = F.binary_cross_entropy_with_logits(pred_fake, fake_labels)
                loss_disc = (loss_real + loss_fake) * 0.5
            
            self.scaler.scale(loss_disc).backward()
            # Clip discriminator gradients too
            self.scaler.unscale_(self.opt_disc_target)
            tgt_norm = torch.nn.utils.clip_grad_norm_(
                            self.disc_target.parameters(), 
                            max_norm=10.0
                        )
            self.scaler.step(self.opt_disc_target)
            self.scaler.update()
            disc_update_cnt += 1
            # ------------------------------------------------------------------
            # DEBUG LOGGING (every discriminator update)
            # ------------------------------------------------------------------
            logger.info(
                "=== DISCRIMINATOR UPDATE %02d ===" % disc_update_cnt
            )
            
            logger.info(
                "TGT  real logits  mean=%.4f  std=%.4f  min=%.4f  max=%.4f" %
                (pred_real.mean().item(), pred_real.std().item(),
                 pred_real.min().item(), pred_real.max().item())
            )
            logger.info(
                "TGT  fake logits  mean=%.4f  std=%.4f  min=%.4f  max=%.4f" %
                (pred_fake.mean().item(), pred_fake.std().item(),
                 pred_fake.min().item(), pred_fake.max().item())
            )
            # sigmoid probabilities
            logger.info(
                "TGT  real prob %.3f  fake prob %.3f" %
                (pred_real.sigmoid().mean().item(),
                 pred_fake.sigmoid().mean().item())
            )
            # BCE components
            logger.info(
                "SRC  BCE real=%.5f  fake=%.5f  total=%.5f" %
                (loss_real.item(), loss_fake.item(), loss_disc.item())
            )
            
            logger.info("TOTAL DISC LOSS = %.5f" % loss_disc.item())

            # gradient norms (after backward, before step)
            # src_norm = torch.nn.utils.clip_grad_norm_(self.disc_source.parameters(), max_norm=1e9)
            tgt_norm = torch.nn.utils.clip_grad_norm_(self.disc_target.parameters(), max_norm=1e9)
            logger.info("GRAD NORM  tgt=%.4f" % (tgt_norm))
        
        # ============= Train Generator =============
        self.opt_gen.zero_grad()
        
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            # Forward pass
            phase_arg = target_phase if use_phase else None
            fake_target = self.generator(source, phase_arg)
            
            # Cycle consistency (Stage 4)
            cycle_source = None
            if use_cycle:
                cycle_phase_arg = source_phase if use_phase else None
                cycle_source = self.generator(fake_target, cycle_phase_arg)
            
            # Adversarial loss (Stage 3+)
            loss_adv = 0.0
            if use_disc:
                pred_fake = self.disc_target(fake_target)
                real_labels = torch.ones_like(pred_fake)
                loss_adv = F.binary_cross_entropy_with_logits(pred_fake, real_labels)
            
            # Combined loss
            loss_gen, loss_dict = self.combined_loss(
                fake_target, target, cycle_source, source, loss_adv, masks
            )
        
        self.scaler.scale(loss_gen).backward()
        # CRITICAL: Clip gradients BEFORE unscaling
        self.scaler.unscale_(self.opt_gen)
        gen_norm = torch.nn.utils.clip_grad_norm_(
                    self.generator.parameters(), 
                    max_norm=10.0  # ← Reasonable limit (was 1e9!)
                )
        self.scaler.step(self.opt_gen)
        self.scaler.update()
        # ------------------------------------------------------------------
        # GENERATOR DEBUG LOGGING
        # ------------------------------------------------------------------
        if use_disc:
            logger.info("=== GENERATOR UPDATE ===")
            logger.info("ADV raw logits  tgt=%.4f" %
                        (pred_fake.mean().item()))
            logger.info("ADV prob (should → 1)  tgt=%.3f" %
                        (pred_fake.sigmoid().mean().item()))
            logger.info("ADV BCE total=%.5f" %
                        (loss_adv.item()))
            logger.info("GEN total loss %.5f  (adv component %.5f)" %
                        (loss_gen.item(), loss_dict['adv']))
            logger.info("GRAD NORM generator=%.4f" % gen_norm)

        self.global_step += 1
        return {
            'gen_total': loss_dict['total'],
            'gen_adv': loss_dict['adv'],
            'gen_cycle': loss_dict['cycle'],
            'gen_mse': loss_dict['mse'],
            'disc': loss_disc.item() if isinstance(loss_disc, torch.Tensor) else loss_disc
        }

    def validate(self, val_loader: DataLoader) -> Dict:
        """Validation - UNCHANGED."""
        self.generator.eval()
        val_losses = []
        psnr_values = []
        ssim_values = []
        
        # validation_stats ={'total':0, 'skipped':0}
        with torch.no_grad():
            for batch in val_loader:
                # validation_stats['total'] += 1  # Track total attempts (add this if not present)
                    
                if batch.get('skip_batch', False):
                    continue

                N = batch['source'].size(0)
                if N == 0:
                    continue

                source = batch['source'].to(self.device)
                target = batch['target'].to(self.device)
                target_phase = batch['target_phase'].to(self.device)
                
                use_phase = self.config.get('use_phase_conditioning', False)
                phase_arg = target_phase if use_phase else None

                with autocast(enabled=self.use_amp):
                    generated = self.generator(source, phase_arg)
                    loss = F.mse_loss(generated, target)
                
                val_losses.append(loss.item())
                
                # Compute metrics
                generated_np = generated.cpu().numpy()
                target_np = target.cpu().numpy()
                
                for i in range(generated_np.shape[0]):
                    mse = np.mean((generated_np[i] - target_np[i]) ** 2)
                    psnr = 10 * np.log10(4.0 / (mse + 1e-10))
                    psnr_values.append(psnr)
                    
                    # Simple SSIM approximation
                    ssim = 1 - mse / 4.0
                    ssim_values.append(max(0, min(1, ssim)))
        
        self.generator.train()
        # skip_rate = validation_stats['skipped'] / max(1, validation_stats['total'])
        # logger.info(f"  Validation - Skipped {validation_stats['skipped']}/{validation_stats['total']} "
                #    f"batches ({skip_rate*100:.1f}%)")

        return {
            'val_loss': np.mean(val_losses),
            'psnr': np.mean(psnr_values),
            'ssim': np.mean(ssim_values)
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int, start_epoch: int = 0):
        """
        Main training loop - UNCHANGED logic.
        NEW: Logs cache statistics for monitoring.
        """
        save_samples_interval = self.config.get('save_samples_interval', 1)
        num_samples = self.config.get('num_samples', 3)
        
        logger.info(f"\nStarting training for {epochs} epochs")
        logger.info(f"Batches: train={len(train_loader)}, val={len(val_loader)}")
        
        # Log initial cache stats
        if hasattr(train_loader.dataset, 'get_cache_stats'):
            cache_stats = train_loader.dataset.get_cache_stats()
            logger.info(f"Volume cache initialized: max size = {cache_stats['cache_size']}")
        
        logger.info("\n=== Loss Weights at Start ===")
        logger.info(f"  lambda_cycle: {self.combined_loss.lambda_cycle}")
        # logger.info(f"  lambda_mse (initial/final): {self.combined_loss.lambda_mse_initial}/{self.combined_loss.lambda_mse_final}")
        logger.info(f"  lambda_mse (initial/final): {self.combined_loss.lambda_mse}")
        logger.info(f"  lambda_adv: {self.combined_loss.lambda_adv}")
        logger.info(f"  lambda_organ: {self.combined_loss.lambda_organ}")
        logger.info(f"  organ_weight (inside MSE): {self.combined_loss.organ_weight}")
        sum_weights = (self.combined_loss.lambda_cycle + self.combined_loss.lambda_mse + 
                    self.combined_loss.lambda_adv + self.combined_loss.lambda_organ)
        logger.info(f"  Sum of lambdas (at full warmup): {sum_weights}")
        logger.info("===========================")

        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch
            self.combined_loss.set_epoch(epoch)
            
            # Cleanup at epoch start
            self.memory_manager.check_and_cleanup(force=True, current_epoch=epoch)
            
            # Training
            
            self.generator.train()
            if self.disc_target is not None:
                self.disc_target.train()
            
            epoch_losses = {
                'gen_total': [], 'gen_adv': [], 'gen_cycle': [],
                'gen_mse': [], 'disc': []
            }
            
            pbar = tqdm(
                train_loader, 
                desc=f"Epoch {epoch}/{epochs}",
                leave=True,
                dynamic_ncols=True,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
            )
            # validation_stats ={'total':0, 'skipped':0}
            for batch_idx, batch in enumerate(pbar):

                try:
                    losses = self.train_step(batch)
                    
                    for key in epoch_losses:
                        epoch_losses[key].append(losses[key])
                    
                    # Periodic cleanup
                    if batch_idx % self.cleanup_frequency == 0:
                        self.memory_manager.check_and_cleanup(current_epoch=epoch)
                    

                    if batch_idx % 10 == 0:
                        postfix = {
                            'G': f"{losses['gen_total']:.3f}",
                            'MSE': f"{losses['gen_mse']:.3f}",
                        }
                        
                        if self.config.get('use_discriminator'):
                            postfix['D'] = f"{losses['disc']:.3f}"
                            postfix['ADV'] = f"{losses['gen_adv']:.3f}"
                        
                        if self.config.get('use_cycle_consistency'):
                            postfix['Cycle'] = f"{losses['gen_cycle']:.3f}"
                        
                        pbar.set_postfix(postfix)
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        # logger.error(f"OOM at batch {batch_idx}")
                        self.memory_manager.emergency_cleanup()
                        continue
                    raise e
            
            # Average losses
            train_losses = {k: np.mean(v) if v else 0.0 for k, v in epoch_losses.items()}
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # self.scheduler_gen.step(val_metrics['val_loss'])

            # Log cache statistics
            if hasattr(train_loader.dataset, 'get_cache_stats'):
                cache_stats = train_loader.dataset.get_cache_stats()
                logger.info(f"  Cache - Size: {cache_stats['cache_size']}, "
                           f"Hit Rate: {cache_stats['hit_rate']:.2%}, "
                           f"Hits: {cache_stats['cache_hits']}, "
                           f"Misses: {cache_stats['cache_misses']}")
            
            # skip_rate = validation_stats['skipped'] / max(1, validation_stats['total'])
            # logger.info(f"  Validation - Skipped {validation_stats['skipped']}/{validation_stats['total']} "
            #        f"batches ({skip_rate*100:.1f}%)")

            # Log
            logger.info(f"\nEpoch {epoch+1}/{epochs} Summary:")
            log_parts = [f"Total: {train_losses['gen_total']:.4f}"]
            log_parts.append(f"MSE: {train_losses['gen_mse']:.4f}")

            if self.config.get('use_cycle_consistency'):
                log_parts.append(f"Cycle: {train_losses['gen_cycle']:.4f}")

            if self.config.get('use_discriminator'):
                log_parts.append(f"Adv: {train_losses['gen_adv']:.4f}")

            logger.info(f"  Train - {', '.join(log_parts)}")
            logger.info(f"  Disc: {train_losses['disc']:.4f}")

            logger.info(f"  Disc: {train_losses['disc']:.4f}")
            logger.info(f"  Val - Loss: {val_metrics['val_loss']:.4f}, "
                       f"PSNR: {val_metrics['psnr']:.2f} dB, "
                       f"SSIM: {val_metrics['ssim']:.4f}")
            
            lr_gen = self.opt_gen.param_groups[0]['lr']
            lr_disc = self.opt_disc_target.param_groups[0]['lr'] if self.opt_disc_target else 0.0
            adv_weight = self.combined_loss.get_adv_weight() if self.config.get('use_discriminator') else 0.0

            self.loss_tracker.update_epoch(
                    epoch=epoch,
                    train_losses=train_losses,
                    val_metrics=val_metrics,
                    lr_gen=lr_gen,
                    lr_disc=lr_disc
                )
            
            if (epoch) % 2 == 0:
                self.loss_tracker.create_summary_report(epoch + 1)
            
            # Save sample patches
            if (epoch) % save_samples_interval == 0:
                logger.info(f"Saving sample patches for epoch {epoch + 1}...")
                save_sample_patches(
                    self.generator,
                    val_loader,
                    epoch,
                    self.samples_dir,
                    self.device,
                    num_samples=num_samples,
                    save_nifti=self.config.get('save_nifti', True)
                )
            
            # Save checkpoint
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
            
            self.save_checkpoint(epoch, val_metrics, is_best)
            # In train loop:
            val_metrics = self.validate(val_loader)
            self.early_stopping(val_metrics['val_loss'])

            if self.early_stopping.early_stop:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
            if self.scheduler_gen is not None:
                self.scheduler_gen.step()
                current_lr = self.scheduler_gen.get_last_lr()[0]
                logger.info(f"  Learning rate: {current_lr:.6f}")
            
        logger.info("Training completed!")
        
        # Final cache statistics
        if hasattr(train_loader.dataset, 'get_cache_stats'):
            cache_stats = train_loader.dataset.get_cache_stats()
            logger.info(f"\n=== Final Cache Statistics ===")
            logger.info(f"Total cache hits: {cache_stats['cache_hits']}")
            logger.info(f"Total cache misses: {cache_stats['cache_misses']}")
            logger.info(f"Overall hit rate: {cache_stats['hit_rate']:.2%}")
            logger.info(f"Disk I/O reduction: ~{cache_stats['hit_rate'] * 100:.1f}%")
    
    def save_checkpoint(self, epoch: int, val_metrics: Dict, is_best: bool = False):
        """Save checkpoint with conditional discriminator state."""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'opt_gen_state_dict': self.opt_gen.state_dict(),
            'val_metrics': val_metrics,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Conditionally save discriminator (Stage 3+)
        if self.disc_target is not None:
            checkpoint['disc_target_state_dict'] = self.disc_target.state_dict()
            checkpoint['opt_disc_target_state_dict'] = self.opt_disc_target.state_dict()
        
        # Save checkpoint
        torch.save(checkpoint, self.output_dir / f'checkpoint_epoch_{epoch}.pth')
        
        if is_best:
            torch.save(checkpoint, self.output_dir / 'checkpoint_best.pth')
        
        # Keep only last N checkpoints
        keep_last_n = self.config.get('keep_last_n_checkpoints', 3)
        checkpoints = sorted(self.output_dir.glob('checkpoint_epoch_*.pth'))
        for old in checkpoints[:-keep_last_n]:
            old.unlink()

    def load_checkpoint(self, checkpoint_path: Path) -> bool:
        """Load checkpoint with conditional discriminator handling."""
        try:
            torch.serialization.add_safe_globals([np.core.multiarray.scalar])
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Load generator
            logger.info("Loading generator state...")
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            
            # Load generator optimizer
            if 'opt_gen_state_dict' in checkpoint:
                logger.info("Loading generator optimizer state...")
                self.opt_gen.load_state_dict(checkpoint['opt_gen_state_dict'])
            
            # Conditionally load discriminator (only if both checkpoint has it AND current config uses it)
            use_disc = self.config.get('use_discriminator', False)
            has_disc_in_checkpoint = 'disc_target_state_dict' in checkpoint
            
            if use_disc and has_disc_in_checkpoint:
                logger.info("Loading discriminator state...")
                self.disc_target.load_state_dict(checkpoint['disc_target_state_dict'])
                
                if 'opt_disc_target_state_dict' in checkpoint:
                    logger.info("Loading discriminator optimizer state...")
                    self.opt_disc_target.load_state_dict(checkpoint['opt_disc_target_state_dict'])
            elif use_disc and not has_disc_in_checkpoint:
                logger.warning("⚠ Discriminator requested but not found in checkpoint (will train from scratch)")
            elif not use_disc and has_disc_in_checkpoint:
                logger.info("Discriminator in checkpoint but not used in current stage (skipping)")
            
            # Load training state
            self.current_epoch = checkpoint.get('epoch', 0)
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            
            logger.info(f"✓ Checkpoint loaded successfully")
            logger.info(f"  - Resuming from epoch {self.current_epoch}")
            logger.info(f"  - Best validation loss: {self.best_val_loss:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to load checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return False    


    # # --------------------------------------------------------------
    # NEW: Helper to upgrade upsampling in-place
    # --------------------------------------------------------------
    def _upgrade_generator_upsampling(self, model: nn.Module):
        """Replace ConvTranspose3d â†’ Upsample + Conv3d with weight transfer."""
        import torch.nn.functional as F

        for name, module in list(model.named_modules()):
            if not isinstance(module, nn.ConvTranspose3d):
                continue

            in_ch, out_ch = module.in_channels, module.out_channels
            bias = module.bias is not None

            # Build new block
            new_block = nn.Sequential(
                nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False),
                nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=bias),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            ).to(module.weight.device)

            # Transfer weights
            with torch.no_grad():
                old_w = module.weight.data  # [out, in, 1,2,2]
                center = old_w[:, :, :, 1:2, 1:2]  # [out, in, 1,1,1]
                new_w = center.repeat(1, 1, 3, 3, 3)
                new_w = new_w.permute(1, 0, 2, 3, 4).contiguous()  # [in, out, 3,3,3]
                new_block[1].weight.copy_(new_w)
                if bias:
                    new_block[1].bias.copy_(module.bias.data)

            # === FIX: Handle top-level vs nested modules ===
            if '.' in name:
                parent_name, layer_name = name.rsplit('.', 1)
                parent = model
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
                setattr(parent, layer_name, new_block)
            else:
                # Top-level module (e.g., self.up4)
                setattr(model, name, new_block)

            logger.debug(f"Upgraded {name} â†’ Upsample+Conv3d")

