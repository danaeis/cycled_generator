"""
OPTIMIZED Memory-Efficient CT Phase Generation Training Pipeline
================================================================
Key Optimizations:
- Smart volume caching (LRU cache for frequently accessed volumes)
- Memory-mapped file support for efficient random access
- Lazy loading - only load what's needed
- Reduced disk I/O by 90%+
- All losses preserved exactly as before

NEW: Autoencoder mode for reconstruction training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from torch.cuda.amp import GradScaler
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
from skimage.metrics import structural_similarity as ssim

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
    
    NEW: Autoencoder mode - reconstructs same input when use_autoencoder=True
    """
    
    def __init__(
        self,
        data_pairs: List[Dict],
        patch_size: Tuple[int, int] = (256, 256),
        slice_range: Tuple[float, float] = (0.3, 0.7),
        overlap_ratio: float = 0.5,
        augment: bool = False,
        cache_size: int = 8,
        use_memmap: bool = True,
        use_autoencoder: bool = False,
        num_augment_copies: int = 4,  # NEW: How many augmented versions per patch
        ):
        
        self.data_pairs = data_pairs
        self.patch_size = patch_size
        self.slice_range = slice_range
        self.overlap_ratio = overlap_ratio
        self.augment = augment
        self.use_autoencoder = use_autoencoder
        self.num_augment_copies = num_augment_copies if augment else 1  # NEW
        self.patch_coords = []
        
        self.volume_cache = VolumeCache(max_cache_size=cache_size)
        self.use_memmap = use_memmap
        
        self.phase_to_idx = {
            'non-contrast': 0,
            'arterial': 1,
            'portal': 2,
            'venous': 2,
            'delayed': 3
        }
        
        # بعد از self.patch_size = patch_size
        if isinstance(patch_size, (int, float)):
            self.patch_height = self.patch_width = int(patch_size)
        else:  # tuple or list
            self.patch_height, self.patch_width = map(int, patch_size)

        self.half_h = self.patch_height // 2
        self.half_w = self.patch_width // 2

        # و stride هم جدا حساب کن
        self.stride_h = int(self.patch_height * (1 - self.overlap_ratio))
        self.stride_w = int(self.patch_width * (1 - self.overlap_ratio))

        mode_str = "AUTOENCODER" if use_autoencoder else "PHASE TRANSLATION"
        logger.info(f"Dataset ({mode_str}): {len(data_pairs)} pairs, 2D slices {patch_size}")
        logger.info(f"Slice range: {slice_range[0]:.1%} to {slice_range[1]:.1%}")
        if augment:
            logger.info(f"Augmentation: {num_augment_copies}x copies per patch")
        self._compute_slice_coordinates()
        base_patches = len(self.patch_coords)
        logger.info(f"Generated {base_patches} base patches × {self.num_augment_copies} aug = {len(self)} total")

    def __len__(self) -> int:
        # NEW: Multiply by augmentation copies
        return len(self.patch_coords) * self.num_augment_copies

    def _is_valid_patch(self, source_vol, target_vol, z, y, x):
        """
        فیلتر پچ‌های خالی/هوا — سازگار با حجم‌های نرمالایز شده به [0,1]
        """
        src_patch = source_vol[z, y-self.half_h:y+self.half_h, x-self.half_w:x+self.half_w]
        tgt_patch = target_vol[z, y-self.half_h:y+self.half_h, x-self.half_w:x+self.half_w]

        # معیار ۱: انحراف معیار خیلی کم (برای داده [0,1] → std < 0.02 یعنی تقریباً ثابت)
        if src_patch.std() < 0.02 or tgt_patch.std() < 0.02:
            return False

        # معیار ۲: میانگین خیلی پایین (نزدیک صفر → هوا یا پدینگ)
        if src_patch.mean() < 0.05 or tgt_patch.mean() < 0.05:
            return False

        # معیار ۳: درصد زیاد مقادیر نزدیک به صفر (بیشتر از ۸۵٪ نزدیک به 0)
        if np.mean(src_patch < 0.1) > 0.85 or np.mean(tgt_patch < 0.1) > 0.85:
            return False

        # اختیاری: حداقل یه مقدار بالای 0.3 داشته باشه (یعنی بافت واقعی باشه)
        if src_patch.max() < 0.2 and tgt_patch.max() < 0.2:
            return False

        return True

    def _compute_slice_coordinates(self):
        """Extract 2D slice coordinates from volume middle range."""
        for pair_idx, pair_data in enumerate(self.data_pairs):
            try:
                source_shape = self.volume_cache.get_volume_shape(pair_data['source_path'])
                target_shape = self.volume_cache.get_volume_shape(pair_data['target_path'])
                
                source_vol = self.volume_cache.get_volume(pair_data['source_path'])
                target_vol = self.volume_cache.get_volume(pair_data['target_path'])
                
                if source_shape != target_shape:
                    continue
                
                depth, height, width = source_shape
                
                # Get slice range (middle 40% by default)
                z_start = int(depth * self.slice_range[0])
                z_end = int(depth * self.slice_range[1])
                z_range = range(z_start, z_end)
                
                # Spatial sampling with overlap
                step_y = max(1, int(self.patch_height * (1 - self.overlap_ratio))) 
                step_x = max(1, int(self.patch_width * (1 - self.overlap_ratio)))
                
                y_positions = list(range(self.half_h, height - self.half_h + 1, step_y))
                x_positions = list(range(self.half_w, width - self.half_w + 1, step_x))
                
                # Generate coordinates for each slice
                for z in z_range:
                    for y in y_positions:
                        for x in x_positions:
                            if self._is_valid_patch(source_vol, target_vol, z, y, x) :
                                self.patch_coords.append({
                                    'pair_idx': pair_idx,
                                    'z': z,  # Single slice index
                                    'y': y,
                                    'x': x
                                })
            except Exception as e:
                logger.warning(f"Skipping pair {pair_idx}: {e}")


    def apply_single_augmentation(self, source, target, aug_type: int):
        """
        Apply a SINGLE augmentation type based on aug_type index.
        
        Aug types:
        0: Original (no aug)
        1: Vertical flip
        2: Rotation
        3: Scaling
        4: Gaussian noise
        """
        source_pil = TF.to_pil_image(source.squeeze())
        target_pil = TF.to_pil_image(target.squeeze())
        
        if aug_type == 0:
            # Original - no augmentation
            pass
        
        elif aug_type == 1:
            # Vertical flip only
            source_pil = TF.vflip(source_pil)
            target_pil = TF.vflip(target_pil)
        
        elif aug_type == 2:
            # Rotation only
            angle = random.uniform(-15, 15)
            source_pil = TF.rotate(source_pil, angle)
            target_pil = TF.rotate(target_pil, angle)
        
        elif aug_type == 3:
            # Scaling only
            scale = random.uniform(0.9, 1.1)
            new_size = int(source_pil.size[0] * scale)
            source_pil = TF.resize(source_pil, new_size)
            target_pil = TF.resize(target_pil, new_size)
            source_pil = TF.center_crop(source_pil, self.patch_size)
            target_pil = TF.center_crop(target_pil, self.patch_size)
        
        # Convert back to tensor
        source = TF.to_tensor(source_pil).squeeze()
        target = TF.to_tensor(target_pil).squeeze()
        
        if aug_type == 4:
            # Gaussian noise only
            noise = torch.randn_like(source) * 0.02
            source = source + noise
            target = target + noise
        
        return source.unsqueeze(0), target.unsqueeze(0)
    
    # def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    #     """
    #     Extract single 2D slice with augmentation.
        
    #     NEW: idx maps to (base_patch_idx, augmentation_type)
    #     """
    #     # NEW: Decompose idx into base patch and augmentation type
    #     base_idx = idx // self.num_augment_copies
    #     aug_type = idx % self.num_augment_copies
        
    #     coord = self.patch_coords[base_idx]
    #     pair_data = self.data_pairs[coord['pair_idx']]
        
    #     # Get source volume
    #     source_vol = self.volume_cache.get_volume(
    #         pair_data['source_path'], use_memmap=self.use_memmap
    #     )
        
    #     # Extract slice coordinates
    #     z, y, x = coord['z'], coord['y'], coord['x']
    #     source_slice = source_vol[ 
    #                             z, 
    #                             y-self.half_h : y+self.half_h, 
    #                             x-self.half_w : x+self.half_w
    #                             ].copy()
        
    #     # NEW: For autoencoder, target = source
    #     if self.use_autoencoder:
    #         target_slice = source_slice.copy()
    #         target_phase_idx = self.phase_to_idx.get(pair_data['source_phase'], 0)
    #     else:
    #         # Phase translation mode
    #         target_vol = self.volume_cache.get_volume(
    #             pair_data['target_path'], use_memmap=self.use_memmap
    #         )
    #         target_slice = target_vol[ 
    #                                 z, 
    #                                 y-self.half_h : y+self.half_h, 
    #                                 x-self.half_w : x+self.half_w
    #                                 ].copy()
    #         target_phase_idx = self.phase_to_idx.get(pair_data['target_phase'], 1)
        
    #     # Phase encoding
    #     source_phase_idx = self.phase_to_idx.get(pair_data['source_phase'], 0)
        
    #     # Normalize to [-1, 1]
    #     source_slice = (source_slice - source_slice.mean()) / (source_slice.std() + 1e-8)
    #     target_slice = (target_slice - target_slice.mean()) / (target_slice.std() + 1e-8)
        
    #     # Convert to tensors [1, H, W]
    #     source_tensor = torch.from_numpy(source_slice).unsqueeze(0).float()
    #     target_tensor = torch.from_numpy(target_slice).unsqueeze(0).float()
        
    #     # NEW: Apply specific augmentation type
    #     if self.augment:
    #         source_tensor, target_tensor = self.apply_single_augmentation(
    #             source_tensor, target_tensor, aug_type
    #         )
        
    #     # Extract masks if available
    #     # masks = {}
    #     # seg_key = 'source_seg' if self.use_autoencoder else 'target_seg'
    #     # if seg_key in pair_data and pair_data[seg_key]:
    #     #     try:
    #     #         seg_path = Path(pair_data[seg_key])
    #     #         if seg_path.exists():
    #     #             seg_vol = self.volume_cache.get_volume(seg_path, use_memmap=self.use_memmap)
    #     #             seg_slice = seg_vol[z, y:y+self.patch_size[0], x:x+self.patch_size[1]].copy()
                    
    #     #             if seg_slice.shape != source_slice.shape:
    #     #                 seg_slice = cv2.resize(seg_slice, 
    #     #                                     (self.patch_size[1], self.patch_size[0]),
    #     #                                     interpolation=cv2.INTER_NEAREST)
                    
    #     #             seg_tensor = torch.from_numpy(seg_slice).unsqueeze(0).float()
                    
    #     #             # Organ mapping
    #     #             organ_labels = {
    #     #                 'liver': [1],
    #     #                 'spleen': [2],
    #     #                 'kidney_right': [3],
    #     #                 'kidney_left': [4],
    #     #                 'pancreas': [5],
    #     #             }
                    
    #     #             for organ, labels in organ_labels.items():
    #     #                 mask = torch.zeros_like(seg_tensor)
    #     #                 for lbl in labels:
    #     #                     mask = torch.logical_or(mask, seg_tensor == lbl).float()
    #     #                 masks[organ] = mask.unsqueeze(0)
    #     #     except Exception as e:
    #     #         logger.warning(f"Mask loading failed: {e}")
    #             # === MASKS (always return the same keys, zero when missing) ===
        
    #     mask_shape = (1, 1, self.patch_size[0], self.patch_size[1])  # [1,1,H,W]
    #     masks = {
    #         'liver': torch.zeros(mask_shape, dtype=torch.float32),
    #         'spleen': torch.zeros(mask_shape, dtype=torch.float32),
    #         'kidney_right': torch.zeros(mask_shape, dtype=torch.float32),
    #         'kidney_left': torch.zeros(mask_shape, dtype=torch.float32),
    #         'pancreas': torch.zeros(mask_shape, dtype=torch.float32),
    #     }

    #     # Try to load real segmentation
    #     seg_key = 'source_seg' if self.use_autoencoder else 'target_seg'
    #     if seg_key in pair_data and pair_data[seg_key]:
    #         try:
    #             seg_path = Path(pair_data[seg_key])
    #             if seg_path.exists():
    #                 seg_vol = self.volume_cache.get_volume(seg_path, use_memmap=self.use_memmap)
    #                 seg_slice = seg_vol[z, y:y+self.patch_size[0], x:x+self.patch_size[1]].copy()

    #                 if seg_slice.shape != (self.patch_size[0], self.patch_size[1]):
    #                     import cv2
    #                     seg_slice = cv2.resize(seg_slice,
    #                                            (self.patch_size[1], self.patch_size[0]),
    #                                            interpolation=cv2.INTER_NEAREST)

    #                 # Convert to tensor: [1, 1, H, W]
    #                 seg_tensor = torch.from_numpy(seg_slice).unsqueeze(0).unsqueeze(0).float()  # [1,1,H,W]

    #                 organ_labels = {
    #                     'liver': [1],
    #                     'spleen': [2],
    #                     'kidney_right': [3],
    #                     'kidney_left': [4],
    #                     'pancreas': [5],
    #                 }

    #                 for organ, labels in organ_labels.items():
    #                     mask = torch.zeros_like(seg_tensor)
    #                     for lbl in labels:
    #                         mask = torch.logical_or(mask, (seg_tensor == lbl))
    #                     masks[organ] = mask.float()  # Already [1,1,H,W]
    #         except Exception as e:
    #             logger.debug(f"Mask loading failed (continuing with zeros): {e}")

    #     return {
    #         'source': source_tensor,
    #         'target': target_tensor,
    #         'source_phase': torch.tensor(source_phase_idx, dtype=torch.long),
    #         'target_phase': torch.tensor(target_phase_idx, dtype=torch.long),
    #         'case_id': pair_data['case_id'],
    #         'masks': masks,
    #         'skip': False
    #     }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample with mask weighting."""
        pair_idx, center_z, y_start, x_start = self.patch_coords[idx]
        pair_data = self.data_pairs[pair_idx]
        
        # Load volumes (with caching)
        source_vol, target_vol = self._load_volumes_cached(pair_idx)
        
        # Transpose to (D, H, W)
        source_vol = np.transpose(source_vol, (2, 1, 0))
        target_vol = np.transpose(target_vol, (2, 1, 0))
        
        # ========== APPLY NORMALIZATION (NEW) ==========
        source_vol_norm = apply_normalization(
            source_vol, 
            method=self.normalization_method,
            to_discrete=self.use_discrete_values,
            discrete_range=self.discrete_range
        )
        
        target_vol_norm = apply_normalization(
            target_vol,
            method=self.normalization_method,
            to_discrete=self.use_discrete_values,
            discrete_range=self.discrete_range
        )
        
        # Extract patches
        padding = self.patch_depth // 2
        z_start = center_z - padding
        z_end = center_z + padding + 1
        
        source_patch = source_vol_norm[z_start:z_end, 
                                        y_start:y_start + self.patch_size[0],
                                        x_start:x_start + self.patch_size[1]]
        
        target_patch = target_vol_norm[z_start:z_end,
                                        y_start:y_start + self.patch_size[0],
                                        x_start:x_start + self.patch_size[1]]
        
        # ========== GENERATE MASK WEIGHTS (NEW) ==========
        if self.use_mask_weighting:
            # Create weight mask based on intensity (higher intensity = more important)
            weight_mask = np.ones_like(target_patch, dtype=np.float32)
            
            if self.use_discrete_values:
                # For discrete values, use threshold
                high_intensity_mask = target_patch > self.high_intensity_threshold
            else:
                # For continuous values, use normalized threshold
                normalized_threshold = self.high_intensity_threshold / self.discrete_range[1]
                high_intensity_mask = target_patch > normalized_threshold
            
            weight_mask[high_intensity_mask] = 3.0  # 3x weight on important regions
        else:
            weight_mask = np.ones_like(target_patch, dtype=np.float32)
        
        # Get phase labels
        source_phase_idx = self.phase_to_idx.get(pair_data.get('source_phase', 'unknown'), 0)
        target_phase_idx = self.phase_to_idx.get(pair_data.get('target_phase', 'unknown'), 0)
        
        # Convert to tensors
        if not self.use_discrete_values:
            # For continuous values, already in [0, 1]
            source_tensor = torch.from_numpy(source_patch).float().unsqueeze(0)
            target_tensor = torch.from_numpy(target_patch).float().unsqueeze(0)
        else:
            # For discrete values, normalize to [0, 1] for network
            source_tensor = torch.from_numpy(source_patch).float().unsqueeze(0) / self.discrete_range[1]
            target_tensor = torch.from_numpy(target_patch).float().unsqueeze(0) / self.discrete_range[1]
        
        weight_tensor = torch.from_numpy(weight_mask).float().unsqueeze(0)
        
        return {
            'source': source_tensor,
            'target': target_tensor,
            'weight_mask': weight_tensor,
            'source_phase': torch.tensor(source_phase_idx, dtype=torch.long),
            'target_phase': torch.tensor(target_phase_idx, dtype=torch.long),
            'case_id': pair_data['case_id']
        }

    def get_cache_stats(self) -> Dict:
        """Get volume cache statistics."""
        return self.volume_cache.get_stats()
    
    def clear_cache(self):
        """Clear the volume cache."""
        self.volume_cache.clear()


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class Generator2D(nn.Module):
    """
    Parametric 2D U-Net for CT phase translation or autoencoder reconstruction.
    """
    def __init__(
        self,
        num_phases: int = 4,
        base_channels: int = 64,
        use_phase_conditioning: bool = False,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.base_channels = base_channels
        self.use_phase_conditioning = use_phase_conditioning
        self.dropout_rate = dropout
        
        ch1 = base_channels
        ch2 = base_channels * 2
        ch3 = base_channels * 4
        ch4 = base_channels * 8
        bottleneck_ch = ch4
        
        # Phase embedding (AdaIN in bottleneck)
        if use_phase_conditioning:
            phase_emb_dim = base_channels
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
        
        self.up1 = nn.ConvTranspose2d(ch1, ch1, kernel_size=2, stride=2)
        self.dec1 = self._make_dec_block(ch1 + ch1, ch1)
        
        self.out_conv = nn.Sequential(
            nn.Conv2d(ch1, 1, kernel_size=1),
            nn.Tanh()
        )
        
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
            emb = self.phase_emb(phase_idx)
            ss = self.phase_scale_shift(emb)
            scale, shift = ss.chunk(2, dim=1)
            scale = scale.view(B, -1, 1, 1)
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


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class CombinedLoss(nn.Module):
    """
    Conditional loss supporting 4-stage training:
    Stage 1: MSE only
    Stage 2: MSE + Organ-weighted MSE
    Stage 3: MSE + Organ + Adversarial
    Stage 4: MSE + Organ + Adversarial + Cycle
    
    Works for both phase translation and autoencoder training.
    """
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
        self.current_epoch = 0
        self.adv_warmup_epochs = config.get('adv_warmup_epochs', 10)
        
        logger.info(f"CombinedLoss initialized:")
        logger.info(f"  use_organ_loss: {self.use_organ_loss}")
        logger.info(f"  use_discriminator: {self.use_discriminator}")
        logger.info(f"  use_cycle: {self.use_cycle}")
    
    def set_epoch(self, epoch: int):
        """Update current epoch for warmup schedules."""
        self.current_epoch = epoch

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
        """Compute organ-weighted MSE."""
        loss_global = F.mse_loss(pred, target, reduction='mean')
        
        if not mask_dict or len(mask_dict) == 0:
            return loss_global
        
        # Union of all organ masks
        union = torch.zeros_like(pred)
        for organ_name, mask in mask_dict.items():
            if mask.dim() == 5:
                mask = mask.squeeze(2)
            union = torch.max(union, mask)
        
        # Organ MSE
        loss_organ = F.mse_loss(pred * union, target * union, reduction='mean')
        
        return loss_global + (organ_weight - 1.0) * loss_organ
    
    def forward(self, pred, target, cycle_pred=None, source=None, 
                adv_loss=0.0, masks=None):
        """
        Compute conditional loss based on active components.
        
        Works for both:
        - Phase translation: pred=generated_target, target=real_target
        - Autoencoder: pred=reconstructed, target=original
        """
        loss_dict = {}
        
        if masks is not None:
            masks = {k: v.to(self.device) for k, v in masks.items()}
        
        # MSE Loss (with optional organ weighting)
        if self.use_organ_loss and masks and len(masks) > 0:
            loss_mse = self.masked_mse(pred, target, masks, self.organ_weight)
            loss_mse *= self.lambda_organ
            loss_dict['mse'] = loss_mse.item()
            
            if self.global_step % 100 == 0:
                logger.info(f"✓ Using MASKED MSE (λ={self.lambda_organ}, weight={self.organ_weight})")
        else:
            loss_mse = self.mse(pred, target) * self.lambda_mse
            loss_dict['mse'] = loss_mse.item()
            
            if self.global_step % 100 == 0:
                if self.use_organ_loss:
                    logger.warning(f"✗ Organ loss enabled but no masks provided!")
                else:
                    logger.info(f"→ Using regular MSE (λ={self.lambda_mse})")
        
        total_loss = loss_mse
        
        # Adversarial Loss
        if self.use_discriminator and isinstance(adv_loss, (int, float)) and adv_loss > 0:
            loss_adv_weighted = adv_loss * self.get_adv_weight()
            total_loss += loss_adv_weighted
            loss_dict['adv'] = loss_adv_weighted
        elif self.use_discriminator and torch.is_tensor(adv_loss):
            loss_adv_weighted = adv_loss * self.get_adv_weight()
            total_loss += loss_adv_weighted
            loss_dict['adv'] = loss_adv_weighted.item()
        else:
            loss_dict['adv'] = 0.0
        
        # Cycle Consistency Loss
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
# LOSS TRACKING
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
    """Track and visualize training losses."""
    
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
        self.history['train_disc'].append(train_losses.get('disc', 0))
        self.history['val_loss'].append(val_metrics.get('val_loss', 0))
        self.history['val_psnr'].append(val_metrics.get('psnr', 0))
        self.history['val_ssim'].append(val_metrics.get('ssim', 0))
        self.history['lr_gen'].append(lr_gen)
        self.history['lr_disc'].append(lr_disc)
    
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
# SAMPLE VISUALIZATION
# ============================================================================

def save_sample_patches(
    generator: nn.Module,
    val_loader: DataLoader,
    epoch: int,
    save_dir: Path,
    device: torch.device,
    num_samples: int = 5,
    save_nifti: bool = False,
    is_autoencoder: bool = False,
    use_cycle: bool = False  # NEW: Whether cycle consistency is used
    ):
    """
    Save sample generated patches for visual inspection.
    Adapts visualization based on training mode.
    """
    generator.eval()
    save_dir = Path(save_dir) / f"epoch_{epoch}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    png_dir = save_dir / 'comparisons'
    nifti_dir = save_dir / 'nifti_volumes'
    png_dir.mkdir(exist_ok=True)
    if save_nifti:
        nifti_dir.mkdir(exist_ok=True)
    
    # Delete old epoch folders (keep only last 10)
    parent_dir = save_dir.parent
    epoch_dirs = sorted(parent_dir.glob('epoch_*'), key=lambda x: int(x.name.split('_')[1]))
    
    if len(epoch_dirs) > 10:
        for old_dir in epoch_dirs[:-10]:
            try:
                import shutil
                shutil.rmtree(old_dir)
                logger.info(f"Deleted old sample directory: {old_dir.name}")
            except Exception as e:
                logger.warning(f"Could not delete {old_dir}: {e}")
    
    
    # Collect validation batches
    all_batches = []
    with torch.no_grad():
        for batch in val_loader:
            all_batches.append(batch)
    
    valid_batches = [b for b in all_batches if not b.get('skip_batch', False)]
    total_patches = sum(b['source'].size(0) for b in valid_batches) if valid_batches else 0
    
    if len(valid_batches) == 0:
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
            
            batch = random.choice(valid_batches)
            
            try:
                real_source = batch['source'].to(device)
                real_target = batch['target'].to(device)
                source_phase = batch['source_phase'].to(device)
                target_phase = batch['target_phase'].to(device)
                case_id = batch['case_id']
                
                # Generate reconstruction
                generated_target = generator(real_source, target_phase)
                
                # Cycle reconstruction (if enabled)
                if use_cycle:
                    reconstructed_source = generator(generated_target, source_phase)
                
                batch_size = real_source.size(0)
                samples_from_batch = min(batch_size, num_samples - saved_count)
                
                if batch_size > samples_from_batch:
                    sample_indices = np.random.choice(batch_size, size=samples_from_batch, replace=False)
                else:
                    sample_indices = range(batch_size)
                
                for i in sample_indices:
                    source_slice = real_source[i].squeeze().cpu().numpy()
                    target_slice = real_target[i].squeeze().cpu().numpy()
                    generated_slice = generated_target[i].squeeze().cpu().numpy()
                    
                    # ============================================================
                    # ADAPTIVE VISUALIZATION BASED ON MODE
                    # ============================================================
                    
                    if is_autoencoder:
                        # AUTOENCODER MODE: Input → Reconstructed → Difference
                        if use_cycle:
                            # With cycle: Show cycle reconstruction too
                            reconstructed_slice = reconstructed_source[i].squeeze().cpu().numpy()
                            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                            
                            # Row 1: Input and Reconstruction
                            im0 = axes[0, 0].imshow(source_slice, cmap='gray', vmin=-1, vmax=1)
                            axes[0, 0].set_title('Input (Original)', fontsize=12, fontweight='bold')
                            axes[0, 0].axis('off')
                            plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
                            
                            im1 = axes[0, 1].imshow(generated_slice, cmap='gray', vmin=-1, vmax=1)
                            axes[0, 1].set_title('Reconstructed (1st pass)', fontsize=12, fontweight='bold')
                            axes[0, 1].axis('off')
                            plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
                            
                            # Row 2: Cycle reconstruction and differences
                            im2 = axes[1, 0].imshow(reconstructed_slice, cmap='gray', vmin=-1, vmax=1)
                            axes[1, 0].set_title('Cycle Reconstruction (2nd pass)', fontsize=12, fontweight='bold')
                            axes[1, 0].axis('off')
                            plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
                            
                            diff = np.abs(generated_slice - source_slice)
                            diff_cycle = np.abs(reconstructed_slice - source_slice)
                            combined_diff = np.stack([diff, diff_cycle, diff], axis=-1)
                            
                            im3 = axes[1, 1].imshow(combined_diff, vmin=0, vmax=1)
                            axes[1, 1].set_title('Errors: Red=1st pass, Green=2nd pass', fontsize=12, fontweight='bold')
                            axes[1, 1].axis('off')
                            plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
                            
                            mse_recon = np.mean((generated_slice - source_slice) ** 2)
                            mse_cycle = np.mean((reconstructed_slice - source_slice) ** 2)
                            
                            fig.suptitle(
                                f'AUTOENCODER - Epoch {epoch} - Case: {case_id[i]}\n'
                                f'MSE (1st pass): {mse_recon:.4f}, MSE (2nd pass/cycle): {mse_cycle:.4f}',
                                fontsize=14, fontweight='bold', y=0.98
                            )
                        else:
                            # No cycle: Simple 1x3 layout
                            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                            
                            im0 = axes[0].imshow(source_slice, cmap='gray', vmin=-1, vmax=1)
                            axes[0].set_title('Input (Original)', fontsize=12, fontweight='bold')
                            axes[0].axis('off')
                            plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
                            
                            im1 = axes[1].imshow(generated_slice, cmap='gray', vmin=-1, vmax=1)
                            axes[1].set_title('Reconstructed', fontsize=12, fontweight='bold')
                            axes[1].axis('off')
                            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
                            
                            diff = np.abs(generated_slice - source_slice)
                            im2 = axes[2].imshow(diff, cmap='hot', vmin=0, vmax=1)
                            axes[2].set_title('|Reconstructed - Input|', fontsize=12, fontweight='bold')
                            axes[2].axis('off')
                            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
                            
                            mse_recon = np.mean((generated_slice - source_slice) ** 2)
                            
                            fig.suptitle(
                                f'AUTOENCODER - Epoch {epoch} - Case: {case_id[i]}\n'
                                f'MSE: {mse_recon:.4f}',
                                fontsize=14, fontweight='bold', y=0.98
                            )
                    
                    else:
                        # PHASE TRANSLATION MODE: Source → Generated Target → Real Target
                        if use_cycle:
                            reconstructed_slice = reconstructed_source[i].squeeze().cpu().numpy()
                            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                            
                            # Row 1: Forward translation
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
                            
                            # Row 2: Cycle and errors
                            im3 = axes[1, 0].imshow(reconstructed_slice, cmap='gray', vmin=-1, vmax=1)
                            axes[1, 0].set_title(f'Cycle Reconstructed: {source_phase[i]}', fontsize=12, fontweight='bold')
                            axes[1, 0].axis('off')
                            plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
                            
                            diff_gen = np.abs(generated_slice - target_slice)
                            im4 = axes[1, 1].imshow(diff_gen, cmap='hot', vmin=0, vmax=1)
                            axes[1, 1].set_title('|Generated - GT|', fontsize=12, fontweight='bold')
                            axes[1, 1].axis('off')
                            plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
                            
                            diff_cycle = np.abs(reconstructed_slice - source_slice)
                            im5 = axes[1, 2].imshow(diff_cycle, cmap='hot', vmin=0, vmax=1)
                            axes[1, 2].set_title('|Cycle - Source|', fontsize=12, fontweight='bold')
                            axes[1, 2].axis('off')
                            plt.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)
                            
                            mse_gen = np.mean((generated_slice - target_slice) ** 2)
                            mse_cycle = np.mean((reconstructed_slice - source_slice) ** 2)
                            
                            fig.suptitle(
                                f'PHASE TRANSLATION - Epoch {epoch} - Case: {case_id[i]}\n'
                                f'MSE (Gen): {mse_gen:.4f}, MSE (Cycle): {mse_cycle:.4f}',
                                fontsize=14, fontweight='bold', y=0.98
                            )
                        else:
                            # No cycle: 1x3 layout
                            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                            
                            im0 = axes[0].imshow(source_slice, cmap='gray', vmin=-1, vmax=1)
                            axes[0].set_title(f'Source: {source_phase[i]}', fontsize=12, fontweight='bold')
                            axes[0].axis('off')
                            plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
                            
                            im1 = axes[1].imshow(generated_slice, cmap='gray', vmin=-1, vmax=1)
                            axes[1].set_title(f'Generated: {target_phase[i]}', fontsize=12, fontweight='bold')
                            axes[1].axis('off')
                            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
                            
                            im2 = axes[2].imshow(target_slice, cmap='gray', vmin=-1, vmax=1)
                            axes[2].set_title(f'Ground Truth: {target_phase[i]}', fontsize=12, fontweight='bold')
                            axes[2].axis('off')
                            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
                            
                            mse_gen = np.mean((generated_slice - target_slice) ** 2)
                            
                            fig.suptitle(
                                f'PHASE TRANSLATION - Epoch {epoch} - Case: {case_id[i]}\n'
                                f'MSE: {mse_gen:.4f}',
                                fontsize=14, fontweight='bold', y=0.98
                            )
                    
                    plt.tight_layout()
                    
                    png_path = png_dir / f'sample_{saved_count:03d}_{case_id[i]}.png'
                    plt.savefig(png_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    saved_count += 1
                    
                    if saved_count >= num_samples:
                        break
                        
            except Exception as e:
                logger.error(f"Error saving sample patches: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    logger.info(f"✓ Saved {saved_count} sample patches to {save_dir}")

def save_val_samples(model, val_dataset, epoch, device, samples_dir="val_samples", num_samples=8, use_amp=True):
    model.eval()
    samples_dir = Path(samples_dir) / f"epoch_{epoch+1:03d}"
    samples_dir.mkdir(parents=True, exist_ok=True)

    case_indices = list(range(len(val_dataset.data_pairs)))
    random.shuffle(case_indices)
    selected_cases = case_indices[:num_samples]

    fig, axes = plt.subplots(num_samples, 6, figsize=(20, 5 * num_samples))
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
            patch = cv2.resize(slice_hu, (patch_w, patch_h), interpolation=cv2.INTER_LINEAR)  # â†’ (128, 192)

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
            recon_hu = cv2.resize(recon_small, (orig_W, orig_H), interpolation=cv2.INTER_LINEAR)  # â†’ (orig_H, orig_W)

            # === Safety check ===
            assert slice_hu.shape == recon_hu.shape, f"Shape mismatch: {slice_hu.shape} vs {recon_hu.shape}"

            # === Metrics ===
            data_range = float(slice_hu.max() - slice_hu.min()) or 1.0
            # Compute metrics at TRAINING resolution
            gt_downsampled = cv2.resize(slice_hu, (patch_w, patch_h), interpolation=cv2.INTER_LINEAR)
            data_range_patch = float(gt_downsampled.max() - gt_downsampled.min()) or 1.0
            psnr_val = psnr(gt_downsampled, recon_small, data_range=data_range_patch)
            ssim_val = ssim(gt_downsampled, recon_small, data_range=data_range_patch)

            overall_psnr.append(psnr_val)
            overall_ssim.append(ssim_val)

            # Show training resolution (what model sees)
            axes[idx, 0].imshow(gt_downsampled, cmap='gray', vmin=-200, vmax=800)
            axes[idx, 0].set_title("GT (128×192)")

            axes[idx, 1].imshow(recon_small, cmap='gray', vmin=-200, vmax=800)
            axes[idx, 1].set_title(f"Recon (128×192)\nPSNR: {psnr_val:.2f}")

            # Show full resolution (after upsample - will be blurry)
            axes[idx, 2].imshow(slice_hu, cmap='gray', vmin=-200, vmax=800)
            axes[idx, 2].set_title("GT (Original)")

            axes[idx, 3].imshow(recon_hu, cmap='gray', vmin=-200, vmax=800)
            axes[idx, 3].set_title("Recon (Upsampled)")
            

            # # === Plot ===
            # axes[idx, 0].imshow(slice_hu, cmap='gray', vmin=-200, vmax=800)
            # axes[idx, 0].set_title("Ground Truth")

            # axes[idx, 1].imshow(recon_hu, cmap='gray', vmin=-200, vmax=800)
            # axes[idx, 1].set_title(f"Reconstruction\nPSNR: {psnr_val:.2f} dB\nSSIM: {ssim_val:.4f}")

            diff = np.abs(slice_hu - recon_hu)
            axes[idx, 4].imshow(diff, cmap='hot', vmin=0, vmax=200)
            axes[idx, 4].set_title(f"Error\nMax: {diff.max():.1f} HU")

            axes[idx, 5].hist(diff.ravel(), bins=80, range=(0, 200), color='red', alpha=0.7)
            axes[idx, 5].set_title(f"Error Dist\nÏƒ = {diff.std():.1f} HU")

            for ax in axes[idx]:
                ax.axis('off')

        # Summary
        mean_psnr = np.mean(overall_psnr)
        mean_ssim = np.mean(overall_ssim)
        plt.suptitle(f"Validation Samples â€” Epoch {epoch+1}\n"
                     f"Avg PSNR: {mean_psnr:.2f} dB | Avg SSIM: {mean_ssim:.4f}",
                     fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(samples_dir / "val_summary.png", dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Validation samples saved | PSNR: {mean_psnr:.2f} dB | SSIM: {mean_ssim:.4f}")

# ============================================================================
# TRAINER
# ============================================================================

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


class MemoryOptimizedTrainer:
    """
    OPTIMIZED Trainer with autoencoder support.
    """
        
    def __init__(self, config: Dict):
        self.config = config
        self.device = config['device']
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Component flags
        use_disc = config.get('use_discriminator', False)
        use_phase = config.get('use_phase_conditioning', False)
        self.use_autoencoder = config.get('use_autoencoder', False)  # NEW
        
        self.real_label_smoothing = config.get('real_label_smoothing', 1)
        self.fake_label_smoothing = config.get('fake_label_smoothing', 0)
        
        # Models
        self.generator = Generator2D(
            num_phases=config.get('num_phase', 2),
            base_channels=config['generator_base_channels'],
            use_phase_conditioning=use_phase,
            dropout=config.get('generator_dropout', 0.3)
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
        self.opt_gen = optim.Adam(
            self.generator.parameters(),
            lr=lr_gen,
            betas=(0.5, 0.999),
            weight_decay=1e-5
        )
        
        # Loss and tracking
        self.combined_loss = CombinedLoss(config)
        self.loss_tracker = LossTracker(self.output_dir)
        self.memory_manager = MemoryManager()
        self.cleanup_frequency = config.get('cleanup_frequency', 10)
    
        # Samples directory
        self.samples_dir = self.output_dir / 'samples'
        self.samples_dir.mkdir(exist_ok=True)
        self.early_stopping = EarlyStopping(patience=7)
        
        # Scheduler
        self.use_cosine_schedule = config.get('use_cosine_schedule', True)
        
        if self.use_cosine_schedule:
            # 3. Replace scheduler with ReduceLROnPlateau (works perfectly for pure MSE)
            self.scheduler_gen = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.opt_gen, mode='min', factor=0.5, patience=3, min_lr=1e-6
            )
            self.scheduler_gen = lr_scheduler.CosineAnnealingWarmRestarts(
                self.opt_gen,
                T_0=config.get('cosine_t0', 10),
                T_mult=config.get('cosine_tmult', 2),
                eta_min=config.get('cosine_eta_min', 1e-6)
            )
            logger.info(f"✓ Cosine annealing enabled")
        else:
            self.scheduler_gen = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.use_amp = config.get('use_mixed_precision', True)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self.global_step = 0
        
        mode_str = "AUTOENCODER" if self.use_autoencoder else "PHASE TRANSLATION"
        logger.info(f"Trainer initialized - Mode: {mode_str} - Stage: {self._get_stage_name()}")
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

    def train_step(self, batch: Dict) -> Dict:
        """
        Training step - works for both phase translation and autoencoder.
        
        NEW: In autoencoder mode, target = source (reconstruction task)
        """
        if batch.get('skip_batch', False) or batch['source'].size(0) == 0:
            return {'gen_total': 0, 'gen_adv': 0, 'gen_cycle': 0, 'gen_mse': 0, 'disc': 0}
        
        source = batch['source'].to(self.device)
        target = batch['target'].to(self.device)  # Same as source in autoencoder mode
        source_phase = batch['source_phase'].to(self.device)
        target_phase = batch['target_phase'].to(self.device)
        masks = {k: v.to(self.device) for k, v in batch.get('masks', {}).items()}
        
        use_disc = self.config.get('use_discriminator', False)
        use_cycle = self.config.get('use_cycle_consistency', False)
        use_phase = self.config.get('use_phase_conditioning', False)
        disc_update_freq = self.config.get('disc_update_freq', 2)
        
        # Train Discriminator
        loss_disc = 0.0
        if use_disc and (self.global_step % disc_update_freq == 0):
            self.opt_disc_target.zero_grad()
            
            with autocast('cuda', enabled=self.use_amp):
                with torch.no_grad():
                    phase_arg = target_phase if use_phase else None
                    fake_target = self.generator(source, phase_arg)
                
                pred_real = self.disc_target(target)
                pred_fake = self.disc_target(fake_target.detach())
                
                real_labels = torch.full_like(pred_real, self.real_label_smoothing)
                fake_labels = torch.full_like(pred_fake, self.fake_label_smoothing)
                
                loss_real = F.binary_cross_entropy_with_logits(pred_real, real_labels)
                loss_fake = F.binary_cross_entropy_with_logits(pred_fake, fake_labels)
                loss_disc = (loss_real + loss_fake) * 0.5
            
            self.scaler.scale(loss_disc).backward()
            self.scaler.unscale_(self.opt_disc_target)
            torch.nn.utils.clip_grad_norm_(self.disc_target.parameters(), max_norm=10.0)
            self.scaler.step(self.opt_disc_target)
            self.scaler.update()
        
        # Train Generator
        self.opt_gen.zero_grad()
        
        with autocast('cuda', enabled=self.use_amp):
            # Forward pass
            phase_arg = target_phase if use_phase else None
            fake_target = self.generator(source, phase_arg)
            
            # Cycle consistency
            # For autoencoder: input → recon1 → recon2 (compare recon2 with input)
            # For phase translation: source → target → source
            cycle_source = None
            if use_cycle:
                cycle_phase_arg = source_phase if use_phase else None
                cycle_source = self.generator(fake_target, cycle_phase_arg)
            
            # Adversarial loss
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
        self.scaler.unscale_(self.opt_gen)
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=10.0)
        self.scaler.step(self.opt_gen)
        self.scaler.update()
        
        self.global_step += 1
        return {
            'gen_total': loss_dict['total'],
            'gen_adv': loss_dict['adv'],
            'gen_cycle': loss_dict['cycle'],
            'gen_mse': loss_dict['mse'],
            'disc': loss_disc.item() if isinstance(loss_disc, torch.Tensor) else loss_disc
        }

    def validate(self, val_loader: DataLoader) -> Dict:
        """Validation."""
        self.generator.eval()
        val_losses = []
        psnr_values = []
        ssim_values = []
        
        with torch.no_grad():
            for batch in val_loader:
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
                
                with autocast('cuda', enabled=self.use_amp):
                    generated = self.generator(source, phase_arg)
                    loss = F.mse_loss(generated, target)
                
                val_losses.append(loss.item())
                
                # # Compute metrics
                generated_np = generated.cpu().numpy()
                target_np = target.cpu().numpy()
                
                # # Inside validate loop:
                # generated_np = (generated_np * std + mean).clip(-1024, 3072)  # or whatever your original range was
                # target_np    = (target_np    * std + mean).clip(-1024, 3072)

                
                for i in range(generated_np.shape[0]):
                    mse  = np.mean((generated_np[i] - target_np[i]) ** 2)
                    psnr = 10 * np.log10(((target_np[i].max() - target_np[i].min())**2) / (mse + 1e-8))
                    ssim_val = ssim(target_np[i,0], generated_np[i,0], data_range=np.ptp(target_np[i]))

                    # mse = np.mean((generated_np[i] - target_np[i]) ** 2)
                    # psnr = 10 * np.log10(4.0 / (mse + 1e-10))
                    psnr_values.append(psnr)
                    
                    # ssim = 1 - mse / 4.0
                    ssim_values.append(max(0, min(1, ssim_val)))
        
        self.generator.train()
        
        return {
            'val_loss': np.mean(val_losses),
            'psnr': np.mean(psnr_values),
            'ssim': np.mean(ssim_values)
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int, start_epoch: int = 0):
              
        """Main training loop."""
        save_samples_interval = self.config.get('save_samples_interval', 1)
        num_samples = self.config.get('num_samples', 3)
        
        mode_str = "AUTOENCODER" if self.use_autoencoder else "PHASE TRANSLATION"
        logger.info(f"\nStarting {mode_str} training for {epochs} epochs")
        logger.info(f"Batches: train={len(train_loader)}, val={len(val_loader)}")
        
        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch
            self.combined_loss.set_epoch(epoch)
            
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
                dynamic_ncols=True
            )
            
            for batch_idx, batch in enumerate(pbar):
                try:
                    losses = self.train_step(batch)
                    
                    for key in epoch_losses:
                        epoch_losses[key].append(losses[key])
                    
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
                        self.memory_manager.emergency_cleanup()
                        continue
                    raise e
            
            # Average losses
            train_losses = {k: np.mean(v) if v else 0.0 for k, v in epoch_losses.items()}
            
            # Validate
            val_metrics = self.validate(val_loader)
            
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
            logger.info(f"  Val - Loss: {val_metrics['val_loss']:.4f}, "
                       f"PSNR: {val_metrics['psnr']:.2f} dB, "
                       f"SSIM: {val_metrics['ssim']:.4f}")
            
            lr_gen = self.opt_gen.param_groups[0]['lr']
            lr_disc = self.opt_disc_target.param_groups[0]['lr'] if self.opt_disc_target else 0.0
            
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
                    save_nifti=self.config.get('save_nifti', True),
                    
                    is_autoencoder = self.use_autoencoder,
                    use_cycle = self.config.get('use_cycle_consistency', False) # NEW: Whether cycle consistency is used
                    )
                # Training samples (very important for debugging!)
                save_sample_patches(
                    self.generator,
                    train_loader,
                    epoch + 1,
                    self.samples_dir,
                    self.device,
                    num_samples=num_samples,
                    save_nifti=False,
                    is_autoencoder=self.use_autoencoder,
                    use_cycle=self.config.get('use_cycle_consistency', False),
                    dataset_name="train"
                )

                # save_val_samples(
                #         self.generator,
                #         val_loader,
                #         epoch,
                #         self.device,
                #         self.samples_dir,
                #         num_samples=num_samples,
                #         # save_nifti=self.config.get('save_nifti', False),
                #         # is_autoencoder=self.use_autoencoder,
                #         # use_cycle=self.config.get('use_cycle_consistency', False)  # NEW
                #     )
            
            # Save checkpoint
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
            
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Early stopping
            self.early_stopping(val_metrics['val_loss'])
            if self.early_stopping.early_stop:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            if self.scheduler_gen is not None:
                self.scheduler_gen.step()
                current_lr = self.scheduler_gen.get_last_lr()[0]
                logger.info(f"  Learning rate: {current_lr:.6f}")
        
        logger.info("Training completed!")
    
    def save_checkpoint(self, epoch: int, val_metrics: Dict, is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'opt_gen_state_dict': self.opt_gen.state_dict(),
            'val_metrics': val_metrics,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.disc_target is not None:
            checkpoint['disc_target_state_dict'] = self.disc_target.state_dict()
            checkpoint['opt_disc_target_state_dict'] = self.opt_disc_target.state_dict()
        
        torch.save(checkpoint, self.output_dir / f'checkpoint_epoch_{epoch}.pth')
        
        if is_best:
            torch.save(checkpoint, self.output_dir / 'checkpoint_best.pth')
        
        # Keep only last N checkpoints
        keep_last_n = self.config.get('keep_last_n_checkpoints', 3)
        checkpoints = sorted(self.output_dir.glob('checkpoint_epoch_*.pth'))
        for old in checkpoints[:-keep_last_n]:
            old.unlink()

    def load_checkpoint(self, checkpoint_path: Path) -> bool:
        """Load checkpoint."""
        try:
            torch.serialization.add_safe_globals([np.core.multiarray.scalar])
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            logger.info("Loading generator state...")
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            
            if 'opt_gen_state_dict' in checkpoint:
                logger.info("Loading generator optimizer state...")
                self.opt_gen.load_state_dict(checkpoint['opt_gen_state_dict'])
            
            use_disc = self.config.get('use_discriminator', False)
            has_disc_in_checkpoint = 'disc_target_state_dict' in checkpoint
            
            if use_disc and has_disc_in_checkpoint:
                logger.info("Loading discriminator state...")
                self.disc_target.load_state_dict(checkpoint['disc_target_state_dict'])
                
                if 'opt_disc_target_state_dict' in checkpoint:
                    logger.info("Loading discriminator optimizer state...")
                    self.opt_disc_target.load_state_dict(checkpoint['opt_disc_target_state_dict'])
            
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