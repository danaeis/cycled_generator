"""
OPTIMIZED CT Phase Generation Training
=====================================
Features:
- Discrete (0-255) or continuous value support
- Phase-conditioned generation
- Mask-weighted losses with high-value region penalty
- Volume caching and gradient scheduling
- Patch saving after each epoch
- Multiple normalization strategies
"""

import os
import gc
import json
import random
import logging
import warnings
from pathlib import Path
from collections import OrderedDict, Counter
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
import nibabel as nib
import psutil

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# NORMALIZATION STRATEGIES
# ============================================================================

class NormalizationStrategy:
    """
    Multiple normalization strategies for CT data.
    Considers benefits of histogram equalization for better contrast.
    """
    
    @staticmethod
    def minmax_normalize(volume: np.ndarray, min_val: float = 0, max_val: float = 255) -> np.ndarray:
        """Simple min-max normalization to [-1, 1]."""
        volume = np.clip(volume, min_val, max_val)
        return (volume - min_val) / (max_val - min_val + 1e-8) * 2.0 - 1.0
    
    @staticmethod
    def zscore_normalize(volume: np.ndarray, clip_range: float = 3.0) -> np.ndarray:
        """Z-score normalization with optional clipping."""
        mean = volume.mean()
        std = volume.std() + 1e-8
        normalized = (volume - mean) / std
        if clip_range > 0:
            normalized = np.clip(normalized, -clip_range, clip_range)
            normalized = normalized / clip_range  # Scale to [-1, 1]
        return normalized
    
    @staticmethod
    def percentile_normalize(volume: np.ndarray, 
                            low_pct: float = 1.0, 
                            high_pct: float = 99.0) -> np.ndarray:
        """Percentile-based normalization (robust to outliers)."""
        p_low = np.percentile(volume, low_pct)
        p_high = np.percentile(volume, high_pct)
        volume = np.clip(volume, p_low, p_high)
        return (volume - p_low) / (p_high - p_low + 1e-8) * 2.0 - 1.0
    
    @staticmethod
    def histogram_equalization_normalize(volume: np.ndarray, 
                                         num_bins: int = 256) -> np.ndarray:
        """
        Histogram equalization for better contrast distribution.
        Useful when discrete values have uneven distribution.
        """
        # Flatten and compute histogram
        flat = volume.ravel()
        hist, bin_edges = np.histogram(flat, bins=num_bins, range=(0, 255))
        cdf = hist.cumsum()
        cdf_normalized = cdf / cdf[-1]  # Normalize to [0, 1]
        
        # Map values using CDF
        volume_normalized = np.interp(volume.ravel(), bin_edges[:-1], cdf_normalized)
        volume_normalized = volume_normalized.reshape(volume.shape)
        
        # Scale to [-1, 1]
        return volume_normalized * 2.0 - 1.0
    
    @staticmethod
    def clahe_normalize(volume: np.ndarray, 
                       clip_limit: float = 2.0, 
                       tile_size: int = 8) -> np.ndarray:
        """
        CLAHE (Contrast Limited Adaptive Histogram Equalization).
        Best for preserving local contrast while limiting noise amplification.
        Applied slice-by-slice for 3D volumes.
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, 
                                tileGridSize=(tile_size, tile_size))
        
        normalized = np.zeros_like(volume, dtype=np.float32)
        
        for d in range(volume.shape[0]):
            slice_2d = volume[d].astype(np.uint8)
            equalized = clahe.apply(slice_2d)
            normalized[d] = equalized.astype(np.float32)
        
        # Scale to [-1, 1]
        return normalized / 127.5 - 1.0
    
    @staticmethod
    def adaptive_normalize(volume: np.ndarray, 
                          discrete_mode: bool = True,
                          strategy: str = 'clahe') -> np.ndarray:
        """
        Adaptive normalization based on data type and strategy.
        
        Args:
            volume: Input volume
            discrete_mode: True if values are discrete 0-255
            strategy: 'minmax', 'zscore', 'percentile', 'histeq', 'clahe'
        """
        if discrete_mode:
            if strategy == 'minmax':
                return NormalizationStrategy.minmax_normalize(volume, 0, 255)
            elif strategy == 'zscore':
                return NormalizationStrategy.zscore_normalize(volume)
            elif strategy == 'percentile':
                return NormalizationStrategy.percentile_normalize(volume, 1, 99)
            elif strategy == 'histeq':
                return NormalizationStrategy.histogram_equalization_normalize(volume)
            elif strategy == 'clahe':
                return NormalizationStrategy.clahe_normalize(volume)
            else:
                return NormalizationStrategy.minmax_normalize(volume, 0, 255)
        else:
            # Continuous HU values
            if strategy == 'zscore':
                return NormalizationStrategy.zscore_normalize(volume)
            elif strategy == 'percentile':
                return NormalizationStrategy.percentile_normalize(volume, 0.5, 99.5)
            else:
                # Default HU normalization
                volume = np.clip(volume, -1000, 1000)
                return (volume + 1000) / 2000.0 * 2.0 - 1.0


# ============================================================================
# VOLUME CACHE (UNCHANGED)
# ============================================================================

class VolumeCache:
    """LRU cache for NIfTI volumes to minimize disk I/O."""
    
    def __init__(self, max_cache_size: int = 8):
        self.max_cache_size = max_cache_size
        self.cache = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        
    def get_volume(self, path: Path, use_memmap: bool = True) -> np.ndarray:
        path_str = str(path)
        
        if path_str in self.cache:
            self.cache_hits += 1
            self.cache.move_to_end(path_str)
            return self.cache[path_str]
        
        self.cache_misses += 1
        nii = nib.load(path)
        
        if use_memmap and hasattr(nii.dataobj, '_mmap'):
            volume = np.asarray(nii.dataobj)
        else:
            volume = nii.get_fdata()
        
        volume = np.transpose(volume, (2, 1, 0))
        
        self.cache[path_str] = volume
        self.cache.move_to_end(path_str)
        
        if len(self.cache) > self.max_cache_size:
            self.cache.popitem(last=False)
        
        return volume
    
    def get_volume_shape(self, path: Path) -> Tuple[int, int, int]:
        nii = nib.load(path)
        shape = nii.shape
        return (shape[2], shape[1], shape[0])
    
    def clear(self):
        self.cache.clear()
        gc.collect()
    
    def get_stats(self) -> Dict:
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
        mem = self.get_memory_info()
        logger.info(f"Memory - GPU: {mem['gpu_allocated_gb']:.2f}GB, "
                   f"RAM: {mem['ram_percent']:.1f}%, Disk: {mem['disk_percent']:.1f}%")
    
    def cleanup_temp_files(self):
        cleaned_size = 0
        for pattern in self.cleanup_patterns:
            for file in self.temp_dir.glob(pattern):
                try:
                    size = file.stat().st_size
                    file.unlink()
                    cleaned_size += size
                except:
                    pass
        
        if cleaned_size > 1024**3:
            logger.info(f"  Freed {cleaned_size / 1024**3:.2f}GB from temp")
        
        plt.close('all')
    
    def cleanup_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
    def check_and_cleanup(self, force: bool = False, current_epoch: int = None):
        mem = self.get_memory_info()
        should_warn = (mem['ram_percent'] > 80 or mem['disk_percent'] > 80)
        if current_epoch is not None:
            should_warn = should_warn and (current_epoch != self.last_warning_epoch)
        
        if should_warn or force:
            if should_warn and current_epoch is not None:
                logger.warning(f"High memory! RAM: {mem['ram_percent']:.1f}%, "
                             f"Disk: {mem['disk_percent']:.1f}% (Epoch {current_epoch})")
                self.last_warning_epoch = current_epoch
            self.cleanup_cache()
            self.cleanup_temp_files()
    
    def emergency_cleanup(self):
        plt.close('all')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
        for _ in range(3):
            gc.collect()
        self.cleanup_temp_files()


# ============================================================================
# OPTIMIZED DATASET WITH DISCRETE/CONTINUOUS SUPPORT
# ============================================================================

class CTPhaseDataset(Dataset):
    """
    Memory-efficient dataset with:
    - Volume caching (LRU eviction)
    - Discrete (0-255) or continuous value support
    - Multiple normalization strategies
    - Mask extraction for weighted losses
    """
    
    def __init__(
        self,
        data_pairs: List[Dict],
        patch_size: Tuple[int, int] = (128, 192),
        patch_depth: int = 15,
        overlap_ratio: float = 0.5,
        augment: bool = True,
        body_focused: bool = True,
        body_threshold: float = 50.0,  # For discrete 0-255 data
        min_fg_ratio: float = 0.6,
        max_same_value_ratio: float = 0.80,
        cache_size: int = 8,
        use_memmap: bool = True,
        # NEW: Discrete mode parameters
        discrete_mode: bool = True,
        normalization_strategy: str = 'clahe',  # 'minmax', 'zscore', 'percentile', 'histeq', 'clahe'
        # Validation parameters
        min_intensity_ratio: float = 0.7,
        min_mean: float = -0.3,
        min_std: float = 0.2,
        validate_patches: bool = False
    ):
        self.data_pairs = data_pairs
        self.patch_size = patch_size
        self.patch_depth = patch_depth
        self.overlap_ratio = overlap_ratio
        self.augment = augment
        self.body_focused = body_focused
        self.min_fg_ratio = min_fg_ratio
        self.max_same_value_ratio = max_same_value_ratio
        self.patch_coords = []
        self.padding_info = {}
        
        # NEW: Discrete mode settings
        self.discrete_mode = discrete_mode
        self.normalization_strategy = normalization_strategy
        
        # Adjust body threshold based on mode
        if discrete_mode:
            self.body_threshold = body_threshold if body_threshold > 0 else 50.0
        else:
            self.body_threshold = -500.0  # HU threshold for continuous
        
        # Volume cache
        self.volume_cache = VolumeCache(max_cache_size=cache_size)
        self.use_memmap = use_memmap
        
        # Validation parameters
        self.min_intensity_ratio = min_intensity_ratio
        self.min_mean = min_mean
        self.min_std = min_std
        self.validate_patches = validate_patches
        
        # Phase mapping
        self.phase_to_idx = {
            'non-contrast': 0,
            'arterial': 1,
            'portal': 2,
            'venous': 2,
            'delayed': 3
        }
        
        logger.info(f"Dataset: {len(data_pairs)} pairs, patch {patch_size}x{patch_depth}")
        logger.info(f"Discrete mode: {discrete_mode}, Normalization: {normalization_strategy}")
        logger.info(f"Volume cache: max {cache_size} volumes")
        
        self._compute_patch_coordinates()
        logger.info(f"Generated {len(self.patch_coords)} patches")
    
    def _normalize_patch(self, patch: np.ndarray) -> np.ndarray:
        """Apply selected normalization strategy."""
        return NormalizationStrategy.adaptive_normalize(
            patch, 
            discrete_mode=self.discrete_mode,
            strategy=self.normalization_strategy
        )
    
    def _same_value_ratio(self, patch: np.ndarray) -> float:
        """Fraction of voxels equal to the mode."""
        flat = patch.ravel()
        if flat.size == 0:
            return 1.0
        values, counts = np.unique(flat, return_counts=True)
        return float(counts.max() / flat.size)

    def _fg_ratio(self, patch: np.ndarray) -> float:
        """Fraction of voxels above body threshold."""
        return (patch > self.body_threshold).mean()

    def is_valid_patch(self, source_raw: np.ndarray, target_raw: np.ndarray) -> Tuple[bool, str]:
        """Validate patch quality."""
        # Normalize for checks
        source_norm = (source_raw - source_raw.mean()) / (source_raw.std() + 1e-8)
        target_norm = (target_raw - target_raw.mean()) / (target_raw.std() + 1e-8)

        # Dark ratio check
        dark_ratio_s = (source_norm < -0.9).mean()
        dark_ratio_t = (target_norm < -0.9).mean()
        if dark_ratio_s > (1 - self.min_intensity_ratio) or dark_ratio_t > (1 - self.min_intensity_ratio):
            return False, f"dark_s={dark_ratio_s:.3f}, dark_t={dark_ratio_t:.3f}"

        # Mean intensity check
        mean_s = source_norm.mean()
        mean_t = target_norm.mean()
        if mean_s < self.min_mean or mean_t < self.min_mean:
            return False, f"mean_s={mean_s:.3f}, mean_t={mean_t:.3f}"

        # Standard deviation check
        std_s = source_norm.std()
        std_t = target_norm.std()
        if std_s < self.min_std or std_t < self.min_std:
            return False, f"std_s={std_s:.3f}, std_t={std_t:.3f}"

        # Foreground ratio check
        fg_s = self._fg_ratio(source_raw)
        fg_t = self._fg_ratio(target_raw)
        if fg_s < self.min_fg_ratio or fg_t < self.min_fg_ratio:
            return False, f"fg_s={fg_s:.3f}, fg_t={fg_t:.3f}"

        # Uniformity check
        same_s = self._same_value_ratio(source_raw)
        same_t = self._same_value_ratio(target_raw)
        if same_s > self.max_same_value_ratio or same_t > self.max_same_value_ratio:
            return False, f"uniform_s={same_s:.3f}, uniform_t={same_t:.3f}"

        return True, ""

    def _find_body_center(self, volume: np.ndarray) -> Tuple[int, int]:
        """Find body center from middle slices."""
        mid_start = volume.shape[0] // 3
        mid_end = 2 * volume.shape[0] // 3
        mid_slices = volume[mid_start:mid_end]
        body_mask = mid_slices > self.body_threshold
        
        if body_mask.sum() > 0:
            coords = np.argwhere(body_mask)
            return int(np.median(coords[:, 1])), int(np.median(coords[:, 2]))
        return volume.shape[1] // 2, volume.shape[2] // 2
    
    def _compute_patch_coordinates(self):
        """Compute all valid patch coordinates."""
        padding = self.patch_depth // 2
        
        for pair_idx, pair_data in enumerate(self.data_pairs):
            try:
                source_shape = self.volume_cache.get_volume_shape(pair_data['source_path'])
                target_shape = self.volume_cache.get_volume_shape(pair_data['target_path'])
                
                if source_shape != target_shape:
                    continue
                
                depth, height, width = source_shape
                
                if depth < self.patch_depth + 2 or height < self.patch_size[0] or width < self.patch_size[1]:
                    continue
                
                step_y = max(1, int(self.patch_size[0] * (1 - self.overlap_ratio)))
                step_x = max(1, int(self.patch_size[1] * (1 - self.overlap_ratio)))
                
                z_start = padding + (self.patch_depth // 2)
                z_end = depth - padding - (self.patch_depth // 2)
                z_range = range(z_start, z_end) if z_end > z_start else []

                if self.body_focused:
                    source_vol = self.volume_cache.get_volume(
                        pair_data['source_path'], 
                        use_memmap=self.use_memmap
                    )
                    center_y, center_x = self._find_body_center(source_vol)
                    y_start_center = max(0, min(center_y - self.patch_size[0] // 2, height - self.patch_size[0]))
                    x_start_center = max(0, min(center_x - self.patch_size[1] // 2, width - self.patch_size[1]))
                    
                    y_positions = [y_start_center]
                    x_positions = [x_start_center]
                    
                    for dy in [-step_y, step_y]:
                        y = y_start_center + dy
                        if 0 <= y <= height - self.patch_size[0]:
                            y_positions.append(y)
                    
                    for dx in [-step_x, step_x]:
                        x = x_start_center + dx
                        if 0 <= x <= width - self.patch_size[1]:
                            x_positions.append(x)
                else:
                    y_positions = list(range(0, height - self.patch_size[0] + 1, step_y))
                    x_positions = list(range(0, width - self.patch_size[1] + 1, step_x))
                
                for z in z_range:
                    for y in y_positions:
                        for x in x_positions:
                            self.patch_coords.append({
                                'pair_idx': pair_idx,
                                'z': z,
                                'y': y,
                                'x': x
                            })
                
            except Exception as e:
                logger.warning(f"Skipping pair {pair_idx}: {e}")
                continue
        
        if len(self.patch_coords) == 0:
            logger.error("⚠️ NO PATCHES GENERATED!")
    
    def __len__(self) -> int:
        return len(self.patch_coords)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Extract patch with proper normalization and masks."""
        coord = self.patch_coords[idx]
        pair_data = self.data_pairs[coord['pair_idx']]
        
        # Get cached volumes
        source_vol = self.volume_cache.get_volume(
            pair_data['source_path'], 
            use_memmap=self.use_memmap
        )
        target_vol = self.volume_cache.get_volume(
            pair_data['target_path'], 
            use_memmap=self.use_memmap
        )
        
        # Extract patch
        z, y, x = coord['z'], coord['y'], coord['x']
        half_depth = self.patch_depth // 2
        
        z_start = max(0, z - half_depth)
        z_end = min(source_vol.shape[0], z + half_depth + 1)
        
        source_patch = source_vol[z_start:z_end, y:y+self.patch_size[0], x:x+self.patch_size[1]].copy()
        target_patch = target_vol[z_start:z_end, y:y+self.patch_size[0], x:x+self.patch_size[1]].copy()
        
        # Store raw values for high-value penalty computation
        source_raw = source_patch.copy()
        target_raw = target_patch.copy()
        
        # Phase encoding
        source_phase_idx = self.phase_to_idx.get(pair_data['source_phase'], 0)
        target_phase_idx = self.phase_to_idx.get(pair_data['target_phase'], 1)
        
        # Handle depth padding
        if source_patch.shape[0] < self.patch_depth:
            pad_before = (self.patch_depth - source_patch.shape[0]) // 2
            pad_after = self.patch_depth - source_patch.shape[0] - pad_before
            source_patch = np.pad(source_patch, ((pad_before, pad_after), (0, 0), (0, 0)), mode='edge')
            target_patch = np.pad(target_patch, ((pad_before, pad_after), (0, 0), (0, 0)), mode='edge')
            source_raw = np.pad(source_raw, ((pad_before, pad_after), (0, 0), (0, 0)), mode='edge')
            target_raw = np.pad(target_raw, ((pad_before, pad_after), (0, 0), (0, 0)), mode='edge')
        
        # Validate patches if enabled
        if self.validate_patches:
            is_valid, reason = self.is_valid_patch(source_raw, target_raw)
            if not is_valid:
                return {
                    'skip': True,
                    'reason': reason,
                    'idx': idx,
                    'case_id': pair_data.get('case_id', 'unknown')
                }
        
        # Apply normalization
        source_patch = self._normalize_patch(source_patch)
        target_patch = self._normalize_patch(target_patch)
        
        # Convert to tensors
        source_tensor = torch.from_numpy(source_patch).unsqueeze(0).float()
        target_tensor = torch.from_numpy(target_patch).unsqueeze(0).float()
        
        # Create intensity weight map for high-value penalty
        # Higher weights for high-intensity regions
        if self.discrete_mode:
            # For 0-255 data, high values are > 180
            high_value_threshold = 180
            intensity_weights = np.where(target_raw > high_value_threshold, 2.0, 1.0)
        else:
            # For HU data, high values are > 200 HU (contrast-enhanced regions)
            high_value_threshold = 200
            intensity_weights = np.where(target_raw > high_value_threshold, 2.0, 1.0)
        
        intensity_weight_tensor = torch.from_numpy(intensity_weights).unsqueeze(0).float()
        
        # Extract masks
        masks = {}
        if 'target_seg' in pair_data and pair_data['target_seg']:
            try:
                seg_path = Path(pair_data['target_seg'])
                if seg_path.exists():
                    seg_vol = self.volume_cache.get_volume(seg_path, use_memmap=self.use_memmap)

                    seg_patch = seg_vol[z_start:z_end, y:y+self.patch_size[0], x:x+self.patch_size[1]].copy()

                    # Resize to match CT patch
                    target_d, target_h, target_w = self.patch_depth, self.patch_size[0], self.patch_size[1]
                    current_d, current_h, current_w = seg_patch.shape

                    if current_d < target_d:
                        pad_before = (target_d - current_d) // 2
                        pad_after = target_d - current_d - pad_before
                        seg_patch = np.pad(seg_patch, ((pad_before, pad_after), (0,0), (0,0)), mode='edge')
                    elif current_d > target_d:
                        start = (current_d - target_d) // 2
                        seg_patch = seg_patch[start:start + target_d, :, :]

                    if (current_h, current_w) != (target_h, target_w):
                        resized = np.zeros((target_d, target_h, target_w), dtype=seg_patch.dtype)
                        for d in range(target_d):
                            resized[d] = cv2.resize(
                                seg_patch[d],
                                (target_w, target_h),
                                interpolation=cv2.INTER_NEAREST
                            )
                        seg_patch = resized

                    seg_tensor = torch.from_numpy(seg_patch)[None, ...].float()
                    
                    # Organ mapping
                    organ_labels = {
                        'liver': [1],
                        'spleen': [2],
                        'kidney_right': [3],
                        'kidney_left': [4],
                        'pancreas': [5],
                    }

                    for organ, labels in organ_labels.items():
                        mask = torch.zeros_like(seg_tensor)
                        for lbl in labels:
                            mask = torch.logical_or(mask, seg_tensor == lbl).float()
                        masks[organ] = mask

            except Exception as e:
                logger.warning(f"Mask loading failed for {pair_data['case_id']}: {e}")
                masks = {}
        
        return {
            'source': source_tensor,
            'target': target_tensor,
            'source_phase': torch.tensor(source_phase_idx, dtype=torch.long),
            'target_phase': torch.tensor(target_phase_idx, dtype=torch.long),
            'case_id': pair_data['case_id'],
            'masks': masks,
            'intensity_weights': intensity_weight_tensor,
            'skip': False
        }
    
    def get_cache_stats(self) -> Dict:
        return self.volume_cache.get_stats()
    
    def clear_cache(self):
        self.volume_cache.clear()


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class ResidualBlock3D(nn.Module):
    """3D Residual Block with InstanceNorm."""
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
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        residual = x
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.norm(x + residual)
        return self.activation(x)


class Generator3D(nn.Module):
    """3D U-Net Generator with phase conditioning."""
    
    def __init__(self, num_phases: int = 4, use_checkpoint: bool = True):
        super().__init__()
        self.num_phases = num_phases
        self.use_checkpoint = use_checkpoint

        self.phase_emb = nn.Embedding(num_phases, 64)

        self.enc1 = ResidualBlock3D(1, 64)
        self.enc2 = ResidualBlock3D(64, 128)
        self.enc3 = ResidualBlock3D(128, 256)
        self.enc4 = ResidualBlock3D(256, 512)

        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.bottleneck = nn.Sequential(
            nn.Conv3d(512, 768, kernel_size=3, padding=1),
            nn.InstanceNorm3d(768),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(768, 512, kernel_size=3, padding=1),
            nn.InstanceNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.phase_scale_shift = nn.Linear(64, 512 * 2)

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False),
            nn.Conv3d(512, 256, kernel_size=3, padding=1),
            nn.InstanceNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec4 = self._make_dec(256 + 512, 256)

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False),
            nn.Conv3d(256, 128, kernel_size=3, padding=1),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec3 = self._make_dec(128 + 256, 128)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False),
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec2 = self._make_dec(64 + 128, 64)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec1 = self._make_dec(64 + 64, 64)

        self.out_conv = nn.Sequential(
            nn.Conv3d(64, 1, kernel_size=1),
            nn.Tanh()
        )

        logger.info(f"Generator3D: {sum(p.numel() for p in self.parameters())/1e6:.2f}M params")

    def _make_dec(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x: torch.Tensor, phase_idx: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        emb = self.phase_emb(phase_idx)

        def _enc(block, inp):
            if self.use_checkpoint and self.training:
                return torch.utils.checkpoint.checkpoint(block, inp, use_reentrant=False)
            return block(inp)

        e1 = _enc(self.enc1, x)
        p1 = self.pool(e1)
        e2 = _enc(self.enc2, p1)
        p2 = self.pool(e2)
        e3 = _enc(self.enc3, p2)
        p3 = self.pool(e3)
        e4 = _enc(self.enc4, p3)
        b = self.pool(e4)

        b = self.bottleneck(b)

        ss = self.phase_scale_shift(emb)
        scale, shift = ss.chunk(2, dim=1)
        scale = scale.view(B, 512, 1, 1, 1)
        shift = shift.view(B, 512, 1, 1, 1)
        b = b * (1 + scale) + shift

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

        out = self.out_conv(d)
        return out


class Discriminator3D(nn.Module):
    """3D PatchGAN Discriminator."""
    
    def __init__(self, in_channels: int = 1):
        super().__init__()
        
        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv3d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm3d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(in_channels, 32, normalize=False),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            nn.Conv3d(128, 1, 3, padding=1)
        )
    
    def forward(self, x):
        return self.model(x)


# ============================================================================
# LOSS FUNCTIONS WITH HIGH-VALUE PENALTY
# ============================================================================

class FocalFrequencyLoss(nn.Module):
    """Focal Frequency Loss for better texture generation."""
    
    def __init__(self, alpha: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss with:
    - MSE with mask weighting
    - High-value region penalty
    - Focal frequency loss
    - Cycle consistency
    - Adversarial loss
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.mse = nn.MSELoss(reduction='none')
        self.focal = FocalFrequencyLoss(alpha=1.0)
        self.current_epoch = 0
        
        # Loss weights
        self.lambda_cycle = config.get('lambda_cycle', 10.0)
        self.lambda_mse_initial = config.get('lambda_mse_initial', 1.0)
        self.lambda_mse_final = config.get('lambda_mse_final', 100.0)
        self.mse_warmup_epochs = config.get('mse_warmup_epochs', 50)
        self.lambda_adv = config.get('lambda_adv', 1.0)
        self.adv_warmup_epochs = config.get('adv_warmup_epochs', 10)
        self.lambda_organ = config.get('lambda_organ', 5.0)
        self.organ_weight = config.get('organ_weight', 10.0)
        
        # NEW: High-value penalty weight
        self.lambda_high_value = config.get('lambda_high_value', 2.0)
        
    def set_epoch(self, epoch: int):
        self.current_epoch = epoch
    
    def get_mse_weight(self) -> float:
        if self.current_epoch >= self.mse_warmup_epochs:
            return self.lambda_mse_final
        progress = self.current_epoch / self.mse_warmup_epochs
        return self.lambda_mse_initial + (self.lambda_mse_final - self.lambda_mse_initial) * progress
    
    def get_adv_weight(self) -> float:
        if self.current_epoch >= self.adv_warmup_epochs:
            return self.lambda_adv
        progress = self.current_epoch / self.adv_warmup_epochs
        return self.lambda_adv * progress
    
    def weighted_mse_loss(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
        mask_dict: Dict[str, torch.Tensor],
        intensity_weights: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute MSE loss with:
        - Organ mask weighting
        - High-value region penalty
        """
        # Base pixel-wise MSE
        pixel_mse = self.mse(pred, target)  # (B, 1, D, H, W)
        
        # Apply intensity weights for high-value penalty
        if intensity_weights is not None:
            pixel_mse = pixel_mse * intensity_weights
        
        # Global loss
        loss_global = pixel_mse.mean()
        
        # Organ-weighted loss
        if mask_dict:
            B, _, D, H, W = pred.shape
            union = torch.zeros((B, 1, D, H, W), dtype=pred.dtype, device=pred.device)
            for m in mask_dict.values():
                union = torch.max(union, m[:, :1, ...])
            
            if union.sum() > 0:
                # Weighted organ loss
                organ_mse = (pixel_mse * union).sum() / (union.sum() + 1e-8)
                loss_global = loss_global + (self.organ_weight - 1.0) * organ_mse
        
        return loss_global
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        cycle_pred: torch.Tensor, 
        source: torch.Tensor,
        adv_loss: float, 
        masks: Dict[str, torch.Tensor],
        intensity_weights: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute combined loss."""
        
        # MSE with weights
        mse_weight = self.get_mse_weight()
        loss_mse = self.weighted_mse_loss(pred, target, masks, intensity_weights) * mse_weight
        
        # Focal frequency loss with organ weighting
        loss_focal = self.lambda_organ * self.weighted_mse_loss(
            pred, target, masks, intensity_weights
        )
        
        # Cycle consistency
        loss_cycle = F.mse_loss(cycle_pred, source) * self.lambda_cycle
        
        # Adversarial loss
        adv_weight = self.get_adv_weight()
        loss_adv = adv_loss * adv_weight
        
        # Total
        total_loss = loss_mse + loss_focal + loss_cycle + loss_adv

        loss_dict = {
            'total': total_loss.item(),
            'mse': loss_mse.item(),
            'focal': loss_focal.item(),
            'cycle': loss_cycle.item(),
            'adv': loss_adv.item()
        }
        
        return total_loss, loss_dict


# ============================================================================
# LOSS TRACKING
# ============================================================================

def convert_numpy(obj):
    """Convert numpy types to Python native types."""
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
            'train_gen_focal': [],
            'train_disc': [],
            'val_loss': [],
            'val_psnr': [],
            'val_ssim': [],
            'lr_gen': [],
            'lr_disc': [],
            'adv_weight': []
        }
        
    def update_epoch(self, epoch: int, train_losses: Dict, val_metrics: Dict,
                     lr_gen: float, lr_disc: float, adv_weight: float):
        self.history['epoch'].append(epoch)
        self.history['train_gen_total'].append(train_losses.get('gen_total', 0))
        self.history['train_gen_adv'].append(train_losses.get('gen_adv', 0))
        self.history['train_gen_cycle'].append(train_losses.get('gen_cycle', 0))
        self.history['train_gen_mse'].append(train_losses.get('gen_mse', 0))
        self.history['train_gen_focal'].append(train_losses.get('gen_focal', 0))
        self.history['train_disc'].append(train_losses.get('disc', 0))
        self.history['val_loss'].append(val_metrics.get('val_loss', 0))
        self.history['val_psnr'].append(val_metrics.get('psnr', 0))
        self.history['val_ssim'].append(val_metrics.get('ssim', 0))
        self.history['lr_gen'].append(lr_gen)
        self.history['lr_disc'].append(lr_disc)
        self.history['adv_weight'].append(adv_weight)
    
    def save_json(self):
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(convert_numpy(self.history), f, indent=2)
    
    def create_summary_report(self, current_epoch: int):
        if not self.history['epoch']:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Training Summary - Epoch {current_epoch}', fontsize=16)
        
        epochs = self.history['epoch']
        
        # Generator losses
        ax = axes[0, 0]
        ax.plot(epochs, self.history['train_gen_mse'], label='MSE', linewidth=2)
        ax.plot(epochs, self.history['train_gen_focal'], label='Focal', linewidth=2)
        ax.plot(epochs, self.history['train_gen_cycle'], label='Cycle', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Generator Component Losses')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Total losses
        ax = axes[0, 1]
        ax.plot(epochs, self.history['train_gen_total'], label='Generator', linewidth=2)
        ax.plot(epochs, self.history['train_disc'], label='Discriminator', linewidth=2)
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
        ax.plot(epochs, self.history['lr_disc'], label='Discriminator LR', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rates')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Adversarial weight
        ax = axes[1, 1]
        ax.plot(epochs, self.history['adv_weight'], linewidth=2, color='red')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Weight')
        ax.set_title('Adversarial Loss Weight (Warmup)')
        ax.grid(True, alpha=0.3)
        
        # Validation loss
        ax = axes[1, 2]
        ax.plot(epochs, self.history['val_loss'], linewidth=2, color='purple')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Validation Loss')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'training_summary_epoch_{current_epoch}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        self.save_json()


# ============================================================================
# PATCH SAVING UTILITY
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
    """Save sample generated patches for visual inspection."""
    generator.eval()
    save_dir = Path(save_dir) / f"epoch_{epoch}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    png_dir = save_dir / 'comparisons'
    nifti_dir = save_dir / 