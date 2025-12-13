import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import OrderedDict
from tqdm import tqdm
import logging
import json
import gc
import psutil
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from enum import Enum

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# NORMALIZATION STRATEGIES
# ============================================================================

class NormalizationMode(Enum):
    """Supported normalization modes for CT volumes."""
    DISCRETE_MINMAX = "discrete_minmax"          # For 0-255 discrete values
    DISCRETE_ZSCORE = "discrete_zscore"          # Z-score for discrete values
    CONTINUOUS_HU = "continuous_hu"               # Raw HU values [-1000, 1000]
    CONTINUOUS_ZSCORE = "continuous_zscore"       # Z-score for continuous
    PERCENTILE = "percentile"                     # Percentile-based clipping
    ADAPTIVE_HISTOGRAM = "adaptive_histogram"     # CLAHE-aware normalization
    ROBUST_ZSCORE = "robust_zscore"               # Median/IQR based


class CTNormalizer:
    """
    Unified normalizer for CT volumes supporting discrete and continuous modes.
    
    Key features:
    - Supports histogram-equalized (discrete 0-255) and raw HU values
    - Multiple normalization strategies
    - Reversible normalization for inference
    """
    
    def __init__(
        self,
        mode: Union[str, NormalizationMode] = NormalizationMode.DISCRETE_MINMAX,
        clip_range: Tuple[float, float] = None,
        percentile_range: Tuple[float, float] = (1, 99),
        output_range: Tuple[float, float] = (-1, 1),
        global_stats: Dict = None  # For consistent normalization across dataset
    ):
        """
        Args:
            mode: Normalization mode
            clip_range: Value clipping range (auto-detected if None)
            percentile_range: Percentiles for percentile-based normalization
            output_range: Target output range (default [-1, 1])
            global_stats: Pre-computed dataset statistics for consistent normalization
        """
        if isinstance(mode, str):
            mode = NormalizationMode(mode)
        self.mode = mode
        self.clip_range = clip_range
        self.percentile_range = percentile_range
        self.output_range = output_range
        self.global_stats = global_stats or {}
        
        # Auto-detect clip range based on mode
        if self.clip_range is None:
            if mode in [NormalizationMode.DISCRETE_MINMAX, NormalizationMode.DISCRETE_ZSCORE]:
                self.clip_range = (0, 255)
            elif mode in [NormalizationMode.CONTINUOUS_HU, NormalizationMode.CONTINUOUS_ZSCORE]:
                self.clip_range = (-1000, 1000)
            else:
                self.clip_range = None  # Computed per-patch
        
        logger.info(f"CTNormalizer initialized: mode={mode.value}, clip_range={self.clip_range}")
    
    def normalize(self, volume: np.ndarray, return_params: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        Normalize a volume based on the selected mode.
        
        Args:
            volume: Input volume (any shape)
            return_params: If True, return normalization parameters for reversal
            
        Returns:
            Normalized volume, optionally with parameters
        """
        params = {}
        
        if self.mode == NormalizationMode.DISCRETE_MINMAX:
            normalized = self._discrete_minmax(volume)
            params = {'mode': 'discrete_minmax', 'range': self.clip_range}
            
        elif self.mode == NormalizationMode.DISCRETE_ZSCORE:
            normalized, params = self._zscore(volume, discrete=True)
            
        elif self.mode == NormalizationMode.CONTINUOUS_HU:
            normalized = self._continuous_hu(volume)
            params = {'mode': 'continuous_hu', 'range': self.clip_range}
            
        elif self.mode == NormalizationMode.CONTINUOUS_ZSCORE:
            normalized, params = self._zscore(volume, discrete=False)
            
        elif self.mode == NormalizationMode.PERCENTILE:
            normalized, params = self._percentile_normalize(volume)
            
        elif self.mode == NormalizationMode.ADAPTIVE_HISTOGRAM:
            normalized, params = self._adaptive_histogram_normalize(volume)
            
        elif self.mode == NormalizationMode.ROBUST_ZSCORE:
            normalized, params = self._robust_zscore(volume)
            
        else:
            raise ValueError(f"Unknown normalization mode: {self.mode}")
        
        if return_params:
            return normalized, params
        return normalized
    
    def denormalize(self, volume: np.ndarray, params: Dict) -> np.ndarray:
        """Reverse normalization using stored parameters."""
        mode = params.get('mode', self.mode.value)
        out_min, out_max = self.output_range
        
        if mode in ['discrete_minmax', 'continuous_hu']:
            in_min, in_max = params.get('range', self.clip_range)
            # Reverse: output_range -> clip_range
            volume = (volume - out_min) / (out_max - out_min)
            return volume * (in_max - in_min) + in_min
            
        elif 'zscore' in mode:
            mean = params.get('mean', 0)
            std = params.get('std', 1)
            return volume * std + mean
            
        elif mode == 'percentile':
            p_low, p_high = params.get('percentiles', (0, 255))
            volume = (volume - out_min) / (out_max - out_min)
            return volume * (p_high - p_low) + p_low
            
        else:
            logger.warning(f"Denormalization not implemented for mode: {mode}")
            return volume
    
    def _discrete_minmax(self, volume: np.ndarray) -> np.ndarray:
        """Normalize discrete 0-255 values to output_range."""
        volume = np.clip(volume, self.clip_range[0], self.clip_range[1])
        out_min, out_max = self.output_range
        return (volume / 255.0) * (out_max - out_min) + out_min
    
    def _continuous_hu(self, volume: np.ndarray) -> np.ndarray:
        """Normalize continuous HU values to output_range."""
        volume = np.clip(volume, self.clip_range[0], self.clip_range[1])
        out_min, out_max = self.output_range
        in_min, in_max = self.clip_range
        return (volume - in_min) / (in_max - in_min) * (out_max - out_min) + out_min
    
    def _zscore(self, volume: np.ndarray, discrete: bool = True) -> Tuple[np.ndarray, Dict]:
        """Z-score normalization with optional global stats."""
        if discrete:
            volume = np.clip(volume, 0, 255).astype(np.float32)
        
        if self.global_stats:
            mean = self.global_stats.get('mean', volume.mean())
            std = self.global_stats.get('std', volume.std())
        else:
            mean = volume.mean()
            std = volume.std() + 1e-8
        
        normalized = (volume - mean) / std
        
        # Clip to reasonable range and scale to output_range
        normalized = np.clip(normalized, -3, 3)
        out_min, out_max = self.output_range
        normalized = (normalized / 3.0) * ((out_max - out_min) / 2) + (out_max + out_min) / 2
        
        params = {'mode': 'zscore', 'mean': float(mean), 'std': float(std)}
        return normalized, params
    
    def _percentile_normalize(self, volume: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Percentile-based normalization (robust to outliers)."""
        p_low, p_high = self.percentile_range
        v_low = np.percentile(volume, p_low)
        v_high = np.percentile(volume, p_high)
        
        volume = np.clip(volume, v_low, v_high)
        out_min, out_max = self.output_range
        normalized = (volume - v_low) / (v_high - v_low + 1e-8) * (out_max - out_min) + out_min
        
        params = {'mode': 'percentile', 'percentiles': (float(v_low), float(v_high))}
        return normalized, params
    
    def _adaptive_histogram_normalize(self, volume: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Normalization that preserves histogram equalization benefits.
        Assumes input is already histogram-equalized (0-255).
        """
        # For CLAHE-processed data, use percentile to handle edge cases
        volume = np.clip(volume, 0, 255).astype(np.float32)
        
        # Use robust statistics
        p1 = np.percentile(volume, 1)
        p99 = np.percentile(volume, 99)
        
        # Soft clipping to preserve histogram structure
        volume_clipped = np.clip(volume, p1, p99)
        
        # Scale to output range
        out_min, out_max = self.output_range
        normalized = (volume_clipped - p1) / (p99 - p1 + 1e-8) * (out_max - out_min) + out_min
        
        params = {
            'mode': 'adaptive_histogram',
            'p1': float(p1),
            'p99': float(p99),
            'histogram_preserved': True
        }
        return normalized, params
    
    def _robust_zscore(self, volume: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Robust Z-score using median and IQR."""
        median = np.median(volume)
        q25, q75 = np.percentile(volume, [25, 75])
        iqr = q75 - q25 + 1e-8
        
        # Robust standardization
        normalized = (volume - median) / (iqr * 0.7413)  # 0.7413 for normal distribution
        
        # Clip and scale
        normalized = np.clip(normalized, -3, 3)
        out_min, out_max = self.output_range
        normalized = (normalized / 3.0) * ((out_max - out_min) / 2) + (out_max + out_min) / 2
        
        params = {'mode': 'robust_zscore', 'median': float(median), 'iqr': float(iqr)}
        return normalized, params


# ============================================================================
# VOLUME CACHE (Enhanced)
# ============================================================================

class VolumeCache:
    """LRU cache for NIfTI volumes with memory monitoring."""
    
    def __init__(self, max_cache_size: int = 8, max_memory_gb: float = None):
        self.max_cache_size = max_cache_size
        self.max_memory_gb = max_memory_gb or (psutil.virtual_memory().total / 1024**3 * 0.3)
        self.cache = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        self.memory_usage = 0
        
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
        volume_size = volume.nbytes / 1024**3
        
        # Memory-aware caching
        while (self.memory_usage + volume_size > self.max_memory_gb and 
               len(self.cache) > 0):
            _, evicted = self.cache.popitem(last=False)
            self.memory_usage -= evicted.nbytes / 1024**3
        
        self.cache[path_str] = volume
        self.memory_usage += volume_size
        self.cache.move_to_end(path_str)
        
        if len(self.cache) > self.max_cache_size:
            _, evicted = self.cache.popitem(last=False)
            self.memory_usage -= evicted.nbytes / 1024**3
        
        return volume
    
    def get_volume_shape(self, path: Path) -> Tuple[int, int, int]:
        nii = nib.load(path)
        shape = nii.shape
        return (shape[2], shape[1], shape[0])
    
    def clear(self):
        self.cache.clear()
        self.memory_usage = 0
        gc.collect()
    
    def get_stats(self) -> Dict:
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            'cache_size': len(self.cache),
            'memory_usage_gb': self.memory_usage,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate
        }


# ============================================================================
# MEMORY MANAGER
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
                             f"Disk: {mem['disk_percent']:.1f}%")
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
# OPTIMIZED DATASET WITH NORMALIZATION CONTROL
# ============================================================================

class CTPhaseDataset(Dataset):
    """
    Memory-efficient dataset with configurable normalization.
    
    Supports:
    - Discrete (0-255) and continuous (HU) value modes
    - Multiple normalization strategies
    - Mask-based weighting
    - Smart volume caching
    """
    
    def __init__(
        self,
        data_pairs: List[Dict],
        patch_size: Tuple[int, int] = (128, 192),
        patch_depth: int = 15,
        overlap_ratio: float = 0.5,
        augment: bool = True,
        body_focused: bool = True,
        body_threshold: float = -500.0,
        min_fg_ratio: float = 0.6487,
        max_same_value_ratio: float = 0.80,
        cache_size: int = 8,
        use_memmap: bool = True,
        min_intensity_ratio: float = 0.7205,
        min_mean: float = -0.2886,
        min_std: float = 0.2135,
        validate_patches: bool = False,
        # NEW: Normalization control
        value_mode: str = 'discrete',  # 'discrete' or 'continuous'
        normalization: Union[str, NormalizationMode] = 'discrete_minmax',
        global_stats: Dict = None,  # Pre-computed dataset statistics
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
        
        # Value mode control
        self.value_mode = value_mode
        self.is_discrete = (value_mode == 'discrete')
        
        # Body threshold adjustment based on value mode
        if self.is_discrete:
            # For 0-255 discrete values
            self.body_threshold = 50  # Approximate equivalent
        else:
            self.body_threshold = body_threshold  # Original HU threshold
        
        # Initialize normalizer
        if isinstance(normalization, str):
            # Auto-select normalization based on value mode
            if value_mode == 'discrete' and normalization in ['auto', 'default']:
                normalization = NormalizationMode.DISCRETE_MINMAX
            elif value_mode == 'continuous' and normalization in ['auto', 'default']:
                normalization = NormalizationMode.CONTINUOUS_HU
        
        self.normalizer = CTNormalizer(
            mode=normalization,
            global_stats=global_stats
        )
        
        # Volume cache
        self.volume_cache = VolumeCache(max_cache_size=cache_size)
        self.use_memmap = use_memmap
        
        # Validation parameters
        self.min_intensity_ratio = min_intensity_ratio
        self.min_mean = min_mean
        self.min_std = min_std
        self.validate_patches = validate_patches
        
        # Phase encoding
        self.phase_to_idx = {
            'non-contrast': 0,
            'arterial': 1,
            'portal': 2,
            'venous': 2,
            'delayed': 3
        }
        
        logger.info(f"Dataset: {len(data_pairs)} pairs, patch {patch_size}x{patch_depth}")
        logger.info(f"Value mode: {value_mode}, Normalization: {self.normalizer.mode.value}")
        logger.info(f"Volume cache: max {cache_size} volumes")
        
        self._compute_patch_coordinates()
    
    def _get_body_mask(self, volume: np.ndarray) -> np.ndarray:
        """Get body mask based on value mode."""
        if self.is_discrete:
            # For 0-255 discrete values
            return volume > self.body_threshold
        else:
            # For HU values
            return volume > self.body_threshold
    
    def _same_value_ratio(self, patch: np.ndarray) -> float:
        flat = patch.ravel()
        if flat.size == 0:
            return 1.0
        values, counts = np.unique(flat, return_counts=True)
        return float(counts.max() / flat.size)

    def _fg_ratio(self, patch: np.ndarray) -> float:
        return (patch > self.body_threshold).mean()

    def _normalize_patch(self, patch: np.ndarray) -> np.ndarray:
        """Normalize patch using configured normalizer."""
        return self.normalizer.normalize(patch)

    def is_valid_patch(self, source_raw: np.ndarray, target_raw: np.ndarray) -> Tuple[bool, str]:
        # Use configured normalizer
        source_norm = self._normalize_patch(source_raw)
        target_norm = self._normalize_patch(target_raw)

        # Adjust thresholds based on value mode
        if self.is_discrete:
            dark_threshold = -0.7  # Adjusted for 0-255 scaled data
        else:
            dark_threshold = -0.9

        dark_ratio_s = (source_norm < dark_threshold).mean()
        dark_ratio_t = (target_norm < dark_threshold).mean()
        if dark_ratio_s > (1 - self.min_intensity_ratio) or dark_ratio_t > (1 - self.min_intensity_ratio):
            return False, f"dark_s={dark_ratio_s:.3f}, dark_t={dark_ratio_t:.3f}"

        mean_s = source_norm.mean()
        mean_t = target_norm.mean()
        if mean_s < self.min_mean or mean_t < self.min_mean:
            return False, f"mean_s={mean_s:.3f}, mean_t={mean_t:.3f}"

        std_s = source_norm.std()
        std_t = target_norm.std()
        if std_s < self.min_std or std_t < self.min_std:
            return False, f"std_s={std_s:.3f}, std_t={std_t:.3f}"

        fg_s = self._fg_ratio(source_raw)
        fg_t = self._fg_ratio(target_raw)
        if fg_s < self.min_fg_ratio or fg_t < self.min_fg_ratio:
            return False, f"fg_s={fg_s:.3f}, fg_t={fg_t:.3f}"

        same_s = self._same_value_ratio(source_raw)
        same_t = self._same_value_ratio(target_raw)
        if same_s > self.max_same_value_ratio or same_t > self.max_same_value_ratio:
            return False, f"uniform_s={same_s:.3f}, uniform_t={same_t:.3f}"

        return True, ""

    def _find_body_center(self, volume: np.ndarray) -> Tuple[int, int]:
        mid_start = volume.shape[0] // 3
        mid_end = 2 * volume.shape[0] // 3
        mid_slices = volume[mid_start:mid_end]
        body_mask = self._get_body_mask(mid_slices)
        
        if body_mask.sum() > 0:
            coords = np.argwhere(body_mask)
            return int(np.median(coords[:, 1])), int(np.median(coords[:, 2]))
        return volume.shape[1] // 2, volume.shape[2] // 2
    
    def _compute_patch_coordinates(self):
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
        
        logger.info(f"Generated {len(self.patch_coords)} patches")
        
        if len(self.patch_coords) == 0:
            logger.error("⚠️ NO PATCHES GENERATED!")
    
    def __len__(self) -> int:
        return len(self.patch_coords)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        coord = self.patch_coords[idx]
        pair_data = self.data_pairs[coord['pair_idx']]
        
        source_vol = self.volume_cache.get_volume(
            pair_data['source_path'], 
            use_memmap=self.use_memmap
        )
        target_vol = self.volume_cache.get_volume(
            pair_data['target_path'], 
            use_memmap=self.use_memmap
        )
        
        z, y, x = coord['z'], coord['y'], coord['x']
        half_depth = self.patch_depth // 2
        
        z_start = max(0, z - half_depth)
        z_end = min(source_vol.shape[0], z + half_depth + 1)
        
        source_patch = source_vol[z_start:z_end, y:y+self.patch_size[0], x:x+self.patch_size[1]].copy()
        target_patch = target_vol[z_start:z_end, y:y+self.patch_size[0], x:x+self.patch_size[1]].copy()
        
        # Phase encoding
        source_phase_idx = self.phase_to_idx.get(pair_data['source_phase'], 0)
        target_phase_idx = self.phase_to_idx.get(pair_data['target_phase'], 1)
        
        # Handle depth padding
        if source_patch.shape[0] < self.patch_depth:
            pad_before = (self.patch_depth - source_patch.shape[0]) // 2
            pad_after = self.patch_depth - source_patch.shape[0] - pad_before
            source_patch = np.pad(source_patch, ((pad_before, pad_after), (0, 0), (0, 0)), mode='edge')
            target_patch = np.pad(target_patch, ((pad_before, pad_after), (0, 0), (0, 0)), mode='edge')
        
        # Validate if enabled
        if self.validate_patches:
            is_valid, reason = self.is_valid_patch(source_patch, target_patch)
            if not is_valid:
                return {
                    'skip': True,
                    'reason': reason,
                    'idx': idx,
                    'case_id': pair_data.get('case_id', 'unknown')
                }
        
        # Normalize using configured normalizer (returns params for potential reversal)
        source_patch, source_params = self.normalizer.normalize(source_patch, return_params=True)
        target_patch, target_params = self.normalizer.normalize(target_patch, return_params=True)
        
        # Convert to tensors
        source_tensor = torch.from_numpy(source_patch).unsqueeze(0).float()
        target_tensor = torch.from_numpy(target_patch).unsqueeze(0).float()
        
        # Compute intensity weights for high-value penalty
        # Weight matrix: higher weight for higher intensity values
        if self.is_discrete:
            # For 0-255 values normalized to [-1, 1]
            intensity_weight = (target_tensor + 1) / 2  # Maps to [0, 1]
        else:
            intensity_weight = (target_tensor + 1) / 2
        
        # Mask extraction
        masks = {}
        if 'target_seg' in pair_data and pair_data['target_seg']:
            try:
                seg_path = Path(pair_data['target_seg'])
                if seg_path.exists():
                    seg_vol = self.volume_cache.get_volume(seg_path, use_memmap=self.use_memmap)

                    seg_patch = seg_vol[z_start:z_end, y:y+self.patch_size[0], x:x+self.patch_size[1]].copy()

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
            'intensity_weight': intensity_weight,  # For high-value penalty
            'norm_params': {'source': source_params, 'target': target_params},
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
    """3D U-Net with phase conditioning."""
    
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

        return self.out_conv(d)


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
# LOSS FUNCTIONS WITH MASK WEIGHTING AND HIGH-VALUE PENALTY
# ============================================================================

class FocalFrequencyLoss(nn.Module):
    """Focal Frequency Loss for preserving high-frequency details."""
    
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
    Combined loss with mask weighting and high-value penalty.
    
    Features:
    - MSE with warmup schedule
    - Mask-weighted loss for organ regions
    - High-intensity region penalty
    - Focal frequency loss
    - Cycle consistency loss
    - Adversarial loss with warmup
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.mse = nn.MSELoss(reduction='none')  # Per-pixel for weighting
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
        
        # HIGH-VALUE PENALTY
        self.lambda_high_value = config.get('lambda_high_value', 2.0)
        self.high_value_threshold = config.get('high_value_threshold', 0.5)  # In normalized [-1,1] space
        self.high_value_weight = config.get('high_value_weight', 3.0)
        
        # Focal loss weight
        self.lambda_focal = config.get('lambda_focal', 0.1)
        
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
    
    def intensity_weighted_mse(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        intensity_weight: torch.Tensor = None
    ) -> torch.Tensor:
        """
        MSE loss with intensity-based weighting.
        Higher weight for high-intensity (bright) regions.
        """
        mse_per_pixel = self.mse(pred, target)
        
        if intensity_weight is not None:
            # Apply intensity weighting
            # intensity_weight is in [0, 1] range
            weight = 1.0 + self.high_value_weight * intensity_weight
            weighted_mse = mse_per_pixel * weight
        else:
            # Compute intensity weight from target
            # Map target from [-1, 1] to [0, 1]
            normalized_target = (target + 1) / 2
            weight = 1.0 + self.high_value_weight * normalized_target
            weighted_mse = mse_per_pixel * weight
        
        return weighted_mse.mean()
    
    def high_value_penalty(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        threshold: float = None
    ) -> torch.Tensor:
        """
        Extra penalty for errors in high-intensity regions.
        """
        if threshold is None:
            threshold = self.high_value_threshold
        
        # Create mask for high-value regions
        high_value_mask = (target > threshold).float()
        
        if high_value_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        
        # MSE only in high-value regions
        mse_per_pixel = self.mse(pred, target)
        high_value_mse = (mse_per_pixel * high_value_mask).sum() / (high_value_mask.sum() + 1e-8)
        
        return high_value_mse
    
    def masked_mse_with_intensity(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask_dict: Dict[str, torch.Tensor],
        intensity_weight: torch.Tensor = None,
        organ_weight: float = None,
        dilate_radius: int = 0
    ) -> torch.Tensor:
        """
        Combined mask-weighted and intensity-weighted MSE.
        """
        if organ_weight is None:
            organ_weight = self.organ_weight
        
        # Base MSE with intensity weighting
        loss_global = self.intensity_weighted_mse(pred, target, intensity_weight)
        
        if not mask_dict:
            return loss_global
        
        B, _, D, H, W = pred.shape
        union = torch.zeros((B, 1, D, H, W), dtype=pred.dtype, device=pred.device)
        for m in mask_dict.values():
            union = torch.max(union, m[:, :1, ...])
        
        if dilate_radius > 0:
            k = 2 * dilate_radius + 1
            kernel = torch.ones((1, 1, k, k, k), dtype=union.dtype, device=union.device)
            dilated = F.conv3d(
                F.pad(union, (dilate_radius,) * 6, mode='constant', value=0),
                kernel,
                padding=0
            )
            mask = (dilated > 0).float()
            # Crop back to original size
            mask = mask[:, :, :D, :H, :W]
        else:
            mask = union
        
        if mask.sum() == 0:
            return loss_global
        
        # Organ MSE with intensity weighting
        mse_per_pixel = self.mse(pred, target)
        
        if intensity_weight is not None:
            weight = 1.0 + self.high_value_weight * intensity_weight
            mse_per_pixel = mse_per_pixel * weight
        
        loss_organ = (mse_per_pixel * mask).sum() / (mask.sum() + 1e-8)
        
        return loss_global + (organ_weight - 1.0) * loss_organ
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        cycle_pred: torch.Tensor, 
        source: torch.Tensor,
        adv_loss: float, 
        masks: Dict[str, torch.Tensor],
        intensity_weight: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute combined loss with all components.
        """
        mse_weight = self.get_mse_weight()
        
        # MSE with mask and intensity weighting
        loss_mse_weighted = self.masked_mse_with_intensity(
            pred=pred,
            target=target,
            mask_dict=masks,
            intensity_weight=intensity_weight,
            organ_weight=self.organ_weight,
            dilate_radius=3
        ) * mse_weight
        
        # High-value region penalty
        loss_high_value = self.high_value_penalty(pred, target) * self.lambda_high_value
        
        # Focal frequency loss
        loss_focal = self.focal(pred, target) * self.lambda_focal
        
        # Cycle consistency loss
        loss_cycle = F.mse_loss(cycle_pred, source) * self.lambda_cycle
        
        # Adversarial loss with warmup
        adv_weight = self.get_adv_weight()
        loss_adv = adv_loss * adv_weight
        
        # Total loss
        total_loss = loss_mse_weighted + loss_high_value + loss_focal + loss_cycle + loss_adv
        
        loss_dict = {
            'total': total_loss.item(),
            'mse': loss_mse_weighted.item(),
            'high_value': loss_high_value.item() if isinstance(loss_high_value, torch.Tensor) else loss_high_value,
            'focal': loss_focal.item(),
            'cycle': loss_cycle.item(),
            'adv': loss_adv.item() if isinstance(loss_adv, torch.Tensor) else loss_adv
        }
        
        return total_loss, loss_dict


# ============================================================================
# LOSS TRACKER
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
            'train_gen_focal': [],
            'train_gen_high_value': [],
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
        self.history['train_gen_high_value'].append(train_losses.get('gen_high_value', 0))
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
        ax.plot(epochs, self.history['train_gen_high_value'], label='High-Value', linewidth=2)
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
        ax.set_title('Adversarial Loss Weight')
        ax.grid(True, alpha=0.3)
        
        # Validation loss
        ax = axes[1, 2]
        ax.plot(epochs, self.history['val_loss'], linewidth=2, color='purple')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Validation Loss')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'training_summary_epoch_{current_epoch}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        self.save_json()


# ============================================================================
# SAMPLE SAVING
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
    save_dir = Path(save_dir) / f"epoch_{epoch:04d}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    png_dir = save_dir / 'comparisons'
    nifti_dir = save_dir / 'nifti_volumes'
    png_dir.mkdir(exist_ok=True)
    if save_nifti:
        nifti_dir.mkdir(exist_ok=True)

    # Clean old epoch folders (keep last 10)
    parent_dir = save_dir.parent
    epoch_dirs = sorted(parent_dir.glob('epoch_*'), 
                       key=lambda x: int(x.name.split('_')[1]))
    
    if len(epoch_dirs) > 10:
        import shutil
        for old_dir in epoch_dirs[:-10]:
            try:
                shutil.rmtree(old_dir)
                logger.info(f"Deleted old sample directory: {old_dir.name}")
            except Exception as e:
                logger.warning(f"Could not delete {old_dir}: {e}")
    
    # Collect valid batches
    all_batches = []
    with torch.no_grad():
        for batch in val_loader:
            all_batches.append(batch)
    
    valid_batches = [b for b in all_batches if not b.get('skip_batch', False)]
    total_patches = sum(b['source'].size(0) for b in valid_batches) if valid_batches else 0
    
    if len(valid_batches) == 0:
        logger.warning("No validation batches available!")
        return
    
    num_batches_to_sample = min(num_samples, len(valid_batches))
    
    saved_count = 0
    with torch.no_grad():
        for _ in range(num_batches_to_sample):
            if saved_count >= num_samples:
                break
            
            batch = random.choice(valid_batches)
            
            try:
                real_source = batch['source'].to(device)
                real_target = batch['target'].to(device)
                source_phase = batch['source_phase'].to(device)
                target_phase = batch['target_phase'].to(device)
                case_id = batch['case_id']
                
                generated_target = generator(real_source, target_phase)
                reconstructed_source = generator(generated_target, source_phase)
                
                batch_size = real_source.size(0)
                samples_from_batch = min(batch_size, num_samples - saved_count)
                
                if batch_size > samples_from_batch:
                    sample_indices = np.random.choice(batch_size, size=samples_from_batch, replace=False)
                else:
                    sample_indices = range(batch_size)
                
                for i in sample_indices:
                    mid_slice = real_source.shape[2] // 2
                    
                    source_slice = real_source[i, 0, mid_slice].cpu().numpy()
                    target_slice = real_target[i, 0, mid_slice].cpu().numpy()
                    generated_slice = generated_target[i, 0, mid_slice].cpu().numpy()
                    reconstructed_slice = reconstructed_source[i, 0, mid_slice].cpu().numpy()
                    
                    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                    
                    im0 = axes[0, 0].imshow(source_slice, cmap='gray', vmin=-1, vmax=1)
                    axes[0, 0].set_title(f'Source: {source_phase[i].item()}', fontsize=12, fontweight='bold')
                    axes[0, 0].axis('off')
                    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
                    
                    im1 = axes[0, 1].imshow(generated_slice, cmap='gray', vmin=-1, vmax=1)
                    axes[0, 1].set_title(f'Generated: {target_phase[i].item()}', fontsize=12, fontweight='bold')
                    axes[0, 1].axis('off')
                    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
                    
                    im2 = axes[0, 2].imshow(target_slice, cmap='gray', vmin=-1, vmax=1)
                    axes[0, 2].set_title(f'Ground Truth: {target_phase[i].item()}', fontsize=12, fontweight='bold')
                    axes[0, 2].axis('off')
                    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
                    
                    im3 = axes[1, 0].imshow(reconstructed_slice, cmap='gray', vmin=-1, vmax=1)
                    axes[1, 0].set_title(f'Reconstructed: {source_phase[i].item()}', fontsize=12, fontweight='bold')
                    axes[1, 0].axis('off')
                    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
                    
                    diff_gen = np.abs(generated_slice - target_slice)
                    im4 = axes[1, 1].imshow(diff_gen, cmap='hot', vmin=0, vmax=1)
                    axes[1, 1].set_title('|Generated - GT|', fontsize=12, fontweight='bold')
                    axes[1, 1].axis('off')
                    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
                    
                    diff_cycle = np.abs(reconstructed_slice - source_slice)
                    im5 = axes[1, 2].imshow(diff_cycle, cmap='hot', vmin=0, vmax=1)
                    axes[1, 2].set_title('|Reconstructed - Source|', fontsize=12, fontweight='bold')
                    axes[1, 2].axis('off')
                    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)
                    
                    mse_gen = np.mean((generated_slice - target_slice) ** 2)
                    mse_cycle = np.mean((reconstructed_slice - source_slice) ** 2)
                    
                    fig.suptitle(
                        f'Epoch {epoch} - Case: {case_id[i]} - '
                        f'MSE (Gen): {mse_gen:.4f}, MSE (Cycle): {mse_cycle:.4f}',
                        fontsize=14, fontweight='bold', y=0.98
                    )
                    
                    plt.tight_layout()
                    
                    png_path = png_dir / f'sample_{saved_count:03d}_{case_id[i]}.png'
                    plt.savefig(png_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    if save_nifti:
                        source_vol = real_source[i, 0].cpu().numpy()
                        target_vol = real_target[i, 0].cpu().numpy()
                        generated_vol = generated_target[i, 0].cpu().numpy()
                        reconstructed_vol = reconstructed_source[i, 0].cpu().numpy()
                        
                        case_nifti_dir = nifti_dir / f'sample_{saved_count:03d}_{case_id[i]}'
                        case_nifti_dir.mkdir(exist_ok=True)
                        
                        def save_nifti_volume(volume, filepath):
                            volume_transposed = np.transpose(volume, (2, 1, 0))
                            nifti_img = nib.Nifti1Image(volume_transposed, affine=np.eye(4))
                            nib.save(nifti_img, filepath)
                        
                        save_nifti_volume(source_vol, case_nifti_dir / 'source.nii.gz')
                        save_nifti_volume(target_vol, case_nifti_dir / 'target_groundtruth.nii.gz')
                        save_nifti_volume(generated_vol, case_nifti_dir / 'target_generated.nii.gz')
                        save_nifti_volume(reconstructed_vol, case_nifti_dir / 'source_reconstructed.nii.gz')
                        
                        metadata = {
                            'epoch': epoch,
                            'case_id': case_id[i],
                            'source_phase': int(source_phase[i].item()),
                            'target_phase': int(target_phase[i].item()),
                            'patch_shape': list(source_vol.shape),
                            'mse_generated': float(mse_gen),
                            'mse_cycle': float(mse_cycle)
                        }
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
    
    logger.info(f"✓ Saved {saved_count} sample patches to {save_dir}")


# ============================================================================
# TRAINER
# ============================================================================

def safe_collate(batch):
    """Collate function that filters out skipped samples."""
    valid_samples = [item for item in batch if not item.get('skip', False)]
    if len(valid_samples) == 0:
        return {'skip_batch': True}
    return torch.utils.data.dataloader.default_collate(valid_samples)


class MemoryOptimizedTrainer:
    """
    Optimized trainer with:
    - Volume caching
    - Gradient scheduling
    - Memory management
    - Sample saving after each epoch
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = config['device']
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.samples_dir = self.output_dir / 'samples'
        self.samples_dir.mkdir(exist_ok=True)
        
        self.memory_manager = MemoryManager()
        self.cleanup_frequency = config.get('cleanup_frequency', 10)
        
        # Models
        self.generator = Generator3D().to(self.device)
        self.disc_source = Discriminator3D().to(self.device)
        self.disc_target = Discriminator3D().to(self.device)
        
        # Optimizers
        lr_gen = config.get('learning_rate', 2e-4)
        lr_disc = lr_gen * config.get('disc_lr_multiplier', 1.0)
        
        self.opt_gen = optim.Adam(self.generator.parameters(), lr=lr_gen, betas=(0.5, 0.999))
        self.opt_disc_source = optim.Adam(self.disc_source.parameters(), lr=lr_disc, betas=(0.5, 0.999))
        self.opt_disc_target = optim.Adam(self.disc_target.parameters(), lr=lr_disc, betas=(0.5, 0.999))
        
        # Learning rate schedulers
        self.scheduler_gen = optim.lr_scheduler.CosineAnnealingLR(
            self.opt_gen, T_max=config.get('epochs', 100), eta_min=1e-6
        )
        self.scheduler_disc_source = optim.lr_scheduler.CosineAnnealingLR(
            self.opt_disc_source, T_max=config.get('epochs', 100), eta_min=1e-6
        )
        self.scheduler_disc_target = optim.lr_scheduler.CosineAnnealingLR(
            self.opt_disc_target, T_max=config.get('epochs', 100), eta_min=1e-6
        )
        
        # Loss and tracking
        self.combined_loss = CombinedLoss(config)
        self.loss_tracker = LossTracker(self.output_dir)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        self.use_amp = config.get('use_mixed_precision', True)
        self.scaler_gen = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self.scaler_disc = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        # Label smoothing
        self.real_label_smoothing = config.get('real_label_smoothing', 0.9)
        self.fake_label_smoothing = config.get('fake_label_smoothing', 0.1)
        self.disc_updates_per_gen = config.get('disc_updates_per_gen', 1)
        
        logger.info(f"Trainer initialized on {self.device}")
        logger.info(f"Generator LR: {lr_gen:.2e}, Discriminator LR: {lr_disc:.2e}")

    def train_step(self, batch: Dict) -> Dict:
        if batch.get('skip_batch', False):
            return {
                'gen_total': 0, 'gen_adv': 0, 'gen_cycle': 0,
                'gen_mse': 0, 'gen_focal': 0, 'gen_high_value': 0, 'disc': 0
            }
        
        N = batch['source'].size(0)
        if N == 0:
            return {
                'gen_total': 0, 'gen_adv': 0, 'gen_cycle': 0,
                'gen_mse': 0, 'gen_focal': 0, 'gen_high_value': 0, 'disc': 0
            }

        source = batch['source'].to(self.device)
        target = batch['target'].to(self.device)
        source_phase = batch['source_phase'].to(self.device)
        target_phase = batch['target_phase'].to(self.device)
        masks = batch.get('masks', {})
        intensity_weight = batch.get('intensity_weight', None)
        
        if masks:
            masks = {k: v.to(self.device) for k, v in masks.items()}
        if intensity_weight is not None:
            intensity_weight = intensity_weight.to(self.device)

        # Discriminator training
        loss_disc = torch.tensor(0.0, device=self.device)
        for _ in range(self.disc_updates_per_gen):
            self.opt_disc_source.zero_grad()
            self.opt_disc_target.zero_grad()

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                with torch.no_grad():
                    fake_target = self.generator(source, target_phase)
                    fake_source = self.generator(target, source_phase)

                pred_real_src = self.disc_source(source)
                pred_fake_src = self.disc_source(fake_source.detach())
                real_lbl_src = torch.full_like(pred_real_src, self.real_label_smoothing)
                fake_lbl_src = torch.full_like(pred_fake_src, self.fake_label_smoothing)
                loss_disc_src = (F.binary_cross_entropy_with_logits(pred_real_src, real_lbl_src) +
                                F.binary_cross_entropy_with_logits(pred_fake_src, fake_lbl_src)) * 0.5

                pred_real_tgt = self.disc_target(target)
                pred_fake_tgt = self.disc_target(fake_target.detach())
                real_lbl_tgt = torch.full_like(pred_real_tgt, self.real_label_smoothing)
                fake_lbl_tgt = torch.full_like(pred_fake_tgt, self.fake_label_smoothing)
                loss_disc_tgt = (F.binary_cross_entropy_with_logits(pred_real_tgt, real_lbl_tgt) +
                                F.binary_cross_entropy_with_logits(pred_fake_tgt, fake_lbl_tgt)) * 0.5

                loss_disc = loss_disc_src + loss_disc_tgt

            self.scaler_disc.scale(loss_disc).backward()
            self.scaler_disc.step(self.opt_disc_source)
            self.scaler_disc.step(self.opt_disc_target)
            self.scaler_disc.update()

        # Generator training
        self.opt_gen.zero_grad()

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            fake_target = self.generator(source, target_phase)
            fake_source = self.generator(target, source_phase)
            cycle_source = self.generator(fake_target, source_phase)
            cycle_target = self.generator(fake_source, target_phase)

            pred_fake_tgt = self.disc_target(fake_target)
            pred_fake_src = self.disc_source(fake_source)
            
            adv_tgt = F.binary_cross_entropy_with_logits(
                pred_fake_tgt, torch.ones_like(pred_fake_tgt)
            )
            adv_src = F.binary_cross_entropy_with_logits(
                pred_fake_src, torch.ones_like(pred_fake_src)
            )
            loss_adv = (adv_tgt + adv_src) * 0.5

            loss_gen, loss_dict = self.combined_loss(
                fake_target, target, cycle_source, source, 
                loss_adv, masks, intensity_weight
            )

        self.scaler_gen.scale(loss_gen).backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        self.scaler_gen.step(self.opt_gen)
        self.scaler_gen.update()

        return {
            'gen_total': loss_dict['total'],
            'gen_adv': loss_dict['adv'],
            'gen_cycle': loss_dict['cycle'],
            'gen_mse': loss_dict['mse'],
            'gen_focal': loss_dict['focal'],
            'gen_high_value': loss_dict.get('high_value', 0),
            'disc': loss_disc.item()
        }
    
    def validate(self, val_loader: DataLoader) -> Dict:
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
                
                with autocast(enabled=self.use_amp):
                    generated = self.generator(source, target_phase)
                    loss = F.mse_loss(generated, target)
                
                val_losses.append(loss.item())
                
                generated_np = generated.cpu().numpy()
                target_np = target.cpu().numpy()
                
                for i in range(generated_np.shape[0]):
                    mse = np.mean((generated_np[i] - target_np[i]) ** 2)
                    psnr = 10 * np.log10(4.0 / (mse + 1e-10))
                    psnr_values.append(psnr)
                    ssim = 1 - mse / 4.0
                    ssim_values.append(max(0, min(1, ssim)))
        
        self.generator.train()
        
        return {
            'val_loss': np.mean(val_losses) if val_losses else 0,
            'psnr': np.mean(psnr_values) if psnr_values else 0,
            'ssim': np.mean(ssim_values) if ssim_values else 0
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int, start_epoch: int = 0):
        """Main training loop with sample saving after each epoch."""
        save_samples_interval = self.config.get('save_samples_interval', 1)
        num_samples = self.config.get('num_samples', 3)
        
        logger.info(f"\nStarting training for {epochs} epochs")
        logger.info(f"Batches: train={len(train_loader)}, val={len(val_loader)}")
        
        if hasattr(train_loader.dataset, 'get_cache_stats'):
            cache_stats = train_loader.dataset.get_cache_stats()
            logger.info(f"Volume cache initialized: max size = {cache_stats['cache_size']}")
        
        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch
            self.combined_loss.set_epoch(epoch)
            
            self.memory_manager.check_and_cleanup(force=True, current_epoch=epoch)
            
            self.generator.train()
            self.disc_source.train()
            self.disc_target.train()
            
            epoch_losses = {
                'gen_total': [], 'gen_adv': [], 'gen_cycle': [],
                'gen_mse': [], 'gen_focal': [], 'gen_high_value': [], 'disc': []
            }
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
            
            for batch_idx, batch in enumerate(pbar):
                try:
                    losses = self.train_step(batch)
                    
                    for key in epoch_losses:
                        epoch_losses[key].append(losses[key])
                    
                    if batch_idx % self.cleanup_frequency == 0:
                        self.memory_manager.check_and_cleanup(current_epoch=epoch)
                    
                    if batch_idx % 10 == 0:
                        pbar.set_postfix({
                            'G': f"{losses['gen_total']:.3f}",
                            'D': f"{losses['disc']:.3f}",
                            'MSE': f"{losses['gen_mse']:.3f}",
                            'HV': f"{losses['gen_high_value']:.3f}"
                        })
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        self.memory_manager.emergency_cleanup()
                        continue
                    raise e
            
            # Step schedulers
            self.scheduler_gen.step()
            self.scheduler_disc_source.step()
            self.scheduler_disc_target.step()
            
            train_losses = {k: np.mean(v) if v else 0.0 for k, v in epoch_losses.items()}
            val_metrics = self.validate(val_loader)
            
            if hasattr(train_loader.dataset, 'get_cache_stats'):
                cache_stats = train_loader.dataset.get_cache_stats()
                logger.info(f"  Cache - Hit Rate: {cache_stats['hit_rate']:.2%}")
            
            logger.info(f"\nEpoch {epoch+1}/{epochs} Summary:")
            logger.info(f"  Train - Total: {train_losses['gen_total']:.4f}, "
                       f"MSE: {train_losses['gen_mse']:.4f}, "
                       f"HV: {train_losses['gen_high_value']:.4f}")
            logger.info(f"  Val - Loss: {val_metrics['val_loss']:.4f}, "
                       f"PSNR: {val_metrics['psnr']:.2f} dB")
            
            lr_gen = self.opt_gen.param_groups[0]['lr']
            lr_disc = self.opt_disc_source.param_groups[0]['lr']
            adv_weight = self.combined_loss.get_adv_weight()
            
            self.loss_tracker.update_epoch(
                epoch=epoch,
                train_losses=train_losses,
                val_metrics=val_metrics,
                lr_gen=lr_gen,
                lr_disc=lr_disc,
                adv_weight=adv_weight
            )
            
            if (epoch + 1) % 2 == 0:
                self.loss_tracker.create_summary_report(epoch + 1)
            
            # SAVE SAMPLES AFTER EACH EPOCH
            if (epoch + 1) % save_samples_interval == 0:
                logger.info(f"Saving sample patches for epoch {epoch + 1}...")
                save_sample_patches(
                    self.generator,
                    val_loader,
                    epoch + 1,
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
        
        logger.info("Training completed!")
    
    def save_checkpoint(self, epoch: int, val_metrics: Dict, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'disc_source_state_dict': self.disc_source.state_dict(),
            'disc_target_state_dict': self.disc_target.state_dict(),
            'opt_gen_state_dict': self.opt_gen.state_dict(),
            'opt_disc_source_state_dict': self.opt_disc_source.state_dict(),
            'opt_disc_target_state_dict': self.opt_disc_target.state_dict(),
            'scheduler_gen_state_dict': self.scheduler_gen.state_dict(),
            'scheduler_disc_source_state_dict': self.scheduler_disc_source.state_dict(),
            'scheduler_disc_target_state_dict': self.scheduler_disc_target.state_dict(),
            'scaler_gen_state_dict': self.scaler_gen.state_dict(),
            'scaler_disc_state_dict': self.scaler_disc.state_dict(),
            'val_metrics': val_metrics,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, self.output_dir / f'checkpoint_epoch_{epoch:04d}.pth')
        
        if is_best:
            torch.save(checkpoint, self.output_dir / 'checkpoint_best.pth')
        
        keep_last_n = self.config.get('keep_last_n_checkpoints', 3)
        checkpoints = sorted(self.output_dir.glob('checkpoint_epoch_*.pth'))
        for old in checkpoints[:-keep_last_n]:
            old.unlink()
    
    def load_checkpoint(self, checkpoint_path: Path) -> bool:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.disc_source.load_state_dict(checkpoint['disc_source_state_dict'])
            self.disc_target.load_state_dict(checkpoint['disc_target_state_dict'])
            
            self.opt_gen.load_state_dict(checkpoint['opt_gen_state_dict'])
            self.opt_disc_source.load_state_dict(checkpoint['opt_disc_source_state_dict'])
            self.opt_disc_target.load_state_dict(checkpoint['opt_disc_target_state_dict'])
            
            if 'scheduler_gen_state_dict' in checkpoint:
                self.scheduler_gen.load_state_dict(checkpoint['scheduler_gen_state_dict'])
                self.scheduler_disc_source.load_state_dict(checkpoint['scheduler_disc_source_state_dict'])
                self.scheduler_disc_target.load_state_dict(checkpoint['scheduler_disc_target_state_dict'])
            
            if 'scaler_gen_state_dict' in checkpoint:
                self.scaler_gen.load_state_dict(checkpoint['scaler_gen_state_dict'])
                self.scaler_disc.load_state_dict(checkpoint['scaler_disc_state_dict'])
            
            self.current_epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            
            logger.info(f"✓ Checkpoint loaded - Resuming from epoch {self.current_epoch}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to load checkpoint: {e}")
            return False


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def main():
    """Example usage with discrete 0-255 values."""
    
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'output_dir': './output',
        'learning_rate': 2e-4,
        'disc_lr_multiplier': 1.0,
        'epochs': 100,
        'batch_size': 2,
        
        # Loss weights
        'lambda_cycle': 10.0,
        'lambda_mse_initial': 1.0,
        'lambda_mse_final': 100.0,
        'mse_warmup_epochs': 50,
        'lambda_adv': 1.0,
        'adv_warmup_epochs': 10,
        'lambda_organ': 5.0,
        'organ_weight': 10.0,
        
        # HIGH-VALUE PENALTY (NEW)
        'lambda_high_value': 2.0,      # Weight for high-value penalty
        'high_value_threshold': 0.5,   # Threshold in normalized space
        'high_value_weight': 3.0,      # Extra weight for bright regions
        
        'lambda_focal': 0.1,
        
        # Sample saving
        'save_samples_interval': 1,    # Save after every epoch
        'num_samples': 5,
        'save_nifti': True,
        
        # Value mode
        'value_mode': 'discrete',      # 'discrete' (0-255) or 'continuous' (HU)
        'normalization': 'discrete_minmax',  # Or 'adaptive_histogram' for CLAHE data
        
        # Misc
        'use_mixed_precision': True,
        'cleanup_frequency': 10,
        'keep_last_n_checkpoints': 3,
    }
    
    # Create dataset with discrete value mode
    # train_dataset = CTPhaseDataset(
    #     data_pairs=train_pairs,
    #     patch_size=(128, 192),
    #     patch_depth=15,
    #     value_mode='discrete',           # For 0-255 data
    #     normalization='discrete_minmax',  # Or 'adaptive_histogram'
    #     cache_size=8,
    # )
    
    # Create dataloaders
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=config['batch_size'],
    #     shuffle=True,
    #     num_workers=4,
    #     collate_fn=safe_collate
    # )
    
    # Initialize trainer
    # trainer = MemoryOptimizedTrainer(config)
    
    # Train
    # trainer.train(train_loader, val_loader, epochs=config['epochs'])
    