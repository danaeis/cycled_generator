"""
OPTIMIZED Data Loading for CT Phase Training
==============================================
Features:
- Direct loading of preprocessed 0-255 data (no re-normalization)
- Mask-weighted loss generation (high-intensity region emphasis)
- Volume caching for fast loading
- Memory-mapped file access
- Phase conditioning support
- Automatic similarity analysis with caching
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional, Union
import logging
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from torch.utils.data import Dataset, DataLoader
import re
from tqdm import tqdm
import nibabel as nib
from collections import OrderedDict
import pickle

from training_phase_gen import MemoryOptimizedTrainer


logger = logging.getLogger(__name__)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_phase_mapping(labels_csv_path: str) -> Dict[str, Dict[str, str]]:
    """
    Load phase mapping from CSV file.
    
    Args:
        labels_csv_path: Path to labels CSV
        
    Returns:
        Mapping: {case_uid: {series_uid: phase_label}}
    """
    df = pd.read_csv(labels_csv_path)
    
    phase_mapping = {}
    for _, row in df.iterrows():
        case_uid = row['StudyInstanceUID']
        series_uid = row['SeriesInstanceUID']
        phase_label = row['Label'].lower()
        
        if case_uid not in phase_mapping:
            phase_mapping[case_uid] = {}
        phase_mapping[case_uid][series_uid] = phase_label
    
    return phase_mapping


def infer_phase_from_filename(filename: str) -> str:
    """Infer CT phase from filename."""
    filename = filename.lower()
    
    phase_keywords = {
        'non-contrast': ['noncontrast', 'non-contrast', 'pre', 'baseline', 'native', 'nc'],
        'arterial': ['arterial', 'art', 'early', 'phase1', 'p1'],
        'portal': ['portal', 'venous', 'pv', 'phase2', 'p2', 'late'],
        'delayed': ['delayed', 'delay', 'equilibrium', 'phase3', 'p3']
    }
    
    for phase, keywords in phase_keywords.items():
        if any(keyword in filename for keyword in keywords):
            return phase
    
    return 'unknown'


def safe_collate(batch):
    """Safe collate function that filters out None samples."""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


# ============================================================================
# DATASET CLASS
# ============================================================================

# class CTPhaseDataset(Dataset):
#     """
#     Optimized dataset for CT phase generation with:
#     - Preprocessed data loading (already 0-255)
#     - Mask weight generation for important regions
#     - Volume caching for speed
#     - Phase conditioning support
#     """
    
#     def __init__(
#         self,
#         data_pairs: List[Dict],
#         patch_size: Tuple[int, int] = (64, 64),
#         patch_depth: int = 7,
#         overlap_ratio: float = 0.5,
#         augment: bool = True,
#         cache_size: int = 10,
#         use_memmap: bool = True,
#         min_intensity_ratio: float = 0.1,
#         min_mean: float = 10.0,
#         min_std: float = 5.0,
#         validate_patches: bool = True,
#         # New parameters for preprocessing
#         data_is_preprocessed: bool = True,
#         discrete_range: Tuple[int, int] = (0, 255),
#         use_mask_weighting: bool = True,
#         high_intensity_threshold: float = 150,
#         high_intensity_weight: float = 3.0
#     ):
#         self.data_pairs = data_pairs
#         self.patch_size = patch_size
#         self.patch_depth = patch_depth
#         self.overlap_ratio = overlap_ratio
#         self.augment = augment
#         self.cache_size = cache_size
#         self.use_memmap = use_memmap
#         self.min_intensity_ratio = min_intensity_ratio
#         self.min_mean = min_mean
#         self.min_std = min_std
#         self.validate_patches = validate_patches
        
#         # Preprocessing settings
#         self.data_is_preprocessed = data_is_preprocessed
#         self.discrete_range = discrete_range
#         self.use_mask_weighting = use_mask_weighting
#         self.high_intensity_threshold = high_intensity_threshold
#         self.high_intensity_weight = high_intensity_weight
        
#         self.patch_coords = []
        
#         # Phase mapping
#         self.phase_to_idx = {
#             'non-contrast': 0,
#             'arterial': 1,
#             'portal': 2,
#             'venous': 2,
#             'delayed': 3
#         }
        
#         # Volume cache (LRU)
#         self.volume_cache = OrderedDict()
#         self.cache_hits = 0
#         self.cache_misses = 0
        
#         logger.info(f"Initializing dataset with {len(data_pairs)} pairs")
#         logger.info(f"Patch size: {patch_size}, Depth: {patch_depth}, Overlap: {overlap_ratio}")
#         logger.info(f"Data preprocessed: {data_is_preprocessed}, Range: {discrete_range}")
#         logger.info(f"Mask weighting: {use_mask_weighting}, Threshold: {high_intensity_threshold}")
        
#         self._compute_patch_coordinates()
#         logger.info(f"Generated {len(self.patch_coords)} total patches")
    
#     def _load_volumes_cached(self, pair_idx: int) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
#         """Load volumes with LRU caching, including segmentation."""
#         pair_data = self.data_pairs[pair_idx]
#         cache_key = (pair_data['source_path'], pair_data['target_path'], pair_data.get('target_seg_path'))
        
#         # Check cache
#         if cache_key in self.volume_cache:
#             self.cache_hits += 1
#             self.volume_cache.move_to_end(cache_key)
#             return self.volume_cache[cache_key]
        
#         # Cache miss - load from disk
#         self.cache_misses += 1
        
#         source_nii = nib.load(pair_data['source_path'])
#         target_nii = nib.load(pair_data['target_path'])
        
#         if self.use_memmap:
#             source_vol = np.asarray(source_nii.dataobj)
#             target_vol = np.asarray(target_nii.dataobj)
#         else:
#             source_vol = source_nii.get_fdata()
#             target_vol = target_nii.get_fdata()
        
#         # Load segmentation if available
#         seg_vol = None
#         seg_path = pair_data.get('target_seg_path')
#         if seg_path and Path(seg_path).exists():
#             try:
#                 seg_nii = nib.load(seg_path)
#                 seg_vol = np.asarray(seg_nii.dataobj) if self.use_memmap else seg_nii.get_fdata()
#             except:
#                 seg_vol = None
        
#         # Add to cache
#         self.volume_cache[cache_key] = (source_vol, target_vol, seg_vol)
        
#         # Remove oldest if cache is full
#         if len(self.volume_cache) > self.cache_size:
#             self.volume_cache.popitem(last=False)
        
#         return source_vol, target_vol, seg_vol

#     def _compute_patch_coordinates(self):
#         """Pre-compute patch coordinates for all volumes."""
#         padding = self.patch_depth // 2
        
#         for pair_idx, pair_data in enumerate(self.data_pairs):
#             try:
#                 # Load volumes
#                 source_nii = nib.load(pair_data['source_path'])
#                 source_vol = np.asarray(source_nii.dataobj)
                
#                 # Get dimensions (W, H, D) from nibabel
#                 width, height, depth = source_vol.shape
                
#                 # Transpose to (D, H, W) for processing
#                 source_vol = np.transpose(source_vol, (2, 1, 0))
#                 depth_new, height_new, width_new = source_vol.shape
                
#                 # Check minimum requirements
#                 if depth_new < self.patch_depth + 2:
#                     logger.warning(f"Insufficient depth ({depth_new}) for pair {pair_idx}, skipping")
#                     continue
                
#                 # if height_new < 5 or width_new < 5:
#                 #     logger.warning(f"Insufficient spatial size for pair {pair_idx}, skipping")
#                 #     continue
                
#                 # Calculate step sizes
#                 step_y = max(1, int(self.patch_size[0] * (1 - self.overlap_ratio)))
#                 step_x = max(1, int(self.patch_size[1] * (1 - self.overlap_ratio)))
                
#                 # Generate coordinates
#                 # y_coords = list(range(0, height_new - self.patch_size[0] + 1, step_y))
#                 if height_new >= self.patch_size[0]:
#                     y_coords = list(range(0, height_new - self.patch_size[0] + 1, step_y))
#                 else:
#                     y_coords = [0]  # Still generate patch, will be padded later
#                 # x_coords = list(range(0, width_new - self.patch_size[1] + 1, step_x))
#                 if height_new >= self.patch_size[0]:
#                     x_coords = list(range(0, height_new - self.patch_size[0] + 1, step_y))
#                 else:
#                     x_coords = [0]  # Still generate patch, will be padded later
#                 z_range = range(padding, depth_new - padding)
                
#                 # Store patch coordinates
#                 for center_z in z_range:
#                     for y_start in y_coords:
#                         for x_start in x_coords:
#                             self.patch_coords.append((pair_idx, center_z, y_start, x_start))
                
#             except Exception as e:
#                 logger.error(f"Error processing pair {pair_idx}: {e}")
#                 continue
    
#     def __len__(self) -> int:
#         return len(self.patch_coords)
    
#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         """
#         Get a training sample with mask weighting.
#         Data is already preprocessed to 0-255 range.
#         """
#         pair_idx, center_z, y_start, x_start = self.patch_coords[idx]
#         pair_data = self.data_pairs[pair_idx]
        
#         # Load volumes (with caching)
#         source_vol, target_vol, seg_vol = self._load_volumes_cached(pair_idx)
        
#         # Transpose to (D, H, W)
#         source_vol = np.transpose(source_vol, (2, 1, 0))
#         target_vol = np.transpose(target_vol, (2, 1, 0))
        
#         if seg_vol is not None:
#             seg_vol = np.transpose(seg_vol, (2, 1, 0))

#         # Calculate patch coordinates
#         padding = self.patch_depth // 2
#         z_start = center_z - padding
#         z_end = center_z + padding + 1
        
#         # Extract patches
#         source_patch = source_vol[z_start:z_end, 
#                                 y_start:y_start + self.patch_size[0],
#                                 x_start:x_start + self.patch_size[1]]
        
#         target_patch = target_vol[z_start:z_end,
#                                 y_start:y_start + self.patch_size[0],
#                                 x_start:x_start + self.patch_size[1]]
        
#         # ========== LOAD SEGMENTATION MASK FIRST (BEFORE PADDING) ==========
#         seg_patch = None

#         if seg_vol is not None:
#             seg_patch = seg_vol[z_start:z_end, y_start:y_end, x_start:x_end]
#         else:
#             seg_patch = None

#         if self.use_mask_weighting:
#             seg_path = pair_data.get('target_seg_path', None)
            
#             if seg_path and Path(seg_path).exists():
#                 try:
#                     # Load segmentation mask
#                     seg_nii = nib.load(seg_path)
#                     seg_vol = np.asarray(seg_nii.dataobj)
#                     seg_vol = np.transpose(seg_vol, (2, 1, 0))  # (D, H, W)
                    
#                     # Extract corresponding patch
#                     seg_patch = seg_vol[z_start:z_end,
#                                     y_start:y_start + self.patch_size[0],
#                                     x_start:x_start + self.patch_size[1]]
#                 except Exception as e:
#                     logger.debug(f"Failed to load seg mask: {e}")
#                     seg_patch = None
        
#         # ========== AUTO-RESIZE/PAD IF PATCH IS SMALLER THAN EXPECTED ==========
#         expected_shape = (self.patch_depth, self.patch_size[0], self.patch_size[1])
        
#         if source_patch.shape != expected_shape:
#             # Calculate padding needed
#             pad_d = max(0, expected_shape[0] - source_patch.shape[0])
#             pad_h = max(0, expected_shape[1] - source_patch.shape[1])
#             pad_w = max(0, expected_shape[2] - source_patch.shape[2])
            
#             # Pad source and target with edge values
#             source_patch = np.pad(
#                 source_patch,
#                 ((0, pad_d), (0, pad_h), (0, pad_w)),
#                 mode='edge'
#             )
            
#             target_patch = np.pad(
#                 target_patch,
#                 ((0, pad_d), (0, pad_h), (0, pad_w)),
#                 mode='edge'
#             )
            
#             # Pad segmentation with zeros if it exists
#             if seg_patch is not None:
#                 seg_patch = np.pad(
#                     seg_patch,
#                     ((0, pad_d), (0, pad_h), (0, pad_w)),
#                     mode='constant',
#                     constant_values=0
#                 )
        
#         # ========== GENERATE WEIGHT MASK FROM SEGMENTATION OR INTENSITY ==========
#         if self.use_mask_weighting:
#             weight_mask = np.ones_like(target_patch, dtype=np.float32)
            
#             if seg_patch is not None:
#                 # Use segmentation: organs/vessels (seg > 0) get higher weight
#                 weight_mask[seg_patch > 0] = self.high_intensity_weight
#             else:
#                 # Fallback to intensity-based weights
#                 high_intensity_mask = target_patch > self.high_intensity_threshold
#                 weight_mask[high_intensity_mask] = self.high_intensity_weight
#         else:
#             weight_mask = np.ones_like(target_patch, dtype=np.float32)
        
#         # ========== GET PHASE LABELS ==========
#         source_phase_idx = self.phase_to_idx.get(pair_data.get('source_phase', 'unknown'), 0)
#         target_phase_idx = self.phase_to_idx.get(pair_data.get('target_phase', 'unknown'), 0)
        
#         # ========== CONVERT TO TENSORS ==========
#         # Normalize to [0, 1] for network
#         source_tensor = torch.from_numpy(source_patch).float().unsqueeze(0) / self.discrete_range[1]
#         target_tensor = torch.from_numpy(target_patch).float().unsqueeze(0) / self.discrete_range[1]
#         weight_tensor = torch.from_numpy(weight_mask).float().unsqueeze(0)
        
#         return {
#             'source': source_tensor,
#             'target': target_tensor,
#             'weight_mask': weight_tensor,
#             'source_phase': torch.tensor(source_phase_idx, dtype=torch.long),
#             'target_phase': torch.tensor(target_phase_idx, dtype=torch.long),
#             'case_id': pair_data['case_id']
#         }

#     def get_cache_stats(self) -> Dict:
#         """Get cache performance statistics."""
#         total_accesses = self.cache_hits + self.cache_misses
#         hit_rate = self.cache_hits / total_accesses if total_accesses > 0 else 0
        
#         return {
#             'cache_hits': self.cache_hits,
#             'cache_misses': self.cache_misses,
#             'hit_rate': hit_rate,
#             'cache_size': len(self.volume_cache)
#         }

class CTPhaseDataset(Dataset):
    """
    Optimized dataset for CT phase generation with:
    - Preprocessed data loading (already 0-255)
    - Mask weight generation for important regions
    - Volume caching for speed
    - Phase conditioning support
    - Proper handling of volumes smaller than patch_size
    """
    
    def __init__(
        self,
        data_pairs: List[Dict],
        patch_size: Tuple[int, int] = (64, 64),
        patch_depth: int = 7,
        overlap_ratio: float = 0.5,
        augment: bool = True,
        cache_size: int = 10,
        use_memmap: bool = True,
        min_intensity_ratio: float = 0.1,
        min_mean: float = 10.0,
        min_std: float = 5.0,
        validate_patches: bool = True,
        # New parameters for preprocessing
        data_is_preprocessed: bool = True,
        discrete_range: Tuple[int, int] = (0, 255),
        use_mask_weighting: bool = True,
        high_intensity_threshold: float = 150,
        high_intensity_weight: float = 3.0,
        mask_labels: Optional[Union[int, List[int]]] = None,  # Which labels to weight
        mask_weights: Optional[Union[float, List[float]]] = None  # Custom weights per label

    ):
        self.data_pairs = data_pairs
        self.patch_size = patch_size
        self.patch_depth = patch_depth
        self.overlap_ratio = overlap_ratio
        self.augment = augment
        self.cache_size = cache_size
        self.use_memmap = use_memmap
        self.min_intensity_ratio = min_intensity_ratio
        self.min_mean = min_mean
        self.min_std = min_std
        self.validate_patches = validate_patches
        
        # Preprocessing settings
        self.data_is_preprocessed = data_is_preprocessed
        self.discrete_range = discrete_range
        self.use_mask_weighting = use_mask_weighting
        self.high_intensity_threshold = high_intensity_threshold
        self.high_intensity_weight = high_intensity_weight
        
        self.patch_coords = []
        
        # ========== NEW: Mask label configuration ==========
        # Configure which mask labels to use for weighting
        if mask_labels is None:
            # Default: use all non-zero labels
            self.mask_labels = None
        elif isinstance(mask_labels, int):
            # Single label
            self.mask_labels = [mask_labels]
        else:
            # Multiple labels
            self.mask_labels = list(mask_labels)
        
        # Configure custom weights per label
        if mask_weights is None:
            # Default: use high_intensity_weight for all selected labels
            self.mask_weights = None
        elif isinstance(mask_weights, (int, float)):
            # Single weight for all labels
            self.mask_weights = {label: float(mask_weights) for label in (self.mask_labels or [])}
        else:
            # Custom weight per label
            if self.mask_labels and len(mask_weights) == len(self.mask_labels):
                self.mask_weights = {label: float(weight) for label, weight in zip(self.mask_labels, mask_weights)}
            else:
                raise ValueError("mask_weights must match length of mask_labels")
        
        logger.info(f"Mask label weighting: {self.mask_labels}")
        logger.info(f"Mask weights: {self.mask_weights if self.mask_weights else self.high_intensity_weight}")


        # Phase mapping
        self.phase_to_idx = {
            'non-contrast': 0,
            'arterial': 1,
            'portal': 2,
            'venous': 2,
            'delayed': 3
        }
        
        # Volume cache (LRU)
        self.volume_cache = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info(f"Initializing dataset with {len(data_pairs)} pairs")
        logger.info(f"Patch size: {patch_size}, Depth: {patch_depth}, Overlap: {overlap_ratio}")
        logger.info(f"Data preprocessed: {data_is_preprocessed}, Range: {discrete_range}")
        logger.info(f"Mask weighting: {use_mask_weighting}, Threshold: {high_intensity_threshold}")
        
        self._compute_patch_coordinates()
        logger.info(f"Generated {len(self.patch_coords)} total patches")
    
    def _load_volumes_cached(self, pair_idx: int) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Load volumes with LRU caching, including segmentation."""
        pair_data = self.data_pairs[pair_idx]
        cache_key = (pair_data['source_path'], pair_data['target_path'], pair_data.get('target_seg_path'))
        
        # Check cache
        if cache_key in self.volume_cache:
            self.cache_hits += 1
            self.volume_cache.move_to_end(cache_key)
            return self.volume_cache[cache_key]
        
        # Cache miss - load from disk
        self.cache_misses += 1
        
        source_nii = nib.load(pair_data['source_path'])
        target_nii = nib.load(pair_data['target_path'])
        
        if self.use_memmap:
            source_vol = np.asarray(source_nii.dataobj)
            target_vol = np.asarray(target_nii.dataobj)
        else:
            source_vol = source_nii.get_fdata()
            target_vol = target_nii.get_fdata()
        
        # Load segmentation if available
        seg_vol = None
        seg_path = pair_data.get('target_seg_path')
        if seg_path and Path(seg_path).exists():
            try:
                seg_nii = nib.load(seg_path)
                seg_vol = np.asarray(seg_nii.dataobj) if self.use_memmap else seg_nii.get_fdata()
            except:
                seg_vol = None
        
        # Add to cache
        self.volume_cache[cache_key] = (source_vol, target_vol, seg_vol)
        
        # Remove oldest if cache is full
        if len(self.volume_cache) > self.cache_size:
            self.volume_cache.popitem(last=False)
        
        return source_vol, target_vol, seg_vol

    def _compute_patch_coordinates(self):
        """
        Pre-compute patch coordinates for all volumes.
        
        FIXED: Now handles volumes smaller than patch_size by:
        1. Generating at least one patch per volume (centered)
        2. Allowing coordinates that will require padding
        """
        padding = self.patch_depth // 2
        skipped_volumes = 0
        
        for pair_idx, pair_data in enumerate(self.data_pairs):
            try:
                # Load volumes
                source_nii = nib.load(pair_data['source_path'])
                source_vol = np.asarray(source_nii.dataobj)
                
                # Get dimensions (W, H, D) from nibabel
                width, height, depth = source_vol.shape
                
                # Transpose to (D, H, W) for processing
                source_vol = np.transpose(source_vol, (2, 1, 0))
                depth_new, height_new, width_new = source_vol.shape
                
                # Check minimum depth requirement
                if depth_new < self.patch_depth + 2:
                    logger.warning(f"Insufficient depth ({depth_new}) for pair {pair_idx}, skipping")
                    skipped_volumes += 1
                    continue
                
                # Calculate step sizes
                step_y = max(1, int(self.patch_size[0] * (1 - self.overlap_ratio)))
                step_x = max(1, int(self.patch_size[1] * (1 - self.overlap_ratio)))
                
                # ========== FIXED: Handle volumes smaller than patch_size ==========
                
                # For Y (height) coordinates
                if height_new >= self.patch_size[0]:
                    # Normal case: volume is large enough
                    y_coords = list(range(0, height_new - self.patch_size[0] + 1, step_y))
                    # Make sure we include the last possible patch
                    last_y = height_new - self.patch_size[0]
                    if last_y not in y_coords and last_y >= 0:
                        y_coords.append(last_y)
                else:
                    # Volume smaller than patch: start at 0, will be padded in __getitem__
                    y_coords = [0]
                
                # For X (width) coordinates
                if width_new >= self.patch_size[1]:
                    # Normal case: volume is large enough
                    x_coords = list(range(0, width_new - self.patch_size[1] + 1, step_x))
                    # Make sure we include the last possible patch
                    last_x = width_new - self.patch_size[1]
                    if last_x not in x_coords and last_x >= 0:
                        x_coords.append(last_x)
                else:
                    # Volume smaller than patch: start at 0, will be padded in __getitem__
                    x_coords = [0]
                
                # Z range for slices
                z_range = range(padding, depth_new - padding)
                
                # Store patch coordinates
                patches_added = 0
                for center_z in z_range:
                    for y_start in y_coords:
                        for x_start in x_coords:
                            self.patch_coords.append({
                                'pair_idx': pair_idx,
                                'center_z': center_z,
                                'y_start': y_start,
                                'x_start': x_start,
                                'vol_shape': (depth_new, height_new, width_new)  # Store for debugging
                            })
                            patches_added += 1
                
                if patches_added == 0:
                    logger.warning(f"No patches generated for pair {pair_idx} "
                                 f"(shape: {depth_new}x{height_new}x{width_new})")
                    skipped_volumes += 1
                
            except Exception as e:
                logger.error(f"Error processing pair {pair_idx}: {e}")
                skipped_volumes += 1
                continue
        
        if skipped_volumes > 0:
            logger.warning(f"Skipped {skipped_volumes}/{len(self.data_pairs)} volumes")
    
    def __len__(self) -> int:
        return len(self.patch_coords)
    
    def _pad_or_crop_to_size(self, patch: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Ensure patch has exactly the target shape by padding or cropping.
        
        Args:
            patch: Input patch of shape (D, H, W)
            target_shape: Desired shape (D, H, W)
        
        Returns:
            Patch with exactly target_shape
        """
        current_shape = patch.shape
        result = patch
        
        # Handle each dimension
        pad_width = []
        for i in range(3):
            diff = target_shape[i] - current_shape[i]
            if diff > 0:
                # Need to pad
                pad_width.append((0, diff))
            elif diff < 0:
                # Need to crop
                pad_width.append((0, 0))
            else:
                pad_width.append((0, 0))
        
        # Apply padding if needed
        if any(p[1] > 0 for p in pad_width):
            result = np.pad(result, pad_width, mode='constant', constant_values=0)
             
        # Apply cropping if needed
        slices = []
        for i in range(3):
            if result.shape[i] > target_shape[i]:
                slices.append(slice(0, target_shape[i]))
            else:
                slices.append(slice(None))
        
        result = result[tuple(slices)]
        
        return result
    
    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get a training sample with mask weighting.
        Data is already preprocessed to 0-255 range.
        
        FIXED: Properly handles patches that need padding.
        """
        try:
            coord_info = self.patch_coords[idx]
            pair_idx = coord_info['pair_idx']
            center_z = coord_info['center_z']
            y_start = coord_info['y_start']
            x_start = coord_info['x_start']
            
            pair_data = self.data_pairs[pair_idx]
            
            # Load volumes (with caching)
            source_vol, target_vol, seg_vol = self._load_volumes_cached(pair_idx)
            
            # Transpose to (D, H, W)
            source_vol = np.transpose(source_vol, (2, 1, 0))
            target_vol = np.transpose(target_vol, (2, 1, 0))
            
            if seg_vol is not None:
                seg_vol = np.transpose(seg_vol, (2, 1, 0))
            
            depth, height, width = source_vol.shape

            # Calculate patch coordinates
            padding = self.patch_depth // 2
            z_start = center_z - padding
            z_end = center_z + padding + 1
            
            # Calculate actual end coordinates (clamped to volume bounds)
            y_end = min(y_start + self.patch_size[0], height)
            x_end = min(x_start + self.patch_size[1], width)
            
            # Extract patches (may be smaller than patch_size near edges or for small volumes)
            source_patch = source_vol[z_start:z_end, y_start:y_end, x_start:x_end].copy()
            target_patch = target_vol[z_start:z_end, y_start:y_end, x_start:x_end].copy()
            
            # Extract segmentation patch if available
            seg_patch = None
            if seg_vol is not None:
                seg_patch = seg_vol[z_start:z_end, y_start:y_end, x_start:x_end].copy()
            
            # ========== ENSURE EXACT PATCH SIZE ==========
            expected_shape = (self.patch_depth, self.patch_size[0], self.patch_size[1])
            
            source_patch = self._pad_or_crop_to_size(source_patch, expected_shape)
            target_patch = self._pad_or_crop_to_size(target_patch, expected_shape)
            
            if seg_patch is not None:
                seg_patch = self._pad_or_crop_to_size(seg_patch, expected_shape)
            
            # Verify shape (debugging)
            assert source_patch.shape == expected_shape, \
                f"Source patch shape mismatch: {source_patch.shape} vs {expected_shape}"
            assert target_patch.shape == expected_shape, \
                f"Target patch shape mismatch: {target_patch.shape} vs {expected_shape}"
            
            # ========== GENERATE WEIGHT MASK ==========
            # if self.use_mask_weighting:
            #     weight_mask = np.ones_like(target_patch, dtype=np.float32)
                
            #     if seg_patch is not None:
            #         # Use segmentation: organs/vessels (seg > 0) get higher weight
            #         weight_mask[seg_patch > 0] = self.high_intensity_weight
            #     else:
            #         # Fallback to intensity-based weights
            #         high_intensity_mask = target_patch > self.high_intensity_threshold
            #         weight_mask[high_intensity_mask] = self.high_intensity_weight
            # else:
            #     weight_mask = np.ones_like(target_patch, dtype=np.float32)
            
            # ========== GENERATE WEIGHT MASK FROM SEGMENTATION OR INTENSITY ==========
            if self.use_mask_weighting:
                weight_mask = np.ones_like(target_patch, dtype=np.float32)
                
                if seg_patch is not None:
                    # ========== USE SPECIFIED MASK LABELS ==========
                    if self.mask_labels is None:
                        # Use all non-zero labels
                        if self.mask_weights:
                            # Apply custom weights per label
                            unique_labels = np.unique(seg_patch[seg_patch > 0])
                            for label in unique_labels:
                                label_mask = (seg_patch == label)
                                weight = self.mask_weights.get(label, self.high_intensity_weight)
                                weight_mask[label_mask] = weight
                        else:
                            # Use default weight for all non-zero
                            weight_mask[seg_patch > 0] = self.high_intensity_weight
                    else:
                        # Use only specified labels
                        for label in self.mask_labels:
                            label_mask = (seg_patch == label)
                            
                            if self.mask_weights:
                                # Use custom weight for this label
                                weight = self.mask_weights.get(label, self.high_intensity_weight)
                            else:
                                # Use default weight
                                weight = self.high_intensity_weight
                            
                            weight_mask[label_mask] = weight
                    
                    logger.debug(f"Applied weights to mask labels: {self.mask_labels if self.mask_labels else 'all non-zero'}")
                    
                else:
                    # Fallback to intensity-based weights
                    high_intensity_mask = target_patch > self.high_intensity_threshold
                    weight_mask[high_intensity_mask] = self.high_intensity_weight
            else:
                weight_mask = np.ones_like(target_patch, dtype=np.float32)

            # ========== GET PHASE LABELS ==========
            source_phase_idx = self.phase_to_idx.get(pair_data.get('source_phase', 'unknown'), 0)
            target_phase_idx = self.phase_to_idx.get(pair_data.get('target_phase', 'unknown'), 0)
            
            # ========== CONVERT TO TENSORS ==========
            # Normalize to [0, 1] for network
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
            
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            return None

    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics."""
        total_accesses = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_accesses if total_accesses > 0 else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.volume_cache)
        }
# ============================================================================
# DATA PREPARATION
# ============================================================================

def create_data_pairs(
        data_dir: str,
        tag: str = 'registered',
        phase_mapping: Optional[Dict] = None,
        exclude_pairs: Optional[Dict[str, List[Tuple[str, str, str]]]] = None,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42
    ) -> Dict[str, List[Dict]]:
    """
    Create training/validation/test data pairs.
    
    Args:
        data_dir: Root directory containing case folders
        phase_mapping: Optional mapping from CSV
        exclude_pairs: Pairs to exclude from similarity analysis
        test_size: Proportion for test set
        val_size: Proportion for validation set
        random_state: Random seed
    
    Returns:
        Dictionary with 'train', 'val', 'test' lists of data pairs
    """
    data_dir = Path(data_dir)
    all_pairs = []
    
    logger.info(f"Scanning directory: {data_dir}")
    
    # Scan all case directories
    case_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    logger.info(f"Found {len(case_dirs)} case directories")
    
    for case_dir in tqdm(case_dirs, desc="Processing cases"):
        case_id = case_dir.name
        
        # Find all registered volumes
        volume_files = sorted(case_dir.glob(f'*_{tag}.nii.gz'))
        
        if len(volume_files) < 2:
            continue
        
        # Extract phases for each volume
        volume_phases = []
        for vol_file in volume_files:
            if phase_mapping:
                # Try to get phase from mapping
                series_uid = vol_file.stem.split('_')[1] if '_' in vol_file.stem else None
                phase = phase_mapping.get(case_id, {}).get(series_uid, None)
                if not phase:
                    phase = infer_phase_from_filename(vol_file.name)
            else:
                phase = infer_phase_from_filename(vol_file.name)
            
            volume_phases.append((vol_file, phase))
        
        # Create pairs: non-contrast -> contrast phases
        non_contrast = [v for v in volume_phases if v[1] == 'non-contrast']
        contrast = [v for v in volume_phases if v[1] in ['arterial', 'portal', 'venous', 'delayed']]
        
        for nc_vol, nc_phase in non_contrast:
            for c_vol, c_phase in contrast:
                # Check if pair should be excluded
                if exclude_pairs:
                    should_exclude = False
                    for split_name, exclude_list in exclude_pairs.items():
                        if (case_id, nc_phase, c_phase) in exclude_list:
                            should_exclude = True
                            break
                    
                    if should_exclude:
                        continue
                # Look for corresponding segmentation files
                source_seg = nc_vol.parent / nc_vol.name.replace('.nii.gz', '_seg.nii.gz')
                target_seg = c_vol.parent / c_vol.name.replace('.nii.gz', '_seg.nii.gz')

                all_pairs.append({
                    'case_id': case_id,
                    'source_path': str(nc_vol),
                    'target_path': str(c_vol),
                    'source_phase': nc_phase,
                    'target_phase': c_phase,
                    'source_seg_path': str(source_seg) if source_seg.exists() else None,
                    'target_seg_path': str(target_seg) if target_seg.exists() else None
                })
                
    
    logger.info(f"Created {len(all_pairs)} total pairs")
    
    # Split by case to avoid data leakage
    unique_cases = list(set([p['case_id'] for p in all_pairs]))
    
    # First split: train+val vs test
    train_val_cases, test_cases = train_test_split(
        unique_cases,
        test_size=test_size,
        random_state=random_state
    )
    
    # Second split: train vs val
    train_cases, val_cases = train_test_split(
        train_val_cases,
        test_size=val_size / (1 - test_size),
        random_state=random_state
    )
    
    # Create splits
    splits = {
        'train': [p for p in all_pairs if p['case_id'] in train_cases],
        'val': [p for p in all_pairs if p['case_id'] in val_cases],
        'test': [p for p in all_pairs if p['case_id'] in test_cases]
    }
    
    logger.info(f"Train: {len(splits['train'])} pairs from {len(train_cases)} cases")
    logger.info(f"Val: {len(splits['val'])} pairs from {len(val_cases)} cases")
    logger.info(f"Test: {len(splits['test'])} pairs from {len(test_cases)} cases")
    
    return splits


# ============================================================================
# TRAINING INTEGRATION
# ============================================================================

def run_full_training(config: Optional[Dict] = None):
    """
    Run complete training pipeline with optimized dataloader.
    
    Args:
        config: Training configuration dictionary
    """
    if config is None:
        # Default config
        config = {
            'data_dir': '../../ncct_cect/vindr_ds/registered_output',
            'labels_csv': '../../ncct_cect/vindr_ds/labels.csv',
            'output_dir': '../../ncct_cect/vindr_ds/optimized_train/discrete_masked_conditioned',
            
            'data_is_preprocessed': True,
            'discrete_range': (0, 255),
            'use_mask_weighting': True,
            'high_intensity_threshold': 150,
            'high_intensity_weight': 3.0,
            'use_phase_conditioning': True,
            'phase_embedding_dim': 64,
            
            'patch_size': (128, 192),
            'patch_depth': 1,
            'overlap_ratio': 0.5,
            
            'batch_size': 16,
            'epochs': 50,
            
            'cache_size': 12,
            'use_memmap': True,
            'use_mixed_precision': True,
            
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
    
    print("=" * 80)
    print("OPTIMIZED CT PHASE TRAINING - DATA LOADING")
    print("=" * 80)
    print(f"✓ Data preprocessed: {config.get('data_is_preprocessed', True)}")
    print(f"✓ Discrete range: {config.get('discrete_range', (0, 255))}")
    print(f"✓ Mask weighting: {config.get('use_mask_weighting', True)}")
    print(f"✓ Volume caching: {config['cache_size']} volumes")
    print(f"✓ Memory mapping: {config['use_memmap']}")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(checkpoint_dir)
    # Check for existing checkpoints
    start_epoch = 0
    checkpoint_path = None
    checkpoint_files = list(checkpoint_dir.glob('checkpoint_epoch_*.pth'))
    
    if checkpoint_files:
        epoch_numbers = []
        for ckpt in checkpoint_files:
            match = re.search(r'checkpoint_epoch_(\d+)\.pth', ckpt.name)
            if match:
                epoch_numbers.append(int(match.group(1)))
        
        if epoch_numbers:
            latest_epoch = max(epoch_numbers)
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{latest_epoch}.pth'
            start_epoch = latest_epoch
            print(f"✓ Found checkpoint at epoch {latest_epoch }, resuming from epoch {start_epoch}")
    else:
        print("✓ Starting fresh training")
    
    
    # Load phase mapping
    phase_mapping = None
    if Path(config['labels_csv']).exists():
        try:
            phase_mapping = load_phase_mapping(config['labels_csv'])
            print(f"✓ Loaded phase mapping for {len(phase_mapping)} cases")
        except Exception as e:
            print(f"✗ Could not load phase mapping: {e}")
    
    # Create data pairs
    data_splits = create_data_pairs(config['data_dir'], config['tag'], phase_mapping=phase_mapping)
    
    # Create datasets
    print("\nCreating datasets with volume caching...")
    train_dataset = CTPhaseDataset(
        data_splits['train'],
        patch_size=config['patch_size'],
        patch_depth=config['patch_depth'],
        overlap_ratio=config['overlap_ratio'],
        augment=False,
        cache_size=config['cache_size'],
        use_memmap=config['use_memmap'],
        data_is_preprocessed=config.get('data_is_preprocessed', True),
        discrete_range=config.get('discrete_range', (0, 255)),
        use_mask_weighting=config.get('use_mask_weighting', True),
        high_intensity_threshold=config.get('high_intensity_threshold', 150),
        high_intensity_weight=config.get('high_intensity_weight', 3.0),
        mask_labels=config.get('mask_labels', None),
        mask_weights=config.get('mask_weights', None)
    )
    
    val_dataset = CTPhaseDataset(
        data_splits['val'],
        patch_size=config['patch_size'],
        patch_depth=config['patch_depth'],
        overlap_ratio=0.5,
        augment=False,
        cache_size=config['cache_size'],
        use_memmap=config['use_memmap'],
        validate_patches=False,
        data_is_preprocessed=config.get('data_is_preprocessed', True),
        discrete_range=config.get('discrete_range', (0, 255)),
        use_mask_weighting=config.get('use_mask_weighting', True),
        high_intensity_threshold=config.get('high_intensity_threshold', 150),
        high_intensity_weight=config.get('high_intensity_weight', 3.0),
        mask_labels=config.get('mask_labels', None),
        mask_weights=config.get('mask_weights', None)
    )
    
    print(f"✓ Train dataset: {len(train_dataset)} patches")
    print(f"✓ Val dataset: {len(val_dataset)} patches")
    
    # Create dataloaders
    print("\nCreating DataLoaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        collate_fn=safe_collate
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=safe_collate
    )
    
    print(f"✓ Train loader: {len(train_loader)} batches")
    print(f"✓ Val loader: {len(val_loader)} batches")
    
    # Initialize trainer
    print("\nInitializing trainer...")
    
    trainer = MemoryOptimizedTrainer(config)
    
    # Start training
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    # Load checkpoint if available
    if checkpoint_path and start_epoch > 0:
        try:
            is_loaded = trainer.load_checkpoint(checkpoint_path)
            if not is_loaded:
                raise Exception("Failed to load checkpoint")
            print(f"✓ Successfully loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"✗ Failed to load checkpoint: {e}, starting from scratch")
            start_epoch = 0
            trainer = MemoryOptimizedTrainer(config)
    
    # Step 6: Start training
    print("\nStep 6: Starting OPTIMIZED Training")
    print("=" * 80)

    try:
        trainer.train(train_loader, val_loader, config['epochs'], start_epoch=start_epoch)
        print("\n✓ Training completed successfully!")
        
        # Print cache statistics
        cache_stats = train_dataset.get_cache_stats()
        print("\n" + "=" * 80)
        print("CACHE PERFORMANCE STATISTICS")
        print("=" * 80)
        print(f"Volume Cache Hit Rate: {cache_stats['hit_rate']:.2%}")
        print(f"Total Cache Hits: {cache_stats['cache_hits']}")
        print(f"Total Cache Misses: {cache_stats['cache_misses']}")
        print(f"Disk I/O Reduction: ~{cache_stats['hit_rate'] * 100:.1f}%")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()

from configs import stage1_config, stage2_config, stage3_config_p, stage3_config_2d
if __name__ == "__main__":
    run_full_training(stage3_config_p)