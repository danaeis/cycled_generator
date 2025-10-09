
"""
Clean CT Phase Generation Training Pipeline
============================================
A well-organized, debuggable implementation for generating different contrast phases of CT images.

Architecture:
- Phase-conditional CycleGAN
- Input phase -> Generate target phase (conditional)
- Generated target -> Regenerate input phase (cycle consistency)
- Discriminator on generated target
- Focal loss on organ masks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# DATASET
# ============================================================================

# class CTPhaseDataset(Dataset):
#     """
#     Optimized dataset for CT phase generation with center-focused patching.
    
#     Extracts patches centered around the body region (image center) rather than
#     starting from (0,0), ensuring better coverage of anatomical structures.
#     """
    
#     def __init__(
#         self,
#         data_pairs: List[Dict],
#         patch_size: Tuple[int, int] = (64, 64),
#         patch_depth: int = 7,
#         overlap_ratio: float = 0.5,
#         augment: bool = True,
#         body_focused: bool = False,  # New parameter
#         body_threshold: float = -500.0  # HU threshold for body detection
#     ):
#         self.data_pairs = data_pairs
#         self.patch_size = patch_size
#         self.patch_depth = patch_depth
#         self.overlap_ratio = overlap_ratio
#         self.augment = augment
#         self.body_focused = body_focused
#         self.body_threshold = body_threshold
#         self.patch_coords = []
        
#         # Phase mapping
#         self.phase_to_idx = {
#             'non-contrast': 0,
#             'arterial': 1,
#             'portal': 2,
#             'venous': 2,  # Map venous to portal
#             'delayed': 3
#         }
        
#         logger.info(f"Initializing dataset with {len(data_pairs)} pairs")
#         logger.info(f"Patch size: {patch_size}, Depth: {patch_depth}, Overlap: {overlap_ratio}")
#         logger.info(f"Body-focused patching: {body_focused}")
        
#         self._compute_patch_coordinates()
#         logger.info(f"Generated {len(self.patch_coords)} total patches")
    
#     def _find_body_center(self, volume: np.ndarray) -> Tuple[int, int]:
#         """
#         Find the center of the body region by detecting non-air voxels.
        
#         Args:
#             volume: 3D CT volume [D, H, W]
            
#         Returns:
#             (center_y, center_x): Coordinates of body center
#         """
#         # Take middle slice for body detection
#         mid_slice = volume[volume.shape[2] // 2]
        
#         # # Threshold to find body (HU > -500 for soft tissue)
#         # body_mask = mid_slice > self.body_threshold
        
#         # # Find bounding box of body region
#         # if body_mask.sum() > 0:
#         #     y_indices, x_indices = np.where(body_mask)
            
#         #     # Calculate center of mass of body region
#         #     center_y = int(np.mean(y_indices))
#         #     center_x = int(np.mean(x_indices))
            
#         #     logger.debug(f"Body center detected at: ({center_y}, {center_x})")
#         # else:
#         # Fallback to image center if no body detected
#         center_y = volume.shape[0] // 2
#         center_x = volume.shape[1] // 2
#         logger.debug(f"Using image center: ({center_y}, {center_x})")
        
#         return center_y, center_x
    
#     def _compute_patch_coordinates(self):
#         """
#         Pre-compute centered patch coordinates focusing on body region.
        
#         Patches are generated centered around the body (image center),
#         expanding outward with specified overlap ratio.
#         """
#         padding = self.patch_depth // 2
        
#         for pair_idx, pair_data in enumerate(self.data_pairs):
#             try:
#                 # Load volumes to get dimensions
#                 source_vol = nib.load(pair_data['source_path']).get_fdata()
#                 target_vol = nib.load(pair_data['target_path']).get_fdata()
                
#                 # Validate shapes match
#                 if source_vol.shape != target_vol.shape:
#                     logger.warning(f"Shape mismatch for pair {pair_idx}, skipping")
#                     continue
                
#                 height, width, depth = source_vol.shape
                
#                 # Check minimum requirements
#                 print(f"{depth} & {self.patch_depth}")

#                 if depth < self.patch_depth + 2:
#                     logger.warning(f"Insufficient depth ({depth}) for pair {pair_idx}, skipping")
#                     continue
                
#                 if height < self.patch_size[0] or width < self.patch_size[1]:
#                     logger.warning(f"Insufficient spatial size for pair {pair_idx}, skipping")
#                     continue
                
#                 # Find center of body region
#                 if self.body_focused:
#                     center_y, center_x = self._find_body_center(source_vol)
#                 else:
#                     center_y = height // 2
#                     center_x = width // 2
                
#                 # Calculate step sizes for overlap
#                 step_y = max(1, int(self.patch_size[0] * (1 - self.overlap_ratio)))
#                 step_x = max(1, int(self.patch_size[1] * (1 - self.overlap_ratio)))
                
#                 # Generate Y coordinates centered around body center
#                 y_coords = self._generate_centered_coordinates(
#                     center=center_y,
#                     patch_size=self.patch_size[0],
#                     volume_size=height,
#                     step=step_y
#                 )
                
#                 # Generate X coordinates centered around body center
#                 x_coords = self._generate_centered_coordinates(
#                     center=center_x,
#                     patch_size=self.patch_size[1],
#                     volume_size=width,
#                     step=step_x
#                 )
                
#                 # Generate Z coordinates (all valid slices)
#                 z_range = range(padding, depth - padding)
                
#                 # Store all centered patch coordinates
#                 patch_count = 0
#                 for center_z in z_range:
#                     for y_start in y_coords:
#                         for x_start in x_coords:
#                             self.patch_coords.append((pair_idx, center_z, y_start, x_start))
#                             patch_count += 1
                
#                 logger.info(f"Pair {pair_idx}: Generated {patch_count} centered patches")
#                 logger.info(f"  Body center: ({center_y}, {center_x})")
#                 logger.info(f"  Spatial coverage: {len(y_coords)}(Y) x {len(x_coords)}(X) patches")
                
#             except Exception as e:
#                 logger.error(f"Error processing pair {pair_idx}: {e}")
#                 continue
    
#     def _generate_centered_coordinates(
#         self, 
#         center: int, 
#         patch_size: int, 
#         volume_size: int, 
#         step: int
#     ) -> List[int]:
#         """
#         Generate patch coordinates centered around a point.
        
#         Args:
#             center: Center coordinate of body region
#             patch_size: Size of patch in this dimension
#             volume_size: Total size of volume in this dimension
#             step: Step size between patches
            
#         Returns:
#             List of start coordinates for patches
#         """
#         coords = []
#         half_patch = patch_size // 2
        
#         # Start with center patch
#         center_start = max(0, min(center - half_patch, volume_size - patch_size))
#         coords.append(center_start)
        
#         # Expand symmetrically from center
#         offset = step
#         while True:
#             # Try to add patch above/left
#             coord_before = center_start - offset
#             # Try to add patch below/right
#             coord_after = center_start + offset
            
#             added = False
            
#             # Add before if valid
#             if coord_before >= 0 and coord_before + patch_size <= volume_size:
#                 coords.insert(0, coord_before)
#                 added = True
            
#             # Add after if valid
#             if coord_after >= 0 and coord_after + patch_size <= volume_size:
#                 coords.append(coord_after)
#                 added = True
            
#             # Stop if we can't add any more patches
#             if not added:
#                 break
            
#             offset += step
        
#         return coords
    
#     def _normalize_intensity(self, image: np.ndarray) -> np.ndarray:
#         """Normalize CT intensity values."""
#         # Clip to abdomen HU range
#         image = np.clip(image, -100, 300)
        
#         # Z-score normalization
#         mean_val = np.mean(image)
#         std_val = np.std(image)
#         if std_val > 1e-6:
#             image = (image - mean_val) / std_val
        
#         return image.astype(np.float32)
    
#     def _extract_organ_masks(
#         self, 
#         seg_volume: np.ndarray, 
#         z_start: int, 
#         z_end: int,
#         y_start: int, 
#         y_end: int, 
#         x_start: int, 
#         x_end: int
#     ) -> Dict[str, np.ndarray]:
#         """Extract organ masks from segmentation volume."""
#         patch_seg = seg_volume[z_start:z_end, y_start:y_end, x_start:x_end]
        
#         # Define organ labels (adjust based on your segmentation)
#         organ_labels = {
#             'liver': [1, 2],
#             'kidney_right': [3],
#             'kidney_left': [4],
#             'spleen': [5],
#             'pancreas': [6],
#         }
        
#         masks = {}
#         for organ, labels in organ_labels.items():
#             mask = np.zeros_like(patch_seg, dtype=np.float32)
#             for label in labels:
#                 mask[patch_seg == label] = 1.0
#             masks[organ] = mask
        
#         return masks
    
#     def _extract_organ_masks_from_patch(self, patch_seg: np.ndarray) -> Dict[str, np.ndarray]:
#         """Extract organ masks from a segmentation patch."""
#         # Define organ labels (adjust based on your segmentation)
#         organ_labels = {
#             'liver': [1, 2],
#             'kidney_right': [3],
#             'kidney_left': [4],
#             'spleen': [5],
#             'pancreas': [6],
#         }
        
#         masks = {}
#         for organ, labels in organ_labels.items():
#             mask = np.zeros_like(patch_seg, dtype=np.float32)
#             for label in labels:
#                 mask[patch_seg == label] = 1.0
#             masks[organ] = mask
        
#         return masks
    
#     def _augment(self, source, target, masks):
#         """Apply 3D augmentations."""
#         # Random horizontal flip
#         if np.random.random() > 0.5:
#             source = np.flip(source, axis=2).copy()
#             target = np.flip(target, axis=2).copy()
#             masks = {k: np.flip(v, axis=2).copy() for k, v in masks.items()}
        
#         # Random vertical flip
#         if np.random.random() > 0.5:
#             source = np.flip(source, axis=1).copy()
#             target = np.flip(target, axis=1).copy()
#             masks = {k: np.flip(v, axis=1).copy() for k, v in masks.items()}
        
#         # Random 90-degree rotations (in axial plane)
#         if np.random.random() > 0.5:
#             k = np.random.randint(1, 4)  # 1, 2, or 3 rotations
#             source = np.rot90(source, k=k, axes=(1, 2)).copy()
#             target = np.rot90(target, k=k, axes=(1, 2)).copy()
#             masks = {organ: np.rot90(mask, k=k, axes=(1, 2)).copy() 
#                     for organ, mask in masks.items()}
        
#         return source, target, masks
    
#     def __len__(self) -> int:
#         return len(self.patch_coords)
    
#     def __getitem__(self, idx: int) -> Dict:
#         pair_idx, center_z, y_start, x_start = self.patch_coords[idx]
#         pair_data = self.data_pairs[pair_idx]
        
#         try:
#             # Load volumes
#             source_vol = nib.load(pair_data['source_path']).get_fdata()
#             target_vol = nib.load(pair_data['target_path']).get_fdata()
            
#             # Normalize
#             source_vol = self._normalize_intensity(source_vol)
#             target_vol = self._normalize_intensity(target_vol)
            
#             # Extract centered patch with guaranteed consistent size
#             height, width, depth = source_vol.shape
#             expected_shape = (self.patch_depth, self.patch_size[0], self.patch_size[1])
            
#             # Calculate patch boundaries
#             padding = self.patch_depth // 2
#             z_start = center_z - padding
#             z_end = center_z + padding + 1
#             y_end = y_start + self.patch_size[0]
#             x_end = x_start + self.patch_size[1]
            
#             # Ensure we don't go out of bounds
#             z_start = max(0, z_start)
#             z_end = min(depth, z_end)
#             y_start = max(0, y_start)
#             y_end = min(height, y_end)
#             x_start = max(0, x_start)
#             x_end = min(width, x_end)
            
#             # Extract patches
#             source_patch = source_vol[z_start:z_end, y_start:y_end, x_start:x_end]
#             target_patch = target_vol[z_start:z_end, y_start:y_end, x_start:x_end]
            
#             # Always pad to ensure exact expected shape
#             current_shape = source_patch.shape
            
#             # Calculate padding needed for each dimension
#             pad_z = expected_shape[0] - current_shape[0]
#             pad_y = expected_shape[1] - current_shape[1]
#             pad_x = expected_shape[2] - current_shape[2]
            
#             # Pad symmetrically
#             pad_z_before = pad_z // 2 if pad_z > 0 else 0
#             pad_z_after = pad_z - pad_z_before if pad_z > 0 else 0
#             pad_y_before = pad_y // 2 if pad_y > 0 else 0
#             pad_y_after = pad_y - pad_y_before if pad_y > 0 else 0
#             pad_x_before = pad_x // 2 if pad_x > 0 else 0
#             pad_x_after = pad_x - pad_x_before if pad_x > 0 else 0
            
#             # Apply padding if needed
#             if pad_z > 0 or pad_y > 0 or pad_x > 0:
#                 source_patch = np.pad(source_patch, 
#                                     ((pad_z_before, pad_z_after), 
#                                      (pad_y_before, pad_y_after), 
#                                      (pad_x_before, pad_x_after)), 
#                                     mode='constant', constant_values=0)
#                 target_patch = np.pad(target_patch, 
#                                     ((pad_z_before, pad_z_after), 
#                                      (pad_y_before, pad_y_after), 
#                                      (pad_x_before, pad_x_after)), 
#                                     mode='constant', constant_values=0)
            
#             # Verify final shape is exactly what we expect
#             assert source_patch.shape == expected_shape, f"Expected {expected_shape}, got {source_patch.shape}"
#             assert target_patch.shape == expected_shape, f"Expected {expected_shape}, got {target_patch.shape}"
            
#             # Verify patch is centered correctly
#             # Center of patch should align with body center
#             patch_center_y = y_start + self.patch_size[0] // 2
#             patch_center_x = x_start + self.patch_size[1] // 2
            
#             # Extract organ masks if available
#             masks = {}
#             if pair_data.get('target_seg'):
#                 try:
#                     seg_vol = nib.load(pair_data['target_seg']).get_fdata()
#                     mask_patch = seg_vol[z_start:z_end, y_start:y_end, x_start:x_end]
                    
#                     # Apply same padding to masks if needed
#                     if mask_patch.shape != expected_shape:
#                         mask_patch = np.pad(mask_patch, 
#                                           ((pad_z_before, pad_z_after), 
#                                            (pad_y_before, pad_y_after), 
#                                            (pad_x_before, pad_x_after)), 
#                                           mode='constant', constant_values=0)
                    
#                     masks = self._extract_organ_masks_from_patch(mask_patch)
#                 except Exception as e:
#                     logger.debug(f"Could not load masks: {e}")
            
#             # Augmentation
#             if self.augment:
#                 source_patch, target_patch, masks = self._augment(
#                     source_patch, target_patch, masks
#                 )
            
#             # Convert to tensors
#             source_tensor = torch.from_numpy(source_patch).unsqueeze(0).float()
#             target_tensor = torch.from_numpy(target_patch).unsqueeze(0).float()
            
#             mask_tensors = {
#                 organ: torch.from_numpy(mask).unsqueeze(0).float() 
#                 for organ, mask in masks.items()
#             }
            
#             # Get phase information
#             source_phase = pair_data['source_phase']
#             target_phase = pair_data['target_phase']
            
#             return {
#                 'source': source_tensor,
#                 'target': target_tensor,
#                 'source_phase': source_phase,
#                 'target_phase': target_phase,
#                 'source_phase_idx': self.phase_to_idx.get(source_phase, 0),
#                 'target_phase_idx': self.phase_to_idx.get(target_phase, 1),
#                 'masks': mask_tensors,
#                 'case_id': pair_data['case_id'],
#                 'patch_center': (patch_center_y, patch_center_x)  # For debugging
#             }
            
#         except Exception as e:
#             logger.error(f"Error loading patch {idx}: {e}")
#             # Return dummy data
#             return {
#                 'source': torch.zeros(1, self.patch_depth, *self.patch_size),
#                 'target': torch.zeros(1, self.patch_depth, *self.patch_size),
#                 'source_phase': 'error',
#                 'target_phase': 'error',
#                 'source_phase_idx': 0,
#                 'target_phase_idx': 1,
#                 'masks': {},
#                 'case_id': 'error',
#                 'patch_center': (0, 0)
#             }




class CTPhaseDataset(Dataset):
    """
    Optimized dataset for CT phase generation with center-focused patching.
    
    IMPORTANT: Volumes are in (width, height, depth) format from nibabel.
    We convert to (depth, height, width) for processing.
    """
    
    def __init__(
        self,
        data_pairs: List[Dict],
        patch_size: Tuple[int, int] = (64, 64),
        patch_depth: int = 7,
        overlap_ratio: float = 0.5,
        augment: bool = True,
        body_focused: bool = True,
        body_threshold: float = -500.0
    ):
        self.data_pairs = data_pairs
        self.patch_size = patch_size  # (H, W) for axial patches
        self.patch_depth = patch_depth  # Number of slices
        self.overlap_ratio = overlap_ratio
        self.augment = augment
        self.body_focused = body_focused
        self.body_threshold = body_threshold
        self.patch_coords = []
        
        # Phase mapping
        self.phase_to_idx = {
            'non-contrast': 0,
            'arterial': 1,
            'portal': 2,
            'venous': 2,
            'delayed': 3
        }
        
        logger.info(f"Initializing dataset with {len(data_pairs)} pairs")
        logger.info(f"Patch size: {patch_size}, Depth: {patch_depth}, Overlap: {overlap_ratio}")
        logger.info(f"Body-focused patching: {body_focused}")
        
        self._compute_patch_coordinates()
        logger.info(f"Generated {len(self.patch_coords)} total patches")
    
    def _find_body_center(self, volume: np.ndarray) -> Tuple[int, int]:
        """
        Find the center of the body region.
        
        Args:
            volume: 3D CT volume [D, H, W] (after transposition)
            
        Returns:
            (center_y, center_x): Coordinates of body center in axial plane
        """
        # Use image center as default
        height, width = volume.shape[1], volume.shape[2]
        center_y = height // 2
        center_x = width // 2
        
        logger.debug(f"Using image center: ({center_y}, {center_x})")
        return center_y, center_x
    
    def _compute_patch_coordinates(self):
        """
        Pre-compute centered patch coordinates.
        
        Handles volume shape conversion: (W, H, D) -> (D, H, W)
        """
        padding = self.patch_depth // 2
        
        for pair_idx, pair_data in enumerate(self.data_pairs):
            try:
                # Load volumes - shape is (W, H, D) from nibabel
                source_vol = nib.load(pair_data['source_path']).get_fdata()
                target_vol = nib.load(pair_data['target_path']).get_fdata()
                
                # Validate shapes match
                if source_vol.shape != target_vol.shape:
                    logger.warning(f"Shape mismatch for pair {pair_idx}, skipping")
                    continue
                
                # Get dimensions: nibabel gives (W, H, D)
                width, height, depth = source_vol.shape
                
                logger.info(f"Pair {pair_idx}: Original shape (W, H, D) = {source_vol.shape}")
                
                # Transpose to (D, H, W) for processing
                source_vol = np.transpose(source_vol, (2, 1, 0))  # (W,H,D) -> (D,H,W)
                target_vol = np.transpose(target_vol, (2, 1, 0))
                
                depth_new, height_new, width_new = source_vol.shape
                logger.info(f"  After transpose (D, H, W) = {source_vol.shape}")
                
                # Check minimum requirements
                if depth_new < self.patch_depth + 2:
                    logger.warning(f"Insufficient depth ({depth_new}) for pair {pair_idx}, skipping")
                    continue
                
                if height_new < self.patch_size[0] or width_new < self.patch_size[1]:
                    logger.warning(f"Insufficient spatial size for pair {pair_idx}, skipping")
                    continue
                
                # Find center of body region (in axial plane)
                if self.body_focused:
                    center_y, center_x = self._find_body_center(source_vol)
                else:
                    center_y = height_new // 2
                    center_x = width_new // 2
                
                # Calculate step sizes for overlap
                step_y = max(1, int(self.patch_size[0] * (1 - self.overlap_ratio)))
                step_x = max(1, int(self.patch_size[1] * (1 - self.overlap_ratio)))
                
                # Generate Y coordinates (height dimension)
                y_coords = self._generate_centered_coordinates(
                    center=center_y,
                    patch_size=self.patch_size[0],
                    volume_size=height_new,
                    step=step_y
                )
                
                # Generate X coordinates (width dimension)
                x_coords = self._generate_centered_coordinates(
                    center=center_x,
                    patch_size=self.patch_size[1],
                    volume_size=width_new,
                    step=step_x
                )
                
                # Generate Z coordinates (depth/slice dimension)
                z_range = range(padding, depth_new - padding)
                
                # Store all centered patch coordinates
                patch_count = 0
                for center_z in z_range:
                    for y_start in y_coords:
                        for x_start in x_coords:
                            self.patch_coords.append((pair_idx, center_z, y_start, x_start))
                            patch_count += 1
                
                logger.info(f"  Generated {patch_count} centered patches")
                logger.info(f"  Body center: ({center_y}, {center_x})")
                logger.info(f"  Spatial coverage: {len(y_coords)}(Y) x {len(x_coords)}(X) x {len(z_range)}(Z)")
                
            except Exception as e:
                logger.error(f"Error processing pair {pair_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    def _generate_centered_coordinates(
        self, 
        center: int, 
        patch_size: int, 
        volume_size: int, 
        step: int
    ) -> List[int]:
        """
        Generate patch coordinates centered around a point.
        
        Args:
            center: Center coordinate
            patch_size: Size of patch in this dimension
            volume_size: Total size of volume in this dimension
            step: Step size between patches
            
        Returns:
            List of start coordinates for patches
        """
        coords = []
        half_patch = patch_size // 2
        
        # Start with center patch
        center_start = max(0, min(center - half_patch, volume_size - patch_size))
        coords.append(center_start)
        
        # Expand symmetrically from center
        offset = step
        while True:
            coord_before = center_start - offset
            coord_after = center_start + offset
            
            added = False
            
            if coord_before >= 0 and coord_before + patch_size <= volume_size:
                coords.insert(0, coord_before)
                added = True
            
            if coord_after >= 0 and coord_after + patch_size <= volume_size:
                coords.append(coord_after)
                added = True
            
            if not added:
                break
            
            offset += step
        
        return coords
    
    def _normalize_intensity(self, image: np.ndarray) -> np.ndarray:
        """Normalize CT intensity values."""
        # Clip to abdomen HU range
        image = np.clip(image, -100, 300)
        
        # Z-score normalization
        mean_val = np.mean(image)
        std_val = np.std(image)
        if std_val > 1e-6:
            image = (image - mean_val) / std_val
        
        return image.astype(np.float32)
    
    def _extract_organ_masks(
        self, 
        seg_volume: np.ndarray, 
        z_start: int, 
        z_end: int,
        y_start: int, 
        y_end: int, 
        x_start: int, 
        x_end: int
    ) -> Dict[str, np.ndarray]:
        """
        Extract organ masks from segmentation volume.
        
        Args:
            seg_volume: Segmentation [D, H, W]
            z_start, z_end: Depth range
            y_start, y_end: Height range
            x_start, x_end: Width range
        """
        patch_seg = seg_volume[z_start:z_end, y_start:y_end, x_start:x_end]
        
        organ_labels = {
            'liver': [1, 2],
            'kidney_right': [3],
            'kidney_left': [4],
            'spleen': [5],
            'pancreas': [6],
        }
        
        masks = {}
        for organ, labels in organ_labels.items():
            mask = np.zeros_like(patch_seg, dtype=np.float32)
            for label in labels:
                mask[patch_seg == label] = 1.0
            masks[organ] = mask
        
        return masks
    
    def _augment(self, source, target, masks):
        """
        Apply 3D augmentations.
        
        Args:
            source, target: [D, H, W] arrays
            masks: Dictionary of [D, H, W] mask arrays
        """
        # Random horizontal flip (flip width)
        if np.random.random() > 0.5:
            source = np.flip(source, axis=2).copy()
            target = np.flip(target, axis=2).copy()
            masks = {k: np.flip(v, axis=2).copy() for k, v in masks.items()}
        
        # Random vertical flip (flip height)
        if np.random.random() > 0.5:
            source = np.flip(source, axis=1).copy()
            target = np.flip(target, axis=1).copy()
            masks = {k: np.flip(v, axis=1).copy() for k, v in masks.items()}
        
        # Random 90-degree rotations (in axial plane: H-W)
        if np.random.random() > 0.5:
            k = np.random.randint(1, 4)
            source = np.rot90(source, k=k, axes=(1, 2)).copy()
            target = np.rot90(target, k=k, axes=(1, 2)).copy()
            masks = {organ: np.rot90(mask, k=k, axes=(1, 2)).copy() 
                    for organ, mask in masks.items()}
        
        return source, target, masks
    
    def __len__(self) -> int:
        return len(self.patch_coords)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single patch.
        
        Returns:
            Dictionary with:
                - source: [1, D, H, W] tensor
                - target: [1, D, H, W] tensor
                - masks: {organ: [1, D, H, W] tensor}
                - phase information
        """
        pair_idx, center_z, y_start, x_start = self.patch_coords[idx]
        pair_data = self.data_pairs[pair_idx]
        
        try:
            # Load volumes - shape is (W, H, D) from nibabel
            source_vol = nib.load(pair_data['source_path']).get_fdata()
            target_vol = nib.load(pair_data['target_path']).get_fdata()
            
            # Transpose to (D, H, W) for processing
            source_vol = np.transpose(source_vol, (2, 1, 0))
            target_vol = np.transpose(target_vol, (2, 1, 0))
            
            # Normalize
            source_vol = self._normalize_intensity(source_vol)
            target_vol = self._normalize_intensity(target_vol)
            
            # Extract patch: [D, H, W]
            padding = self.patch_depth // 2
            z_start = center_z - padding
            z_end = center_z + padding + 1
            y_end = y_start + self.patch_size[0]
            x_end = x_start + self.patch_size[1]
            
            # Extract patches
            source_patch = source_vol[z_start:z_end, y_start:y_end, x_start:x_end]
            target_patch = target_vol[z_start:z_end, y_start:y_end, x_start:x_end]
            
            # Verify shape
            expected_shape = (self.patch_depth+1, self.patch_size[0], self.patch_size[1])
            if source_patch.shape != expected_shape:
                logger.warning(f"Patch shape mismatch: expected {expected_shape}, got {source_patch.shape}")
            
            # Calculate patch center for debugging
            patch_center_y = y_start + self.patch_size[0] // 2
            patch_center_x = x_start + self.patch_size[1] // 2
            
            # Extract organ masks if available
            masks = {}
            if pair_data.get('target_seg'):
                try:
                    seg_vol = nib.load(pair_data['target_seg']).get_fdata()
                    # Transpose segmentation too
                    seg_vol = np.transpose(seg_vol, (2, 1, 0))
                    masks = self._extract_organ_masks(
                        seg_vol, z_start, z_end, y_start, y_end, x_start, x_end
                    )
                except Exception as e:
                    logger.debug(f"Could not load masks: {e}")
            
            # Augmentation
            if self.augment:
                source_patch, target_patch, masks = self._augment(
                    source_patch, target_patch, masks
                )
            
            # Convert to tensors: [D, H, W] -> [1, D, H, W]
            source_tensor = torch.from_numpy(source_patch).unsqueeze(0).float()
            target_tensor = torch.from_numpy(target_patch).unsqueeze(0).float()
            
            mask_tensors = {
                organ: torch.from_numpy(mask).unsqueeze(0).float() 
                for organ, mask in masks.items()
            }
            
            # Get phase information
            source_phase = pair_data['source_phase']
            target_phase = pair_data['target_phase']
            
            return {
                'source': source_tensor,
                'target': target_tensor,
                'source_phase': source_phase,
                'target_phase': target_phase,
                'source_phase_idx': self.phase_to_idx.get(source_phase, 0),
                'target_phase_idx': self.phase_to_idx.get(target_phase, 1),
                'masks': mask_tensors,
                'case_id': pair_data['case_id'],
                'patch_center': (patch_center_y, patch_center_x)
            }
            
        except Exception as e:
            logger.error(f"Error loading patch {idx}: {e}")
            import traceback
            traceback.print_exc()
            
            # Return dummy data with correct shape
            return {
                'source': torch.zeros(1, self.patch_depth, *self.patch_size),
                'target': torch.zeros(1, self.patch_depth, *self.patch_size),
                'source_phase': 'error',
                'target_phase': 'error',
                'source_phase_idx': 0,
                'target_phase_idx': 1,
                'masks': {},
                'case_id': 'error',
                'patch_center': (0, 0)
            }
# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class PhaseConditionedGenerator(nn.Module):
    """
    3D U-Net Generator with phase conditioning.
    Takes input phase and generates target phase conditioned on phase label.
    """
    
    def __init__(self, num_phases: int = 4):
        super().__init__()
        
        self.num_phases = num_phases
        
        # Phase embedding
        self.phase_embedding = nn.Embedding(num_phases, 64)
        
        # Encoder
        self.enc1 = self._make_encoder_block(1, 64)      # Input + phase info
        self.enc2 = self._make_encoder_block(64, 128)
        self.enc3 = self._make_encoder_block(128, 256)
        self.enc4 = self._make_encoder_block(256, 512)
        
        self.pool = nn.MaxPool3d((1, 2, 2))  # Pool spatial dims only
        
        # Bottleneck with phase modulation
        self.bottleneck = nn.Sequential(
            nn.Conv3d(512, 1024, 3, padding=1),
            nn.InstanceNorm3d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(1024, 512, 3, padding=1),
            nn.InstanceNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Phase-adaptive modulation
        self.phase_scale_shift = nn.Linear(64, 512 * 2)
        
        # Decoder
        self.up4 = nn.ConvTranspose3d(512, 256, (1, 2, 2), stride=(1, 2, 2))
        self.dec4 = self._make_decoder_block(512 + 256, 256)
        
        self.up3 = nn.ConvTranspose3d(256, 128, (1, 2, 2), stride=(1, 2, 2))
        self.dec3 = self._make_decoder_block(256 + 128, 128)
        
        self.up2 = nn.ConvTranspose3d(128, 64, (1, 2, 2), stride=(1, 2, 2))
        self.dec2 = self._make_decoder_block(128 + 64, 64)
        
        self.up1 = nn.ConvTranspose3d(64, 64, (1, 2, 2), stride=(1, 2, 2))
        self.dec1 = self._make_decoder_block(64 + 64, 64)
        
        # Output
        self.output = nn.Sequential(
            nn.Conv3d(64, 1, 1),
            nn.Tanh()
        )
    
    def _make_encoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def _make_decoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x: torch.Tensor, phase_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image [B, 1, D, H, W]
            phase_idx: Target phase indices [B]
        """
        batch_size = x.size(0)
        
        # Get phase embedding [B, 64]
        phase_emb = self.phase_embedding(phase_idx)
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck with phase modulation
        b = self.bottleneck(self.pool(e4))
        
        # Apply phase-specific modulation
        phase_params = self.phase_scale_shift(phase_emb)
        scale = phase_params[:, :512].view(batch_size, 512, 1, 1, 1)
        shift = phase_params[:, 512:].view(batch_size, 512, 1, 1, 1)
        b = b * (1 + scale) + shift
        
        # Decoder with skip connections - handle size mismatches
        up4_out = self.up4(b)
        # Ensure spatial dimensions match for skip connection
        if up4_out.shape[2:] != e4.shape[2:]:
            up4_out = F.interpolate(up4_out, size=e4.shape[2:], mode='trilinear', align_corners=False)
        d4 = self.dec4(torch.cat([up4_out, e4], dim=1))
        
        up3_out = self.up3(d4)
        if up3_out.shape[2:] != e3.shape[2:]:
            up3_out = F.interpolate(up3_out, size=e3.shape[2:], mode='trilinear', align_corners=False)
        d3 = self.dec3(torch.cat([up3_out, e3], dim=1))
        
        up2_out = self.up2(d3)
        if up2_out.shape[2:] != e2.shape[2:]:
            up2_out = F.interpolate(up2_out, size=e2.shape[2:], mode='trilinear', align_corners=False)
        d2 = self.dec2(torch.cat([up2_out, e2], dim=1))
        
        up1_out = self.up1(d2)
        if up1_out.shape[2:] != e1.shape[2:]:
            up1_out = F.interpolate(up1_out, size=e1.shape[2:], mode='trilinear', align_corners=False)
        d1 = self.dec1(torch.cat([up1_out, e1], dim=1))
        
        return self.output(d1)


class Discriminator3D(nn.Module):
    """3D PatchGAN Discriminator with spectral normalization for stability."""
    
    def __init__(self):
        super().__init__()
        
        # Use smaller kernels and adaptive pooling for robustness
        self.model = nn.Sequential(
            # Layer 1 - Use smaller kernel
            nn.utils.spectral_norm(
                nn.Conv3d(1, 64, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
            ),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2 - Use smaller kernel
            nn.utils.spectral_norm(
                nn.Conv3d(64, 128, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
            ),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3 - Use smaller kernel and stride
            nn.utils.spectral_norm(
                nn.Conv3d(128, 256, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
            ),
            nn.InstanceNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4 - Use smaller kernel
            nn.utils.spectral_norm(
                nn.Conv3d(256, 512, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
            ),
            nn.InstanceNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output - Use adaptive pooling to handle varying sizes
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.utils.spectral_norm(
                nn.Conv3d(512, 1, (1, 1, 1), stride=1, padding=0)
            )
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ============================================================================
# METRICS
# ============================================================================

class MetricsCalculator:
    """Calculate image quality metrics (SSIM, PSNR)."""
    
    @staticmethod
    def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 2.0) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio.
        
        Args:
            pred: Predicted image [-1, 1] range
            target: Target image [-1, 1] range
            data_range: Range of the data (2.0 for [-1, 1])
        """
        mse = torch.mean((pred - target) ** 2)
        if mse == 0:
            return 100.0
        psnr = 20 * torch.log10(data_range / torch.sqrt(mse))
        return psnr.item()
    
    @staticmethod
    def calculate_ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 2.0) -> float:
        """
        Calculate Structural Similarity Index.
        
        Args:
            pred: Predicted image [-1, 1] range [B, C, D, H, W]
            target: Target image [-1, 1] range [B, C, D, H, W]
            data_range: Range of the data
        """
        # Take middle slice for 2D SSIM calculation
        if pred.dim() == 5:  # [B, C, D, H, W]
            pred = pred[:, :, pred.shape[2]//2, :, :]  # [B, C, H, W]
            target = target[:, :, target.shape[2]//2, :, :]
        
        # Constants for SSIM
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        
        # Calculate means
        mu1 = F.avg_pool2d(pred, 3, 1, 1)
        mu2 = F.avg_pool2d(target, 3, 1, 1)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # Calculate variances and covariance
        sigma1_sq = F.avg_pool2d(pred ** 2, 3, 1, 1) - mu1_sq
        sigma2_sq = F.avg_pool2d(target ** 2, 3, 1, 1) - mu2_sq
        sigma12 = F.avg_pool2d(pred * target, 3, 1, 1) - mu1_mu2
        
        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean().item()


def save_sample_patches(
    generator: nn.Module,
    val_loader: DataLoader,
    epoch: int,
    save_dir: Path,
    device: torch.device,
    num_samples: int = 5
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
    
    saved_count = 0
    
    with torch.no_grad():
        for batch in val_loader:
            if saved_count >= num_samples:
                break
            
            try:
                real_source = batch['source'].to(device)
                real_target = batch['target'].to(device)
                source_phase_idx = batch['source_phase_idx'].to(device)
                target_phase_idx = batch['target_phase_idx'].to(device)
                source_phase = batch['source_phase']
                target_phase = batch['target_phase']
                case_id = batch['case_id']
                
                # Generate target and reconstruct source
                generated_target = generator(real_source, target_phase_idx)
                reconstructed_source = generator(generated_target, source_phase_idx)
                
                # Process each sample in batch
                batch_size = real_source.size(0)
                for i in range(min(batch_size, num_samples - saved_count)):
                    # Get middle slice from 3D patch [1, D, H, W] -> [H, W]
                    mid_slice = real_source.shape[1] // 2
                    
                    source_slice = real_source[i, 0, mid_slice].cpu().numpy()
                    target_slice = real_target[i, 0, mid_slice].cpu().numpy()
                    generated_slice = generated_target[i, 0, mid_slice].cpu().numpy()
                    reconstructed_slice = reconstructed_source[i, 0, mid_slice].cpu().numpy()
                    
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
                    
                    # Save figure
                    save_path = save_dir / f'sample_{saved_count:03d}_{case_id[i]}.png'
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    saved_count += 1
                    
                    if saved_count >= num_samples:
                        break
                        
            except Exception as e:
                logger.error(f"Error saving sample patches: {e}")
                continue
    
    logger.info(f"Saved {saved_count} sample patches to {save_dir}")


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class FocalLoss(nn.Module):
    """Focal loss for organ-specific enhancement."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 1.5):  # REDUCED defaults
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
        masks: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted image [B, 1, D, H, W]
            target: Target image [B, 1, D, H, W]
            masks: Organ masks {organ_name: [B, 1, D, H, W]}
        """
        # Base MSE loss
        mse = F.mse_loss(pred, target, reduction='none')
        
        # Create weighted mask emphasizing organs
        weight_mask = torch.ones_like(mse)
        for organ_name, mask in masks.items():
            if mask.sum() > 0:
                # Higher weight for important organs
                organ_weight = 3.0 if organ_name == 'liver' else 2.0
                weight_mask += (organ_weight - 1.0) * mask
        
        # Apply focal weighting
        focal_weight = mse ** self.gamma
        weighted_loss = self.alpha * focal_weight * mse * weight_mask
        
        return weighted_loss.mean()


class CombinedLoss(nn.Module):
    """Combined loss for CycleGAN training with gradual adversarial loss warm-up."""
    
    def __init__(
        self,
        lambda_cycle: float = 10.0,
        lambda_mse: float = 100.0,
        lambda_focal: float = 5.0,
        lambda_adv: float = 1.0,  # Adversarial loss weight
        adv_warmup_epochs: int = 10  # Number of epochs to gradually increase adversarial loss
    ):
        super().__init__()
        self.lambda_cycle = lambda_cycle
        self.lambda_mse = lambda_mse
        self.lambda_focal = lambda_focal
        self.lambda_adv = lambda_adv
        self.adv_warmup_epochs = adv_warmup_epochs
        
        self.mse_loss = nn.MSELoss()
        self.focal_loss = FocalLoss(alpha=1.0, gamma=1.5)
        self.adv_loss = nn.BCEWithLogitsLoss()
        
        self.current_epoch = 0
    
    def set_epoch(self, epoch: int):
        """Update current epoch for warm-up schedule."""
        self.current_epoch = epoch
    
    def get_adv_weight(self) -> float:
        """Calculate adversarial loss weight with warm-up."""
        if self.current_epoch < self.adv_warmup_epochs:
            # Gradual warm-up: 0 -> lambda_adv over warmup_epochs
            return self.lambda_adv * (self.current_epoch / self.adv_warmup_epochs)
        return self.lambda_adv
    
    def forward(
        self,
        real_source: torch.Tensor,
        real_target: torch.Tensor,
        generated_target: torch.Tensor,
        reconstructed_source: torch.Tensor,
        disc_fake_target: torch.Tensor,
        disc_fake_source: torch.Tensor,  # NEW: discriminator on reconstructed source
        masks: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Calculate all generator losses."""
        
        losses = {}
        
        # Get current adversarial weight (with warm-up)
        adv_weight = self.get_adv_weight()
        
        # Adversarial losses (gradually increased)
        losses['adv_target'] = self.adv_loss(disc_fake_target, torch.ones_like(disc_fake_target)) * adv_weight
        losses['adv_source'] = self.adv_loss(disc_fake_source, torch.ones_like(disc_fake_source)) * adv_weight
        losses['adv'] = losses['adv_target'] + losses['adv_source']
        
        # Cycle consistency loss (always active)
        losses['cycle'] = self.mse_loss(reconstructed_source, real_source) * self.lambda_cycle
        
        # Direct MSE loss (always active)
        losses['mse'] = self.mse_loss(generated_target, real_target) * self.lambda_mse
        
        # Focal loss with organ masks (always active)
        if masks:
            losses['focal'] = self.focal_loss(generated_target, real_target, masks) * self.lambda_focal
        else:
            losses['focal'] = torch.tensor(0.0, device=real_source.device)
        
        # Total generator loss
        losses['total'] = losses['adv'] + losses['cycle'] + losses['mse'] + losses['focal']
        
        return losses


# ============================================================================
# TRAINER
# ============================================================================
try:
    from .loss_tracker import LossTracker
except ImportError:
    from loss_tracker import LossTracker
class CTPhaseTrainer:
    """Clean trainer for CT phase generation."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config['device'])
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create samples directory
        self.samples_dir = self.output_dir / 'samples'
        self.samples_dir.mkdir(exist_ok=True)
        
        # Initialize models
        logger.info("Initializing models...")
        self.generator = PhaseConditionedGenerator(num_phases=4).to(self.device)
        self.disc_source = Discriminator3D().to(self.device)
        self.disc_target = Discriminator3D().to(self.device)
        
        self.loss_tracker = LossTracker(
            output_dir=self.output_dir / 'loss_plots',
            save_interval=1  # Save plots every epoch
        )
        logger.info("✓ Loss tracker initialized")

        # Optimizers with discriminator learning rate 2x higher
        disc_lr_multiplier = config.get('disc_lr_multiplier', 2.0)
        
        self.opt_gen = optim.Adam(
            self.generator.parameters(),
            lr=config['learning_rate'],
            betas=(0.5, 0.999)
        )
        self.opt_disc_source = optim.Adam(
            self.disc_source.parameters(),
            lr=config['learning_rate'] * disc_lr_multiplier,
            betas=(0.5, 0.999)
        )
        self.opt_disc_target = optim.Adam(
            self.disc_target.parameters(),
            lr=config['learning_rate'] * disc_lr_multiplier,
            betas=(0.5, 0.999)
        )
        
        # Loss functions
        self.combined_loss = CombinedLoss().to(self.device)
        self.disc_loss = nn.BCEWithLogitsLoss()
        
        # Label smoothing for discriminator stability
        self.real_label_smoothing = config.get('real_label_smoothing', 0.9)  # 0.9 instead of 1.0
        self.fake_label_smoothing = config.get('fake_label_smoothing', 0.1)  # 0.1 instead of 0.0
        
        # How many times to train discriminator per generator update
        self.disc_updates_per_gen = config.get('disc_updates_per_gen', 2)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        # Save config
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Trainer initialized on {self.device}")
        logger.info(f"Generator LR: {config['learning_rate']:.6f}")
        logger.info(f"Discriminator LR: {config['learning_rate'] * disc_lr_multiplier:.6f}")
        logger.info(f"Label smoothing: Real={self.real_label_smoothing}, Fake={self.fake_label_smoothing}")
        logger.info(f"Discriminator updates per generator update: {self.disc_updates_per_gen}")
    
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Single training step."""
        
        # Get data
        real_source = batch['source'].to(self.device)
        real_target = batch['target'].to(self.device)
        source_phase_idx = batch['source_phase_idx'].to(self.device)
        target_phase_idx = batch['target_phase_idx'].to(self.device)
        masks = {k: v.to(self.device) for k, v in batch['masks'].items()}
        # =====================
        # Train Discriminators (multiple times if specified)
        # =====================
        disc_source_losses = []
        disc_target_losses = []
        
        # First, generate fakes (will be detached for disc training)
        with torch.no_grad():
            generated_target = self.generator(real_source, target_phase_idx)
            reconstructed_source = self.generator(generated_target, source_phase_idx)
        
        for _ in range(self.disc_updates_per_gen):
            # Train disc_target (on target domain)
            self.opt_disc_target.zero_grad()
            
            # Real target
            disc_real_target = self.disc_target(real_target)
            loss_real_target = self.disc_loss(
                disc_real_target,
                torch.full_like(disc_real_target, self.real_label_smoothing)
            )
            
            # Fake generated target
            disc_fake_target_det = self.disc_target(generated_target.detach())
            loss_fake_target = self.disc_loss(
                disc_fake_target_det,
                torch.full_like(disc_fake_target_det, self.fake_label_smoothing)
            )
            
            # Total for target disc
            disc_target_loss = (loss_real_target + loss_fake_target) * 0.5
            disc_target_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.disc_target.parameters(), max_norm=1.0)
            self.opt_disc_target.step()
            
            disc_target_losses.append(disc_target_loss.item())
            
            # Train disc_source (on source domain)
            self.opt_disc_source.zero_grad()
            
            # Real source
            disc_real_source = self.disc_source(real_source)
            loss_real_source = self.disc_loss(
                disc_real_source,
                torch.full_like(disc_real_source, self.real_label_smoothing)
            )
            
            # Fake reconstructed source
            disc_fake_source_det = self.disc_source(reconstructed_source.detach())
            loss_fake_source = self.disc_loss(
                disc_fake_source_det,
                torch.full_like(disc_fake_source_det, self.fake_label_smoothing)
            )
            
            # Total for source disc
            disc_source_loss = (loss_real_source + loss_fake_source) * 0.5
            disc_source_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.disc_source.parameters(), max_norm=1.0)
            self.opt_disc_source.step()
            
            disc_source_losses.append(disc_source_loss.item())
        # =====================
        # Train Generator
        # =====================
        self.opt_gen.zero_grad()
        
        # Forward pass
        # Source -> Target (conditioned)
        generated_target = self.generator(real_source, target_phase_idx)
        
        # Generated Target -> Reconstructed Source (cycle)
        reconstructed_source = self.generator(generated_target, source_phase_idx)
        
        # Discriminator output for generated target
        # disc_fake = self.discriminator(generated_target)
        
        disc_fake_target = self.disc_target(generated_target)
        disc_fake_source = self.disc_source(reconstructed_source)

        # Calculate generator losses
        gen_losses = self.combined_loss(
            real_source, real_target, generated_target, 
            reconstructed_source, disc_fake_target, disc_fake_source, masks
        )

        # Backward
        gen_losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)  # Add gradient clipping
        self.opt_gen.step()
        
        
        # Return losses
        losses = {
            'gen_total': gen_losses['total'].item(),
            'gen_adv': gen_losses['adv'].item(),
            'gen_cycle': gen_losses['cycle'].item(),
            'gen_mse': gen_losses['mse'].item(),
            'gen_focal': gen_losses['focal'].item(),
            'disc_source': np.mean(disc_source_losses),
            'disc_target': np.mean(disc_target_losses),
            'disc': np.mean(disc_source_losses + disc_target_losses)
        }

        self.loss_tracker.update_batch(losses)
        return losses
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.generator.train()
        self.disc_source.train()
        self.disc_target.train()
        
        epoch_losses = {
            'gen_total': [], 'gen_adv': [], 'gen_cycle': [],
            'gen_mse': [], 'gen_focal': [], 
            'disc_source': [], 'disc_target': [], 'disc': []
        }
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch in progress_bar:
            try:
                losses = self.train_step(batch)
                
                # Accumulate losses
                for key, value in losses.items():
                    epoch_losses[key].append(value)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Gen': f"{losses['gen_total']:.3f}",
                    'Disc': f"{losses['disc']:.3f}",
                    'Phase': f"{batch['source_phase'][0]}->{batch['target_phase'][0]}"
                })
                
            except Exception as e:
                logger.error(f"Error in training step: {e}")
                continue
        
        # Average losses
        
        avg_losses = {k: np.mean(v) if v else 0.0 for k, v in epoch_losses.items()}
        
        if (self.current_epoch + 1) % 5 == 0:
            self.loss_tracker.plot_batch_losses(self.current_epoch)
        
        return avg_losses
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.generator.eval()
        val_losses = []
        psnr_scores = []
        ssim_scores = []
        
        metrics_calc = MetricsCalculator()
        
        progress_bar = tqdm(val_loader, desc="Validation")
        
        for batch in progress_bar:
            try:
                real_source = batch['source'].to(self.device)
                real_target = batch['target'].to(self.device)
                target_phase_idx = batch['target_phase_idx'].to(self.device)
                
                # Generate target
                generated_target = self.generator(real_source, target_phase_idx)
                
                # Calculate MSE loss
                val_loss = F.mse_loss(generated_target, real_target)
                val_losses.append(val_loss.item())
                
                # Calculate PSNR
                psnr = metrics_calc.calculate_psnr(generated_target, real_target)
                psnr_scores.append(psnr)
                
                # Calculate SSIM
                ssim = metrics_calc.calculate_ssim(generated_target, real_target)
                ssim_scores.append(ssim)
                
                # Update progress bar with current metrics
                progress_bar.set_postfix({
                    'Loss': f"{val_loss.item():.4f}",
                    'PSNR': f"{psnr:.2f}",
                    'SSIM': f"{ssim:.4f}"
                })
                
            except Exception as e:
                logger.error(f"Error in validation: {e}")
                continue
        
        return {
            'val_loss': np.mean(val_losses) if val_losses else float('inf'),
            'psnr': np.mean(psnr_scores) if psnr_scores else 0.0,
            'ssim': np.mean(ssim_scores) if ssim_scores else 0.0
        }
    
    def save_checkpoint(self, val_metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        
        checkpoint = {
            'epoch': self.current_epoch,
            'generator_state': self.generator.state_dict(),
            'disc_source_state': self.disc_source.state_dict(),
            'disc_target_state': self.disc_target.state_dict(),
            'opt_gen_state': self.opt_gen.state_dict(),
            'opt_disc_source_state': self.opt_disc_source.state_dict(),
            'opt_disc_target_state': self.opt_disc_target.state_dict(),
            'val_metrics': val_metrics,
            'config': self.config
        }
        checkpoint = {
            'epoch': self.current_epoch,
            'generator_state': self.generator.state_dict(),
            'disc_source_state': self.disc_source.state_dict(),
            'disc_target_state': self.disc_target.state_dict(),
            'opt_gen_state': self.opt_gen.state_dict(),
            'opt_disc_source_state': self.opt_disc_source.state_dict(),
            'opt_disc_target_state': self.opt_disc_target.state_dict(),
            'val_metrics': val_metrics,
            'config': self.config
        }
        
        # Save regular checkpoint
        path = self.output_dir / f'checkpoint_epoch_{self.current_epoch}.pth'
        torch.save(checkpoint, path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved! Val loss: {val_metrics['val_loss']:.4f}, "
                       f"PSNR: {val_metrics['psnr']:.2f}, SSIM: {val_metrics['ssim']:.4f}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        """Main training loop."""
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Training batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        # How often to save samples (default: every epoch)
        save_samples_interval = self.config.get('save_samples_interval', 1)
        num_samples = self.config.get('num_samples_to_save', 5)
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Train
            train_losses = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Log
            logger.info(f"\nEpoch {epoch + 1}/{epochs} Summary:")
            logger.info(f"  Train - Total: {train_losses['gen_total']:.4f}, "
                       f"Cycle: {train_losses['gen_cycle']:.4f}, "
                       f"MSE: {train_losses['gen_mse']:.4f}, "
                       f"Focal: {train_losses['gen_focal']:.4f}")
            logger.info(f"  Disc: {train_losses['disc']:.4f}")
            logger.info(f"  Val - Loss: {val_metrics['val_loss']:.4f}, "
                       f"PSNR: {val_metrics['psnr']:.2f} dB, "
                       f"SSIM: {val_metrics['ssim']:.4f}")
            lr_gen = self.opt_gen.param_groups[0]['lr']
            lr_disc = self.opt_disc_source.param_groups[0]['lr'] if hasattr(self, 'opt_disc_source') else 0.0
        
            adv_weight = 0.0
            if hasattr(self, 'combined_loss') and hasattr(self.combined_loss, 'current_epoch'):
                adv_weight = self.combined_loss.get_adv_weight()
            elif hasattr(self, 'adv_schedule'):
                adv_weight = self.adv_schedule.get_weight(epoch)
            else:
                # Simple warmup calculation
                warmup_epochs = self.config.get('adv_warmup_epochs', 10)
                adv_weight = min(epoch / warmup_epochs, 1.0) * self.config.get('lambda_adv', 1.0)
            
            self.loss_tracker.update_epoch(
                epoch=epoch,
                train_losses=train_losses,
                val_metrics=val_metrics,
                lr_gen=lr_gen,
                lr_disc=lr_disc,
                adv_weight=adv_weight
            )

            if (epoch + 1) % 10 == 0:
                self.loss_tracker.create_summary_report(epoch)
            # Save sample patches
            if (epoch + 1) % save_samples_interval == 0:
                logger.info(f"Saving sample patches for epoch {epoch + 1}...")
                save_sample_patches(
                    self.generator,
                    val_loader,
                    epoch + 1,
                    self.samples_dir,
                    self.device,
                    num_samples=num_samples
                )

            


            # Save checkpoint
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
            
            self.save_checkpoint(val_metrics, is_best)
            
        logger.info("Training completed!")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main training script."""
    
    # Configuration
    config = {
        'data_dir': '../ncct_cect/vindr_ds/registered_cases',
        'labels_csv': '../ncct_cect/vindr_ds/labels.csv',
        'output_dir': '../ncct_cect/vindr_ds/clean_training',
        
        'patch_size': (64, 64),
        'patch_depth': 7,
        'overlap_ratio': 0.75,
        
        'batch_size': 4,
        'learning_rate': 2e-4,
        'epochs': 100,
        
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # Sample visualization options
        'save_samples_interval': 1,  # Save every epoch (set to 5 for every 5 epochs)
        'num_samples_to_save': 5     # Number of samples per epoch
    }
    
    logger.info("="*80)
    logger.info("CT PHASE GENERATION TRAINING")
    logger.info("="*80)
    
    # Check if data directory exists
    if not Path(config['data_dir']).exists():
        logger.error(f"❌ Data directory does not exist: {config['data_dir']}")
        logger.error("Please update the 'data_dir' in config to point to your data")
        return
    
    # Load phase mapping if available
    phase_mapping = None
    if Path(config['labels_csv']).exists():
        try:
            # Import the data loader function
            from dataloader_train import load_phase_mapping, create_data_pairs
            
            phase_mapping = load_phase_mapping(config['labels_csv'])
            logger.info(f"✓ Loaded phase mapping for {len(phase_mapping)} cases")
        except Exception as e:
            logger.warning(f"⚠ Could not load phase mapping: {e}")
            logger.warning("Will try to infer phases from filenames")
    else:
        logger.warning(f"⚠ Labels CSV not found: {config['labels_csv']}")
        logger.warning("Will try to infer phases from filenames")
    
    # Create data pairs
    try:
        from dataloader_train import create_data_pairs
        
        logger.info(f"Loading data from: {config['data_dir']}")
        data_splits = create_data_pairs(
            config['data_dir'],
            phase_mapping=phase_mapping
        )
        
        if not data_splits['train']:
            logger.error("❌ No training data pairs found!")
            logger.error("\nPossible issues:")
            logger.error("1. Data directory is empty or has wrong structure")
            logger.error("2. Files are not named with '*_registered.nii.gz' pattern")
            logger.error("3. No cases have both non-contrast and contrast phases")
            logger.error("\nExpected directory structure:")
            logger.error("  data_dir/")
            logger.error("    case_001/")
            logger.error("      case_001_series1_registered.nii.gz")
            logger.error("      case_001_series2_registered.nii.gz")
            logger.error("    case_002/")
            logger.error("      ...")
            return
        
        logger.info(f"✓ Successfully loaded {len(data_splits['train'])} training pairs")
        logger.info(f"✓ Successfully loaded {len(data_splits['val'])} validation pairs")
        
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # TODO: Load your data pairs here
    # data_pairs = load_data_pairs(config['data_dir'], config['labels_csv'])
    # train_pairs, val_pairs = split_data(data_pairs)
    
    # For now, using placeholder
    train_pairs = []  # Your training data pairs
    val_pairs = []    # Your validation data pairs
    
    # Create datasets
    train_dataset = CTPhaseDataset(
        train_pairs,
        data_splits['train'],
        patch_size=config['patch_size'],
        patch_depth=config['patch_depth'],
        overlap_ratio=config['overlap_ratio'],
        augment=True
    )
    
    val_dataset = CTPhaseDataset(
        val_pairs,
        data_splits['val'],
        patch_size=config['patch_size'],
        patch_depth=config['patch_depth'],
        overlap_ratio=0.5,
        augment=False
    )
    
    if len(train_dataset) == 0:
        logger.error("❌ Training dataset is empty after patch generation!")
        logger.error("This usually means:")
        logger.error("1. Volumes are too small for the patch size")
        logger.error("2. Not enough slices (need at least depth + 2)")
        logger.error(f"Current patch size: {config['patch_size']}, depth: {config['patch_depth']}")
        return
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    logger.info(f"✓ Training dataset: {len(train_dataset)} patches")
    logger.info(f"✓ Validation dataset: {len(val_dataset)} patches")
    
    # Initialize trainer
    trainer = CTPhaseTrainer(config)
    
    # Start training
    try:
        trainer.train(train_loader, val_loader, config['epochs'])
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer.save_checkpoint(float('inf'), is_best=False)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()