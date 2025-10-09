import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class CTPhaseDataset(Dataset):
    """
    Optimized dataset for CT phase generation with center-focused patching.
    
    Extracts patches centered around the body region (image center) rather than
    starting from (0,0), ensuring better coverage of anatomical structures.
    """
    
    def __init__(
        self,
        data_pairs: List[Dict],
        patch_size: Tuple[int, int] = (64, 64),
        patch_depth: int = 7,
        overlap_ratio: float = 0.5,
        augment: bool = True,
        body_focused: bool = False,  # New parameter
        body_threshold: float = -500.0  # HU threshold for body detection
    ):
        self.data_pairs = data_pairs
        self.patch_size = patch_size
        self.patch_depth = patch_depth
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
            'venous': 2,  # Map venous to portal
            'delayed': 3
        }
        
        logger.info(f"Initializing dataset with {len(data_pairs)} pairs")
        logger.info(f"Patch size: {patch_size}, Depth: {patch_depth}, Overlap: {overlap_ratio}")
        logger.info(f"Body-focused patching: {body_focused}")
        
        self._compute_patch_coordinates()
        logger.info(f"Generated {len(self.patch_coords)} total patches")
    
    def _find_body_center(self, volume: np.ndarray) -> Tuple[int, int]:
        """
        Find the center of the body region by detecting non-air voxels.
        
        Args:
            volume: 3D CT volume [D, H, W]
            
        Returns:
            (center_y, center_x): Coordinates of body center
        """
        # Take middle slice for body detection
        mid_slice = volume[volume.shape[2] // 2]
        
        # # Threshold to find body (HU > -500 for soft tissue)
        # body_mask = mid_slice > self.body_threshold
        
        # # Find bounding box of body region
        # if body_mask.sum() > 0:
        #     y_indices, x_indices = np.where(body_mask)
            
        #     # Calculate center of mass of body region
        #     center_y = int(np.mean(y_indices))
        #     center_x = int(np.mean(x_indices))
            
        #     logger.debug(f"Body center detected at: ({center_y}, {center_x})")
        # else:
        # Fallback to image center if no body detected
        center_y = volume.shape[0] // 2
        center_x = volume.shape[1] // 2
        logger.debug(f"Using image center: ({center_y}, {center_x})")
        
        return center_y, center_x
    
    def _compute_patch_coordinates(self):
        """
        Pre-compute centered patch coordinates focusing on body region.
        
        Patches are generated centered around the body (image center),
        expanding outward with specified overlap ratio.
        """
        padding = self.patch_depth // 2
        
        for pair_idx, pair_data in enumerate(self.data_pairs):
            try:
                # Load volumes to get dimensions
                source_vol = nib.load(pair_data['source_path']).get_fdata()
                target_vol = nib.load(pair_data['target_path']).get_fdata()
                
                # Validate shapes match
                if source_vol.shape != target_vol.shape:
                    logger.warning(f"Shape mismatch for pair {pair_idx}, skipping")
                    continue
                
                height, width, depth = source_vol.shape
                
                # Check minimum requirements
                print(f"{depth} & {self.patch_depth}")

                if depth < self.patch_depth + 2:
                    logger.warning(f"Insufficient depth ({depth}) for pair {pair_idx}, skipping")
                    continue
                
                if height < self.patch_size[0] or width < self.patch_size[1]:
                    logger.warning(f"Insufficient spatial size for pair {pair_idx}, skipping")
                    continue
                
                # Find center of body region
                if self.body_focused:
                    center_y, center_x = self._find_body_center(source_vol)
                else:
                    center_y = height // 2
                    center_x = width // 2
                
                # Calculate step sizes for overlap
                step_y = max(1, int(self.patch_size[0] * (1 - self.overlap_ratio)))
                step_x = max(1, int(self.patch_size[1] * (1 - self.overlap_ratio)))
                
                # Generate Y coordinates centered around body center
                y_coords = self._generate_centered_coordinates(
                    center=center_y,
                    patch_size=self.patch_size[0],
                    volume_size=height,
                    step=step_y
                )
                
                # Generate X coordinates centered around body center
                x_coords = self._generate_centered_coordinates(
                    center=center_x,
                    patch_size=self.patch_size[1],
                    volume_size=width,
                    step=step_x
                )
                
                # Generate Z coordinates (all valid slices)
                z_range = range(padding, depth - padding)
                
                # Store all centered patch coordinates
                patch_count = 0
                for center_z in z_range:
                    for y_start in y_coords:
                        for x_start in x_coords:
                            self.patch_coords.append((pair_idx, center_z, y_start, x_start))
                            patch_count += 1
                
                logger.info(f"Pair {pair_idx}: Generated {patch_count} centered patches")
                logger.info(f"  Body center: ({center_y}, {center_x})")
                logger.info(f"  Spatial coverage: {len(y_coords)}(Y) x {len(x_coords)}(X) patches")
                
            except Exception as e:
                logger.error(f"Error processing pair {pair_idx}: {e}")
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
            center: Center coordinate of body region
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
            # Try to add patch above/left
            coord_before = center_start - offset
            # Try to add patch below/right
            coord_after = center_start + offset
            
            added = False
            
            # Add before if valid
            if coord_before >= 0 and coord_before + patch_size <= volume_size:
                coords.insert(0, coord_before)
                added = True
            
            # Add after if valid
            if coord_after >= 0 and coord_after + patch_size <= volume_size:
                coords.append(coord_after)
                added = True
            
            # Stop if we can't add any more patches
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
        """Extract organ masks from segmentation volume."""
        patch_seg = seg_volume[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Define organ labels (adjust based on your segmentation)
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
        """Apply 3D augmentations."""
        # Random horizontal flip
        if np.random.random() > 0.5:
            source = np.flip(source, axis=2).copy()
            target = np.flip(target, axis=2).copy()
            masks = {k: np.flip(v, axis=2).copy() for k, v in masks.items()}
        
        # Random vertical flip
        if np.random.random() > 0.5:
            source = np.flip(source, axis=1).copy()
            target = np.flip(target, axis=1).copy()
            masks = {k: np.flip(v, axis=1).copy() for k, v in masks.items()}
        
        # Random 90-degree rotations (in axial plane)
        if np.random.random() > 0.5:
            k = np.random.randint(1, 4)  # 1, 2, or 3 rotations
            source = np.rot90(source, k=k, axes=(1, 2)).copy()
            target = np.rot90(target, k=k, axes=(1, 2)).copy()
            masks = {organ: np.rot90(mask, k=k, axes=(1, 2)).copy() 
                    for organ, mask in masks.items()}
        
        return source, target, masks
    
    def __len__(self) -> int:
        return len(self.patch_coords)
    
    def __getitem__(self, idx: int) -> Dict:
        pair_idx, center_z, y_start, x_start = self.patch_coords[idx]
        pair_data = self.data_pairs[pair_idx]
        
        try:
            # Load volumes
            source_vol = nib.load(pair_data['source_path']).get_fdata()
            target_vol = nib.load(pair_data['target_path']).get_fdata()
            
            # Normalize
            source_vol = self._normalize_intensity(source_vol)
            target_vol = self._normalize_intensity(target_vol)
            
            # Extract centered patch
            padding = self.patch_depth // 2
            z_start = center_z - padding
            z_end = center_z + padding + 1
            y_end = y_start + self.patch_size[0]
            x_end = x_start + self.patch_size[1]
            
            source_patch = source_vol[z_start:z_end, y_start:y_end, x_start:x_end]
            target_patch = target_vol[z_start:z_end, y_start:y_end, x_start:x_end]
            
            # Verify patch is centered correctly
            # Center of patch should align with body center
            patch_center_y = y_start + self.patch_size[0] // 2
            patch_center_x = x_start + self.patch_size[1] // 2
            
            # Extract organ masks if available
            masks = {}
            if pair_data.get('target_seg'):
                try:
                    seg_vol = nib.load(pair_data['target_seg']).get_fdata()
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
            
            # Convert to tensors
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
                'patch_center': (patch_center_y, patch_center_x)  # For debugging
            }
            
        except Exception as e:
            logger.error(f"Error loading patch {idx}: {e}")
            # Return dummy data
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


# Visualization utility to verify centered patching
def visualize_patch_coverage(dataset: CTPhaseDataset, pair_idx: int = 0):
    """
    Visualize how patches cover the volume (for debugging).
    
    Args:
        dataset: CTPhaseDataset instance
        pair_idx: Which data pair to visualize
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    # Get all patches for this pair
    pair_patches = [
        coord for coord in dataset.patch_coords if coord[0] == pair_idx
    ]
    
    if not pair_patches:
        print(f"No patches found for pair {pair_idx}")
        return
    
    # Load volume to get dimensions
    pair_data = dataset.data_pairs[pair_idx]
    volume = nib.load(pair_data['source_path']).get_fdata()
    height, width, depth = volume.shape
    print(width, height, depth)
    # Take middle slice for visualization
    mid_slice_idx = depth // 2
    mid_slice = volume[:,:,mid_slice_idx]
    
    # Find body center
    center_y, center_x = dataset._find_body_center(volume)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(mid_slice, cmap='gray', aspect='auto')
    
    # Draw all patches that include this slice
    print(pair_patches)
    for _, center_z, y_start, x_start in pair_patches:
        padding = dataset.patch_depth // 2
        if abs(center_z - mid_slice_idx) <= padding:
            # This patch includes the middle slice
            rect = mpatches.Rectangle(
                (x_start, y_start), 
                dataset.patch_size[1], 
                dataset.patch_size[0],
                linewidth=1, 
                edgecolor='cyan', 
                facecolor='none',
                alpha=0.5
            )
            ax.add_patch(rect)
    
    # Mark body center
    ax.plot(center_x, center_y, 'r+', markersize=20, markeredgewidth=3, 
            label='Body Center')
    
    # Mark image center for comparison
    ax.plot(width // 2, height // 2, 'g+', markersize=15, markeredgewidth=2,
            label='Image Center')
    
    ax.set_title(f'Patch Coverage (Pair {pair_idx}, Slice {mid_slice_idx})\n'
                 f'Total patches: {len(pair_patches)}')
    ax.legend()
    ax.set_xlabel('X (Width)')
    ax.set_ylabel('Y (Height)')
    
    plt.tight_layout()
    plt.savefig(f'patch_coverage_pair_{pair_idx}.png', dpi=150, bbox_inches='tight')
    print(f"Saved visualization to patch_coverage_pair_{pair_idx}.png")
    plt.close()


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Example data pairs (replace with your actual data)
    data_pairs = [
        {
            'source_path': Path('../ncct_cect/vindr_ds/test_registered_cases/1.2.840.113619.2.278.3.717616.306.1582703645.511/1.2.840.113619.2.278.3.717616.306.1582703645.511_1.2.840.113619.2.278.3.717616.306.1582703645.516.3_registered.nii.gz'),
            'target_path': Path('../ncct_cect/vindr_ds/test_registered_cases/1.2.840.113619.2.278.3.717616.306.1582703645.511/1.2.840.113619.2.278.3.717616.306.1582703645.511_1.2.840.113619.2.278.3.717616.306.1582703645.622.4198401_registered.nii.gz'),
            'target_seg': Path('../ncct_cect/vindr_ds/test_registered_cases/1.2.840.113619.2.278.3.717616.306.1582703645.511/1.2.840.113619.2.278.3.717616.306.1582703645.511_1.2.840.113619.2.278.3.717616.306.1582703645.622.4198401_registered_seg.nii.gz'),
            'source_phase': 'non-contrast',
            'target_phase': 'arterial',
            'case_id': '1.2.840.113619.2.278.3.717616.306.1582703645.511'
        }
    ]
    
    # Create dataset with body-focused patching
    dataset = CTPhaseDataset(
        data_pairs=data_pairs,
        patch_size=(96, 96),
        patch_depth=7,
        overlap_ratio=0.5,
        augment=True,
        body_focused=False  # Enable center-focused patching
    )
    
    print(f"Total patches: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    print(f"Sample shape: {sample['source'].shape}")
    print(f"Patch center: {sample['patch_center']}")
    
    # Visualize patch coverage
    visualize_patch_coverage(dataset, pair_idx=0)