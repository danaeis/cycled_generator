"""
Anatomical Slice Matching and Alignment
Ensures pixel-level correspondence across volumes
"""

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import center_of_mass
from scipy.spatial.distance import cdist
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import os
import json


def convert_numpy_types(obj):
    """
    Convert numpy types to Python native types for JSON serialization
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


class AnatomicalSliceMatcher:
    """
    Match slices across volumes using anatomical landmarks
    """
    
    def __init__(self):
        self.correspondence_map = {}
    
    def compute_organ_centroids(
        self,
        segmentation: sitk.Image,
        labels: List[int] = None
    ) -> Dict[int, np.ndarray]:
        """
        Compute centroid for each organ label
        
        Returns:
            Dict mapping label_id -> (z, y, x) centroid position
        """
        seg_array = sitk.GetArrayFromImage(segmentation)
        spacing = np.array(segmentation.GetSpacing())[::-1]  # To Z,Y,X
        origin = np.array(segmentation.GetOrigin())[::-1]
        
        if labels is None:
            labels = np.unique(seg_array)
            labels = labels[labels > 0]  # Exclude background
        
        centroids = {}
        
        for label in labels:
            mask = (seg_array == label)
            if np.sum(mask) == 0:
                continue
            
            # Compute center of mass
            com = center_of_mass(mask)
            
            # Convert to physical coordinates
            physical_coords = origin + np.array(com) * spacing
            
            centroids[label] = physical_coords
        
        return centroids
    
    def align_volumes_by_centroids(
        self,
        fixed_seg: sitk.Image,
        moving_seg: sitk.Image,
        reference_label: int = 1  # e.g., liver
    ) -> Dict[str, float]:
        """
        Compute translation offset between volumes based on organ centroid
        
        Returns:
            Dict with 'z_offset', 'y_offset', 'x_offset' in mm
        """
        fixed_centroids = self.compute_organ_centroids(fixed_seg, [reference_label])
        moving_centroids = self.compute_organ_centroids(moving_seg, [reference_label])
        
        if reference_label not in fixed_centroids or reference_label not in moving_centroids:
            raise ValueError(f"Label {reference_label} not found in segmentations")
        
        fixed_center = fixed_centroids[reference_label]
        moving_center = moving_centroids[reference_label]
        
        offset = fixed_center - moving_center
        
        return {
            'z_offset': offset[0],
            'y_offset': offset[1],
            'x_offset': offset[2]
        }
    
    def find_anatomical_correspondence(
        self,
        fixed_seg: sitk.Image,
        moving_seg: sitk.Image,
        num_landmarks: int = 10
    ) -> Dict[int, int]:
        """
        Find slice-to-slice correspondence using segmentation profiles
        
        Returns:
            Dict mapping fixed_slice_idx -> moving_slice_idx
        """
        fixed_array = sitk.GetArrayFromImage(fixed_seg)
        moving_array = sitk.GetArrayFromImage(moving_seg)
        
        # Compute anatomical "signature" per slice (area of each organ)
        fixed_profiles = self._compute_slice_profiles(fixed_array)
        moving_profiles = self._compute_slice_profiles(moving_array)
        
        # Match slices based on profile similarity
        correspondence = {}
        
        for fixed_idx in range(len(fixed_profiles)):
            # Find most similar moving slice
            similarities = []
            for moving_idx in range(len(moving_profiles)):
                sim = self._profile_similarity(
                    fixed_profiles[fixed_idx],
                    moving_profiles[moving_idx]
                )
                similarities.append(sim)
            
            best_match = np.argmax(similarities)
            correspondence[fixed_idx] = best_match
        
        self.correspondence_map = correspondence
        return correspondence
    
    def _compute_slice_profiles(self, seg_array: np.ndarray) -> List[Dict[int, float]]:
        """
        Compute anatomical profile for each axial slice
        
        Returns:
            List of dicts: [{label: area, ...}, ...]
        """
        profiles = []
        
        for z in range(seg_array.shape[0]):
            slice_2d = seg_array[z, :, :]
            profile = {}
            
            labels = np.unique(slice_2d)
            for label in labels:
                if label == 0:
                    continue
                area = np.sum(slice_2d == label)
                profile[label] = float(area)
            
            profiles.append(profile)
        
        return profiles
    
    def _profile_similarity(self, profile1: Dict, profile2: Dict) -> float:
        """
        Compute similarity between two slice profiles
        Uses cosine similarity on label areas
        """
        # Get all labels
        all_labels = set(profile1.keys()) | set(profile2.keys())
        
        if len(all_labels) == 0:
            return 0.0
        
        # Create vectors
        vec1 = np.array([profile1.get(label, 0.0) for label in all_labels])
        vec2 = np.array([profile2.get(label, 0.0) for label in all_labels])
        
        # Cosine similarity
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        return similarity
    
    def resample_to_fixed_space(
        self,
        fixed: sitk.Image,
        moving: sitk.Image,
        transform: sitk.Transform = None,
        interpolator: int = sitk.sitkLinear
    ) -> sitk.Image:
        """
        Resample moving image to fixed image space
        
        Args:
            fixed: Reference image
            moving: Image to resample
            transform: Optional transform to apply
            interpolator: sitk.sitkLinear or sitk.sitkNearestNeighbor
        """
        if transform is None:
            transform = sitk.Transform()  # Identity
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed)
        resampler.SetInterpolator(interpolator)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(transform)
        
        resampled = resampler.Execute(moving)
        
        return resampled
    
    def extract_matched_slices(
        self,
        fixed_volume: sitk.Image,
        moving_volume: sitk.Image,
        correspondence: Dict[int, int],
        output_dir: str
    ):
        """
        Extract and save matched slice pairs for verification
        """
        os.makedirs(output_dir, exist_ok=True)
        
        fixed_array = sitk.GetArrayFromImage(fixed_volume)
        moving_array = sitk.GetArrayFromImage(moving_volume)
        
        for fixed_idx, moving_idx in correspondence.items():
            if fixed_idx % 10 != 0:  # Save every 10th slice
                continue
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Fixed slice
            axes[0].imshow(fixed_array[fixed_idx], cmap='gray')
            axes[0].set_title(f'Fixed - Slice {fixed_idx}')
            axes[0].axis('off')
            
            # Moving slice
            axes[1].imshow(moving_array[moving_idx], cmap='gray')
            axes[1].set_title(f'Moving - Slice {moving_idx}')
            axes[1].axis('off')
            
            # Difference
            diff = np.abs(
                fixed_array[fixed_idx].astype(float) - 
                moving_array[moving_idx].astype(float)
            )
            axes[2].imshow(diff, cmap='hot')
            axes[2].set_title('Absolute Difference')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f'match_fixed{fixed_idx}_moving{moving_idx}.png'),
                dpi=150, bbox_inches='tight'
            )
            plt.close()
        
        print(f"✅ Saved matched slice visualizations to {output_dir}")
    
    def create_aligned_volume(
        self,
        moving_volume: sitk.Image,
        correspondence: Dict[int, int],
        target_num_slices: int
    ) -> sitk.Image:
        """
        Create new volume with slices aligned according to correspondence
        
        Args:
            moving_volume: Source volume
            correspondence: Mapping from target_idx -> source_idx
            target_num_slices: Number of slices in output
        """
        moving_array = sitk.GetArrayFromImage(moving_volume)
        
        # Create empty volume
        aligned_array = np.zeros(
            (target_num_slices, moving_array.shape[1], moving_array.shape[2]),
            dtype=moving_array.dtype
        )
        
        for target_idx in range(target_num_slices):
            if target_idx in correspondence:
                source_idx = correspondence[target_idx]
                if 0 <= source_idx < moving_array.shape[0]:
                    aligned_array[target_idx] = moving_array[source_idx]
        
        # Convert back to SimpleITK
        aligned_sitk = sitk.GetImageFromArray(aligned_array)
        aligned_sitk.CopyInformation(moving_volume)
        
        return aligned_sitk


class IntensityBasedSliceMatcher:
    """
    Alternative: Match slices using intensity-based features
    Useful when segmentations are not available
    """
    
    def __init__(self):
        self.fixed_features = None
        self.moving_features = None
    
    def extract_slice_features(self, volume: sitk.Image) -> np.ndarray:
        """
        Extract feature vector for each slice
        
        Features:
        - Mean intensity
        - Std intensity
        - Histogram bins
        - Gradient magnitude mean
        """
        volume_array = sitk.GetArrayFromImage(volume)
        num_slices = volume_array.shape[0]
        
        features = []
        
        for z in range(num_slices):
            slice_2d = volume_array[z, :, :]
            
            # Basic statistics
            mean_int = np.mean(slice_2d)
            std_int = np.std(slice_2d)
            
            # Histogram (10 bins)
            hist, _ = np.histogram(slice_2d, bins=10, density=True)
            
            # Gradient magnitude
            gy, gx = np.gradient(slice_2d.astype(float))
            grad_mag = np.sqrt(gx**2 + gy**2)
            mean_grad = np.mean(grad_mag)
            
            # Combine features
            feat = np.concatenate([
                [mean_int, std_int, mean_grad],
                hist
            ])
            
            features.append(feat)
        
        return np.array(features)
    
    def match_slices(
        self,
        fixed: sitk.Image,
        moving: sitk.Image
    ) -> Dict[int, int]:
        """
        Find slice correspondence using intensity features
        """
        # Extract features
        fixed_features = self.extract_slice_features(fixed)
        moving_features = self.extract_slice_features(moving)
        
        self.fixed_features = fixed_features
        self.moving_features = moving_features
        
        # Compute pairwise distances
        distances = cdist(fixed_features, moving_features, metric='euclidean')
        
        # Find best match for each fixed slice
        correspondence = {}
        for fixed_idx in range(len(fixed_features)):
            moving_idx = np.argmin(distances[fixed_idx, :])
            correspondence[fixed_idx] = moving_idx
        
        return correspondence


# ============================================================================
# Pixel-Level Correspondence Refinement using Feature Matching
# ============================================================================

class DenseCorrespondenceMatcher:
    """
    Refine pixel-level correspondence using dense feature matching
    Can use LoFTR-like approaches for refinement after deformable registration
    """
    
    def __init__(self, grid_size: int = 8):
        """
        Args:
            grid_size: Grid resolution for correspondence points
        """
        self.grid_size = grid_size
        self.correspondences = []
    
    def extract_grid_features(
        self,
        image_2d: np.ndarray,
        patch_size: int = 16
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features on regular grid
        
        Returns:
            positions: (N, 2) grid positions
            features: (N, D) feature vectors
        """
        h, w = image_2d.shape
        positions = []
        features = []
        
        for y in range(0, h - patch_size, self.grid_size):
            for x in range(0, w - patch_size, self.grid_size):
                # Extract patch
                patch = image_2d[y:y+patch_size, x:x+patch_size]
                
                # Simple feature: flattened patch (could use learned features)
                feat = patch.flatten()
                
                positions.append([y + patch_size//2, x + patch_size//2])
                features.append(feat)
        
        return np.array(positions), np.array(features)
    
    def match_features(
        self,
        features1: np.ndarray,
        features2: np.ndarray,
        threshold: float = 0.8
    ) -> np.ndarray:
        """
        Match features using nearest neighbor with ratio test
        
        Returns:
            matches: (M, 2) array of matched indices
        """
        # Compute pairwise distances
        distances = cdist(features1, features2, metric='euclidean')
        
        matches = []
        
        for i in range(len(features1)):
            # Two nearest neighbors
            sorted_indices = np.argsort(distances[i, :])
            nearest = sorted_indices[0]
            second_nearest = sorted_indices[1]
            
            # Ratio test (Lowe's ratio test)
            ratio = distances[i, nearest] / distances[i, second_nearest]
            
            if ratio < threshold:
                matches.append([i, nearest])
        
        return np.array(matches)
    
    def compute_dense_correspondence(
        self,
        slice1: np.ndarray,
        slice2: np.ndarray,
        visualize: bool = False
    ) -> Dict:
        """
        Compute dense correspondence between two slices
        
        Returns:
            Dict with correspondence info
        """
        # Extract features
        pos1, feat1 = self.extract_grid_features(slice1)
        pos2, feat2 = self.extract_grid_features(slice2)
        
        # Match
        matches = self.match_features(feat1, feat2)
        
        # Get matched positions
        matched_pos1 = pos1[matches[:, 0]]
        matched_pos2 = pos2[matches[:, 1]]
        
        # Compute displacement vectors
        displacements = matched_pos2 - matched_pos1
        
        result = {
            'positions1': matched_pos1,
            'positions2': matched_pos2,
            'displacements': displacements,
            'num_matches': len(matches),
            'mean_displacement': np.mean(np.linalg.norm(displacements, axis=1))
        }
        
        if visualize:
            self._visualize_correspondences(slice1, slice2, matched_pos1, matched_pos2)
        
        return result
    
    def _visualize_correspondences(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        pos1: np.ndarray,
        pos2: np.ndarray,
        num_show: int = 20
    ):
        """Visualize feature correspondences"""
        import matplotlib.pyplot as plt
        
        # Sample subset
        if len(pos1) > num_show:
            indices = np.random.choice(len(pos1), num_show, replace=False)
            pos1 = pos1[indices]
            pos2 = pos2[indices]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(img1, cmap='gray')
        axes[0].scatter(pos1[:, 1], pos1[:, 0], c='red', s=50, marker='x')
        axes[0].set_title('Fixed Slice')
        axes[0].axis('off')
        
        axes[1].imshow(img2, cmap='gray')
        axes[1].scatter(pos2[:, 1], pos2[:, 0], c='lime', s=50, marker='x')
        axes[1].set_title('Moving Slice')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()


# ============================================================================
# Complete Pipeline Integration
# ============================================================================

def align_and_match_study(
    study_id: str,
    deformable_dir: str,
    labels_df,
    output_dir: str,
    use_segmentation: bool = True
):
    """
    Complete anatomical alignment and slice matching pipeline
    
    Args:
        deformable_dir: Directory with deformable registration results
        output_dir: Where to save matched results
        use_segmentation: If True, use seg-based matching; else intensity-based
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get non-contrast reference
    nc_row = labels_df[
        (labels_df["StudyInstanceUID"] == study_id) & 
        (labels_df["Label"] == "Non-contrast")
    ]
    
    if nc_row.empty:
        print(f"⚠️ No non-contrast for {study_id}")
        return
    
    nc_series = nc_row.iloc[0]["SeriesInstanceUID"]
    
    # Load reference
    fixed_vol_path = os.path.join(
        deformable_dir, f"{study_id}_{nc_series}_deformable.nii.gz"
    )
    fixed_seg_path = os.path.join(
        deformable_dir, f"{study_id}_{nc_series}_deformable_seg.nii.gz"
    )
    
    if not os.path.exists(fixed_vol_path):
        print(f"⚠️ Reference not found: {fixed_vol_path}")
        return
    
    fixed_vol = sitk.ReadImage(fixed_vol_path)
    
    if use_segmentation and os.path.exists(fixed_seg_path):
        fixed_seg = sitk.ReadImage(fixed_seg_path)
        matcher = AnatomicalSliceMatcher()
    else:
        matcher = IntensityBasedSliceMatcher()
        use_segmentation = False
    
    # Process each series
    series_rows = labels_df[labels_df["StudyInstanceUID"] == study_id]
    
    for _, row in series_rows.iterrows():
        series_id = row["SeriesInstanceUID"]
        
        if series_id == nc_series:
            continue
        
        moving_vol_path = os.path.join(
            deformable_dir, f"{study_id}_{series_id}_deformable.nii.gz"
        )
        moving_seg_path = os.path.join(
            deformable_dir, f"{study_id}_{series_id}_deformable_seg.nii.gz"
        )
        
        if not os.path.exists(moving_vol_path):
            continue
        
        try:
            print(f"\n🔍 Matching slices: {study_id} | {series_id}")
            
            moving_vol = sitk.ReadImage(moving_vol_path)
            
            # Find correspondence
            if use_segmentation:
                moving_seg = sitk.ReadImage(moving_seg_path)
                correspondence = matcher.find_anatomical_correspondence(
                    fixed_seg, moving_seg
                )
            else:
                correspondence = matcher.match_slices(fixed_vol, moving_vol)
            
            print(f"   Found {len(correspondence)} slice correspondences")
            
            # Create aligned volume
            aligned_vol = matcher.create_aligned_volume(
                moving_vol,
                correspondence,
                sitk.GetArrayFromImage(fixed_vol).shape[0]
            )
            
            # Save
            aligned_path = os.path.join(
                output_dir, f"{study_id}_{series_id}_aligned.nii.gz"
            )
            sitk.WriteImage(aligned_vol, aligned_path)
            print(f"   ✅ Saved aligned volume: {aligned_path}")
            
            # Save correspondence map
            corr_path = os.path.join(
                output_dir, f"{study_id}_{series_id}_correspondence.json"
            )
            
            # Convert to JSON-serializable format
            correspondence_serializable = convert_numpy_types(correspondence)
            
            with open(corr_path, 'w') as f:
                json.dump(correspondence_serializable, f, indent=2)
            
            print(f"   ✅ Saved correspondence map: {corr_path}")
            
            # Visualize matches
            vis_dir = os.path.join(output_dir, "visualizations", f"{study_id}_{series_id}")
            matcher.extract_matched_slices(
                fixed_vol, moving_vol, correspondence, vis_dir
            )
        
        except Exception as e:
            print(f"❌ Error matching {series_id}: {e}")
    
    print(f"\n✅ Slice matching completed for {study_id}")


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    from configs import MAIN_PATH
    import pandas as pd
    
    labels_csv = MAIN_PATH + "labels.csv"
    labels_df = pd.read_csv(labels_csv)
    
    deformable_dir = MAIN_PATH + "deformable_registered"
    aligned_output = MAIN_PATH + "anatomically_aligned"
    
    os.makedirs(aligned_output, exist_ok=True)
    
    # Process each study
    for study_id in labels_df["StudyInstanceUID"].unique():  # Test first
        study_deform = os.path.join(deformable_dir, study_id)
        study_aligned = os.path.join(aligned_output, study_id)
        if not os.path.exists(study_deform):
            continue
        align_and_match_study(
            study_id,
            study_deform,
            labels_df,
            study_aligned,
            use_segmentation=True
        )
    
    print("\n✅ Anatomical alignment completed")