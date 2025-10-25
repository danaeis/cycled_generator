"""
Registration Quality Validation
Quantitative metrics: Dice, Hausdorff, Jacobian determinant
"""

import numpy as np
import SimpleITK as sitk
from scipy.spatial.distance import directed_hausdorff
from typing import Dict, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


class RegistrationValidator:
    """
    Validate registration quality using segmentation masks
    """
    
    def __init__(self):
        self.metrics_history = []
    
    def compute_dice(
        self, 
        mask1: sitk.Image, 
        mask2: sitk.Image, 
        label: int = None
    ) -> float:
        """
        Dice Similarity Coefficient
        
        Args:
            mask1, mask2: Binary or multi-label masks
            label: Specific label to compute (None = all non-zero)
        """
        arr1 = sitk.GetArrayFromImage(mask1)
        arr2 = sitk.GetArrayFromImage(mask2)
        
        if label is not None:
            arr1 = (arr1 == label).astype(np.uint8)
            arr2 = (arr2 == label).astype(np.uint8)
        else:
            arr1 = (arr1 > 0).astype(np.uint8)
            arr2 = (arr2 > 0).astype(np.uint8)
        
        intersection = np.sum(arr1 * arr2)
        size1 = np.sum(arr1)
        size2 = np.sum(arr2)
        
        if size1 + size2 == 0:
            return 1.0  # Both empty
        
        dice = 2.0 * intersection / (size1 + size2)
        return dice
    
    def compute_hausdorff(
        self,
        mask1: sitk.Image,
        mask2: sitk.Image,
        percentile: float = 95.0,
        max_points: int = 10000,  # Limit points for memory efficiency
        use_sitk: bool = True      # Use SimpleITK's fast implementation
    ) -> float:
        """
        Hausdorff Distance (95th percentile by default)
        
        Memory-efficient implementation with point sampling
        Lower is better (in mm)
        
        Args:
            mask1, mask2: Binary or multi-label masks
            percentile: Percentile for robust HD (95 is standard)
            max_points: Maximum points to use (memory protection)
            use_sitk: If True, use SimpleITK's fast implementation (recommended)
        """
        if use_sitk:
            # Use SimpleITK's efficient implementation
            try:
                hausdorff_filter = sitk.HausdorffDistanceImageFilter()
                hausdorff_filter.Execute(mask1 > 0, mask2 > 0)
                
                if percentile >= 95:
                    # SimpleITK computes max Hausdorff by default
                    return hausdorff_filter.GetHausdorffDistance()
                else:
                    # For other percentiles, fall back to manual computation
                    pass
            except Exception as e:
                print(f"   ⚠️  SimpleITK Hausdorff failed, using manual: {e}")
        
        # Manual computation with memory protection
        arr1 = sitk.GetArrayFromImage(mask1) > 0
        arr2 = sitk.GetArrayFromImage(mask2) > 0
        
        # Get surface points (boundary voxels only) to reduce memory
        from scipy.ndimage import binary_erosion
        
        # Extract surfaces by erosion
        surface1 = arr1 & ~binary_erosion(arr1)
        surface2 = arr2 & ~binary_erosion(arr2)
        
        coords1 = np.argwhere(surface1)
        coords2 = np.argwhere(surface2)
        
        if len(coords1) == 0 or len(coords2) == 0:
            return np.inf
        
        # Sample points if too many (memory protection)
        if len(coords1) > max_points:
            indices = np.random.choice(len(coords1), max_points, replace=False)
            coords1 = coords1[indices]
        
        if len(coords2) > max_points:
            indices = np.random.choice(len(coords2), max_points, replace=False)
            coords2 = coords2[indices]
        
        # Convert to physical coordinates
        spacing = np.array(mask1.GetSpacing())[::-1]  # Z,Y,X to spacing
        coords1_mm = coords1 * spacing
        coords2_mm = coords2 * spacing
        
        # Compute percentile Hausdorff using chunking to avoid memory issues
        if percentile < 100:
            # Compute distances in chunks
            chunk_size = 1000
            all_min_dists = []
            
            for i in range(0, len(coords1_mm), chunk_size):
                chunk = coords1_mm[i:i+chunk_size]
                # Compute distances for this chunk
                dists = cdist(chunk, coords2_mm, metric='euclidean')
                min_dists = np.min(dists, axis=1)
                all_min_dists.extend(min_dists)
            
            hausdorff = np.percentile(all_min_dists, percentile)
        else:
            # Full Hausdorff (max of directed distances)
            # Still use chunking for memory efficiency
            max_dist_1to2 = 0
            chunk_size = 1000
            
            for i in range(0, len(coords1_mm), chunk_size):
                chunk = coords1_mm[i:i+chunk_size]
                dists = cdist(chunk, coords2_mm, metric='euclidean')
                max_dist_1to2 = max(max_dist_1to2, np.max(np.min(dists, axis=1)))
            
            max_dist_2to1 = 0
            for i in range(0, len(coords2_mm), chunk_size):
                chunk = coords2_mm[i:i+chunk_size]
                dists = cdist(chunk, coords1_mm, metric='euclidean')
                max_dist_2to1 = max(max_dist_2to1, np.max(np.min(dists, axis=1)))
            
            hausdorff = max(max_dist_1to2, max_dist_2to1)
        
        return hausdorff
    
    def compute_jacobian_determinant(
        self,
        displacement_field: sitk.Image
    ) -> Dict[str, float]:
        """
        Jacobian determinant statistics
        
        Measures local volume change:
        - |J| < 0: Folding (BAD)
        - |J| = 1: No change
        - |J| > 1: Expansion
        - 0 < |J| < 1: Contraction
        
        Returns dict with min, max, mean, std, and % negative
        """
        jacobian_filter = sitk.DisplacementFieldJacobianDeterminantFilter()
        jacobian_image = jacobian_filter.Execute(displacement_field)
        
        jacobian_array = sitk.GetArrayFromImage(jacobian_image)
        
        stats = {
            'min': float(np.min(jacobian_array)),
            'max': float(np.max(jacobian_array)),
            'mean': float(np.mean(jacobian_array)),
            'std': float(np.std(jacobian_array)),
            'negative_percent': float(100 * np.mean(jacobian_array < 0))
        }
        
        return stats
    
    def compute_target_registration_error(
        self,
        landmarks_fixed: np.ndarray,
        landmarks_moving: np.ndarray,
        transform: sitk.Transform
    ) -> Dict[str, float]:
        """
        Target Registration Error (TRE)
        
        Args:
            landmarks_fixed: Nx3 array of landmark positions in fixed image
            landmarks_moving: Nx3 array of corresponding positions in moving
            transform: Registration transform
            
        Returns:
            Dict with mean, max, std TRE in mm
        """
        errors = []
        
        for lm_fixed, lm_moving in zip(landmarks_fixed, landmarks_moving):
            # Transform moving landmark
            lm_transformed = transform.TransformPoint(lm_moving)
            
            # Euclidean distance
            error = np.linalg.norm(np.array(lm_transformed) - np.array(lm_fixed))
            errors.append(error)
        
        errors = np.array(errors)
        
        return {
            'mean_tre': float(np.mean(errors)),
            'max_tre': float(np.max(errors)),
            'std_tre': float(np.std(errors)),
            'median_tre': float(np.median(errors))
        }
    
    def validate_registration(
        self,
        fixed_seg: sitk.Image,
        moving_seg: sitk.Image,
        transform: sitk.Transform = None,
        displacement_field: sitk.Image = None,
        labels: List[int] = None,
        transform_path: str = None  # Added: path to load transform
    ) -> Dict:
        """
        Complete validation pipeline
        
        Args:
            fixed_seg: Reference segmentation
            moving_seg: Registered segmentation
            transform: Registration transform (optional, for TRE)
            displacement_field: For Jacobian analysis
            labels: List of label IDs to evaluate separately
            transform_path: Path to .h5 transform file to load and compute Jacobian
        """
        results = {}
        
        # Overall Dice and Hausdorff
        results['dice_overall'] = self.compute_dice(fixed_seg, moving_seg)
        results['hausdorff_95'] = self.compute_hausdorff(fixed_seg, moving_seg)
        
        # Per-label metrics
        if labels:
            results['per_label'] = {}
            for label in labels:
                results['per_label'][label] = {
                    'dice': self.compute_dice(fixed_seg, moving_seg, label),
                }
        
        # Jacobian - try to get displacement field
        if displacement_field is not None:
            results['jacobian'] = self.compute_jacobian_determinant(displacement_field)
        elif transform_path and os.path.exists(transform_path):
            # Try to load transform and extract displacement field
            try:
                loaded_transform = sitk.ReadTransform(transform_path)
                disp_field = self._extract_displacement_field(loaded_transform, fixed_seg)
                if disp_field is not None:
                    results['jacobian'] = self.compute_jacobian_determinant(disp_field)
            except Exception as e:
                print(f"   ⚠️ Could not compute Jacobian from transform: {e}")
        elif transform is not None:
            # Try to extract from provided transform
            disp_field = self._extract_displacement_field(transform, fixed_seg)
            if disp_field is not None:
                results['jacobian'] = self.compute_jacobian_determinant(disp_field)
        
        self.metrics_history.append(results)
        return results
    
    def _extract_displacement_field(
        self, 
        transform: sitk.Transform, 
        reference_image: sitk.Image
    ):
        """
        Extract displacement field from transform
        """
        try:
            # If it's already a displacement field transform
            if hasattr(transform, 'GetDisplacementField'):
                return transform.GetDisplacementField()
            
            # If it's a CompositeTransform, get the last transform (B-spline)
            if isinstance(transform, sitk.CompositeTransform):
                num_transforms = transform.GetNumberOfTransforms()
                for i in range(num_transforms - 1, -1, -1):  # Start from last
                    component = transform.GetNthTransform(i)
                    if hasattr(component, 'GetDisplacementField'):
                        return component.GetDisplacementField()
            
            # Convert transform to displacement field
            return sitk.TransformToDisplacementField(
                transform,
                sitk.sitkVectorFloat64,
                reference_image.GetSize(),
                reference_image.GetOrigin(),
                reference_image.GetSpacing(),
                reference_image.GetDirection()
            )
        except Exception as e:
            print(f"   ⚠️ Failed to extract displacement field: {e}")
            return None
    
    def print_metrics(self, results: Dict, case_name: str = ""):
        """Pretty print validation results"""
        print(f"\n{'='*60}")
        print(f"Registration Quality Metrics{' - ' + case_name if case_name else ''}")
        print(f"{'='*60}")
        
        print(f"📊 Overall Dice Score: {results['dice_overall']:.4f}")
        print(f"📏 Hausdorff Distance (95%): {results['hausdorff_95']:.2f} mm")
        
        if 'per_label' in results:
            print(f"\n🏷️  Per-Label Dice Scores:")
            for label, metrics in results['per_label'].items():
                print(f"   Label {label}: {metrics['dice']:.4f}")
        
        if 'jacobian' in results:
            jac = results['jacobian']
            print(f"\n🔄 Jacobian Determinant:")
            print(f"   Mean: {jac['mean']:.4f}")
            print(f"   Range: [{jac['min']:.4f}, {jac['max']:.4f}]")
            print(f"   Std: {jac['std']:.4f}")
            print(f"   Negative %: {jac['negative_percent']:.2f}%")
            
            if jac['negative_percent'] > 5:
                print(f"   ⚠️  WARNING: >5% negative Jacobian (folding detected)")
    
    def save_metrics_csv(self, output_path: str):
        """Save all metrics to CSV"""
        df = pd.DataFrame(self.metrics_history)
        df.to_csv(output_path, index=False)
        print(f"📊 Metrics saved to: {output_path}")


def validate_study_registrations(
    study_id: str,
    deformable_dir: str,
    labels_df: pd.DataFrame,
    output_dir: str
):
    """
    Validate all registrations for a study
    
    Args:
        deformable_dir: Directory with deformable registration results
        output_dir: Where to save validation results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    validator = RegistrationValidator()
    
    # Get non-contrast reference
    nc_row = labels_df[
        (labels_df["StudyInstanceUID"] == study_id) & 
        (labels_df["Label"] == "Non-contrast")
    ]
    
    if nc_row.empty:
        print(f"⚠️ No non-contrast for {study_id}")
        return
    
    nc_series = nc_row.iloc[0]["SeriesInstanceUID"]
    fixed_seg_path = os.path.join(
        deformable_dir,
        f"{study_id}_{nc_series}_deformable_seg.nii.gz"
    )
    
    if not os.path.exists(fixed_seg_path):
        print(f"⚠️ Reference segmentation not found: {fixed_seg_path}")
        return
    
    fixed_seg = sitk.ReadImage(fixed_seg_path)
    
    # Also load fixed volume for displacement field computation
    fixed_vol_path = os.path.join(
        deformable_dir,
        f"{study_id}_{nc_series}_deformable.nii.gz"
    )
    fixed_vol = sitk.ReadImage(fixed_vol_path) if os.path.exists(fixed_vol_path) else None
    
    # Validate each registered series
    series_rows = labels_df[labels_df["StudyInstanceUID"] == study_id]
    
    for _, row in series_rows.iterrows():
        series_id = row["SeriesInstanceUID"]
        
        if series_id == nc_series:
            continue
        
        moving_seg_path = os.path.join(
            deformable_dir,
            f"{study_id}_{series_id}_deformable_seg.nii.gz"
        )
        
        if not os.path.exists(moving_seg_path):
            continue
        
        try:
            moving_seg = sitk.ReadImage(moving_seg_path)
            
            # Look for transform file (.h5 or components)
            transform_path = os.path.join(
                deformable_dir,
                f"{study_id}_{series_id}_transform.h5"
            )
            
            # Load transform if exists
            transform = None
            if os.path.exists(transform_path):
                try:
                    transform = sitk.ReadTransform(transform_path)
                except Exception as e:
                    print(f"   ⚠️ Could not load transform: {e}")
            
            # Validate with Jacobian computation
            results = validator.validate_registration(
                fixed_seg, 
                moving_seg, 
                transform=transform,
                transform_path=transform_path  # Pass path for Jacobian
            )
            
            case_name = f"{study_id}_{series_id}"
            validator.print_metrics(results, case_name)
            
            # Save individual result
            result_file = os.path.join(
                output_dir,
                f"{study_id}_{series_id}_metrics.json"
            )
            
            import json
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)
        
        except Exception as e:
            print(f"❌ Error validating {series_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save summary CSV
    summary_csv = os.path.join(output_dir, f"{study_id}_validation_summary.csv")
    validator.save_metrics_csv(summary_csv)


# Comparison: Affine vs Deformable
def compare_registration_methods(
    study_id: str,
    affine_dir: str,
    deformable_dir: str,
    labels_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compare affine-only vs deformable registration quality
    """
    validator_affine = RegistrationValidator()
    validator_deform = RegistrationValidator()
    
    nc_row = labels_df[
        (labels_df["StudyInstanceUID"] == study_id) & 
        (labels_df["Label"] == "Non-contrast")
    ]
    nc_series = nc_row.iloc[0]["SeriesInstanceUID"]
    
    # Fixed reference
    fixed_seg = sitk.ReadImage(
        os.path.join(affine_dir, f"{study_id}_{nc_series}_registered_seg.nii.gz")
    )
    
    comparison = []
    
    series_rows = labels_df[labels_df["StudyInstanceUID"] == study_id]
    for _, row in series_rows.iterrows():
        series_id = row["SeriesInstanceUID"]
        if series_id == nc_series:
            continue
        
        # Affine
        affine_seg_path = os.path.join(
            affine_dir, f"{study_id}_{series_id}_registered_seg.nii.gz"
        )
        
        # Deformable
        deform_seg_path = os.path.join(
            deformable_dir, f"{study_id}_{series_id}_deformable_seg.nii.gz"
        )
        
        if not os.path.exists(affine_seg_path) or not os.path.exists(deform_seg_path):
            continue
        
        affine_seg = sitk.ReadImage(affine_seg_path)
        deform_seg = sitk.ReadImage(deform_seg_path)
        
        # Compute metrics
        dice_affine = validator_affine.compute_dice(fixed_seg, affine_seg)
        dice_deform = validator_deform.compute_dice(fixed_seg, deform_seg)
        
        hd_affine = validator_affine.compute_hausdorff(fixed_seg, affine_seg)
        hd_deform = validator_deform.compute_hausdorff(fixed_seg, deform_seg)
        
        comparison.append({
            'study_id': study_id,
            'series_id': series_id,
            'phase': row['Label'],
            'dice_affine': dice_affine,
            'dice_deformable': dice_deform,
            'dice_improvement': dice_deform - dice_affine,
            'hausdorff_affine': hd_affine,
            'hausdorff_deformable': hd_deform,
            'hausdorff_reduction': hd_affine - hd_deform
        })
    
    df = pd.DataFrame(comparison)
    
    print(f"\n{'='*70}")
    print(f"Registration Method Comparison - {study_id}")
    print(f"{'='*70}")
    print(f"\nAverage Dice Improvement: {df['dice_improvement'].mean():.4f}")
    print(f"Average Hausdorff Reduction: {df['hausdorff_reduction'].mean():.2f} mm")
    print(f"\n{df.to_string(index=False)}")
    
    return df


# Usage Example
if __name__ == "__main__":
    import os
    from configs import MAIN_PATH
    
    labels_csv = MAIN_PATH + "labels.csv"
    labels_df = pd.read_csv(labels_csv)
    
    # Directories
    affine_dir = MAIN_PATH + "test_registered_cases"
    deformable_dir = MAIN_PATH + "deformable_registeredbspline"
    validation_output = MAIN_PATH + "validation_resultsbspline"
    
    os.makedirs(validation_output, exist_ok=True)
    
    # Validate each study
    for study_id in labels_df["StudyInstanceUID"].unique():  # Test first

        study_deform = os.path.join(deformable_dir, study_id)
        study_valid = os.path.join(validation_output, study_id)
        if not os.path.exists(study_deform):
            continue
        
        validate_study_registrations(
            study_id, study_deform, labels_df, study_valid
        )
        
        # Compare methods
        study_affine = os.path.join(affine_dir, study_id)
        comparison_df = compare_registration_methods(
            study_id, study_affine, study_deform, labels_df
        )
        
        # Save comparison
        comparison_csv = os.path.join(
            validation_output, f"{study_id}_method_comparison.csv"
        )
        comparison_df.to_csv(comparison_csv, index=False)