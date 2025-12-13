import os
import pandas as pd
import SimpleITK as sitk
import numpy as np
from pathlib import Path

def ensure_same_dtype(fixed_img: sitk.Image, 
                      moving_img: sitk.Image, 
                      target_type: int = sitk.sitkFloat32) -> tuple:
    """
    Ensure both images have the same pixel type.
    
    Args:
        fixed_img: Fixed image
        moving_img: Moving image
        target_type: Target pixel type (default: Float32)
        
    Returns:
        (fixed_casted, moving_casted)
    """
    fixed_type = fixed_img.GetPixelID()
    moving_type = moving_img.GetPixelID()
    
    if fixed_type != target_type:
        print(f"   Casting fixed: {fixed_img.GetPixelIDTypeAsString()} → Float32")
        fixed_img = sitk.Cast(fixed_img, target_type)
    
    if moving_type != target_type:
        print(f"   Casting moving: {moving_img.GetPixelIDTypeAsString()} → Float32")
        moving_img = sitk.Cast(moving_img, target_type)
    
    return fixed_img, moving_img

def create_combined_organ_mask(seg_volume: sitk.Image, 
                                organ_labels: list = None,
                                dilate_mm: float = 5.0) -> sitk.Image:
    """
    Create a binary mask from segmentation that covers all organs.
    
    Args:
        seg_volume: Segmentation volume with organ labels
        organ_labels: List of label values to include (None = all non-zero)
        dilate_mm: Dilation radius in mm to include boundary regions
        
    Returns:
        Binary mask (1 = organs, 0 = background/padding)
    """
    seg_array = sitk.GetArrayFromImage(seg_volume)
    
    if organ_labels is None:
        # Use all non-zero labels
        mask_array = (seg_array > 0).astype(np.uint8)
    else:
        # Combine specific organ labels
        mask_array = np.zeros_like(seg_array, dtype=np.uint8)
        for label in organ_labels:
            mask_array[seg_array == label] = 1
    
    mask_img = sitk.GetImageFromArray(mask_array)
    mask_img.CopyInformation(seg_volume)
    
    # Dilate to include boundary regions (important for registration)
    if dilate_mm > 0:
        # Convert mm to voxels
        spacing = seg_volume.GetSpacing()
        dilate_radius = [int(dilate_mm / s) for s in spacing]
        
        mask_img = sitk.BinaryDilate(mask_img, dilate_radius)
        print(f"   ✓ Dilated organ mask by {dilate_mm}mm ({dilate_radius} voxels)")
    
    # Fill holes
    mask_img = sitk.BinaryFillhole(mask_img)
    
    return mask_img


def setup_bspline_registration(fixed_img: sitk.Image,
                               moving_img: sitk.Image,
                               fixed_mask: sitk.Image = None,
                               moving_mask: sitk.Image = None,
                               grid_spacing_mm: float = 40.0,
                               num_iterations: int = 100) -> sitk.ImageRegistrationMethod:
    """
    Configure B-spline deformable registration with organ-focused metrics.
    
    Args:
        fixed_img: Reference image
        moving_img: Image to deform
        fixed_mask: Binary mask of organs in fixed image
        moving_mask: Binary mask of organs in moving image
        grid_spacing_mm: B-spline grid spacing in mm (smaller = more flexible)
        num_iterations: Maximum optimization iterations
        
    Returns:
        Configured registration method
    """
    registration = sitk.ImageRegistrationMethod()
    
    # --- Similarity Metric ---
    # Use Mattes Mutual Information (best for multi-modal CT phases)
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    
    # CRITICAL: Apply masks to focus on organs only
    if fixed_mask is not None:
        registration.SetMetricFixedMask(fixed_mask)
        print("   ✓ Using fixed organ mask")
    
    if moving_mask is not None:
        registration.SetMetricMovingMask(moving_mask)
        print("   ✓ Using moving organ mask")
    
    # Sampling strategy
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.10)  # Sample 10% of masked region
    
    # --- Interpolator ---
    registration.SetInterpolator(sitk.sitkLinear)
    
    # --- Optimizer ---
    registration.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=num_iterations,
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=500,
        costFunctionConvergenceFactor=1e7
    )
    
    # --- Multi-resolution strategy ---
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    print(f"   B-spline grid spacing: {grid_spacing_mm}mm")
    print(f"   Max iterations: {num_iterations}")
    # print(f"   Multi-resolution levels: {registration.GetNumberOfLevels()}")
    
    return registration


def register_study_deformable(
    study_id: str,
    rigid_dir: str,
    out_base_dir: str,
    labels_df: pd.DataFrame,
    grid_spacing_mm: float = 40.0,
    organ_labels: list = None,
    save_transform: bool = True
):
    """
    Apply B-spline deformable registration to all series in a study,
    using organ masks to prevent deformation in padding regions.
    
    Args:
        study_id: StudyInstanceUID
        rigid_dir: Path to rigidly registered volumes
        out_base_dir: Output directory for deformable results
        labels_df: DataFrame with study/series labels
        grid_spacing_mm: B-spline control point spacing (40mm recommended)
        organ_labels: List of organ label IDs (None = all organs)
        save_transform: Whether to save displacement field
    """
    # --- Setup paths ---
    study_rigid_dir = os.path.join(rigid_dir, study_id)
    
    
    if not os.path.exists(study_rigid_dir):
        # print(f"⚠️  Rigid directory not found: {study_rigid_dir}")
        return
    
    # --- Find non-contrast (reference) ---
    nc_row = labels_df[
        (labels_df["StudyInstanceUID"] == study_id) & 
        (labels_df["Label"] == "Non-contrast")
    ]
    
    if nc_row.empty:
        # print(f"⚠️  No non-contrast found for {study_id}")
        return
    
    study_out_dir = os.path.join(out_base_dir, study_id)
    os.makedirs(study_out_dir, exist_ok=True)
    
    nc_series = nc_row.iloc[0]["SeriesInstanceUID"]
    
    # Load fixed (non-contrast) image
    fixed_vol_path = os.path.join(study_rigid_dir, f"{study_id}_{nc_series}_cropped.nii.gz")
    fixed_seg_path = os.path.join(study_rigid_dir, f"{study_id}_{nc_series}_registered_seg.nii.gz")
    
    if not os.path.exists(fixed_vol_path):
        print(f"⚠️  Fixed volume not found: {fixed_vol_path}")
        return
    
    fixed_img = sitk.ReadImage(fixed_vol_path)
    
    # Load fixed mask if available
    fixed_mask = None
    if os.path.exists(fixed_seg_path):
        fixed_seg = sitk.ReadImage(fixed_seg_path)
        fixed_mask = create_combined_organ_mask(
            fixed_seg, 
            organ_labels=organ_labels,
            dilate_mm=5.0
        )
        print(f"   ✓ Loaded fixed organ mask from {nc_series}")
    else:
        print(f"   ⚠️  No segmentation for fixed image, using full volume")
    
    # --- Process each moving series ---
    series_rows = labels_df[labels_df["StudyInstanceUID"] == study_id]
    
    for _, row in series_rows.iterrows():
        series_id = row["SeriesInstanceUID"]
        phase_label = row["Label"]
        
        # Skip non-contrast (it's the reference)
        if series_id == nc_series:
            # Just copy the reference
            out_vol = os.path.join(study_out_dir, f"{series_id}_deformable.nii.gz")
            out_seg = os.path.join(study_out_dir, f"{series_id}_deformable_seg.nii.gz")
            
            sitk.WriteImage(fixed_img, out_vol)
            if os.path.exists(fixed_seg_path):
                sitk.WriteImage(fixed_seg, out_seg)
            
            print(f"   ✓ Copied reference: {nc_series} (Non-contrast)")
            continue
        
        # Load moving image
        moving_vol_path = os.path.join(study_rigid_dir, f"{study_id}_{series_id}_cropped.nii.gz")
        moving_seg_path = os.path.join(study_rigid_dir, f"{study_id}_{series_id}_registered_seg.nii.gz")
        
        if not os.path.exists(moving_vol_path):
            print(f"   ⚠️  Moving volume not found: {series_id}")
            continue
        
        moving_img = sitk.ReadImage(moving_vol_path)
        
        fixed_img_cast, moving_img_cast = ensure_same_dtype(fixed_img, moving_img)
        
        # Load moving mask if available
        moving_mask = None
        if os.path.exists(moving_seg_path):
            moving_seg = sitk.ReadImage(moving_seg_path)
            moving_mask = create_combined_organ_mask(
                moving_seg,
                organ_labels=organ_labels,
                dilate_mm=5.0
            )
            print(f"   ✓ Loaded moving organ mask from {series_id}")
        
        try:
            print(f"\n{'='*80}")
            print(f"Deformable Registration: {study_id}")
            print(f"  Phase: {phase_label} (Series: {series_id})")
            print(f"{'='*80}")
            
            # --- Initialize B-spline transform ---
            transform_domain_mesh_size = [
                int(sz / grid_spacing_mm) 
                for sz in fixed_img_cast.GetSize()
            ]
            
            initial_transform = sitk.BSplineTransformInitializer(
                fixed_img_cast,
                transformDomainMeshSize=transform_domain_mesh_size,
                order=3
            )
            
            print(f"   B-spline mesh size: {transform_domain_mesh_size}")
            
            # --- Setup registration ---
            registration = setup_bspline_registration(
                fixed_img,
                moving_img,
                fixed_mask=fixed_mask,
                moving_mask=moving_mask,
                grid_spacing_mm=grid_spacing_mm,
                num_iterations=100
            )
            
            registration.SetInitialTransform(initial_transform, inPlace=True)
            
            # --- Execute registration ---
            print(f"   Starting B-spline optimization...")
            final_transform = registration.Execute(fixed_img_cast, moving_img_cast)
            
            # Print results
            print(f"   ✓ Registration completed")
            print(f"     Final metric: {registration.GetMetricValue():.6f}")
            print(f"     Optimizer iterations: {registration.GetOptimizerIteration()}")
            print(f"     Stop condition: {registration.GetOptimizerStopConditionDescription()}")
            
            # --- Apply transform to moving image ---
            registered_img = sitk.Resample(
                moving_img_cast,
                fixed_img_cast,
                final_transform,
                sitk.sitkLinear,
                0.0,
                moving_img.GetPixelID()
            )
            
            # Save deformed image (FIXED PATH)
            out_vol = os.path.join(study_out_dir, f"{series_id}_deformable.nii.gz")
            sitk.WriteImage(registered_img, out_vol)
            print(f"   ✓ Saved: {out_vol}")
            
            # --- Apply transform to segmentation (if exists) ---
            if os.path.exists(moving_seg_path):
                moving_seg = sitk.ReadImage(moving_seg_path)
                
                registered_seg = sitk.Resample(
                    moving_seg,
                    fixed_img,
                    final_transform,
                    sitk.sitkNearestNeighbor,  # Use nearest neighbor for labels
                    0,
                    moving_seg.GetPixelID()
                )
                
                out_seg = os.path.join(study_out_dir, f"{series_id}_deformable_seg.nii.gz")
                sitk.WriteImage(registered_seg, out_seg)
                print(f"   ✓ Saved segmentation: {out_seg}")
            
            # --- Save displacement field (optional) ---
            if save_transform:
                displacement_field = sitk.TransformToDisplacementField(
                    final_transform,
                    sitk.sitkVectorFloat64,
                    fixed_img.GetSize(),
                    fixed_img.GetOrigin(),
                    fixed_img.GetSpacing(),
                    fixed_img.GetDirection()
                )
                
                out_field = os.path.join(study_out_dir, f"{series_id}_displacement_field.nii.gz")
                sitk.WriteImage(displacement_field, out_field)
                print(f"   ✓ Saved displacement field: {out_field}")
            
        except Exception as e:
            print(f"   ❌ Error processing {series_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n✅ Deformable registration completed for {study_id}\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    from configs import MAIN_PATH
    
    # --- Configuration ---
    labels_csv = MAIN_PATH + "labels.csv"
    rigid_dir = MAIN_PATH + "registered_aligned"  # Input: rigid registration results
    out_base_dir = MAIN_PATH + "deformable_aligned_equalized_registered_bspline"  # Output directory
    
    # B-spline settings
    GRID_SPACING_MM = 40.0  # Control point spacing (40mm = moderate flexibility)
    
    # Organ labels to focus on (adjust based on your segmentation)
    # Example: [1=liver, 2=spleen, 3=right kidney, 4=left kidney, 5=pancreas]
    ORGAN_LABELS = None  # None = use all organs in segmentation
    
    # --- Load labels ---
    labels_df = pd.read_csv(labels_csv)
    print(f"Loaded {len(labels_df)} series from {len(labels_df['StudyInstanceUID'].unique())} studies")
    
    # --- Create output directory ---
    os.makedirs(out_base_dir, exist_ok=True)
    
    # --- Process each study ---
    study_ids = labels_df["StudyInstanceUID"].unique()
    
    for idx, study_id in enumerate(study_ids, 1):
        # print(f"\n{'#'*80}")
        # print(f"Processing Study {idx}/{len(study_ids)}: {study_id}")
        # print(f"{'#'*80}")
        
        register_study_deformable(
            study_id=study_id,
            rigid_dir=rigid_dir,
            out_base_dir=out_base_dir,
            labels_df=labels_df,
            grid_spacing_mm=GRID_SPACING_MM,
            organ_labels=ORGAN_LABELS,
            save_transform=True
        )
    
    print("\n" + "="*80)
    print("✅ ALL DEFORMABLE REGISTRATIONS COMPLETED")
    print("="*80)
