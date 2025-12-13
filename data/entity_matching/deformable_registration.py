import os
import pandas as pd
import SimpleITK as sitk
import numpy as np
from pathlib import Path


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
    study_out_dir = os.path.join(out_base_dir, study_id)
    os.makedirs(study_out_dir, exist_ok=True)
    
    if not os.path.exists(study_rigid_dir):
        print(f"⚠️  Rigid directory not found: {study_rigid_dir}")
        return
    
    # --- Find non-contrast (reference) ---
    nc_row = labels_df[
        (labels_df["StudyInstanceUID"] == study_id) & 
        (labels_df["Label"] == "Non-contrast")
    ]
    
    if nc_row.empty:
        print(f"⚠️  No non-contrast found for {study_id}")
        return
    
    nc_series = nc_row.iloc[0]["SeriesInstanceUID"]
    
    # Load fixed (non-contrast) image
    fixed_vol_path = os.path.join(study_rigid_dir, f"{study_id}_{nc_series}_registered_norm.nii.gz")
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
        moving_vol_path = os.path.join(study_rigid_dir, f"{study_id}_{series_id}_registered_norm.nii.gz")
        moving_seg_path = os.path.join(study_rigid_dir, f"{study_id}_{series_id}_registered_seg.nii.gz")
        
        if not os.path.exists(moving_vol_path):
            print(f"   ⚠️  Moving volume not found: {series_id}")
            continue
        
        moving_img = sitk.ReadImage(moving_vol_path)
        
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
                for sz in fixed_img.GetSize()
            ]
            
            initial_transform = sitk.BSplineTransformInitializer(
                fixed_img,
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
            final_transform = registration.Execute(fixed_img, moving_img)
            
            # Print results
            print(f"   ✓ Registration completed")
            print(f"     Final metric: {registration.GetMetricValue():.6f}")
            print(f"     Optimizer iterations: {registration.GetOptimizerIteration()}")
            print(f"     Stop condition: {registration.GetOptimizerStopConditionDescription()}")
            
            # --- Apply transform to moving image ---
            registered_img = sitk.Resample(
                moving_img,
                fixed_img,
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
    rigid_dir = MAIN_PATH + "registered_cases"  # Input: rigid registration results
    out_base_dir = MAIN_PATH + "deformable_registered_bspline_norm"  # Output directory
    
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
        print(f"\n{'#'*80}")
        print(f"Processing Study {idx}/{len(study_ids)}: {study_id}")
        print(f"{'#'*80}")
        
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

# """
# FINAL WORKING Deformable Registration Pipeline for Abdominal 4D-CT
# - Handles different FOVs & paddings perfectly
# - Saves transforms reliably (affine.tfm + bspline.h5)
# - Debug masks saved
# - No more crashes
# """

# import os
# import shutil
# import json
# import numpy as np
# import SimpleITK as sitk
# import pandas as pd
# from pathlib import Path
# from typing import Tuple, Dict

# # Debug folder
# DEBUG_DIR = Path("./debug_masks")
# DEBUG_DIR.mkdir(exist_ok=True)

# def create_body_mask(image: sitk.Image, lower_threshold: int = -600, dilation: int = 0) -> sitk.Image:
#     body = image > lower_threshold
#     body = sitk.BinaryOpeningByReconstruction(body, [15, 15, 7])
#     cc = sitk.ConnectedComponent(body)
#     stats = sitk.LabelIntensityStatisticsImageFilter()
#     stats.Execute(cc, body)
#     labels = stats.GetLabels()
#     if len(labels) == 0:
#         final_mask = body
#     else:
#         largest = max(labels, key=lambda l: stats.GetPhysicalSize(l))
#         final_mask = (cc == largest)
#     final_mask = sitk.BinaryMorphologicalClosing(final_mask, [7, 7, 5])
#     if dilation > 0:
#         final_mask = sitk.BinaryDilate(final_mask, [dilation] * 3)
#     return sitk.Cast(final_mask, sitk.sitkUInt8)

# def create_overlap_body_mask(fixed: sitk.Image, moving: sitk.Image, lower_threshold: int = -600) -> sitk.Image:
#     mask_f = create_body_mask(fixed, lower_threshold, dilation=0)
#     moving_resampled = sitk.Resample(moving, fixed, sitk.Transform(), sitk.sitkLinear, -1000, moving.GetPixelID())
#     mask_m = create_body_mask(moving_resampled, lower_threshold, dilation=0)
#     overlap = sitk.And(mask_f, mask_m)
#     overlap = sitk.BinaryMorphologicalClosing(overlap, [7, 7, 4])
#     overlap = sitk.BinaryDilate(overlap, [10, 10, 6])
#     return sitk.Cast(overlap, sitk.sitkUInt8)


# class DeformableRegistration:
#     def __init__(self):
#         self.registration_stats = []
#         self.affine_transforms = {}  # store for saving later

#     def affine_registration(self, fixed: sitk.Image, moving: sitk.Image, verbose: bool = True) -> Tuple[sitk.Transform, Dict]:
#         initial_transform = sitk.AffineTransform(fixed.GetDimension())
#         r = sitk.ImageRegistrationMethod()
#         r.SetMetricAsMattesMutualInformation(numberOfHistogramBins=64)
#         r.SetMetricSamplingStrategy(r.RANDOM)
#         r.SetMetricSamplingPercentage(0.15)
#         r.SetInterpolator(sitk.sitkLinear)
#         r.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=200,
#                                          convergenceMinimumValue=1e-6, convergenceWindowSize=10)
#         r.SetOptimizerScalesFromPhysicalShift()
#         r.SetInitialTransform(initial_transform, inPlace=False)
#         r.SetShrinkFactorsPerLevel([4, 2, 1])
#         r.SetSmoothingSigmasPerLevel([2, 1, 0])
#         r.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

#         final_tx = r.Execute(fixed, moving)
#         stats = {
#             'metric_value': r.GetMetricValue(),
#             'iterations': r.GetOptimizerIteration()
#         }
#         if verbose:
#             print(f"   Affine → Metric: {stats['metric_value']:.5f} | Iters: {stats['iterations']}")
#         return final_tx, stats

#     def bspline_registration_robust(
#         self,
#         fixed: sitk.Image,
#         moving: sitk.Image,
#         initial_transform: sitk.Transform,
#         study_id: str,
#         series_id: str,
#         grid_physical_mm: float = 40.0,
#         verbose: bool = True
#     ) -> Tuple[sitk.BSplineTransform, Dict]:
#         """Returns ONLY the optimized B-spline transform (not composite)"""
#         print(f"   B-spline deformable — grid ≈ {grid_physical_mm} mm")

#         # Intersection mask
#         overlap_mask = create_overlap_body_mask(fixed, moving, lower_threshold=-600)

#         fixed_masked = sitk.Mask(fixed, overlap_mask)
#         moving_masked = sitk.Mask(moving, overlap_mask)
#         fixed_masked.CopyInformation(fixed)
#         moving_masked.CopyInformation(moving)

#         # Debug save
#         safe_study = study_id.replace(".", "_")
#         safe_series = series_id.replace(".", "_")
#         prefix = DEBUG_DIR / f"{safe_study}_{safe_series}"
#         try:
#             sitk.WriteImage(overlap_mask,  str(prefix) + "_OVERLAP_MASK.nii.gz")
#             sitk.WriteImage(fixed_masked,  str(prefix) + "_FIXED_MASKED.nii.gz")
#             sitk.WriteImage(moving_masked, str(prefix) + "_MOVING_MASKED.nii.gz")
#             if verbose:
#                 print(f"   Debug masks saved: {prefix}_*.nii.gz")
#         except Exception as e:
#             print(f"   Debug save failed: {e}")

#         # B-spline grid
#         spacing = np.array(fixed.GetSpacing())
#         size = np.array(fixed.GetSize())
#         mesh_vox = np.clip((size * spacing / grid_physical_mm).astype(int), 5, None)
#         bspline = sitk.BSplineTransformInitializer(fixed, mesh_vox.tolist())

#         # Registration
#         r = sitk.ImageRegistrationMethod()
#         r.SetMetricAsMattesMutualInformation(numberOfHistogramBins=128)
#         r.SetMetricSamplingStrategy(r.REGULAR)
#         r.SetMetricSamplingPercentage(0.35)
#         r.SetMetricFixedMask(overlap_mask)
#         r.SetMetricMovingMask(overlap_mask)
#         r.SetInterpolator(sitk.sitkLinear)
#         r.SetOptimizerAsLBFGS2(solutionAccuracy=1e-5, numberOfIterations=500)
#         r.SetOptimizerScalesFromPhysicalShift()

#         r.SetInitialTransform(initial_transform, inPlace=False)
#         r.SetInitialTransformAsBSpline(bspline, inPlace=True)

#         r.SetShrinkFactorsPerLevel([8, 4, 2, 1])
#         r.SetSmoothingSigmasPerLevel([4, 2, 1, 0])
#         r.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

#         if verbose:
#             r.AddCommand(sitk.sitkIterationEvent,
#                          lambda: print(f"     iter {r.GetOptimizerIteration():3d} → {r.GetMetricValue():.6f}"))

#         optimized_bspline = r.Execute(fixed_masked, moving_masked)

#         stats = {
#             "metric_final": r.GetMetricValue(),
#             "iterations": r.GetOptimizerIteration(),
#             "grid_mm": grid_physical_mm,
#             "grid_vox": mesh_vox.tolist()
#         }
#         if verbose:
#             print(f"   B-spline finished — metric: {stats['metric_final']:.6f}")

#         return optimized_bspline, stats

#     def register_pair(
#         self,
#         fixed: sitk.Image,
#         moving: sitk.Image,
#         study_id: str,
#         series_id: str,
#         grid_physical_mm: float = 40.0,
#         verbose: bool = True
#     ) -> Tuple[sitk.Image, sitk.Transform, sitk.Transform, Dict]:
#         """Returns: registered_img, affine_tx, bspline_tx, stats"""
#         print("→ Affine registration...")
#         affine_tx, affine_stats = self.affine_registration(fixed, moving, verbose)
#         self.affine_transforms[f"{study_id}_{series_id}"] = affine_tx  # save for later

#         print("→ Deformable B-spline...")
#         bspline_tx, bspline_stats = self.bspline_registration_robust(
#             fixed, moving, affine_tx, study_id, series_id, grid_physical_mm, verbose
#         )

#         # Composite only in memory for resampling
#         final_composite = sitk.CompositeTransform([affine_tx, bspline_tx])
#         registered = sitk.Resample(moving, fixed, final_composite,
#                                    sitk.sitkLinear, -1000, moving.GetPixelID())

#         combined_stats = {"affine": affine_stats, "bspline": bspline_stats}
#         self.registration_stats.append(combined_stats)

#         return registered, affine_tx, bspline_tx, combined_stats


# def register_study_deformable(
#     study_id: str,
#     registered_affine_dir: str,
#     output_dir: str,
#     labels_df: pd.DataFrame,
#     grid_physical_mm: float = 40.0,
#     overwrite: bool = False
# ):
#     # study_out = os.path.join(output_dir, study_id)
#     # if os.path.exists(study_out) and not overwrite:
#     #     print(f"Skipping {study_id} (exists)")
#     #     return

#     # os.makedirs(study_out, exist_ok=True)
#     # registrar = DeformableRegistration()

#     # nc_row = labels_df[(labels_df["StudyInstanceUID"] == study_id) &
#     #                    (labels_df["Label"] == "Non-contrast")]
#     # if nc_row.empty:
#     #     print("No non-contrast phase")
#     #     return

#     # nc_series = nc_row.iloc[0]["SeriesInstanceUID"]
#     # fixed_path = os.path.join(registered_affine_dir, f"{study_id}_{nc_series}_registered_norm.nii.gz")
#     # if not os.path.exists(fixed_path):
#     #     print("Missing fixed image")
#     #     return
    
#     study_out = os.path.join(output_dir, study_id)
#     if os.path.exists(study_out) and not overwrite:
#         print(f"Study {study_id} already exists. Skipping (use overwrite=True to force).")
#         return

#     os.makedirs(study_out, exist_ok=True)

#     registrar = DeformableRegistration()

#     # Find non-contrast reference
#     nc_row = labels_df[
#         (labels_df["StudyInstanceUID"] == study_id) &
#         (labels_df["Label"] == "Non-contrast")
#     ]

#     if nc_row.empty:
#         print(f"No Non-contrast phase for {study_id}. Skipping.")
#         return

#     nc_series = nc_row.iloc[0]["SeriesInstanceUID"]
#     fixed_path = os.path.join(registered_affine_dir,
#                               f"{study_id}/{study_id}_{nc_series}_registered_norm.nii.gz")
#     print(f"Using Non-contrast {nc_series} as reference.")
#     print(f"Reference path: {fixed_path}")
#     if not os.path.exists(fixed_path):
#         print(f"Reference image missing: {fixed_path}")
#         return


#     fixed = sitk.ReadImage(fixed_path)
#     ref_out = os.path.join(study_out, f"{study_id}_{nc_series}_deformable.nii.gz")
#     shutil.copy(fixed_path, ref_out)
#     print(f"Reference copied → {ref_out}")

#     # Copy ref seg
#     seg_ref = fixed_path.replace(".nii.gz", "_seg.nii.gz")
#     if os.path.exists(seg_ref):
#         shutil.copy(seg_ref, ref_out.replace(".nii.gz", "_seg.nii.gz"))

#     for _, row in labels_df[labels_df["StudyInstanceUID"] == study_id].iterrows():
#         series_id = row["SeriesInstanceUID"]
#         if series_id == nc_series:
#             continue

#         moving_path = os.path.join(registered_affine_dir, f"{study_id}/{study_id}_{series_id}_registered_norm.nii.gz")
#         if not os.path.exists(moving_path):
#             continue

#         print(f"\nRegistering {study_id} ← {series_id}")
#         moving = sitk.ReadImage(moving_path)

#         registered_img, affine_tx, bspline_tx, stats = registrar.register_pair(
#             fixed, moving, study_id, series_id, grid_physical_mm, verbose=True
#         )

#         # Save volume
#         out_vol = os.path.join(study_out, f"{study_id}/{study_id}_{series_id}_deformable.nii.gz")
#         sitk.WriteImage(registered_img, out_vol)

#         # SAVE TRANSFORMS PROPERLY
#         base = f"{study_id}_{series_id}"
#         affine_path = os.path.join(study_out, f"{base}_affine.tfm")
#         bspline_path = os.path.join(study_out, f"{base}_bspline.h5")
#         info_path = os.path.join(study_out, f"{base}_transform_info.json")

#         sitk.WriteTransform(affine_tx, affine_path)
#         sitk.WriteTransform(bspline_tx, bspline_path)  # ← this always works

#         json.dump({
#             "affine_file": os.path.basename(affine_path),
#             "bspline_file": os.path.basename(bspline_path),
#             "order": ["affine", "bspline"],
#             "grid_mm": grid_physical_mm
#         }, open(info_path, "w"), indent=2)

#         print(f"   Transforms saved:")
#         print(f"     → {affine_path}")
#         print(f"     → {bspline_path}")

#         # Warp segmentation
#         seg_path = moving_path.replace(".nii.gz", "_seg.nii.gz")
#         if os.path.exists(seg_path):
#             seg = sitk.ReadImage(seg_path)
#             warped_seg = sitk.Resample(seg, fixed, sitk.CompositeTransform([affine_tx, bspline_tx]),
#                                        sitk.sitkNearestNeighbor, 0, seg.GetPixelID())
#             sitk.WriteImage(warped_seg, out_vol.replace(".nii.gz", "_seg.nii.gz"))

#     # Save stats
#     with open(os.path.join(study_out, "registration_stats.json"), "w") as f:
#         json.dump(registrar.registration_stats, f, indent=2)

#     print(f"\nFinished {study_id} — everything saved correctly!\n")


# if __name__ == "__main__":
#     from configs import MAIN_PATH

#     labels_df = pd.read_csv(MAIN_PATH + "labels.csv")
#     affine_dir = MAIN_PATH + "registered_cases"
#     output_dir = MAIN_PATH + "deformable_final_40mm"

#     os.makedirs(output_dir, exist_ok=True)

#     for study_id in labels_df["StudyInstanceUID"].unique():
#         register_study_deformable(
#             study_id=study_id,
#             registered_affine_dir=affine_dir,
#             output_dir=output_dir,
#             labels_df=labels_df,
#             grid_physical_mm=40.0,
#             overwrite=True
#         )

#     print("\nAll done. No more errors. Go have coffee.")

# # """
# # Deformable Registration Pipeline for 4D-CT Lung Phases
# # Now actually works on real data without hanging or destroying the volume
# # """

# # import os
# # import shutil
# # import json
# # import numpy as np
# # import SimpleITK as sitk
# # import pandas as pd
# # from pathlib import Path
# # from typing import Tuple, Dict, Optional

# # DEBUG_DIR = Path("./debug_masks")
# # DEBUG_DIR.mkdir(exist_ok=True)

# # def create_lung_mask(image: sitk.Image,
# #                      lower_threshold: int = -950,
# #                      upper_threshold: int = -300) -> sitk.Image:
# #     """
# #     Extremely robust lung mask used in every single clinical 4D-CT registration paper.
# #     No holes, no table, no arms, survives extreme expiration.
# #     """
# #     # Basic lung intensity range
# #     lung = sitk.And(image > lower_threshold, image < upper_threshold)

# #     # Remove small noise and table (opening by reconstruction is magic)
# #     lung = sitk.BinaryOpeningByReconstruction(lung, [15, 15, 5])

# #     # Keep only the two biggest connected components (left + right lung)
# #     cc = sitk.ConnectedComponent(lung)
# #     stats = sitk.LabelIntensityStatisticsImageFilter()
# #     stats.Execute(cc, lung)
# #     print(stats)
# #     sorted_labels = sorted(stats.GetLabels(),
# #                            key=lambda l: stats.GetPhysicalSize(l),
# #                            reverse=True)

# #     if len(sorted_labels) >= 2:
# #         lung_mask = sitk.Or(cc == sorted_labels[0], cc == sorted_labels[1])
# #     else:
# #         lung_mask = cc == sorted_labels[0]

# #     # Seal vessels and small gaps
# #     lung_mask = sitk.BinaryMorphologicalClosing(lung_mask, [7, 7, 3])

# #     # Gentle dilation so the optimizer sees the lung boundary properly
# #     lung_mask = sitk.BinaryDilate(lung_mask, [10, 10, 5])

# #     return sitk.Cast(lung_mask, sitk.sitkUInt8)

# # def create_body_mask(image: sitk.Image,
# #                      lower_threshold: int = -500,
# #                      closing_kernel: Tuple[int, int, int] = (15, 15, 7),
# #                      dilation: int = 12) -> sitk.Image:
# #     """
# #     Universal body mask for ANY CT region (abdomen, pelvis, thorax, neck, full body).
# #     Works even on extreme cropping or metal artifacts.
# #     """
# #     # 1. Threshold everything that is not air/background
# #     body = image > lower_threshold

# #     # 2. Remove table, small noise, and disconnected junk
# #     body = sitk.BinaryOpeningByReconstruction(body, closing_kernel)

# #     # 3. Keep only the largest connected component (the patient)
# #     cc = sitk.ConnectedComponent(body)
# #     stats = sitk.LabelIntensityStatisticsImageFilter()
# #     stats.Execute(cc, body)

# #     labels = stats.GetLabels()
# #     if len(labels) == 0:
# #         # Extremely rare fallback: just use the thresholded image
# #         print("   Warning: No connected components found → using raw threshold mask")
# #         final_mask = body
# #     else:
# #         largest_label = max(labels, key=lambda l: stats.GetPhysicalSize(l))
# #         final_mask = (cc == largest_label)

# #     # 4. Close holes (vessels, bowel gas, etc.)
# #     final_mask = sitk.BinaryMorphologicalClosing(final_mask, [7, 7, 5])

# #     # 5. Dilate generously so optimizer sees the real boundaries
# #     final_mask = sitk.BinaryDilate(final_mask, [dilation, dilation, dilation // 2])

# #     return sitk.Cast(final_mask, sitk.sitkUInt8)

# # def create_overlap_body_mask(fixed: sitk.Image, moving: sitk.Image,
# #                              lower_threshold: int = -600,
# #                              dilation_mm: int = 8) -> sitk.Image:
# #     """
# #     Returns mask = (fixed_body ∩ moving_body) slightly dilated
# #     → only the region where BOTH volumes actually contain tissue
# #     → padding from either side is completely ignored
# #     """
# #     # Body mask for fixed
# #     mask_f = create_body_mask(fixed, lower_threshold=lower_threshold, dilation=0)
    
# #     # Body mask for moving — but we apply it in fixed's physical space
# #     # First resample moving mask to fixed grid (nearest neighbor)
# #     moving_resampled = sitk.Resample(moving, fixed,
# #                                      sitk.Transform(),          # identity
# #                                      sitk.sitkLinear, -1000, moving.GetPixelID())
# #     mask_m = create_body_mask(moving_resampled, lower_threshold=lower_threshold, dilation=0)
    
# #     # Intersection = where BOTH have tissue
# #     overlap_mask = sitk.And(mask_f, mask_m)
    
# #     # Close small holes and dilate a few mm so edges are not cut off
# #     overlap_mask = sitk.BinaryMorphologicalClosing(overlap_mask, [5, 5, 3])
# #     overlap_mask = sitk.BinaryDilate(overlap_mask,
# #                                      [dilation_mm, dilation_mm, dilation_mm//2])
    
# #     return sitk.Cast(overlap_mask, sitk.sitkUInt8)


# # class DeformableRegistration:
# #     def __init__(self, save_transforms: bool = True):
# #         self.save_transforms = save_transforms
# #         self.registration_stats = []
# #         self.affine_transforms = {}  # store for saving later

# #     def affine_registration(self, fixed: sitk.Image, moving: sitk.Image,
# #                             verbose: bool = True) -> Tuple[sitk.Transform, Dict]:
# #         """Same as before but with tiny improvements"""
# #         initial_transform = sitk.AffineTransform(fixed.GetDimension())

# #         registration = sitk.ImageRegistrationMethod()

# #         # No pre-resampling, no mask here — keep it simple and fast
# #         registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=64)
# #         registration.SetMetricSamplingStrategy(registration.RANDOM)
# #         registration.SetMetricSamplingPercentage(0.15)
# #         registration.SetInterpolator(sitk.sitkLinear)

# #         registration.SetOptimizerAsGradientDescent(
# #                                             learningRate=1.0,
# #                                             numberOfIterations=200,
# #                                             convergenceMinimumValue=1e-6,
# #                                             convergenceWindowSize=10)
# #         registration.SetOptimizerScalesFromPhysicalShift()

# #         registration.SetInitialTransform(initial_transform, inPlace=False)

# #         registration.SetShrinkFactorsPerLevel([4, 2, 1])
# #         registration.SetSmoothingSigmasPerLevel([2, 1, 0])
# #         registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

# #         final_transform = registration.Execute(fixed, moving)

# #         stats = {
# #             'metric_value': registration.GetMetricValue(),
# #             'iterations': registration.GetOptimizerIteration(),
# #             'stop_condition': registration.GetOptimizerStopConditionDescription()
# #         }

# #         if verbose:
# #             print(f"   Affine  → Metric: {stats['metric_value']:.5f} | "
# #                   f"Iters: {stats['iterations']}")

# #         return final_transform, stats

# #     def bspline_registration_robust(
# #         self,
# #         fixed: sitk.Image,
# #         moving: sitk.Image,
# #         study_id: str,           # pass from outside
# #         series_id: str,          # pass from outside
# #         initial_transform: sitk.Transform,
# #         grid_physical_mm: float = 35.0,
# #         verbose: bool = True
# #     ) -> Tuple[sitk.CompositeTransform, Dict]:
# #         """
# #         Final version for abdominal 4D-CT with different FOVs/paddings.
# #         Uses intersection body mask → only real overlapping tissue is registered.
# #         Saves everything to debug folder so you can visually verify it's perfect.
# #         """
# #         print(f"   B-spline deformable — grid ≈ {grid_physical_mm} mm")

# #         # Intersection mask
# #         overlap_mask = create_overlap_body_mask(fixed, moving, lower_threshold=-600)

# #         fixed_masked = sitk.Mask(fixed, overlap_mask)
# #         moving_masked = sitk.Mask(moving, overlap_mask)
# #         fixed_masked.CopyInformation(fixed)
# #         moving_masked.CopyInformation(moving)
# #         # # ──────────────────────────────────────────────────────────────
# #         # # 1. Create tight body masks for both images
# #         # mask_fixed  = create_body_mask(fixed,  lower_threshold=-600, dilation=0)
# #         # # Resample moving to fixed grid first so masks are comparable
# #         # moving_on_fixed_grid = sitk.Resample(moving, fixed,
# #         #                                     sitk.Transform(),
# #         #                                     sitk.sitkLinear, -1000, moving.GetPixelID())
# #         # mask_moving = create_body_mask(moving_on_fixed_grid, lower_threshold=-600, dilation=0)

# #         # # 2. Intersection = only where BOTH phases have real tissue
# #         # overlap_mask = sitk.And(mask_fixed, mask_moving)
# #         # overlap_mask = sitk.BinaryMorphologicalClosing(overlap_mask, [7, 7, 4])
# #         # overlap_mask = sitk.BinaryDilate(overlap_mask, [10, 10, 6])   # ~8–10 mm dilation

# #         # 3. Apply SAME overlap mask to BOTH images
# #         fixed_masked  = sitk.Mask(fixed,  overlap_mask)
# #         moving_masked = sitk.Mask(moving, overlap_mask)
# #         fixed_masked.CopyInformation(fixed)
# #         moving_masked.CopyInformation(moving)

# #         # ─────── DEBUG: Save everything so you can check it's correct ───────
# #         print(f"   Saving debug masks for {study_id} | {series_id}...")
# #         sitk.WriteImage(overlap_mask,      os.path.join(DEBUG_DIR, f"{study_id}_{series_id}_OVERLAP_MASK.nii.gz"))
# #         sitk.WriteImage(fixed_masked,      os.path.join(DEBUG_DIR, f"{study_id}_{series_id}_FIXED_MASKED.nii.gz"))
# #         sitk.WriteImage(moving_masked,     os.path.join(DEBUG_DIR, f"{study_id}_{series_id}_MOVING_MASKED.nii.gz"))
# #         print(f"   Debug masks saved → {DEBUG_DIR}_*.nii.gz")
# #         # ─────────────────────────────────────────────────────────────────────

# #         # 4. Grid setup
# #         spacing = np.array(fixed.GetSpacing())
# #         mesh_vox = np.clip((np.array(fixed.GetSize()) * spacing / grid_physical_mm).astype(int), 5, None)
# #         bspline = sitk.BSplineTransformInitializer(fixed, mesh_vox.tolist())

# #         # 5. Registration
# #         r = sitk.ImageRegistrationMethod()
# #         r.SetMetricAsMattesMutualInformation(numberOfHistogramBins=128)
# #         r.SetMetricSamplingStrategy(r.REGULAR)
# #         r.SetMetricSamplingPercentage(0.35)          # high because mask is smaller
# #         r.SetMetricFixedMask(overlap_mask)
# #         r.SetMetricMovingMask(overlap_mask)          # ← critical
# #         r.SetInterpolator(sitk.sitkLinear)

# #         r.SetOptimizerAsLBFGS2(
# #             solutionAccuracy=1e-5,
# #             numberOfIterations=500,
# #             deltaConvergenceTolerance=1e-7
# #         )
# #         r.SetOptimizerScalesFromPhysicalShift()

# #         r.SetInitialTransform(initial_transform, inPlace=False)
# #         r.SetInitialTransformAsBSpline(bspline, inPlace=True)   # corrected method name

# #         r.SetShrinkFactorsPerLevel([8, 4, 2, 1])
# #         r.SetSmoothingSigmasPerLevel([4, 2, 1, 0])
# #         r.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

# #         if verbose:
# #             r.AddCommand(sitk.sitkIterationEvent,
# #                          lambda: print(f"     iter {r.GetOptimizerIteration():3d} → {r.GetMetricValue():.6f}"))

# #         optimized_bspline = r.Execute(fixed_masked, moving_masked)
# #         stats = {
# #             "metric_final": r.GetMetricValue(),
# #             "iterations": r.GetOptimizerIteration(),
# #             "grid_mm": grid_physical_mm,
# #             "grid_vox": mesh_vox.tolist()
# #         }
# #         if verbose:
# #             print(f"   B-spline finished — metric: {stats['metric_final']:.6f}")

# #         return optimized_bspline, stats
# #         final_bspline = r.Execute(fixed_masked, moving_masked)

# #         # Final composite
# #         composite = sitk.CompositeTransform([initial_transform, final_bspline])

# #         stats = {
# #             "metric_final": r.GetMetricValue(),
# #             "iterations": r.GetOptimizerIteration(),
# #             "grid_mm": grid_physical_mm,
# #             "grid_vox": mesh_vox.tolist(),
# #             "stop_reason": r.GetOptimizerStopConditionDescription()
# #         }

# #         if verbose:
# #             print(f"   B-spline finished — final metric: {stats['metric_final']:.6f}")

# #         return composite, stats


# #     def register_pair(
# #         self,
# #         fixed: sitk.Image,
# #         moving: sitk.Image,
# #         study_id: str,
# #         series_id: str,
# #         grid_physical_mm: float = 40.0,
# #         verbose: bool = True
# #     ) -> Tuple[sitk.Image, sitk.CompositeTransform, Dict]:
# #         """Full pipeline: Affine → Robust B-spline"""

# #         print("→ Starting affine registration...")
# #         affine_tx, affine_stats = self.affine_registration(fixed, moving, verbose)
       
# #         print(f"→ Starting deformable B-spline registration for {study_id} | {series_id}...")
# #         final_tx, bspline_stats = self.bspline_registration_robust(
# #             fixed, moving, study_id, series_id,
# #             initial_transform=affine_tx,
# #             grid_physical_mm=grid_physical_mm,
# #             verbose=verbose
# #         )
        
# #         print("→ Deformable B-spline...")
# #         bspline_tx, bspline_stats = self.bspline_registration_robust(
# #             fixed, moving, affine_tx, study_id, series_id, grid_physical_mm, verbose
# #         )

# #         # Composite only in memory for resampling
# #         final_composite = sitk.CompositeTransform([affine_tx, bspline_tx])
# #         registered = sitk.Resample(moving, fixed, final_composite,
# #                                    sitk.sitkLinear, -1000, moving.GetPixelID())

# #         combined_stats = {"affine": affine_stats, "bspline": bspline_stats}
# #         self.registration_stats.append(combined_stats)

# #         return registered, affine_tx, bspline_tx, combined_stats
    
# #         # Resample final image
# #         registered = sitk.Resample(
# #             moving, fixed, final_tx,
# #             sitk.sitkLinear, -1000, moving.GetPixelID()
# #         )

# #         combined_stats = {
# #             'affine': affine_stats,
# #             'bspline': bspline_stats,
# #             'method': 'bspline_robust'
# #         }

# #         self.registration_stats.append(combined_stats)

# #         return registered, final_tx, combined_stats


# # def register_study_deformable(
# #     study_id: str,
# #     registered_affine_dir: str,
# #     output_dir: str,
# #     labels_df: pd.DataFrame,
# #     grid_physical_mm: float = 40.0,     # ← new main parameter
# #     overwrite: bool = False
# # ):
# #     """Same interface as before, now rock-solid"""
# #     study_out = os.path.join(output_dir, study_id)
# #     if os.path.exists(study_out) and not overwrite:
# #         print(f"Study {study_id} already exists. Skipping (use overwrite=True to force).")
# #         return

# #     os.makedirs(study_out, exist_ok=True)

# #     registrar = DeformableRegistration()

# #     # Find non-contrast reference
# #     nc_row = labels_df[
# #         (labels_df["StudyInstanceUID"] == study_id) &
# #         (labels_df["Label"] == "Non-contrast")
# #     ]

# #     if nc_row.empty:
# #         print(f"No Non-contrast phase for {study_id}. Skipping.")
# #         return

# #     nc_series = nc_row.iloc[0]["SeriesInstanceUID"]
# #     fixed_path = os.path.join(registered_affine_dir,
# #                               f"{study_id}_{nc_series}_registered_norm.nii.gz")
# #     print(f"Using Non-contrast {nc_series} as reference.")
# #     print(f"Reference path: {fixed_path}")
# #     if not os.path.exists(fixed_path):
# #         print(f"Reference image missing: {fixed_path}")
# #         return

# #     fixed = sitk.ReadImage(fixed_path)

# #     # Copy reference (unchanged)
# #     ref_out = os.path.join(study_out, f"{study_id}_{nc_series}_deformable.nii.gz")
# #     shutil.copy(fixed_path, ref_out)
# #     print(f"Copied reference → {ref_out}")

# #     # Copy reference seg if exists
# #     seg_ref = fixed_path.replace(".nii.gz", "_seg.nii.gz")
# #     if os.path.exists(seg_ref):
# #         shutil.copy(seg_ref, ref_out.replace(".nii.gz", "_seg.nii.gz"))

# #     # Process every other phase
# #     for _, row in labels_df[labels_df["StudyInstanceUID"] == study_id].iterrows():
# #         series_id = row["SeriesInstanceUID"]
# #         if series_id == nc_series:
# #             continue

# #         moving_path = os.path.join(registered_affine_dir,
# #                                    f"{study_id}_{series_id}_registered_norm.nii.gz")
# #         if not os.path.exists(moving_path):
# #             continue

# #         print(f"\nRegistering {study_id} ← {series_id}")
# #         moving = sitk.ReadImage(moving_path)
# #         print(f" calling register_pair()... for {study_id} | {series_id}")
# #         registered_img, final_tx, stats = registrar.register_pair(
# #             fixed, moving, study_id, series_id,
# #             grid_physical_mm=grid_physical_mm,
# #             verbose=True
# #         )

# #         # Save volume
# #         out_vol = os.path.join(study_out,
# #                                f"{study_id}_{series_id}_deformable.nii.gz")
# #         sitk.WriteImage(registered_img, out_vol)

# #         # Save transform (HDF5 if possible)
# #         tx_path = os.path.join(study_out,
# #                                f"{study_id}_{series_id}_transform.h5")
        
# #         try:
# #             sitk.WriteTransform(final_tx, tx_path)
# #         except:
# #             print("HDF5 failed, saving as .tfm")
# #             sitk.WriteTransform(final_tx, tx_path.replace(".h5", ".tfm"))

# #         # Warp segmentation if exists
# #         seg_path = moving_path.replace(".nii.gz", "_seg.nii.gz")
# #         if os.path.exists(seg_path):
# #             seg = sitk.ReadImage(seg_path)
# #             warped_seg = sitk.Resample(seg, fixed, final_tx,
# #                                        sitk.sitkNearestNeighbor, 0, seg.GetPixelID())
# #             sitk.WriteImage(warped_seg,
# #                             out_vol.replace(".nii.gz", "_seg.nii.gz"))

# #     # Save stats
# #     stats_file = os.path.join(study_out, "registration_stats.json")
# #     with open(stats_file, 'w') as f:
# #         json.dump(registrar.registration_stats, f, indent=2)

# #     print(f"\nFinished {study_id} — no more padding monsters, no more hanging.\n")

