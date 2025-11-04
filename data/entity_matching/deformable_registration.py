"""
Deformable Registration Pipeline for CT Phase Alignment
Extends affine registration with B-spline deformable registration
"""

import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import json


class DeformableRegistration:
    """
    Multi-stage registration: Affine → B-spline deformable
    """
    
    def __init__(self, save_transforms: bool = True):
        self.save_transforms = save_transforms
        self.registration_stats = []
    
    def affine_registration(
        self, 
        fixed: sitk.Image, 
        moving: sitk.Image,
        verbose: bool = True
    ) -> Tuple[sitk.Transform, Dict]:
        """
        Stage 1: Affine registration (your current approach)
        """
        # Initialize transform
        # initial_transform = sitk.CenteredTransformInitializer(
        #     fixed, moving, 
        #     sitk.Euler3DTransform(),
        #     sitk.CenteredTransformInitializerFilter.GEOMETRY
        # )
        # Initialize transform as Identity
        initial_transform = sitk.AffineTransform(fixed.GetDimension())

        # Setup registration
        registration = sitk.ImageRegistrationMethod()
        # Set the identity transform as initial
        registration.SetInitialTransform(initial_transform, inPlace=False)
        # registration.SetInitialTransformAsToRigid() # Optional: Tell the registration to start with a rigid-like optimization
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(0.2)
        registration.SetInterpolator(sitk.sitkLinear)
        
        registration.SetOptimizerAsGradientDescent(
            learningRate=1.0,
            numberOfIterations=100,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10
        )
        registration.SetOptimizerScalesFromPhysicalShift()
        registration.SetInitialTransform(initial_transform, inPlace=False)
        
        # Multi-resolution
        registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        
        # Execute
        final_transform = registration.Execute(fixed, moving)
        
        stats = {
            'metric_value': registration.GetMetricValue(),
            'stop_condition': registration.GetOptimizerStopConditionDescription(),
            'iterations': registration.GetOptimizerIteration()
        }
        
        if verbose:
            print(f"   Affine - Metric: {stats['metric_value']:.6f}, "
                  f"Iterations: {stats['iterations']}")
        
        return final_transform, stats
    
    def bspline_registration(
        self,
        fixed: sitk.Image,
        moving: sitk.Image,
        initial_transform: sitk.Transform,
        mesh_size: int = 10,
        verbose: bool = True
    ) -> Tuple[sitk.Transform, Dict]:
        """
        Stage 2: B-spline deformable registration
        
        Args:
            mesh_size: Control point grid spacing (lower = more deformable)
                      10-15 for subtle deformations, 5-8 for more aggressive
        """
        # Transform domain setup
        transform_domain_mesh_size = [mesh_size] * moving.GetDimension()
        
        # Initialize B-spline transform
        bspline_transform = sitk.BSplineTransformInitializer(
            fixed, 
            transform_domain_mesh_size
        )
        
        # Composite transform: affine + bspline
        composite = sitk.CompositeTransform(fixed.GetDimension())
        composite.AddTransform(initial_transform)
        composite.AddTransform(bspline_transform)
        
        # Setup registration
        registration = sitk.ImageRegistrationMethod()
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(0.1)
        registration.SetInterpolator(sitk.sitkLinear)
        
        # LBFGSB optimizer for B-spline (better for many parameters)
        registration.SetOptimizerAsLBFGSB(
            gradientConvergenceTolerance=1e-5,
            numberOfIterations=100,
            maximumNumberOfCorrections=5,
            maximumNumberOfFunctionEvaluations=1000
        )
        
        registration.SetInitialTransform(bspline_transform, inPlace=True)
        
        # Multi-resolution with smoothing
        registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        
        # Execute
        final_bspline = registration.Execute(fixed, moving)
        
        # Return composite transform
        final_composite = sitk.CompositeTransform(fixed.GetDimension())
        final_composite.AddTransform(initial_transform)
        final_composite.AddTransform(final_bspline)
        
        stats = {
            'metric_value': registration.GetMetricValue(),
            'stop_condition': registration.GetOptimizerStopConditionDescription(),
            'iterations': registration.GetOptimizerIteration()
        }
        
        if verbose:
            print(f"   B-spline - Metric: {stats['metric_value']:.6f}, "
                  f"Iterations: {stats['iterations']}")
        
        return final_composite, stats
    
    def demons_registration(
        self,
        fixed: sitk.Image,
        moving: sitk.Image,
        initial_transform: sitk.Transform,
        num_iterations: int = 50,
        standard_deviations: float = 1.0,
        verbose: bool = True
    ) -> Tuple[sitk.Image, sitk.DisplacementFieldTransform, Dict]:
        """
        Alternative: Demons deformable registration
        Good for capturing local deformations
        
        Args:
            standard_deviations: Smoothing of displacement field (higher = smoother)
        """
        # Apply initial affine transform
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(initial_transform)
        moving_affine = resampler.Execute(moving)
        
        # Demons filter
        demons = sitk.DiffeomorphicDemonsRegistrationFilter()
        demons.SetNumberOfIterations(num_iterations)
        demons.SetStandardDeviations(standard_deviations)
        demons.SetSmoothDisplacementField(True)
        demons.SetSmoothUpdateField(True)
        
        # Execute
        displacement_field = demons.Execute(fixed, moving_affine)
        
        # Create displacement field transform
        displacement_transform = sitk.DisplacementFieldTransform(displacement_field)
        
        # Composite: affine + displacement
        composite = sitk.CompositeTransform(fixed.GetDimension())
        composite.AddTransform(initial_transform)
        composite.AddTransform(displacement_transform)
        
        stats = {
            'metric_value': demons.GetMetric(),
            'rms_change': demons.GetRMSChange()
        }
        
        if verbose:
            print(f"   Demons - Metric: {stats['metric_value']:.6f}, "
                  f"RMS: {stats['rms_change']:.6f}")
        
        return moving_affine, composite, stats
    
    def register_pair(
        self,
        fixed: sitk.Image,
        moving: sitk.Image,
        method: str = 'bspline',  # 'bspline' or 'demons'
        mesh_size: int = 10,
        verbose: bool = True
    ) -> Tuple[sitk.Image, sitk.Transform, Dict]:
        """
        Complete registration pipeline
        
        Returns:
            registered_image, final_transform, stats_dict
        """
        # Stage 1: Affine
        affine_transform, affine_stats = self.affine_registration(
            fixed, moving, verbose=verbose
        )
        
        # Stage 2: Deformable
        if method == 'bspline':
            final_transform, deform_stats = self.bspline_registration(
                fixed, moving, affine_transform, mesh_size, verbose=verbose
            )
        elif method == 'demons':
            _, final_transform, deform_stats = self.demons_registration(
                fixed, moving, affine_transform, verbose=verbose
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Resample with final transform
        registered = sitk.Resample(
            moving, fixed, final_transform,
            sitk.sitkLinear, 0.0, moving.GetPixelID()
        )
        
        # Combine stats
        combined_stats = {
            'affine': affine_stats,
            'deformable': deform_stats,
            'method': method
        }
        
        self.registration_stats.append(combined_stats)
        
        return registered, final_transform, combined_stats


def register_study_deformable(
    study_id: str,
    registered_affine_dir: str,  # Your existing registered_cases output
    output_dir: str,
    labels_df: pd.DataFrame,
    method: str = 'bspline',
    mesh_size: int = 10
):
    """
    Apply deformable registration to already-affine-registered volumes
    
    Args:
        registered_affine_dir: Output from your current register.py
        output_dir: Where to save deformable results
        method: 'bspline' or 'demons'
    """
    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define study-specific output directory
    study_out = os.path.join(output_dir, study_id)
    
    # Check if study output directory already exists
    if os.path.exists(study_out):
        print(f"⚠️ Study {study_id} already processed at {study_out}, skipping...")
        return

    registrar = DeformableRegistration()
    
    # Find non-contrast (reference)
    nc_row = labels_df[
        (labels_df["StudyInstanceUID"] == study_id) & 
        (labels_df["Label"] == "Non-contrast")
    ]
    
    if nc_row.empty:
        print(f"⚠️ No non-contrast found for {study_id}")
        return
    
    nc_series = nc_row.iloc[0]["SeriesInstanceUID"]
    nc_file = os.path.join(
        registered_affine_dir, 
        f"{study_id}_{nc_series}_registered.nii.gz"
    )
    
    if not os.path.exists(nc_file):
        print(f"⚠️ Reference file not found: {nc_file}")
        return
    # Create study-specific output directory
    os.makedirs(study_out, exist_ok=True)

    fixed = sitk.ReadImage(nc_file)
    
    # Copy reference image
    import shutil
    ref_out = os.path.join(
        study_out,
        f"{study_id}_{nc_series}_deformable.nii.gz"
    )
    shutil.copy(nc_file, ref_out)
    print(f"   ✅ Copied reference image: {ref_out}")
    
    # Copy reference segmentation if it exists
    nc_seg_file = os.path.join(
        registered_affine_dir,
        f"{study_id}_{nc_series}_registered_seg.nii.gz"
    )
    if os.path.exists(nc_seg_file):
        nc_seg_out = os.path.join(
            study_out,
            f"{study_id}_{nc_series}_deformable_seg.nii.gz"
        )
        shutil.copy(nc_seg_file, nc_seg_out)
        print(f"   ✅ Copied reference segmentation: {nc_seg_out}")

    # Process all series
    series_rows = labels_df[labels_df["StudyInstanceUID"] == study_id]
    
    for _, row in series_rows.iterrows():
        series_id = row["SeriesInstanceUID"]
        moving_file = os.path.join(
            registered_affine_dir,
            f"{study_id}_{series_id}_registered.nii.gz"
        )
        
        if not os.path.exists(moving_file) or series_id == nc_series:
            continue
        
        try:
            print(f"\n🔄 Processing {study_id} | {series_id}")
            moving = sitk.ReadImage(moving_file)
            
            # Register with deformable method
            registered, transform, stats = registrar.register_pair(
                fixed, moving, method=method, mesh_size=mesh_size
            )
            
            # Save registered volume
            out_path = os.path.join(
                study_out,
                f"{study_id}_{series_id}_deformable.nii.gz"
            )
            sitk.WriteImage(registered, out_path)
            print(f"   ✅ Saved: {out_path}")
            
            # Save transform - use HDF5 format to support CompositeTransform
            transform_path = os.path.join(
                study_out,
                f"{study_id}_{series_id}_transform.h5"
            )
            try:
                sitk.WriteTransform(transform, transform_path)
                print(f"   ✅ Saved transform: {transform_path}")
            except Exception as e:
                # Fallback: save components separately
                print(f"   ⚠️  HDF5 save failed, saving components...")
                if isinstance(transform, sitk.CompositeTransform):
                    for i in range(transform.GetNumberOfTransforms()):
                        component = transform.GetNthTransform(i)
                        comp_path = os.path.join(
                            study_out,
                            f"{study_id}_{series_id}_transform_comp{i}.tfm"
                        )
                        sitk.WriteTransform(component, comp_path)
                        print(f"      ✅ Saved component {i}")
            
            # Apply to segmentation mask if exists
            seg_file = os.path.join(
                registered_affine_dir,
                f"{study_id}_{series_id}_registered_seg.nii.gz"
            )
            
            if os.path.exists(seg_file):
                seg = sitk.ReadImage(seg_file)
                registered_seg = sitk.Resample(
                    seg, fixed, transform,
                    sitk.sitkNearestNeighbor, 0, seg.GetPixelID()
                )
                
                seg_out = os.path.join(
                    study_out,
                    f"{study_id}_{series_id}_deformable_seg.nii.gz"
                )
                sitk.WriteImage(registered_seg, seg_out)
                print(f"   ✅ Saved mask: {seg_out}")
        
        except Exception as e:
            print(f"❌ Error: {e}")
    
    
    # Save stats
    stats_file = os.path.join(study_out, f"{study_id}_registration_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(registrar.registration_stats, f, indent=2)
    
    print(f"\n✅ Deformable registration completed for {study_id}")


# Usage example
if __name__ == "__main__":
    from configs import MAIN_PATH
    
    labels_csv = MAIN_PATH + "labels.csv"
    labels_df = pd.read_csv(labels_csv)
    
    # Input: your existing affine-registered results
    registered_affine_dir = MAIN_PATH + "registered_cases"
    method = "bspline"
    # Output: deformable registration results
    deformable_output = MAIN_PATH + "deformable_registered_" + method
    os.makedirs(deformable_output, exist_ok=True)
    
    # Register each study with B-spline deformable
    for study_id in labels_df["StudyInstanceUID"].unique():
        study_affine = os.path.join(registered_affine_dir, study_id)
        
        register_study_deformable(
            study_id,
            study_affine,
            deformable_output,
            labels_df,
            method=method,  # 'demons' or 'bspline'
            mesh_size=8  # 5-8 for aggressive, 10-15 for subtle
        )
    
    print("\n✅ All studies processed with deformable registration")