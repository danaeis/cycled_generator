"""
Preprocessing Techniques That Actually Help CT Registration
============================================================

YOUR QUESTION: Can we normalize/adjust HU values to help registration algorithms?

SHORT ANSWER: YES! But not CLAHE. Use these instead:
1. HU windowing (clip to body range)
2. Intensity standardization (z-score normalization)
3. Histogram matching between phases
4. N4 bias field correction (for non-uniform intensity)

Let's explore each...
"""

import SimpleITK as sitk
import numpy as np
from typing import Tuple, List


# =============================================================================
# TECHNIQUE 1: SMART HU WINDOWING (RECOMMENDED!)
# =============================================================================

def apply_body_windowing(
    volume_sitk: sitk.Image,
    body_threshold: int = -500,
    percentile_clip: Tuple[float, float] = (1, 99)
) -> sitk.Image:
    """
    Clip HU values to relevant body range while preserving relationships.
    
    This HELPS registration by:
    - Removing extreme outliers (air, metal artifacts)
    - Reducing dynamic range to focus on soft tissue
    - Still preserves tissue contrast relationships
    
    Args:
        volume_sitk: Input volume in HU
        body_threshold: Threshold to identify body (-500 HU typical)
        percentile_clip: Clip to these percentiles of body values
        
    Returns:
        Volume with clipped HU values (still in HU units!)
    """
    vol_np = sitk.GetArrayFromImage(volume_sitk)
    
    # Find body region
    body_mask = vol_np > body_threshold
    
    if body_mask.sum() < 1000:
        # Mostly empty - use default range
        w_min, w_max = -1000, 600
    else:
        # Get body values only
        body_values = vol_np[body_mask]
        
        # Clip to percentiles
        w_min = np.percentile(body_values, percentile_clip[0])
        w_max = np.percentile(body_values, percentile_clip[1])
        
        # Expand slightly to avoid hard clipping
        iqr = np.percentile(body_values, 75) - np.percentile(body_values, 25)
        margin = max(int(1.5 * iqr), 100)
        
        w_min = max(w_min - margin, -1500)
        w_max = min(w_max + margin, 2000)
    
    # Clip to window (but keep HU units!)
    vol_clipped = np.clip(vol_np, w_min, w_max)
    
    # Convert to Float32 for registration
    vol_f32 = vol_clipped.astype(np.float32)
    
    result = sitk.GetImageFromArray(vol_f32)
    result.CopyInformation(volume_sitk)
    
    print(f"   ✓ Body windowing: [{w_min:.0f}, {w_max:.0f}] HU")
    print(f"   ✓ Removed {np.sum(vol_np < w_min)} low outliers")
    print(f"   ✓ Removed {np.sum(vol_np > w_max)} high outliers")
    
    return result


# =============================================================================
# TECHNIQUE 2: INTENSITY STANDARDIZATION (HELPS A LOT!)
# =============================================================================

def apply_intensity_standardization(
    volume_sitk: sitk.Image,
    body_threshold: int = -500
) -> sitk.Image:
    """
    Standardize intensity distribution (z-score normalization).
    
    This HELPS registration by:
    - Making intensity distributions comparable across phases
    - Handling dose/scanner variations
    - Improving Mutual Information convergence
    
    Formula: (HU - mean) / std
    
    This is DIFFERENT from CLAHE because:
    - Global transformation (not local)
    - Preserves relative relationships
    - Linear transformation (predictable)
    """
    vol_np = sitk.GetArrayFromImage(volume_sitk)
    
    # Get body statistics
    body_mask = vol_np > body_threshold
    
    if body_mask.sum() < 1000:
        mean_hu = 0
        std_hu = 100
    else:
        body_values = vol_np[body_mask]
        mean_hu = body_values.mean()
        std_hu = body_values.std()
    
    # Z-score normalization
    vol_norm = (vol_np - mean_hu) / (std_hu + 1e-8)
    
    # Clip extreme outliers (beyond 5 standard deviations)
    vol_norm = np.clip(vol_norm, -5, 5)
    
    result = sitk.GetImageFromArray(vol_norm.astype(np.float32))
    result.CopyInformation(volume_sitk)
    
    print(f"   ✓ Standardized: mean={mean_hu:.1f}, std={std_hu:.1f}")
    print(f"   ✓ New range: [{vol_norm.min():.2f}, {vol_norm.max():.2f}]")
    
    return result


# =============================================================================
# TECHNIQUE 3: HISTOGRAM MATCHING (ADVANCED)
# =============================================================================

def apply_histogram_matching(
    moving_sitk: sitk.Image,
    fixed_sitk: sitk.Image,
    num_bins: int = 256,
    num_match_points: int = 10
) -> sitk.Image:
    """
    Match histogram of moving image to fixed image.
    
    This HELPS registration by:
    - Making intensity distributions identical
    - Excellent for MI-based registration
    - Handles contrast differences between phases
    
    SimpleITK has built-in histogram matching filter!
    """
    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(num_bins)
    matcher.SetNumberOfMatchPoints(num_match_points)
    matcher.ThresholdAtMeanIntensityOn()
    
    matched = matcher.Execute(moving_sitk, fixed_sitk)
    
    print(f"   ✓ Histogram matched to reference")
    
    return matched


# =============================================================================
# TECHNIQUE 4: ADAPTIVE HISTOGRAM EQUALIZATION (ALTERNATIVE TO CLAHE)
# =============================================================================

def apply_simple_histogram_equalization(
    volume_sitk: sitk.Image,
    body_threshold: int = -500
) -> sitk.Image:
    """
    Simple global histogram equalization (NOT CLAHE).
    
    This is BETTER than CLAHE for registration because:
    - Global transformation (consistent)
    - Still preserves monotonic relationships
    - But generally NOT needed if you use techniques 1-3
    
    Only use if your phases have very different contrast.
    """
    vol_np = sitk.GetArrayFromImage(volume_sitk)
    
    # Find body region
    body_mask = vol_np > body_threshold
    
    if body_mask.sum() < 1000:
        return volume_sitk
    
    body_values = vol_np[body_mask]
    
    # Compute CDF
    hist, bins = np.histogram(body_values.ravel(), bins=1000)
    cdf = hist.cumsum()
    cdf = cdf / cdf[-1]  # Normalize
    
    # Interpolate to get new values
    vol_eq = np.interp(vol_np.ravel(), bins[:-1], cdf)
    vol_eq = vol_eq.reshape(vol_np.shape)
    
    # Scale to reasonable range
    vol_eq = vol_eq * 1000 - 500  # Roughly -500 to 500
    
    result = sitk.GetImageFromArray(vol_eq.astype(np.float32))
    result.CopyInformation(volume_sitk)
    
    print(f"   ✓ Global histogram equalization applied")
    
    return result


# =============================================================================
# TECHNIQUE 5: N4 BIAS FIELD CORRECTION
# =============================================================================

def apply_bias_field_correction(
    volume_sitk: sitk.Image,
    shrink_factor: int = 4
) -> sitk.Image:
    """
    Correct for intensity non-uniformity (bias field).
    
    This HELPS registration when:
    - There's shading across the image
    - Scanner artifacts present
    - Non-uniform intensity distribution
    
    Typically more useful for MRI than CT, but can help CT too.
    """
    # Convert to proper type
    if volume_sitk.GetPixelID() != sitk.sitkFloat32:
        volume_sitk = sitk.Cast(volume_sitk, sitk.sitkFloat32)
    
    # Shrink for speed
    if shrink_factor > 1:
        volume_shrunk = sitk.Shrink(
            volume_sitk, 
            [shrink_factor] * volume_sitk.GetDimension()
        )
    else:
        volume_shrunk = volume_sitk
    
    # Run N4 correction
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])
    
    corrected_shrunk = corrector.Execute(volume_shrunk)
    
    # Resample back to original size
    if shrink_factor > 1:
        corrected = sitk.Resample(
            corrected_shrunk,
            volume_sitk,
            sitk.Transform(),
            sitk.sitkLinear,
            0.0,
            volume_sitk.GetPixelID()
        )
    else:
        corrected = corrected_shrunk
    
    print(f"   ✓ Bias field correction applied")
    
    return corrected


# =============================================================================
# RECOMMENDED PREPROCESSING PIPELINE
# =============================================================================

def preprocess_for_registration(
    volume_sitk: sitk.Image,
    method: str = "windowing+standardization",
    reference_sitk: sitk.Image = None
) -> sitk.Image:
    """
    Complete preprocessing pipeline for CT registration.
    
    Methods:
        "windowing" - Just clip outliers (safe, recommended)
        "standardization" - Z-score normalization (good for MI)
        "windowing+standardization" - Both (BEST for most cases)
        "histogram_matching" - Match to reference (needs reference)
        "none" - No preprocessing (use raw HU)
    
    Args:
        volume_sitk: Input volume
        method: Preprocessing method
        reference_sitk: Reference volume (for histogram matching)
        
    Returns:
        Preprocessed volume ready for registration
    """
    print(f"\n   🔧 Preprocessing: {method}")
    
    if method == "none":
        # Just convert to Float32
        return sitk.Cast(volume_sitk, sitk.sitkFloat32)
    
    elif method == "windowing":
        # Clip outliers only
        return apply_body_windowing(volume_sitk)
    
    elif method == "standardization":
        # Z-score normalization
        return apply_intensity_standardization(volume_sitk)
    
    elif method == "windowing+standardization":
        # Recommended combination
        vol = apply_body_windowing(volume_sitk)
        vol = apply_intensity_standardization(vol)
        return vol
    
    elif method == "histogram_matching":
        if reference_sitk is None:
            raise ValueError("Reference image required for histogram matching")
        # First window both
        vol = apply_body_windowing(volume_sitk)
        ref = apply_body_windowing(reference_sitk)
        # Then match
        return apply_histogram_matching(vol, ref)
    
    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# COMPARISON: EFFECT ON REGISTRATION
# =============================================================================

def compare_preprocessing_methods(
    fixed_path: str,
    moving_path: str,
    output_dir: str
):
    """
    Test different preprocessing methods and compare registration quality.
    """
    import os
    
    print("\n" + "="*80)
    print("TESTING PREPROCESSING METHODS")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load volumes
    fixed = sitk.ReadImage(fixed_path)
    moving = sitk.ReadImage(moving_path)
    
    methods = [
        "none",
        "windowing",
        "standardization",
        "windowing+standardization",
        "histogram_matching"
    ]
    
    results = {}
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Testing: {method}")
        print(f"{'='*60}")
        
        # Preprocess
        if method == "histogram_matching":
            fixed_prep = preprocess_for_registration(fixed, "windowing")
            moving_prep = preprocess_for_registration(
                moving, method, reference_sitk=fixed_prep
            )
        else:
            fixed_prep = preprocess_for_registration(fixed, method)
            moving_prep = preprocess_for_registration(moving, method)
        
        # Quick registration test
        registration = sitk.ImageRegistrationMethod()
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration.SetMetricSamplingPercentage(0.1)
        registration.SetInterpolator(sitk.sitkLinear)
        
        registration.SetOptimizerAsGradientDescent(
            learningRate=1.0,
            numberOfIterations=100,
            convergenceMinimumValue=1e-6
        )
        
        initial_transform = sitk.Euler3DTransform()
        registration.SetInitialTransform(initial_transform)
        
        try:
            final_transform = registration.Execute(fixed_prep, moving_prep)
            final_metric = registration.GetMetricValue()
            iterations = registration.GetOptimizerIteration()
            
            results[method] = {
                'metric': final_metric,
                'iterations': iterations,
                'success': True
            }
            
            print(f"   ✓ Metric: {final_metric:.6f}")
            print(f"   ✓ Iterations: {iterations}")
            
        except Exception as e:
            print(f"   ✗ Failed: {e}")
            results[method] = {
                'metric': None,
                'iterations': None,
                'success': False
            }
    
    # Summary
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Method':<30} {'Metric':<15} {'Iterations':<10} {'Status'}")
    print("-"*80)
    
    for method, data in results.items():
        if data['success']:
            metric_str = f"{data['metric']:.6f}"
            iter_str = str(data['iterations'])
            status = "✓"
        else:
            metric_str = "FAILED"
            iter_str = "-"
            status = "✗"
        
        print(f"{method:<30} {metric_str:<15} {iter_str:<10} {status}")
    
    # Find best
    successful = {k: v for k, v in results.items() if v['success']}
    if successful:
        best = min(successful.items(), key=lambda x: x[1]['metric'])
        print(f"\n✓ Best method: {best[0]} (metric: {best[1]['metric']:.6f})")


from configs import MAIN_PATH
test_cropped = MAIN_PATH + "test_cropped"