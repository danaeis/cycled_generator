"""
Quick Test: Compare Preprocessing Methods on Single Study
==========================================================

This is a simplified version for quick testing.
Use this to test on ONE study before running the full batch.

Usage:
    python quick_preprocessing_test.py --study-id "YOUR_STUDY_ID"

Or edit the STUDY_ID variable below and run:
    python quick_preprocessing_test.py
"""

import os
import json
import time
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


# =============================================================================
# QUICK CONFIGURATION
# =============================================================================

# Edit these to match your setup
MAIN_PATH = "../ncct_cect/vindr_ds/"
INPUT_DIR = Path(MAIN_PATH + "cropped_volumes")
OUTPUT_DIR = Path(MAIN_PATH + "quick_test_results")

# Study to test (or pass as command line argument)
STUDY_ID = "1.2.840.113619.2.359.3.2831208971.47.1588634888.707"

# Which methods to compare
METHODS = [
    "none",                      # Baseline
    "windowing",                 # Remove outliers
    "standardization",           # Z-score
    "windowing+standardization", # Recommended
    "histogram_matching"         # Best for multi-phase
]


# =============================================================================
# PREPROCESSING FUNCTIONS
# =============================================================================

def apply_body_windowing(volume_sitk):
    """Clip HU to body range."""
    vol_np = sitk.GetArrayFromImage(volume_sitk)
    body_mask = vol_np > -500
    
    if body_mask.sum() > 1000:
        body_values = vol_np[body_mask]
        w_min = np.percentile(body_values, 1)
        w_max = np.percentile(body_values, 99)
        iqr = np.percentile(body_values, 75) - np.percentile(body_values, 25)
        margin = max(int(1.5 * iqr), 100)
        w_min = max(w_min - margin, -1500)
        w_max = min(w_max + margin, 2000)
    else:
        w_min, w_max = -1000, 600
    
    vol_clipped = np.clip(vol_np, w_min, w_max).astype(np.float32)
    
    result = sitk.GetImageFromArray(vol_clipped)
    result.CopyInformation(volume_sitk)
    return result


def apply_standardization(volume_sitk):
    """Z-score normalization."""
    vol_np = sitk.GetArrayFromImage(volume_sitk)
    body_mask = vol_np > -500
    
    if body_mask.sum() > 1000:
        body_values = vol_np[body_mask]
        mean_hu = body_values.mean()
        std_hu = body_values.std()
    else:
        mean_hu, std_hu = 0, 100
    
    vol_norm = (vol_np - mean_hu) / (std_hu + 1e-8)
    vol_norm = np.clip(vol_norm, -5, 5).astype(np.float32)
    
    result = sitk.GetImageFromArray(vol_norm)
    result.CopyInformation(volume_sitk)
    return result


def apply_histogram_matching(moving_sitk, fixed_sitk):
    """Match histogram."""
    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(256)
    matcher.SetNumberOfMatchPoints(10)
    matcher.ThresholdAtMeanIntensityOn()
    return matcher.Execute(moving_sitk, fixed_sitk)


def preprocess(volume_sitk, method, reference_sitk=None):
    """Apply preprocessing method."""
    if method == "none":
        return sitk.Cast(volume_sitk, sitk.sitkFloat32)
    elif method == "windowing":
        return apply_body_windowing(volume_sitk)
    elif method == "standardization":
        return apply_standardization(volume_sitk)
    elif method == "windowing+standardization":
        vol = apply_body_windowing(volume_sitk)
        return apply_standardization(vol)
    elif method == "histogram_matching":
        vol = apply_body_windowing(volume_sitk)
        ref = apply_body_windowing(reference_sitk)
        return apply_histogram_matching(vol, ref)
    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# REGISTRATION
# =============================================================================

def register_and_measure(fixed, moving):
    """Register and return metrics."""
    
    # Cast to Float32
    fixed = sitk.Cast(fixed, sitk.sitkFloat32)
    moving = sitk.Cast(moving, sitk.sitkFloat32)
    
    # Setup registration
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.20)
    registration.SetInterpolator(sitk.sitkLinear)
    
    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=200,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    
    # Initial transform
    try:
        initial_transform = sitk.CenteredTransformInitializer(
            fixed, moving,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
    except:
        initial_transform = sitk.Euler3DTransform()
    
    registration.SetInitialTransform(initial_transform, inPlace=False)
    
    # Register
    start_time = time.time()
    final_transform = registration.Execute(fixed, moving)
    reg_time = time.time() - start_time
    
    # Apply transform
    registered = sitk.Resample(
        moving, fixed, final_transform,
        sitk.sitkLinear, 0.0, moving.GetPixelID()
    )
    
    # Compute metrics
    fixed_np = sitk.GetArrayFromImage(fixed)
    reg_np = sitk.GetArrayFromImage(registered)
    
    body_mask = fixed_np > -500
    if body_mask.sum() > 1000:
        fixed_body = fixed_np[body_mask]
        reg_body = reg_np[body_mask]
        
        mae = float(np.abs(fixed_body - reg_body).mean())
        
        fixed_norm = (fixed_body - fixed_body.mean()) / (fixed_body.std() + 1e-8)
        reg_norm = (reg_body - reg_body.mean()) / (reg_body.std() + 1e-8)
        ncc = float(np.mean(fixed_norm * reg_norm))
    else:
        mae = None
        ncc = None
    
    # SSIM on middle slice
    try:
        mid = fixed_np.shape[0] // 2
        data_range = fixed_np.max() - fixed_np.min()
        ssim_val = ssim(fixed_np[mid], reg_np[mid], data_range=data_range)
    except:
        ssim_val = None
    
    return {
        'final_metric': float(registration.GetMetricValue()),
        'iterations': int(registration.GetOptimizerIteration()),
        'time': reg_time,
        'mae': mae,
        'ncc': ncc,
        'ssim': ssim_val,
        'registered': registered
    }


# =============================================================================
# MAIN TEST
# =============================================================================

def run_quick_test(study_id):
    """Run quick test on one study."""
    
    print("\n" + "="*80)
    print(f"QUICK PREPROCESSING TEST")
    print("="*80)
    print(f"Study: {study_id}")
    print(f"Methods: {', '.join(METHODS)}")
    print("="*80)
    
    # Find study directory
    study_dir = INPUT_DIR / study_id
    if not study_dir.exists():
        print(f"\n❌ Study directory not found: {study_dir}")
        print(f"\nAvailable studies:")
        for d in INPUT_DIR.iterdir():
            if d.is_dir():
                print(f"  - {d.name}")
        return
    
    # Find volumes
    volumes = list(study_dir.glob("*_cropped.nii.gz"))
    if len(volumes) == 0:
        print(f"\n❌ No cropped volumes found in {study_dir}")
        return
    
    print(f"\nFound {len(volumes)} volumes:")
    for v in volumes:
        print(f"  - {v.name}")
    
    if len(volumes) < 2:
        print(f"\n❌ Need at least 2 volumes (fixed + moving)")
        return
    
    # Use first as fixed, second as moving (simplified)
    fixed_path = volumes[0]
    moving_path = volumes[1]
    
    print(f"\nUsing:")
    print(f"  Fixed:  {fixed_path.name}")
    print(f"  Moving: {moving_path.name}")
    
    # Load volumes
    print(f"\nLoading volumes...")
    fixed_original = sitk.ReadImage(str(fixed_path))
    moving_original = sitk.ReadImage(str(moving_path))
    
    # Create output directory
    output_dir = OUTPUT_DIR / study_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test each method
    results = {}
    
    for method in METHODS:
        print(f"\n{'─'*60}")
        print(f"Testing: {method}")
        print(f"{'─'*60}")
        
        try:
            # Preprocess
            print(f"  Preprocessing...")
            fixed_prep = preprocess(fixed_original, method)
            moving_prep = preprocess(moving_original, method, reference_sitk=fixed_prep)
            
            # Save preprocessed (for inspection)
            method_dir = output_dir / method
            method_dir.mkdir(exist_ok=True)
            
            sitk.WriteImage(fixed_prep, str(method_dir / "fixed_preprocessed.nii.gz"))
            sitk.WriteImage(moving_prep, str(method_dir / "moving_preprocessed.nii.gz"))
            
            # Register
            print(f"  Registering...")
            metrics = register_and_measure(fixed_prep, moving_prep)
            
            # Save registered
            sitk.WriteImage(metrics['registered'], str(method_dir / "registered.nii.gz"))
            
            results[method] = metrics
            
            # Print metrics
            print(f"  ✓ Complete!")
            print(f"    Final metric: {metrics['final_metric']:.6f}")
            print(f"    Time:         {metrics['time']:.2f}s")
            print(f"    Iterations:   {metrics['iterations']}")
            if metrics['mae'] is not None:
                print(f"    MAE:          {metrics['mae']:.2f}")
                print(f"    NCC:          {metrics['ncc']:.4f}")
            if metrics['ssim'] is not None:
                print(f"    SSIM:         {metrics['ssim']:.4f}")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results[method] = {'error': str(e)}
    
    # Create comparison
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")
    
    print(f"\n{'Method':<30} {'Metric':<12} {'Time':<8} {'MAE':<8} {'NCC':<8}")
    print("─"*80)
    
    for method in METHODS:
        if method in results and 'error' not in results[method]:
            r = results[method]
            print(f"{method:<30} {r['final_metric']:>11.6f} {r['time']:>7.2f}s "
                  f"{r['mae']:>7.2f} {r['ncc']:>7.4f}")
        else:
            print(f"{method:<30} {'FAILED'}")
    
    # Find best
    successful = {k: v for k, v in results.items() if 'error' not in v}
    
    if successful:
        best_metric = min(successful.items(), key=lambda x: x[1]['final_metric'])
        best_mae = min(successful.items(), key=lambda x: x[1]['mae'] if x[1]['mae'] else float('inf'))
        best_ncc = max(successful.items(), key=lambda x: x[1]['ncc'] if x[1]['ncc'] else -float('inf'))
        
        print(f"\n{'='*80}")
        print("BEST METHODS")
        print(f"{'='*80}")
        print(f"  By Final Metric: {best_metric[0]}")
        print(f"  By MAE:          {best_mae[0]}")
        print(f"  By NCC:          {best_ncc[0]}")
    
    # Create plot
    create_comparison_plot(results, output_dir)
    
    # Save results
    results_clean = {}
    for method, data in results.items():
        if 'registered' in data:
            del data['registered']  # Can't serialize SimpleITK images
        results_clean[method] = data
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results_clean, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_dir}")
    print(f"✓ Preprocessed volumes: {output_dir}/{method}/")
    print(f"✓ Comparison plot: {output_dir}/comparison.png")


def create_comparison_plot(results, output_dir):
    """Create simple comparison plot."""
    
    # Filter successful results
    successful = {k: v for k, v in results.items() if 'error' not in v}
    
    if len(successful) == 0:
        return
    
    methods = list(successful.keys())
    final_metrics = [successful[m]['final_metric'] for m in methods]
    maes = [successful[m]['mae'] for m in methods]
    nccs = [successful[m]['ncc'] for m in methods]
    times = [successful[m]['time'] for m in methods]
    
    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Final Metric
    axes[0, 0].bar(methods, final_metrics, color='lightblue')
    axes[0, 0].set_ylabel('Final Metric\n(lower is better)')
    axes[0, 0].set_title('Registration Quality')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # MAE
    axes[0, 1].bar(methods, maes, color='lightcoral')
    axes[0, 1].set_ylabel('MAE\n(lower is better)')
    axes[0, 1].set_title('Mean Absolute Error')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # NCC
    axes[1, 0].bar(methods, nccs, color='lightgreen')
    axes[1, 0].set_ylabel('NCC\n(higher is better)')
    axes[1, 0].set_title('Normalized Cross Correlation')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Time
    axes[1, 1].bar(methods, times, color='wheat')
    axes[1, 1].set_ylabel('Time (seconds)')
    axes[1, 1].set_title('Computational Time')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.suptitle('Preprocessing Methods Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    plt.savefig(output_dir / 'comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n  ✓ Plot saved: {output_dir}/comparison.png")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys
    
    # Check for command line argument
    if len(sys.argv) > 1:
        study_id = sys.argv[1]
    else:
        study_id = STUDY_ID
    
    run_quick_test(study_id)
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print(f"\n1. Check preprocessed volumes: {OUTPUT_DIR}/{study_id}/{{method}}/")
    print(f"2. View comparison plot: {OUTPUT_DIR}/{study_id}/comparison.png")
    print(f"3. If satisfied, run full batch: preprocessing_comparison_pipeline.py")