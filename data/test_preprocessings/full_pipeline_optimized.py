"""
Abdominal CT Preprocessing Comparison - FIXED FOR ORGANS
==========================================================

FIXES:
1. Organ-specific HU windows (not bone-focused)
2. Histogram matching bug fixed
3. Abdominal-appropriate thresholds
4. Better handling of contrast-enhanced phases
"""

import os
import json
import time
import pandas as pd
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration for abdominal organ preprocessing comparison."""
    
    # Paths
    MAIN_PATH = Path("../ncct_cect/vindr_ds/")
    INPUT_DIR = MAIN_PATH / "cropped_volumes"  
    OUTPUT_BASE_DIR = MAIN_PATH / "abdominal_preprocessing_comparison"
    LABELS_CSV = MAIN_PATH / "labels.csv"
    
    # Preprocessing methods to test
    METHODS = [
        "none",                          # Baseline
        "abdominal_windowing",           # Organ-focused HU window
        "soft_tissue_windowing",         # Soft tissue emphasis
        "standardization",               # Z-score
        "adaptive_windowing",            # Adaptive based on contrast
        "abdominal_windowing+standardization",  # Combined
        "histogram_matching"             # Histogram matched
    ]
    
    # ABDOMINAL-SPECIFIC SETTINGS (CRITICAL!)
    # For cropped abdomen (liver to bladder):
    ORGAN_THRESHOLD = -100  # Changed from -500/-600!
    # Abdominal organs: liver (~50), spleen (~50), kidneys (~40), pancreas (~40)
    # Fat: -100 to -50
    # Contrast-enhanced: 100-300
    
    # Alignment settings
    MIN_OVERLAP_SLICES = 30
    SEARCH_REGION = 'upper'  # Or 'middle' for full abdomen
    
    # Registration settings
    NUM_ITERATIONS = 200
    LEARNING_RATE = 1.0
    SAMPLING_PERCENTAGE = 0.20
    
    # How many studies to test
    MAX_STUDIES = 3


# =============================================================================
# ABDOMINAL-SPECIFIC PREPROCESSING FUNCTIONS
# =============================================================================

def apply_abdominal_windowing(volume_sitk: sitk.Image) -> sitk.Image:
    """
    Apply abdominal organ window.
    
    Standard abdominal window: [-160, 240] HU
    - Includes: fat, soft tissue organs, contrast-enhanced vessels
    - Excludes: air, bone (not needed for organ alignment)
    """
    vol_np = sitk.GetArrayFromImage(volume_sitk)
    
    # Clinical abdominal window
    W_MIN = -160  # Include fat
    W_MAX = 240   # Include enhanced vessels
    
    vol_windowed = np.clip(vol_np, W_MIN, W_MAX).astype(np.float32)
    
    result = sitk.GetImageFromArray(vol_windowed)
    result.CopyInformation(volume_sitk)
    # return result
    return volume_sitk


def apply_soft_tissue_windowing(volume_sitk: sitk.Image) -> sitk.Image:
    """
    Soft tissue window for organ detail.
    
    Window: [-50, 150] HU
    - Optimized for: liver, spleen, kidneys, pancreas
    - Narrower than general abdomen (better organ contrast)
    """
    vol_np = sitk.GetArrayFromImage(volume_sitk)
    
    W_MIN = -50   # Exclude most fat
    W_MAX = 150   # Organs + mild enhancement
    
    vol_windowed = np.clip(vol_np, W_MIN, W_MAX).astype(np.float32)
    
    result = sitk.GetImageFromArray(vol_windowed)
    result.CopyInformation(volume_sitk)
    return result


def apply_adaptive_windowing(volume_sitk: sitk.Image, organ_threshold: int = -100) -> sitk.Image:
    """
    Adaptive windowing based on actual organ HU distribution.
    
    Uses organ region statistics to set optimal window.
    """
    vol_np = sitk.GetArrayFromImage(volume_sitk)
    
    # Get organ region (exclude fat and air)
    organ_mask = (vol_np > organ_threshold) & (vol_np < 300)
    
    if organ_mask.sum() > 1000:
        organ_values = vol_np[organ_mask]
        
        # Use 1st-99th percentile of organs
        w_min = np.percentile(organ_values, 1)
        w_max = np.percentile(organ_values, 99)
        
        # Add margin for contrast variation
        margin = (w_max - w_min) * 0.2
        w_min = max(w_min - margin, -200)
        w_max = min(w_max + margin, 400)
    else:
        # Fallback to standard abdominal window
        w_min, w_max = -160, 240
    
    vol_windowed = np.clip(vol_np, w_min, w_max).astype(np.float32)
    
    result = sitk.GetImageFromArray(vol_windowed)
    result.CopyInformation(volume_sitk)
    return result


def apply_standardization(volume_sitk: sitk.Image, organ_threshold: int = -100) -> sitk.Image:
    """
    Z-score normalization using organ region.
    
    Uses organ HU values (not all voxels) for mean/std.
    """
    vol_np = sitk.GetArrayFromImage(volume_sitk)
    
    # Get organ region
    organ_mask = (vol_np > organ_threshold) & (vol_np < 300)
    
    if organ_mask.sum() > 1000:
        organ_values = vol_np[organ_mask]
        mean_hu = organ_values.mean()
        std_hu = organ_values.std()
    else:
        # Fallback
        mean_hu = 50  # Typical organ HU
        std_hu = 30
    
    vol_norm = (vol_np - mean_hu) / (std_hu + 1e-8)
    vol_norm = np.clip(vol_norm, -5, 5).astype(np.float32)
    
    result = sitk.GetImageFromArray(vol_norm)
    result.CopyInformation(volume_sitk)
    return result


def apply_histogram_matching(moving_sitk: sitk.Image, fixed_sitk: sitk.Image) -> sitk.Image:
    """
    Match histogram of moving to fixed.
    
    Best for multi-phase CT with different contrast levels.
    """
    # Window both first
    moving_windowed = apply_abdominal_windowing(moving_sitk)
    fixed_windowed = apply_abdominal_windowing(fixed_sitk)
    
    # Match
    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(256)
    matcher.SetNumberOfMatchPoints(10)
    matcher.ThresholdAtMeanIntensityOn()
    
    matched = matcher.Execute(moving_windowed, fixed_windowed)
    
    return matched


def preprocess_volume(
    volume_sitk: sitk.Image, 
    method: str, 
    reference_sitk: Optional[sitk.Image] = None
) -> sitk.Image:
    """
    Apply preprocessing method.
    
    FIXED: Properly handles histogram matching with reference.
    """
    if method == "none":
        return sitk.Cast(volume_sitk, sitk.sitkFloat32)
    
    elif method == "abdominal_windowing":
        return apply_abdominal_windowing(volume_sitk)
    
    elif method == "soft_tissue_windowing":
        return apply_soft_tissue_windowing(volume_sitk)
    
    elif method == "adaptive_windowing":
        return apply_adaptive_windowing(volume_sitk, Config.ORGAN_THRESHOLD)
    
    elif method == "standardization":
        return apply_standardization(volume_sitk, Config.ORGAN_THRESHOLD)
    
    elif method == "abdominal_windowing+standardization":
        vol = apply_abdominal_windowing(volume_sitk)
        return apply_standardization(vol, Config.ORGAN_THRESHOLD)
    
    elif method == "histogram_matching":
        if reference_sitk is None:
            #raise ValueError("Reference required for histogram matching")
            return apply_abdominal_windowing(volume_sitk)
        return apply_histogram_matching(apply_abdominal_windowing(volume_sitk), apply_abdominal_windowing(reference_sitk))

    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# ALIGNMENT WITH METRICS (FIXED FOR ORGANS)
# =============================================================================

def find_best_reference_slice(
    reference_vol: np.ndarray, 
    organ_threshold: int = -100,  # Changed!
    search_region: str = 'upper'
) -> int:
    """Find optimal reference slice in abdomen."""
    D = reference_vol.shape[0]
    
    if search_region == 'upper':
        search_start = int(D * 0.65)
        search_end = int(D * 0.90)
    elif search_region == 'middle':
        search_start = int(D * 0.35)
        search_end = int(D * 0.65)
    else:
        search_start = int(D * 0.10)
        search_end = int(D * 0.35)
    
    search_start = max(0, search_start)
    search_end = min(D, search_end)
    
    best_idx = (search_start + search_end) // 2
    best_score = 0
    
    for idx in range(search_start, search_end):
        slice_data = reference_vol[idx]
        
        # Use organ threshold (not body threshold!)
        organ_mask = slice_data > organ_threshold
        
        if organ_mask.sum() < 1000:
            continue
        
        organ_values = slice_data[organ_mask]
        score = organ_mask.sum() * organ_values.std()
        
        if score > best_score:
            best_score = score
            best_idx = idx
    
    return best_idx


def match_single_slice(
    reference_slice: np.ndarray, 
    moving_vol: np.ndarray, 
    search_range: Tuple[int, int],
    organ_threshold: int = -100  # Changed!
) -> Tuple[int, float]:
    """Match reference slice to best slice in moving volume."""
    target_shape = reference_slice.shape
    best_idx = (search_range[0] + search_range[1]) // 2
    best_score = -np.inf
    
    # Normalize reference slice
    ref_norm = (reference_slice - reference_slice.mean()) / (reference_slice.std() + 1e-8)
    
    for m_idx in range(search_range[0], min(search_range[1], moving_vol.shape[0])):
        moving_slice = moving_vol[m_idx]
        
        # Check organ content (not body!)
        if (moving_slice > organ_threshold).sum() < 1000:
            continue
        
        if moving_slice.shape != target_shape:
            import cv2
            moving_slice = cv2.resize(
                moving_slice.astype(np.float32),
                (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
        
        try:
            m_norm = (moving_slice - moving_slice.mean()) / (moving_slice.std() + 1e-8)
            score = np.mean(ref_norm * m_norm)  # NCC
            
            if score > best_score:
                best_score = score
                best_idx = m_idx
        except:
            continue
    
    return best_idx, best_score


def align_study_with_metrics(
    study_id: str,
    series_volumes: Dict[str, sitk.Image],
    reference_series_id: str,
    organ_threshold: int = -100,  # Changed!
    search_region: str = 'upper'
) -> Dict:
    """Align all series with detailed metrics."""
    
    ref_sitk = series_volumes[reference_series_id]
    ref_np = sitk.GetArrayFromImage(ref_sitk)
    
    ref_slice_idx = find_best_reference_slice(ref_np, organ_threshold, search_region)
    ref_slice = ref_np[ref_slice_idx]
    
    results = {
        'study_id': study_id,
        'reference_series': reference_series_id,
        'reference_slice_idx': ref_slice_idx,
        'series_metrics': {},
        'alignment_quality': {}
    }
    
    for series_id, vol_sitk in series_volumes.items():
        if series_id == reference_series_id:
            results['series_metrics'][series_id] = {
                'z_offset': 0,
                'matched_idx': ref_slice_idx,
                'ncc_score': 1.0,
                'is_reference': True
            }
            continue
        
        vol_np = sitk.GetArrayFromImage(vol_sitk)
        D = vol_np.shape[0]
        
        if search_region == 'upper':
            search_start = int(D * 0.50)
            search_end = int(D * 0.95)
        elif search_region == 'middle':
            search_start = int(D * 0.25)
            search_end = int(D * 0.75)
        else:
            search_start = int(D * 0.05)
            search_end = int(D * 0.50)
        
        matched_idx, ncc_score = match_single_slice(
            ref_slice, vol_np,
            search_range=(search_start, search_end),
            organ_threshold=organ_threshold
        )
        
        z_offset = ref_slice_idx - matched_idx
        
        results['series_metrics'][series_id] = {
            'z_offset': int(z_offset),
            'matched_idx': int(matched_idx),
            'ncc_score': float(ncc_score),
            'is_reference': False
        }
    
    ncc_scores = [m['ncc_score'] for m in results['series_metrics'].values() if not m['is_reference']]
    
    if len(ncc_scores) > 0:
        results['alignment_quality'] = {
            'mean_ncc': float(np.mean(ncc_scores)),
            'std_ncc': float(np.std(ncc_scores)),
            'min_ncc': float(np.min(ncc_scores)),
            'max_ncc': float(np.max(ncc_scores)),
            'num_series_aligned': len(ncc_scores)
        }
    
    return results


# =============================================================================
# REGISTRATION WITH METRICS (FIXED)
# =============================================================================

def register_with_metrics(
    fixed: sitk.Image,
    moving: sitk.Image,
    num_iterations: int = 200,
    learning_rate: float = 1.0,
    sampling_percentage: float = 0.20,
    organ_threshold: int = -100  # Changed!
) -> Dict:
    """Run registration with comprehensive metrics."""
    
    metrics = {
        'success': False,
        'error': None,
        'registration_time': 0.0,
        'final_metric': None,
        'iterations_used': 0,
        'initial_metric': None,
        'metric_improvement': None,
        'mae': None,
        'ncc': None,
        'ssim': None
    }
    
    try:
        fixed = sitk.Cast(fixed, sitk.sitkFloat32)
        moving = sitk.Cast(moving, sitk.sitkFloat32)
        
        registration = sitk.ImageRegistrationMethod()
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(sampling_percentage)
        registration.SetInterpolator(sitk.sitkLinear)
        
        registration.SetOptimizerAsGradientDescent(
            learningRate=learning_rate,
            numberOfIterations=num_iterations,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10
        )
        registration.SetOptimizerScalesFromPhysicalShift()
        
        try:
            initial_transform = sitk.CenteredTransformInitializer(
                fixed, moving,
                sitk.Euler3DTransform(),
                sitk.CenteredTransformInitializerFilter.GEOMETRY
            )
        except:
            initial_transform = sitk.Euler3DTransform()
        
        registration.SetInitialTransform(initial_transform, inPlace=False)
        
        try:
            registration.SetMetricSamplingPercentage(0.1)
            initial_metric = registration.MetricEvaluate(fixed, moving)
            metrics['initial_metric'] = float(initial_metric)
            registration.SetMetricSamplingPercentage(sampling_percentage)
        except:
            pass
        
        start_time = time.time()
        final_transform = registration.Execute(fixed, moving)
        metrics['registration_time'] = time.time() - start_time
        
        metrics['final_metric'] = float(registration.GetMetricValue())
        metrics['iterations_used'] = int(registration.GetOptimizerIteration())
        
        if metrics['initial_metric'] is not None:
            metrics['metric_improvement'] = metrics['initial_metric'] - metrics['final_metric']
        
        registered = sitk.Resample(moving, fixed, final_transform, sitk.sitkLinear, 0.0, moving.GetPixelID())
        
        fixed_np = sitk.GetArrayFromImage(fixed)
        reg_np = sitk.GetArrayFromImage(registered)
        
        if fixed_np.shape == reg_np.shape:
            # Use organ mask (not body!)
            organ_mask = fixed_np > organ_threshold
            
            if organ_mask.sum() > 1000:
                fixed_organ = fixed_np[organ_mask]
                reg_organ = reg_np[organ_mask]
                
                metrics['mae'] = float(np.abs(fixed_organ - reg_organ).mean())
                
                fixed_norm = (fixed_organ - fixed_organ.mean()) / (fixed_organ.std() + 1e-8)
                reg_norm = (reg_organ - reg_organ.mean()) / (reg_organ.std() + 1e-8)
                metrics['ncc'] = float(np.mean(fixed_norm * reg_norm))
            
            try:
                mid = fixed_np.shape[0] // 2
                data_range = fixed_np.max() - fixed_np.min()
                metrics['ssim'] = float(ssim(fixed_np[mid], reg_np[mid], data_range=data_range))
            except:
                pass
        
        metrics['success'] = True
        
    except Exception as e:
        metrics['error'] = str(e)
        metrics['success'] = False
    
    return metrics


# =============================================================================
# FULL PIPELINE (FIXED)
# =============================================================================

def run_full_pipeline_single_study(
    study_id: str,
    study_dir: Path,
    labels_df: pd.DataFrame,
    method: str,
    output_dir: Path
) -> Dict:
    """Run complete pipeline with abdominal preprocessing."""
    
    print(f"\n  {'─'*60}")
    print(f"  Method: {method}")
    print(f"  {'─'*60}")
    
    # Find series
    volume_files = list(study_dir.glob(f"{study_id}_*_crop.nii.gz"))
    
    if len(volume_files) == 0:
        return {'error': 'No volumes found', 'success': False}
    
    # Map to phases
    series_info = {}
    for vol_path in volume_files:
        basename = vol_path.name
        series_id = basename.replace(f"{study_id}_", "").replace("_crop.nii.gz", "")
        
        # Find phase
        phase = "Unknown"
        study_labels = labels_df[labels_df["StudyInstanceUID"] == study_id]
        for _, row in study_labels.iterrows():
            if row["SeriesInstanceUID"] == series_id:
                phase = row["Label"]
                break
        
        series_info[series_id] = {
            'path': vol_path,
            'phase': phase
        }
    
    # Find non-contrast
    nc_series = None
    for series_id, info in series_info.items():
        if info['phase'] == 'Non-contrast':
            nc_series = series_id
            break
    
    if nc_series is None:
        return {'error': 'No non-contrast series', 'success': False}
    
    if len(series_info) < 2:
        return {'error': f'Only {len(series_info)} series', 'success': False}
    
    print(f"    Found {len(series_info)} series (reference: {series_info[nc_series]['phase']})")
    
    # Create output directory
    method_dir = output_dir / method
    method_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'study_id': study_id,
        'method': method,
        'num_series': len(series_info),
        'reference_series': nc_series
    }
    
    try:
        # STEP 1: PREPROCESSING
        print(f"    [1/3] Preprocessing...")
        
        # Load all volumes
        original_volumes = {}
        for series_id, info in series_info.items():
            original_volumes[series_id] = sitk.ReadImage(str(info['path']))
        
        # Preprocess - FIXED: Pass reference for histogram matching!
        preprocessed_volumes = {}
        ref_vol = original_volumes[nc_series]
        
        for series_id, vol in original_volumes.items():
            if series_id == nc_series:
                # Process reference without needing another reference
                if method == "histogram_matching":
                    # For reference, just window it
                    preprocessed_volumes[series_id] = apply_abdominal_windowing(vol)
                else:
                    preprocessed_volumes[series_id] = preprocess_volume(vol, method)
            else:
                # Pass reference for histogram matching
                preprocessed_volumes[series_id] = preprocess_volume(
                    vol, method, reference_sitk=ref_vol
                )
            
            # Save preprocessed
            prep_path = method_dir / f"{series_id}_preprocessed.nii.gz"
            sitk.WriteImage(preprocessed_volumes[series_id], str(prep_path))
        
        print(f"        ✓ Preprocessed {len(preprocessed_volumes)} volumes")
        
        # STEP 2: ALIGNMENT
        print(f"    [2/3] Z-Alignment...")
        
        alignment_start = time.time()
        alignment_results = align_study_with_metrics(
            study_id=study_id,
            series_volumes=preprocessed_volumes,
            reference_series_id=nc_series,
            organ_threshold=Config.ORGAN_THRESHOLD,
            search_region=Config.SEARCH_REGION
        )
        alignment_time = time.time() - alignment_start
        
        results['alignment'] = alignment_results['alignment_quality']
        results['alignment']['time'] = alignment_time
        results['alignment']['series_details'] = alignment_results['series_metrics']
        
        print(f"        ✓ Aligned: mean_ncc={results['alignment']['mean_ncc']:.4f}")
        
        # STEP 3: REGISTRATION
        print(f"    [3/3] Rigid Registration...")
        
        registration_results = {}
        
        fixed_prep = preprocessed_volumes[nc_series]
        
        for series_id, info in series_info.items():
            if series_id == nc_series:
                continue
            
            moving_prep = preprocessed_volumes[series_id]
            
            reg_metrics = register_with_metrics(
                fixed_prep, moving_prep,
                num_iterations=Config.NUM_ITERATIONS,
                learning_rate=Config.LEARNING_RATE,
                sampling_percentage=Config.SAMPLING_PERCENTAGE,
                organ_threshold=Config.ORGAN_THRESHOLD
            )
            
            registration_results[series_id] = {
                'phase': info['phase'],
                'metrics': reg_metrics
            }
        
        results['registration'] = registration_results
        
        successful_regs = [
            r['metrics'] for r in registration_results.values() 
            if r['metrics']['success']
        ]
        
        if len(successful_regs) > 0:
            results['registration_summary'] = {
                'mean_final_metric': float(np.mean([r['final_metric'] for r in successful_regs])),
                'mean_mae': float(np.mean([r['mae'] for r in successful_regs if r['mae'] is not None])),
                'mean_ncc': float(np.mean([r['ncc'] for r in successful_regs if r['ncc'] is not None])),
                'mean_time': float(np.mean([r['registration_time'] for r in successful_regs])),
                'num_successful': len(successful_regs)
            }
            
            print(f"        ✓ Registered {len(successful_regs)} series")
            print(f"          Final metric: {results['registration_summary']['mean_final_metric']:.6f}")
            print(f"          MAE: {results['registration_summary']['mean_mae']:.2f}")
        
        results['success'] = True
        
    except Exception as e:
        print(f"        ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        results['error'] = str(e)
        results['success'] = False
    
    with open(method_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


# =============================================================================
# REST OF FUNCTIONS (copy from original, no changes needed)
# =============================================================================

def process_study_all_methods(
    study_id: str,
    input_dir: Path,
    output_dir: Path,
    labels_df: pd.DataFrame,
    methods: List[str]
) -> Dict:
    """Process one study with all preprocessing methods."""
    
    print(f"\n{'='*80}")
    print(f"Study: {study_id}")
    print(f"{'='*80}")
    
    study_dir = input_dir / study_id
    
    if not study_dir.exists():
        print(f"  ✗ Directory not found: {study_dir}")
        return None
    
    study_output_dir = output_dir / study_id
    study_output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for method in methods:
        result = run_full_pipeline_single_study(
            study_id, study_dir, labels_df, method, study_output_dir
        )
        all_results[method] = result
    
    with open(study_output_dir / 'all_methods_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    create_study_comparison_table(all_results, study_output_dir)
    
    return all_results


def create_study_comparison_table(results: Dict, output_dir: Path):
    """Create comparison table for one study."""
    
    print(f"\n  {'='*60}")
    print(f"  COMPARISON FOR THIS STUDY")
    print(f"  {'='*60}")
    
    header = f"{'Method':<35} {'Align NCC':<12} {'Reg Metric':<12} {'MAE':<8} {'Reg NCC':<8}"
    print(f"\n  {header}")
    print(f"  {'─'*83}")
    
    for method, data in results.items():
        if not data.get('success', False):
            error_msg = data.get('error', 'FAILED')
            print(f"  {method:<35} {error_msg}")
            continue
        
        align_ncc = data['alignment']['mean_ncc']
        
        if 'registration_summary' in data:
            reg_metric = data['registration_summary']['mean_final_metric']
            mae = data['registration_summary']['mean_mae']
            reg_ncc = data['registration_summary']['mean_ncc']
            
            print(f"  {method:<35} {align_ncc:>11.4f} {reg_metric:>11.6f} {mae:>7.2f} {reg_ncc:>7.4f}")
        else:
            print(f"  {method:<35} {align_ncc:>11.4f} {'N/A'}")


def process_all_studies(
    input_dir: Path,
    output_dir: Path,
    labels_csv: Path,
    methods: List[str],
    max_studies: Optional[int] = None
):
    """Process all studies with all methods."""
    
    print("\n" + "="*80)
    print("ABDOMINAL PREPROCESSING COMPARISON")
    print("="*80)
    print(f"Input:   {input_dir}")
    print(f"Output:  {output_dir}")
    print(f"Methods: {', '.join(methods)}")
    print(f"Organ threshold: {Config.ORGAN_THRESHOLD} HU (abdomen-specific!)")
    if max_studies:
        print(f"Testing: First {max_studies} studies")
    print("="*80)
    
    labels_df = pd.read_csv(labels_csv)
    
    study_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    
    if max_studies:
        study_dirs = study_dirs[:max_studies]
    
    print(f"\nProcessing {len(study_dirs)} studies...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for idx, study_dir in enumerate(study_dirs, 1):
        study_id = study_dir.name
        print(f"\n[{idx}/{len(study_dirs)}]")
        
        try:
            result = process_study_all_methods(
                study_id, input_dir, output_dir, labels_df, methods
            )
            
            if result:
                all_results.append(result)
                
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("✅ PROCESSING COMPLETE")
    print("="*80)
    print(f"\nProcessed {len(all_results)} studies successfully")
    print(f"Results saved in: {output_dir}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    
    process_all_studies(
        input_dir=Config.INPUT_DIR,
        output_dir=Config.OUTPUT_BASE_DIR,
        labels_csv=Config.LABELS_CSV,
        methods=Config.METHODS,
        max_studies=Config.MAX_STUDIES
    )
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print(f"\n1. Check: {Config.OUTPUT_BASE_DIR}/{{study_id}}/")
    print(f"2. Compare preprocessed volumes visually")
    print(f"3. Look for methods with good Align_NCC AND low Reg_MAE")
    print(f"\n4. Expected improvements:")
    print(f"   - abdominal_windowing: Should preserve organs now!")
    print(f"   - histogram_matching: Should work without errors")
    print(f"   - All methods: Better align_ncc (not -inf)")