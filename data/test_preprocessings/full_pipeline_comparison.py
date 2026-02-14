"""
Complete Pipeline Preprocessing Comparison
===========================================

Tests preprocessing methods on BOTH:
1. Z-Alignment (slice matching)
2. Rigid Registration

Compares metrics for the entire pipeline.
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
    """Configuration for full pipeline comparison."""
    
    # Paths
    MAIN_PATH = "../ncct_cect/vindr_ds/"
    INPUT_DIR = Path(MAIN_PATH + "cropped_volumes")  
    OUTPUT_BASE_DIR = Path(MAIN_PATH + "full_pipeline_comparison")
    LABELS_CSV = MAIN_PATH + "labels.csv"
    
    # Preprocessing methods to test
    METHODS = [
        "none",                      # Baseline
        "windowing",                 # Outlier removal
        "standardization",           # Z-score
        "windowing+standardization", # Combined
        "histogram_matching"         # Histogram matched
    ]
    
    # Alignment settings
    BODY_THRESHOLD = -100
    MIN_OVERLAP_SLICES = 30
    SEARCH_REGION = 'upper'
    
    # Registration settings
    NUM_ITERATIONS = 200
    LEARNING_RATE = 1.0
    SAMPLING_PERCENTAGE = 0.20
    
    # How many studies to test
    MAX_STUDIES = 10
    

# =============================================================================
# PREPROCESSING FUNCTIONS
# =============================================================================

def apply_body_windowing(volume_sitk: sitk.Image, body_threshold: int = -500) -> sitk.Image:
    """Clip HU to body range."""
    vol_np = sitk.GetArrayFromImage(volume_sitk)
    body_mask = vol_np > body_threshold
    
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


def apply_standardization(volume_sitk: sitk.Image, body_threshold: int = -500) -> sitk.Image:
    """Z-score normalization."""
    vol_np = sitk.GetArrayFromImage(volume_sitk)
    body_mask = vol_np > body_threshold
    
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


def apply_histogram_matching(moving_sitk: sitk.Image, fixed_sitk: sitk.Image) -> sitk.Image:
    """Match histogram."""
    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(256)
    matcher.SetNumberOfMatchPoints(10)
    matcher.ThresholdAtMeanIntensityOn()
    return matcher.Execute(moving_sitk, fixed_sitk)


def preprocess_volume(volume_sitk: sitk.Image, method: str, reference_sitk: Optional[sitk.Image] = None) -> sitk.Image:
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
        if reference_sitk is None:
            raise ValueError("Reference required for histogram matching")
        vol = apply_body_windowing(volume_sitk)
        ref = apply_body_windowing(reference_sitk)
        return apply_histogram_matching(vol, ref)
    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# ALIGNMENT WITH METRICS
# =============================================================================

def find_best_reference_slice(reference_vol: np.ndarray, body_threshold: int = -600, search_region: str = 'upper') -> int:
    """Find optimal reference slice."""
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
        body_mask = slice_data > body_threshold
        
        if body_mask.sum() < 1000:
            continue
        
        body_values = slice_data[body_mask]
        score = body_mask.sum() * body_values.std()
        
        if score > best_score:
            best_score = score
            best_idx = idx
    
    return best_idx


def match_single_slice(reference_slice: np.ndarray, moving_vol: np.ndarray, search_range: Tuple[int, int], 
                       body_threshold: int = -600) -> Tuple[int, float]:
    """Match reference slice to best slice in moving volume."""
    target_shape = reference_slice.shape
    best_idx = (search_range[0] + search_range[1]) // 2
    best_score = -np.inf
    
    ref_norm = (reference_slice - reference_slice.mean()) / (reference_slice.std() + 1e-8)
    
    for m_idx in range(search_range[0], min(search_range[1], moving_vol.shape[0])):
        moving_slice = moving_vol[m_idx]
        
        if (moving_slice > body_threshold).sum() < 1000:
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
    body_threshold: int = -600,
    search_region: str = 'upper'
) -> Dict:
    """
    Align all series with detailed metrics.
    
    Returns:
        Dictionary with alignment results and metrics
    """
    
    # Load reference
    ref_sitk = series_volumes[reference_series_id]
    ref_np = sitk.GetArrayFromImage(ref_sitk)
    
    # Find reference slice
    ref_slice_idx = find_best_reference_slice(ref_np, body_threshold, search_region)
    ref_slice = ref_np[ref_slice_idx]
    
    # Results
    results = {
        'study_id': study_id,
        'reference_series': reference_series_id,
        'reference_slice_idx': ref_slice_idx,
        'series_metrics': {},
        'alignment_quality': {}
    }
    
    # Match each series
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
        
        # Define search range
        if search_region == 'upper':
            search_start = int(D * 0.50)
            search_end = int(D * 0.95)
        elif search_region == 'middle':
            search_start = int(D * 0.25)
            search_end = int(D * 0.75)
        else:
            search_start = int(D * 0.05)
            search_end = int(D * 0.50)
        
        # Match
        matched_idx, ncc_score = match_single_slice(
            ref_slice, vol_np,
            search_range=(search_start, search_end),
            body_threshold=body_threshold
        )
        
        z_offset = ref_slice_idx - matched_idx
        
        results['series_metrics'][series_id] = {
            'z_offset': int(z_offset),
            'matched_idx': int(matched_idx),
            'ncc_score': float(ncc_score),
            'is_reference': False
        }
    
    # Compute alignment quality metrics
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
# REGISTRATION WITH METRICS
# =============================================================================

def register_with_metrics(
    fixed: sitk.Image,
    moving: sitk.Image,
    num_iterations: int = 200,
    learning_rate: float = 1.0,
    sampling_percentage: float = 0.20
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
        # Ensure Float32
        fixed = sitk.Cast(fixed, sitk.sitkFloat32)
        moving = sitk.Cast(moving, sitk.sitkFloat32)
        
        # Setup registration
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
        
        # Get initial metric
        try:
            registration.SetMetricSamplingPercentage(0.1)
            initial_metric = registration.MetricEvaluate(fixed, moving)
            metrics['initial_metric'] = float(initial_metric)
            registration.SetMetricSamplingPercentage(sampling_percentage)
        except:
            pass
        
        # Execute
        start_time = time.time()
        final_transform = registration.Execute(fixed, moving)
        metrics['registration_time'] = time.time() - start_time
        
        metrics['final_metric'] = float(registration.GetMetricValue())
        metrics['iterations_used'] = int(registration.GetOptimizerIteration())
        
        if metrics['initial_metric'] is not None:
            metrics['metric_improvement'] = metrics['initial_metric'] - metrics['final_metric']
        
        # Apply transform
        registered = sitk.Resample(moving, fixed, final_transform, sitk.sitkLinear, 0.0, moving.GetPixelID())
        
        # Image similarity metrics
        fixed_np = sitk.GetArrayFromImage(fixed)
        reg_np = sitk.GetArrayFromImage(registered)
        
        if fixed_np.shape == reg_np.shape:
            body_mask = fixed_np > -100
            
            if body_mask.sum() > 1000:
                fixed_body = fixed_np[body_mask]
                reg_body = reg_np[body_mask]
                
                metrics['mae'] = float(np.abs(fixed_body - reg_body).mean())
                
                fixed_norm = (fixed_body - fixed_body.mean()) / (fixed_body.std() + 1e-8)
                reg_norm = (reg_body - reg_body.mean()) / (reg_body.std() + 1e-8)
                metrics['ncc'] = float(np.mean(fixed_norm * reg_norm))
            
            # SSIM
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
# FULL PIPELINE
# =============================================================================

def run_full_pipeline_single_study(
    study_id: str,
    study_dir: Path,
    labels_df: pd.DataFrame,
    method: str,
    output_dir: Path
) -> Dict:
    """
    Run complete pipeline (preprocessing → alignment → registration) for one study.
    
    Returns:
        Complete results with metrics from all steps
    """
    
    print(f"\n  {'─'*60}")
    print(f"  Method: {method}")
    print(f"  {'─'*60}")
    

    # print("\nDEBUG - directory content:")
    # print(f"{study_dir} directory")
    # for f in sorted(study_dir.glob("*crop.nii.gz")):
    #     print(f"  {f.name}")

    # Find series
    volume_files = list(study_dir.glob(f"{study_id}_*_crop.nii.gz"))
    
    if len(volume_files) == 0:
        return {'error': 'No volumes found'}
    
    # Map to phases
    series_info = {}
    for vol_path in volume_files:
        basename = vol_path.name
        series_id = basename.replace(f"{study_id}_", "").replace("_crop.nii.gz", "")
        print("series_id", series_id)
        # Find phase
        phase = "Unknown"
        study_labels = labels_df[labels_df["StudyInstanceUID"] == study_id]
        for _, row in study_labels.iterrows():
            print(" in csv label ")
            if row["SeriesInstanceUID"] == series_id:
                phase = row["Label"]
                break
        
        series_info[series_id] = {
            'path': vol_path,
            'phase': phase
        }

        print(series_info[series_id])
    
    # Find non-contrast
    nc_series = None
    for series_id, info in series_info.items():
        if info['phase'] == 'Non-contrast':
            nc_series = series_id
            break
    
    if nc_series is None:
        return {'error': 'No non-contrast series'}
    
    if len(series_info) < 2:
        return {'error': f'Only {len(series_info)} series'}
    
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
        # =====================================================================
        # STEP 1: PREPROCESSING
        # =====================================================================
        print(f"    [1/3] Preprocessing...")
        
        # Load all volumes
        original_volumes = {}
        for series_id, info in series_info.items():
            original_volumes[series_id] = sitk.ReadImage(str(info['path']))
        
        # Preprocess
        preprocessed_volumes = {}
        ref_vol = original_volumes[nc_series]
        
        for series_id, vol in original_volumes.items():
            if series_id == nc_series:
                preprocessed_volumes[series_id] = preprocess_volume(vol, method)
            else:
                preprocessed_volumes[series_id] = preprocess_volume(
                    vol, method, reference_sitk=ref_vol
                )
            
            # Save preprocessed
            prep_path = method_dir / f"{series_id}_preprocessed.nii.gz"
            sitk.WriteImage(preprocessed_volumes[series_id], str(prep_path))
        
        print(f"        ✓ Preprocessed {len(preprocessed_volumes)} volumes")
        
        # =====================================================================
        # STEP 2: ALIGNMENT
        # =====================================================================
        print(f"    [2/3] Z-Alignment...")
        
        alignment_start = time.time()
        alignment_results = align_study_with_metrics(
            study_id=study_id,
            series_volumes=preprocessed_volumes,
            reference_series_id=nc_series,
            body_threshold=Config.BODY_THRESHOLD,
            search_region=Config.SEARCH_REGION
        )
        alignment_time = time.time() - alignment_start
        
        results['alignment'] = alignment_results['alignment_quality']
        results['alignment']['time'] = alignment_time
        results['alignment']['series_details'] = alignment_results['series_metrics']
        
        print(f"        ✓ Aligned: mean_ncc={results['alignment']['mean_ncc']:.4f}")
        
        # =====================================================================
        # STEP 3: REGISTRATION
        # =====================================================================
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
                sampling_percentage=Config.SAMPLING_PERCENTAGE
            )
            
            registration_results[series_id] = {
                'phase': info['phase'],
                'metrics': reg_metrics
            }
        
        results['registration'] = registration_results
        
        # Compute average registration metrics
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
        results['error'] = str(e)
        results['success'] = False
    
    # Save results
    with open(method_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


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
        print(f"  ✗ Directory not found")
        return None
    
    study_output_dir = output_dir / study_id
    study_output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for method in methods:
        result = run_full_pipeline_single_study(
            study_id, study_dir, labels_df, method, study_output_dir
        )
        all_results[method] = result
    
    # Save combined results
    with open(study_output_dir / 'all_methods_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create comparison table
    create_study_comparison_table(all_results, study_output_dir)
    
    return all_results


def create_study_comparison_table(results: Dict, output_dir: Path):
    """Create comparison table for one study."""
    
    print(f"\n  {'='*60}")
    print(f"  COMPARISON FOR THIS STUDY")
    print(f"  {'='*60}")
    
    header = f"{'Method':<30} {'Align NCC':<12} {'Reg Metric':<12} {'MAE':<8} {'Reg NCC':<8}"
    print(f"\n  {header}")
    print(f"  {'─'*78}")
    
    for method, data in results.items():
        if not data.get('success', False):
            print(f"  {method:<30} {'FAILED'}")
            continue
        
        align_ncc = data['alignment']['mean_ncc']
        
        if 'registration_summary' in data:
            reg_metric = data['registration_summary']['mean_final_metric']
            mae = data['registration_summary']['mean_mae']
            reg_ncc = data['registration_summary']['mean_ncc']
            
            print(f"  {method:<30} {align_ncc:>11.4f} {reg_metric:>11.6f} {mae:>7.2f} {reg_ncc:>7.4f}")
        else:
            print(f"  {method:<30} {align_ncc:>11.4f} {'N/A'}")


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def process_all_studies(
    input_dir: Path,
    output_dir: Path,
    labels_csv: str,
    methods: List[str],
    max_studies: Optional[int] = None
):
    """Process all studies with all methods."""
    
    print("\n" + "="*80)
    print("FULL PIPELINE PREPROCESSING COMPARISON")
    print("="*80)
    print(f"Input:   {input_dir}")
    print(f"Output:  {output_dir}")
    print(f"Methods: {', '.join(methods)}")
    if max_studies:
        print(f"Testing: First {max_studies} studies")
    print("="*80)
    
    # Load labels
    labels_df = pd.read_csv(labels_csv)
    
    # Find studies
    study_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    
    if max_studies:
        study_dirs = study_dirs[:max_studies]
    
    print(f"\nProcessing {len(study_dirs)} studies...")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each study
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
    
    # Generate final report
    if len(all_results) > 0:
        generate_final_report(all_results, methods, output_dir)
    
    print("\n" + "="*80)
    print("✅ PROCESSING COMPLETE")
    print("="*80)


# =============================================================================
# FINAL REPORT
# =============================================================================

def generate_final_report(all_results: List[Dict], methods: List[str], output_dir: Path):
    """Generate comprehensive comparison report."""
    
    print("\n" + "="*80)
    print("GENERATING FINAL REPORT")
    print("="*80)
    
    # Collect metrics
    rows = []
    
    for study_results in all_results:
        for method in methods:
            if method not in study_results:
                continue
            
            data = study_results[method]
            
            if not data.get('success', False):
                continue
            
            row = {
                'method': method,
                'study_id': data['study_id'],
                'num_series': data['num_series'],
                'align_mean_ncc': data['alignment']['mean_ncc'],
                'align_std_ncc': data['alignment']['std_ncc'],
                'align_time': data['alignment']['time']
            }
            
            if 'registration_summary' in data:
                row.update({
                    'reg_final_metric': data['registration_summary']['mean_final_metric'],
                    'reg_mae': data['registration_summary']['mean_mae'],
                    'reg_ncc': data['registration_summary']['mean_ncc'],
                    'reg_time': data['registration_summary']['mean_time']
                })
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save detailed results
    detailed_path = output_dir / "detailed_results.csv"
    df.to_csv(detailed_path, index=False)
    print(f"\n  ✓ Detailed results: {detailed_path}")
    
    # Summary by method
    print("\n" + "─"*80)
    print("SUMMARY BY METHOD")
    print("─"*80)
    
    summary_rows = []
    
    for method in methods:
        method_df = df[df['method'] == method]
        
        if len(method_df) == 0:
            continue
        
        summary = {
            'Method': method,
            'N': len(method_df),
            'Align_NCC': f"{method_df['align_mean_ncc'].mean():.4f} ± {method_df['align_mean_ncc'].std():.4f}",
            'Align_Time': f"{method_df['align_time'].mean():.2f}s"
        }
        
        if 'reg_final_metric' in method_df.columns and method_df['reg_final_metric'].notna().any():
            summary.update({
                'Reg_Metric': f"{method_df['reg_final_metric'].mean():.6f} ± {method_df['reg_final_metric'].std():.6f}",
                'Reg_MAE': f"{method_df['reg_mae'].mean():.2f} ± {method_df['reg_mae'].std():.2f}",
                'Reg_NCC': f"{method_df['reg_ncc'].mean():.4f} ± {method_df['reg_ncc'].std():.4f}",
                'Reg_Time': f"{method_df['reg_time'].mean():.2f}s"
            })
        
        summary_rows.append(summary)
        
        print(f"\n{method}:")
        for key, value in summary.items():
            if key != 'Method':
                print(f"  {key:15s}: {value}")
    
    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "summary_by_method.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  ✓ Summary: {summary_path}")
    
    # Create plots
    create_comparison_plots(df, output_dir, methods)
    
    # Recommendations
    print("\n" + "─"*80)
    print("RECOMMENDATIONS")
    print("─"*80)
    
    if 'align_mean_ncc' in df.columns:
        best_align = df.groupby('method')['align_mean_ncc'].mean().idxmax()
        print(f"\n  Best for Alignment: {best_align}")
    
    if 'reg_final_metric' in df.columns and df['reg_final_metric'].notna().any():
        best_reg = df.groupby('method')['reg_final_metric'].mean().idxmin()
        print(f"  Best for Registration: {best_reg}")
    
    if 'reg_mae' in df.columns and df['reg_mae'].notna().any():
        best_mae = df.groupby('method')['reg_mae'].mean().idxmin()
        print(f"  Best by MAE: {best_mae}")


def create_comparison_plots(df: pd.DataFrame, output_dir: Path, methods: List[str]):
    """Create visualization plots."""
    
    print("\n  Creating plots...")
    
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Plot 1: Alignment NCC
    if 'align_mean_ncc' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        data = [df[df['method'] == m]['align_mean_ncc'].values for m in methods if m in df['method'].unique()]
        labels = [m for m in methods if m in df['method'].unique()]
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        
        ax.set_ylabel('Alignment NCC (higher is better)', fontsize=12)
        ax.set_title('Z-Alignment Quality by Preprocessing Method', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(plots_dir / 'alignment_ncc_comparison.png', dpi=150)
        plt.close()
    
    # Plot 2: Registration Metric
    if 'reg_final_metric' in df.columns and df['reg_final_metric'].notna().any():
        fig, ax = plt.subplots(figsize=(12, 6))
        data = [df[df['method'] == m]['reg_final_metric'].dropna().values 
                for m in methods if m in df['method'].unique()]
        labels = [m for m in methods if m in df['method'].unique()]
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightcoral')
        
        ax.set_ylabel('Registration Final Metric (lower is better)', fontsize=12)
        ax.set_title('Registration Quality by Preprocessing Method', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(plots_dir / 'registration_metric_comparison.png', dpi=150)
        plt.close()
    
    # Plot 3: Combined metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics_to_plot = [
        ('align_mean_ncc', 'Alignment NCC', 'lightblue', axes[0, 0]),
        ('reg_final_metric', 'Registration Metric', 'lightcoral', axes[0, 1]),
        ('reg_mae', 'Registration MAE', 'lightgreen', axes[1, 0]),
        ('reg_ncc', 'Registration NCC', 'wheat', axes[1, 1])
    ]
    
    for metric, title, color, ax in metrics_to_plot:
        if metric in df.columns and df[metric].notna().any():
            data = [df[df['method'] == m][metric].dropna().values 
                    for m in methods if m in df['method'].unique()]
            labels = [m for m in methods if m in df['method'].unique()]
            
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor(color)
            
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle('Complete Pipeline Metrics Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(plots_dir / 'complete_pipeline_comparison.png', dpi=150)
    plt.close()
    
    print(f"  ✓ Plots saved: {plots_dir}")


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
    print(f"\n1. Check: {Config.OUTPUT_BASE_DIR}/summary_by_method.csv")
    print(f"2. View plots: {Config.OUTPUT_BASE_DIR}/plots/")
    print(f"3. Inspect: {Config.OUTPUT_BASE_DIR}/{{study_id}}/{{method}}/")
    print(f"4. Choose best method for both alignment and registration")
    print(f"\n5. To test all studies: Set Config.MAX_STUDIES = None")