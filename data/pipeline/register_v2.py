"""
CT Multi-Phase Registration Pipeline v3.2
==========================================
FIXED:
- Proper NC reference volume saving (NOT overwritten by seg)
- All segmentations saved correctly
- Explicit file handling with debug output
"""

import os
import shutil
import numpy as np
import SimpleITK as sitk
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import glob


# =============================================================================
# FILE DISCOVERY
# =============================================================================

def scan_aligned_directory(aligned_dir: str) -> Dict[str, Dict[str, dict]]:
    """Scan aligned directory and catalog ALL existing files."""
    catalog = {}
    
    if not os.path.exists(aligned_dir):
        print(f"⚠️  Directory does not exist: {aligned_dir}")
        return catalog
    
    study_dirs = [d for d in os.listdir(aligned_dir) 
                  if os.path.isdir(os.path.join(aligned_dir, d))]
    
    print(f"\n📂 Scanning {aligned_dir}...")
    print(f"   Found {len(study_dirs)} study directories")
    
    for study_id in study_dirs:
        study_path = os.path.join(aligned_dir, study_id)
        catalog[study_id] = {}
        
        pattern = os.path.join(study_path, "*_aligned.nii.gz")
        files = glob.glob(pattern)
        image_files = [f for f in files if '_seg' not in f]
        
        for img_path in image_files:
            basename = os.path.basename(img_path)
            
            if basename.startswith(study_id + "_"):
                remainder = basename[len(study_id) + 1:]
                series_id = remainder.replace("_aligned.nii.gz", "")
            else:
                series_id = basename.replace("_aligned.nii.gz", "")
            
            # Check for segmentation
            seg_path = img_path.replace("_aligned.nii.gz", "_aligned_seg.nii.gz")
            has_seg = os.path.exists(seg_path)
            
            catalog[study_id][series_id] = {
                'image_path': img_path,
                'seg_path': seg_path if has_seg else None,
                'has_seg': has_seg,
                'basename': basename
            }
    
    total_volumes = sum(len(series) for series in catalog.values())
    total_with_seg = sum(
        1 for study in catalog.values() 
        for series in study.values() 
        if series['has_seg']
    )
    
    print(f"   Total volumes: {total_volumes}")
    print(f"   With segmentation: {total_with_seg}")
    
    return catalog


def get_available_series_for_study(
        catalog: Dict[str, Dict[str, dict]],
        study_id: str,
        labels_df: pd.DataFrame
    ) -> List[dict]:
    """Get list of series that exist on disk with their labels."""
    if study_id not in catalog:
        return []
    
    available = []
    study_labels = labels_df[labels_df["StudyInstanceUID"] == study_id]
    
    for series_id, file_info in catalog[study_id].items():
        phase = "Unknown"
        
        for _, row in study_labels.iterrows():
            label_series_id = row["SeriesInstanceUID"]
            
            if label_series_id == series_id:
                phase = row["Label"]
                break
            
            if (series_id in label_series_id or 
                label_series_id in series_id or
                series_id.rsplit('.', 1)[0] == label_series_id.rsplit('.', 1)[0]):
                phase = row["Label"]
                break
        
        available.append({
            'series_id': series_id,
            'phase': phase,
            'image_path': file_info['image_path'],
            'seg_path': file_info['seg_path'],
            'has_seg': file_info['has_seg']
        })
    
    return available


# =============================================================================
# IMAGE UTILITIES
# =============================================================================

def ensure_float32(img: sitk.Image) -> sitk.Image:
    if img.GetPixelID() != sitk.sitkFloat32:
        return sitk.Cast(img, sitk.sitkFloat32)
    return img


def diagnose_image(img: sitk.Image, name: str = "Image") -> dict:
    arr = sitk.GetArrayFromImage(img)
    return {
        'name': name,
        'size': img.GetSize(),
        'spacing': img.GetSpacing(),
        'pixel_type': img.GetPixelIDTypeAsString(),
        'min': float(arr.min()),
        'max': float(arr.max()),
        'mean': float(arr.mean()),
        'unique_values': len(np.unique(arr)) if arr.size < 1e7 else -1
    }


def is_segmentation(img: sitk.Image) -> bool:
    """Check if an image looks like a segmentation mask."""
    arr = sitk.GetArrayFromImage(img)
    unique = np.unique(arr)
    
    # Segmentation typically has few unique integer values
    if len(unique) < 50 and arr.dtype in [np.uint8, np.int8, np.uint16, np.int16, np.int32]:
        return True
    
    # Also check if values are small integers
    if arr.max() < 100 and arr.min() >= 0 and len(unique) < 20:
        return True
    
    return False


def prepare_for_registration(
        fixed: sitk.Image,
        moving: sitk.Image,
        body_threshold: int = -500,
        verbose: bool = True
    ) -> Tuple[sitk.Image, sitk.Image, sitk.Image]:
    """Prepare images for registration."""
    
    if verbose:
        print("\n   📋 Preparing images...")
    
    fixed_f32 = ensure_float32(fixed)
    moving_f32 = ensure_float32(moving)
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_f32)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1000.0)
    resampler.SetOutputPixelType(sitk.sitkFloat32)
    
    moving_resampled = resampler.Execute(moving_f32)
    
    mask = sitk.BinaryThreshold(fixed_f32, lowerThreshold=body_threshold, upperThreshold=3000)
    mask = sitk.BinaryMorphologicalClosing(mask, [3, 3, 3])
    mask = sitk.BinaryFillhole(mask)
    mask = sitk.Cast(mask, sitk.sitkUInt8)
    
    if verbose:
        print(f"      ✓ Prepared: both Float32, same grid")
    
    return fixed_f32, moving_resampled, mask


# =============================================================================
# REGISTRATION
# =============================================================================

def register_volumes(
        fixed: sitk.Image,
        moving: sitk.Image,
        fixed_mask: Optional[sitk.Image] = None,
        num_iterations: int = 200,
        learning_rate: float = 1.0,
        sampling_percentage: float = 0.2,
        final_interpolator: int = sitk.sitkLanczosWindowedSinc,
        verbose: bool = True
    ) -> Tuple[sitk.Image, sitk.Transform, dict]:
    """Register moving to fixed using rigid registration."""
    
    if verbose:
        print("\n   🔧 Running registration...")
    
    fixed = ensure_float32(fixed)
    moving = ensure_float32(moving)
    
    try:
        initial_transform = sitk.CenteredTransformInitializer(
            fixed, moving,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
    except Exception as e:
        print(f"      ⚠️  Transform init failed: {e}")
        initial_transform = sitk.Euler3DTransform()
    
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(sampling_percentage)
    
    if fixed_mask is not None:
        registration.SetMetricFixedMask(fixed_mask)
    
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsGradientDescent(
        learningRate=learning_rate,
        numberOfIterations=num_iterations,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetInitialTransform(initial_transform, inPlace=False)
    
    final_transform = registration.Execute(fixed, moving)
    final_metric = registration.GetMetricValue()
    
    if verbose:
        print(f"      ✓ Complete: metric={final_metric:.6f}")
    
    # High-quality resampling
    registered = sitk.Resample(
        moving, fixed, final_transform,
        final_interpolator,
        -1000.0,
        moving.GetPixelID()
    )
    
    return registered, final_transform, {'final_metric': float(final_metric)}


def apply_transform_to_segmentation(
        segmentation: sitk.Image,
        reference: sitk.Image,
        transform: sitk.Transform
    ) -> sitk.Image:
    """Apply transform to segmentation using NEAREST NEIGHBOR."""
    return sitk.Resample(
        segmentation, reference, transform,
        sitk.sitkNearestNeighbor,
        0,
        segmentation.GetPixelID()
    )


# =============================================================================
# NORMALIZATION
# =============================================================================

def smart_hu_window(volume: np.ndarray, body_threshold: int = -500) -> Tuple[float, float]:
    """Compute adaptive HU window."""
    
    if volume.max() <= 1.5 and volume.min() >= -0.5:
        return 0.0, 1.0
    
    body_mask = volume > body_threshold
    
    if body_mask.sum() < 1000:
        return -1000.0, 600.0
    
    body_values = volume[body_mask].ravel()
    
    low = np.percentile(body_values, 1)
    high = np.percentile(body_values, 99)
    
    q25, q75 = np.percentile(body_values, [25, 75])
    iqr = q75 - q25
    margin = max(int(1.5 * iqr), 100)
    
    w_min = max(low - margin, -1500)
    w_max = min(high + margin, 2000)
    
    if w_max - w_min < 500:
        center = (w_max + w_min) / 2
        w_min = center - 250
        w_max = center + 250
    
    return float(w_min), float(w_max)


def normalize_volume(
        vol_sitk: sitk.Image,
        reference_sitk: Optional[sitk.Image] = None
    ) -> sitk.Image:
    """Normalize volume to [0, 1]."""
    
    vol_np = sitk.GetArrayFromImage(vol_sitk)
    
    if vol_np.max() <= 1.5 and vol_np.min() >= -0.5:
        return vol_sitk
    
    w_min, w_max = smart_hu_window(vol_np)
    
    vol_clipped = np.clip(vol_np, w_min, w_max)
    vol_norm = (vol_clipped - w_min) / (w_max - w_min + 1e-8)
    
    out = sitk.GetImageFromArray(vol_norm.astype(np.float32))
    out.CopyInformation(vol_sitk)
    
    return out


# =============================================================================
# EXPLICIT SAVE FUNCTIONS
# =============================================================================

def save_volume(
        vol_sitk: sitk.Image,
        output_path: str,
        description: str = "volume"
    ) -> bool:
    """
    Save a volume to disk with verification.
    
    Returns True if successful.
    """
    try:
        # Verify it's not a segmentation being saved as volume
        info = diagnose_image(vol_sitk, description)
        
        if is_segmentation(vol_sitk):
            print(f"      ⚠️  WARNING: {description} looks like a segmentation!")
            print(f"         Values: min={info['min']}, max={info['max']}, unique={info['unique_values']}")
        
        sitk.WriteImage(vol_sitk, output_path)
        
        # Verify file was written
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            print(f"      ✓ Saved {description}: {os.path.basename(output_path)} ({file_size:.1f} MB)")
            return True
        else:
            print(f"      ❌ Failed to save {description}: {output_path}")
            return False
            
    except Exception as e:
        print(f"      ❌ Error saving {description}: {e}")
        return False


def save_segmentation(
        seg_sitk: sitk.Image,
        output_path: str,
        description: str = "segmentation"
    ) -> bool:
    """
    Save a segmentation to disk with verification.
    
    Returns True if successful.
    """
    try:
        info = diagnose_image(seg_sitk, description)
        
        if not is_segmentation(seg_sitk):
            print(f"      ⚠️  WARNING: {description} doesn't look like a segmentation!")
            print(f"         Values: min={info['min']}, max={info['max']}")
        
        sitk.WriteImage(seg_sitk, output_path)
        
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"      ✓ Saved {description}: {os.path.basename(output_path)} ({file_size:.1f} MB)")
            return True
        else:
            print(f"      ❌ Failed to save {description}: {output_path}")
            return False
            
    except Exception as e:
        print(f"      ❌ Error saving {description}: {e}")
        return False


def copy_file_safe(
        source_path: str,
        dest_path: str,
        description: str = "file"
    ) -> bool:
    """
    Safely copy a file with verification.
    
    Returns True if successful.
    """
    try:
        if not os.path.exists(source_path):
            print(f"      ❌ Source not found: {source_path}")
            return False
        
        shutil.copy2(source_path, dest_path)
        
        if os.path.exists(dest_path):
            file_size = os.path.getsize(dest_path) / (1024 * 1024)
            print(f"      ✓ Copied {description}: {os.path.basename(dest_path)} ({file_size:.1f} MB)")
            return True
        else:
            print(f"      ❌ Failed to copy {description}")
            return False
            
    except Exception as e:
        print(f"      ❌ Error copying {description}: {e}")
        return False


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_checkerboard(img1: np.ndarray, img2: np.ndarray, block_size: int = 30) -> np.ndarray:
    h, w = img1.shape
    result = np.zeros_like(img1)
    
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            i_end = min(i + block_size, h)
            j_end = min(j + block_size, w)
            
            if ((i // block_size) + (j // block_size)) % 2 == 0:
                result[i:i_end, j:j_end] = img1[i:i_end, j:j_end]
            else:
                result[i:i_end, j:j_end] = img2[i:i_end, j:j_end]
    
    return result


def visualize_registration(
        fixed: sitk.Image,
        moving_before: sitk.Image,
        registered: sitk.Image,
        output_path: str,
        title: str = "Registration Quality"
    ):
    """Create before/after visualization."""
    import matplotlib.pyplot as plt
    
    fixed_np = sitk.GetArrayFromImage(fixed)
    moving_np = sitk.GetArrayFromImage(moving_before)
    reg_np = sitk.GetArrayFromImage(registered)
    
    mid = fixed_np.shape[0] // 2
    
    body = fixed_np[fixed_np > -500]
    vmin, vmax = (np.percentile(body, [1, 99]) if len(body) > 0 else (-1000, 500))
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    axes[0, 0].imshow(fixed_np[mid], cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('Fixed (Reference)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(moving_np[mid], cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title('Moving (Before)')
    axes[0, 1].axis('off')
    
    diff_before = np.abs(fixed_np[mid] - moving_np[mid])
    im = axes[0, 2].imshow(diff_before, cmap='hot', vmin=0, vmax=500)
    axes[0, 2].set_title(f'Diff Before (MAE={diff_before.mean():.1f})')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)
    
    checker = create_checkerboard(fixed_np[mid], moving_np[mid])
    axes[0, 3].imshow(checker, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 3].set_title('Checkerboard (Before)')
    axes[0, 3].axis('off')
    
    axes[1, 0].imshow(fixed_np[mid], cmap='gray', vmin=vmin, vmax=vmax)
    axes[1, 0].set_title('Fixed (Reference)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(reg_np[mid], cmap='gray', vmin=vmin, vmax=vmax)
    axes[1, 1].set_title('Registered (After)', color='green', fontweight='bold')
    axes[1, 1].axis('off')
    
    diff_after = np.abs(fixed_np[mid] - reg_np[mid])
    im = axes[1, 2].imshow(diff_after, cmap='hot', vmin=0, vmax=500)
    axes[1, 2].set_title(f'Diff After (MAE={diff_after.mean():.1f})', color='green', fontweight='bold')
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046)
    
    checker = create_checkerboard(fixed_np[mid], reg_np[mid])
    axes[1, 3].imshow(checker, cmap='gray', vmin=vmin, vmax=vmax)
    axes[1, 3].set_title('Checkerboard (After)', color='green', fontweight='bold')
    axes[1, 3].axis('off')
    
    improvement = (diff_before.mean() - diff_after.mean()) / (diff_before.mean() + 1e-8) * 100
    fig.suptitle(f'{title}\nMAE Improvement: {improvement:.1f}%', fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN REGISTRATION FUNCTION
# =============================================================================

def register_study_v4(
        study_id: str,
        file_catalog: Dict[str, Dict[str, dict]],
        output_dir: str,
        labels_df: pd.DataFrame,
        interpolator: int = sitk.sitkLanczosWindowedSinc,
        skip_existing: bool = True,
        save_raw: bool = True,
        save_normalized: bool = True,
        save_visualizations: bool = True
    ) -> dict:
    """
    Register all series in a study to non-contrast.
    
    PROPERLY SAVES:
    1. NC reference volume (raw + normalized) - VERIFIED not a seg!
    2. NC segmentation (if exists) - COPIED from source
    3. All contrast volumes (registered, raw + normalized)
    4. All contrast segmentations (registered with transform)
    """
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    if study_id not in file_catalog:
        return {'status': 'skipped', 'reason': 'not_in_catalog'}
    
    study_files = file_catalog[study_id]
    if len(study_files) == 0:
        return {'status': 'skipped', 'reason': 'no_files'}
    
    available_series = get_available_series_for_study(file_catalog, study_id, labels_df)
    if len(available_series) == 0:
        return {'status': 'skipped', 'reason': 'no_labeled_series'}
    
    # Find non-contrast
    nc_series = None
    for s in available_series:
        if s['phase'] == 'Non-contrast':
            nc_series = s
            break
    
    if nc_series is None:
        return {'status': 'skipped', 'reason': 'no_noncontrast'}
    
    # =========================================================================
    # SETUP
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"📁 Registering Study: {study_id}")
    print(f"{'='*80}")
    
    # Print all available series
    print(f"\n   Available series ({len(available_series)}):")
    for s in available_series:
        seg_mark = "📄" if s['has_seg'] else "  "
        is_nc = "★" if s['phase'] == 'Non-contrast' else " "
        print(f"      {is_nc}{seg_mark} {s['phase']}: {os.path.basename(s['image_path'])}")
        if s['has_seg']:
            print(f"            Seg: {os.path.basename(s['seg_path'])}")
    
    study_output_dir = os.path.join(output_dir, study_id)
    os.makedirs(study_output_dir, exist_ok=True)
    
    results = {
        'study_id': study_id,
        'reference': nc_series['series_id'],
        'saved_files': {
            'volumes_raw': [],
            'volumes_norm': [],
            'segmentations': [],
            'visualizations': []
        },
        'successful': [],
        'failed': [],
        'skipped': []
    }
    
    # =========================================================================
    # STEP 1: LOAD AND VERIFY NC REFERENCE
    # =========================================================================
    print(f"\n   📥 Loading Non-contrast reference...")
    print(f"      Path: {nc_series['image_path']}")
    
    nc_vol_sitk = sitk.ReadImage(nc_series['image_path'])
    nc_info = diagnose_image(nc_vol_sitk, "NC Volume")
    
    print(f"      Size: {nc_info['size']}")
    print(f"      Range: [{nc_info['min']:.1f}, {nc_info['max']:.1f}]")
    print(f"      Type: {nc_info['pixel_type']}")
    
    # VERIFY it's actually a volume, not a segmentation!
    if is_segmentation(nc_vol_sitk):
        print(f"      ❌ ERROR: NC volume looks like a segmentation!")
        print(f"         This is a bug - please check your input files.")
        return {'status': 'failed', 'reason': 'nc_is_segmentation'}
    else:
        print(f"      ✓ Verified: NC is a CT volume (not segmentation)")
    
    # =========================================================================
    # STEP 2: SAVE NC REFERENCE VOLUME
    # =========================================================================
    print(f"\n   💾 Saving Non-contrast reference volume...")
    
    nc_base = f"{study_id}_{nc_series['series_id']}_registered"
    
    # Save RAW volume
    if save_raw:
        nc_raw_path = os.path.join(study_output_dir, f"{nc_base}.nii.gz")
        if save_volume(nc_vol_sitk, nc_raw_path, "NC raw volume"):
            results['saved_files']['volumes_raw'].append(nc_raw_path)
    
    # Save NORMALIZED volume
    if save_normalized:
        nc_norm_sitk = normalize_volume(nc_vol_sitk, reference_sitk=None)
        nc_norm_path = os.path.join(study_output_dir, f"{nc_base}_norm.nii.gz")
        if save_volume(nc_norm_sitk, nc_norm_path, "NC normalized volume"):
            results['saved_files']['volumes_norm'].append(nc_norm_path)
    
    # =========================================================================
    # STEP 3: SAVE NC SEGMENTATION (IF EXISTS)
    # =========================================================================
    print(f"\n   💾 Saving Non-contrast segmentation...")
    # print("nc_series info ", nc_series['has_seg'] , nc_series['seg_path'])
    if nc_series['has_seg'] and nc_series['seg_path']:
        print(f"      Source: {nc_series['seg_path']}")
        
        if os.path.exists(nc_series['seg_path']):
            # Load and verify it's a segmentation
            nc_seg_sitk = sitk.ReadImage(nc_series['seg_path'])
            nc_seg_info = diagnose_image(nc_seg_sitk, "NC Segmentation")
            
            print(f"      Size: {nc_seg_info['size']}")
            print(f"      Range: [{nc_seg_info['min']:.1f}, {nc_seg_info['max']:.1f}]")
            print(f"      Unique values: ~{nc_seg_info['unique_values']}")
            
            if not is_segmentation(nc_seg_sitk):
                print(f"      ⚠️  WARNING: NC seg doesn't look like a segmentation!")
            
            # Save segmentation
            nc_seg_path = os.path.join(study_output_dir, f"{nc_base}_seg.nii.gz")
            if save_segmentation(nc_seg_sitk, nc_seg_path, "NC segmentation"):
                results['saved_files']['segmentations'].append(nc_seg_path)
        else:
            print(f"      ❌ Segmentation file not found!")
    else:
        print(f"      ⊘ No segmentation available for NC")
    
    # =========================================================================
    # STEP 4: PROCESS EACH CONTRAST SERIES
    # =========================================================================
    for series_info in available_series:
        phase = series_info['phase']
        
        # Skip non-contrast (already processed)
        if phase == 'Non-contrast':
            continue
        
        series_id = series_info['series_id']
        image_path = series_info['image_path']
        
        print(f"\n   {'='*60}")
        print(f"   🔄 Processing: {phase}")
        print(f"      Series ID: {series_id}")
        print(f"      Image: {os.path.basename(image_path)}")
        if series_info['has_seg']:
            print(f"      Seg: {os.path.basename(series_info['seg_path'])}")
        
        # Check file exists
        if not os.path.exists(image_path):
            print(f"      ❌ Image file not found!")
            results['failed'].append({'series': series_id, 'phase': phase, 'reason': 'file_missing'})
            continue
        
        # Check if already processed
        output_base = f"{study_id}_{series_id}_registered"
        output_raw_path = os.path.join(study_output_dir, f"{output_base}.nii.gz")
        
        if skip_existing and os.path.exists(output_raw_path):
            print(f"      ⏭️  Already processed, skipping")
            results['skipped'].append(series_id)
            continue
        
        try:
            # -----------------------------------------------------------------
            # Load moving volume
            # -----------------------------------------------------------------
            print(f"\n      📥 Loading volume...")
            moving_sitk = sitk.ReadImage(image_path)
            moving_info = diagnose_image(moving_sitk, f"{phase} Volume")
            
            print(f"         Size: {moving_info['size']}")
            print(f"         Range: [{moving_info['min']:.1f}, {moving_info['max']:.1f}]")
            
            if is_segmentation(moving_sitk):
                print(f"         ❌ ERROR: Volume looks like segmentation!")
                results['failed'].append({'series': series_id, 'phase': phase, 'reason': 'volume_is_seg'})
                continue
            
            # -----------------------------------------------------------------
            # Prepare and register
            # -----------------------------------------------------------------
            fixed_prepared, moving_prepared, body_mask = prepare_for_registration(
                nc_vol_sitk, moving_sitk, verbose=True
            )
            
            registered, transform, metrics = register_volumes(
                fixed_prepared, moving_prepared,
                fixed_mask=body_mask,
                final_interpolator=interpolator,
                verbose=True
            )
            
            # -----------------------------------------------------------------
            # Save registered volume (RAW)
            # -----------------------------------------------------------------
            if save_raw:
                if save_volume(registered, output_raw_path, f"{phase} registered raw"):
                    results['saved_files']['volumes_raw'].append(output_raw_path)
            
            # -----------------------------------------------------------------
            # Save registered volume (NORMALIZED)
            # -----------------------------------------------------------------
            if save_normalized:
                reg_norm = normalize_volume(registered, reference_sitk=nc_vol_sitk)
                output_norm_path = os.path.join(study_output_dir, f"{output_base}_norm.nii.gz")
                if save_volume(reg_norm, output_norm_path, f"{phase} registered normalized"):
                    results['saved_files']['volumes_norm'].append(output_norm_path)
            
            # -----------------------------------------------------------------
            # Save visualization
            # -----------------------------------------------------------------
            if save_visualizations:
                viz_path = os.path.join(study_output_dir, f"{output_base}_quality.png")
                visualize_registration(
                    fixed_prepared, moving_prepared, registered,
                    viz_path, title=f"{phase} → Non-contrast"
                )
                results['saved_files']['visualizations'].append(viz_path)
                print(f"      ✓ Saved visualization")
            
            # -----------------------------------------------------------------
            # Process segmentation (if exists)
            # -----------------------------------------------------------------
            print(f"\n      📄 Processing segmentation...")
            # print("series info ", series_info['has_seg'] , series_info['seg_path'])
            if series_info['has_seg'] and series_info['seg_path']:
                seg_source_path = series_info['seg_path']
                
                if os.path.exists(seg_source_path):
                    print(f"         Source: {os.path.basename(seg_source_path)}")
                    
                    # Load segmentation
                    seg_sitk = sitk.ReadImage(seg_source_path)
                    seg_info = diagnose_image(seg_sitk, f"{phase} Segmentation")
                    
                    print(f"         Size: {seg_info['size']}")
                    print(f"         Range: [{seg_info['min']:.1f}, {seg_info['max']:.1f}]")
                    
                    # Apply registration transform
                    print(f"         Applying transform...")
                    reg_seg = apply_transform_to_segmentation(
                        seg_sitk, fixed_prepared, transform
                    )
                    
                    # Save registered segmentation
                    seg_output_path = os.path.join(study_output_dir, f"{output_base}_seg.nii.gz")
                    if save_segmentation(reg_seg, seg_output_path, f"{phase} registered segmentation"):
                        results['saved_files']['segmentations'].append(seg_output_path)
                else:
                    print(f"         ❌ Segmentation file not found: {seg_source_path}")
            else:
                print(f"         ⊘ No segmentation for this series")
            
            results['successful'].append({
                'series': series_id,
                'phase': phase,
                'metric': metrics['final_metric']
            })
            
        except Exception as e:
            print(f"      ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results['failed'].append({'series': series_id, 'phase': phase, 'reason': str(e)})
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print(f"\n   {'='*60}")
    print(f"   📊 Study Summary: {study_id[:40]}...")
    print(f"   {'='*60}")
    print(f"      ✓ Successful: {len(results['successful'])}")
    print(f"      ⏭️  Skipped:   {len(results['skipped'])}")
    print(f"      ❌ Failed:    {len(results['failed'])}")
    
    print(f"\n   📁 Files saved:")
    print(f"      Raw volumes:        {len(results['saved_files']['volumes_raw'])}")
    print(f"      Normalized volumes: {len(results['saved_files']['volumes_norm'])}")
    print(f"      Segmentations:      {len(results['saved_files']['segmentations'])}")
    print(f"      Visualizations:     {len(results['saved_files']['visualizations'])}")
    
    print(f"\n   📋 File list:")
    all_files = (
        results['saved_files']['volumes_raw'] + 
        results['saved_files']['volumes_norm'] + 
        results['saved_files']['segmentations']
    )
    for f in sorted(all_files):
        print(f"      - {os.path.basename(f)}")
    
    results['status'] = 'complete'
    return results


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def register_all_studies_v4(
        aligned_dir: str,
        output_dir: str,
        labels_csv: str,
        interpolator: int = sitk.sitkLanczosWindowedSinc,
        skip_existing: bool = True,
        save_raw: bool = True,
        save_normalized: bool = True,
        save_visualizations: bool = True
    ):
    """
    Register all studies with proper validation and complete file saving.
    """
    print(f"\n{'='*80}")
    print("BATCH REGISTRATION v3.2 (FIXED)")
    print(f"{'='*80}")
    
    interp_name = {
        sitk.sitkLinear: "Linear",
        sitk.sitkBSpline: "BSpline",
        sitk.sitkLanczosWindowedSinc: "LanczosWindowedSinc (BEST)",
    }.get(interpolator, "Unknown")
    
    print(f"Input:          {aligned_dir}")
    print(f"Output:         {output_dir}")
    print(f"Interpolator:   {interp_name}")
    print(f"Save raw:       {save_raw}")
    print(f"Save normalized:{save_normalized}")
    print(f"Skip existing:  {skip_existing}")
    
    # Scan directory
    file_catalog = scan_aligned_directory(aligned_dir)
    
    if len(file_catalog) == 0:
        print("\n❌ No studies found!")
        return
    
    # Load labels
    labels_df = pd.read_csv(labels_csv)
    
    # Get valid studies
    catalog_studies = set(file_catalog.keys())
    label_studies = set(labels_df["StudyInstanceUID"].unique())
    valid_studies = catalog_studies & label_studies
    
    print(f"\n   Studies in catalog: {len(catalog_studies)}")
    print(f"   Studies in labels:  {len(label_studies)}")
    print(f"   Valid (both):       {len(valid_studies)}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Counters
    successful = 0
    failed = 0
    skipped = 0
    
    total_files = {
        'volumes_raw': 0,
        'volumes_norm': 0,
        'segmentations': 0,
        'visualizations': 0
    }
    
    for idx, study_id in enumerate(sorted(valid_studies), 1):
        print(f"\n[{idx}/{len(valid_studies)}] {study_id}")
        
        try:
            result = register_study_v4(
                study_id=study_id,
                file_catalog=file_catalog,
                output_dir=output_dir,
                labels_df=labels_df,
                interpolator=interpolator,
                skip_existing=skip_existing,
                save_raw=save_raw,
                save_normalized=save_normalized,
                save_visualizations=save_visualizations
            )
            
            if result['status'] == 'complete':
                if len(result['successful']) > 0:
                    successful += 1
                else:
                    skipped += 1
                
                # Count files
                for key in total_files:
                    total_files[key] += len(result['saved_files'].get(key, []))
            else:
                skipped += 1
                
        except Exception as e:
            print(f"   ❌ Study failed: {e}")
            failed += 1
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print(f"\n{'='*80}")
    print("BATCH COMPLETE")
    print(f"{'='*80}")
    print(f"   ✓ Successful: {successful}/{len(valid_studies)}")
    print(f"   ⏭️  Skipped:   {skipped}/{len(valid_studies)}")
    print(f"   ❌ Failed:    {failed}/{len(valid_studies)}")
    
    print(f"\n   📁 Total files saved:")
    print(f"      Raw volumes:        {total_files['volumes_raw']}")
    print(f"      Normalized volumes: {total_files['volumes_norm']}")
    print(f"      Segmentations:      {total_files['segmentations']}")
    print(f"      Visualizations:     {total_files['visualizations']}")
    print(f"      ─────────────────────────────")
    print(f"      TOTAL:              {sum(total_files.values())}")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    # Configuration
    MAIN_PATH = "../../ncct_cect/vindr_ds/"
    ALIGNED_DIR = MAIN_PATH + "aligned_cases"
    OUTPUT_DIR = MAIN_PATH + "registered_output"
    LABELS_CSV = MAIN_PATH + "labels.csv"
    
    # Choose interpolator quality
    # Options:
    #   sitk.sitkLinear              - Fast but blurry
    #   sitk.sitkBSpline             - Balanced
    #   sitk.sitkLanczosWindowedSinc - BEST quality (recommended)
    
    INTERPOLATOR = sitk.sitkLanczosWindowedSinc
    
    # Run
    register_all_studies_v4(
        aligned_dir=ALIGNED_DIR,
        output_dir=OUTPUT_DIR,
        labels_csv=LABELS_CSV,
        interpolator=INTERPOLATOR,
        skip_existing=False,  # Set True to skip already processed
        save_raw=True,
        save_normalized=True,
        save_visualizations=True
    )