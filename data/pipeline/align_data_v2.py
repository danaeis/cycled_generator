"""
Multi-Series CT Volume Alignment with Origin Normalization
===========================================================
Ensures all aligned volumes share the same physical coordinate system.
"""

import os
import json
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import cv2
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

# =============================================================================
# ORIGIN NORMALIZATION FUNCTIONS
# =============================================================================

def normalize_origin_to_reference(
        moving_sitk: sitk.Image,
        reference_sitk: sitk.Image,
        method: str = 'copy'  # 'copy', 'center', or 'resample'
    ) -> sitk.Image:
    """
    Normalize the origin of a moving image to match the reference.
    
    Args:
        moving_sitk: Image to modify
        reference_sitk: Reference image whose origin to use
        method: 
            - 'copy': Simply copy the reference origin (fastest, assumes aligned)
            - 'center': Center both images (good for different FOVs)
            - 'resample': Resample moving to reference grid (most accurate)
    
    Returns:
        Image with normalized origin
    """
    if method == 'copy':
        # Simply copy the origin from reference
        result = sitk.Image(moving_sitk)
        result.SetOrigin(reference_sitk.GetOrigin())
        return result
        
    elif method == 'center':
        # Compute physical center of both images and align centers
        ref_center = get_physical_center(reference_sitk)
        mov_center = get_physical_center(moving_sitk)
        
        # Shift moving origin so centers align
        origin_shift = np.array(ref_center) - np.array(mov_center)
        new_origin = np.array(moving_sitk.GetOrigin()) + origin_shift
        
        result = sitk.Image(moving_sitk)
        result.SetOrigin(new_origin.tolist())
        return result
        
    elif method == 'resample':
        # Resample moving to exact reference grid
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference_sitk)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(-1000)
        resampler.SetTransform(sitk.Transform())
        return resampler.Execute(moving_sitk)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def get_physical_center(image: sitk.Image) -> Tuple[float, float, float]:
    """Get the physical center of an image."""
    size = np.array(image.GetSize())
    spacing = np.array(image.GetSpacing())
    origin = np.array(image.GetOrigin())
    direction = np.array(image.GetDirection()).reshape(3, 3)
    
    # Physical extent
    extent = (size - 1) * spacing
    
    # Center in physical coordinates
    center = origin + direction @ (extent / 2)
    
    return tuple(center)


def resample_to_common_grid(
        volumes: Dict[str, sitk.Image],
        reference_id: str,
        interpolator: int = sitk.sitkLinear,
        default_value: float = -1000
    ) -> Dict[str, sitk.Image]:
    """
    Resample all volumes to the exact same physical grid as reference.
    
    This ensures:
    - Same origin
    - Same spacing
    - Same size
    - Same direction
    
    Args:
        volumes: Dict of series_id -> sitk.Image
        reference_id: Which series to use as reference
        interpolator: SimpleITK interpolator
        default_value: Value for out-of-bounds
        
    Returns:
        Dict of resampled volumes
    """
    reference = volumes[reference_id]
    result = {}
    
    print(f"\n   Resampling all volumes to reference grid...")
    print(f"   Reference: {reference.GetSize()}, {reference.GetSpacing()}")
    print(f"   Reference origin: {np.round(reference.GetOrigin(), 2)}")
    
    for series_id, vol in volumes.items():
        if series_id == reference_id:
            result[series_id] = vol
            continue
        
        # Check if already matches
        if (vol.GetSize() == reference.GetSize() and
            np.allclose(vol.GetSpacing(), reference.GetSpacing(), rtol=0.01) and
            np.allclose(vol.GetOrigin(), reference.GetOrigin(), atol=0.1)):
            print(f"   {series_id[:30]}... already matches reference")
            result[series_id] = vol
            continue
        
        # Resample to reference grid
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference)
        resampler.SetInterpolator(interpolator)
        resampler.SetDefaultPixelValue(default_value)
        resampler.SetTransform(sitk.Transform())  # Identity
        
        resampled = resampler.Execute(vol)
        result[series_id] = resampled
        
        print(f"   {series_id[:30]}... resampled")
        print(f"      Before: {vol.GetSize()}, origin: {np.round(vol.GetOrigin(), 2)}")
        print(f"      After:  {resampled.GetSize()}, origin: {np.round(resampled.GetOrigin(), 2)}")
    
    return result


def force_common_origin(
        volumes: Dict[str, sitk.Image],
        reference_id: str
    ) -> Dict[str, sitk.Image]:
    """
    Force all volumes to have the same origin as reference.
    
    This is a simple approach that works when:
    - Volumes are already Z-aligned (same slice = same anatomy)
    - Only the origin metadata differs
    
    Args:
        volumes: Dict of series_id -> sitk.Image
        reference_id: Which series to use as reference
        
    Returns:
        Dict of volumes with unified origin
    """
    reference = volumes[reference_id]
    ref_origin = reference.GetOrigin()
    
    print(f"\n   Forcing common origin: {np.round(ref_origin, 2)}")
    
    result = {}
    for series_id, vol in volumes.items():
        if series_id == reference_id:
            result[series_id] = vol
            continue
        
        old_origin = vol.GetOrigin()
        
        # Create new image with reference origin
        new_vol = sitk.Image(vol)
        new_vol.SetOrigin(ref_origin)
        
        result[series_id] = new_vol
        
        print(f"   {series_id[:30]}...")
        print(f"      Origin: {np.round(old_origin, 2)} → {np.round(ref_origin, 2)}")
    
    return result


def verify_physical_alignment(
        volumes: Dict[str, sitk.Image],
        tolerance_mm: float = 1.0
    ) -> Dict:
    """
    Verify all volumes are in the same physical space.
    
    Args:
        volumes: Dict of series_id -> sitk.Image
        tolerance_mm: Maximum allowed origin difference in mm
        
    Returns:
        Verification report
    """
    print(f"\n   PHYSICAL ALIGNMENT VERIFICATION")
    print(f"   {'='*60}")
    
    # Get reference (first volume)
    ref_id = list(volumes.keys())[0]
    ref_vol = volumes[ref_id]
    
    ref_origin = np.array(ref_vol.GetOrigin())
    ref_spacing = np.array(ref_vol.GetSpacing())
    ref_size = ref_vol.GetSize()
    ref_direction = np.array(ref_vol.GetDirection())
    
    print(f"   Reference: {ref_id[:40]}...")
    print(f"   Origin:    {np.round(ref_origin, 2)}")
    print(f"   Spacing:   {np.round(ref_spacing, 3)}")
    print(f"   Size:      {ref_size}")
    
    report = {
        'reference': ref_id,
        'all_match': True,
        'series': {}
    }
    
    for series_id, vol in volumes.items():
        origin = np.array(vol.GetOrigin())
        spacing = np.array(vol.GetSpacing())
        size = vol.GetSize()
        direction = np.array(vol.GetDirection())
        
        # Check differences
        origin_diff = np.linalg.norm(origin - ref_origin)
        spacing_match = np.allclose(spacing, ref_spacing, rtol=0.01)
        size_match = size == ref_size
        direction_match = np.allclose(direction, ref_direction, rtol=0.01)
        
        origin_ok = origin_diff < tolerance_mm
        all_ok = origin_ok and spacing_match and size_match and direction_match
        
        if not all_ok:
            report['all_match'] = False
        
        status = "✓" if all_ok else "✗"
        print(f"\n   {status} {series_id[:40]}...")
        print(f"      Origin:    {np.round(origin, 2)} (diff: {origin_diff:.2f} mm)")
        print(f"      Spacing:   {np.round(spacing, 3)} ({'✓' if spacing_match else '✗'})")
        print(f"      Size:      {size} ({'✓' if size_match else '✗'})")
        
        report['series'][series_id] = {
            'origin': origin.tolist(),
            'origin_diff_mm': float(origin_diff),
            'spacing': spacing.tolist(),
            'size': list(size),
            'origin_ok': origin_ok,
            'spacing_match': spacing_match,
            'size_match': size_match,
            'direction_match': direction_match,
            'all_ok': all_ok
        }
    
    print(f"\n   {'='*60}")
    if report['all_match']:
        print(f"   ✓✓✓ All volumes are properly aligned in physical space")
    else:
        print(f"   ✗✗✗ Physical alignment issues detected!")
    
    return report


# =============================================================================
# UPDATED ALIGNMENT FUNCTIONS WITH ORIGIN NORMALIZATION
# =============================================================================

def align_all_series_in_study_v3(
        study_id: str,
        series_paths: Dict[str, dict],
        reference_series_id: str,
        output_dir: str,
        body_threshold: int = -600,
        min_overlap_slices: int = 30,
        save_visualizations: bool = True,
        force_recompute: bool = False,
        search_region: str = 'upper',
        origin_normalization: str = 'force'  # NEW: 'force', 'resample', or 'none'
    ) -> dict:
    """
    Align ALL series in a study to a common coordinate system.
    
    NEW: Includes origin normalization to ensure all volumes are in same physical space.
    
    Args:
        study_id: Study identifier
        series_paths: Dict mapping series_id to paths and metadata
        reference_series_id: Which series to use as reference
        output_dir: Where to save aligned volumes
        body_threshold: HU threshold for body detection
        min_overlap_slices: Minimum overlap required
        save_visualizations: Whether to create comparison images
        force_recompute: If True, recompute even if already processed
        search_region: 'upper', 'middle', or 'lower' abdomen
        origin_normalization: 
            - 'force': Copy reference origin to all (fast, recommended)
            - 'resample': Resample all to reference grid (accurate but slower)
            - 'none': Keep original origins (not recommended)
        
    Returns:
        Dictionary with alignment results
    """
    # =========================================================================
    # Check if already processed
    # =========================================================================
    viz_dir = os.path.join(output_dir, "alignment_visualizations")
    metadata_path = os.path.join(output_dir, f"{study_id}_alignment_metadata.json")
    
    if not force_recompute and os.path.exists(metadata_path):
        print(f"\n✓ Study {study_id[:30]}... already processed")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return {
            'aligned_volumes': {},
            'aligned_segmentations': {},
            'metadata': metadata,
            'all_match': True,
            'skipped': True
        }
    
    print(f"\n{'='*80}")
    print(f"Multi-Series Z-Alignment: {study_id}")
    print(f"{'='*80}")
    print(f"Reference series: {reference_series_id[:40]}...")
    print(f"Total series: {len(series_paths)}")
    print(f"Origin normalization: {origin_normalization}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # =========================================================================
    # STEP 1: Load reference volume
    # =========================================================================
    print(f"\n[STEP 1] Loading reference volume...")
    
    ref_path = series_paths[reference_series_id]["image"]
    ref_sitk = sitk.ReadImage(ref_path)
    ref_np = sitk.GetArrayFromImage(ref_sitk)
    ref_phase = series_paths[reference_series_id].get('phase', 'Unknown')
    
    print(f"   Reference: {ref_phase}")
    print(f"   Shape: {ref_np.shape}")
    print(f"   Spacing: {ref_sitk.GetSpacing()}")
    print(f"   Origin: {np.round(ref_sitk.GetOrigin(), 2)}")
    
    # Load reference segmentation if available
    ref_seg_path = series_paths[reference_series_id].get("seg")
    ref_seg_sitk = None
    ref_seg_np = None
    
    if ref_seg_path and os.path.exists(ref_seg_path):
        ref_seg_sitk = sitk.ReadImage(ref_seg_path)
        ref_seg_np = sitk.GetArrayFromImage(ref_seg_sitk)
        print(f"   ✓ Reference segmentation: {ref_seg_np.shape}")
    
    # =========================================================================
    # STEP 2: Find optimal reference slice
    # =========================================================================
    print(f"\n[STEP 2] Finding optimal reference slice...")
    
    ref_slice_idx = find_best_reference_slice(
        ref_np,
        body_threshold=body_threshold,
        search_region=search_region
    )
    ref_slice = ref_np[ref_slice_idx]
    
    # =========================================================================
    # STEP 3: Match each series to reference slice
    # =========================================================================
    print(f"\n[STEP 3] Matching all series to reference slice {ref_slice_idx}...")
    
    alignment_data = {}
    needs_resampling = False
    
    # Initialize reference data
    alignment_data[reference_series_id] = {
        'z_offset': 0,
        'matched_idx': ref_slice_idx,
        'reference_idx': ref_slice_idx,
        'score': 1.0,
        'original_shape': ref_np.shape,
        'phase': ref_phase,
        'sitk': ref_sitk,
        'numpy': ref_np,
        'seg_sitk': ref_seg_sitk,
        'seg_numpy': ref_seg_np,
        'needs_resampling': False,
        'original_origin': ref_sitk.GetOrigin(),
        'original_spacing': ref_sitk.GetSpacing()
    }
    
    for series_id, info in series_paths.items():
        if series_id == reference_series_id:
            continue
        
        phase = info.get('phase', 'Unknown')
        print(f"\n   Matching {series_id[:35]}... ({phase})")
        
        # Load moving volume
        moving_sitk = sitk.ReadImage(info["image"])
        moving_np = sitk.GetArrayFromImage(moving_sitk)
        
        print(f"      Shape: {moving_np.shape}")
        print(f"      Spacing: {moving_sitk.GetSpacing()}")
        print(f"      Origin: {np.round(moving_sitk.GetOrigin(), 2)}")
        
        # Check origin difference
        origin_diff = np.linalg.norm(
            np.array(moving_sitk.GetOrigin()) - np.array(ref_sitk.GetOrigin())
        )
        print(f"      Origin diff from reference: {origin_diff:.2f} mm")
        
        # Check if in-plane dimensions match
        in_plane_match = (moving_np.shape[1:] == ref_np.shape[1:])
        
        if not in_plane_match:
            print(f"      ⚠️  In-plane mismatch: {moving_np.shape[1:]} vs {ref_np.shape[1:]}")
            needs_resampling = True
        
        # Define search range
        D_moving = moving_np.shape[0]
        if search_region == 'upper':
            search_start = int(D_moving * 0.50)
            search_end = int(D_moving * 0.95)
        elif search_region == 'middle':
            search_start = int(D_moving * 0.25)
            search_end = int(D_moving * 0.75)
        else:
            search_start = int(D_moving * 0.05)
            search_end = int(D_moving * 0.50)
        
        # Match to reference slice
        matched_idx, score = match_single_slice(
            ref_slice,
            moving_np,
            search_range=(search_start, search_end),
            metric='ncc',
            body_threshold=body_threshold
        )
        
        z_offset = ref_slice_idx - matched_idx
        
        print(f"      ✓ Matched at slice {matched_idx}")
        print(f"      Z-offset: {z_offset}, Score: {score:.4f}")
        
        # Load segmentation if available
        seg_path = info.get("seg")
        seg_sitk = None
        seg_np = None
        
        if seg_path and os.path.exists(seg_path):
            seg_sitk = sitk.ReadImage(seg_path)
            seg_np = sitk.GetArrayFromImage(seg_sitk)
            print(f"      ✓ Segmentation: {seg_np.shape}")
        
        alignment_data[series_id] = {
            'z_offset': z_offset,
            'matched_idx': matched_idx,
            'reference_idx': ref_slice_idx,
            'score': score,
            'original_shape': moving_np.shape,
            'phase': phase,
            'sitk': moving_sitk,
            'numpy': moving_np,
            'seg_sitk': seg_sitk,
            'seg_numpy': seg_np,
            'needs_resampling': not in_plane_match,
            'original_origin': moving_sitk.GetOrigin(),
            'original_spacing': moving_sitk.GetSpacing()
        }
    
    # =========================================================================
    # STEP 4: Compute common overlap region
    # =========================================================================
    print(f"\n[STEP 4] Computing common overlap region...")
    
    valid_ranges = []
    
    for series_id, data in alignment_data.items():
        D = data['original_shape'][0]
        z_offset = data['z_offset']
        
        if z_offset >= 0:
            ref_start = z_offset
            ref_end = z_offset + D
        else:
            ref_start = 0
            ref_end = D + z_offset
        
        ref_start = max(0, ref_start)
        ref_end = min(ref_np.shape[0], ref_end)
        
        valid_ranges.append((ref_start, ref_end))
        print(f"   {alignment_data[series_id]['phase']:20s}: ref[{ref_start}:{ref_end}]")
    
    common_start = max(r[0] for r in valid_ranges)
    common_end = min(r[1] for r in valid_ranges)
    common_depth = common_end - common_start
    
    print(f"\n   ✓ Common overlap: ref[{common_start}:{common_end}] = {common_depth} slices")
    
    if common_depth < min_overlap_slices:
        raise ValueError(f"Insufficient overlap: {common_depth} slices")
    
    # =========================================================================
    # STEP 5: Process all volumes (resample, crop, normalize origin)
    # =========================================================================
    print(f"\n[STEP 5] Processing all volumes...")
    
    aligned_volumes = {}
    aligned_segmentations = {}
    
    # Process reference first
    processing_order = [reference_series_id] + [
        sid for sid in alignment_data.keys() if sid != reference_series_id
    ]
    
    # Store the FINAL reference origin (after cropping)
    final_reference_origin = None
    
    for series_id in processing_order:
        data = alignment_data[series_id]
        phase = data['phase']
        
        print(f"\n   Processing {phase}...")
        
        vol_sitk = data['sitk']
        vol_np = data['numpy']
        seg_sitk = data.get('seg_sitk')
        seg_np = data.get('seg_numpy')
        z_offset = data['z_offset']
        
        # -----------------------------------------------------------------
        # Resample in-plane if needed
        # -----------------------------------------------------------------
        if data['needs_resampling']:
            print(f"      Resampling in-plane to reference...")
            vol_sitk = resample_volume_inplane(vol_sitk, ref_sitk)
            vol_np = sitk.GetArrayFromImage(vol_sitk)
            
            if seg_sitk is not None:
                seg_sitk = resample_volume_inplane(
                    seg_sitk, ref_sitk, 
                    interpolator=sitk.sitkNearestNeighbor,
                    default_value=0
                )
                seg_np = sitk.GetArrayFromImage(seg_sitk)
            
            data['resampled_shape'] = vol_np.shape
        
        # -----------------------------------------------------------------
        # Calculate crop range
        # -----------------------------------------------------------------
        if series_id == reference_series_id:
            crop_start = common_start
            crop_end = common_end
        else:
            crop_start = common_start - z_offset
            crop_end = common_end - z_offset
            crop_start = max(0, crop_start)
            crop_end = min(vol_np.shape[0], crop_end)
        
        actual_depth = crop_end - crop_start
        print(f"      Cropping: [{crop_start}:{crop_end}] ({actual_depth} slices)")
        
        # -----------------------------------------------------------------
        # Crop image
        # -----------------------------------------------------------------
        cropped_np = vol_np[crop_start:crop_end]
        cropped_sitk = sitk.GetImageFromArray(cropped_np)
        
        # Set spacing and direction (same as original)
        cropped_sitk.SetSpacing(vol_sitk.GetSpacing())
        cropped_sitk.SetDirection(vol_sitk.GetDirection())
        
        # Compute cropped origin
        cropped_origin = compute_new_origin(vol_sitk, crop_start)
        cropped_sitk.SetOrigin(cropped_origin)
        
        print(f"      Cropped origin: {np.round(cropped_origin, 2)}")
        
        # -----------------------------------------------------------------
        # Store reference origin for normalization
        # -----------------------------------------------------------------
        if series_id == reference_series_id:
            final_reference_origin = cropped_origin
            print(f"      ★ This is the REFERENCE origin")
        
        aligned_volumes[series_id] = cropped_sitk
        data['crop_range'] = (crop_start, crop_end)
        data['aligned_shape'] = cropped_np.shape
        data['cropped_origin'] = cropped_origin
        
        # -----------------------------------------------------------------
        # Crop segmentation
        # -----------------------------------------------------------------
        if seg_np is not None:
            seg_cropped_np = seg_np[crop_start:crop_end]
            seg_cropped_sitk = sitk.GetImageFromArray(seg_cropped_np.astype(np.uint8))
            seg_cropped_sitk.SetSpacing(cropped_sitk.GetSpacing())
            seg_cropped_sitk.SetDirection(cropped_sitk.GetDirection())
            seg_cropped_sitk.SetOrigin(cropped_origin)
            
            aligned_segmentations[series_id] = seg_cropped_sitk
        else:
            aligned_segmentations[series_id] = None
    
    # =========================================================================
    # STEP 6: ORIGIN NORMALIZATION (KEY FIX!)
    # =========================================================================
    print(f"\n[STEP 6] Normalizing origins (method: {origin_normalization})...")
    
    if origin_normalization == 'force':
        # Force all volumes to have the same origin as reference
        aligned_volumes = force_common_origin(aligned_volumes, reference_series_id)
        aligned_segmentations = force_common_origin_segmentations(
            aligned_segmentations, aligned_volumes, reference_series_id
        )
        
    elif origin_normalization == 'resample':
        # Resample all to exact reference grid
        aligned_volumes = resample_to_common_grid(aligned_volumes, reference_series_id)
        aligned_segmentations = resample_segmentations_to_common_grid(
            aligned_segmentations, aligned_volumes, reference_series_id
        )
        
    elif origin_normalization == 'none':
        print(f"   ⚠️  Skipping origin normalization (not recommended!)")
    
    # =========================================================================
    # STEP 7: Verify physical alignment
    # =========================================================================
    print(f"\n[STEP 7] Verifying physical alignment...")
    
    alignment_report = verify_physical_alignment(aligned_volumes, tolerance_mm=1.0)
    
    # =========================================================================
    # STEP 8: Save aligned volumes
    # =========================================================================
    print(f"\n[STEP 8] Saving aligned volumes...")
    
    for series_id in processing_order:
        data = alignment_data[series_id]
        phase = data['phase']
        
        vol_sitk = aligned_volumes[series_id]
        
        # Save image
        output_path = os.path.join(output_dir, f"{study_id}_{series_id}_aligned.nii.gz")
        sitk.WriteImage(vol_sitk, output_path)
        print(f"   ✓ Saved {phase}: {os.path.basename(output_path)}")
        print(f"      Size: {vol_sitk.GetSize()}, Origin: {np.round(vol_sitk.GetOrigin(), 2)}")
        
        data['output_path'] = output_path
        
        # Save segmentation
        if aligned_segmentations.get(series_id) is not None:
            seg_sitk = aligned_segmentations[series_id]
            seg_output_path = os.path.join(
                output_dir, 
                f"{study_id}_{series_id}_aligned_seg.nii.gz"
            )
            sitk.WriteImage(seg_sitk, seg_output_path)
            print(f"      + Segmentation saved")
            data['seg_output_path'] = seg_output_path
    
    # =========================================================================
    # STEP 9: Create visualizations
    # =========================================================================
    if save_visualizations:
        print(f"\n[STEP 9] Creating visualizations...")
        
        os.makedirs(viz_dir, exist_ok=True)
        
        # Per-series comparison
        for series_id, data in alignment_data.items():
            if series_id == reference_series_id:
                continue
            
            try:
                create_series_alignment_comparison_v3(
                    series_id=series_id,
                    series_data=data,
                    reference_data=alignment_data[reference_series_id],
                    aligned_volumes=aligned_volumes,
                    aligned_segmentations=aligned_segmentations,
                    output_dir=viz_dir,
                    study_id=study_id
                )
            except Exception as e:
                print(f"      ⚠️  Visualization failed: {e}")
        
        # Origin comparison visualization
        try:
            create_origin_comparison_visualization(
                alignment_data=alignment_data,
                aligned_volumes=aligned_volumes,
                output_dir=viz_dir,
                study_id=study_id
            )
        except Exception as e:
            print(f"      ⚠️  Origin visualization failed: {e}")
    
    # =========================================================================
    # STEP 10: Save metadata
    # =========================================================================
    print(f"\n[STEP 10] Saving metadata...")
    
    metadata = {
        'study_id': study_id,
        'reference_series': reference_series_id,
        'reference_slice_idx': ref_slice_idx,
        'origin_normalization': origin_normalization,
        'final_origin': list(aligned_volumes[reference_series_id].GetOrigin()),
        'final_spacing': list(aligned_volumes[reference_series_id].GetSpacing()),
        'final_size': list(aligned_volumes[reference_series_id].GetSize()),
        'common_overlap': {
            'start': int(common_start),
            'end': int(common_end),
            'depth': int(common_depth)
        },
        'alignment_verified': alignment_report['all_match'],
        'series': {}
    }
    
    for series_id, data in alignment_data.items():
        has_seg = aligned_segmentations.get(series_id) is not None
        
        metadata['series'][series_id] = {
            'phase': data['phase'],
            'z_offset': int(data['z_offset']),
            'matched_idx': int(data['matched_idx']),
            'similarity_score': float(data['score']),
            'original_shape': [int(x) for x in data['original_shape']],
            'original_origin': [float(x) for x in data['original_origin']],
            'aligned_shape': [int(x) for x in data['aligned_shape']],
            'crop_range': [int(x) for x in data['crop_range']],
            'has_segmentation': has_seg,
            'output_path': data.get('output_path', ''),
            'seg_output_path': data.get('seg_output_path', '') if has_seg else None
        }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   ✓ Metadata saved: {os.path.basename(metadata_path)}")
    
    # =========================================================================
    # Final summary
    # =========================================================================
    print(f"\n{'='*80}")
    print("ALIGNMENT COMPLETE")
    print(f"{'='*80}")
    
    ref_vol = aligned_volumes[reference_series_id]
    print(f"\nFinal unified grid:")
    print(f"   Size:    {ref_vol.GetSize()}")
    print(f"   Spacing: {np.round(ref_vol.GetSpacing(), 3)}")
    print(f"   Origin:  {np.round(ref_vol.GetOrigin(), 2)}")
    
    print(f"\nSeries summary:")
    for series_id, vol in aligned_volumes.items():
        phase = alignment_data[series_id]['phase']
        origin = np.round(vol.GetOrigin(), 2)
        has_seg = "📄" if aligned_segmentations.get(series_id) is not None else "  "
        print(f"   {has_seg} {phase:20s}: origin = {origin}")
    
    all_match = alignment_report['all_match']
    print(f"\n{'✓✓✓' if all_match else '✗✗✗'} Physical alignment: "
          f"{'VERIFIED' if all_match else 'FAILED'}")
    
    return {
        'aligned_volumes': aligned_volumes,
        'aligned_segmentations': aligned_segmentations,
        'metadata': metadata,
        'all_match': all_match,
        'alignment_report': alignment_report,
        'skipped': False
    }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def resample_volume_inplane(
        moving_sitk: sitk.Image,
        reference_sitk: sitk.Image,
        interpolator: int = sitk.sitkLinear,
        default_value: float = -1000
    ) -> sitk.Image:
    """Resample moving volume to match reference in-plane, keeping Z depth."""
    
    ref_spacing = reference_sitk.GetSpacing()
    ref_size = reference_sitk.GetSize()
    
    # New size: reference X,Y but keep moving Z
    new_size = (
        ref_size[0],
        ref_size[1],
        moving_sitk.GetSize()[2]
    )
    
    # New spacing: reference X,Y but keep moving Z
    new_spacing = (
        ref_spacing[0],
        ref_spacing[1],
        moving_sitk.GetSpacing()[2]
    )
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(moving_sitk.GetOrigin())
    resampler.SetOutputDirection(moving_sitk.GetDirection())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_value)
    resampler.SetTransform(sitk.Transform())
    
    return resampler.Execute(moving_sitk)


def compute_new_origin(original_sitk: sitk.Image, crop_start: int) -> List[float]:
    """Compute new origin after cropping slices from beginning."""
    
    origin = np.array(original_sitk.GetOrigin())
    spacing = np.array(original_sitk.GetSpacing())
    direction = np.array(original_sitk.GetDirection()).reshape(3, 3)
    z_vector = direction[:, 2]
    shift = crop_start * spacing[2] * z_vector
    new_origin = origin + shift
    
    return new_origin.tolist()


def force_common_origin_segmentations(
        segmentations: Dict[str, Optional[sitk.Image]],
        volumes: Dict[str, sitk.Image],
        reference_id: str
    ) -> Dict[str, Optional[sitk.Image]]:
    """Force segmentations to have same origin as their corresponding volumes."""
    
    result = {}
    for series_id, seg in segmentations.items():
        if seg is None:
            result[series_id] = None
            continue
        
        vol = volumes[series_id]
        new_seg = sitk.Image(seg)
        new_seg.SetOrigin(vol.GetOrigin())
        new_seg.SetSpacing(vol.GetSpacing())
        new_seg.SetDirection(vol.GetDirection())
        result[series_id] = new_seg
    
    return result


def resample_segmentations_to_common_grid(
        segmentations: Dict[str, Optional[sitk.Image]],
        volumes: Dict[str, sitk.Image],
        reference_id: str
    ) -> Dict[str, Optional[sitk.Image]]:
    """Resample segmentations to match their corresponding resampled volumes."""
    
    reference_vol = volumes[reference_id]
    result = {}
    
    for series_id, seg in segmentations.items():
        if seg is None:
            result[series_id] = None
            continue
        
        vol = volumes[series_id]
        
        # Resample segmentation to match volume
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(vol)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(sitk.Transform())
        
        result[series_id] = resampler.Execute(seg)
    
    return result


def find_best_reference_slice(
        reference_vol: np.ndarray,
        body_threshold: int = -600,
        search_region: str = 'upper'
    ) -> int:
    """Find optimal reference slice in the volume."""
    
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
    
    print(f"   ✓ Best reference slice: {best_idx}")
    
    return best_idx


def match_single_slice(
        reference_slice: np.ndarray,
        moving_vol: np.ndarray,
        search_range: Tuple[int, int],
        metric: str = 'ncc',
        body_threshold: int = -600
    ) -> Tuple[int, float]:
    """Match a single reference slice to best slice in moving volume."""
    
    target_shape = reference_slice.shape
    best_idx = (search_range[0] + search_range[1]) // 2
    best_score = -np.inf
    
    ref_norm = (reference_slice - reference_slice.mean()) / (reference_slice.std() + 1e-8)
    
    for m_idx in range(search_range[0], min(search_range[1], moving_vol.shape[0])):
        moving_slice = moving_vol[m_idx]
        
        if (moving_slice > body_threshold).sum() < 1000:
            continue
        
        if moving_slice.shape != target_shape:
            moving_slice = cv2.resize(
                moving_slice.astype(np.float32),
                (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
        
        try:
            m_norm = (moving_slice - moving_slice.mean()) / (moving_slice.std() + 1e-8)
            score = np.mean(ref_norm * m_norm)
            
            if score > best_score:
                best_score = score
                best_idx = m_idx
        except:
            continue
    
    return best_idx, best_score


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_origin_comparison_visualization(
        alignment_data: dict,
        aligned_volumes: dict,
        output_dir: str,
        study_id: str
    ):
    """Create visualization comparing origins before and after normalization."""
    
    num_series = len(aligned_volumes)
    
    fig, axes = plt.subplots(2, num_series, figsize=(5*num_series, 10))
    if num_series == 1:
        axes = axes.reshape(2, 1)
    
    for idx, (series_id, vol) in enumerate(sorted(aligned_volumes.items())):
        data = alignment_data[series_id]
        phase = data['phase']
        
        vol_np = sitk.GetArrayFromImage(vol)
        mid_slice = vol_np.shape[0] // 2
        
        # Row 1: Image
        axes[0, idx].imshow(vol_np[mid_slice], cmap='gray', vmin=-1000, vmax=500)
        axes[0, idx].set_title(f'{phase}\n{vol.GetSize()}', fontsize=10, fontweight='bold')
        axes[0, idx].axis('off')
        
        # Row 2: Origin info
        axes[1, idx].axis('off')
        
        orig_origin = np.array(data['original_origin'])
        final_origin = np.array(vol.GetOrigin())
        origin_change = np.linalg.norm(final_origin - orig_origin)
        
        info_text = f"""
Original Origin:
  {np.round(orig_origin, 1)}

Final Origin:
  {np.round(final_origin, 1)}

Change: {origin_change:.1f} mm

Z-offset: {data['z_offset']} slices
Score: {data['score']:.3f}
"""
        axes[1, idx].text(
            0.1, 0.9, info_text, 
            transform=axes[1, idx].transAxes,
            fontsize=9, 
            verticalalignment='top',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
        )
    
    fig.suptitle(
        f'Origin Normalization: {study_id[:50]}...\n'
        f'All volumes now share the same origin',
        fontsize=12, fontweight='bold'
    )
    
    output_path = os.path.join(output_dir, 'origin_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"      ✓ Origin comparison saved: {os.path.basename(output_path)}")


def create_series_alignment_comparison_v3(
        series_id: str,
        series_data: dict,
        reference_data: dict,
        aligned_volumes: dict,
        aligned_segmentations: dict,
        output_dir: str,
        study_id: str
    ):
    """Create alignment comparison visualization."""
    
    phase = series_data['phase']
    ref_phase = reference_data['phase']
    
    # Get aligned volumes
    aligned_vol = aligned_volumes[series_id]
    ref_aligned_vol = aligned_volumes[list(aligned_volumes.keys())[0]]
    
    aligned_np = sitk.GetArrayFromImage(aligned_vol)
    ref_aligned_np = sitk.GetArrayFromImage(ref_aligned_vol)
    
    mid_slice = aligned_np.shape[0] // 2
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Aligned images
    axes[0, 0].imshow(aligned_np[mid_slice], cmap='gray', vmin=-1000, vmax=500)
    axes[0, 0].set_title(f'{phase}\nSlice {mid_slice}', fontsize=10)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(ref_aligned_np[mid_slice], cmap='gray', vmin=-1000, vmax=500)
    axes[0, 1].set_title(f'{ref_phase} (Reference)\nSlice {mid_slice}', fontsize=10)
    axes[0, 1].axis('off')
    
    # Difference
    diff = np.abs(aligned_np[mid_slice] - ref_aligned_np[mid_slice])
    im = axes[0, 2].imshow(diff, cmap='hot', vmin=0, vmax=500)
    axes[0, 2].set_title(f'Difference\nMAE: {diff.mean():.1f} HU', fontsize=10)
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # Row 2: Multiple slices
    for i, frac in enumerate([0.25, 0.5, 0.75]):
        slice_idx = int(aligned_np.shape[0] * frac)
        axes[1, i].imshow(aligned_np[slice_idx], cmap='gray', vmin=-1000, vmax=500)
        axes[1, i].set_title(f'{int(frac*100)}% - Slice {slice_idx}', fontsize=9)
        axes[1, i].axis('off')
    
    # Info
    origin = np.round(aligned_vol.GetOrigin(), 2)
    ref_origin = np.round(ref_aligned_vol.GetOrigin(), 2)
    
    fig.text(
        0.02, 0.02,
        f"This origin: {origin} | Reference origin: {ref_origin} | "
        f"Z-offset: {series_data['z_offset']} | Score: {series_data['score']:.3f}",
        fontsize=9,
        transform=fig.transFigure
    )
    
    fig.suptitle(f'Alignment: {phase} → {ref_phase}', fontsize=12, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    safe_id = series_id.replace('.', '_')[:40]
    output_path = os.path.join(output_dir, f'{safe_id}_{phase.replace(" ", "_")}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# POST-PROCESSING: FIX ALREADY ALIGNED VOLUMES
# =============================================================================

def fix_origins_in_directory(
        study_dir: str,
        reference_pattern: str = "*Non-contrast*_aligned.nii.gz"
    ) -> bool:
    """
    Fix origins of already-aligned volumes in a directory.
    
    Use this to fix volumes that were aligned with the old code.
    
    Args:
        study_dir: Directory containing aligned volumes
        reference_pattern: Glob pattern for reference volume
        
    Returns:
        True if successful
    """
    import glob
    
    # Find all aligned volumes
    all_files = glob.glob(os.path.join(study_dir, "*_aligned.nii.gz"))
    image_files = [f for f in all_files if '_seg' not in f]
    
    if not image_files:
        print(f"No aligned volumes found in {study_dir}")
        return False
    
    # Find reference
    ref_files = glob.glob(os.path.join(study_dir, reference_pattern))
    if not ref_files:
        # Use first file as reference
        ref_files = [image_files[0]]
    
    ref_path = ref_files[0]
    ref_sitk = sitk.ReadImage(ref_path)
    ref_origin = ref_sitk.GetOrigin()
    
    print(f"\nFixing origins in: {study_dir}")
    print(f"Reference: {os.path.basename(ref_path)}")
    print(f"Reference origin: {np.round(ref_origin, 2)}")
    
    for image_path in image_files:
        if image_path == ref_path:
            continue
        
        # Load image
        img = sitk.ReadImage(image_path)
        old_origin = img.GetOrigin()
        
        # Check if fix needed
        if np.allclose(old_origin, ref_origin, atol=0.1):
            print(f"  ✓ {os.path.basename(image_path)}: already OK")
            continue
        
        # Fix origin
        img.SetOrigin(ref_origin)
        
        # Save (backup original first)
        backup_path = image_path.replace('.nii.gz', '_old_origin.nii.gz')
        os.rename(image_path, backup_path)
        sitk.WriteImage(img, image_path)
        
        print(f"  ✓ {os.path.basename(image_path)}")
        print(f"      {np.round(old_origin, 2)} → {np.round(ref_origin, 2)}")
        
        # Fix corresponding segmentation
        seg_path = image_path.replace('_aligned.nii.gz', '_aligned_seg.nii.gz')
        if os.path.exists(seg_path):
            seg = sitk.ReadImage(seg_path)
            seg.SetOrigin(ref_origin)
            
            seg_backup = seg_path.replace('.nii.gz', '_old_origin.nii.gz')
            os.rename(seg_path, seg_backup)
            sitk.WriteImage(seg, seg_path)
            
            print(f"      + Fixed segmentation")
    
    print(f"\n✓ Origins fixed!")
    return True


def batch_fix_origins(aligned_dir: str):
    """Fix origins for all studies in a directory."""
    
    study_dirs = [d for d in os.listdir(aligned_dir) 
                  if os.path.isdir(os.path.join(aligned_dir, d))]
    
    print(f"\n{'='*80}")
    print(f"BATCH ORIGIN FIX: {len(study_dirs)} studies")
    print(f"{'='*80}")
    
    for study_id in study_dirs:
        study_path = os.path.join(aligned_dir, study_id)
        try:
            fix_origins_in_directory(study_path)
        except Exception as e:
            print(f"  ✗ {study_id}: {e}")


# =============================================================================
# MAIN
# =============================================================================
# =============================================================================
# BATCH ALIGNMENT FUNCTIONS
# =============================================================================

def scan_cropped_directory(cropped_dir: str) -> Dict[str, Dict[str, dict]]:
    """
    Scan flat cropped directory and group files by study_id.
    
    Input format:
        {study_id}_{series_id}_crop.nii.gz
        {study_id}_{series_id}_seg.nii.gz
    
    Returns:
        Dict[study_id][series_id] = {
            'image_path': path,
            'seg_path': path or None
        }
    """
    import glob
    
    catalog = {}
    
    if not os.path.exists(cropped_dir):
        print(f"❌ Directory not found: {cropped_dir}")
        return catalog
    
    print(f"\n📂 Scanning {cropped_dir}...")
    
    # Find all crop files (not seg files)
    pattern = os.path.join(cropped_dir, "*_crop.nii.gz")
    crop_files = glob.glob(pattern)
    
    print(f"   Found {len(crop_files)} cropped volumes")
    
    for crop_path in crop_files:
        basename = os.path.basename(crop_path)
        
        # Parse filename: {study_id}_{series_id}_crop.nii.gz
        # Study ID and Series ID both contain dots and underscores
        # Pattern: everything before last _crop.nii.gz, split by first underscore after study pattern
        
        name_without_suffix = basename.replace("_crop.nii.gz", "")
        
        # Find the split point between study_id and series_id
        # Both IDs typically start with "1.2.840" or similar
        # Strategy: Look for the pattern where we have two UIDs separated by underscore
        
        parts = name_without_suffix.split("_")
        
        # Reconstruct: first UID is study_id, rest is series_id
        # UIDs typically look like: 1.2.840.113619.2.359.3.2831208971.47.1588634888.707
        
        # Find where study_id ends and series_id begins
        # Usually after a number followed by underscore and another "1." pattern
        
        study_id = None
        series_id = None
        
        for i in range(1, len(parts)):
            potential_study = "_".join(parts[:i])
            potential_series = "_".join(parts[i:])
            
            # Check if both look like UIDs (contain multiple dots)
            if potential_study.count(".") >= 3 and potential_series.count(".") >= 3:
                study_id = potential_study
                series_id = potential_series
                break
        
        if study_id is None or series_id is None:
            print(f"   ⚠️  Could not parse: {basename}")
            continue
        
        # Check for segmentation file
        seg_path = crop_path.replace("_crop.nii.gz", "_seg.nii.gz")
        has_seg = os.path.exists(seg_path)
        
        # Add to catalog
        if study_id not in catalog:
            catalog[study_id] = {}
        
        catalog[study_id][series_id] = {
            'image_path': crop_path,
            'seg_path': seg_path if has_seg else None,
            'has_seg': has_seg
        }
    
    # Summary
    total_studies = len(catalog)
    total_volumes = sum(len(s) for s in catalog.values())
    total_with_seg = sum(
        1 for study in catalog.values()
        for series in study.values()
        if series['has_seg']
    )
    
    print(f"   Studies: {total_studies}")
    print(f"   Total volumes: {total_volumes}")
    print(f"   With segmentation: {total_with_seg}")
    
    return catalog


def build_series_paths_for_study(
        study_id: str,
        study_files: Dict[str, dict],
        labels_df: pd.DataFrame
    ) -> Tuple[Dict[str, dict], Optional[str]]:
    """
    Build series_paths dict for a study and identify reference (non-contrast).
    
    Returns:
        (series_paths dict, reference_series_id or None)
    """
    series_paths = {}
    reference_series_id = None
    
    # Get labels for this study
    study_labels = labels_df[labels_df["StudyInstanceUID"] == study_id]
    
    for series_id, file_info in study_files.items():
        # Find phase label
        phase = "Unknown"
        
        for _, row in study_labels.iterrows():
            label_series_id = row["SeriesInstanceUID"]
            
            # Exact match
            if label_series_id == series_id:
                phase = row["Label"]
                break
            
            # Fuzzy match (handle slight variations)
            if (series_id in label_series_id or 
                label_series_id in series_id or
                series_id.rsplit('.', 1)[0] == label_series_id.rsplit('.', 1)[0]):
                phase = row["Label"]
                break
        
        series_paths[series_id] = {
            'image': file_info['image_path'],
            'phase': phase,
            'seg': file_info['seg_path']
        }
        
        # Check if this is non-contrast reference
        if phase == "Non-contrast" and reference_series_id is None:
            reference_series_id = series_id
    
    return series_paths, reference_series_id


def align_all_studies(
        cropped_dir: str,
        output_dir: str,
        labels_csv: str,
        origin_normalization: str = 'force',
        force_recompute: bool = False,
        min_overlap_slices: int = 30,
        search_region: str = 'upper'
    ):
    """
    Batch align all studies.
    
    Args:
        cropped_dir: Input directory with cropped volumes (flat structure)
        output_dir: Output directory (will create study subfolders)
        labels_csv: Path to labels CSV
        origin_normalization: 'force', 'resample', or 'none'
        force_recompute: Reprocess even if output exists
        min_overlap_slices: Minimum Z overlap required
        search_region: 'upper', 'middle', or 'lower'
    """
    print(f"\n{'='*80}")
    print("BATCH ALIGNMENT")
    print(f"{'='*80}")
    print(f"Input:  {cropped_dir}")
    print(f"Output: {output_dir}")
    print(f"Origin normalization: {origin_normalization}")
    print(f"Force recompute: {force_recompute}")
    
    # Load labels
    labels_df = pd.read_csv(labels_csv)
    print(f"Labels: {len(labels_df)} entries")
    
    # Scan input directory
    catalog = scan_cropped_directory(cropped_dir)
    
    if len(catalog) == 0:
        print("❌ No studies found!")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each study
    successful = 0
    failed = 0
    skipped = 0
    
    failed_studies = []
    
    for idx, (study_id, study_files) in enumerate(sorted(catalog.items()), 1):
        print(f"\n[{idx}/{len(catalog)}] {study_id[:50]}...")
        
        # Check if already processed
        study_output_dir = os.path.join(output_dir, study_id)
        metadata_path = os.path.join(study_output_dir, f"{study_id}_alignment_metadata.json")
        
        if not force_recompute and os.path.exists(metadata_path):
            print(f"   ⏭️  Already processed, skipping")
            skipped += 1
            continue
        
        # Build series_paths
        series_paths, reference_series_id = build_series_paths_for_study(
            study_id, study_files, labels_df
        )
        
        if reference_series_id is None:
            print(f"   ⚠️  No non-contrast series found")
            failed += 1
            failed_studies.append((study_id, "No non-contrast"))
            continue
        
        if len(series_paths) < 2:
            print(f"   ⚠️  Only {len(series_paths)} series, need at least 2")
            failed += 1
            failed_studies.append((study_id, f"Only {len(series_paths)} series"))
            continue
        
        # Print series info
        print(f"   Series: {len(series_paths)}")
        for sid, info in series_paths.items():
            is_ref = "★" if sid == reference_series_id else " "
            has_seg = "📄" if info['seg'] else "  "
            print(f"      {is_ref}{has_seg} {info['phase']}: {sid[:40]}...")
        
        # Run alignment
        try:
            result = align_all_series_in_study_v3(
                study_id=study_id,
                series_paths=series_paths,
                reference_series_id=reference_series_id,
                output_dir=study_output_dir,
                origin_normalization=origin_normalization,
                force_recompute=True,  # Already checked above
                min_overlap_slices=min_overlap_slices,
                search_region=search_region
            )
            
            if result.get('all_match', False):
                successful += 1
                print(f"   ✓ Success")
            else:
                failed += 1
                failed_studies.append((study_id, "Alignment mismatch"))
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            failed += 1
            failed_studies.append((study_id, str(e)[:50]))
    
    # Summary
    print(f"\n{'='*80}")
    print("BATCH ALIGNMENT COMPLETE")
    print(f"{'='*80}")
    print(f"✓ Successful: {successful}/{len(catalog)}")
    print(f"⏭️  Skipped:   {skipped}/{len(catalog)}")
    print(f"❌ Failed:    {failed}/{len(catalog)}")
    
    if failed_studies:
        print(f"\nFailed studies:")
        for study_id, reason in failed_studies[:10]:
            print(f"   - {study_id[:40]}... → {reason}")
        if len(failed_studies) > 10:
            print(f"   ... and {len(failed_studies) - 10} more")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    
    
    # Configuration
    CROPPED_DIR = "../../ncct_cect/vindr_ds/cropped_volumes"
    OUTPUT_DIR = "../../ncct_cect/vindr_ds/aligned_cases"
    LABELS_CSV = "../../ncct_cect/vindr_ds/labels.csv"
    
    # Run batch alignment
    align_all_studies(
        cropped_dir=CROPPED_DIR,
        output_dir=OUTPUT_DIR,
        labels_csv=LABELS_CSV,
        origin_normalization='force',
        force_recompute=False,  # Skip already processed
        min_overlap_slices=30,
        search_region='upper'
    )

# if __name__ == "__main__":
#     import pandas as pd
    
#     # Configuration
#     MAIN_PATH = "../../ncct_cect/vindr_ds/cropped_volumes/"
    
#     # Option 1: Fix already-aligned volumes
#     print("\n" + "="*80)
#     print("FIXING EXISTING ALIGNED VOLUMES")
#     print("="*80)
    
#     # batch_fix_origins(MAIN_PATH + "aligned_volumes")
    
#     # Option 2: Re-run alignment with origin normalization
#     print("\n" + "="*80)
#     print("RE-RUNNING ALIGNMENT WITH ORIGIN NORMALIZATION")
#     print("="*80)
    
#     # Example single study
#     study_id = "1.2.840.113619.2.359.3.2831208971.47.1588634888.707"
#     series_paths = {
#         "series1": {
#             "image": MAIN_PATH + "1.2.840.113619.2.359.3.2831208971.47.1588634888.707_1.2.840.113619.2.359.3.2831208971.47.1588634888.712_crop.nii.gz",
#             "phase": "Non-contrast",
#             "seg": MAIN_PATH + "1.2.840.113619.2.359.3.2831208971.47.1588634888.707_1.2.840.113619.2.359.3.2831208971.47.1588634888.712_seg.nii.gz"
#         },
#         "series2": {
#             "image": MAIN_PATH + "1.2.840.113619.2.359.3.2831208971.47.1588634888.707_1.2.840.113619.2.359.3.2831208971.47.1588634888.806.4198401_crop.nii.gz", 
#             "phase": "Arterial",
#             "seg": MAIN_PATH + "1.2.840.113619.2.359.3.2831208971.47.1588634888.707_1.2.840.113619.2.359.3.2831208971.47.1588634888.806.4198401_seg.nii.gz"
#         },
#         "series3": {
#             "image": MAIN_PATH + "1.2.840.113619.2.359.3.2831208971.47.1588634888.707_1.2.840.113619.2.359.3.2831208971.47.1588634888.806.4206593_crop.nii.gz", 
#             "phase": "Venous",
#             "seg": MAIN_PATH + "1.2.840.113619.2.359.3.2831208971.47.1588634888.707_1.2.840.113619.2.359.3.2831208971.47.1588634888.806.4206593_seg.nii.gz"
#         },
#         "series4": {
#             "image": MAIN_PATH + "1.2.840.113619.2.359.3.2831208971.47.1588634888.707_1.2.840.113619.2.359.3.2831208971.47.1588634888.806.4202497_crop.nii.gz",
#             "phase": "other",
#             "seg": MAIN_PATH + "1.2.840.113619.2.359.3.2831208971.47.1588634888.707_1.2.840.113619.2.359.3.2831208971.47.1588634888.806.4202497_seg.nii.gz"
#         }   

#     }
    
#     result = align_all_series_in_study_v3(
#         study_id=study_id,
#         series_paths=series_paths,
#         reference_series_id="series1",
#         output_dir="./aligned_output",
#         origin_normalization='force',  # KEY: Force common origin
#         force_recompute=True
#     )