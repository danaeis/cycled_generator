import os
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from skimage import exposure
from tqdm import tqdm
import shutil


# def flip_volume_z_axis(vol_sitk: sitk.Image) -> sitk.Image:
#     """
#     Flip volume along Z-axis (upside down).
#     Upper abdomen becomes first slices.
    
#     Returns:
#         Flipped volume with correct metadata
#     """
#     vol_np = sitk.GetArrayFromImage(vol_sitk)
    
#     # Flip along Z (depth) axis
#     flipped_np = np.flip(vol_np, axis=0).copy()
    
#     # Create new image
#     flipped_sitk = sitk.GetImageFromArray(flipped_np)
    
#     # Copy spacing and direction
#     flipped_sitk.SetSpacing(vol_sitk.GetSpacing())
#     flipped_sitk.SetDirection(vol_sitk.GetDirection())
    
#     # Update origin to reflect flip
#     origin = np.array(vol_sitk.GetOrigin())
#     spacing = np.array(vol_sitk.GetSpacing())
#     direction_matrix = np.array(vol_sitk.GetDirection()).reshape(3, 3)
#     z_unit_vector = direction_matrix[:, 2]
    
#     # Shift origin by total Z extent
#     depth = vol_np.shape[0]
#     origin_shift = (depth - 1) * spacing[2] * z_unit_vector
#     new_origin = origin + origin_shift
    
#     flipped_sitk.SetOrigin(new_origin.tolist())
    
#     return flipped_sitk


# def find_slice_correspondence_arc(
#         fixed_vol: np.ndarray,
#         moving_vol: np.ndarray,
#         body_threshold: int = -600,
#         search_range: int = 50,
#         metric: str = 'ncc'  # 'ncc', 'ssim', or 'mi'
#     ) -> tuple[int, int, float]:
#     """
#     Find best matching slice pair between two volumes.
#     Handles different in-plane dimensions by resampling.
    
#     Args:
#         fixed_vol: Reference volume [D, H, W]
#         moving_vol: Volume to align [D, H, W]
#         body_threshold: HU threshold for body mask
#         search_range: How many slices to search (from middle)
#         metric: 'ncc' (fast), 'ssim' (robust), or 'mi' (multi-modal)
    
#     Returns:
#         (fixed_slice_idx, moving_slice_idx, similarity_score)
#     """
#     from skimage.metrics import structural_similarity as ssim
#     import cv2
    
#     # Get target dimensions from fixed volume
#     target_shape = (fixed_vol.shape[1], fixed_vol.shape[2])  # (H, W)
    
#     # Focus on middle slices (more likely to have anatomy)
#     D_fixed = fixed_vol.shape[0]
#     D_moving = moving_vol.shape[0]
#     print(f"   Fixed depth: {D_fixed}, Moving depth: {D_moving}")   
#     # # moving_mid = D_moving // 2
#     # # # With this (much better for abdomen):
#     # fixed_mid = int(D_fixed * 0.75)   # 75th percentile → upper abdomen
#     # moving_mid = int(D_moving * 0.75)

#     # # Search range around middle
#     # fixed_range = range(
#     #     max(0, fixed_mid - search_range),
#     #     min(D_fixed, fixed_mid + search_range)
#     # )
#     # moving_range = range(
#     #     max(0, moving_mid - search_range),
#     #     min(D_moving, moving_mid + search_range)
#     # )
#     # For flipped volumes: search from beginning (upper abdomen is now at slice 0)
#     # For normal volumes: search from 75% position
#     use_flipped_search = True  # Set based on whether volumes are flipped

#     if use_flipped_search:
#         # Search first 20% of slices (upper abdomen in flipped volumes)
#         fixed_mid = int(D_fixed * 0.10)
#         moving_mid = int(D_moving * 0.10)
#     else:
#         # Search last 25% of slices (upper abdomen in normal orientation)
#         fixed_mid = int(D_fixed * 0.75)
#         moving_mid = int(D_moving * 0.75)

#     # Search range
#     fixed_range = range(
#         max(0, fixed_mid - search_range),
#         min(D_fixed, fixed_mid + search_range)
#     )
#     moving_range = range(
#         max(0, moving_mid - search_range),
#         min(D_moving, moving_mid + search_range)
#     )
#     best_score = -np.inf  # Changed from -1 to handle negative scores
#     best_fixed_idx = fixed_mid
#     best_moving_idx = moving_mid
    
#     print(f"   Searching {len(fixed_range)} × {len(moving_range)} slice pairs...")
    
#     # Compute similarity for all pairs
#     for f_idx in fixed_range:
#         fixed_slice = fixed_vol[f_idx]
        
#         # Skip if mostly padding
#         if (fixed_slice > body_threshold).sum() < 1000:
#             continue
        
#         for m_idx in moving_range:
#             moving_slice = moving_vol[m_idx]
            
#             # Skip if mostly padding
#             if (moving_slice > body_threshold).sum() < 1000:
#                 continue
            
#             # ============================================================
#             # FIX: Resample moving slice to match fixed dimensions
#             # ============================================================
#             if moving_slice.shape != target_shape:
#                 moving_slice_resampled = cv2.resize(
#                     moving_slice,
#                     (target_shape[1], target_shape[0]),  # cv2 uses (W, H)
#                     interpolation=cv2.INTER_LINEAR
#                 )
#             else:
#                 moving_slice_resampled = moving_slice
            
#             # Now both slices have same shape - safe to compare!
#             try:
#                 # Compute similarity
#                 if metric == 'ncc':
#                     # Normalized Cross-Correlation
#                     f_norm = (fixed_slice - fixed_slice.mean()) / (fixed_slice.std() + 1e-8)
#                     m_norm = (moving_slice_resampled - moving_slice_resampled.mean()) / (moving_slice_resampled.std() + 1e-8)
                    
#                     # Compute correlation
#                     score = np.mean(f_norm * m_norm)  # Simpler and more robust than corrcoef
                    
#                 elif metric == 'ssim':
#                     # Structural Similarity
#                     f_norm = (fixed_slice - fixed_slice.mean()) / (fixed_slice.std() + 1e-8)
#                     m_norm = (moving_slice_resampled - moving_slice_resampled.mean()) / (moving_slice_resampled.std() + 1e-8)
                    
#                     data_range = max(f_norm.max() - f_norm.min(), 1e-8)
#                     score = ssim(f_norm, m_norm, data_range=data_range)
                    
#                 elif metric == 'mi':
#                     # Mutual Information (histogram-based, shape-invariant)
#                     hist_2d, _, _ = np.histogram2d(
#                         fixed_slice.ravel(),
#                         moving_slice_resampled.ravel(),
#                         bins=50
#                     )
                    
#                     # Normalize
#                     pxy = hist_2d / hist_2d.sum()
#                     px = pxy.sum(axis=1)
#                     py = pxy.sum(axis=0)
                    
#                     # MI = sum(p(x,y) * log(p(x,y) / (p(x)*p(y))))
#                     px_py = px[:, None] * py[None, :]
#                     nzs = pxy > 0  # Non-zero elements
#                     score = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
                
#                 if score > best_score:
#                     best_score = score
#                     best_fixed_idx = f_idx
#                     best_moving_idx = m_idx
                    
#             except Exception as e:
#                 # Skip this pair if computation fails
#                 continue
    
#     print(f"   Best match: fixed[{best_fixed_idx}] ↔ moving[{best_moving_idx}] "
#           f"(score: {best_score:.4f})")
    
#     return best_fixed_idx, best_moving_idx, best_score

def find_slice_correspondence(
        fixed_vol: np.ndarray,
        moving_vol: np.ndarray,
        body_threshold: int = -600,
        search_range: int = 50,
        metric: str = 'ncc',
        search_from_end: bool = True  # NEW: Search from upper abdomen
    ) -> tuple[int, int, float]:
    """
    Find best matching slice pair between two volumes.
    
    Args:
        search_from_end: If True, search from high Z indices (upper abdomen in standard DICOM)
    """
    from skimage.metrics import structural_similarity as ssim
    import cv2
    
    target_shape = (fixed_vol.shape[1], fixed_vol.shape[2])
    
    D_fixed = fixed_vol.shape[0]
    D_moving = moving_vol.shape[0]
    print(f"   Fixed depth: {D_fixed}, Moving depth: {D_moving}")
    
    # ============================================================
    # KEY FIX: Search from end (upper abdomen) without flipping
    # ============================================================
    if search_from_end:
        # Search last 25% of slices (upper abdomen/heart region)
        fixed_center = int(D_fixed * 0.80)  # 80th percentile
        moving_center = int(D_moving * 0.80)
        print(f"   Searching from END (upper abdomen): fixed@{fixed_center}, moving@{moving_center}")
    else:
        # Search from middle
        fixed_center = D_fixed // 2
        moving_center = D_moving // 2
        print(f"   Searching from MIDDLE: fixed@{fixed_center}, moving@{moving_center}")
    
    # Build search ranges
    fixed_range = range(
        max(0, fixed_center - search_range),
        min(D_fixed, fixed_center + search_range)
    )
    moving_range = range(
        max(0, moving_center - search_range),
        min(D_moving, moving_center + search_range)
    )
    
    print(f"   Search space: fixed[{fixed_range.start}:{fixed_range.stop}] × "
          f"moving[{moving_range.start}:{moving_range.stop}]")
    
    best_score = -np.inf
    best_fixed_idx = fixed_center
    best_moving_idx = moving_center
    
    # Compute similarity for all pairs
    for f_idx in fixed_range:
        fixed_slice = fixed_vol[f_idx]
        
        if (fixed_slice > body_threshold).sum() < 1000:
            continue
        
        for m_idx in moving_range:
            moving_slice = moving_vol[m_idx]
            
            if (moving_slice > body_threshold).sum() < 1000:
                continue
            
            # Resample if needed
            if moving_slice.shape != target_shape:
                moving_slice_resampled = cv2.resize(
                    moving_slice,
                    (target_shape[1], target_shape[0]),
                    interpolation=cv2.INTER_LINEAR
                )
            else:
                moving_slice_resampled = moving_slice
            
            try:
                if metric == 'ncc':
                    f_norm = (fixed_slice - fixed_slice.mean()) / (fixed_slice.std() + 1e-8)
                    m_norm = (moving_slice_resampled - moving_slice_resampled.mean()) / (moving_slice_resampled.std() + 1e-8)
                    score = np.mean(f_norm * m_norm)
                    
                elif metric == 'ssim':
                    f_norm = (fixed_slice - fixed_slice.mean()) / (fixed_slice.std() + 1e-8)
                    m_norm = (moving_slice_resampled - moving_slice_resampled.mean()) / (moving_slice_resampled.std() + 1e-8)
                    data_range = max(f_norm.max() - f_norm.min(), 1e-8)
                    score = ssim(f_norm, m_norm, data_range=data_range)
                
                if score > best_score:
                    best_score = score
                    best_fixed_idx = f_idx
                    best_moving_idx = m_idx
                    
            except Exception:
                continue
    
    print(f"   ✓ Best match: fixed[{best_fixed_idx}] ↔ moving[{best_moving_idx}] "
          f"(score: {best_score:.4f})")
    
    return best_fixed_idx, best_moving_idx, best_score

# def align_volumes_z_axis_arc(
#         fixed_sitk: sitk.Image,
#         moving_sitk: sitk.Image,
#         study_id: str,
#         series_id: str,
#         min_overlap_slices: int = 30,
#         body_threshold: int = -600,
#         flip_for_alignment: bool = True  # NEW PARAMETER
#     ) -> tuple[sitk.Image, sitk.Image, dict]:
#     """
#     Align two volumes in Z-axis based on content matching, then crop to overlap.
    
#     Returns:
#         (aligned_fixed, aligned_moving, alignment_info)
#     """
#     # NEW: Optionally flip volumes
#     if flip_for_alignment:
#         print(f"   Flipping volumes for alignment (upper abdomen → first slices)")
#         fixed_sitk = flip_volume_z_axis(fixed_sitk)
#         moving_sitk = flip_volume_z_axis(moving_sitk)
    
#     # Convert to numpy
#     fixed_np = sitk.GetArrayFromImage(fixed_sitk)   # [D, H, W]
#     moving_np = sitk.GetArrayFromImage(moving_sitk)
    
#     print(f"\n   Z-axis alignment for {study_id}_{series_id}")
#     print(f"   Fixed shape: {fixed_np.shape}, Moving shape: {moving_np.shape}")
    
#     # Find corresponding slices
#     fixed_idx, moving_idx, score = find_slice_correspondence(
#         fixed_np, moving_np, 
#         body_threshold=body_threshold,
#         search_range=80,
#         metric='ncc'  # Fast and works well for same-modality CT
#     )
    
#     # Calculate z-offset
#     z_offset = fixed_idx - moving_idx
#     print(f"   Z-offset detected: {z_offset} slices")
    
#     # Determine cropping range for maximum overlap
#     D_fixed = fixed_np.shape[0]
#     D_moving = moving_np.shape[0]
    
#     # Start indices in original volumes
#     if z_offset >= 0:
#         # Moving starts before fixed
#         fixed_start = z_offset
#         moving_start = 0
#     else:
#         # Fixed starts before moving
#         fixed_start = 0
#         moving_start = -z_offset
    
#     # End indices
#     overlap_slices = min(
#         D_fixed - fixed_start,
#         D_moving - moving_start
#     )
    
#     if overlap_slices < min_overlap_slices:
#         raise ValueError(
#             f"Insufficient overlap: {overlap_slices} slices "
#             f"(minimum: {min_overlap_slices})"
#         )
    
#     fixed_end = fixed_start + overlap_slices
#     moving_end = moving_start + overlap_slices
    
#     print(f"   Cropping to overlap: {overlap_slices} slices")
#     print(f"   Fixed: [{fixed_start}:{fixed_end}], Moving: [{moving_start}:{moving_end}]")
    
#     # Crop numpy arrays
#     fixed_cropped_np = fixed_np[fixed_start:fixed_end]
#     moving_cropped_np = moving_np[moving_start:moving_end]
    
#     # Convert back to SimpleITK and UPDATE ORIGIN
#     fixed_cropped = sitk.GetImageFromArray(fixed_cropped_np)
#     moving_cropped = sitk.GetImageFromArray(moving_cropped_np)
#     # Verify shape is correct (should match fixed_sitk after cropping)
#     expected_size = (fixed_sitk.GetSize()[0], fixed_sitk.GetSize()[1], overlap_slices)
#     print(f"   Expected size: {expected_size}, Got: {fixed_cropped.GetSize()}")

#     # If sizes are transposed, we need to fix the orientation
#     if fixed_cropped.GetSize() != expected_size:
#         print(f"   ⚠️  Shape mismatch detected - this should not happen!")

#     # Copy spacing and direction
#     fixed_cropped.SetSpacing(fixed_sitk.GetSpacing())
#     fixed_cropped.SetDirection(fixed_sitk.GetDirection())
#     moving_cropped.SetSpacing(moving_sitk.GetSpacing())
#     moving_cropped.SetDirection(moving_sitk.GetDirection())
    
#     # ⚠️ CRITICAL: Update origin to reflect cropping
#     fixed_origin = np.array(fixed_sitk.GetOrigin())
#     moving_origin = np.array(moving_sitk.GetOrigin())
    
#     spacing = np.array(fixed_sitk.GetSpacing())
#     direction_matrix = np.array(fixed_sitk.GetDirection()).reshape(3, 3)
#     z_unit_vector = direction_matrix[:, 2]
    
#     # Physical shift = number_of_removed_slices * spacing_z * direction
#     fixed_origin_shift = fixed_start * spacing[2] * z_unit_vector
#     moving_origin_shift = moving_start * spacing[2] * z_unit_vector
    
#     fixed_cropped.SetOrigin((fixed_origin + fixed_origin_shift).tolist())
#     moving_cropped.SetOrigin((moving_origin + moving_origin_shift).tolist())
    
#     alignment_info = {
#         'z_offset': z_offset,
#         'similarity_score': score,
#         'fixed_crop': (fixed_start, fixed_end),
#         'moving_crop': (moving_start, moving_end),
#         'overlap_slices': overlap_slices,
#         'original_shapes': (fixed_np.shape, moving_np.shape),
#         'aligned_shapes': (fixed_cropped_np.shape, moving_cropped_np.shape)
#     }
    
#     print(f"   ✓ Aligned: {fixed_cropped.GetSize()} (both volumes now match)")
#     # At the end, optionally flip back
#     if flip_for_alignment:
#         print(f"   Flipping aligned volumes back to original orientation")
#         fixed_cropped = flip_volume_z_axis(fixed_cropped)
#         moving_cropped = flip_volume_z_axis(moving_cropped)
    
#     return fixed_cropped, moving_cropped, alignment_info
    
def find_best_reference_slice(
        reference_vol: np.ndarray,
        body_threshold: int = -600,
        search_region: str = 'upper'  # 'upper', 'middle', or 'lower'
    ) -> int:
    """
    Find the best reference slice in the non-contrast volume.
    Should have good anatomy visibility (liver, spleen, etc.)
    """
    D = reference_vol.shape[0]
    
    if search_region == 'upper':
        search_start = int(D * 0.65)
        search_end = int(D * 0.90)
    elif search_region == 'middle':
        search_start = int(D * 0.35)
        search_end = int(D * 0.65)
    else:  # lower
        search_start = int(D * 0.10)
        search_end = int(D * 0.35)
    
    print(f"   Searching for reference slice in region [{search_start}:{search_end}]")
    
    best_idx = (search_start + search_end) // 2
    best_body_area = 0
    
    for idx in range(search_start, search_end):
        slice_data = reference_vol[idx]
        body_mask = slice_data > body_threshold
        body_area = body_mask.sum()
        
        # Also check for good contrast (std dev)
        body_std = slice_data[body_mask].std() if body_mask.any() else 0
        
        # Score = area × contrast
        score = body_area * body_std
        
        if score > best_body_area:
            best_body_area = score
            best_idx = idx
    
    print(f"   ✓ Best reference slice: {best_idx} (score: {best_body_area:.0f})")
    
    return best_idx


def match_single_slice(
        reference_slice: np.ndarray,
        moving_vol: np.ndarray,
        search_range: tuple[int, int],
        metric: str = 'ncc'
    ) -> tuple[int, float]:
    """Match a single reference slice to a moving volume."""
    import cv2
    from skimage.metrics import structural_similarity as ssim
    
    target_shape = reference_slice.shape
    
    best_idx = (search_range[0] + search_range[1]) // 2
    best_score = -np.inf
    
    for m_idx in range(search_range[0], search_range[1]):
        moving_slice = moving_vol[m_idx]
        
        # Resample if needed
        if moving_slice.shape != target_shape:
            moving_slice = cv2.resize(
                moving_slice,
                (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
        
        try:
            if metric == 'ncc':
                r_norm = (reference_slice - reference_slice.mean()) / (reference_slice.std() + 1e-8)
                m_norm = (moving_slice - moving_slice.mean()) / (moving_slice.std() + 1e-8)
                score = np.mean(r_norm * m_norm)
            elif metric == 'ssim':
                r_norm = (reference_slice - reference_slice.mean()) / (reference_slice.std() + 1e-8)
                m_norm = (moving_slice - moving_slice.mean()) / (moving_slice.std() + 1e-8)
                score = ssim(r_norm, m_norm, data_range=max(r_norm.max()-r_norm.min(), 1e-8))
            
            if score > best_score:
                best_score = score
                best_idx = m_idx
                
        except Exception:
            continue
    
    return best_idx, best_score

# def align_all_series_in_study(
#         study_id: str,
#         series_paths: dict,
#         reference_series_id: str,
#         output_dir: str,
#         body_threshold: int = -600,
#         min_overlap_slices: int = 30,
#         save_visualizations: bool = True,
#         force_recompute: bool = False  # NEW: Force recomputation
#     ) -> dict:
#     """
#     Align ALL series in a study to a common coordinate system.
#     FIXED: Handles different in-plane dimensions by resampling to reference grid.
    
#     Args:
#         study_id: Study identifier
#         series_paths: Dict mapping series_id to file paths
#         reference_series_id: Which series to use as reference
#         output_dir: Where to save aligned volumes
#         body_threshold: HU threshold for body detection
#         min_overlap_slices: Minimum overlap required
#         save_visualizations: Whether to create comparison images
#         force_recompute: If True, recompute even if visualizations exist
        
#     Returns:
#         Dictionary with alignment info for all series
#     """
#     import json
#     import cv2
#     import matplotlib.pyplot as plt
#     import matplotlib.gridspec as gridspec
    
#     # ============================================================
#     # NEW: Check if already processed
#     # ============================================================
#     viz_dir = os.path.join(output_dir, "alignment_visualizations")
    
#     if not force_recompute and os.path.exists(viz_dir):
#         # Check if it has content
#         existing_files = os.listdir(viz_dir)
#         if len(existing_files) > 0:
#             print(f"\n✓ Study {study_id[:30]}... already processed")
#             print(f"  Found {len(existing_files)} visualization files in {viz_dir}")
#             print(f"  → Skipping (use force_recompute=True to override)")
            
#             # Load existing metadata
#             metadata_path = os.path.join(output_dir, f"{study_id}_alignment_metadata.json")
#             if os.path.exists(metadata_path):
#                 with open(metadata_path, 'r') as f:
#                     metadata = json.load(f)
                
#                 return {
#                     'aligned_volumes': {},  # Not loading volumes to save memory
#                     'aligned_segmentations': {},
#                     'metadata': metadata,
#                     'all_match': True,  # Assume it was successful
#                     'skipped': True
#                 }
    
#     print(f"\n{'='*80}")
#     print(f"Multi-Series Z-Alignment: {study_id}")
#     print(f"{'='*80}")
#     print(f"Reference series: {reference_series_id[:30]}...")
#     print(f"Total series: {len(series_paths)}")
    
#     # ============================================================
#     # STEP 1: Load reference volume
#     # ============================================================
#     ref_path = series_paths[reference_series_id]["image"]
#     ref_sitk = sitk.ReadImage(ref_path)
#     ref_np = sitk.GetArrayFromImage(ref_sitk)
    
#     print(f"\n1. Reference loaded: {ref_np.shape}")
#     print(f"   Reference size (SimpleITK): {ref_sitk.GetSize()}")
#     print(f"   Reference spacing: {ref_sitk.GetSpacing()}")
    
#     # Load reference segmentation if available
#     ref_seg_path = series_paths[reference_series_id].get("seg")
#     ref_seg_sitk = None
#     ref_seg_np = None
    
#     if ref_seg_path and os.path.exists(ref_seg_path):
#         ref_seg_sitk = sitk.ReadImage(ref_seg_path)
#         ref_seg_np = sitk.GetArrayFromImage(ref_seg_sitk)
#         print(f"   ✓ Reference segmentation loaded: {ref_seg_np.shape}")
    
#     # ============================================================
#     # STEP 2: Find slice correspondence AND check dimensions
#     # ============================================================
#     alignment_data = {}
#     needs_resampling = False
    
#     for series_id, series_info in series_paths.items():
#         if series_id == reference_series_id:
#             alignment_data[series_id] = {
#                 'z_offset': 0,
#                 'fixed_idx': ref_np.shape[0] // 2,
#                 'moving_idx': ref_np.shape[0] // 2,
#                 'score': 1.0,
#                 'original_shape': ref_np.shape,
#                 'phase': series_info['phase'],
#                 'needs_resampling': False
#             }
#             continue
        
#         print(f"\n2. Analyzing {series_id[:30]}... ({series_info['phase']})...")
        
#         moving_sitk = sitk.ReadImage(series_info["image"])
#         moving_np = sitk.GetArrayFromImage(moving_sitk)
        
#         print(f"   Shape: {moving_np.shape}")
#         print(f"   Size (SimpleITK): {moving_sitk.GetSize()}")
#         print(f"   Spacing: {moving_sitk.GetSpacing()}")
        
#         # ============================================================
#         # NEW: Check if in-plane dimensions match
#         # ============================================================
#         in_plane_match = (moving_np.shape[1:] == ref_np.shape[1:])
        
#         if not in_plane_match:
#             print(f"   ⚠️  In-plane dimension mismatch!")
#             print(f"      Reference: {ref_np.shape[1:]} (H, W)")
#             print(f"      This series: {moving_np.shape[1:]} (H, W)")
#             print(f"   → Will resample to reference grid")
#             needs_resampling = True
        
#         # Find correspondence
#         fixed_idx, moving_idx, score = find_slice_correspondence(
#             ref_np, moving_np,
#             body_threshold=body_threshold,
#             search_range=80,
#             metric='ncc'
#         )
        
#         z_offset = fixed_idx - moving_idx
        
#         # Load segmentation if available
#         seg_path = series_info.get("seg")
#         seg_sitk = None
#         seg_np = None
        
#         if seg_path and os.path.exists(seg_path):
#             seg_sitk = sitk.ReadImage(seg_path)
#             seg_np = sitk.GetArrayFromImage(seg_sitk)
#             print(f"   ✓ Segmentation loaded: {seg_np.shape}")
        
#         alignment_data[series_id] = {
#             'z_offset': z_offset,
#             'fixed_idx': fixed_idx,
#             'moving_idx': moving_idx,
#             'score': score,
#             'original_shape': moving_np.shape,
#             'phase': series_info['phase'],
#             'sitk': moving_sitk,
#             'numpy': moving_np,
#             'seg_sitk': seg_sitk,
#             'seg_numpy': seg_np,
#             'needs_resampling': not in_plane_match
#         }
        
#         print(f"   Z-offset: {z_offset} slices (score: {score:.4f})")
    
#     if needs_resampling:
#         print(f"\n⚠️  RESAMPLING REQUIRED: Some series have different in-plane dimensions")
#         print(f"   All series will be resampled to reference grid: {ref_np.shape[1:]} (H, W)")
    
#     # ============================================================
#     # STEP 3: Find COMMON overlap region
#     # ============================================================
#     print(f"\n3. Computing common overlap region...")
    
#     valid_ranges = []
#     valid_ranges.append((0, ref_np.shape[0]))
    
#     for series_id, data in alignment_data.items():
#         if series_id == reference_series_id:
#             continue
        
#         moving_depth = data['original_shape'][0]
#         z_offset = data['z_offset']
        
#         if z_offset >= 0:
#             ref_start = z_offset
#             ref_end = z_offset + moving_depth
#         else:
#             ref_start = 0
#             ref_end = moving_depth + z_offset
        
#         ref_start = max(0, ref_start)
#         ref_end = min(ref_np.shape[0], ref_end)
        
#         valid_ranges.append((ref_start, ref_end))
#         print(f"   {series_id[:30]}...: valid [{ref_start}:{ref_end}]")
    
#     common_start = max(r[0] for r in valid_ranges)
#     common_end = min(r[1] for r in valid_ranges)
#     common_depth = common_end - common_start
    
#     print(f"\n   ✓ Common overlap: [{common_start}:{common_end}] = {common_depth} slices")
    
#     if common_depth < min_overlap_slices:
#         raise ValueError(
#             f"Insufficient common overlap: {common_depth} slices "
#             f"(minimum: {min_overlap_slices})"
#         )
    
#     # ============================================================
#     # STEP 4: Resample and crop ALL volumes
#     # ============================================================
#     print(f"\n4. {'Resampling and cropping' if needs_resampling else 'Cropping'} all volumes...")

#     os.makedirs(output_dir, exist_ok=True)
#     aligned_volumes = {}
#     aligned_segmentations = {}

#     # Store reference data
#     alignment_data[reference_series_id]['sitk'] = ref_sitk
#     alignment_data[reference_series_id]['numpy'] = ref_np
#     alignment_data[reference_series_id]['seg_sitk'] = ref_seg_sitk
#     alignment_data[reference_series_id]['seg_numpy'] = ref_seg_np

#     # ============================================================
#     # NEW FIX: Process reference FIRST, then all others
#     # ============================================================
#     # Create processing order: reference first, then others
#     processing_order = [reference_series_id] + [
#         sid for sid in alignment_data.keys() if sid != reference_series_id
#     ]

#     for series_id in processing_order:  # ← Changed from alignment_data.items()
#         data = alignment_data[series_id]
        
#         print(f"\n   Processing {series_id[:30]}... ({data['phase']})...")
        
#         vol_sitk = data['sitk']
#         vol_np = data['numpy']
#         seg_sitk = data.get('seg_sitk')
#         seg_np = data.get('seg_numpy')
#         z_offset = data['z_offset']
#         needs_resample = data['needs_resampling']
        
#         # ============================================================
#         # Resample to reference grid if needed
#         # ============================================================
#         if needs_resample:
#             print(f"     Resampling to reference grid (in-plane only)...")
            
#             # Get reference in-plane spacing and size
#             ref_spacing = ref_sitk.GetSpacing()
#             ref_size = ref_sitk.GetSize()
            
#             # Build new size: use reference X,Y but keep moving's Z
#             new_size = (
#                 ref_size[0],      # X from reference
#                 ref_size[1],      # Y from reference  
#                 vol_sitk.GetSize()[2]  # Z from moving (KEEP ORIGINAL!)
#             )
            
#             # Build new spacing: use reference X,Y but keep moving's Z
#             new_spacing = (
#                 ref_spacing[0],    # X spacing from reference
#                 ref_spacing[1],    # Y spacing from reference
#                 vol_sitk.GetSpacing()[2]  # Z spacing from moving (KEEP ORIGINAL!)
#             )
            
#             # Resample image with correct parameters
#             resampler = sitk.ResampleImageFilter()
#             resampler.SetSize(new_size)  # ✓ Explicit size (not from reference!)
#             resampler.SetOutputSpacing(new_spacing)
#             resampler.SetOutputOrigin(vol_sitk.GetOrigin())
#             resampler.SetOutputDirection(vol_sitk.GetDirection())
#             resampler.SetInterpolator(sitk.sitkLinear)
#             resampler.SetDefaultPixelValue(-1000)
#             resampler.SetTransform(sitk.Transform())  # Identity
            
#             vol_resampled_sitk = resampler.Execute(vol_sitk)
#             vol_np = sitk.GetArrayFromImage(vol_resampled_sitk)
#             vol_sitk = vol_resampled_sitk
            
#             print(f"       ✓ Image resampled: {vol_np.shape} (Z-depth preserved)")
#             # print(f"     Resampling to reference grid...")
            
#             # # Resample image
#             # resampler = sitk.ResampleImageFilter()
#             # resampler.SetReferenceImage(ref_sitk)
#             # resampler.SetInterpolator(sitk.sitkLinear)
#             # resampler.SetDefaultPixelValue(-1000)
#             # resampler.SetTransform(sitk.Transform())
            
#             # vol_resampled_sitk = resampler.Execute(vol_sitk)
#             vol_np = sitk.GetArrayFromImage(vol_resampled_sitk)
#             vol_sitk = vol_resampled_sitk
            
#             print(f"       ✓ Image resampled: {vol_np.shape}")
#             if seg_np is not None:
#                 resampler.SetInterpolator(sitk.sitkNearestNeighbor)
#                 resampler.SetDefaultPixelValue(0)
#                 resampler.SetSize(new_size)  # ✓ Use same new_size
#                 resampler.SetOutputSpacing(new_spacing)
#                 resampler.SetOutputOrigin(seg_sitk.GetOrigin())
#                 resampler.SetOutputDirection(seg_sitk.GetDirection())
                
#                 seg_resampled_sitk = resampler.Execute(seg_sitk)
#                 seg_np = sitk.GetArrayFromImage(seg_resampled_sitk)
#                 seg_sitk = seg_resampled_sitk
                
#                 print(f"       ✓ Segmentation resampled: {seg_np.shape} (Z-depth preserved)")


#             # # Resample segmentation if available
#             # if seg_np is not None:
#             #     resampler.SetInterpolator(sitk.sitkNearestNeighbor)
#             #     resampler.SetDefaultPixelValue(0)
                
#             #     seg_resampled_sitk = resampler.Execute(seg_sitk)
#             #     seg_np = sitk.GetArrayFromImage(seg_resampled_sitk)
#             #     seg_sitk = seg_resampled_sitk
                
#             #     print(f"       ✓ Segmentation resampled: {seg_np.shape}")
            
#             # Update data
#             data['numpy'] = vol_np
#             data['sitk'] = vol_sitk
#             data['seg_numpy'] = seg_np
#             data['seg_sitk'] = seg_sitk
#             data['resampled_shape'] = vol_np.shape
        
#         # Calculate crop range
#         if series_id == reference_series_id:
#             crop_start = common_start
#             crop_end = common_end
#         else:
#             crop_start = common_start - z_offset
#             crop_end = common_end - z_offset
#             crop_start = max(0, crop_start)
#             crop_end = min(vol_np.shape[0], crop_end)
        
#         print(f"     Image: Cropping [{crop_start}:{crop_end}] ({crop_end - crop_start} slices)")
        
#         # ============================================================
#         # Crop IMAGE
#         # ============================================================
#         cropped_np = vol_np[crop_start:crop_end]
#         cropped_sitk = sitk.GetImageFromArray(cropped_np)
#         cropped_sitk.SetSpacing(vol_sitk.GetSpacing())
#         cropped_sitk.SetDirection(vol_sitk.GetDirection())
        
#         # Update origin
#         origin = np.array(vol_sitk.GetOrigin())
#         spacing = np.array(vol_sitk.GetSpacing())
#         direction_matrix = np.array(vol_sitk.GetDirection()).reshape(3, 3)
#         z_unit_vector = direction_matrix[:, 2]
#         origin_shift = crop_start * spacing[2] * z_unit_vector
#         new_origin = origin + origin_shift
#         cropped_sitk.SetOrigin(new_origin.tolist())
        
#         # Save aligned image
#         output_path = os.path.join(output_dir, f"{study_id}_{series_id}_aligned.nii.gz")
#         sitk.WriteImage(cropped_sitk, output_path)
#         print(f"     ✓ Saved image: {os.path.basename(output_path)}")
        
#         aligned_volumes[series_id] = cropped_sitk
#         data['output_path'] = output_path
        
#         # ============================================================
#         # Crop SEGMENTATION
#         # ============================================================
#         if seg_np is not None:
#             print(f"     Segmentation: Cropping [{crop_start}:{crop_end}]")
            
#             seg_cropped_np = seg_np[crop_start:crop_end]
#             seg_cropped_sitk = sitk.GetImageFromArray(seg_cropped_np.astype(np.uint8))
            
#             # Copy metadata from aligned image
#             seg_cropped_sitk.SetSpacing(cropped_sitk.GetSpacing())
#             seg_cropped_sitk.SetDirection(cropped_sitk.GetDirection())
#             seg_cropped_sitk.SetOrigin(cropped_sitk.GetOrigin())
            
#             # Verify size matches
#             if seg_cropped_sitk.GetSize() != cropped_sitk.GetSize():
#                 print(f"     ⚠️  Seg size mismatch: {seg_cropped_sitk.GetSize()} vs {cropped_sitk.GetSize()}")
#             else:
#                 print(f"     ✓ Segmentation size matches: {seg_cropped_sitk.GetSize()}")
            
#             # Save
#             seg_output_path = os.path.join(output_dir, f"{study_id}_{series_id}_aligned_seg.nii.gz")
#             sitk.WriteImage(seg_cropped_sitk, seg_output_path)
#             print(f"     ✓ Saved segmentation: {os.path.basename(seg_output_path)}")
            
#             aligned_segmentations[series_id] = seg_cropped_sitk
#             data['seg_output_path'] = seg_output_path
#         else:
#             print(f"     ⊘ No segmentation available")
#             aligned_segmentations[series_id] = None
        
#         # ============================================================
#         # Verify against reference (NOW SAFE - reference is always first!)
#         # ============================================================
#         if series_id != reference_series_id:
#             ref_aligned = aligned_volumes[reference_series_id]  # ← NOW GUARANTEED TO EXIST
#             if cropped_sitk.GetSize() != ref_aligned.GetSize():
#                 print(f"     ❌ SIZE MISMATCH: {cropped_sitk.GetSize()} vs {ref_aligned.GetSize()}")
#                 raise ValueError(f"Failed to align {series_id} - size mismatch persists!")
#             else:
#                 print(f"     ✓ Size matches reference: {cropped_sitk.GetSize()}")
        
#         data['crop_range'] = (crop_start, crop_end)
#         data['aligned_shape'] = cropped_np.shape
#     # ============================================================
#     # STEP 5: Create visualizations
#     # ============================================================
#     if save_visualizations:
#         print(f"\n5. Creating per-series comparison images...")
        
#         os.makedirs(viz_dir, exist_ok=True)
        
#         for series_id, data in alignment_data.items():
#             if series_id == reference_series_id:
#                 continue
            
#             try:
#                 create_series_alignment_comparison(
#                     series_id=series_id,
#                     series_data=data,
#                     reference_data=alignment_data[reference_series_id],
#                     aligned_volumes=aligned_volumes,
#                     aligned_segmentations=aligned_segmentations,
#                     output_dir=viz_dir,
#                     study_id=study_id
#                 )
#             except Exception as e:
#                 print(f"     ⚠️  Visualization failed for {series_id[:20]}...: {e}")
    
#     # ============================================================
#     # STEP 6: Save metadata
#     # ============================================================
#     print(f"\n6. Saving alignment metadata...")
    
#     metadata = {
#         'study_id': study_id,
#         'reference_series': reference_series_id,
#         'needed_resampling': needs_resampling,
#         'common_overlap': {
#             'start': int(common_start),
#             'end': int(common_end),
#             'depth': int(common_depth)
#         },
#         'series': {}
#     }
    
#     for series_id, data in alignment_data.items():
#         has_seg = data.get('seg_numpy') is not None
        
#         metadata['series'][series_id] = {
#             'phase': data['phase'],
#             'z_offset': int(data['z_offset']),
#             'original_shape': [int(x) for x in data['original_shape']],
#             'aligned_shape': [int(x) for x in data['aligned_shape']],
#             'crop_range': [int(x) for x in data['crop_range']],
#             'similarity_score': float(data['score']),
#             'needed_resampling': data['needs_resampling'],
#             'resampled_shape': [int(x) for x in data.get('resampled_shape', data['original_shape'])],
#             'has_segmentation': has_seg,
#             'output_path': data.get('output_path', ''),
#             'seg_output_path': data.get('seg_output_path', '') if has_seg else None
#         }
    
#     metadata_path = os.path.join(output_dir, f"{study_id}_alignment_metadata.json")
#     with open(metadata_path, 'w') as f:
#         json.dump(metadata, f, indent=2)
    
#     print(f"   ✓ Metadata saved: {os.path.basename(metadata_path)}")
    
#     # ============================================================
#     # STEP 7: Verification
#     # ============================================================
#     print(f"\n{'='*80}")
#     print("VERIFICATION")
#     print(f"{'='*80}")
    
#     ref_size = aligned_volumes[reference_series_id].GetSize()
#     all_match = True
    
#     for series_id, vol in aligned_volumes.items():
#         size = vol.GetSize()
#         has_seg = aligned_segmentations.get(series_id) is not None
#         seg_marker = "📄" if has_seg else "  "
#         match = "✓" if size == ref_size else "✗"
        
#         print(f"  {match} {seg_marker} {series_id[:30]}...: {size}")
        
#         if size != ref_size:
#             all_match = False
        
#         if has_seg:
#             seg_size = aligned_segmentations[series_id].GetSize()
#             if seg_size != size:
#                 print(f"      ⚠️  Seg size mismatch: {seg_size}")
#                 all_match = False
    
#     if all_match:
#         print(f"\n✓✓✓ SUCCESS: All {len(aligned_volumes)} volumes match!")
#     else:
#         print(f"\n❌ FAILED: Size mismatch detected!")
    
#     total_with_seg = sum(1 for s in aligned_segmentations.values() if s is not None)
#     print(f"\nSummary:")
#     print(f"  Total series: {len(aligned_volumes)}")
#     print(f"  With segmentation: {total_with_seg}/{len(aligned_volumes)}")
#     print(f"  Common depth: {common_depth} slices")
#     print(f"  Resampling applied: {'Yes' if needs_resampling else 'No'}")
    
#     return {
#         'aligned_volumes': aligned_volumes,
#         'aligned_segmentations': aligned_segmentations,
#         'metadata': metadata,
#         'all_match': all_match,
#         'skipped': False
#     }


def align_all_series_in_study(
        study_id: str,
        series_paths: dict,
        reference_series_id: str,
        output_dir: str,
        body_threshold: int = -600,
        min_overlap_slices: int = 30
    ) -> dict:
    """
    IMPROVED: Uses single reference slice strategy.
    """
    print(f"\n{'='*80}")
    print(f"Multi-Series Alignment: {study_id}")
    print(f"{'='*80}")
    
    # Step 1: Load reference
    ref_path = series_paths[reference_series_id]["image"]
    ref_sitk = sitk.ReadImage(ref_path)
    ref_np = sitk.GetArrayFromImage(ref_sitk)
    
    print(f"\n1. Reference loaded: {ref_np.shape}")
    
    # Step 2: Find ONE reference slice (upper abdomen)
    ref_slice_idx = find_best_reference_slice(
        ref_np, 
        body_threshold=body_threshold,
        search_region='upper'
    )
    ref_slice = ref_np[ref_slice_idx]
    
    print(f"\n2. Using reference slice {ref_slice_idx} for ALL matching")
    
    # Step 3: Match each series to this ONE reference slice
    alignment_data = {}
    alignment_data[reference_series_id] = {
        'z_offset': 0,
        'matched_idx': ref_slice_idx,
        'reference_idx': ref_slice_idx,
        'score': 1.0,
        'original_shape': ref_np.shape
    }
    
    for series_id, info in series_paths.items():
        if series_id == reference_series_id:
            continue
        
        print(f"\n3. Matching {series_id[:30]}... ({info['phase']})")
        
        moving_sitk = sitk.ReadImage(info["image"])
        moving_np = sitk.GetArrayFromImage(moving_sitk)
        
        print(f"   Shape: {moving_np.shape}")
        
        # Search in upper region of moving volume
        D_moving = moving_np.shape[0]
        search_start = int(D_moving * 0.50)  # Start from middle
        search_end = min(D_moving, int(D_moving * 0.95))  # To near end
        
        matched_idx, score = match_single_slice(
            ref_slice, 
            moving_np,
            search_range=(search_start, search_end),
            metric='ncc'
        )
        
        z_offset = ref_slice_idx - matched_idx
        
        print(f"   ✓ Matched at slice {matched_idx} (z_offset: {z_offset}, score: {score:.4f})")
        
        alignment_data[series_id] = {
            'z_offset': z_offset,
            'matched_idx': matched_idx,
            'reference_idx': ref_slice_idx,
            'score': score,
            'original_shape': moving_np.shape,
            'sitk': moving_sitk,
            'numpy': moving_np
        }
    
    # Step 4: Compute common overlap (same as before)
    # ... rest of your existing code ...
    
    return alignment_data

def verify_alignment_coordinates(fixed: sitk.Image, moving: sitk.Image) -> dict:
    """Verify that two volumes are properly aligned in physical space."""
    
    results = {
        'size_match': fixed.GetSize() == moving.GetSize(),
        'spacing_match': np.allclose(fixed.GetSpacing(), moving.GetSpacing(), rtol=0.01),
        'direction_match': np.allclose(fixed.GetDirection(), moving.GetDirection(), rtol=0.01),
        'origin_close': np.allclose(fixed.GetOrigin(), moving.GetOrigin(), rtol=1.0),  # 1mm tolerance
    }
    
    print("\n   COORDINATE VERIFICATION:")
    print(f"   Size match:      {'✓' if results['size_match'] else '✗'} "
          f"({fixed.GetSize()} vs {moving.GetSize()})")
    print(f"   Spacing match:   {'✓' if results['spacing_match'] else '✗'} "
          f"({np.round(fixed.GetSpacing(), 3)} vs {np.round(moving.GetSpacing(), 3)})")
    print(f"   Direction match: {'✓' if results['direction_match'] else '✗'}")
    print(f"   Origin close:    {'✓' if results['origin_close'] else '⚠️'} "
          f"({np.round(fixed.GetOrigin(), 2)} vs {np.round(moving.GetOrigin(), 2)})")
    
    results['all_good'] = all([
        results['size_match'], 
        results['spacing_match'], 
        results['direction_match']
    ])
    
    return results

def align_volumes_z_axis(
        fixed_sitk: sitk.Image,
        moving_sitk: sitk.Image,
        study_id: str,
        series_id: str,
        min_overlap_slices: int = 30,
        body_threshold: int = -600,
        search_from_end: bool = True  # REPLACES flip_for_alignment
    ) -> tuple[sitk.Image, sitk.Image, dict]:
    """
    Align two volumes in Z-axis based on content matching, then crop to overlap.
    NO FLIPPING - just smart search direction.
    """
    # Convert to numpy (NO FLIP!)
    fixed_np = sitk.GetArrayFromImage(fixed_sitk)
    moving_np = sitk.GetArrayFromImage(moving_sitk)
    
    print(f"\n   Z-axis alignment for {study_id}_{series_id}")
    print(f"   Fixed shape: {fixed_np.shape}, Moving shape: {moving_np.shape}")
    
    # Find corresponding slices with proper search direction
    fixed_idx, moving_idx, score = find_slice_correspondence(
        fixed_np, moving_np, 
        body_threshold=body_threshold,
        search_range=80,
        metric='ncc',
        search_from_end=search_from_end  # Search from upper abdomen
    )
    
    # Calculate z-offset
    z_offset = fixed_idx - moving_idx
    print(f"   Z-offset detected: {z_offset} slices")
    
    # Determine cropping range
    D_fixed = fixed_np.shape[0]
    D_moving = moving_np.shape[0]
    
    if z_offset >= 0:
        fixed_start = z_offset
        moving_start = 0
    else:
        fixed_start = 0
        moving_start = -z_offset
    
    overlap_slices = min(D_fixed - fixed_start, D_moving - moving_start)
    
    if overlap_slices < min_overlap_slices:
        raise ValueError(f"Insufficient overlap: {overlap_slices} slices")
    
    fixed_end = fixed_start + overlap_slices
    moving_end = moving_start + overlap_slices
    
    print(f"   Cropping: fixed[{fixed_start}:{fixed_end}], moving[{moving_start}:{moving_end}]")
    
    # Crop volumes
    fixed_cropped_np = fixed_np[fixed_start:fixed_end]
    moving_cropped_np = moving_np[moving_start:moving_end]
    
    # Create SimpleITK images with CORRECT metadata
    fixed_cropped = sitk.GetImageFromArray(fixed_cropped_np)
    moving_cropped = sitk.GetImageFromArray(moving_cropped_np)
    
    # Copy spacing and direction (UNCHANGED)
    fixed_cropped.SetSpacing(fixed_sitk.GetSpacing())
    fixed_cropped.SetDirection(fixed_sitk.GetDirection())
    moving_cropped.SetSpacing(moving_sitk.GetSpacing())
    moving_cropped.SetDirection(moving_sitk.GetDirection())
    
    # Update origin to reflect cropping
    def compute_new_origin(original_sitk, crop_start):
        origin = np.array(original_sitk.GetOrigin())
        spacing = np.array(original_sitk.GetSpacing())
        direction = np.array(original_sitk.GetDirection()).reshape(3, 3)
        z_vector = direction[:, 2]
        shift = crop_start * spacing[2] * z_vector
        return (origin + shift).tolist()
    
    fixed_cropped.SetOrigin(compute_new_origin(fixed_sitk, fixed_start))
    moving_cropped.SetOrigin(compute_new_origin(moving_sitk, moving_start))
    
    alignment_info = {
        'z_offset': z_offset,
        'similarity_score': score,
        'fixed_crop': (fixed_start, fixed_end),
        'moving_crop': (moving_start, moving_end),
        'overlap_slices': overlap_slices,
        'matched_slices': (fixed_idx, moving_idx)
    }
    
    print(f"   ✓ Aligned: {fixed_cropped.GetSize()}")
    
    return fixed_cropped, moving_cropped, alignment_info


def create_study_summary_visualization(output_dir: str, study_id: str, metadata: dict):
    """Create a comprehensive study-level summary visualization."""
    import matplotlib.pyplot as plt
    import glob
    
    fig = plt.figure(figsize=(18, 10))
    
    # Load all aligned volumes
    volume_files = glob.glob(os.path.join(output_dir, f"{study_id}_*_aligned.nii.gz"))
    volume_files = [f for f in volume_files if '_seg' not in f]
    
    num_series = len(volume_files)
    
    # Create grid
    gs = plt.GridSpec(2, num_series, hspace=0.3, wspace=0.2)
    
    for idx, vol_file in enumerate(sorted(volume_files)):
        # Extract series ID
        basename = os.path.basename(vol_file)
        series_id = basename.replace(f"{study_id}_", "").replace("_aligned.nii.gz", "")
        
        # Load volume
        vol_sitk = sitk.ReadImage(vol_file)
        vol_np = sitk.GetArrayFromImage(vol_sitk)
        
        # Load segmentation if available
        seg_file = vol_file.replace(".nii.gz", "_seg.nii.gz")
        has_seg = os.path.exists(seg_file)
        
        # Get phase from metadata
        phase = "Unknown"
        for sid, info in metadata['series'].items():
            if sid in basename:
                phase = info['phase']
                break
        
        # Row 1: Image only
        ax = fig.add_subplot(gs[0, idx])
        mid_slice = vol_np.shape[0] // 2
        ax.imshow(vol_np[mid_slice], cmap='gray', vmin=-1000, vmax=500)
        ax.set_title(f'{phase}\n{series_id[:15]}...\n{vol_np.shape}', fontsize=9)
        ax.axis('off')
        
        # Row 2: With segmentation overlay if available
        ax = fig.add_subplot(gs[1, idx])
        if has_seg:
            seg_sitk = sitk.ReadImage(seg_file)
            seg_np = sitk.GetArrayFromImage(seg_sitk)
            overlay = create_segmentation_overlay(vol_np[mid_slice], seg_np[mid_slice])
            ax.imshow(overlay)
            ax.set_title('With Segmentation ✓', fontsize=9, color='green')
        else:
            ax.imshow(vol_np[mid_slice], cmap='gray', vmin=-1000, vmax=500)
            ax.set_title('No Segmentation', fontsize=9, color='gray')
        ax.axis('off')
    
    fig.suptitle(
        f'Study Summary: {study_id[:50]}...\n'
        f'All {num_series} series aligned to common space '
        f'({metadata["common_overlap"]["depth"]} slices)',
        fontsize=14, fontweight='bold'
    )
    
    output_path = os.path.join(output_dir, f'{study_id}_study_summary.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Study summary saved: {os.path.basename(output_path)}")

def verify_study_alignment(study_dir: str):
    """Verify all series in a study have matching dimensions."""
    import glob
    
    files = glob.glob(os.path.join(study_dir, "*_aligned.nii.gz"))
    
    if not files:
        print(f"No aligned volumes found in {study_dir}")
        return False
    
    print(f"\nVerifying {len(files)} volumes in {study_dir}...")
    
    sizes = {}
    for f in files:
        img = sitk.ReadImage(f)
        size = img.GetSize()
        spacing = img.GetSpacing()
        sizes[os.path.basename(f)] = (size, spacing)
    
    # Check all match
    reference = list(sizes.values())[0]
    all_match = all(s == reference for s in sizes.values())
    
    print("\nSize check:")
    for name, (size, spacing) in sizes.items():
        match = "✓" if (size, spacing) == reference else "✗"
        print(f"  {match} {name}: {size}, spacing: {spacing}")
    
    return all_match

def create_series_alignment_comparison(
        series_id: str,
        series_data: dict,
        reference_data: dict,
        aligned_volumes: dict,
        aligned_segmentations: dict,
        output_dir: str,
        study_id: str
    ):
    """
    Create detailed before/after comparison for a single series.
    Shows: original vs aligned, with and without segmentation overlay.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import cv2
    
    phase = series_data['phase']
    ref_phase = reference_data['phase']
    
    print(f"     Creating comparison for {series_id[:20]}... ({phase})")
    
    # Load original (before alignment)
    original_sitk = series_data['sitk']
    original_np = series_data['numpy']
    original_seg_np = series_data.get('seg_numpy')
    
    # Load aligned (after alignment)
    aligned_sitk = aligned_volumes[series_id]
    aligned_np = sitk.GetArrayFromImage(aligned_sitk)
    aligned_seg_sitk = aligned_segmentations.get(series_id)
    aligned_seg_np = sitk.GetArrayFromImage(aligned_seg_sitk) if aligned_seg_sitk else None
    
    # Load reference
    ref_sitk = reference_data['sitk']
    ref_np = reference_data['numpy']
    ref_seg_np = reference_data.get('seg_numpy')
    
    # ============================================================
    # Create comprehensive comparison figure
    # ============================================================
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 4, hspace=0.3, wspace=0.3)
    
    # Get representative slices
    mid_original = original_np.shape[0] // 2
    mid_aligned = aligned_np.shape[0] // 2
    mid_ref = ref_np.shape[0] // 2
    
    # ============================================================
    # ROW 1: BEFORE ALIGNMENT
    # ============================================================
    # Original moving
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(original_np[mid_original], cmap='gray', vmin=-1000, vmax=500)
    ax.set_title(f'BEFORE: {phase}\nSlice {mid_original}/{original_np.shape[0]}', 
                 fontsize=10, fontweight='bold')
    ax.axis('off')
    
    # Reference (for comparison)
    ax = fig.add_subplot(gs[0, 1])
    ref_slice_resized = cv2.resize(
        ref_np[mid_ref],
        (original_np.shape[2], original_np.shape[1]),
        interpolation=cv2.INTER_LINEAR
    )
    ax.imshow(ref_slice_resized, cmap='gray', vmin=-1000, vmax=500)
    ax.set_title(f'Reference: {ref_phase}\nSlice {mid_ref}/{ref_np.shape[0]}', 
                 fontsize=10, fontweight='bold')
    ax.axis('off')
    
    # Difference (before)
    ax = fig.add_subplot(gs[0, 2])
    diff_before = np.abs(original_np[mid_original] - ref_slice_resized)
    im = ax.imshow(diff_before, cmap='hot', vmin=0, vmax=500)
    ax.set_title('Difference (Before)', fontsize=10, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Original with segmentation overlay
    ax = fig.add_subplot(gs[0, 3])
    if original_seg_np is not None:
        overlay_before = create_segmentation_overlay(
            original_np[mid_original], 
            original_seg_np[mid_original]
        )
        ax.imshow(overlay_before)
        ax.set_title(f'With Segmentation (Before)', fontsize=10, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No Segmentation', ha='center', va='center', fontsize=12)
        ax.set_title('Segmentation', fontsize=10, fontweight='bold')
    ax.axis('off')
    
    # ============================================================
    # ROW 2: AFTER ALIGNMENT
    # ============================================================
    # Aligned moving
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(aligned_np[mid_aligned], cmap='gray', vmin=-1000, vmax=500)
    ax.set_title(f'AFTER: {phase}\nSlice {mid_aligned}/{aligned_np.shape[0]}', 
                 fontsize=10, fontweight='bold', color='green')
    ax.axis('off')
    
    # Aligned reference (same slice index now!)
    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(ref_np[mid_ref], cmap='gray', vmin=-1000, vmax=500)
    ax.set_title(f'Reference: {ref_phase}\nSlice {mid_ref}/{ref_np.shape[0]}', 
                 fontsize=10, fontweight='bold', color='green')
    ax.axis('off')
    
    # Difference (after) - should be much better!
    ax = fig.add_subplot(gs[1, 2])
    # Now both have same dimensions - direct comparison
    ref_slice_aligned = ref_np[mid_ref]
    if aligned_np[mid_aligned].shape != ref_slice_aligned.shape:
        ref_slice_aligned = cv2.resize(
            ref_slice_aligned,
            (aligned_np.shape[2], aligned_np.shape[1]),
            interpolation=cv2.INTER_LINEAR
        )
    diff_after = np.abs(aligned_np[mid_aligned] - ref_slice_aligned)
    im = ax.imshow(diff_after, cmap='hot', vmin=0, vmax=500)
    ax.set_title('Difference (After)', fontsize=10, fontweight='bold', color='green')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Aligned with segmentation overlay
    ax = fig.add_subplot(gs[1, 3])
    if aligned_seg_np is not None:
        overlay_after = create_segmentation_overlay(
            aligned_np[mid_aligned], 
            aligned_seg_np[mid_aligned]
        )
        ax.imshow(overlay_after)
        ax.set_title(f'With Segmentation (After)', fontsize=10, fontweight='bold', color='green')
    else:
        ax.text(0.5, 0.5, 'No Segmentation', ha='center', va='center', fontsize=12)
        ax.set_title('Segmentation', fontsize=10, fontweight='bold')
    ax.axis('off')
    
    # ============================================================
    # ROW 3: IMPROVEMENT METRICS
    # ============================================================
    # Improvement histogram
    ax = fig.add_subplot(gs[2, 0:2])
    ax.hist(diff_before.ravel(), bins=50, alpha=0.5, label='Before', color='red', density=True)
    ax.hist(diff_after.ravel(), bins=50, alpha=0.5, label='After', color='green', density=True)
    ax.set_xlabel('Absolute Difference (HU)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title('Difference Distribution', fontsize=10, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Metrics table
    ax = fig.add_subplot(gs[2, 2:4])
    ax.axis('off')
    
    # Calculate metrics
    mse_before = np.mean(diff_before ** 2)
    mse_after = np.mean(diff_after ** 2)
    mae_before = np.mean(diff_before)
    mae_after = np.mean(diff_after)
    improvement = ((mse_before - mse_after) / mse_before) * 100
    
    metrics_text = f"""
    ALIGNMENT METRICS
    {'='*40}

    Original Shape:    {series_data['original_shape']}
    Aligned Shape:     {series_data['aligned_shape']}
    Z-offset:          {series_data['z_offset']} slices
    Similarity Score:  {series_data['score']:.4f}

    DIFFERENCE METRICS (HU)
    {'='*40}
                Before      After       Improvement
    MSE:        {mse_before:8.2f}    {mse_after:8.2f}    {improvement:6.1f}%
    MAE:        {mae_before:8.2f}    {mae_after:8.2f}    {((mae_before-mae_after)/mae_before)*100:6.1f}%

    SEGMENTATION
    {'='*40}
    Status:     {'Available ✓' if aligned_seg_np is not None else 'Not Available ✗'}
    """
    
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Overall title
    fig.suptitle(
        f'Z-Alignment Comparison: {phase} vs {ref_phase}\n'
        f'Study: {study_id[:40]}... | Series: {series_id[:40]}...',
        fontsize=14, fontweight='bold'
    )
    
    # Save
    safe_series_id = series_id.replace('.', '_')
    output_path = os.path.join(output_dir, f"{safe_series_id}_{phase.replace(' ', '_')}_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"       ✓ Saved: {os.path.basename(output_path)}")

def create_segmentation_overlay(image_slice: np.ndarray, seg_slice: np.ndarray, alpha: float = 0.3):
    """Create RGB overlay of segmentation on grayscale image."""
    import cv2
    
    # Normalize image to 0-255
    img_norm = np.clip((image_slice + 1000) / 1500 * 255, 0, 255).astype(np.uint8)
    img_rgb = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2RGB)
    
    # Create colored segmentation overlay
    overlay = img_rgb.copy()
    
    # Define colors for different organs
    colors = {
        1: [255, 0, 0],      # Liver - Red
        2: [0, 255, 0],      # Spleen - Green
        3: [0, 0, 255],      # Right Kidney - Blue
        4: [255, 255, 0],    # Left Kidney - Yellow
        5: [255, 0, 255],    # Pancreas - Magenta
    }
    
    for label, color in colors.items():
        mask = seg_slice == label
        overlay[mask] = color
    
    # Blend
    result = cv2.addWeighted(img_rgb, 1 - alpha, overlay, alpha, 0)
    
    return result


def visualize_z_alignment(
        fixed: sitk.Image,
        moving: sitk.Image,
        aligned_fixed: sitk.Image,
        aligned_moving: sitk.Image,
        output_path: str
    ):
    """Create before/after comparison of Z-alignment."""
    import matplotlib.pyplot as plt
    import cv2
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Before alignment - compare middle slices
    fixed_np = sitk.GetArrayFromImage(fixed)      # [D, H, W]
    moving_np = sitk.GetArrayFromImage(moving)    # [D, H, W]
    
    mid_f = fixed_np.shape[0] // 2
    mid_m = moving_np.shape[0] // 2
    
    fixed_slice = fixed_np[mid_f]
    moving_slice = moving_np[mid_m]
    
    # ============================================================
    # FIX: Resample moving to match fixed dimensions
    # ============================================================
    if moving_slice.shape != fixed_slice.shape:
        moving_slice = cv2.resize(
            moving_slice,
            (fixed_slice.shape[1], fixed_slice.shape[0]),  # (W, H)
            interpolation=cv2.INTER_LINEAR
        )
    
    axes[0, 0].imshow(fixed_slice, cmap='gray', vmin=-1000, vmax=500)
    axes[0, 0].set_title(f'Fixed - Before (slice {mid_f}/{fixed_np.shape[0]})')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(moving_slice, cmap='gray', vmin=-1000, vmax=500)
    axes[0, 1].set_title(f'Moving - Before (slice {mid_m}/{moving_np.shape[0]}) [resampled]')
    axes[0, 1].axis('off')
    
    # Now safe to compute difference
    axes[0, 2].imshow(np.abs(fixed_slice - moving_slice), cmap='hot', vmin=0, vmax=500)
    axes[0, 2].set_title('Difference (misaligned)')
    axes[0, 2].axis('off')
    
    # After alignment - same slice index now corresponds to same anatomy!
    aligned_fixed_np = sitk.GetArrayFromImage(aligned_fixed)
    aligned_moving_np = sitk.GetArrayFromImage(aligned_moving)
    
    mid_aligned = aligned_fixed_np.shape[0] // 2
    
    aligned_fixed_slice = aligned_fixed_np[mid_aligned]
    aligned_moving_slice = aligned_moving_np[mid_aligned]
    
    # Resample if needed (should not be needed after alignment, but just in case)
    if aligned_moving_slice.shape != aligned_fixed_slice.shape:
        aligned_moving_slice = cv2.resize(
            aligned_moving_slice,
            (aligned_fixed_slice.shape[1], aligned_fixed_slice.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
    
    axes[1, 0].imshow(aligned_fixed_slice, cmap='gray', vmin=-1000, vmax=500)
    axes[1, 0].set_title(f'Fixed - After (slice {mid_aligned}/{aligned_fixed_np.shape[0]})')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(aligned_moving_slice, cmap='gray', vmin=-1000, vmax=500)
    axes[1, 1].set_title(f'Moving - After (slice {mid_aligned}/{aligned_moving_np.shape[0]})')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(np.abs(aligned_fixed_slice - aligned_moving_slice), cmap='hot', vmin=0, vmax=500)
    axes[1, 2].set_title('Difference (aligned!)')
    axes[1, 2].axis('off')
    
    # Add text summary
    fig.text(0.5, 0.02, 
             f'Before: Fixed {fixed_np.shape}, Moving {moving_np.shape} → '
             f'After: Both {aligned_fixed_np.shape}',
             ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Z-Axis Alignment Quality Check', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Alignment visualization saved: {output_path}")


# Example usage:if __name__ == "__main__":
#     Paths to example volumes
#     Load labels
import os
import pandas as pd
from configs import CROPPED_DIR, MAIN_PATH
labels_csv = MAIN_PATH + "labels.csv"
labels_df = pd.read_csv(labels_csv)   # change sep="," if CSV is comma separated

cropped_dir = MAIN_PATH + "cropped_volumes"
fixed_path = '/Users/macbook/develope/thesis/ncct_cect/vindr_ds/cropped_volumes/1.2.840.113619.2.278.3.717616.204.1584947883.344_1.2.840.113619.2.278.3.717616.204.1584947883.349.3_crop.nii.gz'    
moving_path = '/Users/macbook/develope/thesis/ncct_cect/vindr_ds/cropped_volumes/1.2.840.113619.2.278.3.717616.204.1584947883.344_1.2.840.113619.2.278.3.717616.204.1584947883.792.12_crop.nii.gz'    
output_vis_path = 'alignment_quality.png'
study_id = 'example_study'
series_id = 'example_series'    
# Read volumes
fixed_sitk = sitk.ReadImage(fixed_path)
moving_sitk = sitk.ReadImage(moving_path)    
# Align volumes
aligned_fixed, aligned_moving, align_info = align_volumes_z_axis(
    fixed_sitk, moving_sitk, study_id, series_id
)    


# Visualize alignment quality
visualize_z_alignment(
    fixed_sitk, moving_sitk,
    aligned_fixed, aligned_moving,
    output_vis_path
)
sitk.WriteImage(aligned_fixed, 'aligned_fixed4.nii.gz')
sitk.WriteImage(aligned_moving, 'aligned_moving4.nii.gz')

