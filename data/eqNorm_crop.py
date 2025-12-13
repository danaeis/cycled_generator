import os
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from skimage import exposure
from tqdm import tqdm
import shutil

# =================== CONFIG ===================
INPUT_DIR = Path("../../ncct_cect/vindr_ds/registered_cases")                    # ← پوشه فعلی ثبت شده
OUTPUT_DIR = Path("../../ncct_cect/vindr_ds/equalized_registered_cases")         # ← خروجی نهایی
BODY_THRESHOLD = -700                                   # برای تشخیص بدن
CLAHE_CLIP_LIMIT = 0.01
MIN_BODY_SLICES = 30  # حداقل تعداد اسلایس با بدن برای معتبر بودن کیس
# ==============================================

def analyze_slice_statistics(vol_np: np.ndarray, body_threshold: int = -700):
    """
    Analyze slice statistics to find optimal padding detection thresholds.
    Run this on a few cases to understand your data.
    """
    D, H, W = vol_np.shape
    print(f"\n📊 Analyzing volume: {D} slices, {H}x{W} pixels")
    print("=" * 70)
    print(f"{'Slice':<8} {'Mean':<10} {'Std':<10} {'Body%':<10} {'Min':<10} {'Max':<10}")
    print("=" * 70)
    
    for z in [0, 1, 2, 10, D//4, D//2, 3*D//4, D-3, D-2, D-1]:  # Sample key slices
        if z >= D:
            continue
        slice_2d = vol_np[z]
        mean_val = slice_2d.mean()
        std_val = slice_2d.std()
        body_pct = 100 * (slice_2d > body_threshold).sum() / (H * W)
        min_val = slice_2d.min()
        max_val = slice_2d.max()
        
        print(f"{z:<8} {mean_val:<10.2f} {std_val:<10.2f} {body_pct:<10.2f} {min_val:<10.2f} {max_val:<10.2f}")
    
    print("=" * 70)
    print("\n💡 Look for patterns:")
    print("   • Padding slices typically have: low body%, consistent mean/std")
    print("   • Body slices have: high body%, variable std, wider value range")



def find_padding_slices(vol_np: np.ndarray, 
                        body_threshold: int = -700,
                        min_body_percent: float = 5.0,      # At least 5% body pixels to be valid
                        std_threshold: float = 50.0,        # INCREASED - more lenient
                        max_padding_mean: float = -400,     # Padding usually < -400 HU
                        margin: int = 3) -> tuple:          # Keep 3 extra slices
    """
    Robust padding detection using body content percentage as primary criterion.
    """
    D, H, W = vol_np.shape
    total_pixels = H * W
    
    # Calculate body percentage for each slice
    body_percent = np.zeros(D)
    mean_intensity = np.zeros(D)
    std_intensity = np.zeros(D)
    
    for z in range(D):
        slice_2d = vol_np[z]
        body_percent[z] = 100 * (slice_2d > body_threshold).sum() / total_pixels
        mean_intensity[z] = slice_2d.mean()
        std_intensity[z] = slice_2d.std()
    
    # Primary criterion: body percentage
    is_padding = body_percent < min_body_percent
    
    # Secondary criterion: typical padding intensity (optional reinforcement)
    # Only mark as padding if BOTH low body AND low intensity
    for z in range(D):
        if not is_padding[z]:  # Already marked as body
            continue
        # Reinforce: if low body but high intensity, might be artifact, keep it
        if mean_intensity[z] > max_padding_mean:
            is_padding[z] = False  # Not typical padding
    
    # Find first body slice from top
    z_start = 0
    for z in range(D):
        if not is_padding[z]:
            z_start = max(0, z + margin)
            break
    
    # Find last body slice from bottom
    z_end = D
    for z in range(D-1, -1, -1):
        if not is_padding[z]:
            z_end = min(D, z - 1 - margin)
            break
    
    # Safety: keep at least 30 slices
    if z_end - z_start < 30:
        print(f"    ⚠ Warning: Only {z_end - z_start} valid slices → keeping all")
        return 0, D
    
    padding_top = z_start
    padding_bottom = D - z_end
    total_removed = padding_top + padding_bottom
    
    if total_removed > 0:
        print(f"    ✓ Padding detected: {padding_top} top + {padding_bottom} bottom = {total_removed} slices removed")
        print(f"    → Keeping slices {z_start} to {z_end-1} ({z_end - z_start} slices)")
    else:
        print(f"    → No padding detected (all {D} slices have body content)")
    
    return z_start, z_end


def find_body_range(vol_np: np.ndarray, threshold: int = BODY_THRESHOLD) -> tuple:
    """پیدا کردن اولین و آخرین اسلایس که بدن داره"""
    mask = vol_np > threshold
    body_per_slice = np.sum(mask, axis=(1, 2))
    valid_slices = np.where(body_per_slice > 1000)[0]  # حداقل 1000 پیکسل بدن
    
    if len(valid_slices) < MIN_BODY_SLICES:
        return None, None  # کیس نامعتبر
    
    z_start = valid_slices.min()
    z_end = valid_slices.max() + 1
    return z_start, z_end


def ct_clahe_equalization(vol_np: np.ndarray) -> np.ndarray:
    """CLAHE روی هر اسلایس + خروجی 0–255"""
    result = []
    for z in range(vol_np.shape[0]):
        slice_2d = vol_np[z].astype(np.float32)
        
        # پنجره 2% تا 98%
        p2, p98 = np.percentile(slice_2d[slice_2d > -700], (2, 98))
        slice_clipped = np.clip(slice_2d, p2, p98)
        
        # به 0–65535 ببر برای CLAHE
        slice_uint16 = exposure.rescale_intensity(slice_clipped, in_range=(p2, p98), out_range=(0, 65535)).astype(np.uint16)
        
        # CLAHE
        eq = exposure.equalize_adapthist(slice_uint16, clip_limit=CLAHE_CLIP_LIMIT)
        eq_uint8 = (eq * 255).astype(np.uint8)
        
        result.append(eq_uint8)
    
    return np.stack(result, axis=0)

def process_case(case_path: Path):
    case_id = case_path.name
    print(f"\n{'='*70}")
    print(f"Processing case: {case_id}")
    print('='*70)
    
    image_files = sorted([f for f in case_path.glob("*_registered.nii.gz") if "_seg" not in f.name])
    if not image_files:
        print("  ❌ No volumes found")
        return
    
    print(f"  Found {len(image_files)} series")
    
    # ===================================================================
    # STEP 1: Load all volumes ONCE and detect padding in each
    # ===================================================================
    volumes_data = []
    padding_ranges = []
    
    for idx, img_file in enumerate(image_files, 1):
        try:
            print(f"\n  [{idx}/{len(image_files)}] Analyzing: {img_file.name}")
            vol_sitk = sitk.ReadImage(str(img_file))
            vol_np = sitk.GetArrayFromImage(vol_sitk)
            
            # Optional: show diagnostic for first series only
            if idx == 1:
                print("\n  🔍 Diagnostic stats:")
                analyze_slice_statistics(vol_np, BODY_THRESHOLD)
            
            # Detect padding
            z_start, z_end = find_padding_slices(vol_np, body_threshold=BODY_THRESHOLD, 
                                                 min_body_percent=10.0, margin=10)
            
            volumes_data.append({
                'file': img_file,
                'sitk': vol_sitk,
                'numpy': vol_np,
                'z_range': (z_start, z_end)
            })
            padding_ranges.append((z_start, z_end))
            
        except Exception as e:
            print(f"    ❌ Error loading: {e}")
            return
    
    # ===================================================================
    # STEP 2: Find common range (intersection of all valid ranges)
    # ===================================================================
    common_z_start = max(r[0] for r in padding_ranges)
    common_z_end = min(r[1] for r in padding_ranges)
    
    if common_z_end <= common_z_start:
        print("\n  ❌ No common valid range → skipping case")
        return
    
    common_slices = common_z_end - common_z_start
    print(f"\n  ✅ COMMON RANGE: [{common_z_start}:{common_z_end}] = {common_slices} slices")
    
    # ===================================================================
    # STEP 3: Crop and process all volumes
    # ===================================================================
    out_case_dir = OUTPUT_DIR / case_id
    out_case_dir.mkdir(parents=True, exist_ok=True)
    
    for vol_data in volumes_data:
        try:
            img_file = vol_data['file']
            vol_sitk_original = vol_data['sitk']
            vol_np_full = vol_data['numpy']
            
            # Crop to common range
            cropped_np = vol_np_full[common_z_start:common_z_end]
            
            # Apply CLAHE equalization
            print(f"\n  Processing: {img_file.name}")
            print(f"    Original: {vol_np_full.shape[0]} slices → Cropped: {cropped_np.shape[0]} slices")
            eq_np = ct_clahe_equalization(cropped_np)
            
            # Create NEW sitk image (don't copy from original - size mismatch!)
            # vol_out = sitk.GetImageFromArray(eq_np)
            vol_out = sitk.GetImageFromArray(eq_np, isVector=False)

            # Copy physical properties EXCEPT size
            vol_out.SetSpacing(vol_sitk.GetSpacing())
            vol_out.SetDirection(vol_sitk.GetDirection())

            # Update origin: shift by the number of removed top slices
            origin = np.array(vol_sitk.GetOrigin())
            spacing = np.array(vol_sitk.GetSpacing())
            direction_matrix = np.array(vol_sitk.GetDirection()).reshape(3, 3)
            z_unit_vector = direction_matrix[:, 2]  # direction of Z axis

            # Physical shift = common_z_start * spacing_z * z_direction
            origin_shift = common_z_start * spacing[2] * z_unit_vector
            new_origin = origin + origin_shift

            vol_out.SetOrigin(new_origin.tolist())


            # Copy ONLY spacing, origin, direction (not size-dependent metadata)
            # vol_out.SetSpacing(vol_sitk_original.GetSpacing())
            # vol_out.SetOrigin(vol_sitk_original.GetOrigin())
            # vol_out.SetDirection(vol_sitk_original.GetDirection())
            
            # Save normalized volume
            new_name = img_file.name.replace("_registered.nii.gz", "_registered_norm.nii.gz")
            out_path = out_case_dir / new_name
            sitk.WriteImage(vol_out, str(out_path))
            print(f"    ✅ Saved: {new_name}")
            
            # ===================================================================
            # STEP 4: Process corresponding segmentation mask (if exists)
            # ===================================================================
            for suffix in ["_registered_seg.nii.gz", "_seg.nii.gz"]:
                seg_path = img_file.parent / img_file.name.replace("_registered.nii.gz", suffix)
                if seg_path.exists():
                    seg_sitk = sitk.ReadImage(str(seg_path))
                    seg_np = sitk.GetArrayFromImage(seg_sitk)
                    
                    # Crop mask to same range
                    seg_cropped = seg_np[common_z_start:common_z_end]
                    
                    # Create new mask image
                    seg_out = sitk.GetImageFromArray(seg_cropped.astype(np.uint8))
                    # vol_out = sitk.GetImageFromArray(eq_np, isVector=False)

                    # Copy physical properties EXCEPT size
                    seg_out.SetSpacing(vol_sitk.GetSpacing())
                    seg_out.SetDirection(vol_sitk.GetDirection())

                    # Update origin: shift by the number of removed top slices
                    origin = np.array(vol_sitk.GetOrigin())
                    spacing = np.array(vol_sitk.GetSpacing())
                    direction_matrix = np.array(vol_sitk.GetDirection()).reshape(3, 3)
                    z_unit_vector = direction_matrix[:, 2]  # direction of Z axis

                    # Physical shift = common_z_start * spacing_z * z_direction
                    origin_shift = common_z_start * spacing[2] * z_unit_vector
                    new_origin = origin + origin_shift

                    seg_out.SetOrigin(new_origin.tolist())

                    # seg_out.SetSpacing(seg_sitk.GetSpacing())
                    # seg_out.SetOrigin(seg_sitk.GetOrigin())
                    # seg_out.SetDirection(seg_sitk.GetDirection())
                    
                    seg_out_name = new_name.replace("_norm.nii.gz", "_seg.nii.gz")
                    seg_out_path = out_case_dir / seg_out_name
                    sitk.WriteImage(seg_out, str(seg_out_path))
                    print(f"    ✅ Saved mask: {seg_out_name}")
                    break
                    
        except Exception as e:
            print(f"    ❌ Error processing: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n  ✅ Case {case_id} completed!\n")

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("="*70)
    print("CT NORMALIZATION + PADDING REMOVAL + CLAHE EQUALIZATION")
    print("="*70)
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Body threshold: {BODY_THRESHOLD} HU")
    print(f"CLAHE clip limit: {CLAHE_CLIP_LIMIT}")
    print(f"Min body%: 10.0%")
    print(f"Margin: 7 slices")
    print("="*70)
    
    case_dirs = sorted([p for p in INPUT_DIR.iterdir() if p.is_dir()])
    print(f"\nFound {len(case_dirs)} cases to process\n")
    
    for case_path in case_dirs:
        process_case(case_path)
    
    print("\n" + "="*70)
    print("✅ ALL DONE!")
    print("="*70)
    print(f"Results saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()