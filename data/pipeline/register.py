import os
import pandas as pd
import SimpleITK as sitk
import shutil
import json
import numpy as np

# from align_data import apply_window_and_normalize, visualize_registration_quality, register_study

def visualize_registration_quality(fixed, moving, registered, output_path):
    """Create checkerboard comparison."""
    import matplotlib.pyplot as plt
    
    # Get middle slices
    fixed_np = sitk.GetArrayFromImage(fixed)
    moving_np = sitk.GetArrayFromImage(moving)
    registered_np = sitk.GetArrayFromImage(registered)
    
    mid_z = fixed_np.shape[0] // 2
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(fixed_np[mid_z], cmap='gray')
    axes[0, 0].set_title('Fixed (Non-contrast)')
    
    axes[0, 1].imshow(moving_np[mid_z], cmap='gray')
    axes[0, 1].set_title('Moving (Before Registration)')
    
    axes[0, 2].imshow(registered_np[mid_z], cmap='gray')
    axes[0, 2].set_title('Registered')
    
    # Difference maps
    axes[1, 0].imshow(np.abs(fixed_np[mid_z] - moving_np[mid_z]), cmap='hot')
    axes[1, 0].set_title('|Fixed - Moving|')
    
    axes[1, 1].imshow(np.abs(fixed_np[mid_z] - registered_np[mid_z]), cmap='hot')
    axes[1, 1].set_title('|Fixed - Registered|')
    
    # Checkerboard
    checker = np.copy(fixed_np[mid_z])
    h, w = checker.shape
    checker[::32, :] = registered_np[mid_z][::32, :]
    checker[:, ::32] = registered_np[mid_z][:, ::32]
    axes[1, 2].imshow(checker, cmap='gray')
    axes[1, 2].set_title('Checkerboard Overlay')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def match_histogram(source: np.ndarray, 
                   reference: np.ndarray,
                   body_threshold: int = -500,
                   nbins: int = 256) -> np.ndarray:
    """
    Match histogram of source to reference volume.
    
    This ensures all phases have similar intensity distributions,
    making training more stable.
    
    Based on scikit-image's exposure.match_histograms but optimized for CT.
    """
    from scipy import interpolate
    
    # Get body masks
    source_mask = source > body_threshold
    reference_mask = reference > body_threshold
    
    if not (np.any(source_mask) and np.any(reference_mask)):
        print("Warning: Body mask empty in histogram matching")
        return source
    
    # Extract body intensities
    source_values = source[source_mask].ravel()
    reference_values = reference[reference_mask].ravel()
    
    # Compute CDFs (Cumulative Distribution Functions)
    src_hist, src_bins = np.histogram(source_values, bins=nbins, density=True)
    ref_hist, ref_bins = np.histogram(reference_values, bins=nbins, density=True)
    
    src_cdf = np.cumsum(src_hist)
    src_cdf = src_cdf / src_cdf[-1]  # Normalize
    
    ref_cdf = np.cumsum(ref_hist)
    ref_cdf = ref_cdf / ref_cdf[-1]
    
    # Build lookup table
    src_centers = (src_bins[:-1] + src_bins[1:]) / 2
    ref_centers = (ref_bins[:-1] + ref_bins[1:]) / 2
    
    # Interpolate inverse CDF
    interp_func = interpolate.interp1d(
        src_cdf, src_centers, 
        bounds_error=False, 
        fill_value=(src_centers[0], src_centers[-1])
    )
    
    # Map source to reference distribution
    matched = source.copy()
    matched[source_mask] = np.interp(
        source[source_mask],
        src_centers,
        np.interp(src_cdf, ref_cdf, ref_centers)
    )
    
    return matched


def smart_hu_clip(volume: np.ndarray, 
                  body_threshold: int = -500,
                  percentile_low: float = 1.0,      # ← Changed from 0.5
                  percentile_high: float = 99.0,    # ← Changed from 99.5
                  adaptive_margin: bool = True) -> tuple:
    """
    Adaptive HU windowing that preserves contrast while normalizing distributions.
    
    Key improvement: Uses IQR-based margins instead of fixed values.
    """
    # Step 1: Get body mask
    body_mask = volume > body_threshold
    
    if not np.any(body_mask):
        print("Warning: No body found, using default window")
        return -1000, 600
    
    body_intensities = volume[body_mask].ravel()
    
    # Step 2: Compute robust percentiles
    low = np.percentile(body_intensities, percentile_low)
    high = np.percentile(body_intensities, percentile_high)
    
    # Step 3: ADAPTIVE margin based on IQR (interquartile range)
    if adaptive_margin:
        q25 = np.percentile(body_intensities, 25)
        q75 = np.percentile(body_intensities, 75)
        iqr = q75 - q25
        
        # Use 1.5 * IQR for outlier detection (standard statistical approach)
        margin_low = int(1.5 * iqr)
        margin_high = int(1.5 * iqr)
    else:
        margin_low = 300
        margin_high = 300
    
    window_min = int(low - margin_low)
    window_max = int(high + margin_high)
    
    # Step 4: Clamp to reasonable CT range
    window_min = max(window_min, -1500)
    window_max = min(window_max, 2000)
    
    print(f"Adaptive HU Window → Min: {window_min}, Max: {window_max} | "
          f"Width: {window_max - window_min} | IQR: {iqr:.1f}")
    
    return window_min, window_max

def apply_window_and_normalize(vol_sitk: sitk.Image,
                               reference_vol: sitk.Image = None) -> sitk.Image:
    """
    Apply adaptive windowing + optional histogram matching + normalize to [0, 1].
    
    Args:
        vol_sitk: Input volume
        reference_vol: If provided, match histogram to this volume (e.g., non-contrast)
    """
    vol_np = sitk.GetArrayFromImage(vol_sitk)
    
    # Step 1: Histogram matching (if reference provided)
    if reference_vol is not None:
        ref_np = sitk.GetArrayFromImage(reference_vol)
        vol_np = match_histogram(vol_np, ref_np)
        print("   ✓ Histogram matched to reference")
    
    # Step 2: Adaptive windowing
    w_min, w_max = smart_hu_clip(vol_np, adaptive_margin=True)
    
    # Step 3: Clip + Normalize to [0, 1]
    vol_clipped = np.clip(vol_np, w_min, w_max)
    vol_normalized = (vol_clipped - w_min) / (w_max - w_min + 1e-8)
    
    # Step 4: Convert back to SimpleITK
    vol_out = sitk.GetImageFromArray(vol_normalized)
    vol_out.CopyInformation(vol_sitk)
    
    # Save metadata
    vol_out.SetMetaData("hu_window_min", str(w_min))
    vol_out.SetMetaData("hu_window_max", str(w_max))
    vol_out.SetMetaData("normalization", "adaptive_histogram_matched")
    
    return vol_out

def prepare_images_for_registration(fixed: sitk.Image, moving: sitk.Image, threshold: int = -500):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLanczosWindowedSinc) # use sitkWindowedSinc or BSpline for better quality than linear
    resampler.SetDefaultPixelValue(-1000)
    moving_resampled = resampler.Execute(moving)
    
    mask = fixed > threshold
    mask = sitk.BinaryMorphologicalClosing(mask, [3,3,3])
    mask = sitk.BinaryFillhole(mask)
    mask = sitk.Cast(mask, sitk.sitkUInt8)
    
    return moving_resampled, mask

def register_study(study_id, cropped_dir, output_dir, labels_df):
    """
    Register all series in a study to the non-contrast volume.
    Also applies the transform to segmentation masks if available.

    Args:
        study_id (str): StudyInstanceUID
        cropped_dir (str): Path to cropped volumes (*.nii.gz)
        output_dir (str): Where to save registered results
        labels_df (pd.DataFrame): DataFrame with StudyInstanceUID, SeriesInstanceUID, Label
    """
    

    # --- Find the non-contrast row ---
    nc_row = labels_df[(labels_df["StudyInstanceUID"] == study_id) & (labels_df["Label"] == "Non-contrast")]
    if nc_row.empty:
        # print(f"⚠️ No non-contrast found for {study_id}, skipping...")
        return

    nc_series = nc_row.iloc[0]["SeriesInstanceUID"]
    nc_file = os.path.join(cropped_dir, f"{study_id}/{study_id}_{nc_series}_aligned.nii.gz")
    if not os.path.exists(nc_file):
        # print(f"⚠️ Non-contrast file not found: {nc_file}")
        return


    output_dir = os.path.join(output_dir, study_id)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📁 Registering Study: {study_id} | Non-contrast Series: {nc_series}")
    fixed = sitk.ReadImage(nc_file)

    # --- Process all series in this study ---
    series_rows = labels_df[labels_df["StudyInstanceUID"] == study_id]
    for _, row in series_rows.iterrows():
        series_id = row["SeriesInstanceUID"]
        # print(f"\n🔄 Registering {study_id} | Series: {series_id}")
        moving_file = os.path.join(cropped_dir, f"{study_id}/{study_id}_{series_id}_aligned.nii.gz")

        if not os.path.exists(moving_file) or moving_file == nc_file:
            print(f"   ⏭️ Skipping (file not found or is non-contrast): {moving_file}")
            continue

        try:
            # moving = sitk.ReadImage(moving_file)
            moving_raw = sitk.ReadImage(moving_file)
            moving, fixed_mask = prepare_images_for_registration(fixed, moving_raw)
            # Initialize transform
            initial_transform = sitk.CenteredTransformInitializer(
                fixed, moving, sitk.Euler3DTransform(),
                sitk.CenteredTransformInitializerFilter.GEOMETRY
            )

            # Registration method
            registration = sitk.ImageRegistrationMethod()
            registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
            registration.SetMetricFixedMask(fixed_mask)            # ← ماسک رو اینجا می‌دیم
            registration.SetMetricSamplingStrategy(registration.RANDOM)
            registration.SetMetricSamplingPercentage(0.20)
            registration.SetInterpolator(sitk.sitkLinear)

            registration.SetOptimizerAsGradientDescent(
                learningRate=1.0,
                numberOfIterations=200,    # یه کم بیشتر بذار بهتر همگرا می‌شه
                convergenceMinimumValue=1e-6,
                convergenceWindowSize=10
            )
            registration.SetOptimizerScalesFromPhysicalShift()
            registration.SetInitialTransform(initial_transform, inPlace=False)

            # Execute registration
            final_transform = registration.Execute(fixed, moving)

            # Print metrics
            print(f"📊 {study_id} | {series_id}: {registration.GetMetricValue():.6f}")

            # --- Resample moving image ---
            registered = sitk.Resample(
                moving, fixed, final_transform,
                sitk.sitkLanczosWindowedSinc, 0.0, moving.GetPixelID()
            )
            
            out_vol_path = os.path.join(output_dir, f"{study_id}_{series_id}_registered.nii.gz")

            # Apply histogram matching + normalization (use non-contrast as reference)
            print(f"   Applying histogram matching + normalization...")
            registered_normalized = apply_window_and_normalize(
                registered, 
                reference_vol=fixed  # ← Match to non-contrast distribution
            )
            
            out_norm_path = out_vol_path.replace(".nii.gz", "_norm.nii.gz")
            sitk.WriteImage(registered_normalized, out_norm_path)
            print(f"   Saved normalized volume: {out_norm_path}")

            sitk.WriteImage(registered, out_vol_path)
            print(f"   Saved original (raw HU): {out_vol_path}")

            print(f"   ✅ Saved registered volume: {out_vol_path}")
            # After registration
            visualize_registration_quality(
                fixed, moving, registered,
                os.path.join(output_dir, f"{study_id}_{series_id}_registration_check.png")
            )
            # --- Apply same transform to mask if it exists ---
            seg_file = os.path.join(cropped_dir, f"{study_id}/{study_id}_{series_id}_aligned_seg.nii.gz")
            if os.path.exists(seg_file):
                seg = sitk.ReadImage(seg_file)

                registered_seg = sitk.Resample(
                    seg, fixed, final_transform,
                    sitk.sitkNearestNeighbor, 0, seg.GetPixelID()   # 🚨 nearest neighbor for masks
                )

                out_seg_path = os.path.join(output_dir, f"{study_id}_{series_id}_registered_seg.nii.gz")
                sitk.WriteImage(registered_seg, out_seg_path)
                print(f"   ✅ Saved registered mask: {out_seg_path}")

        except Exception as e:
            print(f"❌ Error registering {study_id}_{series_id}: {e}")

    # --- Copy reference (non-contrast) volume and mask ---
    nc_out = os.path.join(output_dir, f"{study_id}_{nc_series}_registered.nii.gz")
    # shutil.copy(nc_file, nc_out)
    #non-contrast
    fixed_raw = sitk.ReadImage(nc_file)
    fixed_normalized = apply_window_and_normalize(
        fixed_raw,
        reference_vol=None  # ← Non-contrast serves as its own reference
    )

    nc_norm_out = nc_out.replace(".nii.gz", "_norm.nii.gz")
    sitk.WriteImage(fixed_normalized, nc_norm_out)
    print(f"   Saved normalized reference: {nc_norm_out}")

    # نسخه خام هم کپی کن
    shutil.copy(nc_file, nc_out)

    seg_nc_file = os.path.join(cropped_dir, f"{study_id}/{study_id}_{nc_series}_aligned_seg.nii.gz")
    # print("seg_nc_file", seg_nc_file)
    if os.path.exists(seg_nc_file):
        nc_seg_out = os.path.join(output_dir, f"{study_id}_{nc_series}_registered_seg.nii.gz")
        shutil.copy(seg_nc_file, nc_seg_out)

    print(f"   Copied reference non-contrast for {study_id}")

# ------------------ USAGE ------------------
MAIN_PATH = "../../ncct_cect/vindr_ds/"
labels_csv = MAIN_PATH + "labels.csv"
labels_df = pd.read_csv(labels_csv)   # change sep="," if CSV is comma separated

# cropped_dir = MAIN_PATH + "aligned_volumes"
cropped_dir = "./aligned_output"
registered_dir = "./registered_cases_aligned"

os.makedirs(registered_dir, exist_ok=True)
study_ids = labels_df["StudyInstanceUID"].unique()

for idx, study_id in enumerate(study_ids, 1):
    # print(f"\n{'#'*80}")
    # print(f"Processing Study {idx}/{len(study_ids)}: {study_id}")
    # print(f"{'#'*80}")
    
    register_study(study_id, cropped_dir, registered_dir, labels_df)
    
    # print("\n" + "="*80)
    # print("✅ ALL REGISTRATIONS COMPLETED")
    # print("="*80)

# # Register each study
# for study_id in labels_df["StudyInstanceUID"].unique():
#     study_out = os.path.join(registered_dir, study_id)
#     register_study(study_id, cropped_dir, study_out, labels_df)

# print("✅ Registration completed for all studies")
