# import os
# import pandas as pd
# import SimpleITK as sitk
# import shutil

# def register_study(study_id, cropped_dir, output_dir, labels_df):
#     """
#     Register all series in a study to the non-contrast volume.
    
#     Args:
#         study_id (str): StudyInstanceUID
#         cropped_dir (str): Path to cropped volumes (studyname_seriesname_crop.nii.gz)
#         output_dir (str): Where to save registered results
#         labels_df (pd.DataFrame): DataFrame with StudyInstanceUID, SeriesInstanceUID, Label
#     """
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Find the non-contrast row
#     nc_row = labels_df[(labels_df["StudyInstanceUID"] == study_id) & (labels_df["Label"] == "Non-contrast")]
#     if nc_row.empty:
#         print(f"⚠️ No non-contrast found for {study_id}, skipping...")
#         return
    
#     nc_series = nc_row.iloc[0]["SeriesInstanceUID"]
#     nc_file = os.path.join(cropped_dir, f"{study_id}_{nc_series}_crop.nii.gz")
    
#     if not os.path.exists(nc_file):
#         print(f"⚠️ Non-contrast file not found: {nc_file}")
#         return
    
#     fixed = sitk.ReadImage(nc_file)
#     nc_basename = os.path.basename(nc_file)
    
#     # Get all series for this study
#     series_rows = labels_df[labels_df["StudyInstanceUID"] == study_id]
    
#     for _, row in series_rows.iterrows():
#         series_id = row["SeriesInstanceUID"]
#         moving_file = os.path.join(cropped_dir, f"{study_id}_{series_id}_crop.nii.gz")
        
#         if not os.path.exists(moving_file) or moving_file == nc_file:
#             continue
        
#         try:
#             moving = sitk.ReadImage(moving_file)
            
#             # Initialize transform
#             initial_transform = sitk.CenteredTransformInitializer(
#                 fixed, moving, sitk.Euler3DTransform(),
#                 sitk.CenteredTransformInitializerFilter.GEOMETRY
#             )
            
#             # Registration method
#             registration = sitk.ImageRegistrationMethod()
#             registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
#             registration.SetMetricSamplingStrategy(registration.RANDOM)
#             registration.SetMetricSamplingPercentage(0.2)
#             registration.SetInterpolator(sitk.sitkLinear)
#             registration.SetOptimizerAsGradientDescent(
#                 learningRate=1.0,
#                 numberOfIterations=100,
#                 convergenceMinimumValue=1e-6,
#                 convergenceWindowSize=10
#             )
#             registration.SetOptimizerScalesFromPhysicalShift()
#             registration.SetInitialTransform(initial_transform, inPlace=False)
            
#             # Execute
#             final_transform = registration.Execute(fixed, moving)
            
#             # Print metrics
#             print(f"📊 {study_id} | {series_id}:")
#             print(f"   Final metric value: {registration.GetMetricValue():.6f}")
#             print(f"   Optimizer stop condition: {registration.GetOptimizerStopConditionDescription()}")
            
#             # Resample moving image
#             registered = sitk.Resample(
#                 moving, fixed, final_transform,
#                 sitk.sitkLinear, 0.0, moving.GetPixelID()
#             )
            
#             # Save registered image
#             out_path = os.path.join(output_dir, f"{study_id}_{series_id}_registered.nii.gz")
#             sitk.WriteImage(registered, out_path)
#             print(f"   ✅ Saved registered: {out_path}")
        
#         except Exception as e:
#             print(f"❌ Error registering {study_id}_{series_id}: {e}")
    
#     # Copy reference (non-contrast) to output_dir
#     nc_out = os.path.join(output_dir, f"{study_id}_{nc_series}_registered.nii.gz")
#     shutil.copy(nc_file, nc_out)
#     print(f"   Copied reference non-contrast: {nc_out}")

import os
import pandas as pd
import SimpleITK as sitk
import shutil
import json

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
    os.makedirs(output_dir, exist_ok=True)

    # --- Find the non-contrast row ---
    nc_row = labels_df[(labels_df["StudyInstanceUID"] == study_id) & (labels_df["Label"] == "Non-contrast")]
    if nc_row.empty:
        print(f"⚠️ No non-contrast found for {study_id}, skipping...")
        return

    nc_series = nc_row.iloc[0]["SeriesInstanceUID"]
    nc_file = os.path.join(cropped_dir, f"{study_id}_{nc_series}_crop.nii.gz")
    if not os.path.exists(nc_file):
        print(f"⚠️ Non-contrast file not found: {nc_file}")
        return

    fixed = sitk.ReadImage(nc_file)

    # --- Process all series in this study ---
    series_rows = labels_df[labels_df["StudyInstanceUID"] == study_id]
    for _, row in series_rows.iterrows():
        series_id = row["SeriesInstanceUID"]
        moving_file = os.path.join(cropped_dir, f"{study_id}_{series_id}_crop.nii.gz")

        if not os.path.exists(moving_file) or moving_file == nc_file:
            continue

        try:
            moving = sitk.ReadImage(moving_file)

            # Initialize transform
            initial_transform = sitk.CenteredTransformInitializer(
                fixed, moving, sitk.Euler3DTransform(),
                sitk.CenteredTransformInitializerFilter.GEOMETRY
            )

            # Registration method
            registration = sitk.ImageRegistrationMethod()
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

            # Execute registration
            final_transform = registration.Execute(fixed, moving)

            # Print metrics
            print(f"📊 {study_id} | {series_id}: {registration.GetMetricValue():.6f}")

            # --- Resample moving image ---
            registered = sitk.Resample(
                moving, fixed, final_transform,
                sitk.sitkLinear, 0.0, moving.GetPixelID()
            )

            out_vol_path = os.path.join(output_dir, f"{study_id}_{series_id}_registered.nii.gz")
            sitk.WriteImage(registered, out_vol_path)
            print(f"   ✅ Saved registered volume: {out_vol_path}")

            # --- Apply same transform to mask if it exists ---
            seg_file = os.path.join(cropped_dir, f"{study_id}_{series_id}_seg.nii.gz")
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
    shutil.copy(nc_file, nc_out)

    seg_nc_file = os.path.join(cropped_dir, f"{study_id}_{nc_series}_seg.nii.gz")
    # print("seg_nc_file", seg_nc_file)
    if os.path.exists(seg_nc_file):
        nc_seg_out = os.path.join(output_dir, f"{study_id}_{nc_series}_registered_seg.nii.gz")
        shutil.copy(seg_nc_file, nc_seg_out)

    print(f"   Copied reference non-contrast for {study_id}")

# ------------------ USAGE ------------------

# Load labels
from configs import CROPPED_DIR, MAIN_PATH
labels_csv = MAIN_PATH + "labels.csv"
labels_df = pd.read_csv(labels_csv)   # change sep="," if CSV is comma separated

cropped_dir = MAIN_PATH + "cropped_volumes"
registered_dir = MAIN_PATH + "registered_cases"

os.makedirs(registered_dir, exist_ok=True)

# Register each study
for study_id in labels_df["StudyInstanceUID"].unique():
    study_out = os.path.join(registered_dir, study_id)
    register_study(study_id, cropped_dir, study_out, labels_df)

print("✅ Registration completed for all studies")
