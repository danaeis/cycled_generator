import os
import SimpleITK as sitk

def check_spacing_orientation(root_dir, file_ext=".nii.gz"):
    """
    Check if all volumes in each subfolder have the same spacing and orientation.

    Args:
        root_dir (str): Root directory containing study/series folders
        file_ext (str): Expected file extension of volumes (default: .nii.gz)
    """
    for study in os.listdir(root_dir):
        study_path = os.path.join(root_dir, study)
        if not os.path.isdir(study_path):
            continue

        for series in os.listdir(study_path):
            series_path = os.path.join(study_path, series)
            
            if not series.endswith(file_ext):
                continue

            print(f"\n📂 Checking {study}/{series} ...")

            ref_spacing = None
            ref_direction = None
            consistent = True

            try:
                img = sitk.ReadImage(series_path)
                spacing = img.GetSpacing()
                direction = img.GetDirection()

                if ref_spacing is None:
                    ref_spacing = spacing
                    ref_direction = direction
                    print(f"   Reference spacing: {spacing}, orientation: {direction}")
                else:
                    if spacing != ref_spacing or direction != ref_direction:
                        print(f"   ⚠️ Mismatch in {series}: spacing={spacing}, direction={direction}")
                        consistent = False
            except Exception as e:
                print(f"   ❌ Could not read {series}: {e}")

            if consistent:
                print("   ✅ All volumes in this folder are consistent")
            else:
                print("   ❌ Inconsistencies found in this folder")


# ------------------ USAGE ------------------
from configs import CROPPED_DIR, ORIGINAL_fixed_DIR, MAIN_PATH
if __name__ == "__main__":
    # check_spacing_orientation(CROPPED_DIR, file_ext=".nii.gz")
    print("__________________________")
    check_spacing_orientation(MAIN_PATH+"test", file_ext=".nii.gz")

# import os
# import SimpleITK as sitk
# from collections import defaultdict

# def check_study_consistency(cropped_dir):
#     """
#     Check spacing and orientation consistency of volumes belonging to the same study_id.
    
#     Args:
#         cropped_dir (str): Path containing cropped volumes {study_id}_{series_id}_crop.nii.gz
#     """
#     # Group files by study_id
#     study_files = defaultdict(list)
#     for fname in os.listdir(cropped_dir):
#         # if fname.endswith("_crop.nii.gz"):
#         study_id = fname.split("_")[0]  # before first underscore
#         study_files[study_id].append(fname)

#     # Check each study
#     for study_id, files in study_files.items():
#         print(f"\n📂 Checking study {study_id} ...")

#         ref_spacing = None
#         ref_direction = None
#         consistent = True

#         for f in files:
#             fpath = os.path.join(cropped_dir, f)
#             try:
#                 img = sitk.ReadImage(fpath)
#                 spacing = img.GetSpacing()
#                 direction = img.GetDirection()

#                 if ref_spacing is None:
#                     ref_spacing = spacing
#                     ref_direction = direction
#                     print(f"   Reference: {f} | spacing={spacing}, direction={direction}")
#                 else:
#                     if spacing != ref_spacing or direction != ref_direction:
#                         print(f"   ⚠️ Mismatch in {f} | spacing={spacing}, direction={direction}")
#                         consistent = False
#             except Exception as e:
#                 print(f"   ❌ Could not read {f}: {e}")

#         if consistent:
#             print(f"   ✅ Study {study_id}: All series consistent")
#         else:
#             print(f"   ❌ Study {study_id}: Inconsistencies found")

# # ------------------ USAGE ------------------
# from configs import CROPPED_DIR
# if __name__ == "__main__":
#     check_study_consistency(CROPPED_DIR)
