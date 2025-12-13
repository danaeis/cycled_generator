# import os
# import pandas as pd
# import nibabel as nib
# from pathlib import Path
# from collections import defaultdict

# # ---------------------- CONFIGURE YOUR PATHS HERE ----------------------
# base_path = Path("../ncct_cect/vindr_ds/")  # <-- CHANGE THIS TO YOUR MAIN FOLDER

# cropped_volumes_folder = base_path / "cropped_volumes"
# cropped_volumes_folder = Path(cropped_volumes_folder)
# label_csv_path = base_path / "labels.csv"
# exclude_csv_path = "/media/disk1/saeedeh_danaei/cycled_generator/similarity_analysis/pairs_to_exclude.csv"  #base_path / "pairs_to_exclude.csv"

# output_csv_path = base_path / "volume_summary.csv"

# # Expected phase names (change only if your label.csv uses different spelling)
# PHASE_NAMES = ["non-contrast", "arterial", "venous", "other"]

# # ---------------------- LOAD LABELS AND EXCLUSIONS ----------------------
# print("Loading label.csv and pairs_to_exclude.csv...")

# df_labels = pd.read_csv(label_csv_path)
# # Expected columns in label.csv: case_id, series_id, phase (or similar)
# # Adjust column names below if yours are different
# # Common variations:
# case_col = "StudyInstanceUID"      # try also: "caseID", "CaseID", "patient_id"
# series_col = "SeriesInstanceUID"  # try also: "seriesID", "SeriesID"
# phase_col = "Label"       # try also: "Phase", "label"

# # Make sure we have the right column names
# if case_col not in df_labels.columns:
#     print(f"Warning: '{case_col}' not found. Available columns:", list(df_labels.columns))
# if series_col not in df_labels.columns:
#     print(f"Warning: '{series_col}' not found. Available columns:", list(df_labels.columns))
# if phase_col not in df_labels.columns:
#     print(f"Warning: '{phase_col}' not found. Available columns:", list(df_labels.columns))

# # Load exclusions
# exclude_pairs = set()
# if Path(exclude_csv_path).exists():
#     df_exclude = pd.read_csv(exclude_csv_path)
#     for _, row in df_exclude.iterrows():
#         case_id = str(row['case_id']).strip()
#         src = str(row['source_phase']).strip()
#         tgt = str(row['target_phase']).strip()
#         exclude_pairs.add((case_id, src, tgt))
#         # exclude_pairs.add((case_id, tgt, src))  # bidirectional just in case
#     print(f"Loaded {len(exclude_pairs)} exclusion pairs.")
# else:
#     print("No pairs_to_exclude.csv found, assuming none are excluded.")

# # ---------------------- PROCESS EACH CROPPED FILE ----------------------
# records = []

# for crop_file in cropped_volumes_folder.glob("*_crop.nii.gz"):
#     try:
#         seg_file = crop_file.parent / crop_file.name.replace("_crop.nii.gz", "_seg.nii.gz")
#         if not seg_file.exists():
#             print(f"Missing seg file for {crop_file.name}, skipping...")
#             continue

#         # Parse filename: caseID_seriesID_crop.nii.gz
#         basename = crop_file.stem.replace("_crop", "")
#         if "_" not in basename:
#             print(f"Unexpected filename format: {crop_file.name}")
#             continue
#         case_id, series_id = basename.split("_", 1)
#         case_id = case_id.strip()
#         series_id = series_id.strip().replace(".nii", "")
#         # print(f"case_id:{case_id} and series id: {series_id}")
#         # Load the cropped volume to get shape and spacing
#         img = nib.load(crop_file)
#         shape = img.shape
#         spacing = img.header.get_zooms()  # (x, y, z)
#         slice_thickness = spacing[2]      # usually the z-spacing

#         # Find the phase label for this case_id + series_id
#         mask = (df_labels[case_col].astype(str) == case_id) & (df_labels[series_col].astype(str) == series_id)
#         phase_rows = df_labels[mask]
#         if phase_rows.empty:
#             print(f"No phase label found for case {case_id}, series {series_id}")
#             phase = "unknown"
#         else:
#             phase = phase_rows.iloc[0][phase_col].strip().lower()
#             # Normalize phase names
#             if "non" in phase or "plain" in phase or "nc" in phase:
#                 phase = "non-contrast"
#             elif "art" in phase or "ater" in phase:
#                 phase = "arterial"
#             elif "ven" in phase or "pv" in phase or "port" in phase:
#                 phase = "venous"
#             else:
#                 phase = "other"

#         # Check if this case has any excluded phase pair
#         excluded_info = ""
#         for excl_case, src, tgt in exclude_pairs:
#             if excl_case == case_id:
#                 excluded_info = f"{tgt}"
#                 break

#         # Store the info per phase for this case
#         record = {
#             "case_id": case_id,
#             "phase": phase,
#             "series_id": series_id,
#             "shape": "x".join(map(str, shape)),
#             "slice_thickness": round(float(slice_thickness), 4),
#             "excluded_info": excluded_info
#         }
#         records.append(record)

#     except Exception as e:
#         print(f"Error processing {crop_file.name}: {e}")

#     # break

# # ---------------------- GROUP BY CASE_ID ----------------------
# case_dict = defaultdict(dict)
# # print("len records: ",len(records))
# # print(records)
# for rec in records:
#     cid = rec["case_id"]
#     phase = rec["phase"]
#     case_dict[cid][phase] = {
#         "shape": rec["shape"],
#         "thickness": rec["slice_thickness"]
#     }
#     # Carry forward exclusion info (same for all phases of a case)
#     if rec["excluded_info"]:
#         case_dict[cid]["excluded"] = rec["excluded_info"]

# # ---------------------- BUILD FINAL TABLE ----------------------
# final_rows = []

# for case_id, phasedata in case_dict.items():
#     row = {"case_id": case_id}
    
#     for phase in PHASE_NAMES:
#         if phase in phasedata and phase != "excluded":
#             row[f"{phase} shape"] = phasedata[phase]["shape"]
#             row[f"{phase} slice-thickness"] = phasedata[phase]["thickness"]
#         else:
#             row[f"{phase} shape"] = ""
#             row[f"{phase} slice-thickness"] = ""
    
#     excl = phasedata.get("excluded", "")
#     row["is_excluded"] = excl if excl else "no"
    
#     final_rows.append(row)

# final_df = pd.DataFrame(final_rows)
# # Order columns exactly as you requested
# column_order = [
#     "case_id",
#     "non-contrast shape", "arterial shape", "venous shape", "other shape",
#     "non-contrast slice-thickness", "arterial slice-thickness", 
#     "venous slice-thickness", "other slice-thickness",
#     "is_excluded"
# ]
# final_df = final_df[column_order]

# # Sort by case_id for easier reading
# final_df = final_df.sort_values("case_id").reset_index(drop=True)

# # Save
# final_df.to_csv(output_csv_path, index=False)
# print(f"\nDone! Saved summary for {len(final_df)} cases to:")
# print(output_csv_path)
# print("\nFirst few rows:")
# print(final_df.head())



import pandas as pd
from pathlib import Path

# Change this to where your CSV is
csv_path = Path("../ncct_cect/vindr_ds/volume_summary.csv")           # or give full path
output_path = Path("../ncct_cect/vindr_ds/volume_summary_with_z_diff.csv")

df = pd.read_csv(csv_path)

# Helper to extract the z-dimension from the shape string like "191x148x426"
def get_z(shape_str):
    if pd.isna(shape_str) or not shape_str:
        return None
    parts = str(shape_str).split("x")
    if len(parts) != 3:
        return None
    try:
        return int(parts[2])
    except:
        return None

# Extract z-sizes
df["non_contrast_z"] = df["non-contrast shape"].apply(get_z)
df["arterial_z"]      = df["arterial shape"].apply(get_z)
df["venous_z"]        = df["venous shape"].apply(get_z)
df["other_z"]         = df["other shape"].apply(get_z)

# Compute differences compared to non-contrast
df["arterial_z_diff"]   = df["arterial_z"]   - df["non_contrast_z"]
df["venous_z_diff"]     = df["venous_z"]     - df["non_contrast_z"]
df["other_z_diff"]      = df["other_z"]      - df["non_contrast_z"]

# Replace NaN differences with empty string for cleaner look
df["arterial_z_diff"] = df["arterial_z_diff"].fillna("").replace({pd.NA: ""})
df["venous_z_diff"]   = df["venous_z_diff"].fillna("").replace({pd.NA: ""})
df["other_z_diff"]    = df["other_z_diff"].fillna("").replace({pd.NA: ""})

# Reorder columns nicely
new_columns = [
    "case_id",
    "non-contrast shape", "arterial shape", "venous shape", "other shape",
    "non-contrast slice-thickness", "arterial slice-thickness",
    "venous slice-thickness", "other slice-thickness",
    "is_excluded",
    "arterial_z_diff", "venous_z_diff", "other_z_diff"   # ← new columns
]

# Keep only existing columns (in case "other" is sometimes missing)
final_cols = [c for c in new_columns if c in df.columns]
df = df[final_cols]

# Save
df.to_csv(output_path, index=False)

print(f"All done ♡ Saved to {output_path}")
print(f"Total cases: {len(df)}")
print("\nFirst few rows with differences:")
print(df[["case_id", "arterial_z_diff", "venous_z_diff", "other_z_diff"]].head(10))