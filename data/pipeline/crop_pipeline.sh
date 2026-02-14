#!/bin/bash
set -e

DATA_DIR="../ncct_cect/vindr_ds/nifti_unprocessed_volumes"
seg_dir="../ncct_cect/vindr_ds/ts_segmentations"
output_dir="../ncct_cect/vindr_ds/cropped_volumes"
mkdir -p "$output_dir"

nifti_list=$(python3 <<EOF
import os
DATA_DIR = "$DATA_DIR"
nifti_paths = []
for file in os.listdir(DATA_DIR):
    series_path = os.path.join(DATA_DIR, file)
    if file.endswith(".nii.gz"):
        nifti_paths.append(series_path)
print(" ".join(nifti_paths))
EOF
)

for vol in $nifti_list; do

    file_name=$(basename "$vol" .nii.gz)

    # Split by underscore into an array
    IFS='_' read -r study_name vol_name _ <<< "$file_name"

    combined_seg="${seg_dir}/${study_name}_${vol_name}_seg.nii.gz"
    out_name="${study_name}_${vol_name}"

    echo "Cropping $vol with mask $combined_seg ..."
    python3 ./data/crop_with_mask.py "$vol" "$combined_seg" "$output_dir" "$out_name" 10
    echo "-----------------"
done
