#!/bin/bash
set -e

DATA_DIR="../ncct_cect/vindr_ds/nifti_unprocessed_volumes"
output_dir="../ncct_cect/vindr_ds/ts_segmentations"
mkdir -p "$output_dir"

# Get all NIfTI files using Python snippet
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

    seg_dir="${output_dir}/${study_name}_${vol_name}_segs"
    mkdir -p "$seg_dir"

    echo "Running TotalSegmentator on $vol..."
    TotalSegmentator -i "$vol" -o "$seg_dir"

    echo "Combining masks for $vol..."
    python3 data/combine_masks.py "$seg_dir" "${output_dir}/${study_name}_${vol_name}_seg.nii.gz"
done