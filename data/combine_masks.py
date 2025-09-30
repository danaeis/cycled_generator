import nibabel as nib
import numpy as np
import os
import sys

if len(sys.argv) != 3:
    print("Usage: python combine_masks.py <mask_dir> <output_path>")
    sys.exit(1)

mask_path = sys.argv[1]
output_path = sys.argv[2]

# Pick the labels you want to merge
files = {
    1: "liver.nii.gz",
    2: "spleen.nii.gz",
    3: "kidney_left.nii.gz",
    4: "kidney_right.nii.gz",
    5: "stomach.nii.gz",
    6: "pancreas.nii.gz",
    7: "urinary_bladder.nii.gz",
    8: "prostate.nii.gz",
    9: "vertebrae_L1.nii.gz",
    10: "vertebrae_L2.nii.gz",
    11: "vertebrae_L3.nii.gz",
    12: "vertebrae_L4.nii.gz",
    13: "vertebrae_L5.nii.gz",
    14: "vertebrae_S1.nii.gz",
    15: "sacrum.nii.gz"
    # ... add more if needed
}

# Load reference image (first mask available)
first_mask = os.path.join(mask_path, next(iter(files.values())))
ref = nib.load(first_mask)
combined = np.zeros(ref.shape, dtype=np.uint16)

# Merge masks into one label map
for label, fname in files.items():
    mask_file = os.path.join(mask_path, fname)
    if not os.path.exists(mask_file):
        print(f"Warning: {fname} not found, skipping")
        continue
    img = nib.load(mask_file)
    mask = img.get_fdata().astype(bool)
    combined[mask] = label

# Save combined mask
out_img = nib.Nifti1Image(combined, ref.affine, ref.header)
nib.save(out_img, output_path)
print(f"Saved combined mask to {output_path}")
