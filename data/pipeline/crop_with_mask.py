import os
import sys
import nibabel as nib
import numpy as np

if len(sys.argv) < 4:
    print("Usage: python crop_with_mask.py <volume_path> <mask_path> <out_prefix> [margin]")
    sys.exit(1)

vol_path = sys.argv[1]
mask_path = sys.argv[2]
out_dir = sys.argv[3]
out_name = sys.argv[4]
margin = int(sys.argv[5]) if len(sys.argv) > 4 else 10  # default margin = 10 voxels

if os.path.exists(vol_path) and os.path.exists(mask_path):
    pass
else:
    print(f"⚠️ No volume found in path, skipping.")
    sys.exit(0)

# Load images
vol_img = nib.load(vol_path)
mask_img = nib.load(mask_path)

vol_data = vol_img.get_fdata()
mask_data = mask_img.get_fdata()

# Find bounding box of mask
coords = np.argwhere(mask_data > 0)
if coords.size == 0:
    print(f"⚠️ No mask found in {mask_path}, skipping.")
    sys.exit(0)

minz, miny, minx = coords.min(axis=0)
maxz, maxy, maxx = coords.max(axis=0) + 1  # +1 since slice end is exclusive

# Apply margin
minz = max(minz - margin, 0)
miny = max(miny - margin, 0)
minx = max(minx - margin, 0)
maxz = min(maxz + margin, vol_data.shape[0])
maxy = min(maxy + margin, vol_data.shape[1])
maxx = min(maxx + margin, vol_data.shape[2])


# Crop
vol_crop = vol_data[minz:maxz, miny:maxy, minx:maxx]
mask_crop = mask_data[minz:maxz, miny:maxy, minx:maxx]

# Save cropped versions
vol_crop_img = nib.Nifti1Image(vol_crop, vol_img.affine, vol_img.header)
mask_crop_img = nib.Nifti1Image(mask_crop, mask_img.affine, mask_img.header)

vol_out = os.path.join(out_dir, f"{out_name}_crop.nii.gz")
mask_out = os.path.join(out_dir, f"{out_name}_seg.nii.gz")

nib.save(vol_crop_img, vol_out)
nib.save(mask_crop_img, mask_out)

print(f"✅ Saved cropped volume: {vol_out}")
print(f"✅ Saved cropped mask:   {mask_out}")

