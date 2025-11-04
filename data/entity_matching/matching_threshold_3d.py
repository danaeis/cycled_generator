import numpy as np
import nibabel as nib
from scipy.ndimage import convolve, gaussian_filter
import torch

# Approximate SSIM for 2D or 3D
def approx_ssim(img1, img2):
    ndim = img1.ndim
    mu1, mu2 = np.mean(img1), np.mean(img2)
    sigma1, sigma2 = np.std(img1), np.std(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    k1, k2, L = 0.01, 0.03, 255  # Defaults for 8-bit
    c1, c2 = (k1 * L)**2, (k2 * L)**2
    return ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2))

# Load NIfTI patch
def load_nifti_patch(filename):
    img = nib.load(filename)
    data = img.get_fdata().astype(np.float32)  # Convert to float32
    return data, img.affine

# Generate pseudo-target for 2D slice
def generate_pseudo_target_2d(source_slice, target_slices, diff_threshold=0.1, kernel_size=3):
    source_norm = (source_slice - np.min(source_slice)) / (np.max(source_slice) - np.min(source_slice) + 1e-8)
    target_norms = [(t - np.min(t)) / (np.max(t) - np.min(t) + 1e-8) for t in target_slices]
    
    scores = [approx_ssim(source_norm, t_norm) for t_norm in target_norms]
    best_idx = np.argmax(scores)
    matched_target = target_slices[best_idx]
    
    diff = np.abs(source_slice - matched_target)
    intensity_range = np.max(matched_target) - np.min(matched_target)
    unmatched_mask = diff > (diff_threshold * intensity_range)
    
    pseudo_target = matched_target.copy()
    h, w = pseudo_target.shape
    mean_kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    
    for y in range(h):
        for x in range(w):
            if unmatched_mask[y, x]:
                y_start, y_end = max(0, y - kernel_size//2), min(h, y + kernel_size//2 + 1)
                x_start, x_end = max(0, x - kernel_size//2), min(w, x + kernel_size//2 + 1)
                neighborhood = pseudo_target[y_start:y_end, x_start:x_end]
                matched_neigh = neighborhood[~unmatched_mask[y_start:y_end, x_start:x_end]]
                if len(matched_neigh) > 0:
                    pseudo_target[y, x] = np.median(matched_neigh)
                else:
                    pseudo_target[y, x] = np.median(matched_target)
    
    pseudo_target = gaussian_filter(pseudo_target, sigma=0.5)
    return pseudo_target

# Generate pseudo-target for 3D patch
def generate_pseudo_target_3d(source_patch, target_patches, diff_threshold=0.1, kernel_size=3):
    source_norm = (source_patch - np.min(source_patch)) / (np.max(source_patch) - np.min(source_patch) + 1e-8)
    target_norms = [(p - np.min(p)) / (np.max(p) - np.min(p) + 1e-8) for p in target_patches]
    
    scores = [approx_ssim(source_norm, t_norm) for t_norm in target_norms]
    best_idx = np.argmax(scores)
    matched_target = target_patches[best_idx]
    
    diff = np.abs(source_patch - matched_target)
    intensity_range = np.max(matched_target) - np.min(matched_target)
    unmatched_mask = diff > (diff_threshold * intensity_range)
    
    pseudo_target = matched_target.copy()
    d, h, w = pseudo_target.shape
    mean_kernel = np.ones((kernel_size, kernel_size, kernel_size)) / (kernel_size ** 3)
    
    for z in range(d):
        for y in range(h):
            for x in range(w):
                if unmatched_mask[z, y, x]:
                    z_start, z_end = max(0, z - kernel_size//2), min(d, z + kernel_size//2 + 1)
                    y_start, y_end = max(0, y - kernel_size//2), min(h, y + kernel_size//2 + 1)
                    x_start, x_end = max(0, x - kernel_size//2), min(w, x + kernel_size//2 + 1)
                    neighborhood = pseudo_target[z_start:z_end, y_start:y_end, x_start:x_end]
                    matched_neigh = neighborhood[~unmatched_mask[z_start:z_end, y_start:y_end, x_start:x_end]]
                    if len(matched_neigh) > 0:
                        pseudo_target[z, y, x] = np.median(matched_neigh)
                    else:
                        pseudo_target[z, y, x] = np.median(matched_target)
    
    pseudo_target = gaussian_filter(pseudo_target, sigma=0.5)
    return pseudo_target

# Save as NIfTI
def save_as_nifti(data, affine, filename):
    nifti_img = nib.Nifti1Image(data, affine=affine)
    nib.save(nifti_img, filename)
    print(f"Saved to {filename}")

# Pass patches to model (example with PyTorch)
def pass_to_model(source_patch, pseudo_target, model=None):
    source_tensor = torch.from_numpy(source_patch).float().unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, D, H, W)
    pseudo_tensor = torch.from_numpy(pseudo_target).float().unsqueeze(0).unsqueeze(0)
    
    if model is not None:
        model.eval()
        with torch.no_grad():
            generated = model(source_tensor)
            l1_loss = torch.nn.L1Loss()(generated, pseudo_tensor)
            print(f"L1 Loss: {l1_loss.item()}")
    
    return source_tensor, pseudo_tensor

# Main function to process patches and create pseudo-target
def process_patches(source_nii_path, target_nii_paths, mode='3d', diff_threshold=0.2, kernel_size=3, output_nii_path="pseudo_target_patch.nii.gz"):
    """
    Process 3D patches from .nii.gz files and create pseudo-target.
    
    Args:
    - source_nii_path: Path to source patch .nii.gz
    - target_nii_paths: List of paths to target patch .nii.gz files
    - mode: '2d' (slice-by-slice) or '3d' (whole patch)
    - diff_threshold: Threshold for unmatched voxels
    - kernel_size: Size of mean kernel
    - output_nii_path: Path to save pseudo-target .nii.gz
    
    Returns:
    - source_patch: 3D numpy array
    - pseudo_target: 3D numpy array
    - affine: Affine matrix from source
    """
    # Load source patch
    source_patch, affine = load_nifti_patch(source_nii_path)
    
    # Load target patches
    target_patches = [load_nifti_patch(path)[0] for path in target_nii_paths]
    
    # Generate pseudo-target
    if mode == '3d':
        pseudo_target = generate_pseudo_target_3d(source_patch, target_patches, diff_threshold, kernel_size)
    
    elif mode == '2d':
        d, h, w = source_patch.shape
        pseudo_slices = []
        for z in range(d):
            source_slice = source_patch[z]
            target_slices = [p[z] for p in target_patches]
            pseudo_slice = generate_pseudo_target_2d(source_slice, target_slices, diff_threshold, kernel_size)
            pseudo_slices.append(pseudo_slice)
        pseudo_target = np.stack(pseudo_slices, axis=0)
    
    else:
        raise ValueError("Mode must be '2d' or '3d'")
    
    # Save pseudo-target as .nii.gz
    save_as_nifti(pseudo_target, affine, output_nii_path)
    
    return source_patch, pseudo_target, affine

# Example usage
source_nii_path = "source.nii.gz"
target_nii_paths = ["target_groundtruth.nii.gz"]
output_nii_path_3d = "pseudo_target_3d.nii.gz"
output_nii_path_2d = "pseudo_target_2d.nii.gz"

# Process in 3D mode
source_patch, pseudo_target_3d, affine = process_patches(source_nii_path, target_nii_paths, mode='3d', output_nii_path=output_nii_path_3d)
print("3D Pseudo-target shape:", pseudo_target_3d.shape)

# Process in 2D mode
source_patch, pseudo_target_2d, affine = process_patches(source_nii_path, target_nii_paths, mode='2d', output_nii_path=output_nii_path_2d)
print("2D Pseudo-target shape:", pseudo_target_2d.shape)

# Pass to model (example, assuming model exists)
# model = YourModel()  # Define your model
# source_tensor, pseudo_tensor = pass_to_model(source_patch, pseudo_target_3d, model)