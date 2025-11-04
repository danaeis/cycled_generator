import numpy as np
from scipy.ndimage import convolve, gaussian_filter  # For smoothing if needed
from skimage.metrics import structural_similarity as ssim  # Note: skimage not in your env, but assume or implement SSIM manually

# Manual SSIM approximation if skimage unavailable (simple version)
def approx_ssim(img1, img2):
    mu1, mu2 = np.mean(img1), np.mean(img2)
    sigma1, sigma2 = np.std(img1), np.std(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    k1, k2, L = 0.01, 0.03, 255  # Defaults for 8-bit
    c1, c2 = (k1 * L)**2, (k2 * L)**2
    return ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2))

def generate_pseudo_target(source_slice, target_slices, diff_threshold=0.1, kernel_size=3):
    # Step 1: Normalize images to [0,1] for fair comparison
    source_norm = (source_slice - np.min(source_slice)) / (np.max(source_slice) - np.min(source_slice) + 1e-8)
    target_norms = [(t - np.min(t)) / (np.max(t) - np.min(t) + 1e-8) for t in target_slices]
    
    # Step 2: Find best matching target slice via SSIM
    scores = [approx_ssim(source_norm, t_norm) for t_norm in target_norms]
    best_idx = np.argmax(scores)
    matched_target = target_slices[best_idx]
    
    # Step 3: Identify unmatched pixels (where abs diff > threshold * range)
    diff = np.abs(source_slice - matched_target)
    intensity_range = np.max(matched_target) - np.min(matched_target)
    unmatched_mask = diff > (diff_threshold * intensity_range)
    
    # Step 4: Create pseudo-target: copy matched, fill unmatched with mean of surrounding (from matched_target)
    pseudo_target = matched_target.copy()
    h, w = pseudo_target.shape
    mean_kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)  # Mean filter kernel
    
    for y in range(h):
        for x in range(w):
            if unmatched_mask[y, x]:
                # Extract neighborhood, ignoring edges for simplicity (pad if needed)
                y_start, y_end = max(0, y - kernel_size//2), min(h, y + kernel_size//2 + 1)
                x_start, x_end = max(0, x - kernel_size//2), min(w, x + kernel_size//2 + 1)
                neighborhood = pseudo_target[y_start:y_end, x_start:x_end]
                # Only use matched pixels in neighborhood for mean
                matched_neigh = neighborhood[~unmatched_mask[y_start:y_end, x_start:x_end]]
                if len(matched_neigh) > 0:
                    pseudo_target[y, x] = np.mean(matched_neigh)
                else:
                    # Fallback: use global mean if no matched nearby
                    pseudo_target[y, x] = np.mean(matched_target)
    
    # Optional: Light Gaussian smoothing to reduce artifacts
    pseudo_target = gaussian_filter(pseudo_target, sigma=0.5)
    
    return pseudo_target
from PIL import Image
import numpy as np
import nibabel as nib

from configs import MAIN_PATH

def save_as_png(data, filename="pseudo_target.png"):
    # Normalize to [0, 255] for 8-bit PNG
    data_norm = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8) * 255
    data_uint8 = data_norm.astype(np.uint8)
    
    # Save using PIL
    img = Image.fromarray(data_uint8, mode='L')  # 'L' for grayscale
    img.save(filename)
    print(f"Saved pseudo-target to {filename}")
# Assuming pseudo_target is from the previous code (2D numpy array, e.g., shape (256, 256))
def save_as_nifti(data, filename="pseudo_target.nii.gz"):
    # Ensure data is float32, common for medical images
    data = data.astype(np.float32)
    
    # If 2D, reshape to (1, H, W) for NIfTI
    if len(data.shape) == 2:
        data = data[np.newaxis, ...]  # Add singleton dimension
    
    # Create NIfTI image (minimal metadata; adjust affine if needed)
    nifti_img = nib.Nifti1Image(data, affine=np.eye(4))  # Default affine
    nib.save(nifti_img, filename)
    print(f"Saved pseudo-target to {filename}")

# Example usage (dummy data)

source = np.random.randint(0, 100, (256, 256))  # Non-contrast
targets = [np.random.randint(100, 200, (256, 256)) for _ in range(5)]  # Real contrast slices
pseudo = generate_pseudo_target(source, targets)

# Example
save_as_png(pseudo, "pseudo_target.png")


# Example
save_as_nifti(pseudo, "pseudo_target.nii.gz")
# In training: L1 loss = np.mean(np.abs(generated - pseudo))