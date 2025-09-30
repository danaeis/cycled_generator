# """
# Clean CT Phase Generation Training Pipeline
# ============================================
# A well-organized, debuggable implementation for generating different contrast phases of CT images.

# Architecture:
# - Phase-conditional CycleGAN
# - Input phase -> Generate target phase (conditional)
# - Generated target -> Regenerate input phase (cycle consistency)
# - Discriminator on generated target
# - Focal loss on organ masks
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import numpy as np
# import nibabel as nib
# from pathlib import Path
# from tqdm import tqdm
# import json
# from typing import Dict, List, Tuple, Optional
# import logging

# # Setup logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)


# # ============================================================================
# # DATASET
# # ============================================================================

# class CTPhaseDataset(Dataset):
#     """
#     Dataset for CT phase generation with paired patches and organ masks.
    
#     Creates overlapping 3D patches from registered CT volumes with different phases.
#     Includes organ segmentation masks for focal loss.
#     """
    
#     def __init__(
#         self,
#         data_pairs: List[Dict],
#         patch_size: Tuple[int, int] = (64, 64),
#         patch_depth: int = 7,
#         overlap_ratio: float = 0.5,
#         augment: bool = True
#     ):
#         self.data_pairs = data_pairs
#         self.patch_size = patch_size
#         self.patch_depth = patch_depth
#         self.overlap_ratio = overlap_ratio
#         self.augment = augment
#         self.patch_coords = []
        
#         # Phase mapping
#         self.phase_to_idx = {
#             'non-contrast': 0,
#             'arterial': 1,
#             'portal': 2,
#             'venous': 2,  # Map venous to portal
#             'delayed': 3
#         }
        
#         logger.info(f"Initializing dataset with {len(data_pairs)} pairs")
#         logger.info(f"Patch size: {patch_size}, Depth: {patch_depth}, Overlap: {overlap_ratio}")
        
#         self._compute_patch_coordinates()
#         logger.info(f"Generated {len(self.patch_coords)} total patches")
    
#     def _compute_patch_coordinates(self):
#         """Pre-compute all valid patch coordinates from volumes."""
#         padding = self.patch_depth // 2
        
#         for pair_idx, pair_data in enumerate(self.data_pairs):
#             try:
#                 # Load volumes to get dimensions
#                 source_vol = nib.load(pair_data['source_path']).get_fdata()
#                 target_vol = nib.load(pair_data['target_path']).get_fdata()
                
#                 # Validate shapes match
#                 if source_vol.shape != target_vol.shape:
#                     logger.warning(f"Shape mismatch for pair {pair_idx}, skipping")
#                     continue
                
#                 depth, height, width = source_vol.shape
                
#                 # Check minimum requirements
#                 if depth < self.patch_depth + 2:
#                     logger.warning(f"Insufficient depth ({depth}) for pair {pair_idx}, skipping")
#                     continue
                
#                 if height < self.patch_size[0] or width < self.patch_size[1]:
#                     logger.warning(f"Insufficient spatial size for pair {pair_idx}, skipping")
#                     continue
                
#                 # Calculate step sizes for overlap
#                 step_y = max(1, int(self.patch_size[0] * (1 - self.overlap_ratio)))
#                 step_x = max(1, int(self.patch_size[1] * (1 - self.overlap_ratio)))
                
#                 # Generate coordinate ranges
#                 z_range = range(padding, depth - padding)
#                 y_range = range(0, height - self.patch_size[0] + 1, step_y)
#                 x_range = range(0, width - self.patch_size[1] + 1, step_x)
                
#                 # Store all valid patch coordinates
#                 for center_z in z_range:
#                     for y_start in y_range:
#                         for x_start in x_range:
#                             self.patch_coords.append((pair_idx, center_z, y_start, x_start))
                
#                 logger.info(f"Pair {pair_idx}: Generated {len(z_range)*len(y_range)*len(x_range)} patches")
                
#             except Exception as e:
#                 logger.error(f"Error processing pair {pair_idx}: {e}")
#                 continue
    
#     def _normalize_intensity(self, image: np.ndarray) -> np.ndarray:
#         """Normalize CT intensity values."""
#         # Clip to abdomen HU range
#         image = np.clip(image, -100, 300)
        
#         # Z-score normalization
#         mean_val = np.mean(image)
#         std_val = np.std(image)
#         if std_val > 1e-6:
#             image = (image - mean_val) / std_val
        
#         return image.astype(np.float32)
    
#     def _extract_organ_masks(
#         self, 
#         seg_volume: np.ndarray, 
#         z_start: int, 
#         z_end: int,
#         y_start: int, 
#         y_end: int, 
#         x_start: int, 
#         x_end: int
#     ) -> Dict[str, np.ndarray]:
#         """Extract organ masks from segmentation volume."""
#         patch_seg = seg_volume[z_start:z_end, y_start:y_end, x_start:x_end]
        
#         # Define organ labels (adjust based on your segmentation)
#         organ_labels = {
#             'liver': [1, 2],
#             'kidney_right': [3],
#             'kidney_left': [4],
#             'spleen': [5],
#         }
        
#         masks = {}
#         for organ, labels in organ_labels.items():
#             mask = np.zeros_like(patch_seg, dtype=np.float32)
#             for label in labels:
#                 mask[patch_seg == label] = 1.0
#             masks[organ] = mask
        
#         return masks
    
#     def _augment(self, source, target, masks):
#         """Apply 3D augmentations."""
#         # Random horizontal flip
#         if np.random.random() > 0.5:
#             source = np.flip(source, axis=2).copy()
#             target = np.flip(target, axis=2).copy()
#             masks = {k: np.flip(v, axis=2).copy() for k, v in masks.items()}
        
#         # Random vertical flip
#         if np.random.random() > 0.5:
#             source = np.flip(source, axis=1).copy()
#             target = np.flip(target, axis=1).copy()
#             masks = {k: np.flip(v, axis=1).copy() for k, v in masks.items()}
        
#         return source, target, masks
    
#     def __len__(self) -> int:
#         return len(self.patch_coords)
    
#     def __getitem__(self, idx: int) -> Dict:
#         pair_idx, center_z, y_start, x_start = self.patch_coords[idx]
#         pair_data = self.data_pairs[pair_idx]
        
#         try:
#             # Load volumes
#             source_vol = nib.load(pair_data['source_path']).get_fdata()
#             target_vol = nib.load(pair_data['target_path']).get_fdata()
            
#             # Normalize
#             source_vol = self._normalize_intensity(source_vol)
#             target_vol = self._normalize_intensity(target_vol)
            
#             # Extract patch
#             padding = self.patch_depth // 2
#             z_start = center_z - padding
#             z_end = center_z + padding + 1
#             y_end = y_start + self.patch_size[0]
#             x_end = x_start + self.patch_size[1]
            
#             source_patch = source_vol[z_start:z_end, y_start:y_end, x_start:x_end]
#             target_patch = target_vol[z_start:z_end, y_start:y_end, x_start:x_end]
            
#             # Extract organ masks if available
#             masks = {}
#             if pair_data.get('target_seg'):
#                 try:
#                     seg_vol = nib.load(pair_data['target_seg']).get_fdata()
#                     masks = self._extract_organ_masks(seg_vol, z_start, z_end, y_start, y_end, x_start, x_end)
#                 except Exception as e:
#                     logger.debug(f"Could not load masks: {e}")
            
#             # Augmentation
#             if self.augment:
#                 source_patch, target_patch, masks = self._augment(source_patch, target_patch, masks)
            
#             # Convert to tensors
#             source_tensor = torch.from_numpy(source_patch).unsqueeze(0).float()  # [1, D, H, W]
#             target_tensor = torch.from_numpy(target_patch).unsqueeze(0).float()
            
#             mask_tensors = {
#                 organ: torch.from_numpy(mask).unsqueeze(0).float() 
#                 for organ, mask in masks.items()
#             }
            
#             # Get phase indices
#             source_phase = pair_data['source_phase']
#             target_phase = pair_data['target_phase']
            
#             return {
#                 'source': source_tensor,
#                 'target': target_tensor,
#                 'source_phase': source_phase,
#                 'target_phase': target_phase,
#                 'source_phase_idx': self.phase_to_idx.get(source_phase, 0),
#                 'target_phase_idx': self.phase_to_idx.get(target_phase, 1),
#                 'masks': mask_tensors,
#                 'case_id': pair_data['case_id']
#             }
            
#         except Exception as e:
#             logger.error(f"Error loading patch {idx}: {e}")
#             # Return dummy data
#             return {
#                 'source': torch.zeros(1, self.patch_depth, *self.patch_size),
#                 'target': torch.zeros(1, self.patch_depth, *self.patch_size),
#                 'source_phase': 'error',
#                 'target_phase': 'error',
#                 'source_phase_idx': 0,
#                 'target_phase_idx': 1,
#                 'masks': {},
#                 'case_id': 'error'
#             }


# # ============================================================================
# # MODEL ARCHITECTURE
# # ============================================================================

# class PhaseConditionedGenerator(nn.Module):
#     """
#     3D U-Net Generator with phase conditioning.
#     Takes input phase and generates target phase conditioned on phase label.
#     """
    
#     def __init__(self, num_phases: int = 4):
#         super().__init__()
        
#         self.num_phases = num_phases
        
#         # Phase embedding
#         self.phase_embedding = nn.Embedding(num_phases, 64)
        
#         # Encoder
#         self.enc1 = self._make_encoder_block(1, 64)      # Input + phase info
#         self.enc2 = self._make_encoder_block(64, 128)
#         self.enc3 = self._make_encoder_block(128, 256)
#         self.enc4 = self._make_encoder_block(256, 512)
        
#         self.pool = nn.MaxPool3d((1, 2, 2))  # Pool spatial dims only
        
#         # Bottleneck with phase modulation
#         self.bottleneck = nn.Sequential(
#             nn.Conv3d(512, 1024, 3, padding=1),
#             nn.InstanceNorm3d(1024),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv3d(1024, 512, 3, padding=1),
#             nn.InstanceNorm3d(512),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
        
#         # Phase-adaptive modulation
#         self.phase_scale_shift = nn.Linear(64, 512 * 2)
        
#         # Decoder
#         self.up4 = nn.ConvTranspose3d(512, 256, (1, 2, 2), stride=(1, 2, 2))
#         self.dec4 = self._make_decoder_block(512 + 256, 256)
        
#         self.up3 = nn.ConvTranspose3d(256, 128, (1, 2, 2), stride=(1, 2, 2))
#         self.dec3 = self._make_decoder_block(256 + 128, 128)
        
#         self.up2 = nn.ConvTranspose3d(128, 64, (1, 2, 2), stride=(1, 2, 2))
#         self.dec2 = self._make_decoder_block(128 + 64, 64)
        
#         self.up1 = nn.ConvTranspose3d(64, 64, (1, 2, 2), stride=(1, 2, 2))
#         self.dec1 = self._make_decoder_block(64 + 64, 64)
        
#         # Output
#         self.output = nn.Sequential(
#             nn.Conv3d(64, 1, 1),
#             nn.Tanh()
#         )
    
#     def _make_encoder_block(self, in_ch, out_ch):
#         return nn.Sequential(
#             nn.Conv3d(in_ch, out_ch, 3, padding=1),
#             nn.InstanceNorm3d(out_ch),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv3d(out_ch, out_ch, 3, padding=1),
#             nn.InstanceNorm3d(out_ch),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
    
#     def _make_decoder_block(self, in_ch, out_ch):
#         return nn.Sequential(
#             nn.Conv3d(in_ch, out_ch, 3, padding=1),
#             nn.InstanceNorm3d(out_ch),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
    
#     def forward(self, x: torch.Tensor, phase_idx: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x: Input image [B, 1, D, H, W]
#             phase_idx: Target phase indices [B]
#         """
#         batch_size = x.size(0)
        
#         # Get phase embedding [B, 64]
#         phase_emb = self.phase_embedding(phase_idx)
        
#         # Encoder
#         e1 = self.enc1(x)
#         e2 = self.enc2(self.pool(e1))
#         e3 = self.enc3(self.pool(e2))
#         e4 = self.enc4(self.pool(e3))
        
#         # Bottleneck with phase modulation
#         b = self.bottleneck(self.pool(e4))
        
#         # Apply phase-specific modulation
#         phase_params = self.phase_scale_shift(phase_emb)
#         scale = phase_params[:, :512].view(batch_size, 512, 1, 1, 1)
#         shift = phase_params[:, 512:].view(batch_size, 512, 1, 1, 1)
#         b = b * (1 + scale) + shift
        
#         # Decoder with skip connections
#         d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
#         d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
#         d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
#         d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
#         return self.output(d1)


# class Discriminator3D(nn.Module):
#     """3D PatchGAN Discriminator."""
    
#     def __init__(self):
#         super().__init__()
        
#         self.model = nn.Sequential(
#             # Layer 1
#             nn.Conv3d(1, 64, (3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
#             nn.LeakyReLU(0.2, inplace=True),
            
#             # Layer 2
#             nn.Conv3d(64, 128, (3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
#             nn.InstanceNorm3d(128),
#             nn.LeakyReLU(0.2, inplace=True),
            
#             # Layer 3
#             nn.Conv3d(128, 256, (3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
#             nn.InstanceNorm3d(256),
#             nn.LeakyReLU(0.2, inplace=True),
            
#             # Output
#             nn.Conv3d(256, 1, (2, 8, 8), stride=1, padding=0)
#         )
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.model(x)

# class MetricsCalculator:
#     """Calculate image quality metrics (SSIM, PSNR)."""
    
#     @staticmethod
#     def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 2.0) -> float:
#         """
#         Calculate Peak Signal-to-Noise Ratio.
        
#         Args:
#             pred: Predicted image [-1, 1] range
#             target: Target image [-1, 1] range
#             data_range: Range of the data (2.0 for [-1, 1])
#         """
#         mse = torch.mean((pred - target) ** 2)
#         if mse == 0:
#             return 100.0
#         psnr = 20 * torch.log10(data_range / torch.sqrt(mse))
#         return psnr.item()
    
#     @staticmethod
#     def calculate_ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 2.0) -> float:
#         """
#         Calculate Structural Similarity Index.
        
#         Args:
#             pred: Predicted image [-1, 1] range [B, C, D, H, W]
#             target: Target image [-1, 1] range [B, C, D, H, W]
#             data_range: Range of the data
#         """
#         # Take middle slice for 2D SSIM calculation
#         if pred.dim() == 5:  # [B, C, D, H, W]
#             pred = pred[:, :, pred.shape[2]//2, :, :]  # [B, C, H, W]
#             target = target[:, :, target.shape[2]//2, :, :]
        
#         # Constants for SSIM
#         C1 = (0.01 * data_range) ** 2
#         C2 = (0.03 * data_range) ** 2
        
#         # Calculate means
#         mu1 = F.avg_pool2d(pred, 3, 1, 1)
#         mu2 = F.avg_pool2d(target, 3, 1, 1)
        
#         mu1_sq = mu1 ** 2
#         mu2_sq = mu2 ** 2
#         mu1_mu2 = mu1 * mu2
        
#         # Calculate variances and covariance
#         sigma1_sq = F.avg_pool2d(pred ** 2, 3, 1, 1) - mu1_sq
#         sigma2_sq = F.avg_pool2d(target ** 2, 3, 1, 1) - mu2_sq
#         sigma12 = F.avg_pool2d(pred * target, 3, 1, 1) - mu1_mu2
        
#         # SSIM formula
#         ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
#                    ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
#         return ssim_map.mean().item()

# import matplotlib.pyplot as plt

# def save_sample_patches(
#     generator: nn.Module,
#     val_loader: DataLoader,
#     epoch: int,
#     save_dir: Path,
#     device: torch.device,
#     num_samples: int = 5
# ):
#     """
#     Save sample generated patches for visual inspection.
    
#     Args:
#         generator: The generator model
#         val_loader: Validation dataloader
#         epoch: Current epoch number
#         save_dir: Directory to save samples
#         device: Device to run on
#         num_samples: Number of samples to save
#     """
#     generator.eval()
#     save_dir = Path(save_dir) / f"epoch_{epoch}"
#     save_dir.mkdir(parents=True, exist_ok=True)
    
#     saved_count = 0
    
#     with torch.no_grad():
#         for batch in val_loader:
#             if saved_count >= num_samples:
#                 break
            
#             try:
#                 real_source = batch['source'].to(device)
#                 real_target = batch['target'].to(device)
#                 source_phase_idx = batch['source_phase_idx'].to(device)
#                 target_phase_idx = batch['target_phase_idx'].to(device)
#                 source_phase = batch['source_phase']
#                 target_phase = batch['target_phase']
#                 case_id = batch['case_id']
                
#                 # Generate target and reconstruct source
#                 generated_target = generator(real_source, target_phase_idx)
#                 reconstructed_source = generator(generated_target, source_phase_idx)
                
#                 # Process each sample in batch
#                 batch_size = real_source.size(0)
#                 for i in range(min(batch_size, num_samples - saved_count)):
#                     # Get middle slice from 3D patch [1, D, H, W] -> [H, W]
#                     mid_slice = real_source.shape[2] // 2
                    
#                     source_slice = real_source[i, 0, mid_slice].cpu().numpy()
#                     target_slice = real_target[i, 0, mid_slice].cpu().numpy()
#                     generated_slice = generated_target[i, 0, mid_slice].cpu().numpy()
#                     reconstructed_slice = reconstructed_source[i, 0, mid_slice].cpu().numpy()
                    
#                     # Create comparison figure
#                     fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                    
#                     # Row 1: Forward generation (source -> target)
#                     im0 = axes[0, 0].imshow(source_slice, cmap='gray', vmin=-1, vmax=1)
#                     axes[0, 0].set_title(f'Source: {source_phase[i]}', fontsize=12, fontweight='bold')
#                     axes[0, 0].axis('off')
#                     plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
                    
#                     im1 = axes[0, 1].imshow(generated_slice, cmap='gray', vmin=-1, vmax=1)
#                     axes[0, 1].set_title(f'Generated: {target_phase[i]}', fontsize=12, fontweight='bold')
#                     axes[0, 1].axis('off')
#                     plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
                    
#                     im2 = axes[0, 2].imshow(target_slice, cmap='gray', vmin=-1, vmax=1)
#                     axes[0, 2].set_title(f'Ground Truth: {target_phase[i]}', fontsize=12, fontweight='bold')
#                     axes[0, 2].axis('off')
#                     plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
                    
#                     # Row 2: Cycle reconstruction and difference maps
#                     im3 = axes[1, 0].imshow(reconstructed_slice, cmap='gray', vmin=-1, vmax=1)
#                     axes[1, 0].set_title(f'Reconstructed: {source_phase[i]}', fontsize=12, fontweight='bold')
#                     axes[1, 0].axis('off')
#                     plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
                    
#                     # Difference map: generated vs ground truth
#                     diff_gen = np.abs(generated_slice - target_slice)
#                     im4 = axes[1, 1].imshow(diff_gen, cmap='hot', vmin=0, vmax=1)
#                     axes[1, 1].set_title('|Generated - GT|', fontsize=12, fontweight='bold')
#                     axes[1, 1].axis('off')
#                     plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
                    
#                     # Difference map: reconstructed vs source
#                     diff_cycle = np.abs(reconstructed_slice - source_slice)
#                     im5 = axes[1, 2].imshow(diff_cycle, cmap='hot', vmin=0, vmax=1)
#                     axes[1, 2].set_title('|Reconstructed - Source|', fontsize=12, fontweight='bold')
#                     axes[1, 2].axis('off')
#                     plt.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)
                    
#                     # Calculate metrics for this sample
#                     mse_gen = np.mean((generated_slice - target_slice) ** 2)
#                     mse_cycle = np.mean((reconstructed_slice - source_slice) ** 2)
                    
#                     # Add overall title with metrics
#                     fig.suptitle(
#                         f'Epoch {epoch} - Case: {case_id[i]} - '
#                         f'MSE (Gen): {mse_gen:.4f}, MSE (Cycle): {mse_cycle:.4f}',
#                         fontsize=14, fontweight='bold', y=0.98
#                     )
                    
#                     plt.tight_layout()
                    
#                     # Save figure
#                     save_path = save_dir / f'sample_{saved_count:03d}_{case_id[i]}.png'
#                     plt.savefig(save_path, dpi=150, bbox_inches='tight')
#                     plt.close()
                    
#                     saved_count += 1
                    
#                     if saved_count >= num_samples:
#                         break
                        
#             except Exception as e:
#                 logger.error(f"Error saving sample patches: {e}")
#                 continue
    
#     logger.info(f"Saved {saved_count} sample patches to {save_dir}")



# class FocalLoss(nn.Module):
#     """Focal loss for organ-specific enhancement."""
    
#     def __init__(self, alpha: float = 1.0, gamma: float = 1.5):  # REDUCED defaults
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma
    
#     def forward(
#         self, 
#         pred: torch.Tensor, 
#         target: torch.Tensor, 
#         masks: Dict[str, torch.Tensor]
#     ) -> torch.Tensor:
#         """
#         Args:
#             pred: Predicted image [B, 1, D, H, W]
#             target: Target image [B, 1, D, H, W]
#             masks: Organ masks {organ_name: [B, 1, D, H, W]}
#         """
#         # Base MSE loss
#         mse = F.mse_loss(pred, target, reduction='none')
        
#         # Create weighted mask emphasizing organs
#         weight_mask = torch.ones_like(mse)
#         for organ_name, mask in masks.items():
#             if mask.sum() > 0:
#                 # Higher weight for important organs
#                 organ_weight = 3.0 if organ_name == 'liver' else 2.0
#                 weight_mask += (organ_weight - 1.0) * mask
        
#         # Apply focal weighting
#         focal_weight = mse ** self.gamma
#         weighted_loss = self.alpha * focal_weight * mse * weight_mask
        
#         return weighted_loss.mean()


# class CombinedLoss(nn.Module):
#     """Combined loss for CycleGAN training."""
    
#     def __init__(
#         self,
#         lambda_cycle: float = 10.0,
#         lambda_mse: float = 100.0,
#         lambda_focal: float = 5.0,  # REDUCED from 50.0 to 5.0
#         lambda_adv: float = 1.0,  # Adversarial loss weight
#         adv_warmup_epochs: int = 10  # Number of epochs to gradually increase adversarial loss
  
#     ):
#         super().__init__()
#         self.lambda_cycle = lambda_cycle
#         self.lambda_mse = lambda_mse
#         self.lambda_focal = lambda_focal
#         self.lambda_adv = lambda_adv
#         self.adv_warmup_epochs = adv_warmup_epochs
        
#         self.mse_loss = nn.MSELoss()
#         self.focal_loss = FocalLoss(alpha=1.0, gamma=1.5)
#         self.adv_loss = nn.BCEWithLogitsLoss()
        
#         self.current_epoch = 0
    
#     def forward(
#         self,
#         real_source: torch.Tensor,
#         real_target: torch.Tensor,
#         generated_target: torch.Tensor,
#         reconstructed_source: torch.Tensor,
#         disc_fake: torch.Tensor,
#         masks: Dict[str, torch.Tensor]
#     ) -> Dict[str, torch.Tensor]:
#         """Calculate all generator losses."""
        
#         losses = {}
        
#         # Adversarial loss (fool discriminator)
#         losses['adv'] = self.adv_loss(disc_fake, torch.ones_like(disc_fake))
        
#         # Cycle consistency loss
#         losses['cycle'] = self.mse_loss(reconstructed_source, real_source) * self.lambda_cycle
        
#         # Direct MSE loss
#         losses['mse'] = self.mse_loss(generated_target, real_target) * self.lambda_mse
        
#         # Focal loss with organ masks
#         if masks:
#             losses['focal'] = self.focal_loss(generated_target, real_target, masks) * self.lambda_focal
#         else:
#             losses['focal'] = torch.tensor(0.0, device=real_source.device)
        
#         # Total generator loss
#         losses['total'] = losses['adv'] + losses['cycle'] + losses['mse'] + losses['focal']
        
#         return losses


# # ============================================================================
# # TRAINER
# # ============================================================================

# class CTPhaseTrainer:
#     """Clean trainer for CT phase generation."""
    
#     def __init__(self, config: Dict):
#         self.config = config
#         self.device = torch.device(config['device'])
#         self.output_dir = Path(config['output_dir'])
#         self.output_dir.mkdir(parents=True, exist_ok=True)

#         # Create samples directory
#         self.samples_dir = self.output_dir / 'samples'
#         self.samples_dir.mkdir(exist_ok=True)

#         # Initialize models
#         logger.info("Initializing models...")
#         self.generator = PhaseConditionedGenerator(num_phases=4).to(self.device)
#         self.discriminator = Discriminator3D().to(self.device)
        
#         # Optimizers
#         self.opt_gen = optim.Adam(
#             self.generator.parameters(),
#             lr=config['learning_rate'],
#             betas=(0.5, 0.999)
#         )
#         self.opt_disc = optim.Adam(
#             self.discriminator.parameters(),
#             lr=config['learning_rate'] * config.get('disc_lr_multiplier', 1.0),  # 2x learning rate
#             betas=(0.5, 0.999)
#         )
        
#         # Loss functions
#         self.combined_loss = CombinedLoss().to(self.device)
#         self.disc_loss = nn.BCEWithLogitsLoss()
        
#         # Training state
#         self.current_epoch = 0
#         self.best_val_loss = float('inf')
        
#         # Save config
#         with open(self.output_dir / 'config.json', 'w') as f:
#             json.dump(config, f, indent=2)
        
#         logger.info(f"Trainer initialized on {self.device}")
    
#     def train_step(self, batch: Dict) -> Dict[str, float]:
#         """Single training step."""
        
#         # Get data
#         real_source = batch['source'].to(self.device)
#         real_target = batch['target'].to(self.device)
#         source_phase_idx = batch['source_phase_idx'].to(self.device)
#         target_phase_idx = batch['target_phase_idx'].to(self.device)
#         masks = {k: v.to(self.device) for k, v in batch['masks'].items()}
        
#         # =====================
#         # Train Generator
#         # =====================
#         self.opt_gen.zero_grad()
        
#         # Forward pass
#         # Source -> Target (conditioned)
#         generated_target = self.generator(real_source, target_phase_idx)
        
#         # Generated Target -> Reconstructed Source (cycle)
#         reconstructed_source = self.generator(generated_target, source_phase_idx)
        
#         # Discriminator output for generated target
#         disc_fake = self.discriminator(generated_target)
        
#         # Calculate generator losses
#         gen_losses = self.combined_loss(
#             real_source, real_target, generated_target, 
#             reconstructed_source, disc_fake, masks
#         )
        
#         # Backward
#         gen_losses['total'].backward()
#         self.opt_gen.step()
        
#         # =====================
#         # Train Discriminator
#         # =====================
#         self.opt_disc.zero_grad()
        
#         # Real samples
#         disc_real = self.discriminator(real_target)
#         loss_real = self.disc_loss(disc_real, torch.ones_like(disc_real))
        
#         # Fake samples (detached)
#         disc_fake_detached = self.discriminator(generated_target.detach())
#         loss_fake = self.disc_loss(disc_fake_detached, torch.zeros_like(disc_fake_detached))
        
#         # Total discriminator loss
#         disc_loss = (loss_real + loss_fake) * 0.5
        
#         # Backward
#         disc_loss.backward()
#         self.opt_disc.step()
        
#         # Return losses
#         return {
#             'gen_total': gen_losses['total'].item(),
#             'gen_adv': gen_losses['adv'].item(),
#             'gen_cycle': gen_losses['cycle'].item(),
#             'gen_mse': gen_losses['mse'].item(),
#             'gen_focal': gen_losses['focal'].item(),
#             'disc': disc_loss.item()
#         }
    
#     def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
#         """Train for one epoch."""
#         self.generator.train()
#         self.discriminator.train()
        
#         epoch_losses = {
#             'gen_total': [], 'gen_adv': [], 'gen_cycle': [],
#             'gen_mse': [], 'gen_focal': [], 'disc': []
#         }
        
#         progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
#         for batch in progress_bar:
#             try:
#                 losses = self.train_step(batch)
                
#                 # Accumulate losses
#                 for key, value in losses.items():
#                     epoch_losses[key].append(value)
                
#                 # Update progress bar
#                 progress_bar.set_postfix({
#                     'Gen': f"{losses['gen_total']:.3f}",
#                     'Disc': f"{losses['disc']:.3f}",
#                     'Phase': f"{batch['source_phase'][0]}->{batch['target_phase'][0]}"
#                 })
                
#             except Exception as e:
#                 logger.error(f"Error in training step: {e}")
#                 continue
        
#         # Average losses
#         return {k: np.mean(v) if v else 0.0 for k, v in epoch_losses.items()}
    
#     @torch.no_grad()
#     def validate(self, val_loader: DataLoader) -> float:
#         """Validate the model."""
#         self.generator.eval()
#         val_losses = []
#         psnr_scores = []
#         ssim_scores = []
        
#         metrics_calc = MetricsCalculator()
        
#         for batch in tqdm(val_loader, desc="Validation"):
#             try:
#                 real_source = batch['source'].to(self.device)
#                 real_target = batch['target'].to(self.device)
#                 target_phase_idx = batch['target_phase_idx'].to(self.device)
                
#                 # Generate target
#                 generated_target = self.generator(real_source, target_phase_idx)
                
#                 # Calculate MSE loss
#                 val_loss = F.mse_loss(generated_target, real_target)
#                 val_losses.append(val_loss.item())
                
#                 # Calculate PSNR
#                 psnr = metrics_calc.calculate_psnr(generated_target, real_target)
#                 psnr_scores.append(psnr)
                
#                 # Calculate SSIM
#                 ssim = metrics_calc.calculate_ssim(generated_target, real_target)
#                 ssim_scores.append(ssim)
                
#             except Exception as e:
#                 logger.error(f"Error in validation: {e}")
#                 continue
#         return {
#             'val_loss': np.mean(val_losses) if val_losses else float('inf'),
#             'psnr': np.mean(psnr_scores) if psnr_scores else 0.0,
#             'ssim': np.mean(ssim_scores) if ssim_scores else 0.0
#         }
    
#     def save_checkpoint(self, val_loss: float, is_best: bool = False):
#         """Save model checkpoint."""
#         checkpoint = {
#             'epoch': self.current_epoch,
#             'generator_state': self.generator.state_dict(),
#             'discriminator_state': self.discriminator.state_dict(),
#             'opt_gen_state': self.opt_gen.state_dict(),
#             'opt_disc_state': self.opt_disc.state_dict(),
#             'val_loss': val_loss,
#             'config': self.config
#         }
        
#         # Save regular checkpoint
#         path = self.output_dir / f'checkpoint_epoch_{self.current_epoch}.pth'
#         torch.save(checkpoint, path)
        
#         # Save best model
#         if is_best:
#             best_path = self.output_dir / 'best_model.pth'
#             torch.save(checkpoint, best_path)
#             logger.info(f"  Val - Loss: {val_metrics['val_loss']:.4f}, "
#                        f"PSNR: {val_metrics['psnr']:.2f} dB, "
#                        f"SSIM: {val_metrics['ssim']:.4f}")
    
#     def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
#         """Main training loop."""
#         logger.info(f"Starting training for {epochs} epochs")
#         logger.info(f"Training batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
#         # How often to save samples (default: every epoch)
#         save_samples_interval = self.config.get('save_samples_interval', 1)
#         num_samples = self.config.get('num_samples_to_save', 5)

#         for epoch in range(epochs):
#             self.current_epoch = epoch
            
#             # Train
#             train_losses = self.train_epoch(train_loader)
            
#             # Validate
#             val_metrics = self.validate(val_loader)
            
#             # Log
#             logger.info(f"\nEpoch {epoch + 1}/{epochs} Summary:")
#             logger.info(f"  Train - Total: {train_losses['gen_total']:.4f}, "
#                        f"Cycle: {train_losses['gen_cycle']:.4f}, "
#                        f"MSE: {train_losses['gen_mse']:.4f}, "
#                        f"Focal: {train_losses['gen_focal']:.4f}")
#             logger.info(f"  Disc: {train_losses['disc']:.4f}")
#             logger.info(f"  Val - Loss: {val_metrics['val_loss']:.4f}, "
#                        f"PSNR: {val_metrics['psnr']:.2f} dB, "
#                        f"SSIM: {val_metrics['ssim']:.4f}")
            
#             # Save sample patches
#             if (epoch + 1) % save_samples_interval == 0:
#                 logger.info(f"Saving sample patches for epoch {epoch + 1}...")
#                 save_sample_patches(
#                     self.generator,
#                     val_loader,
#                     epoch + 1,
#                     self.samples_dir,
#                     self.device,
#                     num_samples=num_samples
#                 )
#             # Save checkpoint
#             is_best = val_metrics['val_loss'] < self.best_val_loss
#             if is_best:
#                 self.best_val_loss = val_metrics['val_loss']
            
#             self.save_checkpoint(val_metrics, is_best)
            
#         logger.info("Training completed!")


# # ============================================================================
# # MAIN
# # ============================================================================

# def main():
#     """Main training script."""
    
#     # Configuration
#     config = {
#         'data_dir': '../ncct_cect/vindr_ds/test_registered_cases',
#         'labels_csv': '../ncct_cect/vindr_ds/labels.csv',
#         'output_dir': '../ncct_cect/vindr_ds/loss_handled_training',
        
#         'patch_size': (64, 64),
#         'patch_depth': 16,
#         'overlap_ratio': 0.75,
        
#         'batch_size': 16,
#         'learning_rate': 2e-4,
#         'epochs': 100,
#         # Discriminator training stabilization
#         'adv_warmup_epochs': 10,
#         'disc_lr_multiplier': 2.0,      # Discriminator learns 2x faster
#         'disc_updates_per_gen': 2,      # Train discriminator 2x per generator update
#         'real_label_smoothing': 0.9,    # Smooth real labels (0.9 instead of 1.0)
#         'fake_label_smoothing': 0.1,    # Smooth fake labels (0.1 instead of 0.0)
        
#         'device': 'cuda' if torch.cuda.is_available() else 'cpu'
#     }
    
#     logger.info("="*80)
#     logger.info("CT PHASE GENERATION TRAINING")
#     logger.info("="*80)
    
#     # TODO: Load your data pairs here
#     # data_pairs = load_data_pairs(config['data_dir'], config['labels_csv'])
#     # train_pairs, val_pairs = split_data(data_pairs)
    
#     # For now, using placeholder
#     train_pairs = []  # Your training data pairs
#     val_pairs = []    # Your validation data pairs
    
#     # Create datasets
#     train_dataset = CTPhaseDataset(
#         train_pairs,
#         patch_size=config['patch_size'],
#         patch_depth=config['patch_depth'],
#         overlap_ratio=config['overlap_ratio'],
#         augment=True
#     )
    
#     val_dataset = CTPhaseDataset(
#         val_pairs,
#         patch_size=config['patch_size'],
#         patch_depth=config['patch_depth'],
#         overlap_ratio=0.5,
#         augment=False
#     )
    
#     # Create dataloaders
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=config['batch_size'],
#         shuffle=True,
#         num_workers=4,
#         pin_memory=True
#     )
    
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=config['batch_size'],
#         shuffle=False,
#         num_workers=4,
#         pin_memory=True
#     )
    
#     # Initialize trainer
#     trainer = CTPhaseTrainer(config)
    
#     # Start training
#     try:
#         trainer.train(train_loader, val_loader, config['epochs'])
#     except KeyboardInterrupt:
#         logger.info("Training interrupted by user")
#         trainer.save_checkpoint(float('inf'), is_best=False)
#     except Exception as e:
#         logger.error(f"Training failed: {e}")
#         raise


# if __name__ == "__main__":
#     main()

"""
Clean CT Phase Generation Training Pipeline
============================================
A well-organized, debuggable implementation for generating different contrast phases of CT images.

Architecture:
- Phase-conditional CycleGAN
- Input phase -> Generate target phase (conditional)
- Generated target -> Regenerate input phase (cycle consistency)
- Discriminator on generated target
- Focal loss on organ masks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# DATASET
# ============================================================================

class CTPhaseDataset(Dataset):
    """
    Dataset for CT phase generation with paired patches and organ masks.
    
    Creates overlapping 3D patches from registered CT volumes with different phases.
    Includes organ segmentation masks for focal loss.
    """
    
    def __init__(
        self,
        data_pairs: List[Dict],
        patch_size: Tuple[int, int] = (64, 64),
        patch_depth: int = 7,
        overlap_ratio: float = 0.5,
        augment: bool = True
    ):
        self.data_pairs = data_pairs
        self.patch_size = patch_size
        self.patch_depth = patch_depth
        self.overlap_ratio = overlap_ratio
        self.augment = augment
        self.patch_coords = []
        
        # Phase mapping
        self.phase_to_idx = {
            'non-contrast': 0,
            'arterial': 1,
            'portal': 2,
            'venous': 2,  # Map venous to portal
            'delayed': 3
        }
        
        logger.info(f"Initializing dataset with {len(data_pairs)} pairs")
        logger.info(f"Patch size: {patch_size}, Depth: {patch_depth}, Overlap: {overlap_ratio}")
        
        self._compute_patch_coordinates()
        logger.info(f"Generated {len(self.patch_coords)} total patches")
    
    def _compute_patch_coordinates(self):
        """Pre-compute all valid patch coordinates from volumes."""
        padding = self.patch_depth // 2
        
        for pair_idx, pair_data in enumerate(self.data_pairs):
            try:
                # Load volumes to get dimensions
                source_vol = nib.load(pair_data['source_path']).get_fdata()
                target_vol = nib.load(pair_data['target_path']).get_fdata()
                
                # Validate shapes match
                if source_vol.shape != target_vol.shape:
                    logger.warning(f"Shape mismatch for pair {pair_idx}, skipping")
                    continue
                
                depth, height, width = source_vol.shape
                
                # Check minimum requirements
                if depth < self.patch_depth + 2:
                    logger.warning(f"Insufficient depth ({depth}) for pair {pair_idx}, skipping")
                    continue
                
                if height < self.patch_size[0] or width < self.patch_size[1]:
                    logger.warning(f"Insufficient spatial size for pair {pair_idx}, skipping")
                    continue
                
                # Calculate step sizes for overlap
                step_y = max(1, int(self.patch_size[0] * (1 - self.overlap_ratio)))
                step_x = max(1, int(self.patch_size[1] * (1 - self.overlap_ratio)))
                
                # Generate coordinate ranges
                z_range = range(padding, depth - padding)
                y_range = range(0, height - self.patch_size[0] + 1, step_y)
                x_range = range(0, width - self.patch_size[1] + 1, step_x)
                
                # Store all valid patch coordinates
                for center_z in z_range:
                    for y_start in y_range:
                        for x_start in x_range:
                            self.patch_coords.append((pair_idx, center_z, y_start, x_start))
                
                logger.info(f"Pair {pair_idx}: Generated {len(z_range)*len(y_range)*len(x_range)} patches")
                
            except Exception as e:
                logger.error(f"Error processing pair {pair_idx}: {e}")
                continue
    
    def _normalize_intensity(self, image: np.ndarray) -> np.ndarray:
        """Normalize CT intensity values."""
        # Clip to abdomen HU range
        image = np.clip(image, -100, 300)
        
        # Z-score normalization
        mean_val = np.mean(image)
        std_val = np.std(image)
        if std_val > 1e-6:
            image = (image - mean_val) / std_val
        
        return image.astype(np.float32)
    
    def _extract_organ_masks(
        self, 
        seg_volume: np.ndarray, 
        z_start: int, 
        z_end: int,
        y_start: int, 
        y_end: int, 
        x_start: int, 
        x_end: int
    ) -> Dict[str, np.ndarray]:
        """Extract organ masks from segmentation volume."""
        patch_seg = seg_volume[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Define organ labels (adjust based on your segmentation)
        organ_labels = {
            'liver': [1, 2],
            'kidney_right': [3],
            'kidney_left': [4],
            'spleen': [5],
        }
        
        masks = {}
        for organ, labels in organ_labels.items():
            mask = np.zeros_like(patch_seg, dtype=np.float32)
            for label in labels:
                mask[patch_seg == label] = 1.0
            masks[organ] = mask
        
        return masks
    
    def _augment(self, source, target, masks):
        """Apply 3D augmentations."""
        # Random horizontal flip
        if np.random.random() > 0.5:
            source = np.flip(source, axis=2).copy()
            target = np.flip(target, axis=2).copy()
            masks = {k: np.flip(v, axis=2).copy() for k, v in masks.items()}
        
        # Random vertical flip
        if np.random.random() > 0.5:
            source = np.flip(source, axis=1).copy()
            target = np.flip(target, axis=1).copy()
            masks = {k: np.flip(v, axis=1).copy() for k, v in masks.items()}
        
        return source, target, masks
    
    def __len__(self) -> int:
        return len(self.patch_coords)
    
    def __getitem__(self, idx: int) -> Dict:
        pair_idx, center_z, y_start, x_start = self.patch_coords[idx]
        pair_data = self.data_pairs[pair_idx]
        
        try:
            # Load volumes
            source_vol = nib.load(pair_data['source_path']).get_fdata()
            target_vol = nib.load(pair_data['target_path']).get_fdata()
            
            # Normalize
            source_vol = self._normalize_intensity(source_vol)
            target_vol = self._normalize_intensity(target_vol)
            
            # Extract patch
            padding = self.patch_depth // 2
            z_start = center_z - padding
            z_end = center_z + padding + 1
            y_end = y_start + self.patch_size[0]
            x_end = x_start + self.patch_size[1]
            
            source_patch = source_vol[z_start:z_end, y_start:y_end, x_start:x_end]
            target_patch = target_vol[z_start:z_end, y_start:y_end, x_start:x_end]
            
            # Extract organ masks if available
            masks = {}
            if pair_data.get('target_seg'):
                try:
                    seg_vol = nib.load(pair_data['target_seg']).get_fdata()
                    masks = self._extract_organ_masks(seg_vol, z_start, z_end, y_start, y_end, x_start, x_end)
                except Exception as e:
                    logger.debug(f"Could not load masks: {e}")
            
            # Augmentation
            if self.augment:
                source_patch, target_patch, masks = self._augment(source_patch, target_patch, masks)
            
            # Convert to tensors
            source_tensor = torch.from_numpy(source_patch).unsqueeze(0).float()  # [1, D, H, W]
            target_tensor = torch.from_numpy(target_patch).unsqueeze(0).float()
            
            mask_tensors = {
                organ: torch.from_numpy(mask).unsqueeze(0).float() 
                for organ, mask in masks.items()
            }
            
            # Get phase indices
            source_phase = pair_data['source_phase']
            target_phase = pair_data['target_phase']
            
            return {
                'source': source_tensor,
                'target': target_tensor,
                'source_phase': source_phase,
                'target_phase': target_phase,
                'source_phase_idx': self.phase_to_idx.get(source_phase, 0),
                'target_phase_idx': self.phase_to_idx.get(target_phase, 1),
                'masks': mask_tensors,
                'case_id': pair_data['case_id']
            }
            
        except Exception as e:
            logger.error(f"Error loading patch {idx}: {e}")
            # Return dummy data
            return {
                'source': torch.zeros(1, self.patch_depth, *self.patch_size),
                'target': torch.zeros(1, self.patch_depth, *self.patch_size),
                'source_phase': 'error',
                'target_phase': 'error',
                'source_phase_idx': 0,
                'target_phase_idx': 1,
                'masks': {},
                'case_id': 'error'
            }


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class PhaseConditionedGenerator(nn.Module):
    """
    3D U-Net Generator with phase conditioning.
    Takes input phase and generates target phase conditioned on phase label.
    """
    
    def __init__(self, num_phases: int = 4):
        super().__init__()
        
        self.num_phases = num_phases
        
        # Phase embedding
        self.phase_embedding = nn.Embedding(num_phases, 64)
        
        # Encoder
        self.enc1 = self._make_encoder_block(1, 64)      # Input + phase info
        self.enc2 = self._make_encoder_block(64, 128)
        self.enc3 = self._make_encoder_block(128, 256)
        self.enc4 = self._make_encoder_block(256, 512)
        
        self.pool = nn.MaxPool3d((1, 2, 2))  # Pool spatial dims only
        
        # Bottleneck with phase modulation
        self.bottleneck = nn.Sequential(
            nn.Conv3d(512, 1024, 3, padding=1),
            nn.InstanceNorm3d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(1024, 512, 3, padding=1),
            nn.InstanceNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Phase-adaptive modulation
        self.phase_scale_shift = nn.Linear(64, 512 * 2)
        
        # Decoder
        self.up4 = nn.ConvTranspose3d(512, 256, (1, 2, 2), stride=(1, 2, 2))
        self.dec4 = self._make_decoder_block(512 + 256, 256)
        
        self.up3 = nn.ConvTranspose3d(256, 128, (1, 2, 2), stride=(1, 2, 2))
        self.dec3 = self._make_decoder_block(256 + 128, 128)
        
        self.up2 = nn.ConvTranspose3d(128, 64, (1, 2, 2), stride=(1, 2, 2))
        self.dec2 = self._make_decoder_block(128 + 64, 64)
        
        self.up1 = nn.ConvTranspose3d(64, 64, (1, 2, 2), stride=(1, 2, 2))
        self.dec1 = self._make_decoder_block(64 + 64, 64)
        
        # Output
        self.output = nn.Sequential(
            nn.Conv3d(64, 1, 1),
            nn.Tanh()
        )
    
    def _make_encoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def _make_decoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x: torch.Tensor, phase_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image [B, 1, D, H, W]
            phase_idx: Target phase indices [B]
        """
        batch_size = x.size(0)
        
        # Get phase embedding [B, 64]
        phase_emb = self.phase_embedding(phase_idx)
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck with phase modulation
        b = self.bottleneck(self.pool(e4))
        
        # Apply phase-specific modulation
        phase_params = self.phase_scale_shift(phase_emb)
        scale = phase_params[:, :512].view(batch_size, 512, 1, 1, 1)
        shift = phase_params[:, 512:].view(batch_size, 512, 1, 1, 1)
        b = b * (1 + scale) + shift
        
        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return self.output(d1)


class Discriminator3D(nn.Module):
    """3D PatchGAN Discriminator with spectral normalization for stability."""
    
    def __init__(self):
        super().__init__()
        
        # Use spectral normalization for all conv layers to stabilize training
        self.model = nn.Sequential(
            # Layer 1 - No normalization on first layer
            nn.utils.spectral_norm(
                nn.Conv3d(1, 64, (3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1))
            ),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2
            nn.utils.spectral_norm(
                nn.Conv3d(64, 128, (3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1))
            ),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3
            nn.utils.spectral_norm(
                nn.Conv3d(128, 256, (3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))
            ),
            nn.InstanceNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4 - Additional layer for more capacity
            nn.utils.spectral_norm(
                nn.Conv3d(256, 512, (3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))
            ),
            nn.InstanceNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output
            nn.utils.spectral_norm(
                nn.Conv3d(512, 1, (2, 4, 4), stride=1, padding=0)
            )
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ============================================================================
# METRICS
# ============================================================================

class MetricsCalculator:
    """Calculate image quality metrics (SSIM, PSNR)."""
    
    @staticmethod
    def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 2.0) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio.
        
        Args:
            pred: Predicted image [-1, 1] range
            target: Target image [-1, 1] range
            data_range: Range of the data (2.0 for [-1, 1])
        """
        mse = torch.mean((pred - target) ** 2)
        if mse == 0:
            return 100.0
        psnr = 20 * torch.log10(data_range / torch.sqrt(mse))
        return psnr.item()
    
    @staticmethod
    def calculate_ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 2.0) -> float:
        """
        Calculate Structural Similarity Index.
        
        Args:
            pred: Predicted image [-1, 1] range [B, C, D, H, W]
            target: Target image [-1, 1] range [B, C, D, H, W]
            data_range: Range of the data
        """
        # Take middle slice for 2D SSIM calculation
        if pred.dim() == 5:  # [B, C, D, H, W]
            pred = pred[:, :, pred.shape[2]//2, :, :]  # [B, C, H, W]
            target = target[:, :, target.shape[2]//2, :, :]
        
        # Constants for SSIM
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        
        # Calculate means
        mu1 = F.avg_pool2d(pred, 3, 1, 1)
        mu2 = F.avg_pool2d(target, 3, 1, 1)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # Calculate variances and covariance
        sigma1_sq = F.avg_pool2d(pred ** 2, 3, 1, 1) - mu1_sq
        sigma2_sq = F.avg_pool2d(target ** 2, 3, 1, 1) - mu2_sq
        sigma12 = F.avg_pool2d(pred * target, 3, 1, 1) - mu1_mu2
        
        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean().item()


def save_sample_patches(
    generator: nn.Module,
    val_loader: DataLoader,
    epoch: int,
    save_dir: Path,
    device: torch.device,
    num_samples: int = 5
):
    """
    Save sample generated patches for visual inspection.
    
    Args:
        generator: The generator model
        val_loader: Validation dataloader
        epoch: Current epoch number
        save_dir: Directory to save samples
        device: Device to run on
        num_samples: Number of samples to save
    """
    generator.eval()
    save_dir = Path(save_dir) / f"epoch_{epoch}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    saved_count = 0
    
    with torch.no_grad():
        for batch in val_loader:
            if saved_count >= num_samples:
                break
            
            try:
                real_source = batch['source'].to(device)
                real_target = batch['target'].to(device)
                source_phase_idx = batch['source_phase_idx'].to(device)
                target_phase_idx = batch['target_phase_idx'].to(device)
                source_phase = batch['source_phase']
                target_phase = batch['target_phase']
                case_id = batch['case_id']
                
                # Generate target and reconstruct source
                generated_target = generator(real_source, target_phase_idx)
                reconstructed_source = generator(generated_target, source_phase_idx)
                
                # Process each sample in batch
                batch_size = real_source.size(0)
                for i in range(min(batch_size, num_samples - saved_count)):
                    # Get middle slice from 3D patch [1, D, H, W] -> [H, W]
                    mid_slice = real_source.shape[2] // 2
                    
                    source_slice = real_source[i, 0, mid_slice].cpu().numpy()
                    target_slice = real_target[i, 0, mid_slice].cpu().numpy()
                    generated_slice = generated_target[i, 0, mid_slice].cpu().numpy()
                    reconstructed_slice = reconstructed_source[i, 0, mid_slice].cpu().numpy()
                    
                    # Create comparison figure
                    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                    
                    # Row 1: Forward generation (source -> target)
                    im0 = axes[0, 0].imshow(source_slice, cmap='gray', vmin=-1, vmax=1)
                    axes[0, 0].set_title(f'Source: {source_phase[i]}', fontsize=12, fontweight='bold')
                    axes[0, 0].axis('off')
                    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
                    
                    im1 = axes[0, 1].imshow(generated_slice, cmap='gray', vmin=-1, vmax=1)
                    axes[0, 1].set_title(f'Generated: {target_phase[i]}', fontsize=12, fontweight='bold')
                    axes[0, 1].axis('off')
                    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
                    
                    im2 = axes[0, 2].imshow(target_slice, cmap='gray', vmin=-1, vmax=1)
                    axes[0, 2].set_title(f'Ground Truth: {target_phase[i]}', fontsize=12, fontweight='bold')
                    axes[0, 2].axis('off')
                    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
                    
                    # Row 2: Cycle reconstruction and difference maps
                    im3 = axes[1, 0].imshow(reconstructed_slice, cmap='gray', vmin=-1, vmax=1)
                    axes[1, 0].set_title(f'Reconstructed: {source_phase[i]}', fontsize=12, fontweight='bold')
                    axes[1, 0].axis('off')
                    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
                    
                    # Difference map: generated vs ground truth
                    diff_gen = np.abs(generated_slice - target_slice)
                    im4 = axes[1, 1].imshow(diff_gen, cmap='hot', vmin=0, vmax=1)
                    axes[1, 1].set_title('|Generated - GT|', fontsize=12, fontweight='bold')
                    axes[1, 1].axis('off')
                    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
                    
                    # Difference map: reconstructed vs source
                    diff_cycle = np.abs(reconstructed_slice - source_slice)
                    im5 = axes[1, 2].imshow(diff_cycle, cmap='hot', vmin=0, vmax=1)
                    axes[1, 2].set_title('|Reconstructed - Source|', fontsize=12, fontweight='bold')
                    axes[1, 2].axis('off')
                    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)
                    
                    # Calculate metrics for this sample
                    mse_gen = np.mean((generated_slice - target_slice) ** 2)
                    mse_cycle = np.mean((reconstructed_slice - source_slice) ** 2)
                    
                    # Add overall title with metrics
                    fig.suptitle(
                        f'Epoch {epoch} - Case: {case_id[i]} - '
                        f'MSE (Gen): {mse_gen:.4f}, MSE (Cycle): {mse_cycle:.4f}',
                        fontsize=14, fontweight='bold', y=0.98
                    )
                    
                    plt.tight_layout()
                    
                    # Save figure
                    save_path = save_dir / f'sample_{saved_count:03d}_{case_id[i]}.png'
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    saved_count += 1
                    
                    if saved_count >= num_samples:
                        break
                        
            except Exception as e:
                logger.error(f"Error saving sample patches: {e}")
                continue
    
    logger.info(f"Saved {saved_count} sample patches to {save_dir}")


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class FocalLoss(nn.Module):
    """Focal loss for organ-specific enhancement."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 1.5):  # REDUCED defaults
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
        masks: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted image [B, 1, D, H, W]
            target: Target image [B, 1, D, H, W]
            masks: Organ masks {organ_name: [B, 1, D, H, W]}
        """
        # Base MSE loss
        mse = F.mse_loss(pred, target, reduction='none')
        
        # Create weighted mask emphasizing organs
        weight_mask = torch.ones_like(mse)
        for organ_name, mask in masks.items():
            if mask.sum() > 0:
                # Higher weight for important organs
                organ_weight = 3.0 if organ_name == 'liver' else 2.0
                weight_mask += (organ_weight - 1.0) * mask
        
        # Apply focal weighting
        focal_weight = mse ** self.gamma
        weighted_loss = self.alpha * focal_weight * mse * weight_mask
        
        return weighted_loss.mean()


class CombinedLoss(nn.Module):
    """Combined loss for CycleGAN training with gradual adversarial loss warm-up."""
    
    def __init__(
        self,
        lambda_cycle: float = 10.0,
        lambda_mse: float = 100.0,
        lambda_focal: float = 5.0,
        lambda_adv: float = 1.0,  # Adversarial loss weight
        adv_warmup_epochs: int = 10  # Number of epochs to gradually increase adversarial loss
    ):
        super().__init__()
        self.lambda_cycle = lambda_cycle
        self.lambda_mse = lambda_mse
        self.lambda_focal = lambda_focal
        self.lambda_adv = lambda_adv
        self.adv_warmup_epochs = adv_warmup_epochs
        
        self.mse_loss = nn.MSELoss()
        self.focal_loss = FocalLoss(alpha=1.0, gamma=1.5)
        self.adv_loss = nn.BCEWithLogitsLoss()
        
        self.current_epoch = 0
    
    def set_epoch(self, epoch: int):
        """Update current epoch for warm-up schedule."""
        self.current_epoch = epoch
    
    def get_adv_weight(self) -> float:
        """Calculate adversarial loss weight with warm-up."""
        if self.current_epoch < self.adv_warmup_epochs:
            # Gradual warm-up: 0 -> lambda_adv over warmup_epochs
            return self.lambda_adv * (self.current_epoch / self.adv_warmup_epochs)
        return self.lambda_adv
    
    def forward(
        self,
        real_source: torch.Tensor,
        real_target: torch.Tensor,
        generated_target: torch.Tensor,
        reconstructed_source: torch.Tensor,
        disc_fake_target: torch.Tensor,
        disc_fake_source: torch.Tensor,  # NEW: discriminator on reconstructed source
        masks: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Calculate all generator losses."""
        
        losses = {}
        
        # Get current adversarial weight (with warm-up)
        adv_weight = self.get_adv_weight()
        
        # Adversarial losses (gradually increased)
        losses['adv_target'] = self.adv_loss(disc_fake_target, torch.ones_like(disc_fake_target)) * adv_weight
        losses['adv_source'] = self.adv_loss(disc_fake_source, torch.ones_like(disc_fake_source)) * adv_weight
        losses['adv'] = losses['adv_target'] + losses['adv_source']
        
        # Cycle consistency loss (always active)
        losses['cycle'] = self.mse_loss(reconstructed_source, real_source) * self.lambda_cycle
        
        # Direct MSE loss (always active)
        losses['mse'] = self.mse_loss(generated_target, real_target) * self.lambda_mse
        
        # Focal loss with organ masks (always active)
        if masks:
            losses['focal'] = self.focal_loss(generated_target, real_target, masks) * self.lambda_focal
        else:
            losses['focal'] = torch.tensor(0.0, device=real_source.device)
        
        # Total generator loss
        losses['total'] = losses['adv'] + losses['cycle'] + losses['mse'] + losses['focal']
        
        return losses


# ============================================================================
# TRAINER
# ============================================================================

class CTPhaseTrainer:
    """Clean trainer for CT phase generation."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config['device'])
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create samples directory
        self.samples_dir = self.output_dir / 'samples'
        self.samples_dir.mkdir(exist_ok=True)
        
        # Initialize models
        logger.info("Initializing models...")
        self.generator = PhaseConditionedGenerator(num_phases=4).to(self.device)
        self.disc_source = Discriminator3D().to(self.device)
        self.disc_target = Discriminator3D().to(self.device)
        
        # Optimizers with discriminator learning rate 2x higher
        disc_lr_multiplier = config.get('disc_lr_multiplier', 2.0)
        
        self.opt_gen = optim.Adam(
            self.generator.parameters(),
            lr=config['learning_rate'],
            betas=(0.5, 0.999)
        )
        self.opt_disc_source = optim.Adam(
            self.disc_source.parameters(),
            lr=config['learning_rate'] * disc_lr_multiplier,
            betas=(0.5, 0.999)
        )
        self.opt_disc_target = optim.Adam(
            self.disc_target.parameters(),
            lr=config['learning_rate'] * disc_lr_multiplier,
            betas=(0.5, 0.999)
        )
        
        # Loss functions
        self.combined_loss = CombinedLoss().to(self.device)
        self.disc_loss = nn.BCEWithLogitsLoss()
        
        # Label smoothing for discriminator stability
        self.real_label_smoothing = config.get('real_label_smoothing', 0.9)  # 0.9 instead of 1.0
        self.fake_label_smoothing = config.get('fake_label_smoothing', 0.1)  # 0.1 instead of 0.0
        
        # How many times to train discriminator per generator update
        self.disc_updates_per_gen = config.get('disc_updates_per_gen', 2)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        # Save config
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Trainer initialized on {self.device}")
        logger.info(f"Generator LR: {config['learning_rate']:.6f}")
        logger.info(f"Discriminator LR: {config['learning_rate'] * disc_lr_multiplier:.6f}")
        logger.info(f"Label smoothing: Real={self.real_label_smoothing}, Fake={self.fake_label_smoothing}")
        logger.info(f"Discriminator updates per generator update: {self.disc_updates_per_gen}")
    
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Single training step."""
        
        # Get data
        real_source = batch['source'].to(self.device)
        real_target = batch['target'].to(self.device)
        source_phase_idx = batch['source_phase_idx'].to(self.device)
        target_phase_idx = batch['target_phase_idx'].to(self.device)
        masks = {k: v.to(self.device) for k, v in batch['masks'].items()}
        # =====================
        # Train Discriminators (multiple times if specified)
        # =====================
        disc_source_losses = []
        disc_target_losses = []
        
        # First, generate fakes (will be detached for disc training)
        with torch.no_grad():
            generated_target = self.generator(real_source, target_phase_idx)
            reconstructed_source = self.generator(generated_target, source_phase_idx)
        
        for _ in range(self.disc_updates_per_gen):
            # Train disc_target (on target domain)
            self.opt_disc_target.zero_grad()
            
            # Real target
            disc_real_target = self.disc_target(real_target)
            loss_real_target = self.disc_loss(
                disc_real_target,
                torch.full_like(disc_real_target, self.real_label_smoothing)
            )
            
            # Fake generated target
            disc_fake_target_det = self.disc_target(generated_target.detach())
            loss_fake_target = self.disc_loss(
                disc_fake_target_det,
                torch.full_like(disc_fake_target_det, self.fake_label_smoothing)
            )
            
            # Total for target disc
            disc_target_loss = (loss_real_target + loss_fake_target) * 0.5
            disc_target_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.disc_target.parameters(), max_norm=1.0)
            self.opt_disc_target.step()
            
            disc_target_losses.append(disc_target_loss.item())
            
            # Train disc_source (on source domain)
            self.opt_disc_source.zero_grad()
            
            # Real source
            disc_real_source = self.disc_source(real_source)
            loss_real_source = self.disc_loss(
                disc_real_source,
                torch.full_like(disc_real_source, self.real_label_smoothing)
            )
            
            # Fake reconstructed source
            disc_fake_source_det = self.disc_source(reconstructed_source.detach())
            loss_fake_source = self.disc_loss(
                disc_fake_source_det,
                torch.full_like(disc_fake_source_det, self.fake_label_smoothing)
            )
            
            # Total for source disc
            disc_source_loss = (loss_real_source + loss_fake_source) * 0.5
            disc_source_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.disc_source.parameters(), max_norm=1.0)
            self.opt_disc_source.step()
            
            disc_source_losses.append(disc_source_loss.item())
        # =====================
        # Train Generator
        # =====================
        self.opt_gen.zero_grad()
        
        # Forward pass
        # Source -> Target (conditioned)
        generated_target = self.generator(real_source, target_phase_idx)
        
        # Generated Target -> Reconstructed Source (cycle)
        reconstructed_source = self.generator(generated_target, source_phase_idx)
        
        # Discriminator output for generated target
        # disc_fake = self.discriminator(generated_target)
        
        disc_fake_target = self.disc_target(generated_target)
        disc_fake_source = self.disc_source(reconstructed_source)

        # Calculate generator losses
        gen_losses = self.combined_loss(
            real_source, real_target, generated_target, 
            reconstructed_source, disc_fake_target, disc_fake_source, masks
        )

        # Backward
        gen_losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)  # Add gradient clipping
        self.opt_gen.step()
        
        
        # Return losses
        return {
            'gen_total': gen_losses['total'].item(),
            'gen_adv': gen_losses['adv'].item(),
            'gen_cycle': gen_losses['cycle'].item(),
            'gen_mse': gen_losses['mse'].item(),
            'gen_focal': gen_losses['focal'].item(),
            'disc_source': np.mean(disc_source_losses),
            'disc_target': np.mean(disc_target_losses),
            'disc': np.mean(disc_source_losses + disc_target_losses)
        }
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.generator.train()
        self.disc_source.train()
        self.disc_target.train()
        
        epoch_losses = {
            'gen_total': [], 'gen_adv': [], 'gen_cycle': [],
            'gen_mse': [], 'gen_focal': [], 
            'disc_source': [], 'disc_target': [], 'disc': []
        }
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch in progress_bar:
            try:
                losses = self.train_step(batch)
                
                # Accumulate losses
                for key, value in losses.items():
                    epoch_losses[key].append(value)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Gen': f"{losses['gen_total']:.3f}",
                    'Disc': f"{losses['disc']:.3f}",
                    'Phase': f"{batch['source_phase'][0]}->{batch['target_phase'][0]}"
                })
                
            except Exception as e:
                logger.error(f"Error in training step: {e}")
                continue
        
        # Average losses
        return {k: np.mean(v) if v else 0.0 for k, v in epoch_losses.items()}
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.generator.eval()
        val_losses = []
        psnr_scores = []
        ssim_scores = []
        
        metrics_calc = MetricsCalculator()
        
        progress_bar = tqdm(val_loader, desc="Validation")
        
        for batch in progress_bar:
            try:
                real_source = batch['source'].to(self.device)
                real_target = batch['target'].to(self.device)
                target_phase_idx = batch['target_phase_idx'].to(self.device)
                
                # Generate target
                generated_target = self.generator(real_source, target_phase_idx)
                
                # Calculate MSE loss
                val_loss = F.mse_loss(generated_target, real_target)
                val_losses.append(val_loss.item())
                
                # Calculate PSNR
                psnr = metrics_calc.calculate_psnr(generated_target, real_target)
                psnr_scores.append(psnr)
                
                # Calculate SSIM
                ssim = metrics_calc.calculate_ssim(generated_target, real_target)
                ssim_scores.append(ssim)
                
                # Update progress bar with current metrics
                progress_bar.set_postfix({
                    'Loss': f"{val_loss.item():.4f}",
                    'PSNR': f"{psnr:.2f}",
                    'SSIM': f"{ssim:.4f}"
                })
                
            except Exception as e:
                logger.error(f"Error in validation: {e}")
                continue
        
        return {
            'val_loss': np.mean(val_losses) if val_losses else float('inf'),
            'psnr': np.mean(psnr_scores) if psnr_scores else 0.0,
            'ssim': np.mean(ssim_scores) if ssim_scores else 0.0
        }
    
    def save_checkpoint(self, val_metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        
        checkpoint = {
            'epoch': self.current_epoch,
            'generator_state': self.generator.state_dict(),
            'disc_source_state': self.disc_source.state_dict(),
            'disc_target_state': self.disc_target.state_dict(),
            'opt_gen_state': self.opt_gen.state_dict(),
            'opt_disc_source_state': self.opt_disc_source.state_dict(),
            'opt_disc_target_state': self.opt_disc_target.state_dict(),
            'val_metrics': val_metrics,
            'config': self.config
        }
        checkpoint = {
            'epoch': self.current_epoch,
            'generator_state': self.generator.state_dict(),
            'disc_source_state': self.disc_source.state_dict(),
            'disc_target_state': self.disc_target.state_dict(),
            'opt_gen_state': self.opt_gen.state_dict(),
            'opt_disc_source_state': self.opt_disc_source.state_dict(),
            'opt_disc_target_state': self.opt_disc_target.state_dict(),
            'val_metrics': val_metrics,
            'config': self.config
        }
        
        # Save regular checkpoint
        path = self.output_dir / f'checkpoint_epoch_{self.current_epoch}.pth'
        torch.save(checkpoint, path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved! Val loss: {val_metrics['val_loss']:.4f}, "
                       f"PSNR: {val_metrics['psnr']:.2f}, SSIM: {val_metrics['ssim']:.4f}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        """Main training loop."""
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Training batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        # How often to save samples (default: every epoch)
        save_samples_interval = self.config.get('save_samples_interval', 1)
        num_samples = self.config.get('num_samples_to_save', 5)
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Train
            train_losses = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Log
            logger.info(f"\nEpoch {epoch + 1}/{epochs} Summary:")
            logger.info(f"  Train - Total: {train_losses['gen_total']:.4f}, "
                       f"Cycle: {train_losses['gen_cycle']:.4f}, "
                       f"MSE: {train_losses['gen_mse']:.4f}, "
                       f"Focal: {train_losses['gen_focal']:.4f}")
            logger.info(f"  Disc: {train_losses['disc']:.4f}")
            logger.info(f"  Val - Loss: {val_metrics['val_loss']:.4f}, "
                       f"PSNR: {val_metrics['psnr']:.2f} dB, "
                       f"SSIM: {val_metrics['ssim']:.4f}")
            
            # Save sample patches
            if (epoch + 1) % save_samples_interval == 0:
                logger.info(f"Saving sample patches for epoch {epoch + 1}...")
                save_sample_patches(
                    self.generator,
                    val_loader,
                    epoch + 1,
                    self.samples_dir,
                    self.device,
                    num_samples=num_samples
                )
            
            # Save checkpoint
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
            
            self.save_checkpoint(val_metrics, is_best)
            
        logger.info("Training completed!")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main training script."""
    
    # Configuration
    config = {
        'data_dir': '../ncct_cect/vindr_ds/test_registered_cases',
        'labels_csv': '../ncct_cect/vindr_ds/labels.csv',
        'output_dir': '../ncct_cect/vindr_ds/clean_training',
        
        'patch_size': (64, 64),
        'patch_depth': 7,
        'overlap_ratio': 0.75,
        
        'batch_size': 4,
        'learning_rate': 2e-4,
        'epochs': 100,
        
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # Sample visualization options
        'save_samples_interval': 1,  # Save every epoch (set to 5 for every 5 epochs)
        'num_samples_to_save': 5     # Number of samples per epoch
    }
    
    logger.info("="*80)
    logger.info("CT PHASE GENERATION TRAINING")
    logger.info("="*80)
    
    # Check if data directory exists
    if not Path(config['data_dir']).exists():
        logger.error(f"❌ Data directory does not exist: {config['data_dir']}")
        logger.error("Please update the 'data_dir' in config to point to your data")
        return
    
    # Load phase mapping if available
    phase_mapping = None
    if Path(config['labels_csv']).exists():
        try:
            # Import the data loader function
            from dataloader_train import load_phase_mapping, create_data_pairs
            
            phase_mapping = load_phase_mapping(config['labels_csv'])
            logger.info(f"✓ Loaded phase mapping for {len(phase_mapping)} cases")
        except Exception as e:
            logger.warning(f"⚠ Could not load phase mapping: {e}")
            logger.warning("Will try to infer phases from filenames")
    else:
        logger.warning(f"⚠ Labels CSV not found: {config['labels_csv']}")
        logger.warning("Will try to infer phases from filenames")
    
    # Create data pairs
    try:
        from dataloader_train import create_data_pairs
        
        logger.info(f"Loading data from: {config['data_dir']}")
        data_splits = create_data_pairs(
            config['data_dir'],
            phase_mapping=phase_mapping
        )
        
        if not data_splits['train']:
            logger.error("❌ No training data pairs found!")
            logger.error("\nPossible issues:")
            logger.error("1. Data directory is empty or has wrong structure")
            logger.error("2. Files are not named with '*_registered.nii.gz' pattern")
            logger.error("3. No cases have both non-contrast and contrast phases")
            logger.error("\nExpected directory structure:")
            logger.error("  data_dir/")
            logger.error("    case_001/")
            logger.error("      case_001_series1_registered.nii.gz")
            logger.error("      case_001_series2_registered.nii.gz")
            logger.error("    case_002/")
            logger.error("      ...")
            return
        
        logger.info(f"✓ Successfully loaded {len(data_splits['train'])} training pairs")
        logger.info(f"✓ Successfully loaded {len(data_splits['val'])} validation pairs")
        
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # TODO: Load your data pairs here
    # data_pairs = load_data_pairs(config['data_dir'], config['labels_csv'])
    # train_pairs, val_pairs = split_data(data_pairs)
    
    # For now, using placeholder
    train_pairs = []  # Your training data pairs
    val_pairs = []    # Your validation data pairs
    
    # Create datasets
    train_dataset = CTPhaseDataset(
        train_pairs,
        data_splits['train'],
        patch_size=config['patch_size'],
        patch_depth=config['patch_depth'],
        overlap_ratio=config['overlap_ratio'],
        augment=True
    )
    
    val_dataset = CTPhaseDataset(
        val_pairs,
        data_splits['val'],
        patch_size=config['patch_size'],
        patch_depth=config['patch_depth'],
        overlap_ratio=0.5,
        augment=False
    )
    
    if len(train_dataset) == 0:
        logger.error("❌ Training dataset is empty after patch generation!")
        logger.error("This usually means:")
        logger.error("1. Volumes are too small for the patch size")
        logger.error("2. Not enough slices (need at least depth + 2)")
        logger.error(f"Current patch size: {config['patch_size']}, depth: {config['patch_depth']}")
        return
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    logger.info(f"✓ Training dataset: {len(train_dataset)} patches")
    logger.info(f"✓ Validation dataset: {len(val_dataset)} patches")
    
    # Initialize trainer
    trainer = CTPhaseTrainer(config)
    
    # Start training
    try:
        trainer.train(train_loader, val_loader, config['epochs'])
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer.save_checkpoint(float('inf'), is_best=False)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()