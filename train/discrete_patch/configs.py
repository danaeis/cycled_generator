"""
Optimized Configuration for CT Phase Training
==============================================
Data is already preprocessed to 0-255 range - no re-normalization needed.
"""

import torch

# ============================================================================
# MAIN TRAINING CONFIG
# ============================================================================

train_config = {
    # ========== DATA PATHS ==========
    'data_dir': '../../ncct_cect/vindr_ds/registered_aligned',
    'labels_csv': '../../ncct_cect/vindr_ds/labels.csv',
    'output_dir': '../../ncct_cect/vindr_ds/optimized_train/discrete_masked_conditioned_p8',
    
    'tag': "registered_norm",

    # ========== DATA CONFIGURATION ==========
    'data_is_preprocessed': True,  # Data already normalized to 0-255
    'discrete_range': (0, 255),    # Range of discrete values
    
    # ========== MASK WEIGHTING ==========
    'use_mask_weighting': True,      # Enable mask-based loss weighting
    'high_intensity_weight': 3.0,    # 3x penalty on bright regions (vessels/organs)
    'high_intensity_threshold': 240, # Threshold for "high intensity" in 0-255 scale
    'mask_labels': [1, 5, 2, 3, 4, 6, 7],
    'mask_weights': 3.0,
    # ========== PHASE CONDITIONING ==========
    'use_phase_conditioning': True,  # Enable phase embedding in generator
    'num_phase': 4,                  # Number of phase types
    'phase_embedding_dim': 64,       # Dimension of phase embeddings
    
    # ========== PATCH CONFIGURATION ==========
    'patch_size': (128, 192),        # (H, W) for axial patches
    'patch_depth': 8,                # Number of slices (1 for 2D)
    'slice_range': (0.1, 0.9),       # Use middle 40% of volume
    'overlap_ratio': 0.5,            # Patch overlap (0.5 = 50%)
    
    # ========== MODEL ARCHITECTURE ==========
    'generator_base_channels': 64,   # Base channels in U-Net (32 or 64)
    'generator_dropout': 0.3,        # Dropout rate for regularization
    
    # ========== TRAINING PARAMETERS ==========
    'batch_size': 16,                # Batch size (adjust based on GPU memory)
    'learning_rate': 2e-4,           # Initial learning rate
    'epochs': 50,                    # Total training epochs
    
    # ========== FEATURE FLAGS ==========
    'use_focal_loss': False,         # Use focal loss (for hard examples)
    'use_discriminator': False,      # Use adversarial training
    'use_cycle_consistency': False,  # Use cycle consistency loss
    
    # ========== MEMORY OPTIMIZATION ==========
    'use_mixed_precision': True,     # Enable mixed precision training
    'cleanup_frequency': 5,          # GPU memory cleanup frequency
    'keep_last_n_checkpoints': 3,    # Number of recent checkpoints to keep
    
    # ========== CACHING SETTINGS ==========
    'cache_size': 12,                # Keep 12 volumes in cache (adjust based on RAM)
    'use_memmap': True,              # Use memory-mapped file access
    
    # ========== VALIDATION & SAMPLING ==========
    'min_intensity_ratio': 0.1,      # Minimum valid intensity ratio for patches
    'min_mean': 10.0,                # Minimum mean intensity
    'min_std': 5.0,                  # Minimum standard deviation
    'validate_patches': True,        # Validate patches during generation
    
    # ========== SAMPLE SAVING ==========
    'save_samples_interval': 1,           # Save samples every N epochs
    'keep_last_n_sample_epochs': 5,       # Keep samples from last N epochs
    'num_samples_to_save': 10,            # Number of samples per epoch
    
    # ========== DEVICE ==========
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # ========== DEBUG OPTIONS ==========
    'debug_patches': False,          # Enable patch debugging
    'debug_output_dir': './debug_patches',
}


# ============================================================================
# ALTERNATIVE CONFIGS (for different training stages)
# ============================================================================

# Stage 1: MSE only (baseline)
stage1_config = train_config.copy()
stage1_config.update({
    'output_dir': '../../ncct_cect/vindr_ds/optimized_train/stage1_mse_only',
    'use_mask_weighting': False,
    'use_phase_conditioning': False,
    'generator_base_channels': 32,
})

# Stage 2: MSE + Mask Weighting
stage2_config = train_config.copy()
stage2_config.update({
    'output_dir': '../../ncct_cect/vindr_ds/optimized_train/stage2_mse_weighted',
    'use_mask_weighting': True,
    'use_phase_conditioning': False,
    'generator_base_channels': 64,
})

# Stage 3: Full optimization (MSE + Mask + Phase Conditioning)
stage3_config_p = train_config.copy()
stage3_config_p.update({
    'output_dir': '../../ncct_cect/vindr_ds/optimized_train/stage3_full_optimized_p',
    'use_mask_weighting': True,
    'use_phase_conditioning': True,
    'generator_base_channels': 64,
    'patch_depth': 8,
})

# Stage 3: Full optimization (MSE + Mask + Phase Conditioning)
stage3_config_2d = train_config.copy()
stage3_config_2d.update({
    'output_dir': '../../ncct_cect/vindr_ds/optimized_train/stage3_full_optimized_2d',
    'use_mask_weighting': True,
    'use_phase_conditioning': True,
    'generator_base_channels': 64,
    'patch_depth': 1,
})


