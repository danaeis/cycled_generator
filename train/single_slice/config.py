import torch
from pathlib import Path
# train_config = {
#     'data_dir': '../../../ncct_cect/vindr_ds/deformable_registered_bspline',
#     'labels_csv': '../../../ncct_cect/vindr_ds/labels.csv',
#     'output_dir': '../../../ncct_cect/vindr_ds/simlified_train/stage1_mse_only_64_autoenc',
        
#     'use_autoencoder': True,  # NEW: Enable autoencoder mode
    
#     'use_organ_loss': False,
#     'use_discriminator': False,
#     'use_phase_conditioning': False,
#     'use_cycle_consistency': False,
    
#     'target_phase': 'venous',
#     'patch_size': (128, 192),

#     'generator_base_channels': 64,    # or 32 for the smaller model
#     'generator_dropout': 0.1,
#     'num_phase': 2,
    
#     'patch_depth': 1,          # Always 1 for 2D
#     'slice_range': (0.3, 0.7), # Use middle 40% of volume
#     'overlap_ratio': 0.5,
    
#     'batch_size': 16,
#     'epochs': 50,

#     # Memory optimization
#     'use_mixed_precision': True,
#     'cleanup_frequency': 5,
#     'keep_last_n_checkpoints': 3,
#     # OPTIMIZED: Cache settings for faster loading
#     'cache_size': 12,  # Keep 12 volumes in cache (adjust based on RAM)
#     'use_memmap': True,  # Use memory-mapped files

#     'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
#     'keep_last_n_checkpoints': 3,
#     'save_samples_interval': 1,
#     'keep_last_n_sample_epochs': 5,
#     # Debug options
#     'debug_patches': True,
#     'debug_output_dir': './debug_patches',
#     'keep_last_n_checkpoints': 3,
#     'save_samples_interval': 1,
#     'keep_last_n_sample_epochs': 5,
#     }
train_config = {
    'data_dir': '../../ncct_cect/vindr_ds/registered_output',
    'labels_csv': '../../ncct_cect/vindr_ds/labels.csv',
    'output_dir': Path('../../ncct_cect/vindr_ds/simlified_train/stage1_mse_only_autoenc_noaug_e6_reg'),
        
    'use_autoencoder': False,  # NEW: Enable autoencoder mode
    
    'use_organ_loss': False,
    'use_discriminator': False,
    'use_phase_conditioning': True,
    'use_cycle_consistency': True,
    
    'target_phase': 'venous',
    'patch_size': (128, 192),

    'generator_base_channels': 64,    # or 32 for the smaller model
    'generator_dropout': 0.1,
    'num_phase': 2,
    
    'patch_depth': 1,          # Always 1 for 2D
    'slice_range': (0.1, 0.9), # Use middle 40% of volume
    'overlap_ratio': 0.5,
    
    'batch_size': 16,
    'epochs': 50,

    # Memory optimization
    'use_mixed_precision': True,
    'cleanup_frequency': 5,
    'keep_last_n_checkpoints': 3,
    # OPTIMIZED: Cache settings for faster loading
    'cache_size': 12,  # Keep 12 volumes in cache (adjust based on RAM)
    'use_memmap': True,  # Use memory-mapped files
    'learning_rate': 2e-6,
    'cosine_t0': 15,
    'cosine_tmult': 2,
    'cosine_eta_min': 5e-7,
    
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    'keep_last_n_checkpoints': 3,
    'save_samples_interval': 1,
    'keep_last_n_sample_epochs': 5,
    # Debug options
    'debug_patches': True,
    'debug_output_dir': './debug_patches',
    'keep_last_n_checkpoints': 3,
    'save_samples_interval': 1,
    'keep_last_n_sample_epochs': 5,

    }


# train_config = {
#     'data_dir': '../../../ncct_cect/vindr_ds/deformable_registered_bspline',
#     'labels_csv': '../../../ncct_cect/vindr_ds/labels.csv',
#     'output_dir': '../../../ncct_cect/vindr_ds/simlified_train/stage2_mse_mask_weighted_64',
    
#     'use_organ_loss': True,
#     'use_discriminator': False,
#     'use_phase_conditioning': False,
#     'use_cycle_consistency': False,
    
#     'target_phase': 'venous',
#     'patch_size': (128, 192),
#     'generator_base_channels': 64,    # or 32 for the smaller model
#     'generator_dropout': 0.3,
#     'num_phase': 2,
#     'patch_depth': 1,          # Always 1 for 2D
#     'slice_range': (0.3, 0.7), # Use middle 40% of volume
#     'overlap_ratio': 0.5,
    
#     'batch_size': 16,
#     'epochs': 50,

#     # Memory optimization
#     'use_mixed_precision': True,
#     'cleanup_frequency': 5,
#     'keep_last_n_checkpoints': 3,
#     # OPTIMIZED: Cache settings for faster loading
#     'cache_size': 12,  # Keep 12 volumes in cache (adjust based on RAM)
#     'use_memmap': True,  # Use memory-mapped files

#     'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
#     'keep_last_n_checkpoints': 3,
#     'save_samples_interval': 1,
#     'keep_last_n_sample_epochs': 5,
#     # Debug options
#     'debug_patches': True,
#     'debug_output_dir': './debug_patches',
#     'keep_last_n_checkpoints': 3,
#     'save_samples_interval': 1,
#     'keep_last_n_sample_epochs': 5,
#     }


# train_config = {
#     'data_dir': '../../../ncct_cect/vindr_ds/deformable_registered_bspline',
#     'labels_csv': '../../../ncct_cect/vindr_ds/labels.csv',
#     'output_dir': '../../../ncct_cect/vindr_ds/simlified_train/stage3_mse_disc_refined_32',
    
#     'use_autoencoder': False,  # NEW: Enable autoencoder mode
    
#     'use_organ_loss': False,
#     'use_discriminator': True,
#     'use_phase_conditioning': False,
#     'use_cycle_consistency': False,
    
#     'target_phase': 'venous',
#     'patch_size': (128, 192),
    
#     'patch_depth': 1,          # Always 1 for 2D
#     'slice_range': (0.3, 0.7), # Use middle 40% of volume
#     'overlap_ratio': 0.5,
#     'generator_base_channels': 32,    # or 32 for the smaller model
#     'generator_dropout': 0.3,
#     'num_phase': 2,
#     'batch_size': 16,
#     'epochs': 50,

#     # 'real_label_smoothing': 0.9,
#     # 'fake_label_smoothing': 0.1,
#     # Change to:
#     'lambda_adv': 0.05,  # ← 4x lower!
#     'adv_warmup_epochs': 20,  # Slower warmup
    
#     # Memory optimization
#     'use_mixed_precision': True,
#     'cleanup_frequency': 5,
#     'keep_last_n_checkpoints': 3,
#     # OPTIMIZED: Cache settings for faster loading
#     'cache_size': 12,  # Keep 12 volumes in cache (adjust based on RAM)
#     'use_memmap': True,  # Use memory-mapped files

#     'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
#     'keep_last_n_checkpoints': 3,
#     'save_samples_interval': 1,
#     'keep_last_n_sample_epochs': 5,
#     # Debug options
#     'debug_patches': True,
#     'debug_output_dir': './debug_patches',
#     'keep_last_n_checkpoints': 3,
#     'save_samples_interval': 1,
#     'keep_last_n_sample_epochs': 5,
#     }


# train_config = {

#     'data_dir': '../../../ncct_cect/vindr_ds/deformable_registered_bspline',
#     'labels_csv': '../../../ncct_cect/vindr_ds/labels.csv',
#     'output_dir': '../../../ncct_cect/vindr_ds/simlified_train/stage4_mse_cycled_32',
    
#     'use_organ_loss': False,
#     'use_discriminator': False,
#     'use_phase_conditioning': True,
#     'use_cycle_consistency': True,
    
#     'target_phase': 'venous',
#     'patch_size': (128, 192),
#     'generator_base_channels': 32,    # or 32 for the smaller model
#     'generator_dropout': 0.3,
#     'num_phase': 4,
#     'patch_depth': 1,          # Always 1 for 2D
#     'slice_range': (0.3, 0.7), # Use middle 40% of volume
#     'overlap_ratio': 0.5,
    
#     'batch_size': 16,
#     'epochs': 50,

#     # Memory optimization
#     'use_mixed_precision': True,
#     'cleanup_frequency': 5,
#     'keep_last_n_checkpoints': 3,
#     # OPTIMIZED: Cache settings for faster loading
#     'cache_size': 12,  # Keep 12 volumes in cache (adjust based on RAM)
#     'use_memmap': True,  # Use memory-mapped files

#     'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
#     'keep_last_n_checkpoints': 3,
#     'save_samples_interval': 1,
#     'keep_last_n_sample_epochs': 5,
#     # Debug options
#     'debug_patches': True,
#     'debug_output_dir': './debug_patches',
#     'keep_last_n_checkpoints': 3,
#     'save_samples_interval': 1,
#     'keep_last_n_sample_epochs': 5,
#     }


