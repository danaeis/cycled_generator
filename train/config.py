import torch

train_config = {
    'data_dir': '../ncct_cect/vindr_ds/deformable_registered_bspline',
    'labels_csv': '../ncct_cect/vindr_ds/labels.csv',
    'output_dir': '../ncct_cect/vindr_ds/memeff_patch_bspline_exclude_training_thrshF_organ1',
    
    # Patch configuration
    'patch_size': (128, 192),
    'patch_depth': 15,
    'overlap_ratio': 0.5,
    'pad_mode': 'constant',
    'save_nifti': True,
      
    'min_intensity_ratio': 0.7205,  # Reject worst 5%
    'min_mean': -0.2886,
    'min_std': 0.2135,
    'validate_patches': False,
    'min_fg_ratio': 0.7058,
    'max_same_value_ratio': 0.90,

    # Memory optimization
    'use_mixed_precision': True,
    'cleanup_frequency': 5,
    'keep_last_n_checkpoints': 3,
    # OPTIMIZED: Cache settings for faster loading
    'cache_size': 12,  # Keep 12 volumes in cache (adjust based on RAM)
    'use_memmap': True,  # Use memory-mapped files
    
    # Loss weights (ALL PRESERVED)
    'disc_lr_multiplier': 1.599,
    'lambda_cycle': 10,
    'lambda_mse_initial': 1.0,
    'lambda_mse_final': 50.0,
    'mse_warmup_epochs': 50,
    'lambda_adv': 0.1,
    
    'lambda_organ'      : 1,     # weight of the *masked* MSE term
    'organ_weight'      : 5.0,    # how many times organ voxels count inside the MSE
    
    # Training stability (ALL PRESERVED)
    'adv_warmup_epochs': 8,
    'disc_updates_per_gen': 2,
    'real_label_smoothing': 0.896,
    'fake_label_smoothing': 0.118,
    
    # Training parameters
    'batch_size': 2,
    'learning_rate': 2e-4,
    'epochs': 100,
    
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Debug options
    'debug_patches': False,
    'debug_output_dir': './debug_patches',
    'keep_last_n_checkpoints': 3,
    'save_samples_interval': 1,
    'keep_last_n_sample_epochs': 5,
    }