"""
Train with Best Parameters from Optuna Study
============================================
Use the best hyperparameters found to train a full model.
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import logging

from ct_phase_training import CTPhaseDataset, CTPhaseTrainer
from dataloader_train import load_phase_mapping, create_data_pairs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_with_best_params():
    """Train model using best parameters from Optuna study."""
    
    # Best parameters from trial #35
    config = {
        # Data paths
        'data_dir': '../ncct_cect/vindr_ds/registered_cases',
        'labels_csv': '../ncct_cect/vindr_ds/labels.csv',
        'output_dir': '../ncct_cect/vindr_ds/best_params_training',
        
        # Architecture
        'patch_size': (96, 128),  # Can try (96, 96) or (128, 128) for better quality
        'patch_depth': 10,
        'overlap_ratio': 0.5,
        
        # Training
        'batch_size': 16,
        'epochs': 100,  # Full training
        
        # BEST HYPERPARAMETERS FROM OPTUNA (Trial #35)
        'learning_rate': 2e-4,  # You can use this or tune it
        'disc_lr_multiplier': 1.599,
        
        # Loss weights - CRITICAL VALUES FROM BEST TRIAL
        'lambda_cycle': 5.418,
        'lambda_mse': 23.928,
        'lambda_focal': 4.777,
        'lambda_adv': 0.173,
        
        # Training stability
        'adv_warmup_epochs': 8,
        'disc_updates_per_gen': 2,
        'real_label_smoothing': 0.896,
        'fake_label_smoothing': 0.118,
        
        # Visualization
        'save_samples_interval': 5,
        'num_samples_to_save': 10,
        
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    
    logger.info("="*80)
    logger.info("TRAINING WITH BEST OPTUNA PARAMETERS")
    logger.info("="*80)
    logger.info("Best parameters from Trial #35:")
    logger.info(f"  lambda_cycle: {config['lambda_cycle']:.3f}")
    logger.info(f"  lambda_mse: {config['lambda_mse']:.3f}")
    logger.info(f"  lambda_focal: {config['lambda_focal']:.3f}")
    logger.info(f"  lambda_adv: {config['lambda_adv']:.3f}")
    logger.info(f"  disc_lr_multiplier: {config['disc_lr_multiplier']:.3f}")
    logger.info("="*80)
    
    # Check disk space before starting
    import shutil
    total, used, free = shutil.disk_usage("/")
    free_gb = free // (2**30)
    
    if free_gb < 10:
        logger.error(f"❌ Only {free_gb}GB free on root partition!")
        logger.error("Please free up disk space before training!")
        # return
    
    logger.info(f"✅ Disk space check: {free_gb}GB free")
    
    # Load data
    try:
        logger.info("Loading data...")
        phase_mapping = None
        if Path(config['labels_csv']).exists():
            phase_mapping = load_phase_mapping(config['labels_csv'])
        
        data_splits = create_data_pairs(
            config['data_dir'],
            phase_mapping=phase_mapping
        )
        
        logger.info(f"✅ Data loaded:")
        logger.info(f"   Train: {len(data_splits['train'])} pairs")
        logger.info(f"   Val: {len(data_splits['val'])} pairs")
        
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create datasets
    try:
        logger.info("Creating datasets...")
        train_dataset = CTPhaseDataset(
            data_splits['train'],
            patch_size=config['patch_size'],
            patch_depth=config['patch_depth'],
            overlap_ratio=config['overlap_ratio'],
            augment=True
        )
        
        val_dataset = CTPhaseDataset(
            data_splits['val'],
            patch_size=config['patch_size'],
            patch_depth=config['patch_depth'],
            overlap_ratio=0.5,
            augment=False
        )
        
        logger.info(f"✅ Datasets created:")
        logger.info(f"   Train: {len(train_dataset)} patches")
        logger.info(f"   Val: {len(val_dataset)} patches")
        
    except Exception as e:
        logger.error(f"❌ Error creating datasets: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize trainer
    try:
        logger.info("Initializing trainer...")
        trainer = CTPhaseTrainer(config)
        logger.info("✅ Trainer initialized")
        
    except Exception as e:
        logger.error(f"❌ Error initializing trainer: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Start training
    logger.info("\n" + "="*80)
    logger.info("STARTING TRAINING")
    logger.info("="*80)
    
    try:
        trainer.train(train_loader, val_loader, config['epochs'])
        logger.info("\n✅ Training completed successfully!")
        
        # Print final results
        logger.info("\n" + "="*80)
        logger.info("TRAINING SUMMARY")
        logger.info("="*80)
        logger.info(f"Best validation loss: {trainer.best_val_loss:.6f}")
        logger.info(f"Output directory: {config['output_dir']}")
        logger.info(f"Best model saved to: {config['output_dir']}/best_model.pth")
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Training interrupted by user")
        logger.info("Saving checkpoint...")
        trainer.save_checkpoint({'val_loss': float('inf')}, is_best=False)
        
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Save what we have
        try:
            trainer.save_checkpoint({'val_loss': float('inf')}, is_best=False)
            logger.info("Checkpoint saved despite error")
        except:
            pass


if __name__ == "__main__":
    train_with_best_params()