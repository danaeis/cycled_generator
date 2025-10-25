"""
Resume Optuna Hyperparameter Tuning
====================================
Continue optimization from existing database.
"""

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import random

# Import your training modules
from ct_phase_training import CTPhaseDataset, CTPhaseTrainer
from dataloader_train import load_phase_mapping, create_data_pairs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_objective(base_config: dict, data_splits: dict):
    """
    Create Optuna objective function.
    
    Args:
        base_config: Base configuration (non-tuned params)
        data_splits: Pre-loaded data splits
    """
    
    def objective(trial: optuna.Trial) -> float:
        """Objective function for Optuna."""
        
        # Suggest hyperparameters
        config = base_config.copy()
        config.update({
            # Learning rates
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True),
            'disc_lr_multiplier': trial.suggest_float('disc_lr_multiplier', 1.0, 3.0),
            
            # Loss weights
            'lambda_cycle': trial.suggest_float('lambda_cycle', 5.0, 20.0),
            'lambda_mse': trial.suggest_float('lambda_mse', 50.0, 200.0),
            'lambda_focal': trial.suggest_float('lambda_focal', 1.0, 10.0),
            'lambda_adv': trial.suggest_float('lambda_adv', 0.5, 2.0),
            
            # Label smoothing
            'real_label_smoothing': trial.suggest_float('real_label_smoothing', 0.8, 1.0),
            'fake_label_smoothing': trial.suggest_float('fake_label_smoothing', 0.0, 0.2),
            
            # Discriminator training
            'disc_updates_per_gen': trial.suggest_int('disc_updates_per_gen', 1, 3),
            
            # Architecture params (optional)
            'patch_size_exp': trial.suggest_int('patch_size_exp', 5, 7),  # 32, 64, 128
            'patch_depth': trial.suggest_int('patch_depth', 5, 11, step=2),  # odd numbers
        })
        
        # Derive patch size from exponent
        patch_size = 2 ** config['patch_size_exp']
        config['patch_size'] = (patch_size, patch_size)
        
        # Create unique output directory for this trial
        trial_dir = Path(config['output_dir']) / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        config['trial_output_dir'] = str(trial_dir)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Starting Trial {trial.number}")
        logger.info(f"{'='*80}")
        logger.info(f"Parameters:")
        for key, value in config.items():
            if key.startswith('lambda_') or key in ['learning_rate', 'disc_lr_multiplier']:
                logger.info(f"  {key}: {value}")
        
        try:
            # Create datasets
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
            trainer = CTPhaseTrainer(config)
            
            # Train for limited epochs (for quick evaluation)
            num_trial_epochs = config.get('trial_epochs', 20)
            
            best_val_loss = float('inf')
            
            for epoch in range(num_trial_epochs):
                # Train epoch
                train_losses = trainer.train_epoch(train_loader)
                
                # Validate
                val_metrics = trainer.validate(val_loader)
                
                # Track best
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                
                # Report intermediate value for pruning
                trial.report(val_metrics['val_loss'], epoch)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    logger.info(f"Trial {trial.number} pruned at epoch {epoch}")
                    raise optuna.TrialPruned()
                
                logger.info(
                    f"Trial {trial.number} | Epoch {epoch+1}/{num_trial_epochs} | "
                    f"Val Loss: {val_metrics['val_loss']:.4f} | "
                    f"PSNR: {val_metrics['psnr']:.2f} | "
                    f"SSIM: {val_metrics['ssim']:.4f}"
                )
            
            # Clean up to save memory
            del trainer, train_loader, val_loader, train_dataset, val_dataset
            torch.cuda.empty_cache()
            
            logger.info(f"Trial {trial.number} completed with best val loss: {best_val_loss:.6f}")
            
            return best_val_loss
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    return objective

def resume_optimization(
    db_path: str = "optuna_study.db",
    study_name: str = "ct_phase_optimization",
    n_trials: int = 50,
    base_config: dict = None,
    subset_fraction: float = 0.1
):
    """
    Resume Optuna optimization from database.
    
    Args:
        db_path: Path to database file
        study_name: Name of study
        n_trials: Number of additional trials to run
        base_config: Base configuration dict
        subset_fraction: Fraction of data to use for tuning
    """
    
    # Default base config
    if base_config is None:
        base_config = {
            'data_dir': '../ncct_cect/vindr_ds/registered_cases',
            'labels_csv': '../ncct_cect/vindr_ds/labels.csv',
            'output_dir': '../ncct_cect/vindr_ds/optuna_tuning',
            
            'overlap_ratio': 0.75,
            'batch_size': 4,
            'trial_epochs': 20,  # Short training for each trial
            
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        }
    
    # Storage
    storage = f"sqlite:///{db_path}"
    
    # Load or create study
    try:
        # Try to load existing study
        study = optuna.load_study(
            study_name=study_name,
            storage=storage
        )
        logger.info(f"✅ Loaded existing study: {study_name}")
        logger.info(f"   Trials completed: {len(study.trials)}")
        if study.best_trial:
            logger.info(f"   Best value so far: {study.best_value:.6f}")
            logger.info(f"   Best trial: {study.best_trial.number}")
    except KeyError:
        # Create new study if doesn't exist
        logger.info(f"Creating new study: {study_name}")
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="minimize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
            load_if_exists=True
        )
    
    # Load data once (don't reload for each trial)
    logger.info("Loading data splits...")
    phase_mapping = None
    if Path(base_config['labels_csv']).exists():
        phase_mapping = load_phase_mapping(base_config['labels_csv'])
    
    data_splits = create_data_pairs(
        base_config['data_dir'],
        phase_mapping=phase_mapping
    )
    logger.info(f"✅ Data loaded: {len(data_splits['train'])} train pairs")
    
    # Handle subset_fraction for data splits
    if subset_fraction < 1.0:
        random.seed(42)
        
        n_train = int(len(data_splits['train']) * subset_fraction)
        n_val = int(len(data_splits['val']) * subset_fraction)
        
        train_subset = random.sample(data_splits['train'], n_train)
        val_subset = random.sample(data_splits['val'], n_val)
        
        logger.info(f"Using {subset_fraction:.1%} of data for Optuna tuning:")
        logger.info(f"  Train: {n_train}/{len(data_splits['train'])} pairs")
        logger.info(f"  Val: {n_val}/{len(data_splits['val'])} pairs")
    else:
        train_subset = data_splits['train']
        val_subset = data_splits['val']
        logger.info("Using full dataset for Optuna tuning")
    
    # Update data_splits with subsets
    data_splits = {
        'train': train_subset,
        'val': val_subset
    }
    
    # Create objective
    objective = create_objective(base_config, data_splits)
    
    # Resume optimization
    logger.info(f"\n{'='*80}")
    logger.info(f"RESUMING OPTIMIZATION")
    logger.info(f"{'='*80}")
    logger.info(f"Study: {study_name}")
    logger.info(f"Database: {db_path}")
    logger.info(f"Additional trials: {n_trials}")
    logger.info(f"{'='*80}\n")
    
    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=True,
            catch=(Exception,)  # Continue even if some trials fail
        )
        
        # Print results
        logger.info(f"\n{'='*80}")
        logger.info("OPTIMIZATION COMPLETED")
        logger.info(f"{'='*80}")
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best value: {study.best_value:.6f}")
        logger.info(f"\nBest parameters:")
        for key, value in study.best_params.items():
            logger.info(f"  {key}: {value}")
        
        # Save best parameters
        best_params_path = Path(base_config['output_dir']) / 'best_parameters.json'
        import json
        with open(best_params_path, 'w') as f:
            json.dump(study.best_params, f, indent=2)
        logger.info(f"\n💾 Saved best parameters to: {best_params_path}")
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Optimization interrupted by user")
        logger.info(f"Progress saved to database: {db_path}")
        if study.best_trial:
            logger.info(f"Best trial so far: {study.best_trial.number}")
            logger.info(f"Best value so far: {study.best_value:.6f}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Resume Optuna tuning")
    parser.add_argument("--db", type=str, default="optuna_study.db",
                       help="Path to Optuna database")
    parser.add_argument("--study-name", type=str, default="ct_phase_optimization",
                       help="Name of study")
    parser.add_argument("--n-trials", type=int, default=50,
                       help="Number of additional trials")
    parser.add_argument("--data-dir", type=str, 
                       default="../ncct_cect/vindr_ds/registered_cases",
                       help="Path to data directory")
    
    args = parser.parse_args()
    
    # Base config
    base_config = {
        'data_dir': args.data_dir,
        'labels_csv': str(Path(args.data_dir).parent / 'labels.csv'),
        'output_dir': str(Path(args.db).parent / 'optuna_tuning'),
        
        'overlap_ratio': 0.75,
        'batch_size': 8,
        'trial_epochs': 20,
        
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    
    # Resume optimization
    resume_optimization(
        db_path=args.db,
        study_name=args.study_name,
        n_trials=args.n_trials,
        base_config=base_config,
        subset_fraction=0.25
    )