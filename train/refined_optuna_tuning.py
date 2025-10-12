"""
Refined Optuna Hyperparameter Tuning
=====================================
Based on insights from previous ct_phase_loss_tuning study.

Key findings from Trial #35 (best):
- Lower MSE weight works better (24 vs 100)
- Lower adversarial weight (0.17 vs 1.0)
- Moderate cycle weight (5.4 vs 10)
- Discriminator should be weaker

New search strategy:
- Narrow ranges around best values
- Focus on promising regions
- Larger patch sizes for better quality
"""

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import logging

from ct_phase_training import CTPhaseDataset, CTPhaseTrainer
from dataloader_train import load_phase_mapping, create_data_pairs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_refined_objective(base_config: dict, data_splits: dict):
    """
    Create refined objective function with narrower parameter ranges.
    
    Based on analysis of ct_phase_loss_tuning results:
    - Best trial #35: val_loss=0.283, PSNR=17.0, SSIM=0.518
    - Key insight: MUCH lower loss weights work better
    """
    
    def objective(trial: optuna.Trial) -> float:
        """Refined objective function."""
        
        config = base_config.copy()
        
        # =================================================================
        # REFINED PARAMETER RANGES (based on Trial #35 findings)
        # =================================================================
        
        # Learning rates - keep similar to best
        config['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 3e-4, log=True)
        
        # Discriminator LR multiplier - narrow around 1.6
        config['disc_lr_multiplier'] = trial.suggest_float('disc_lr_multiplier', 1.2, 2.2)
        
        # CRITICAL: Loss weights - MUCH NARROWER ranges around best values
        # Previous best: cycle=5.4, mse=24, focal=4.8, adv=0.17
        config['lambda_cycle'] = trial.suggest_float('lambda_cycle', 3.0, 8.0)
        config['lambda_mse'] = trial.suggest_float('lambda_mse', 15.0, 40.0)
        config['lambda_focal'] = trial.suggest_float('lambda_focal', 2.0, 8.0)
        
        # Adversarial weight - VERY LOW (this was key finding!)
        config['lambda_adv'] = trial.suggest_float('lambda_adv', 0.08, 0.30)
        
        # Label smoothing - keep high for real, low for fake
        config['real_label_smoothing'] = trial.suggest_float('real_label_smoothing', 0.85, 0.95)
        config['fake_label_smoothing'] = trial.suggest_float('fake_label_smoothing', 0.05, 0.15)
        
        # Discriminator updates - 2 worked well
        config['disc_updates_per_gen'] = trial.suggest_categorical('disc_updates_per_gen', [1, 2, 3])
        
        # Warmup epochs - 8 worked well
        config['adv_warmup_epochs'] = trial.suggest_int('adv_warmup_epochs', 5, 12)
        
        # NEW: Try larger patch sizes for better quality
        # Previous trials used 64x64 and got PSNR~17
        # Larger patches should improve quality
        patch_size_options = [64, 96, 128]
        patch_size = trial.suggest_categorical('patch_size', patch_size_options)
        config['patch_size'] = (patch_size, patch_size)
        
        # Adjust batch size based on patch size (avoid OOM)
        if patch_size == 128:
            config['batch_size'] = 2
        elif patch_size == 96:
            config['batch_size'] = 3
        else:
            config['batch_size'] = 4
        
        # Patch depth - MUST BE ODD to avoid shape mismatch!
        # Even depths cause extracted patch to be depth+1 due to padding logic
        config['patch_depth'] = trial.suggest_categorical('patch_depth', [7, 9, 11, 13])
        
        # Validate patch depth is odd
        assert config['patch_depth'] % 2 == 1, f"patch_depth must be odd, got {config['patch_depth']}"
        
        # Create unique output directory
        trial_dir = Path(config['output_dir']) / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        config['trial_output_dir'] = str(trial_dir)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Trial {trial.number}")
        logger.info(f"{'='*80}")
        logger.info(f"Patch size: {patch_size}x{patch_size} (batch_size={config['batch_size']})")
        logger.info(f"Loss weights:")
        logger.info(f"  λ_cycle = {config['lambda_cycle']:.3f}")
        logger.info(f"  λ_mse   = {config['lambda_mse']:.3f}")
        logger.info(f"  λ_focal = {config['lambda_focal']:.3f}")
        logger.info(f"  λ_adv   = {config['lambda_adv']:.3f}")
        logger.info(f"Discriminator LR mult: {config['disc_lr_multiplier']:.3f}")
        
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
            
            # Train for evaluation epochs
            num_eval_epochs = config.get('trial_epochs', 30)  # Increased from 20
            
            best_val_loss = float('inf')
            best_psnr = 0.0
            best_ssim = 0.0
            
            for epoch in range(num_eval_epochs):
                # Train epoch
                train_losses = trainer.train_epoch(train_loader)
                
                # Validate
                val_metrics = trainer.validate(val_loader)
                
                # Track best metrics
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    best_psnr = val_metrics['psnr']
                    best_ssim = val_metrics['ssim']
                
                # Report for pruning
                trial.report(val_metrics['val_loss'], epoch)
                
                # Prune if not promising
                if trial.should_prune():
                    logger.info(f"Trial {trial.number} pruned at epoch {epoch}")
                    raise optuna.TrialPruned()
                
                # Log progress
                if epoch % 5 == 0 or epoch == num_eval_epochs - 1:
                    logger.info(
                        f"Epoch {epoch+1}/{num_eval_epochs} | "
                        f"Loss: {val_metrics['val_loss']:.4f} | "
                        f"PSNR: {val_metrics['psnr']:.2f} | "
                        f"SSIM: {val_metrics['ssim']:.4f}"
                    )
            
            # Store quality metrics as user attributes
            trial.set_user_attr('best_psnr', best_psnr)
            trial.set_user_attr('best_ssim', best_ssim)
            trial.set_user_attr('final_val_loss', best_val_loss)
            
            # Clean up
            del trainer, train_loader, val_loader, train_dataset, val_dataset
            torch.cuda.empty_cache()
            
            logger.info(
                f"Trial {trial.number} completed | "
                f"Best: Loss={best_val_loss:.4f}, PSNR={best_psnr:.2f}, SSIM={best_ssim:.4f}"
            )
            
            return best_val_loss
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    return objective


def start_refined_optimization(
    db_path: str = "optuna_study_v2.db",
    study_name: str = "ct_phase_refined",
    n_trials: int = 30,
    base_config: dict = None,
    use_data_subset: float = 1.0  # NEW: Fraction of data to use (0.25 = 25%)
):
    """
    Start refined optimization with lessons learned.
    
    Args:
        db_path: Path to NEW database file
        study_name: Name for new study
        n_trials: Number of trials to run
        base_config: Base configuration
        use_data_subset: Fraction of data to use (e.g., 0.25 for 25%, 1.0 for all)
    """
    
    # Default config
    if base_config is None:
        base_config = {
            'data_dir': '../ncct_cect/vindr_ds/registered_cases',
            'labels_csv': '../ncct_cect/vindr_ds/labels.csv',
            'output_dir': '../ncct_cect/vindr_ds/refined_tuning',
            
            'overlap_ratio': 0.75,
            'trial_epochs': 30,  # Increased from 20 for better evaluation
            
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'use_data_subset': use_data_subset,  # Store for later use
        }
    
    # Storage
    storage = f"sqlite:///{db_path}"
    
    # Create new study
    logger.info("="*80)
    logger.info("REFINED HYPERPARAMETER OPTIMIZATION")
    logger.info("="*80)
    logger.info(f"Based on insights from ct_phase_loss_tuning:")
    logger.info(f"  • Lower MSE weight (15-40 instead of 50-200)")
    logger.info(f"  • Much lower adversarial weight (0.08-0.30 instead of 0.5-2.0)")
    logger.info(f"  • Moderate cycle weight (3-8 instead of 5-20)")
    logger.info(f"  • Testing larger patch sizes (64/96/128 for better quality)")
    logger.info(f"  • ODD patch depths only (7/9/11/13 to avoid shape mismatch)")
    logger.info(f"  • Longer evaluation (30 epochs vs 20)")
    if use_data_subset < 1.0:
        logger.info(f"  • Using {use_data_subset*100:.0f}% of data for faster tuning")
    logger.info("="*80)
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(
            n_startup_trials=5,  # Don't prune first 5 trials
            n_warmup_steps=10,   # Wait 10 epochs before pruning
            interval_steps=5     # Check every 5 epochs
        ),
        load_if_exists=True
    )
    
    logger.info(f"Study: {study_name}")
    logger.info(f"Database: {db_path}")
    logger.info(f"Trials to run: {n_trials}")
    logger.info("="*80 + "\n")
    
    # Load data once
    logger.info("Loading data...")
    phase_mapping = None
    if Path(base_config['labels_csv']).exists():
        phase_mapping = load_phase_mapping(base_config['labels_csv'])
    
    data_splits = create_data_pairs(
        base_config['data_dir'],
        phase_mapping=phase_mapping
    )
    
    # NEW: Subset data if requested (for faster tuning)
    if use_data_subset < 1.0:
        import random
        random.seed(42)  # Reproducible
        
        original_train = len(data_splits['train'])
        original_val = len(data_splits['val'])
        
        # Take subset
        n_train = max(1, int(len(data_splits['train']) * use_data_subset))
        n_val = max(1, int(len(data_splits['val']) * use_data_subset))
        
        data_splits['train'] = random.sample(data_splits['train'], n_train)
        data_splits['val'] = random.sample(data_splits['val'], n_val)
        
        logger.info(f"📊 Using data subset ({use_data_subset*100:.0f}%):")
        logger.info(f"   Train: {n_train}/{original_train} pairs ({n_train/original_train*100:.0f}%)")
        logger.info(f"   Val: {n_val}/{original_val} pairs ({n_val/original_val*100:.0f}%)")
    else:
        logger.info(f"✅ Using full dataset:")
        logger.info(f"   Train: {len(data_splits['train'])} pairs")
        logger.info(f"   Val: {len(data_splits['val'])} pairs")
    
    logger.info("")
    
    # Create objective
    objective = create_refined_objective(base_config, data_splits)
    
    # Run optimization
    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=True,
            catch=(Exception,)
        )
        
        # Results
        logger.info("\n" + "="*80)
        logger.info("OPTIMIZATION COMPLETED")
        logger.info("="*80)
        
        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
        pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        
        logger.info(f"Trials: {len(study.trials)} total")
        logger.info(f"  Completed: {completed}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Pruned: {pruned}")
        
        if completed > 0:
            try:
                logger.info(f"\n🏆 Best Trial #{study.best_trial.number}:")
                logger.info(f"   Val Loss: {study.best_value:.6f}")
                
                # Get quality metrics
                best_psnr = study.best_trial.user_attrs.get('best_psnr', 'N/A')
                best_ssim = study.best_trial.user_attrs.get('best_ssim', 'N/A')
                logger.info(f"   PSNR: {best_psnr}")
                logger.info(f"   SSIM: {best_ssim}")
                
                logger.info(f"\n   Best Parameters:")
                for key, value in study.best_params.items():
                    logger.info(f"      {key}: {value}")
                
                # Save results
                import json
                results = {
                    'best_trial_number': study.best_trial.number,
                    'best_value': study.best_value,
                    'best_psnr': best_psnr,
                    'best_ssim': best_ssim,
                    'best_params': study.best_params,
                    'n_trials': len(study.trials),
                    'n_completed': completed
                }
                
                results_path = Path(base_config['output_dir']) / 'refined_best_params.json'
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=2)
                
                logger.info(f"\n💾 Saved results to: {results_path}")
                
                # Create trials CSV
                import pandas as pd
                trials_df = study.trials_dataframe()
                csv_path = Path(base_config['output_dir']) / 'refined_trials.csv'
                trials_df.to_csv(csv_path, index=False)
                logger.info(f"💾 Saved trials to: {csv_path}")
                
            except ValueError as e:
                logger.warning(f"Could not access best trial: {e}")
        else:
            logger.warning("No trials completed successfully")
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Optimization interrupted")
        logger.info(f"Progress saved to: {db_path}")
        
        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        if completed > 0:
            try:
                logger.info(f"Best so far: Trial #{study.best_trial.number}, Loss={study.best_value:.4f}")
            except:
                pass


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Refined Optuna tuning")
    parser.add_argument("--db", type=str, default="optuna_study_v2.db",
                       help="Path to database (will create new)")
    parser.add_argument("--study-name", type=str, default="ct_phase_refined_v2",
                       help="Name for refined study")
    parser.add_argument("--n-trials", type=int, default=30,
                       help="Number of trials")
    parser.add_argument("--data-dir", type=str,
                       default="../ncct_cect/vindr_ds/registered_cases")
    parser.add_argument("--trial-epochs", type=int, default=30,
                       help="Epochs per trial evaluation")
    parser.add_argument("--data-subset", type=float, default=0.25,
                       help="Fraction of data to use (0.25=25%%, 1.0=100%%)")
    
    args = parser.parse_args()
    
    base_config = {
        'data_dir': args.data_dir,
        'labels_csv': str(Path(args.data_dir).parent / 'labels.csv'),
        'output_dir': str(Path(args.db).parent / 'refined_tuning'),
        'overlap_ratio': 0.75,
        'trial_epochs': args.trial_epochs,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    
    start_refined_optimization(
        db_path=args.db,
        study_name=args.study_name,
        n_trials=args.n_trials,
        base_config=base_config,
        use_data_subset=args.data_subset  # Pass the subset parameter
    )