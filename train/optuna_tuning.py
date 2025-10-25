"""
Optuna Hyperparameter Tuning for CT Phase Training
===================================================
Automatically tune loss weights and training hyperparameters using Optuna.

Features:
- TPE (Tree-structured Parzen Estimator) sampler
- Median pruning for early stopping of unpromising trials
- Multi-objective optimization (PSNR, SSIM, Val Loss)
- Comprehensive visualization of results
- Automatic checkpoint saving for best trials
"""

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, Optional
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import your existing modules
from ct_phase_training import (
    CTPhaseDataset, 
    PhaseConditionedGenerator, 
    Discriminator3D,
    CombinedLoss,
    MetricsCalculator
)
from dataloader_train import create_data_pairs, load_phase_mapping

logger = logging.getLogger(__name__)


class OptunaTrainer:
    """
    Lightweight trainer for Optuna trials with early stopping support.
    """
    
    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        trial: optuna.Trial
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.trial = trial
        self.device = torch.device(config['device'])
        
        # Initialize models
        self.generator = PhaseConditionedGenerator(num_phases=4).to(self.device)
        self.disc_source = Discriminator3D().to(self.device)
        self.disc_target = Discriminator3D().to(self.device)
        
        # Initialize optimizers
        self.opt_gen = torch.optim.Adam(
            self.generator.parameters(),
            lr=config['learning_rate'],
            betas=(0.5, 0.999)
        )
        self.opt_disc_source = torch.optim.Adam(
            self.disc_source.parameters(),
            lr=config['learning_rate'] * config.get('disc_lr_multiplier', 2.0),
            betas=(0.5, 0.999)
        )
        self.opt_disc_target = torch.optim.Adam(
            self.disc_target.parameters(),
            lr=config['learning_rate'] * config.get('disc_lr_multiplier', 2.0),
            betas=(0.5, 0.999)
        )
        
        # Initialize loss function with trial hyperparameters
        self.combined_loss = CombinedLoss(
            lambda_cycle=config['lambda_cycle'],
            lambda_mse=config['lambda_mse'],
            lambda_focal=config['lambda_focal'],
            lambda_adv=config['lambda_adv'],
            adv_warmup_epochs=config['adv_warmup_epochs']
        ).to(self.device)
        
        self.disc_loss = nn.BCEWithLogitsLoss()
        
        # Label smoothing
        self.real_label_smoothing = config.get('real_label_smoothing', 0.9)
        self.fake_label_smoothing = config.get('fake_label_smoothing', 0.1)
        self.disc_updates_per_gen = config.get('disc_updates_per_gen', 2)
        
        self.metrics_calc = MetricsCalculator()
    
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Single training step (simplified version)."""
        real_source = batch['source'].to(self.device)
        real_target = batch['target'].to(self.device)
        source_phase_idx = batch['source_phase_idx'].to(self.device)
        target_phase_idx = batch['target_phase_idx'].to(self.device)
        masks = {k: v.to(self.device) for k, v in batch['masks'].items()}
        
        # Train discriminators
        with torch.no_grad():
            generated_target = self.generator(real_source, target_phase_idx)
            reconstructed_source = self.generator(generated_target, source_phase_idx)
        
        disc_losses = []
        for _ in range(self.disc_updates_per_gen):
            # Disc target
            self.opt_disc_target.zero_grad()
            disc_real_target = self.disc_target(real_target)
            disc_fake_target = self.disc_target(generated_target.detach())
            
            loss_real = self.disc_loss(
                disc_real_target,
                torch.full_like(disc_real_target, self.real_label_smoothing)
            )
            loss_fake = self.disc_loss(
                disc_fake_target,
                torch.full_like(disc_fake_target, self.fake_label_smoothing)
            )
            
            disc_target_loss = (loss_real + loss_fake) * 0.5
            disc_target_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.disc_target.parameters(), max_norm=1.0)
            self.opt_disc_target.step()
            
            # Disc source
            self.opt_disc_source.zero_grad()
            disc_real_source = self.disc_source(real_source)
            disc_fake_source = self.disc_source(reconstructed_source.detach())
            
            loss_real = self.disc_loss(
                disc_real_source,
                torch.full_like(disc_real_source, self.real_label_smoothing)
            )
            loss_fake = self.disc_loss(
                disc_fake_source,
                torch.full_like(disc_fake_source, self.fake_label_smoothing)
            )
            
            disc_source_loss = (loss_real + loss_fake) * 0.5
            disc_source_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.disc_source.parameters(), max_norm=1.0)
            self.opt_disc_source.step()
            
            disc_losses.append((disc_source_loss.item() + disc_target_loss.item()) / 2)
        
        # Train generator
        self.opt_gen.zero_grad()
        
        generated_target = self.generator(real_source, target_phase_idx)
        reconstructed_source = self.generator(generated_target, source_phase_idx)
        
        disc_fake_target = self.disc_target(generated_target)
        disc_fake_source = self.disc_source(reconstructed_source)
        
        gen_losses = self.combined_loss(
            real_source, real_target, generated_target,
            reconstructed_source, disc_fake_target, disc_fake_source, masks
        )
        
        gen_losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        self.opt_gen.step()
        
        return {
            'gen_total': gen_losses['total'].item(),
            'disc': np.mean(disc_losses)
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validation with metrics."""
        self.generator.eval()
        val_losses = []
        psnr_scores = []
        ssim_scores = []
        
        for batch in self.val_loader:
            real_source = batch['source'].to(self.device)
            real_target = batch['target'].to(self.device)
            target_phase_idx = batch['target_phase_idx'].to(self.device)
            
            generated_target = self.generator(real_source, target_phase_idx)
            
            val_loss = torch.nn.functional.mse_loss(generated_target, real_target)
            val_losses.append(val_loss.item())
            
            psnr = self.metrics_calc.calculate_psnr(generated_target, real_target)
            psnr_scores.append(psnr)
            
            ssim = self.metrics_calc.calculate_ssim(generated_target, real_target)
            ssim_scores.append(ssim)
        
        self.generator.train()
        
        return {
            'val_loss': np.mean(val_losses),
            'psnr': np.mean(psnr_scores),
            'ssim': np.mean(ssim_scores)
        }
    
    def train_epochs(self, num_epochs: int) -> Dict[str, float]:
        """
        Train for specified epochs with pruning support.
        
        Returns best validation metrics.
        """
        best_val_loss = float('inf')
        best_metrics = None
        
        for epoch in range(num_epochs):
            # Update epoch for warmup schedule
            self.combined_loss.set_epoch(epoch)
            
            # Training
            self.generator.train()
            self.disc_source.train()
            self.disc_target.train()
            
            for batch in self.train_loader:
                self.train_step(batch)
            
            # Validation
            val_metrics = self.validate()
            
            # Track best
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                best_metrics = val_metrics.copy()
            
            # Report to Optuna for pruning
            self.trial.report(val_metrics['val_loss'], epoch)
            
            # Check if trial should be pruned
            if self.trial.should_prune():
                raise optuna.TrialPruned()
            
            logger.info(f"Trial {self.trial.number} - Epoch {epoch+1}/{num_epochs}: "
                       f"Val Loss={val_metrics['val_loss']:.4f}, "
                       f"PSNR={val_metrics['psnr']:.2f}, "
                       f"SSIM={val_metrics['ssim']:.4f}")
        
        return best_metrics


def check_study_status(storage_path: str, study_name: str) -> Dict:
    """
    Check if a study exists and return information about it.
    
    Args:
        storage_path: Path to SQLite database
        study_name: Name of the study
    
    Returns:
        Dictionary with study information or None if study doesn't exist
    """
    storage_path = Path(storage_path)
    
    if not storage_path.exists():
        logger.info(f"No existing study found at {storage_path}")
        return None
    
    try:
        # Load the existing study to get information
        study = optuna.load_study(
            study_name=study_name,
            storage=f'sqlite:///{storage_path}'
        )
        
        # Get study statistics
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
        
        study_info = {
            'exists': True,
            'n_trials': len(study.trials),
            'n_complete_trials': len(completed_trials),
            'n_pruned_trials': len(pruned_trials),
            'n_failed_trials': len(failed_trials),
            'best_value': study.best_value if completed_trials else None,
            'best_trial': study.best_trial.number if completed_trials else None,
            'study': study
        }
        
        logger.info(f"Found existing study '{study_name}' with {study_info['n_trials']} trials")
        logger.info(f"  Completed: {study_info['n_complete_trials']}")
        logger.info(f"  Pruned: {study_info['n_pruned_trials']}")
        logger.info(f"  Failed: {study_info['n_failed_trials']}")
        if study_info['best_value'] is not None:
            logger.info(f"  Best validation loss: {study_info['best_value']:.6f}")
        
        return study_info
        
    except Exception as e:
        logger.warning(f"Error loading existing study: {e}")
        return None

def create_study(
    study_name: str,
    storage_path: str,
    direction: str = 'minimize'
) -> optuna.Study:
    """
    Create Optuna study with TPE sampler and median pruner.
    
    Args:
        study_name: Name of the study
        storage_path: Path to SQLite database for storing results
        direction: 'minimize' for val_loss, 'maximize' for PSNR/SSIM
    
    Returns:
        Optuna study object
    """
    sampler = TPESampler(
        seed=42,
        n_startup_trials=10,  # Random trials before TPE kicks in
        multivariate=True,    # Consider parameter interactions
        warn_independent_sampling=False
    )
    
    pruner = MedianPruner(
        n_startup_trials=5,   # Don't prune first 5 trials
        n_warmup_steps=5,     # Don't prune first 3 epochs
        interval_steps=1      # Check every epoch
    )
    
    # Check if study already exists
    storage_path_obj = Path(storage_path)
    if storage_path_obj.exists():
        logger.info(f"Database file exists at {storage_path}")
        logger.info(f"Loading existing study '{study_name}' and continuing optimization...")
    else:
        logger.info(f"Creating new study '{study_name}' at {storage_path}")
    
    study = optuna.create_study(
        study_name=study_name,
        storage=f'sqlite:///{storage_path}',
        sampler=sampler,
        pruner=pruner,
        direction=direction,
        load_if_exists=True
    )
    
    return study


def objective(
    trial: optuna.Trial,
    train_loader: DataLoader,
    val_loader: DataLoader,
    base_config: Dict
) -> float:
    """
    Objective function for Optuna optimization.
    
    Args:
        trial: Optuna trial object
        train_loader: Training dataloader
        val_loader: Validation dataloader
        base_config: Base configuration dictionary
    
    Returns:
        Metric to optimize (validation loss)
    """
    # Suggest hyperparameters
    config = base_config.copy()
    
    # Loss weights - log scale for wider range
    config['lambda_cycle'] = trial.suggest_float('lambda_cycle', 3.0, 8.0, log=True)
    config['lambda_mse'] = trial.suggest_float('lambda_mse', 15.0, 40.0, log=True)
    config['lambda_focal'] = trial.suggest_float('lambda_focal', 2.0, 8.0, log=True)
    config['lambda_adv'] = trial.suggest_float('lambda_adv', 0.08, 0.30, log=True)
    
    # Warmup epochs
    config['adv_warmup_epochs'] = trial.suggest_int('adv_warmup_epochs', 5, 12)
    
    # Discriminator training frequency
    config['disc_updates_per_gen'] = trial.suggest_int('disc_updates_per_gen', 1, 4)
    
    # Label smoothing
    config['real_label_smoothing'] = trial.suggest_float('real_label_smoothing', 0.85, 0.95)
    config['fake_label_smoothing'] = trial.suggest_float('fake_label_smoothing', 0.05, 0.15)
    
    # Discriminator learning rate multiplier
    config['disc_lr_multiplier'] = trial.suggest_float('disc_lr_multiplier', 1.2, 2.2)
    
    logger.info(f"\nTrial {trial.number} hyperparameters:")
    logger.info(f"  lambda_cycle: {config['lambda_cycle']:.2f}")
    logger.info(f"  lambda_mse: {config['lambda_mse']:.2f}")
    logger.info(f"  lambda_focal: {config['lambda_focal']:.2f}")
    logger.info(f"  lambda_adv: {config['lambda_adv']:.2f}")
    logger.info(f"  adv_warmup_epochs: {config['adv_warmup_epochs']}")
    logger.info(f"  disc_updates_per_gen: {config['disc_updates_per_gen']}")
    logger.info(f"  disc_lr_multiplier: {config['disc_lr_multiplier']:.2f}")
    
    # Create trainer
    trainer = OptunaTrainer(train_loader, val_loader, config, trial)
    
    # Train for limited epochs (tune this based on your needs)
    num_epochs = base_config.get('optuna_epochs', 15)
    
    try:
        best_metrics = trainer.train_epochs(num_epochs)
        
        # Store additional metrics as user attributes
        trial.set_user_attr('psnr', best_metrics['psnr'])
        trial.set_user_attr('ssim', best_metrics['ssim'])
        trial.set_user_attr('val_loss', best_metrics['val_loss'])
        
        # Return primary metric (validation loss)
        return best_metrics['val_loss']
        
    except optuna.TrialPruned:
        logger.info(f"Trial {trial.number} was pruned")
        raise
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        raise


def visualize_optimization_results(study: optuna.Study, output_dir: Path):
    """
    Create comprehensive visualizations of optimization results.
    
    Args:
        study: Completed Optuna study
        output_dir: Directory to save visualizations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Optimization history
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_html(str(output_dir / 'optimization_history.html'))
    
    # 2. Parallel coordinate plot
    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.write_html(str(output_dir / 'parallel_coordinate.html'))
    
    # 3. Parameter importances
    try:
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(str(output_dir / 'param_importances.html'))
    except:
        logger.warning("Could not create parameter importance plot")
    
    # 4. Slice plot
    fig = optuna.visualization.plot_slice(study)
    fig.write_html(str(output_dir / 'slice_plot.html'))
    
    # 5. Contour plot for key parameters
    try:
        fig = optuna.visualization.plot_contour(
            study,
            params=['lambda_mse', 'lambda_cycle']
        )
        fig.write_html(str(output_dir / 'contour_mse_cycle.html'))
    except:
        pass
    
    # 6. Custom multi-metric comparison
    create_multi_metric_plot(study, output_dir)
    
    logger.info(f"Saved optimization visualizations to {output_dir}")


def create_multi_metric_plot(study: optuna.Study, output_dir: Path):
    """Create custom plot comparing all metrics across trials."""
    trials = study.trials
    
    # Extract data
    trial_numbers = [t.number for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    val_losses = [t.user_attrs.get('val_loss', float('inf')) for t in trials 
                  if t.state == optuna.trial.TrialState.COMPLETE]
    psnrs = [t.user_attrs.get('psnr', 0) for t in trials 
             if t.state == optuna.trial.TrialState.COMPLETE]
    ssims = [t.user_attrs.get('ssim', 0) for t in trials 
             if t.state == optuna.trial.TrialState.COMPLETE]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Validation Loss', 'PSNR (dB)', 'SSIM', 'Best Trial Progression'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Validation Loss
    fig.add_trace(
        go.Scatter(x=trial_numbers, y=val_losses, mode='markers+lines', 
                   name='Val Loss', marker=dict(color='red')),
        row=1, col=1
    )
    
    # PSNR
    fig.add_trace(
        go.Scatter(x=trial_numbers, y=psnrs, mode='markers+lines',
                   name='PSNR', marker=dict(color='green')),
        row=1, col=2
    )
    
    # SSIM
    fig.add_trace(
        go.Scatter(x=trial_numbers, y=ssims, mode='markers+lines',
                   name='SSIM', marker=dict(color='blue')),
        row=2, col=1
    )
    
    # Best trial progression
    best_vals = []
    current_best = float('inf')
    for val in val_losses:
        current_best = min(current_best, val)
        best_vals.append(current_best)
    
    fig.add_trace(
        go.Scatter(x=trial_numbers, y=best_vals, mode='lines',
                   name='Best Val Loss', line=dict(color='purple', width=3)),
        row=2, col=2
    )
    
    fig.update_layout(
        title_text="Optuna Optimization Results - All Metrics",
        showlegend=True,
        height=800
    )
    
    fig.write_html(str(output_dir / 'multi_metric_comparison.html'))


def save_best_hyperparameters(study: optuna.Study, output_path: Path):
    """Save best hyperparameters to JSON file."""
    best_trial = study.best_trial
    
    results = {
        'best_trial_number': best_trial.number,
        'best_val_loss': best_trial.value,
        'best_psnr': best_trial.user_attrs.get('psnr', 'N/A'),
        'best_ssim': best_trial.user_attrs.get('ssim', 'N/A'),
        'best_params': best_trial.params,
        'all_trials_summary': {
            'n_trials': len(study.trials),
            'n_complete': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'n_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved best hyperparameters to {output_path}")
    
    # Print to console
    print("\n" + "="*80)
    print("BEST HYPERPARAMETERS FOUND")
    print("="*80)
    print(f"Trial Number: {best_trial.number}")
    print(f"Validation Loss: {best_trial.value:.6f}")
    print(f"PSNR: {results['best_psnr']}")
    print(f"SSIM: {results['best_ssim']}")
    print("\nParameters:")
    for param, value in best_trial.params.items():
        print(f"  {param}: {value}")
    print("="*80 + "\n")
    
    return results

def custom_collate_fn(batch):
    """Safe collate function that handles varying tensor sizes and mask dictionaries."""
    try:
        # Get expected shapes from first item
        first_item = batch[0]
        expected_source_shape = first_item['source'].shape
        expected_target_shape = first_item['target'].shape
        
        # Pad all tensors to match the maximum dimensions in the batch
        padded_sources = []
        padded_targets = []
        
        for item in batch:
            source = item['source']
            target = item['target']
            
            # Check if padding is needed
            if source.shape != expected_source_shape:
                # Calculate padding needed
                pad_d = expected_source_shape[1] - source.shape[1]
                pad_h = expected_source_shape[2] - source.shape[2]
                pad_w = expected_source_shape[3] - source.shape[3]
                
                # Pad symmetrically
                pad_d_before = pad_d // 2
                pad_d_after = pad_d - pad_d_before
                pad_h_before = pad_h // 2
                pad_h_after = pad_h - pad_h_before
                pad_w_before = pad_w // 2
                pad_w_after = pad_w - pad_w_before
                
                source = torch.nn.functional.pad(
                    source, 
                    (pad_w_before, pad_w_after, pad_h_before, pad_h_after, pad_d_before, pad_d_after),
                    mode='constant', 
                    value=0
                )
                target = torch.nn.functional.pad(
                    target, 
                    (pad_w_before, pad_w_after, pad_h_before, pad_h_after, pad_d_before, pad_d_after),
                    mode='constant', 
                    value=0
                )
            
            padded_sources.append(source.contiguous())
            padded_targets.append(target.contiguous())
        
        # Stack padded tensors
        sources = torch.stack(padded_sources)
        targets = torch.stack(padded_targets)
        
        # Stack phase indices
        source_phase_idx = torch.tensor([item['source_phase_idx'] for item in batch], dtype=torch.long)
        target_phase_idx = torch.tensor([item['target_phase_idx'] for item in batch], dtype=torch.long)
        
        # Handle masks - collect all unique organ names
        all_organs = set()
        for item in batch:
            all_organs.update(item['masks'].keys())
        
        # Create unified mask dict
        batch_masks = {}
        for organ in all_organs:
            organ_masks = []
            for item in batch:
                if organ in item['masks']:
                    mask = item['masks'][organ]
                    # Apply same padding to masks if needed
                    if mask.shape != expected_source_shape:
                        pad_d = expected_source_shape[1] - mask.shape[1]
                        pad_h = expected_source_shape[2] - mask.shape[2]
                        pad_w = expected_source_shape[3] - mask.shape[3]
                        
                        pad_d_before = pad_d // 2
                        pad_d_after = pad_d - pad_d_before
                        pad_h_before = pad_h // 2
                        pad_h_after = pad_h - pad_h_before
                        pad_w_before = pad_w // 2
                        pad_w_after = pad_w - pad_w_before
                        
                        mask = torch.nn.functional.pad(
                            mask, 
                            (pad_w_before, pad_w_after, pad_h_before, pad_h_after, pad_d_before, pad_d_after),
                            mode='constant', 
                            value=0
                        )
                    organ_masks.append(mask.contiguous())
                else:
                    # Create zero mask with correct shape
                    zero_mask = torch.zeros(expected_source_shape, dtype=torch.float32)
                    organ_masks.append(zero_mask)
            batch_masks[organ] = torch.stack(organ_masks)
        
        # Keep string fields as lists
        source_phases = [item['source_phase'] for item in batch]
        target_phases = [item['target_phase'] for item in batch]
        case_ids = [item['case_id'] for item in batch]
        
        return {
            'source': sources,
            'target': targets,
            'source_phase_idx': source_phase_idx,
            'target_phase_idx': target_phase_idx,
            'masks': batch_masks,
            'source_phase': source_phases,
            'target_phase': target_phases,
            'case_id': case_ids
        }
    except Exception as e:
        logger.error(f"Error in custom_collate_fn: {e}")
        logger.error(f"Batch info:")
        for i, item in enumerate(batch):
            logger.error(f"  Item {i}:")
            logger.error(f"    source shape: {item['source'].shape if isinstance(item['source'], torch.Tensor) else 'not a tensor'}")
            logger.error(f"    target shape: {item['target'].shape if isinstance(item['target'], torch.Tensor) else 'not a tensor'}")
            logger.error(f"    masks: {list(item['masks'].keys())}")
        raise
    
def run_optuna_optimization(
    data_splits: Dict,
    base_config: Dict,
    n_trials: int = 50,
    study_name: str = 'ct_phase_optimization',
    subset_fraction: float = 0.33  # NEW: Use 1/3 of data by default
) -> optuna.Study:
    """
    Main function to run Optuna optimization.
    
    Args:
        data_splits: Dictionary with 'train' and 'val' data pairs
        base_config: Base configuration dictionary
        n_trials: Number of Optuna trials
        study_name: Name for the study
        subset_fraction: Fraction of data to use (0.33 = 1/3, 1.0 = all data)
    
    Returns:
        Completed Optuna study
    """
    # Create output directory
    optuna_dir = Path(base_config['output_dir']) / 'optuna_results'
    optuna_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    optuna.logging.enable_propagation()
    optuna.logging.disable_default_handler()
    
    # Subset the data for faster tuning
    if subset_fraction < 1.0:
        import random
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
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = CTPhaseDataset(
        train_subset,
        patch_size=base_config['patch_size'],
        patch_depth=base_config['patch_depth'],
        overlap_ratio=base_config['overlap_ratio'],
        augment=True
    )
    
    val_dataset = CTPhaseDataset(
        val_subset,
        patch_size=base_config['patch_size'],
        patch_depth=base_config['patch_depth'],
        overlap_ratio=0.5,
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=base_config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=base_config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    logger.info(f"Train dataset: {len(train_dataset)} patches")
    logger.info(f"Val dataset: {len(val_dataset)} patches")
    
    # Check for existing study and create or load it
    storage_path = optuna_dir / 'optuna_study.db'
    
    # Check if study exists and get information
    study_info = check_study_status(storage_path, study_name)
    
    # Create or load study
    study = create_study(
        study_name=study_name,
        storage_path=str(storage_path),
        direction='minimize'  # Minimize validation loss
    )
    
    # Calculate remaining trials if study exists
    remaining_trials = n_trials
    if study_info is not None:
        completed_trials = study_info['n_complete_trials']
        remaining_trials = max(0, n_trials - completed_trials)
        logger.info(f"\nResuming Optuna optimization with {remaining_trials} more trials...")
        logger.info(f"Already completed {completed_trials} trials")
    else:
        logger.info(f"\nStarting new Optuna optimization with {n_trials} trials...")
    
    logger.info(f"Results will be saved to: {optuna_dir}")
    
    # Only run optimization if there are trials remaining
    if remaining_trials > 0:
        study.optimize(
            lambda trial: objective(trial, train_loader, val_loader, base_config),
            n_trials=remaining_trials,
            show_progress_bar=True,
            catch=(Exception,)  # Continue even if some trials fail
        )
    else:
        logger.info("All requested trials have already been completed. Skipping optimization.")
    
    # Save results
    logger.info("\nOptimization complete! Saving results...")
    
    # Best hyperparameters
    best_params = save_best_hyperparameters(study, optuna_dir / 'best_hyperparameters.json')
    
    # Visualizations
    visualize_optimization_results(study, optuna_dir / 'plots')
    
    # Save study statistics
    stats = {
        'n_trials': len(study.trials),
        'n_complete_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        'n_pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        'n_failed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
        'best_value': study.best_value,
        'best_trial': study.best_trial.number
    }
    
    with open(optuna_dir / 'optimization_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"\nOptimization Statistics:")
    logger.info(f"  Total trials: {stats['n_trials']}")
    logger.info(f"  Completed: {stats['n_complete_trials']}")
    logger.info(f"  Pruned: {stats['n_pruned_trials']}")
    logger.info(f"  Failed: {stats['n_failed_trials']}")
    logger.info(f"  Best validation loss: {stats['best_value']:.6f}")
    
    return study


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to run Optuna optimization."""
    
    # Base configuration
    base_config = {
        'data_dir': '../ncct_cect/vindr_ds/registered_cases',
        'labels_csv': '../ncct_cect/vindr_ds/labels.csv',
        'output_dir': '../ncct_cect/vindr_ds/optuna_tuning_refined',
        
        'patch_size': (96, 128),
        'patch_depth': 11,
        'overlap_ratio': 0.5,
        
        'batch_size': 8,
        'learning_rate': 2e-4,
        'optuna_epochs': 20,  # Epochs per trial (keep low for faster tuning)
        
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    
    print("="*80)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Trials: 50")
    print(f"  Epochs per trial: {base_config['optuna_epochs']}")
    print(f"  Data subset: 33% (1/3 of full dataset)")
    print(f"  Sampler: TPE (Tree-structured Parzen Estimator)")
    print(f"  Pruner: MedianPruner")
    print(f"  Device: {base_config['device']}")
    print("\n💡 TIP: Using 1/3 of data makes tuning ~3x faster!")
    print("   Hyperparameters found on subset work well on full dataset.")
    print("\nHyperparameters to tune:")
    print("  • lambda_cycle (1.0 - 50.0)")
    print("  • lambda_mse (10.0 - 200.0)")
    print("  • lambda_focal (0.5 - 20.0)")
    print("  • lambda_adv (0.1 - 5.0)")
    print("  • adv_warmup_epochs (3 - 20)")
    print("  • disc_updates_per_gen (1 - 4)")
    print("  • disc_lr_multiplier (1.0 - 4.0)")
    print("  • real_label_smoothing (0.85 - 0.95)")
    print("  • fake_label_smoothing (0.0 - 0.15)")
    print("="*80 + "\n")
    
    # Load data
    logger.info("Loading data...")
    phase_mapping = None
    if Path(base_config['labels_csv']).exists():
        phase_mapping = load_phase_mapping(base_config['labels_csv'])
    
    data_splits = create_data_pairs(
        base_config['data_dir'],
        phase_mapping=phase_mapping
    )
    
    if not data_splits['train'] or not data_splits['val']:
        logger.error("No training or validation data found!")
        return
    
    # Run optimization
    study = run_optuna_optimization(
        data_splits=data_splits,
        base_config=base_config,
        n_trials=50,
        study_name='ct_phase_loss_tuning',
        subset_fraction=0.25  # Use full test dataset since it's small
    )
    
    print("\n✓ Optimization complete!")
    print(f"✓ Results saved to: {base_config['output_dir']}/optuna_results/")
    print("\n⚠️  IMPORTANT: These hyperparameters were tuned on 33% of your data")
    print("   For final training, use the FULL dataset with train_with_optimized_params.py")
    print("\nNext steps:")
    print("1. Check best_hyperparameters.json for optimal values")
    print("2. Review plots in optuna_results/plots/")
    print("3. Run full training: python train_with_optimized_params.py")


if __name__ == "__main__":
    main()