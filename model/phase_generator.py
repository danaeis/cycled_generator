import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import json
import time
from tqdm import tqdm
import warnings
import os
warnings.filterwarnings('ignore')

# Import the updated phase generator components
from train_phase_generator import (
    CTDataset, create_data_splits_from_directories, load_phase_mapping,
    CycleGAN3D, CombinedLoss, MetricsEvaluator, save_radiologist_samples
)

class CTPhaseTrainer:
    """Complete trainer for CT phase generation"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tensorboard logging
        self.writer = SummaryWriter(self.output_dir / 'logs')
        
        # Initialize models
        print("Initializing models...")
        self.model = CycleGAN3D().to(self.device)
        
        # Initialize optimizers
        self.opt_gen = optim.Adam(
            list(self.model.gen_AB.parameters()) + list(self.model.gen_BA.parameters()),
            lr=config['learning_rate'], betas=(0.5, 0.999)
        )
        self.opt_disc = optim.Adam(
            list(self.model.disc_A.parameters()) + list(self.model.disc_B.parameters()),
            lr=config['learning_rate'], betas=(0.5, 0.999)
        )
        
        # Learning rate schedulers
        self.scheduler_gen = optim.lr_scheduler.LinearLR(
            self.opt_gen, start_factor=1.0, end_factor=0.0, total_iters=config['epochs']
        )
        self.scheduler_disc = optim.lr_scheduler.LinearLR(
            self.opt_disc, start_factor=1.0, end_factor=0.0, total_iters=config['epochs']
        )
        
        # Initialize loss functions and metrics
        self.combined_loss = CombinedLoss().to(self.device)
        self.discriminator_loss = torch.nn.BCEWithLogitsLoss()
        self.metrics_evaluator = MetricsEvaluator(self.device)
        
        # Mixed precision training
        if config['mixed_precision']:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        print(f"Models initialized on device: {self.device}")
        print(f"Generator parameters: {sum(p.numel() for p in self.model.gen_AB.parameters() if p.requires_grad):,}")
        print(f"Discriminator parameters: {sum(p.numel() for p in self.model.disc_A.parameters() if p.requires_grad):,}")
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {
            'gen_total': [], 'disc_total': [], 'cycle_A': [], 'cycle_B': [],
            'mse_A': [], 'mse_B': [], 'focal_A': [], 'focal_B': [],
            'adv_A': [], 'adv_B': []
        }
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move data to device
                real_A = batch['source'].to(self.device)
                real_B = batch['target'].to(self.device)
                organ_masks = {k: v.to(self.device) for k, v in batch['masks'].items()}
                
                batch_size = real_A.size(0)
                
                # ===============================
                # Train Generators
                # ===============================
                self.opt_gen.zero_grad()
                
                if self.config['mixed_precision']:
                    with torch.cuda.amp.autocast():
                        # Generate fake images
                        outputs = self.model(real_A, real_B)
                        
                        # Discriminator outputs for generator training
                        disc_fake_A = self.model.disc_A(outputs['fake_A'])
                        disc_fake_B = self.model.disc_B(outputs['fake_B'])
                        
                        disc_outputs = {
                            'disc_fake_A': disc_fake_A,
                            'disc_fake_B': disc_fake_B
                        }
                        
                        # Calculate generator losses
                        gen_losses = self.combined_loss(outputs, real_A, real_B, organ_masks, disc_outputs)
                        total_gen_loss = gen_losses['total_gen']
                    
                    # Backward pass with mixed precision
                    self.scaler.scale(total_gen_loss).backward()
                    self.scaler.step(self.opt_gen)
                    self.scaler.update()
                else:
                    # Generate fake images
                    outputs = self.model(real_A, real_B)
                    
                    # Discriminator outputs for generator training
                    disc_fake_A = self.model.disc_A(outputs['fake_A'])
                    disc_fake_B = self.model.disc_B(outputs['fake_B'])
                    
                    disc_outputs = {
                        'disc_fake_A': disc_fake_A,
                        'disc_fake_B': disc_fake_B
                    }
                    
                    # Calculate generator losses
                    gen_losses = self.combined_loss(outputs, real_A, real_B, organ_masks, disc_outputs)
                    total_gen_loss = gen_losses['total_gen']
                    
                    # Backward pass
                    total_gen_loss.backward()
                    self.opt_gen.step()
                
                # ===============================
                # Train Discriminators
                # ===============================
                self.opt_disc.zero_grad()
                
                if self.config['mixed_precision']:
                    with torch.cuda.amp.autocast():
                        # Discriminator A (distinguishes real A from fake A)
                        disc_real_A = self.model.disc_A(real_A)
                        disc_fake_A_detached = self.model.disc_A(outputs['fake_A'].detach())
                        
                        loss_disc_A_real = self.discriminator_loss(
                            disc_real_A, torch.ones_like(disc_real_A)
                        )
                        loss_disc_A_fake = self.discriminator_loss(
                            disc_fake_A_detached, torch.zeros_like(disc_fake_A_detached)
                        )
                        loss_disc_A = (loss_disc_A_real + loss_disc_A_fake) * 0.5
                        
                        # Discriminator B (distinguishes real B from fake B)
                        disc_real_B = self.model.disc_B(real_B)
                        disc_fake_B_detached = self.model.disc_B(outputs['fake_B'].detach())
                        
                        loss_disc_B_real = self.discriminator_loss(
                            disc_real_B, torch.ones_like(disc_real_B)
                        )
                        loss_disc_B_fake = self.discriminator_loss(
                            disc_fake_B_detached, torch.zeros_like(disc_fake_B_detached)
                        )
                        loss_disc_B = (loss_disc_B_real + loss_disc_B_fake) * 0.5
                        
                        total_disc_loss = loss_disc_A + loss_disc_B
                    
                    # Backward pass with mixed precision
                    self.scaler.scale(total_disc_loss).backward()
                    self.scaler.step(self.opt_disc)
                    self.scaler.update()
                else:
                    # Discriminator A (distinguishes real A from fake A)
                    disc_real_A = self.model.disc_A(real_A)
                    disc_fake_A_detached = self.model.disc_A(outputs['fake_A'].detach())
                    
                    loss_disc_A_real = self.discriminator_loss(
                        disc_real_A, torch.ones_like(disc_real_A)
                    )
                    loss_disc_A_fake = self.discriminator_loss(
                        disc_fake_A_detached, torch.zeros_like(disc_fake_A_detached)
                    )
                    loss_disc_A = (loss_disc_A_real + loss_disc_A_fake) * 0.5
                    
                    # Discriminator B (distinguishes real B from fake B)
                    disc_real_B = self.model.disc_B(real_B)
                    disc_fake_B_detached = self.model.disc_B(outputs['fake_B'].detach())
                    
                    loss_disc_B_real = self.discriminator_loss(
                        disc_real_B, torch.ones_like(disc_real_B)
                    )
                    loss_disc_B_fake = self.discriminator_loss(
                        disc_fake_B_detached, torch.zeros_like(disc_fake_B_detached)
                    )
                    loss_disc_B = (loss_disc_B_real + loss_disc_B_fake) * 0.5
                    
                    total_disc_loss = loss_disc_A + loss_disc_B
                    
                    # Backward pass
                    total_disc_loss.backward()
                    self.opt_disc.step()
                
                # Record losses
                epoch_losses['gen_total'].append(total_gen_loss.item())
                epoch_losses['disc_total'].append(total_disc_loss.item())
                epoch_losses['cycle_A'].append(gen_losses['cycle_A'].item())
                epoch_losses['cycle_B'].append(gen_losses['cycle_B'].item())
                epoch_losses['mse_A'].append(gen_losses['mse_A'].item())
                epoch_losses['mse_B'].append(gen_losses['mse_B'].item())
                epoch_losses['adv_A'].append(gen_losses['adv_A'].item())
                epoch_losses['adv_B'].append(gen_losses['adv_B'].item())
                
                if 'focal_A' in gen_losses:
                    epoch_losses['focal_A'].append(gen_losses['focal_A'].item())
                    epoch_losses['focal_B'].append(gen_losses['focal_B'].item())
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Gen': f"{total_gen_loss.item():.4f}",
                    'Disc': f"{total_disc_loss.item():.4f}",
                    'Phase': batch['target_phase'][0] if len(batch['target_phase']) > 0 else 'unknown'
                })
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        return {k: np.mean(v) if v else 0.0 for k, v in epoch_losses.items()}
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        val_losses = []
        all_metrics = {
            'ssim': [], 'psnr': [], 'organ_ssim': {org: [] for org in ['liver', 'kidney_right', 'spleen']}
        }
        
        progress_bar = tqdm(val_loader, desc="Validation")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                try:
                    real_A = batch['source'].to(self.device)
                    real_B = batch['target'].to(self.device)
                    organ_masks = {k: v.to(self.device) for k, v in batch['masks'].items()}
                    
                    # Generate fake images
                    if self.config['mixed_precision']:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(real_A, real_B)
                    else:
                        outputs = self.model(real_A, real_B)
                    
                    # Calculate validation loss (only cycle consistency + MSE)
                    cycle_loss = torch.nn.functional.mse_loss(outputs['cycle_A'], real_A) + \
                               torch.nn.functional.mse_loss(outputs['cycle_B'], real_B)
                    mse_loss = torch.nn.functional.mse_loss(outputs['fake_B'], real_B)
                    val_loss = cycle_loss + mse_loss
                    
                    val_losses.append(val_loss.item())
                    
                    # Calculate metrics
                    batch_metrics = self.metrics_evaluator.evaluate_batch(
                        outputs['fake_B'], real_B, organ_masks
                    )
                    
                    all_metrics['ssim'].append(batch_metrics['ssim'])
                    all_metrics['psnr'].append(batch_metrics['psnr'])
                    
                    if 'organ_ssim' in batch_metrics:
                        for organ, ssim_val in batch_metrics['organ_ssim'].items():
                            if organ in all_metrics['organ_ssim']:
                                all_metrics['organ_ssim'][organ].append(ssim_val)
                    
                    # Update FID calculation
                    self.metrics_evaluator.update_fid(outputs['fake_B'], real_B)
                    
                    progress_bar.set_postfix({
                        'Val Loss': f"{val_loss.item():.4f}",
                        'SSIM': f"{batch_metrics['ssim']:.3f}",
                        'PSNR': f"{batch_metrics['psnr']:.1f}"
                    })
                    
                except Exception as e:
                    print(f"Error in validation batch {batch_idx}: {e}")
                    continue
        
        # Compute final metrics
        final_metrics = {
            'val_loss': np.mean(val_losses) if val_losses else float('inf'),
            'ssim': np.mean(all_metrics['ssim']) if all_metrics['ssim'] else 0.0,
            'psnr': np.mean(all_metrics['psnr']) if all_metrics['psnr'] else 0.0,
        }
        
        try:
            final_metrics['fid'] = self.metrics_evaluator.compute_fid()
        except:
            final_metrics['fid'] = 0.0
        
        # Add organ-specific metrics
        for organ, values in all_metrics['organ_ssim'].items():
            if values:
                final_metrics[f'ssim_{organ}'] = np.mean(values)
        
        return final_metrics
    
    def save_checkpoint(self, epoch, val_metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_gen_state_dict': self.opt_gen.state_dict(),
            'optimizer_disc_state_dict': self.opt_disc.state_dict(),
            'scheduler_gen_state_dict': self.scheduler_gen.state_dict(),
            'scheduler_disc_state_dict': self.scheduler_disc.state_dict(),
            'val_metrics': val_metrics,
            'config': self.config
        }
        
        if self.config['mixed_precision']:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"New best model saved with validation loss: {val_metrics['val_loss']:.4f}")
        
        # Keep only last 3 checkpoints to save space
        checkpoints = sorted(self.output_dir.glob('checkpoint_epoch_*.pth'))
        if len(checkpoints) > 3:
            for old_checkpoint in checkpoints[:-3]:
                old_checkpoint.unlink()
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.opt_gen.load_state_dict(checkpoint['optimizer_gen_state_dict'])
        self.opt_disc.load_state_dict(checkpoint['optimizer_disc_state_dict'])
        self.scheduler_gen.load_state_dict(checkpoint['scheduler_gen_state_dict'])
        self.scheduler_disc.load_state_dict(checkpoint['scheduler_disc_state_dict'])
        
        if self.config['mixed_precision'] and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
        
        return checkpoint['val_metrics']
    
    def train(self, train_loader, val_loader):
        """Main training loop"""
        print(f"Starting training for {self.config['epochs']} epochs...")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.config['mixed_precision']}")
        print(f"Batch Size: {self.config['batch_size']}")
        
        start_epoch = self.current_epoch
        
        for epoch in range(start_epoch, self.config['epochs']):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Training
            train_losses = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate_epoch(val_loader)
            
            # Update learning rates
            self.scheduler_gen.step()
            self.scheduler_disc.step()
            
            # Log metrics
            self.log_metrics(train_losses, val_metrics, epoch)
            
            # Save radiologist samples
            if (epoch + 1) % self.config['save_interval'] == 0:
                print(f"Saving radiologist samples for epoch {epoch + 1}")
                try:
                    save_radiologist_samples(
                        self.model, val_loader, epoch + 1, 
                        self.output_dir / 'radiologist_samples'
                    )
                except Exception as e:
                    print(f"Error saving samples: {e}")
            
            # Save checkpoint
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
            
            self.save_checkpoint(epoch + 1, val_metrics, is_best)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch {epoch + 1}/{self.config['epochs']} Summary:")
            print(f"  Time: {epoch_time:.1f}s")
            print(f"  Train Loss: Gen={train_losses['gen_total']:.4f}, Disc={train_losses['disc_total']:.4f}")
            print(f"  Val Metrics: Loss={val_metrics['val_loss']:.4f}, SSIM={val_metrics['ssim']:.3f}, PSNR={val_metrics['psnr']:.1f}")
            if 'fid' in val_metrics:
                print(f"  FID: {val_metrics['fid']:.2f}")
            print("-" * 80)
        
        print("Training completed!")
        self.writer.close()
    
    def log_metrics(self, train_losses, val_metrics, epoch):
        """Log metrics to tensorboard"""
        # Training losses
        for loss_name, loss_value in train_losses.items():
            self.writer.add_scalar(f'Train/{loss_name}', loss_value, epoch)
        
        # Validation metrics
        for metric_name, metric_value in val_metrics.items():
            self.writer.add_scalar(f'Validation/{metric_name}', metric_value, epoch)
        
        # Learning rates
        self.writer.add_scalar('Learning_Rate/Generator', self.opt_gen.param_groups[0]['lr'], epoch)
        self.writer.add_scalar('Learning_Rate/Discriminator', self.opt_disc.param_groups[0]['lr'], epoch)

def test_data_loading(config):
    """Test data loading before training"""
    print("=" * 60)
    print("TESTING DATA LOADING")
    print("=" * 60)
    
    # Load phase mapping if available
    phase_mapping = None
    if Path(config['labels_csv']).exists():
        try:
            phase_mapping = load_phase_mapping(config['labels_csv'])
            print(f"✓ Loaded phase mapping for {len(phase_mapping)} cases")
        except Exception as e:
            print(f"⚠ Could not load phase mapping: {e}")
            print("  Will infer phases from filenames")
    else:
        print(f"⚠ Labels CSV not found: {config['labels_csv']}")
        print("  Will infer phases from filenames")
    
    # Create data splits
    try:
        data_splits = create_data_splits_from_directories(
            config['data_dir'], 
            phase_mapping=phase_mapping
        )
        print(f"✓ Data splits created successfully")
    except Exception as e:
        print(f"✗ Error creating data splits: {e}")
        return False
    
    # Test dataset creation
    try:
        train_dataset = CTDataset(
            data_splits['train'][:2],  # Test with first 2 pairs only
            patch_size=config['patch_size'],
            overlap_ratio=config['overlap_ratio'],
            augment=True
        )
        print(f"✓ Training dataset created: {len(train_dataset)} patches")
        
        # Test loading a sample
        sample = train_dataset[0]
        print(f"✓ Sample loaded successfully:")
        print(f"  Source shape: {sample['source'].shape}")
        print(f"  Target shape: {sample['target'].shape}")
        print(f"  Source phase: {sample['source_phase']}")
        print(f"  Target phase: {sample['target_phase']}")
        print(f"  Available masks: {list(sample['masks'].keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main execution function with step-by-step process"""
    
    # Configuration
    config = {
        # Data paths - UPDATE THESE FOR YOUR SETUP
        'data_dir': '../ncct_cect/vindr_ds/test_registered_cases',
        'labels_csv': '../ncct_cect/vindr_ds/labels.csv',
        'output_dir': '../ncct_cect/vindr_ds/predicted',
        
        # Training parameters
        'batch_size': 2,  # Start small for testing
        'learning_rate': 2e-4,
        'epochs': 5,  # Start with few epochs for testing
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'mixed_precision': True,
        
        # Data parameters
        'patch_size': (100, 100),
        'overlap_ratio': 0.5,
        'save_interval': 1,  # Save samples every 2 epochs for testing
        
        # Checkpoint
        'resume_from': None,  # Path to checkpoint to resume from
    }
    
    print("=" * 60)
    print("CT PHASE GENERATION TRAINING PIPELINE")
    print("=" * 60)
    print(f"Device: {config['device']}")
    print(f"Data directory: {config['data_dir']}")
    print(f"Output directory: {config['output_dir']}")
    
    # Step 1: Test data loading
    if not test_data_loading(config):
        print("✗ Data loading test failed. Please check your data paths and structure.")
        return
    
    # Step 2: Create full datasets
    print("\n" + "=" * 60)
    print("CREATING FULL DATASETS")
    print("=" * 60)
    
    # Load phase mapping
    phase_mapping = None
    if Path(config['labels_csv']).exists():
        try:
            phase_mapping = load_phase_mapping(config['labels_csv'])
        except Exception as e:
            print(f"Warning: Could not load phase mapping: {e}")
    
    # Create data splits
    data_splits = create_data_splits_from_directories(
        config['data_dir'], 
        phase_mapping=phase_mapping
    )
    
    # Create datasets
    train_dataset = CTDataset(
        data_splits['train'], 
        patch_size=config['patch_size'],
        overlap_ratio=config['overlap_ratio'],
        augment=True
    )
    
    val_dataset = CTDataset(
        data_splits['val'],
        patch_size=config['patch_size'], 
        overlap_ratio=0.25,  # Less overlap for validation
        augment=False
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=2,  # Reduced for stability
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True,
        persistent_workers=True
    )
    
    print(f"✓ Training dataset: {len(train_dataset)} patches")
    print(f"✓ Validation dataset: {len(val_dataset)} patches")
    print(f"✓ Training batches per epoch: {len(train_loader)}")
    print(f"✓ Validation batches per epoch: {len(val_loader)}")
    
    # Step 3: Initialize trainer
    print("\n" + "=" * 60)
    print("INITIALIZING TRAINER")
    print("=" * 60)
    
    trainer = CTPhaseTrainer(config)
    
    # Step 4: Resume from checkpoint if specified
    if config['resume_from'] and Path(config['resume_from']).exists():
        print(f"Resuming from checkpoint: {config['resume_from']}")
        trainer.load_checkpoint(config['resume_from'])
    
    # Step 5: Start training
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    try:
        trainer.train(train_loader, val_loader)
        print("✓ Training completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
        # Save current state
        trainer.save_checkpoint(trainer.current_epoch, {'val_loss': float('inf')}, is_best=False)
        print("✓ Current state saved")
    except Exception as e:
        print(f"✗ Training failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()