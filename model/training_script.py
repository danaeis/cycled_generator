import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import nibabel as nib
from pathlib import Path
import json
import time
from tqdm import tqdm
import warnings
import os
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


# Import your existing components
from train_phase_generator import (
    CTDataset, create_data_splits_from_directories, load_phase_mapping,
    PhaseConditionalCycleGAN, MetricsEvaluator, save_radiologist_samples,
    ParametricDiscriminator3D
)
# # Import your existing components
# from train_phase_generator import (
#     CTDataset, create_data_splits_from_directories, load_phase_mapping,
#     ParametricCycleGAN3D, CombinedLoss, MetricsEvaluator, save_radiologist_samples
# )

class FullDatasetTrainer:
    """Complete trainer for your full CT dataset"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tensorboard logging
        self.writer = SummaryWriter(self.output_dir / 'logs')
        
        # Initialize models with parametric depth
        print("Initializing parametric models...")
        self.model = ParametricCycleGAN3D(
            input_channels=1, 
            output_channels=1, 
            input_depth=config['patch_depth']  # Use your desired depth
        ).to(self.device)
        
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
            self.opt_gen, start_factor=1.0, end_factor=0.1, total_iters=config['epochs']
        )
        self.scheduler_disc = optim.lr_scheduler.LinearLR(
            self.opt_disc, start_factor=1.0, end_factor=0.1, total_iters=config['epochs']
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
        
        # Save config
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Models initialized on device: {self.device}")
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {
            'gen_total': [], 'disc_total': [], 'cycle_A': [], 'cycle_B': [],
            'mse_A': [], 'mse_B': [], 'adv_A': [], 'adv_B': []
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
                # Map phase names to discriminator keys
                phase_map = {
                    'arterial': 'arterial',
                    'portal': 'portal', 
                    'venous': 'portal',  # Map venous to portal
                    'delayed': 'delayed'
                }
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
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Gen': f"{total_gen_loss.item():.4f}",
                    'Disc': f"{total_disc_loss.item():.4f}",
                    'Phase': f"{batch['source_phase'][0]}->{batch['target_phase'][0]}",
                })
                
                # Free up memory periodically
                if batch_idx % 100 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        return {k: np.mean(v) if v else 0.0 for k, v in epoch_losses.items()}
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        val_losses = []
        all_metrics = {
            'ssim': [], 'psnr': []
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
                    
                    # Calculate metrics (every 10 batches to save time)
                    if batch_idx % 10 == 0:
                        batch_metrics = self.metrics_evaluator.evaluate_batch(
                            outputs['fake_B'], real_B, organ_masks
                        )
                        all_metrics['ssim'].append(batch_metrics['ssim'])
                        all_metrics['psnr'].append(batch_metrics['psnr'])
                        
                        # Update FID calculation
                        self.metrics_evaluator.update_fid(outputs['fake_B'], real_B)
                    
                    progress_bar.set_postfix({
                        'Val Loss': f"{val_loss.item():.4f}",
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
        print(f"Patch Depth: {self.config['patch_depth']}")
        print(f"Training patches per epoch: {len(train_loader)}")
        print(f"Validation patches per epoch: {len(val_loader)}")
        
        start_epoch = self.current_epoch
        
        for epoch in range(start_epoch, self.config['epochs']):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Training
            train_losses = self.train_epoch(train_loader)
            
            # Validation (every N epochs to save time)
            if (epoch + 1) % self.config.get('val_interval', 5) == 0:
                val_metrics = self.validate_epoch(val_loader)
            else:
                val_metrics = {'val_loss': train_losses['gen_total'], 'ssim': 0.0, 'psnr': 0.0, 'fid': 0.0}
            
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
                        self.output_dir / 'radiologist_samples', num_samples=10
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
            if val_metrics['fid'] > 0:
                print(f"  FID: {val_metrics['fid']:.2f}")
            print(f"  Learning rates: Gen={self.opt_gen.param_groups[0]['lr']:.6f}, Disc={self.opt_disc.param_groups[0]['lr']:.6f}")
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

# Create parametric dataset with your desired depth
class ParametricCTDataset(CTDataset):
    def __init__(self, config, data_pairs, patch_size=(64, 64), patch_depth=7, overlap_ratio=0.5, augment=True):
        self.config = config
        self.patch_depth = patch_depth
        super().__init__(data_pairs, patch_size, overlap_ratio, augment)
    
    def _compute_patch_coordinates(self):
        """Modified to use parametric depth"""
        for pair_idx, pair_data in enumerate(self.data_pairs):
            print(f"\nProcessing pair {pair_idx + 1}/{len(self.data_pairs)}")
            
            try:
                source_vol = nib.load(pair_data['source_path']).get_fdata()
                target_vol = nib.load(pair_data['target_path']).get_fdata()
                
                if source_vol.shape != target_vol.shape:
                    print(f"Shape mismatch! Skipping this pair.")
                    continue
                
                depth, height, width = source_vol.shape
                
                # Check minimum requirements for parametric depth
                padding = self.patch_depth // 2
                if depth < self.patch_depth + 2:  # Need some padding
                    print(f"Only {depth} slices, need at least {self.patch_depth + 2}. Skipping.")
                    continue
                
                if height < self.patch_size[0] or width < self.patch_size[1]:
                    print(f"Spatial dimensions too small. Skipping.")
                    continue
                
                # Calculate step size
                step_y = max(1, int(self.patch_size[0] * (1 - self.overlap_ratio)))
                step_x = max(1, int(self.patch_size[1] * (1 - self.overlap_ratio)))
                
                # Generate patch coordinates with parametric depth
                z_range = range(padding, depth - padding)
                y_range = range(0, height - self.patch_size[0] + 1, step_y)
                x_range = range(0, width - self.patch_size[1] + 1, step_x)
                
                pair_patches = 0
                for center_z in z_range:
                    for y_start in y_range:
                        for x_start in x_range:
                            self.patch_coords.append((pair_idx, center_z, y_start, x_start))
                            pair_patches += 1
                
                print(f"Generated {pair_patches} patches for this pair")
                
            except Exception as e:
                print(f"Error processing pair {pair_idx}: {e}")
                continue
    
    def __getitem__(self, idx):
        """Modified to use parametric depth"""
        pair_idx, center_z, y_start, x_start = self.patch_coords[idx]
        pair_data = self.data_pairs[pair_idx]
        
        try:
            source_vol = nib.load(pair_data['source_path']).get_fdata()
            target_vol = nib.load(pair_data['target_path']).get_fdata()
            
            source_vol = self._normalize_intensity(source_vol)
            target_vol = self._normalize_intensity(target_vol)
            
            # Extract patches with parametric depth
            padding = self.patch_depth // 2
            z_start = center_z - padding
            z_end = center_z + padding + 1  # self.patch_depth slices total
            y_end = y_start + self.patch_size[0]
            x_end = x_start + self.patch_size[1]
            
            source_patch = source_vol[z_start:z_end, y_start:y_end, x_start:x_end]
            target_patch = target_vol[z_start:z_end, y_start:y_end, x_start:x_end]
            
            # Verify patch shapes
            expected_shape = (self.patch_depth, self.patch_size[0], self.patch_size[1])
            if source_patch.shape != expected_shape:
                source_patch = self._fix_patch_shape(source_patch, expected_shape)
                target_patch = self._fix_patch_shape(target_patch, expected_shape)
            
            # Load masks (simplified)
            organ_masks = {}
            
            # Apply augmentation
            if self.augment:
                source_patch, target_patch, organ_masks = self._augment_patch(
                    source_patch, target_patch, organ_masks)
            
            # Convert to tensors
            source_patch = torch.from_numpy(source_patch).unsqueeze(0).float()
            target_patch = torch.from_numpy(target_patch).unsqueeze(0).float()
            
            mask_tensors = {}
            for organ, mask in organ_masks.items():
                mask_tensors[organ] = torch.from_numpy(mask).unsqueeze(0).float()
            
            return {
                'source': source_patch,
                'target': target_patch,
                'source_phase': pair_data['source_phase'],
                'target_phase': pair_data['target_phase'],
                'masks': mask_tensors,
                'patch_coords': (pair_idx, center_z, y_start, x_start),
                'case_id': pair_data['case_id']
            }
            
        except Exception as e:
            # Return dummy patch
            dummy_shape = (1, config['patch_depth'], config['patch_size'][0], config['patch_size'][1])
            return {
                'source': torch.zeros(dummy_shape),
                'target': torch.zeros(dummy_shape),
                'source_phase': 'error',
                'target_phase': 'error',
                'masks': {},
                'patch_coords': (pair_idx, 0, 0, 0),
                'case_id': 'error'
            }


def create_dataloaders(config):
    """Create train and validation dataloaders"""
    
    # Load phase mapping if available
    phase_mapping = None
    if Path(config['labels_csv']).exists():
        try:
            phase_mapping = load_phase_mapping(config['labels_csv'])
            print(f"Loaded phase mapping for {len(phase_mapping)} cases")
        except Exception as e:
            print(f"Could not load phase mapping: {e}")
    
    # Create data splits
    data_splits = create_data_splits_from_directories(
        config['data_dir'], 
        phase_mapping=phase_mapping
    )
    # Create datasets with parametric depth
    train_dataset = ParametricCTDataset(
        config=config,
        data_pairs=data_splits['train'], 
        patch_size=config['patch_size'],
        patch_depth=config['patch_depth'],  # Use your desired depth
        overlap_ratio=config['overlap_ratio'],
        augment=True
    )
    
    val_dataset = ParametricCTDataset(
        config=config,
        data_pairs=data_splits['val'],
        patch_size=config['patch_size'],
        patch_depth=config['patch_depth'],  # Same depth for validation
        overlap_ratio=0.5,  # Less overlap for validation
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True,
        drop_last=True  # Ensure consistent batch sizes
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True,
        drop_last=False
    )
    
    print(f"Training dataset: {len(train_dataset)} patches")
    print(f"Validation dataset: {len(val_dataset)} patches")
    print(f"Training batches per epoch: {len(train_loader)}")
    print(f"Validation batches per epoch: {len(val_loader)}")
    
    return train_loader, val_loader


class PhaseConditionalCombinedLoss(nn.Module):
    """Combined loss for phase-conditional training"""
    
    def __init__(self, lambda_cycle=10.0, lambda_mse=100.0, lambda_focal=50.0):
        super().__init__()
        self.lambda_cycle = lambda_cycle
        self.lambda_mse = lambda_mse
        self.lambda_focal = lambda_focal
        
        self.mse_loss = nn.MSELoss()
        self.adversarial_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, generated, real_target, disc_output_fake, disc_output_real=None):
        """Calculate losses for phase-conditional model"""
        losses = {}
        
        # Adversarial loss for generator
        losses['adv'] = self.adversarial_loss(
            disc_output_fake, torch.ones_like(disc_output_fake)
        )
        
        # Direct MSE loss
        losses['mse'] = self.mse_loss(generated, real_target) * self.lambda_mse
        
        # Total generator loss
        losses['total_gen'] = losses['adv'] + losses['mse']
        
        return losses


class PhaseConditionalTrainer:
    """Trainer for phase-conditional CT generation"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Phase mapping
        self.phase_to_idx = {
            'non-contrast': 0,
            'arterial': 1,
            'portal': 2,
            'delayed': 3
        }
        
        print("Initializing phase-conditional model...")
        
        # Initialize phase-conditional model
        self.model = PhaseConditionalCycleGAN(
            input_channels=1,
            output_channels=1
        ).to(self.device)
        
        # Separate discriminator for each phase
        self.discriminators = nn.ModuleDict({
            phase: ParametricDiscriminator3D(
                input_channels=1, 
                input_depth=config.get('patch_depth', 7)
            ).to(self.device)
            for phase in ['arterial', 'portal', 'delayed']
        })
        
        # Optimizers
        self.opt_gen = optim.Adam(
            self.model.generator.parameters(),
            lr=config['learning_rate'], 
            betas=(0.5, 0.999)
        )
        
        self.opt_disc = optim.Adam(
            [p for disc in self.discriminators.values() for p in disc.parameters()],
            lr=config['learning_rate'], 
            betas=(0.5, 0.999)
        )
        
        # Loss functions
        self.generator_loss = PhaseConditionalCombinedLoss().to(self.device)
        self.discriminator_loss = nn.BCEWithLogitsLoss()
        self.metrics_evaluator = MetricsEvaluator(self.device)
        
        # Mixed precision
        if config.get('mixed_precision', True):
            self.scaler = torch.cuda.amp.GradScaler()
        
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        print(f"Model initialized on device: {self.device}")
    
    def _convert_phase_to_idx(self, phase_labels):
        """Convert phase labels (strings) to indices"""
        if isinstance(phase_labels, (list, tuple)):
            return torch.tensor([
                self.phase_to_idx.get(p, 1) for p in phase_labels
            ], device=self.device)
        else:
            return torch.tensor([
                self.phase_to_idx.get(phase_labels, 1)
            ], device=self.device)
    
    def train_epoch(self, train_loader):
        """Train for one epoch with phase conditioning"""
        self.model.train()
        for disc in self.discriminators.values():
            disc.train()
        
        epoch_losses = {
            'gen_total': [], 'disc_total': [], 
            'gen_adv': [], 'gen_mse': []
        }
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Get data
                real_source = batch['source'].to(self.device)  # Non-contrast
                real_target = batch['target'].to(self.device)  # Contrast phase
                target_phases = batch['target_phase']  # Phase names
                
                # Convert phase names to indices
                phase_indices = self._convert_phase_to_idx(target_phases)
                
                batch_size = real_source.size(0)
                
                # ===============================
                # Train Generator
                # ===============================
                self.opt_gen.zero_grad()
                
                if self.config.get('mixed_precision', True):
                    with torch.cuda.amp.autocast():
                        # Generate target phase with phase conditioning
                        generated = self.model(real_source, target_phases)
                        
                        # Get discriminator output for generated images
                        # Use appropriate discriminator based on target phase
                        disc_outputs = []
                        for i, phase in enumerate(target_phases):
                            if phase in self.discriminators:
                                disc_out = self.discriminators[phase](
                                    generated[i:i+1]
                                )
                                disc_outputs.append(disc_out)
                        
                        if disc_outputs:
                            disc_output_fake = torch.cat(disc_outputs, dim=0)
                        else:
                            # Fallback to arterial discriminator
                            disc_output_fake = self.discriminators['arterial'](generated)
                        
                        # Calculate generator losses
                        gen_losses = self.generator_loss(
                            generated, real_target, disc_output_fake
                        )
                        total_gen_loss = gen_losses['total_gen']
                    
                    self.scaler.scale(total_gen_loss).backward()
                    self.scaler.step(self.opt_gen)
                    self.scaler.update()
                else:
                    # Generate with phase conditioning
                    generated = self.model(real_source, target_phases)
                    
                    # Discriminator outputs
                    disc_outputs = []
                    for i, phase in enumerate(target_phases):
                        if phase in self.discriminators:
                            disc_out = self.discriminators[phase](
                                generated[i:i+1]
                            )
                            disc_outputs.append(disc_out)
                    
                    if disc_outputs:
                        disc_output_fake = torch.cat(disc_outputs, dim=0)
                    else:
                        disc_output_fake = self.discriminators['arterial'](generated)
                    
                    gen_losses = self.generator_loss(
                        generated, real_target, disc_output_fake
                    )
                    total_gen_loss = gen_losses['total_gen']
                    
                    total_gen_loss.backward()
                    self.opt_gen.step()
                
                # ===============================
                # Train Discriminators
                # ===============================
                self.opt_disc.zero_grad()
                phase_map = {
                    'arterial': 'arterial',
                    'portal': 'portal', 
                    'venous': 'portal',  # Map venous to portal
                    'delayed': 'delayed'
                }

                if self.config.get('mixed_precision', True):
                    with torch.cuda.amp.autocast():
                        disc_loss_total = 0
                        
                        for i, phase in enumerate(target_phases):
                            # Map phase name to discriminator key
                            disc_key = phase_map.get(phase, 'arterial')  # Default to arterial
                            
                            # if phase in self.discriminators:
                            #     disc = self.discriminators[phase]
                            if disc_key in self.discriminators:
                                disc = self.discriminators[disc_key]
                                
                                # Real images
                                disc_real = disc(real_target[i:i+1])
                                loss_real = self.discriminator_loss(
                                    disc_real, torch.ones_like(disc_real)
                                )
                                
                                # Fake images (detached)
                                disc_fake = disc(generated[i:i+1].detach())
                                loss_fake = self.discriminator_loss(
                                    disc_fake, torch.zeros_like(disc_fake)
                                )
                                
                                disc_loss_total += (loss_real + loss_fake) * 0.5
                        
                        # disc_loss_total = disc_loss_total / len(target_phases)
                        if disc_loss_total:  # Only if we have losses
                            disc_loss_total = torch.stack(disc_loss_total).mean()
                            self.scaler.scale(disc_loss_total).backward()
                            self.scaler.step(self.opt_disc)
                        
                        self.scaler.update()
                    # self.scaler.scale(disc_loss_total).backward()
                    # self.scaler.step(self.opt_disc)
                    # self.scaler.update()
                else:
                    disc_loss_total = 0
                    
                    for i, phase in enumerate(target_phases):
                        if phase in self.discriminators:
                            disc = self.discriminators[phase]
                            
                            disc_real = disc(real_target[i:i+1])
                            loss_real = self.discriminator_loss(
                                disc_real, torch.ones_like(disc_real)
                            )
                            
                            disc_fake = disc(generated[i:i+1].detach())
                            loss_fake = self.discriminator_loss(
                                disc_fake, torch.zeros_like(disc_fake)
                            )
                            
                            disc_loss_total += (loss_real + loss_fake) * 0.5
                    
                    disc_loss_total = disc_loss_total / len(target_phases)
                    disc_loss_total.backward()
                    self.opt_disc.step()
                
                # Record losses
                epoch_losses['gen_total'].append(total_gen_loss.item())
                epoch_losses['disc_total'].append(disc_loss_total.item())
                epoch_losses['gen_adv'].append(gen_losses['adv'].item())
                epoch_losses['gen_mse'].append(gen_losses['mse'].item())
                
                # Update progress
                progress_bar.set_postfix({
                    'Gen': f"{total_gen_loss.item():.4f}",
                    'Disc': f"{disc_loss_total.item():.4f}",
                    'Phase': target_phases[0] if target_phases else 'N/A'
                })
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return {k: np.mean(v) if v else 0.0 for k, v in epoch_losses.items()}
    
    def validate_epoch(self, val_loader):
        """Validate with phase conditioning"""
        self.model.eval()
        val_losses = []
        all_metrics = {'ssim': [], 'psnr': []}
        
        progress_bar = tqdm(val_loader, desc="Validation")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                try:
                    real_source = batch['source'].to(self.device)
                    real_target = batch['target'].to(self.device)
                    target_phases = batch['target_phase']
                    
                    # Generate with phase conditioning
                    if self.config.get('mixed_precision', True):
                        with torch.cuda.amp.autocast():
                            generated = self.model(real_source, target_phases)
                    else:
                        generated = self.model(real_source, target_phases)
                    
                    # Calculate validation loss
                    val_loss = torch.nn.functional.mse_loss(generated, real_target)
                    val_losses.append(val_loss.item())
                    
                    # Calculate metrics (every 10 batches)
                    if batch_idx % 10 == 0:
                        batch_metrics = self.metrics_evaluator.evaluate_batch(
                            generated, real_target, None
                        )
                        all_metrics['ssim'].append(batch_metrics['ssim'])
                        all_metrics['psnr'].append(batch_metrics['psnr'])
                    
                    progress_bar.set_postfix({
                        'Val Loss': f"{val_loss.item():.4f}"
                    })
                    
                except Exception as e:
                    print(f"Error in validation batch {batch_idx}: {e}")
                    continue
        
        return {
            'val_loss': np.mean(val_losses) if val_losses else float('inf'),
            'ssim': np.mean(all_metrics['ssim']) if all_metrics['ssim'] else 0.0,
            'psnr': np.mean(all_metrics['psnr']) if all_metrics['psnr'] else 0.0,
        }
    
    def save_checkpoint(self, epoch, val_metrics, is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'discriminators_state_dict': {
                k: v.state_dict() for k, v in self.discriminators.items()
            },
            'optimizer_gen_state_dict': self.opt_gen.state_dict(),
            'optimizer_disc_state_dict': self.opt_disc.state_dict(),
            'val_metrics': val_metrics,
            'config': self.config,
            'phase_to_idx': self.phase_to_idx
        }
        
        if self.config.get('mixed_precision'):
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.output_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"New best model saved! Val loss: {val_metrics['val_loss']:.4f}")
    
    def train(self, train_loader, val_loader, epochs):
        """Main training loop"""
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            
            # Training
            train_losses = self.train_epoch(train_loader)
            
            # Validation
            if (epoch + 1) % self.config.get('val_interval', 5) == 0:
                val_metrics = self.validate_epoch(val_loader)
            else:
                val_metrics = {'val_loss': train_losses['gen_total'], 'ssim': 0.0, 'psnr': 0.0}
            
            # Save checkpoint
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
            
            self.save_checkpoint(epoch + 1, val_metrics, is_best)
            
            # Print summary
            print(f"\nEpoch {epoch + 1}/{epochs} Summary:")
            print(f"  Train - Gen: {train_losses['gen_total']:.4f}, Disc: {train_losses['disc_total']:.4f}")
            print(f"  Val - Loss: {val_metrics['val_loss']:.4f}, SSIM: {val_metrics['ssim']:.3f}")
            print("-" * 60)


def main():
    """Main training function"""
    
    config = {
        # Data paths
        'data_dir': '../ncct_cect/vindr_ds/test_registered_cases',
        'labels_csv': '../ncct_cect/vindr_ds/labels.csv',
        'output_dir': '../ncct_cect/vindr_ds/phase_conditional_training',
        
        # Model parameters
        'patch_size': (64, 64),
        'patch_depth': 7,
        
        # Training parameters
        'batch_size': 4,
        'learning_rate': 2e-4,
        'epochs': 100,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'mixed_precision': True,
        
        # Data parameters
        'overlap_ratio': 0.75,
        'val_interval': 5,
    }
    
    print("=" * 80)
    print("PHASE-CONDITIONAL CT GENERATION TRAINING")
    print("=" * 80)
    
    # Load phase mapping
    phase_mapping = None
    if Path(config['labels_csv']).exists():
        try:
            phase_mapping = load_phase_mapping(config['labels_csv'])
            print(f"Loaded phase mapping for {len(phase_mapping)} cases")
        except Exception as e:
            print(f"Could not load phase mapping: {e}")
    
    # Create data splits
    data_splits = create_data_splits_from_directories(
        config['data_dir'], 
        phase_mapping=phase_mapping
    )
    
    # Create datasets
    train_dataset = ParametricCTDataset(
        config,
        data_splits['train'], 
        patch_size=config['patch_size'],
        overlap_ratio=config['overlap_ratio'],
        augment=True
    )
    
    val_dataset = ParametricCTDataset(
        config,
        data_splits['val'],
        patch_size=config['patch_size'], 
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
    
    print(f"Training patches: {len(train_dataset)}")
    print(f"Validation patches: {len(val_dataset)}")
    
    # Initialize trainer
    trainer = PhaseConditionalTrainer(config)
    
    # Start training
    try:
        trainer.train(train_loader, val_loader, config['epochs'])
        print("Training completed successfully!")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.save_checkpoint(trainer.current_epoch, {'val_loss': float('inf')})
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()