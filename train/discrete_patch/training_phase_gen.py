"""
Optimized CT Phase Generation Training Pipeline
================================================
Features:
- Preprocessed discrete data (0-255) - no re-normalization
- Mask-weighted loss (higher penalty on high-intensity regions)
- Phase-conditional generation (input phase -> target phase)
- Automatic patch saving after each epoch
- Memory optimization with mixed precision
- Gradient accumulation and scheduling
- Volume caching for fast loading

Architecture:
- 3D U-Net Generator with phase embeddings
- Optional discriminator for adversarial training
- Weighted MSE loss with spatial importance masks
- Cycle consistency (optional)
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
from collections import OrderedDict
import gc

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class WeightedMSELoss(nn.Module):
    """
    MSE Loss with spatial weighting for important regions.
    High-intensity pixels (vessels, enhanced organs) get higher penalty.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted image [B, 1, D, H, W]
            target: Target image [B, 1, D, H, W]
            weight: Weight mask [B, 1, D, H, W] (higher = more important)
        
        Returns:
            Weighted MSE loss
        """
        squared_diff = (pred - target) ** 2
        weighted_loss = squared_diff * weight
        return weighted_loss.mean()


class FocalLoss(nn.Module):
    """
    Focal loss for important regions.
    Focuses training on hard examples and important anatomical regions.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred: Predicted image [B, 1, D, H, W]
            target: Target image [B, 1, D, H, W]
            mask: Optional importance mask [B, 1, D, H, W]
        """
        # Compute pixel-wise error
        diff = torch.abs(pred - target)
        
        # Focal weighting: focus more on large errors
        focal_weight = torch.pow(diff, self.gamma)
        
        # MSE with focal weighting
        loss = focal_weight * (pred - target) ** 2
        
        # Apply importance mask if provided
        if mask is not None:
            loss = loss * mask
        
        return loss.mean()


# ============================================================================
# NETWORK ARCHITECTURES
# ============================================================================

class PhaseGenerator(nn.Module):
    """
    3D U-Net generator with phase conditioning.
    
    Features:
    - Phase embeddings guide the transformation
    - Skip connections for detail preservation
    - Dropout for regularization
    - Supports conditioning on both source and target phases
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        dropout: float = 0.3,
        use_phase_conditioning: bool = True,
        num_phases: int = 4,
        phase_embedding_dim: int = 64
    ):
        super().__init__()
        
        self.use_phase_conditioning = use_phase_conditioning
        
        # Phase embedding
        if self.use_phase_conditioning:
            self.phase_embedding = nn.Embedding(num_phases, phase_embedding_dim)
            # Project phase embedding to spatial features
            self.phase_projection = nn.Sequential(
                nn.Linear(phase_embedding_dim, base_channels * 8 * 8 * 8),
                nn.ReLU(inplace=True)
            )
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, base_channels, dropout)
        self.enc2 = self._conv_block(base_channels, base_channels * 2, dropout)
        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4, dropout)
        
        # Bottleneck
        self.bottleneck = self._conv_block(base_channels * 4, base_channels * 8, dropout)
        
        # Decoder
        self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(base_channels * 8, base_channels * 4, dropout)
        
        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(base_channels * 4, base_channels * 2, dropout)
        
        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(base_channels * 2, base_channels, dropout)
        
        # Output
        self.out_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)
        
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
    
    def _conv_block(self, in_ch: int, out_ch: int, dropout: float) -> nn.Sequential:
        """3D convolutional block with batch norm and dropout."""
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def _match_size(self, x: torch.Tensor, target_size: tuple) -> torch.Tensor:
        """Match spatial dimensions using interpolation if needed."""
        if x.shape[2:] != target_size:
            x = F.interpolate(x, size=target_size, mode='trilinear', align_corners=False)
        return x
        
    def forward(
            self, 
            x: torch.Tensor, 
            source_phase: Optional[torch.Tensor] = None, 
            target_phase: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        """
        Forward pass with optional phase conditioning.
        
        Args:
            x: Input image [B, 1, D, H, W]
            source_phase: Source phase indices [B] (optional)
            target_phase: Target phase indices [B] (optional)
        
        Returns:
            Generated image [B, 1, D, H, W]
        """
        B, C, D, H, W = x.shape
        
        # Encoder path
        e1 = self.enc1(x)  # [B, 32, D, H, W]
        e1_size = e1.shape[2:]
        e2 = self.enc2(self.pool(e1))  # [B, 64, D/2, H/2, W/2]
        e2_size = e2.shape[2:]
        e3 = self.enc3(self.pool(e2))  # [B, 128, D/4, H/4, W/4]
        e3_size = e3.shape[2:]
        
        # Bottleneck
        b = self.bottleneck(self.pool(e3))  # [B, 256, D/8, H/8, W/8]
        
        # Phase conditioning: inject phase information into bottleneck
        if self.use_phase_conditioning and target_phase is not None:
            # Get phase embedding [B, phase_dim]
            phase_emb = self.phase_embedding(target_phase)
            
            # Project to spatial features [B, base_channels*4, 8, 8]
            phase_feat = self.phase_projection(phase_emb).view(
                B, b.size(1), 8, 8
            )
            
            # Upsample to match bottleneck spatial size
            phase_map = F.interpolate(
                phase_feat.unsqueeze(2),  # Add depth dimension
                size=(b.size(2), b.size(3), b.size(4)),
                mode='trilinear',
                align_corners=False
            )
            
            # Add phase information to bottleneck
            b = b + phase_map
        
        # Decoder path with skip connections
        d3 = self.up3(b)
        d3 = self._match_size(d3, e3_size)  # <-- Add this
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = self._match_size(d2, e2_size)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = self._match_size(d1, e1_size)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # Output
        out = self.out_conv(d1)
        
        # Sigmoid to keep output in [0, 1] range
        return torch.sigmoid(out)

class PhaseGenerator2D(nn.Module):
    """
    2D U-Net generator with phase conditioning (for patch_depth=1).
    
    Uses 2D convolutions and pooling instead of 3D.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        dropout: float = 0.3,
        use_phase_conditioning: bool = True,
        num_phases: int = 4,
        phase_embedding_dim: int = 64
    ):
        super().__init__()
        
        self.use_phase_conditioning = use_phase_conditioning
        
        # Phase embedding
        if self.use_phase_conditioning:
            self.phase_embedding = nn.Embedding(num_phases, phase_embedding_dim)
            self.phase_projection = nn.Sequential(
                nn.Linear(phase_embedding_dim, base_channels * 8 * 8 * 8),
                nn.ReLU(inplace=True)
            )
        
        # Encoder (2D)
        self.enc1 = self._conv_block_2d(in_channels, base_channels, dropout)
        self.enc2 = self._conv_block_2d(base_channels, base_channels * 2, dropout)
        self.enc3 = self._conv_block_2d(base_channels * 2, base_channels * 4, dropout)
        
        # Bottleneck
        self.bottleneck = self._conv_block_2d(base_channels * 4, base_channels * 8, dropout)
        
        # Decoder (2D)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = self._conv_block_2d(base_channels * 8, base_channels * 4, dropout)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = self._conv_block_2d(base_channels * 4, base_channels * 2, dropout)
        
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = self._conv_block_2d(base_channels * 2, base_channels, dropout)
        
        # Output
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def _conv_block_2d(self, in_ch: int, out_ch: int, dropout: float) -> nn.Sequential:
        """2D convolutional block."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def _match_size(self, x: torch.Tensor, target_size: tuple) -> torch.Tensor:
        """Match spatial dimensions using interpolation if needed."""
        if x.shape[2:] != target_size:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x
    
    def forward(
            self, 
            x: torch.Tensor, 
            source_phase: Optional[torch.Tensor] = None, 
            target_phase: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        """
        Forward pass for 2D patches.
        
        Args:
            x: Input image [B, 1, 1, H, W] or [B, 1, H, W]
            source_phase: Source phase indices [B]
            target_phase: Target phase indices [B]
        
        Returns:
            Generated image [B, 1, 1, H, W]
        """
        # Store original shape info
        had_depth_dim = False
        
        # If input has depth dim of 1, squeeze it
        if x.dim() == 5 and x.size(2) == 1:
            had_depth_dim = True
            x = x.squeeze(2)  # [B, C, 1, H, W] -> [B, C, H, W]
        
        B, C, H, W = x.shape
        
        # Encoder - store sizes for skip connections
        e1 = self.enc1(x)                    # [B, 32, H, W]
        e1_size = e1.shape[2:]
        
        e2 = self.enc2(self.pool(e1))        # [B, 64, H/2, W/2]
        e2_size = e2.shape[2:]
        
        e3 = self.enc3(self.pool(e2))        # [B, 128, H/4, W/4]
        e3_size = e3.shape[2:]
        
        # Bottleneck
        b = self.bottleneck(self.pool(e3))   # [B, 256, H/8, W/8]
        
        # Phase conditioning
        if self.use_phase_conditioning and target_phase is not None:
            phase_emb = self.phase_embedding(target_phase)  # [B, phase_dim]
            
            # Get bottleneck channels
            bottleneck_channels = b.size(1)  # base_channels * 8
            
            # Project to spatial features
            phase_feat = self.phase_projection(phase_emb).view(
                B, bottleneck_channels, 8, 8
            )
            
            # Upsample to bottleneck spatial size
            phase_map = F.interpolate(
                phase_feat,
                size=(b.size(2), b.size(3)),
                mode='bilinear',
                align_corners=False
            )
            
            b = b + phase_map
        
        # Decoder with size matching for skip connections
        d3 = self.up3(b)
        d3 = self._match_size(d3, e3_size)   # Match to encoder size
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = self._match_size(d2, e2_size)   # Match to encoder size
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = self._match_size(d1, e1_size)   # Match to encoder size
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        out = self.out_conv(d1)
        
        # Add back depth dimension for consistency [B, C, H, W] -> [B, C, 1, H, W]
        out = out.unsqueeze(2)
        
        return torch.sigmoid(out)

        
class PatchDiscriminator(nn.Module):
    """
    Patch-based discriminator for adversarial training.
    Operates on 3D patches to classify real vs fake.
    """
    
    def __init__(self, in_channels: int = 1, base_channels: int = 64):
        super().__init__()
        
        self.model = nn.Sequential(
            # Layer 1
            nn.Conv3d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3
            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output
            nn.Conv3d(base_channels * 4, 1, kernel_size=4, padding=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image [B, 1, D, H, W]
        
        Returns:
            Patch predictions [B, 1, D', H', W']
        """
        return self.model(x)


# ============================================================================
# LOSS TRACKING
# ============================================================================

class LossTracker:
    """
    Track and visualize training losses over time.
    Saves plots and JSON logs.
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {
            'epoch': [],
            'train_gen_loss': [],
            'train_mse': [],
            'train_focal': [],
            'train_disc': [],
            'val_loss': [],
            'val_psnr': [],
            'val_ssim': [],
            'lr_gen': [],
            'lr_disc': [],
            'adv_weight': []
        }
    
    def update_epoch(self, epoch: int, train_losses: Dict, val_metrics: Dict, 
                    lr_gen: float, lr_disc: float, adv_weight: float):
        """Add metrics for completed epoch."""
        self.history['epoch'].append(epoch)
        self.history['train_gen_loss'].append(train_losses.get('gen_loss', 0))
        self.history['train_mse'].append(train_losses.get('gen_mse', 0))
        self.history['train_focal'].append(train_losses.get('gen_focal', 0))
        self.history['train_disc'].append(train_losses.get('disc', 0))
        self.history['val_loss'].append(val_metrics['val_loss'])
        self.history['val_psnr'].append(val_metrics['psnr'])
        self.history['val_ssim'].append(val_metrics['ssim'])
        self.history['lr_gen'].append(lr_gen)
        self.history['lr_disc'].append(lr_disc)
        self.history['adv_weight'].append(adv_weight)
        
        # Save JSON
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def create_summary_report(self, current_epoch: int):
        """Create visualization plots."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        epochs = self.history['epoch']
        
        # Generator losses
        axes[0, 0].plot(epochs, self.history['train_gen_loss'], label='Total Gen Loss')
        axes[0, 0].plot(epochs, self.history['train_mse'], label='MSE')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Generator Losses')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Validation metrics
        axes[0, 1].plot(epochs, self.history['val_loss'], label='Val Loss', color='orange')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Validation Loss')
        axes[0, 1].grid(True)
        
        # PSNR and SSIM
        ax_psnr = axes[0, 2]
        ax_ssim = ax_psnr.twinx()
        ax_psnr.plot(epochs, self.history['val_psnr'], 'b-', label='PSNR')
        ax_ssim.plot(epochs, self.history['val_ssim'], 'r-', label='SSIM')
        ax_psnr.set_xlabel('Epoch')
        ax_psnr.set_ylabel('PSNR (dB)', color='b')
        ax_ssim.set_ylabel('SSIM', color='r')
        ax_psnr.set_title('Image Quality Metrics')
        ax_psnr.grid(True)
        
        # Learning rates
        axes[1, 0].plot(epochs, self.history['lr_gen'], label='Gen LR')
        if any(self.history['lr_disc']):
            axes[1, 0].plot(epochs, self.history['lr_disc'], label='Disc LR')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Adversarial weight
        if any(self.history['adv_weight']):
            axes[1, 1].plot(epochs, self.history['adv_weight'])
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Weight')
            axes[1, 1].set_title('Adversarial Loss Weight')
            axes[1, 1].grid(True)
        
        # Discriminator loss
        if any(self.history['train_disc']):
            axes[1, 2].plot(epochs, self.history['train_disc'])
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Loss')
            axes[1, 2].set_title('Discriminator Loss')
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'training_progress_epoch_{current_epoch}.png', dpi=150)
        plt.close()


# ============================================================================
# TRAINER
# ============================================================================

class MemoryOptimizedTrainer:
    """
    Optimized trainer with:
    - Mixed precision training
    - Gradient accumulation
    - Learning rate scheduling
    - Checkpoint management
    - Memory cleanup
    - Phase conditioning
    - Mask-weighted losses
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config['device'])
        
        # Create output directories
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.samples_dir = self.output_dir / 'sample_patches'
        self.samples_dir.mkdir(exist_ok=True)
        
        # ========== Choose 2D or 3D generator based on patch_depth ==========
        patch_depth = config.get('patch_depth', 1)

        if patch_depth == 1:
            logger.info("✓ Using 2D U-Net (patch_depth=1)")
            self.generator = PhaseGenerator2D(
                in_channels=1,
                out_channels=1,
                base_channels=config.get('generator_base_channels', 32),
                dropout=config.get('generator_dropout', 0.3),
                use_phase_conditioning=config.get('use_phase_conditioning', True),
                num_phases=config.get('num_phase', 4),
                phase_embedding_dim=config.get('phase_embedding_dim', 64)
            ).to(self.device)
        else:
            logger.info(f"✓ Using 3D U-Net (patch_depth={patch_depth})")
            self.generator = PhaseGenerator(
                in_channels=1,
                out_channels=1,
                base_channels=config.get('generator_base_channels', 32),
                dropout=config.get('generator_dropout', 0.3),
                use_phase_conditioning=config.get('use_phase_conditioning', True),
                num_phases=config.get('num_phase', 4),
                phase_embedding_dim=config.get('phase_embedding_dim', 64)
            ).to(self.device)
        
        # Initialize loss functions
        if config.get('use_mask_weighting', True):
            self.mse_criterion = WeightedMSELoss()
            self.use_weighting = True
            logger.info("✓ Using weighted MSE loss with mask importance")
        else:
            self.mse_criterion = nn.MSELoss()
            self.use_weighting = False
            logger.info("✓ Using standard MSE loss")
        
        self.focal_criterion = FocalLoss(alpha=0.25, gamma=2.0)
        
        # Optimizer and scheduler
        self.opt_gen = optim.AdamW(
            self.generator.parameters(),
            lr=config.get('learning_rate', 2e-4),
            betas=(0.5, 0.999),
            weight_decay=1e-4
        )
        
        self.scheduler_gen = optim.lr_scheduler.ReduceLROnPlateau(
            self.opt_gen,
            mode='min',
            factor=0.5,
            patience=5
            # verbose=True
        )
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.get('use_mixed_precision', True))
        
        # Loss tracking
        self.loss_tracker = LossTracker(self.output_dir)
        
        # Best model tracking
        self.best_val_loss = float('inf')
        
        logger.info(f"✓ Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")
        logger.info(f"✓ Output directory: {self.output_dir}")
    
    def compute_psnr(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute Peak Signal-to-Noise Ratio."""
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return 100.0
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        return psnr.item()
    
    def compute_ssim(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute Structural Similarity Index (simplified)."""
        # Use middle slice for 3D patches
        pred_2d = pred[:, :, pred.size(2) // 2]
        target_2d = target[:, :, target.size(2) // 2]
        
        mu_pred = pred_2d.mean()
        mu_target = target_2d.mean()
        
        sigma_pred = pred_2d.std()
        sigma_target = target_2d.std()
        sigma_pred_target = ((pred_2d - mu_pred) * (target_2d - mu_target)).mean()
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim = ((2 * mu_pred * mu_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred ** 2 + sigma_target ** 2 + C2))
        
        return ssim.item()
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch with mask weighting and phase conditioning."""
        self.generator.train()
        
        total_gen_loss = 0
        total_mse = 0
        total_focal = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}')
        
        for batch_idx, batch in enumerate(pbar):
            source = batch['source'].to(self.device)
            target = batch['target'].to(self.device)
            weight_mask = batch['weight_mask'].to(self.device)
            source_phase = batch['source_phase'].to(self.device)
            target_phase = batch['target_phase'].to(self.device)
            
            # Generator step
            self.opt_gen.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=self.config.get('use_mixed_precision', True)):
                # Generate with phase conditioning
                generated = self.generator(source, source_phase=source_phase, target_phase=target_phase)
                
                # Weighted MSE loss
                if self.use_weighting:
                    mse_loss = self.mse_criterion(generated, target, weight_mask)
                else:
                    mse_loss = self.mse_criterion(generated, target)
                
                # Optional focal loss
                if self.config.get('use_focal_loss', False):
                    focal_loss = self.focal_criterion(generated, target, weight_mask)
                    gen_loss = mse_loss + 0.1 * focal_loss
                else:
                    focal_loss = torch.tensor(0.0)
                    gen_loss = mse_loss
            
            # Backward and optimize
            self.scaler.scale(gen_loss).backward()
            self.scaler.step(self.opt_gen)
            self.scaler.update()
            
            # Update metrics
            total_gen_loss += gen_loss.item()
            total_mse += mse_loss.item()
            total_focal += focal_loss.item() if isinstance(focal_loss, torch.Tensor) else 0.0
            num_batches += 1
            
            pbar.set_postfix({
                'Gen': f'{gen_loss.item():.4f}',
                'MSE': f'{mse_loss.item():.4f}'
            })
            
            # Memory cleanup
            if batch_idx % self.config.get('cleanup_frequency', 10) == 0:
                torch.cuda.empty_cache()
        
        return {
            'gen_loss': total_gen_loss / num_batches,
            'gen_mse': total_mse / num_batches,
            'gen_focal': total_focal / num_batches
        }
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate with phase conditioning."""
        self.generator.eval()
        
        total_loss = 0
        total_psnr = 0
        total_ssim = 0
        num_batches = 0
        
        for batch in tqdm(val_loader, desc='Validation'):
            source = batch['source'].to(self.device)
            target = batch['target'].to(self.device)
            weight_mask = batch['weight_mask'].to(self.device)
            source_phase = batch['source_phase'].to(self.device)
            target_phase = batch['target_phase'].to(self.device)
            
            # Generate
            generated = self.generator(source, source_phase=source_phase, target_phase=target_phase)
            
            # Compute loss
            if self.use_weighting:
                loss = self.mse_criterion(generated, target, weight_mask)
            else:
                loss = self.mse_criterion(generated, target)
            
            # Metrics
            psnr = self.compute_psnr(generated, target)
            ssim = self.compute_ssim(generated, target)
            
            total_loss += loss.item()
            total_psnr += psnr
            total_ssim += ssim
            num_batches += 1
        
        return {
            'val_loss': total_loss / num_batches,
            'psnr': total_psnr / num_batches,
            'ssim': total_ssim / num_batches
        }
    
    # def save_epoch_patches(self, val_loader: DataLoader, epoch: int, num_samples: int = 10):
    #     """Save sample patches after each epoch for visual monitoring."""
    #     self.generator.eval()
        
    #     epoch_dir = self.samples_dir / f'epoch_{epoch:03d}'
    #     epoch_dir.mkdir(parents=True, exist_ok=True)
        
    #     saved_count = 0
    #     discrete_range = self.config.get('discrete_range', (0, 255))
        
    #     with torch.no_grad():
    #         for batch in val_loader:
    #             if saved_count >= num_samples:
    #                 break
                
    #             source = batch['source'].to(self.device)
    #             target = batch['target'].to(self.device)
    #             weight_mask = batch['weight_mask'].to(self.device)
    #             source_phase = batch['source_phase'].to(self.device)
    #             target_phase = batch['target_phase'].to(self.device)
    #             case_id = batch['case_id'][0]
                
    #             # Generate
    #             generated = self.generator(source, source_phase=source_phase, target_phase=target_phase)
                
    #             # Convert back to 0-255 for visualization
    #             source_np = (source[0, 0].cpu().numpy() * discrete_range[1]).astype(np.uint8)
    #             target_np = (target[0, 0].cpu().numpy() * discrete_range[1]).astype(np.uint8)
    #             generated_np = (generated[0, 0].cpu().numpy() * discrete_range[1]).astype(np.uint8)
    #             weight_np = weight_mask[0, 0].cpu().numpy()
                
    #             # Take middle slice
    #             mid_slice = source_np.shape[0] // 2
                
    #             # Create comparison figure
    #             fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                
    #             # Row 1: Images
    #             axes[0, 0].imshow(source_np[mid_slice], cmap='gray', vmin=0, vmax=255)
    #             axes[0, 0].set_title(f'Source (Phase {source_phase[0].item()})')
    #             axes[0, 0].axis('off')
                
    #             axes[0, 1].imshow(generated_np[mid_slice], cmap='gray', vmin=0, vmax=255)
    #             axes[0, 1].set_title(f'Generated (→Phase {target_phase[0].item()})')
    #             axes[0, 1].axis('off')
                
    #             axes[0, 2].imshow(target_np[mid_slice], cmap='gray', vmin=0, vmax=255)
    #             axes[0, 2].set_title(f'Target (Phase {target_phase[0].item()})')
    #             axes[0, 2].axis('off')
                
    #             # Row 2: Analysis
    #             # Difference map
    #             diff = np.abs(generated_np[mid_slice] - target_np[mid_slice])
    #             im1 = axes[1, 0].imshow(diff, cmap='hot', vmin=0, vmax=64)
    #             axes[1, 0].set_title('Absolute Difference')
    #             axes[1, 0].axis('off')
    #             plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)
                
    #             # Weight mask
    #             im2 = axes[1, 1].imshow(weight_np[mid_slice], cmap='viridis', vmin=1, vmax=3)
    #             axes[1, 1].set_title('Importance Weights')
    #             axes[1, 1].axis('off')
    #             plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)
                
    #             # Error histogram
    #             axes[1, 2].hist(diff.flatten(), bins=50, color='red', alpha=0.7)
    #             axes[1, 2].set_xlabel('Absolute Error')
    #             axes[1, 2].set_ylabel('Frequency')
    #             axes[1, 2].set_title('Error Distribution')
    #             axes[1, 2].grid(True, alpha=0.3)
                
    #             plt.suptitle(f'Epoch {epoch} - Case {case_id}', fontsize=14, fontweight='bold')
    #             plt.tight_layout()
    #             plt.savefig(epoch_dir / f'sample_{saved_count:03d}_{case_id}.png', dpi=150, bbox_inches='tight')
    #             plt.close()
                
    #             saved_count += 1
        
    #     logger.info(f'✓ Saved {saved_count} patch samples to {epoch_dir}')
    
    def save_epoch_patches(self, val_loader: DataLoader, epoch: int, num_samples: int = 10):
        """
        Save sample patches after each epoch with random selection.
        Randomly selects samples from validation set and random slices from each patch.
        """
        import random
        
        self.generator.eval()
        
        epoch_dir = self.samples_dir / f'epoch_{epoch:03d}'
        epoch_dir.mkdir(parents=True, exist_ok=True)
        
        discrete_range = self.config.get('discrete_range', (0, 255))
        
        # ========== COLLECT ALL BATCHES FIRST ==========
        all_batches = []
        with torch.no_grad():
            for batch in val_loader:
                all_batches.append(batch)
        
        if len(all_batches) == 0:
            logger.warning("No validation batches available for sampling")
            return
        
        # ========== RANDOMLY SELECT BATCHES ==========
        num_batches_to_sample = min(num_samples, len(all_batches))
        selected_batch_indices = random.sample(range(len(all_batches)), num_batches_to_sample)
        
        logger.info(f"Saving {num_batches_to_sample} randomly selected samples...")
        
        saved_count = 0
        
        with torch.no_grad():
            for batch_idx in selected_batch_indices:
                batch = all_batches[batch_idx]
                
                # Get batch data
                source = batch['source'].to(self.device)
                target = batch['target'].to(self.device)
                weight_mask = batch['weight_mask'].to(self.device)
                source_phase = batch['source_phase'].to(self.device)
                target_phase = batch['target_phase'].to(self.device)
                
                # Randomly select one sample from batch
                batch_size = source.size(0)
                sample_idx = random.randint(0, batch_size - 1)
                
                case_id = batch['case_id'][sample_idx]
                
                # Generate
                generated = self.generator(source, source_phase=source_phase, target_phase=target_phase)
                
                # Convert back to 0-255 for visualization
                source_np = (source[sample_idx, 0].cpu().numpy() * discrete_range[1]).astype(np.uint8)
                target_np = (target[sample_idx, 0].cpu().numpy() * discrete_range[1]).astype(np.uint8)
                generated_np = (generated[sample_idx, 0].cpu().numpy() * discrete_range[1]).astype(np.uint8)
                weight_np = weight_mask[sample_idx, 0].cpu().numpy()
                
                # ========== RANDOMLY SELECT SLICE ==========
                depth = source_np.shape[0]
                
                if depth > 1:
                    # For multi-slice patches, randomly select a slice
                    # Avoid edge slices (first and last), select from middle 80%
                    start_slice = max(0, int(depth * 0.1))
                    end_slice = min(depth, int(depth * 0.9))
                    slice_idx = random.randint(start_slice, end_slice - 1) if end_slice > start_slice else depth // 2
                else:
                    # Single slice
                    slice_idx = 0
                
                # ========== CREATE VISUALIZATION ==========
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                
                # Row 1: Images
                axes[0, 0].imshow(source_np[slice_idx], cmap='gray', vmin=0, vmax=discrete_range[1])
                axes[0, 0].set_title(f'Source (Phase {source_phase[sample_idx].item()})', fontsize=12, fontweight='bold')
                axes[0, 0].axis('off')
                
                axes[0, 1].imshow(generated_np[slice_idx], cmap='gray', vmin=0, vmax=discrete_range[1])
                axes[0, 1].set_title(f'Generated (→Phase {target_phase[sample_idx].item()})', fontsize=12, fontweight='bold')
                axes[0, 1].axis('off')
                
                axes[0, 2].imshow(target_np[slice_idx], cmap='gray', vmin=0, vmax=discrete_range[1])
                axes[0, 2].set_title(f'Target (Phase {target_phase[sample_idx].item()})', fontsize=12, fontweight='bold')
                axes[0, 2].axis('off')
                
                # Row 2: Analysis
                # Difference map
                diff = np.abs(generated_np[slice_idx] - target_np[slice_idx])
                im1 = axes[1, 0].imshow(diff, cmap='hot', vmin=0, vmax=discrete_range[1]//4)
                axes[1, 0].set_title('Absolute Difference', fontsize=12, fontweight='bold')
                axes[1, 0].axis('off')
                plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)
                
                # Weight mask
                im2 = axes[1, 1].imshow(weight_np[slice_idx], cmap='viridis', vmin=1, vmax=self.config.get('high_intensity_weight', 3.0))
                axes[1, 1].set_title('Importance Weights', fontsize=12, fontweight='bold')
                axes[1, 1].axis('off')
                plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)
                
                # Error histogram
                axes[1, 2].hist(diff.flatten(), bins=50, color='red', alpha=0.7, edgecolor='black')
                axes[1, 2].set_xlabel('Absolute Error', fontsize=10)
                axes[1, 2].set_ylabel('Frequency', fontsize=10)
                axes[1, 2].set_title('Error Distribution', fontsize=12, fontweight='bold')
                axes[1, 2].grid(True, alpha=0.3)
                
                # Add statistics to histogram
                mean_error = diff.mean()
                max_error = diff.max()
                axes[1, 2].axvline(mean_error, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.1f}')
                axes[1, 2].legend()
                
                # Overall title with slice info
                if depth > 1:
                    slice_info = f'Slice {slice_idx}/{depth-1} (randomly selected)'
                else:
                    slice_info = 'Single slice'
                
                plt.suptitle(
                    f'Epoch {epoch} - Sample {saved_count+1}/{num_batches_to_sample} - Case {case_id}\n{slice_info}',
                    fontsize=14,
                    fontweight='bold'
                )
                
                plt.tight_layout()
                plt.savefig(
                    epoch_dir / f'sample_{saved_count:03d}_{case_id}_slice{slice_idx}.png',
                    dpi=150,
                    bbox_inches='tight'
                )
                plt.close()
                
                saved_count += 1
        
        logger.info(f'✓ Saved {saved_count} randomly selected patch samples to {epoch_dir}')


    def save_checkpoint(self, val_metrics: Dict, is_best: bool, epoch: int):
        """Save model checkpoint with cleanup of old checkpoints."""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'optimizer_gen_state_dict': self.opt_gen.state_dict(),
            'scheduler_gen_state_dict': self.scheduler_gen.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'val_metrics': val_metrics,
            'config': self.config
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, latest_path)
        logger.info(f"✓ Saved checkpoint: {latest_path}")
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"✓ New best model saved! Val Loss: {val_metrics['val_loss']:.4f}")
        
        # Cleanup old checkpoints (keep last N)
        keep_last_n = self.config.get('keep_last_n_checkpoints', 3)
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        if len(checkpoints) > keep_last_n:
            for old_ckpt in checkpoints[:-keep_last_n]:
                old_ckpt.unlink()
                logger.info(f"✓ Removed old checkpoint: {old_ckpt.name}")
    
    def load_checkpoint(self, checkpoint_path: Path) -> bool:
        """Load checkpoint and resume training."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.opt_gen.load_state_dict(checkpoint['optimizer_gen_state_dict'])
            self.scheduler_gen.load_state_dict(checkpoint['scheduler_gen_state_dict'])
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            logger.info(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to load checkpoint: {e}")
            return False
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int, start_epoch: int = 0):
        """Main training loop with all optimizations."""
        logger.info("=" * 80)
        logger.info("STARTING OPTIMIZED TRAINING")
        logger.info("=" * 80)
        logger.info(f"Start epoch: {start_epoch}")
        logger.info(f"Total epochs: {num_epochs}")
        logger.info(f"Batches per epoch: {len(train_loader)}")
        logger.info(f"Phase conditioning: {self.config.get('use_phase_conditioning', True)}")
        logger.info(f"Mask weighting: {self.use_weighting}")
        logger.info("=" * 80)
        
        for epoch in range(start_epoch, num_epochs):
            # Train
            train_losses = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler_gen.step(val_metrics['val_loss'])
            
            # Save sample patches every epoch
            self.save_epoch_patches(val_loader, epoch, num_samples=10)
            
            # Log metrics
            logger.info("=" * 80)
            logger.info(f"Epoch {epoch + 1}/{num_epochs} Complete")
            logger.info("-" * 80)
            logger.info(f"  Train - Gen: {train_losses['gen_loss']:.4f}, "
                       f"MSE: {train_losses['gen_mse']:.4f}, "
                       f"Focal: {train_losses['gen_focal']:.4f}")
            logger.info(f"  Val - Loss: {val_metrics['val_loss']:.4f}, "
                       f"PSNR: {val_metrics['psnr']:.2f} dB, "
                       f"SSIM: {val_metrics['ssim']:.4f}")
            
            lr_gen = self.opt_gen.param_groups[0]['lr']
            logger.info(f"  Learning Rate: {lr_gen:.2e}")
            logger.info("=" * 80)
            
            # Update loss tracker
            self.loss_tracker.update_epoch(
                epoch=epoch,
                train_losses=train_losses,
                val_metrics=val_metrics,
                lr_gen=lr_gen,
                lr_disc=0.0,
                adv_weight=0.0
            )
            
            # Create summary report every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.loss_tracker.create_summary_report(epoch)
            
            # Save checkpoint
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
            
            self.save_checkpoint(val_metrics, is_best, epoch)
            
            # Memory cleanup
            gc.collect()
            torch.cuda.empty_cache()
        
        logger.info("=" * 80)
        logger.info("✓ TRAINING COMPLETED SUCCESSFULLY!")
        logger.info(f"✓ Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"✓ Output directory: {self.output_dir}")
        logger.info("=" * 80)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main training script."""
    
    # Configuration
    config = {
        'data_dir': '../../../ncct_cect/vindr_ds/deformable_registered_bspline',
        'labels_csv': '../../../ncct_cect/vindr_ds/labels.csv',
        'output_dir': '../../../ncct_cect/vindr_ds/optimized_train/discrete_masked_conditioned',
        
        # Data configuration
        'data_is_preprocessed': True,  # Data already normalized to 0-255
        'discrete_range': (0, 255),
        
        # Patch configuration
        'patch_size': (128, 192),
        'patch_depth': 1,
        'slice_range': (0.3, 0.7),
        'overlap_ratio': 0.5,
        
        # Model configuration
        'generator_base_channels': 64,
        'generator_dropout': 0.3,
        'num_phase': 4,
        'phase_embedding_dim': 64,
        
        # Training configuration
        'batch_size': 16,
        'learning_rate': 2e-4,
        'epochs': 50,
        
        # Feature flags
        'use_phase_conditioning': True,
        'use_mask_weighting': True,
        'use_focal_loss': False,
        'use_discriminator': False,
        'use_cycle_consistency': False,
        
        # Mask weighting
        'high_intensity_weight': 3.0,
        'high_intensity_threshold': 150,
        
        # Memory optimization
        'use_mixed_precision': True,
        'cleanup_frequency': 5,
        'keep_last_n_checkpoints': 3,
        
        # Cache settings
        'cache_size': 12,
        'use_memmap': True,
        
        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # Sample saving
        'save_samples_interval': 1,
        'keep_last_n_sample_epochs': 5,
        
        # Debug options
        'debug_patches': False,
        'debug_output_dir': './debug_patches'
    }
    
    logger.info("=" * 80)
    logger.info("CT PHASE GENERATION TRAINING - OPTIMIZED VERSION")
    logger.info("=" * 80)
    logger.info("Features:")
    logger.info("  ✓ Preprocessed discrete data (0-255)")
    logger.info("  ✓ Mask-weighted loss (3x penalty on high-intensity)")
    logger.info("  ✓ Phase-conditional generation")
    logger.info("  ✓ Automatic patch saving after each epoch")
    logger.info("  ✓ Volume caching for fast loading")
    logger.info("  ✓ Mixed precision training")
    logger.info("=" * 80)
    
    # Check if data directory exists
    if not Path(config['data_dir']).exists():
        logger.error(f"✗ Data directory does not exist: {config['data_dir']}")
        logger.error("Please update the 'data_dir' in config to point to your data")
        return
    
    # Load data using the optimized dataloader
    try:
        from dataloader_train_slice import run_full_training
        
        logger.info("Starting training with optimized dataloader...")
        run_full_training(config)
        
    except ImportError:
        logger.error("✗ Could not import dataloader_train_slice")
        logger.error("Please ensure dataloader_train_slice.py is in the same directory")
        return
    except Exception as e:
        logger.error(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()