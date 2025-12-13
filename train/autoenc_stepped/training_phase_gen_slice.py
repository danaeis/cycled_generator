"""
Stepped Autoencoder Generator for Progressive Compression Testing
Supports 3 compression levels:
- Step 1: 256 → 256 → 256 (identity-like, no compression)
- Step 2: 256 → 128 → 256 (single compression layer)
- Step 3: 256 → 128 → 64 → 128 → 256 (double compression)
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class SteppedGenerator2D(nn.Module):
    """
    Progressive compression autoencoder for debugging.
    Tests different bottleneck complexities to isolate reconstruction issues.
    """
    def __init__(
        self,
        compression_step: int = 1,  # 1, 2, or 3
        base_dim: int = 256,        # Starting latent dimension
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.compression_step = compression_step
        self.base_dim = base_dim
        self.dropout_rate = dropout
        
        # Input encoder: image → base_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, padding=3),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # /2
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, base_dim, kernel_size=3, stride=2, padding=1),  # /4
            nn.InstanceNorm2d(base_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Stepped bottleneck (the key difference)
        if compression_step == 1:
            # Step 1: 256 → 256 → 256 (identity-like)
            self.bottleneck = nn.Sequential(
                self._make_block(base_dim, base_dim),
                self._make_block(base_dim, base_dim),
                self._make_block(base_dim, base_dim),
            )
            logger.info(f"✓ Step 1 bottleneck: {base_dim} → {base_dim} → {base_dim} (identity)")
            
        elif compression_step == 2:
            # Step 2: 256 → 128 → 256 (single compression)
            mid_dim = base_dim // 2
            self.bottleneck = nn.Sequential(
                self._make_block(base_dim, mid_dim),      # 256→128
                self._make_block(mid_dim, mid_dim),       # 128→128
                self._make_block(mid_dim, base_dim),      # 128→256
            )
            logger.info(f"✓ Step 2 bottleneck: {base_dim} → {mid_dim} → {base_dim}")
            
        elif compression_step == 3:
            # Step 3: 256 → 128 → 64 → 128 → 256 (double compression)
            mid_dim1 = base_dim // 2
            mid_dim2 = base_dim // 4
            self.bottleneck = nn.Sequential(
                self._make_block(base_dim, mid_dim1),     # 256→128
                self._make_block(mid_dim1, mid_dim2),     # 128→64
                self._make_block(mid_dim2, mid_dim2),     # 64→64
                self._make_block(mid_dim2, mid_dim1),     # 64→128
                self._make_block(mid_dim1, base_dim),     # 128→256
            )
            logger.info(f"✓ Step 3 bottleneck: {base_dim} → {mid_dim1} → {mid_dim2} → {mid_dim1} → {base_dim}")
        
        else:
            raise ValueError(f"Invalid compression_step: {compression_step}. Must be 1, 2, or 3.")
        
        # Output decoder: base_dim → image
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_dim, 128, kernel_size=4, stride=2, padding=1),  # ×2
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # ×4
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 1, kernel_size=7, padding=3),
            nn.Tanh()
        )
        
        total_params = sum(p.numel() for p in self.parameters()) / 1e6
        logger.info(f"SteppedGenerator2D initialized | Step {compression_step} | "
                    f"{total_params:.2f}M params | dropout={dropout}")
    
    def _make_block(self, in_ch: int, out_ch: int):
        """Residual block with bottleneck structure"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(self.dropout_rate),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, 1, H, W)
        Returns:
            Reconstructed tensor (B, 1, H, W)
        """
        z = self.encoder(x)        # → (B, base_dim, H/4, W/4)
        z = self.bottleneck(z)     # → (B, base_dim, H/4, W/4)
        out = self.decoder(z)      # → (B, 1, H, W)
        return out


class SteppedGenerator3D(nn.Module):
    """
    3D version of stepped autoencoder for volumetric data.
    Same compression steps as 2D version.
    """
    def __init__(
        self,
        compression_step: int = 1,
        base_dim: int = 128,  # Lower for 3D due to memory
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.compression_step = compression_step
        self.base_dim = base_dim
        self.dropout_rate = dropout
        
        # Input encoder: volume → base_dim
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=7, padding=3),
            nn.InstanceNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),  # /2
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv3d(64, base_dim, kernel_size=3, stride=2, padding=1),  # /4
            nn.InstanceNorm3d(base_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Stepped bottleneck
        if compression_step == 1:
            self.bottleneck = nn.Sequential(
                self._make_block_3d(base_dim, base_dim),
                self._make_block_3d(base_dim, base_dim),
                self._make_block_3d(base_dim, base_dim),
            )
            logger.info(f"✓ Step 1 bottleneck (3D): {base_dim} → {base_dim} → {base_dim}")
            
        elif compression_step == 2:
            mid_dim = base_dim // 2
            self.bottleneck = nn.Sequential(
                self._make_block_3d(base_dim, mid_dim),
                self._make_block_3d(mid_dim, mid_dim),
                self._make_block_3d(mid_dim, base_dim),
            )
            logger.info(f"✓ Step 2 bottleneck (3D): {base_dim} → {mid_dim} → {base_dim}")
            
        elif compression_step == 3:
            mid_dim1 = base_dim // 2
            mid_dim2 = base_dim // 4
            self.bottleneck = nn.Sequential(
                self._make_block_3d(base_dim, mid_dim1),
                self._make_block_3d(mid_dim1, mid_dim2),
                self._make_block_3d(mid_dim2, mid_dim2),
                self._make_block_3d(mid_dim2, mid_dim1),
                self._make_block_3d(mid_dim1, base_dim),
            )
            logger.info(f"✓ Step 3 bottleneck (3D): {base_dim} → {mid_dim1} → {mid_dim2} → {mid_dim1} → {base_dim}")
        
        else:
            raise ValueError(f"Invalid compression_step: {compression_step}")
        
        # Output decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(base_dim, 64, kernel_size=4, stride=2, padding=1),  # ×2
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),  # ×4
            nn.InstanceNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv3d(32, 1, kernel_size=7, padding=3),
            nn.Tanh()
        )
        
        total_params = sum(p.numel() for p in self.parameters()) / 1e6
        logger.info(f"SteppedGenerator3D initialized | Step {compression_step} | "
                    f"{total_params:.2f}M params")
    
    def _make_block_3d(self, in_ch: int, out_ch: int):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(self.dropout_rate),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, 1, D, H, W)
        Returns:
            Reconstructed tensor (B, 1, D, H, W)
        """
        z = self.encoder(x)
        z = self.bottleneck(z)
        out = self.decoder(z)
        return out