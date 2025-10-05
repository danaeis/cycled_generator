"""
Loss Tracking and Plotting System for CT Phase Training
========================================================
Comprehensive loss tracking with real-time plotting and analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class LossTracker:
    """
    Track all losses during training and provide plotting functionality.
    """
    
    def __init__(self, output_dir: str, save_interval: int = 1):
        """
        Initialize loss tracker.
        
        Args:
            output_dir: Directory to save loss plots and data
            save_interval: How often to save plots (in epochs)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_interval = save_interval
        
        # Loss history storage
        self.history = {
            # Training losses
            'epoch': [],
            'train_gen_total': [],
            'train_gen_adv': [],
            'train_gen_cycle': [],
            'train_gen_mse': [],
            'train_gen_focal': [],
            'train_disc_source': [],
            'train_disc_target': [],
            'train_disc_total': [],
            
            # Validation losses
            'val_loss': [],
            'val_psnr': [],
            'val_ssim': [],
            
            # Learning rates
            'lr_gen': [],
            'lr_disc': [],
            
            # Adversarial weight (for warmup tracking)
            'adv_weight': [],
        }
        
        # Batch-level tracking (within epoch)
        self.batch_history = {
            'train_gen_total': [],
            'train_gen_adv': [],
            'train_gen_cycle': [],
            'train_gen_mse': [],
            'train_gen_focal': [],
            'train_disc_total': [],
        }
        
        logger.info(f"Loss tracker initialized. Outputs will be saved to {self.output_dir}")
    
    def update_batch(self, losses: Dict[str, float]):
        """
        Update batch-level losses (called every training step).
        
        Args:
            losses: Dictionary of loss values from train_step
        """
        self.batch_history['train_gen_total'].append(losses.get('gen_total', 0.0))
        self.batch_history['train_gen_adv'].append(losses.get('gen_adv', 0.0))
        self.batch_history['train_gen_cycle'].append(losses.get('gen_cycle', 0.0))
        self.batch_history['train_gen_mse'].append(losses.get('gen_mse', 0.0))
        self.batch_history['train_gen_focal'].append(losses.get('gen_focal', 0.0))
        self.batch_history['train_disc_total'].append(losses.get('disc', 0.0))
    
    def update_epoch(
        self,
        epoch: int,
        train_losses: Dict[str, float],
        val_metrics: Dict[str, float],
        lr_gen: float,
        lr_disc: float = 0.0,
        adv_weight: float = 0.0
    ):
        """
        Update epoch-level losses (called at end of epoch).
        
        Args:
            epoch: Current epoch number
            train_losses: Average training losses for the epoch
            val_metrics: Validation metrics (loss, psnr, ssim)
            lr_gen: Generator learning rate
            lr_disc: Discriminator learning rate
            adv_weight: Current adversarial weight
        """
        self.history['epoch'].append(epoch)
        
        # Training losses
        self.history['train_gen_total'].append(train_losses.get('gen_total', 0.0))
        self.history['train_gen_adv'].append(train_losses.get('gen_adv', 0.0))
        self.history['train_gen_cycle'].append(train_losses.get('gen_cycle', 0.0))
        self.history['train_gen_mse'].append(train_losses.get('gen_mse', 0.0))
        self.history['train_gen_focal'].append(train_losses.get('gen_focal', 0.0))
        self.history['train_disc_source'].append(train_losses.get('disc_source', 0.0))
        self.history['train_disc_target'].append(train_losses.get('disc_target', 0.0))
        self.history['train_disc_total'].append(train_losses.get('disc', 0.0))
        
        # Validation metrics
        self.history['val_loss'].append(val_metrics.get('val_loss', 0.0))
        self.history['val_psnr'].append(val_metrics.get('psnr', 0.0))
        self.history['val_ssim'].append(val_metrics.get('ssim', 0.0))
        
        # Learning rates and weights
        self.history['lr_gen'].append(lr_gen)
        self.history['lr_disc'].append(lr_disc)
        self.history['adv_weight'].append(adv_weight)
        
        # Clear batch history for next epoch
        self.batch_history = {key: [] for key in self.batch_history.keys()}
        
        # Save periodically
        if (epoch + 1) % self.save_interval == 0:
            self.save_history()
            self.plot_all_losses(epoch)
    
    def save_history(self):
        """Save loss history to JSON file."""
        save_path = self.output_dir / 'loss_history.json'
        
        # Convert to serializable format
        history_serializable = {
            key: [float(v) if isinstance(v, (np.floating, np.integer)) else v 
                  for v in values]
            for key, values in self.history.items()
        }
        
        with open(save_path, 'w') as f:
            json.dump(history_serializable, f, indent=2)
        
        logger.debug(f"Saved loss history to {save_path}")
    
    def load_history(self, path: Optional[str] = None):
        """Load loss history from JSON file."""
        if path is None:
            path = self.output_dir / 'loss_history.json'
        else:
            path = Path(path)
        
        if path.exists():
            with open(path, 'r') as f:
                self.history = json.load(f)
            logger.info(f"Loaded loss history from {path}")
        else:
            logger.warning(f"No history file found at {path}")
    
    def plot_all_losses(self, epoch: int, save_pdf: bool = True):
        """
        Create comprehensive loss plots.
        
        Args:
            epoch: Current epoch number
            save_pdf: Whether to save as PDF (in addition to PNG)
        """
        if len(self.history['epoch']) < 2:
            logger.warning("Not enough data to plot (need at least 2 epochs)")
            return
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        epochs = self.history['epoch']
        
        # 1. Generator Loss Components (weighted)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_generator_components(ax1, epochs)
        
        # 2. Generator Total Loss
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_total_loss(ax2, epochs)
        
        # 3. Discriminator Losses
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_discriminator_losses(ax3, epochs)
        
        # 4. GAN Balance (Gen vs Disc)
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_gan_balance(ax4, epochs)
        
        # 5. Validation Metrics
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_validation_metrics(ax5, epochs)
        
        # 6. Loss Ratios
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_loss_ratios(ax6, epochs)
        
        # 7. Adversarial Weight Schedule
        ax7 = fig.add_subplot(gs[2, 0])
        self._plot_adversarial_schedule(ax7, epochs)
        
        # 8. Learning Rates
        ax8 = fig.add_subplot(gs[2, 1])
        self._plot_learning_rates(ax8, epochs)
        
        # 9. Loss Contribution Pie Chart (latest epoch)
        ax9 = fig.add_subplot(gs[2, 2])
        self._plot_loss_contribution(ax9)
        
        # Overall title
        fig.suptitle(f'Training Progress - Epoch {epoch + 1}', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        # Save plot
        save_path = self.output_dir / f'losses_epoch_{epoch + 1:04d}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved loss plot to {save_path}")
        
        if save_pdf:
            pdf_path = self.output_dir / f'losses_epoch_{epoch + 1:04d}.pdf'
            plt.savefig(pdf_path, bbox_inches='tight')
        
        plt.close(fig)
    
    def _plot_generator_components(self, ax, epochs):
        """Plot individual generator loss components."""
        # Get weighted values
        mse_weighted = [v for v in self.history['train_gen_mse']]
        cycle_weighted = [v for v in self.history['train_gen_cycle']]
        focal_weighted = [v for v in self.history['train_gen_focal']]
        adv_weighted = [v for v in self.history['train_gen_adv']]
        
        ax.plot(epochs, mse_weighted, label='MSE (weighted)', linewidth=2, color='#3498db')
        ax.plot(epochs, cycle_weighted, label='Cycle (weighted)', linewidth=2, color='#2ecc71')
        ax.plot(epochs, focal_weighted, label='Focal (weighted)', linewidth=2, color='#f39c12')
        ax.plot(epochs, adv_weighted, label='Adversarial (weighted)', linewidth=2, color='#e74c3c')
        
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Loss Value', fontweight='bold')
        ax.set_title('Generator Loss Components (Weighted)', fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    def _plot_total_loss(self, ax, epochs):
        """Plot total generator loss and validation loss."""
        ax.plot(epochs, self.history['train_gen_total'], 
                label='Train Gen Total', linewidth=2, color='#3498db', marker='o', markersize=3)
        ax.plot(epochs, self.history['val_loss'], 
                label='Validation Loss', linewidth=2, color='#e74c3c', marker='s', markersize=3)
        
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Loss Value', fontweight='bold')
        ax.set_title('Total Generator Loss', fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Highlight best validation
        if self.history['val_loss']:
            best_val_idx = np.argmin(self.history['val_loss'])
            best_val = self.history['val_loss'][best_val_idx]
            best_epoch = self.history['epoch'][best_val_idx]
            ax.axvline(best_epoch, color='green', linestyle='--', alpha=0.5)
            ax.text(best_epoch, best_val, f'  Best: {best_val:.4f}', 
                   verticalalignment='bottom', fontsize=8)
    
    def _plot_discriminator_losses(self, ax, epochs):
        """Plot discriminator losses."""
        ax.plot(epochs, self.history['train_disc_source'], 
                label='Disc Source', linewidth=2, color='#9b59b6', alpha=0.7)
        ax.plot(epochs, self.history['train_disc_target'], 
                label='Disc Target', linewidth=2, color='#e67e22', alpha=0.7)
        ax.plot(epochs, self.history['train_disc_total'], 
                label='Disc Total', linewidth=2.5, color='#34495e')
        
        # Reference line at 0.5 (balanced)
        ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Balanced (0.5)')
        
        # Warning zones
        ax.axhspan(0, 0.2, alpha=0.1, color='red', label='Too Strong')
        ax.axhspan(0.8, 1.0, alpha=0.1, color='orange', label='Too Weak')
        
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Loss Value', fontweight='bold')
        ax.set_title('Discriminator Losses', fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.0])
    
    def _plot_gan_balance(self, ax, epochs):
        """Plot generator vs discriminator balance."""
        # Normalize losses for comparison
        gen_norm = np.array(self.history['train_gen_total'])
        disc_norm = np.array(self.history['train_disc_total'])
        
        ax.plot(epochs, gen_norm, label='Generator', linewidth=2, color='#3498db')
        
        # Plot discriminator on secondary y-axis
        ax2 = ax.twinx()
        ax2.plot(epochs, disc_norm, label='Discriminator', linewidth=2, color='#e74c3c')
        
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Generator Loss', fontweight='bold', color='#3498db')
        ax2.set_ylabel('Discriminator Loss', fontweight='bold', color='#e74c3c')
        ax.set_title('GAN Balance', fontweight='bold')
        
        ax.tick_params(axis='y', labelcolor='#3498db')
        ax2.tick_params(axis='y', labelcolor='#e74c3c')
        
        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        ax.grid(True, alpha=0.3)
    
    def _plot_validation_metrics(self, ax, epochs):
        """Plot PSNR and SSIM."""
        if not any(self.history['val_psnr']):
            ax.text(0.5, 0.5, 'No validation data yet', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Validation Metrics', fontweight='bold')
            return
        
        ax.plot(epochs, self.history['val_psnr'], 
                label='PSNR (dB)', linewidth=2, color='#2ecc71', marker='o', markersize=3)
        
        # SSIM on secondary axis
        ax2 = ax.twinx()
        ax2.plot(epochs, self.history['val_ssim'], 
                 label='SSIM', linewidth=2, color='#9b59b6', marker='s', markersize=3)
        
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('PSNR (dB)', fontweight='bold', color='#2ecc71')
        ax2.set_ylabel('SSIM', fontweight='bold', color='#9b59b6')
        ax.set_title('Validation Metrics', fontweight='bold')
        
        ax.tick_params(axis='y', labelcolor='#2ecc71')
        ax2.tick_params(axis='y', labelcolor='#9b59b6')
        
        # Target lines
        ax.axhline(y=30, color='green', linestyle='--', alpha=0.3)
        ax2.axhline(y=0.85, color='purple', linestyle='--', alpha=0.3)
        
        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
        
        ax.grid(True, alpha=0.3)
    
    def _plot_loss_ratios(self, ax, epochs):
        """Plot ratios between loss components."""
        # Calculate ratios
        mse_to_cycle = np.array(self.history['train_gen_mse']) / (np.array(self.history['train_gen_cycle']) + 1e-8)
        mse_to_focal = np.array(self.history['train_gen_mse']) / (np.array(self.history['train_gen_focal']) + 1e-8)
        
        ax.plot(epochs, mse_to_cycle, label='MSE / Cycle', linewidth=2, color='#3498db')
        ax.plot(epochs, mse_to_focal, label='MSE / Focal', linewidth=2, color='#e74c3c')
        
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Ratio', fontweight='bold')
        ax.set_title('Loss Component Ratios', fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    def _plot_adversarial_schedule(self, ax, epochs):
        """Plot adversarial weight schedule."""
        ax.plot(epochs, self.history['adv_weight'], 
                linewidth=2.5, color='#e74c3c', marker='o', markersize=4)
        
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Adversarial Weight', fontweight='bold')
        ax.set_title('Adversarial Loss Weight Schedule', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.05, max(self.history['adv_weight']) * 1.1 if self.history['adv_weight'] else 1.1])
        
        # Highlight warmup period
        if any(w < max(self.history['adv_weight']) for w in self.history['adv_weight']):
            warmup_end = next((i for i, w in enumerate(self.history['adv_weight']) 
                              if w >= max(self.history['adv_weight']) * 0.99), len(epochs))
            if warmup_end < len(epochs):
                ax.axvspan(epochs[0], epochs[warmup_end], alpha=0.1, color='orange', label='Warmup')
                ax.legend(loc='best')
    
    def _plot_learning_rates(self, ax, epochs):
        """Plot learning rate schedules."""
        ax.plot(epochs, self.history['lr_gen'], 
                label='Generator LR', linewidth=2, color='#3498db', marker='o', markersize=3)
        ax.plot(epochs, self.history['lr_disc'], 
                label='Discriminator LR', linewidth=2, color='#e74c3c', marker='s', markersize=3)
        
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Learning Rate', fontweight='bold')
        ax.set_title('Learning Rate Schedules', fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    def _plot_loss_contribution(self, ax):
        """Plot pie chart of loss contributions (latest epoch)."""
        if len(self.history['epoch']) == 0:
            return
        
        # Get latest values
        latest_mse = self.history['train_gen_mse'][-1] if self.history['train_gen_mse'] else 0
        latest_cycle = self.history['train_gen_cycle'][-1] if self.history['train_gen_cycle'] else 0
        latest_focal = self.history['train_gen_focal'][-1] if self.history['train_gen_focal'] else 0
        latest_adv = self.history['train_gen_adv'][-1] if self.history['train_gen_adv'] else 0
        
        values = [latest_mse, latest_cycle, latest_focal, latest_adv]
        labels = ['MSE', 'Cycle', 'Focal', 'Adversarial']
        colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
        
        # Only plot non-zero values
        filtered_values = []
        filtered_labels = []
        filtered_colors = []
        for v, l, c in zip(values, labels, colors):
            if v > 1e-6:
                filtered_values.append(v)
                filtered_labels.append(l)
                filtered_colors.append(c)
        
        if sum(filtered_values) > 0:
            wedges, texts, autotexts = ax.pie(
                filtered_values, 
                labels=filtered_labels, 
                colors=filtered_colors,
                autopct='%1.1f%%',
                startangle=90
            )
            
            # Make percentage text bold
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            ax.set_title(f'Loss Contribution (Epoch {self.history["epoch"][-1] + 1})', 
                        fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
    
    def plot_batch_losses(self, epoch: int):
        """
        Plot within-epoch batch losses for debugging.
        
        Args:
            epoch: Current epoch number
        """
        if not self.batch_history['train_gen_total']:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Batch-Level Losses - Epoch {epoch + 1}', fontsize=14, fontweight='bold')
        
        batches = list(range(len(self.batch_history['train_gen_total'])))
        
        # Plot each loss component
        components = [
            ('train_gen_total', 'Generator Total', axes[0, 0]),
            ('train_gen_mse', 'MSE', axes[0, 1]),
            ('train_gen_cycle', 'Cycle', axes[0, 2]),
            ('train_gen_focal', 'Focal', axes[1, 0]),
            ('train_gen_adv', 'Adversarial', axes[1, 1]),
            ('train_disc_total', 'Discriminator', axes[1, 2]),
        ]
        
        for key, title, ax in components:
            if self.batch_history[key]:
                ax.plot(batches, self.batch_history[key], linewidth=1, alpha=0.7)
                
                # Add moving average
                if len(self.batch_history[key]) > 10:
                    window = min(50, len(self.batch_history[key]) // 10)
                    moving_avg = np.convolve(self.batch_history[key], 
                                            np.ones(window)/window, mode='valid')
                    ax.plot(range(window-1, len(batches)), moving_avg, 
                           linewidth=2, color='red', label=f'MA({window})')
                    ax.legend()
                
                ax.set_xlabel('Batch')
                ax.set_ylabel('Loss')
                ax.set_title(title, fontweight='bold')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / f'batch_losses_epoch_{epoch + 1:04d}.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        logger.debug(f"Saved batch losses to {save_path}")
    
    def create_summary_report(self, epoch: int):
        """Create a text summary report of training progress."""
        report_path = self.output_dir / f'training_report_epoch_{epoch + 1:04d}.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"TRAINING SUMMARY REPORT - Epoch {epoch + 1}\n")
            f.write("="*80 + "\n\n")
            
            if len(self.history['epoch']) < 2:
                f.write("Not enough training data yet.\n")
                return
            
            # Latest values
            f.write("LATEST VALUES (Current Epoch):\n")
            f.write("-"*80 + "\n")
            f.write(f"Generator Total:     {self.history['train_gen_total'][-1]:.6f}\n")
            f.write(f"  └─ MSE:            {self.history['train_gen_mse'][-1]:.6f}\n")
            f.write(f"  └─ Cycle:          {self.history['train_gen_cycle'][-1]:.6f}\n")
            f.write(f"  └─ Focal:          {self.history['train_gen_focal'][-1]:.6f}\n")
            f.write(f"  └─ Adversarial:    {self.history['train_gen_adv'][-1]:.6f}\n")
            f.write(f"Discriminator:       {self.history['train_disc_total'][-1]:.6f}\n")
            f.write(f"Validation Loss:     {self.history['val_loss'][-1]:.6f}\n")
            f.write(f"PSNR:                {self.history['val_psnr'][-1]:.2f} dB\n")
            f.write(f"SSIM:                {self.history['val_ssim'][-1]:.4f}\n")
            f.write(f"Adversarial Weight:  {self.history['adv_weight'][-1]:.4f}\n")
            f.write("\n")
            
            # Best values
            f.write("BEST VALUES (So Far):\n")
            f.write("-"*80 + "\n")
            best_val_idx = np.argmin(self.history['val_loss'])
            f.write(f"Best Validation Loss: {self.history['val_loss'][best_val_idx]:.6f} ")
            f.write(f"(Epoch {self.history['epoch'][best_val_idx] + 1})\n")
            
            best_psnr_idx = np.argmax(self.history['val_psnr'])
            f.write(f"Best PSNR:            {self.history['val_psnr'][best_psnr_idx]:.2f} dB ")
            f.write(f"(Epoch {self.history['epoch'][best_psnr_idx] + 1})\n")
            
            best_ssim_idx = np.argmax(self.history['val_ssim'])
            f.write(f"Best SSIM:            {self.history['val_ssim'][best_ssim_idx]:.4f} ")
            f.write(f"(Epoch {self.history['epoch'][best_ssim_idx] + 1})\n")
            f.write("\n")
            
            # Trends (last 10 epochs)
            if len(self.history['epoch']) >= 10:
                f.write("TRENDS (Last 10 Epochs):\n")
                f.write("-"*80 + "\n")
                
                recent_gen = self.history['train_gen_total'][-10:]
                gen_trend = "↓ Decreasing" if recent_gen[-1] < recent_gen[0] else "↑ Increasing"
                f.write(f"Generator Loss:   {gen_trend}\n")
                
                recent_val = self.history['val_loss'][-10:]
                val_trend = "↓ Decreasing" if recent_val[-1] < recent_val[0] else "↑ Increasing"
                f.write(f"Validation Loss:  {val_trend}\n")
                
                recent_psnr = self.history['val_psnr'][-10:]
                psnr_trend = "↑ Increasing" if recent_psnr[-1] > recent_psnr[0] else "↓ Decreasing"
                f.write(f"PSNR:             {psnr_trend}\n")
                f.write("\n")
            
            # Warnings
            f.write("WARNINGS & RECOMMENDATIONS:\n")
            f.write("-"*80 + "\n")
            warnings = []
            
            # Check discriminator
            if self.history['train_disc_total'][-1] < 0.2:
                warnings.append("⚠️  Discriminator too strong (loss < 0.2)")
            elif self.history['train_disc_total'][-1] > 0.8:
                warnings.append("⚠️  Discriminator too weak (loss > 0.8)")
            
            # Check validation trend
            if len(self.history['val_loss']) >= 5:
                if all(self.history['val_loss'][-i] > self.history['val_loss'][-i-1] 
                       for i in range(1, 5)):
                    warnings.append("⚠️  Validation loss increasing for 5 epochs (possible overfitting)")
            
            # Check PSNR target
            if self.history['val_psnr'][-1] < 25:
                warnings.append("⚠️  PSNR below 25 dB (target: >30 dB)")
            
            # Check SSIM target
            if self.history['val_ssim'][-1] < 0.8:
                warnings.append("⚠️  SSIM below 0.8 (target: >0.85)")
            
            if warnings:
                for warning in warnings:
                    f.write(f"{warning}\n")
            else:
                f.write("✓ All metrics look healthy!\n")
            
            f.write("\n" + "="*80 + "\n")
        
        logger.info(f"Saved training report to {report_path}")


# Integration with existing trainer
def integrate_loss_tracker_example():
    """
    Example of how to integrate LossTracker into existing CTPhaseTrainer.
    """
    
    example_code = '''
# In CTPhaseTrainer.__init__:
class CTPhaseTrainer:
    def __init__(self, config: Dict):
        # ... existing initialization ...
        
        # Add loss tracker
        self.loss_tracker = LossTracker(
            output_dir=self.output_dir / 'loss_plots',
            save_interval=1  # Save plots every epoch
        )
    
    def train_step(self, batch: Dict) -> Dict[str, float]:
        # ... existing training step ...
        losses = {
            'gen_total': ...,
            'gen_adv': ...,
            'gen_cycle': ...,
            'gen_mse': ...,
            'gen_focal': ...,
            'disc': ...,
            'disc_source': ...,
            'disc_target': ...,
        }
        
        # Track batch losses
        self.loss_tracker.update_batch(losses)
        
        return losses
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        # ... existing epoch training ...
        
        # Calculate average epoch losses
        epoch_losses = {k: np.mean(v) for k, v in accumulated_losses.items()}
        
        return epoch_losses
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Train
            train_losses = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Get current learning rates
            lr_gen = self.opt_gen.param_groups[0]['lr']
            lr_disc = self.opt_disc_source.param_groups[0]['lr'] if hasattr(self, 'opt_disc_source') else 0.0
            
            # Get adversarial weight
            adv_weight = self.adv_schedule.get_weight(epoch) if hasattr(self, 'adv_schedule') else 0.0
            
            # Update loss tracker
            self.loss_tracker.update_epoch(
                epoch=epoch,
                train_losses=train_losses,
                val_metrics=val_metrics,
                lr_gen=lr_gen,
                lr_disc=lr_disc,
                adv_weight=adv_weight
            )
            
            # Optionally plot batch losses
            if epoch % 5 == 0:  # Every 5 epochs
                self.loss_tracker.plot_batch_losses(epoch)
            
            # Create summary report
            if (epoch + 1) % 10 == 0:  # Every 10 epochs
                self.loss_tracker.create_summary_report(epoch)
            
            # ... rest of training logic ...
    '''
    
    print(example_code)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("LOSS TRACKING SYSTEM")
    print("="*80)
    print("\nThis module provides comprehensive loss tracking and visualization.")
    print("\nFeatures:")
    print("  • Track all loss components (MSE, Cycle, Focal, Adversarial, Discriminator)")
    print("  • Automatic plotting every epoch")
    print("  • Batch-level loss tracking for debugging")
    print("  • Validation metrics (PSNR, SSIM)")
    print("  • Learning rate schedules")
    print("  • Adversarial weight tracking")
    print("  • Summary reports with warnings")
    print("\nOutputs:")
    print("  • loss_history.json - All loss data")
    print("  • losses_epoch_XXXX.png - Comprehensive plots")
    print("  • batch_losses_epoch_XXXX.png - Within-epoch debugging")
    print("  • training_report_epoch_XXXX.txt - Text summary")
    
    print("\n" + "="*80)
    print("INTEGRATION EXAMPLE")
    print("="*80)
    integrate_loss_tracker_example()