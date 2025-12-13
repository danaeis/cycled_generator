"""
Compare Results Across Stepped Autoencoder Training
====================================================
Run after training multiple steps to visualize performance differences.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_history(step, dim="2D", base_dir="."):
    """Load training history for a specific step."""
    history_path = Path(base_dir) / f"autoencoder_stepped_{dim}_step{step}_lr6" / "training_history.json"
    
    if not history_path.exists():
        print(f"⚠️  No results found for Step {step} ({dim})")
        return None
    
    with open(history_path, 'r') as f:
        return json.load(f)


def plot_comparison(histories, dim="2D", save_path="stepped_comparison.png"):
    """
    Plot side-by-side comparison of all steps.
    
    Args:
        histories: dict of {step_num: history_dict}
    """
    if not histories:
        print("No histories to plot!")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = {1: 'blue', 2: 'green', 3: 'red'}
    labels = {
        1: "Step 1 (256→256→256)",
        2: "Step 2 (256→128→256)",
        3: "Step 3 (256→128→64→128→256)"
    }
    
    # Plot 1: Training Loss
    ax = axes[0, 0]
    for step, hist in sorted(histories.items()):
        ax.plot(hist['epoch'], hist['train_loss'], 
                color=colors[step], linewidth=2.5, label=labels[step], alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Training Loss (MSE)', fontsize=12)
    ax.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Validation Loss
    ax = axes[0, 1]
    for step, hist in sorted(histories.items()):
        ax.plot(hist['epoch'], hist['val_loss'], 
                color=colors[step], linewidth=2.5, label=labels[step], alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Loss (MSE)', fontsize=12)
    ax.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: PSNR
    ax = axes[1, 0]
    for step, hist in sorted(histories.items()):
        ax.plot(hist['epoch'], hist['val_psnr'], 
                color=colors[step], linewidth=2.5, label=labels[step], alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title('Validation PSNR Comparison ↑', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: SSIM
    ax = axes[1, 1]
    for step, hist in sorted(histories.items()):
        ax.plot(hist['epoch'], hist['val_ssim'], 
                color=colors[step], linewidth=2.5, label=labels[step], alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('SSIM', fontsize=12)
    ax.set_title('Validation SSIM Comparison ↑', fontsize=14, fontweight='bold')
    ax.set_ylim([0.5, 1.0])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Stepped Autoencoder Performance Comparison ({dim})', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Comparison plot saved: {save_path}")


def print_final_metrics(histories, dim="2D"):
    """Print final epoch metrics for each step."""
    print(f"\n{'='*70}")
    print(f"FINAL METRICS COMPARISON ({dim})")
    print(f"{'='*70}")
    print(f"{'Step':<6} {'Architecture':<30} {'Val Loss':<12} {'PSNR (dB)':<12} {'SSIM':<10}")
    print(f"{'-'*70}")
    
    arch_labels = {
        1: "256→256→256",
        2: "256→128→256",
        3: "256→128→64→128→256"
    }
    
    for step in sorted(histories.keys()):
        hist = histories[step]
        final_loss = hist['val_loss'][-1]
        final_psnr = hist['val_psnr'][-1]
        final_ssim = hist['val_ssim'][-1]
        
        print(f"{step:<6} {arch_labels[step]:<30} {final_loss:<12.6f} {final_psnr:<12.2f} {final_ssim:<10.4f}")
    
    print(f"{'='*70}\n")
    
    # Print analysis
    print("ANALYSIS:")
    if 1 in histories:
        psnr1 = histories[1]['val_psnr'][-1]
        if psnr1 > 35:
            print(f"✓ Step 1: Excellent baseline (PSNR {psnr1:.1f} dB) - training pipeline is solid")
        elif psnr1 > 30:
            print(f"⚠ Step 1: Moderate baseline (PSNR {psnr1:.1f} dB) - may have minor issues")
        else:
            print(f"✗ Step 1: Poor baseline (PSNR {psnr1:.1f} dB) - fundamental training problem!")
    
    if 2 in histories:
        psnr2 = histories[2]['val_psnr'][-1]
        if psnr2 > 32:
            print(f"✓ Step 2: Good single compression (PSNR {psnr2:.1f} dB)")
        elif psnr2 > 28:
            print(f"⚠ Step 2: Acceptable compression (PSNR {psnr2:.1f} dB) - some quality loss")
        else:
            print(f"✗ Step 2: Poor compression (PSNR {psnr2:.1f} dB) - bottleneck too narrow")
    
    if 3 in histories:
        psnr3 = histories[3]['val_psnr'][-1]
        if psnr3 > 28:
            print(f"✓ Step 3: Acceptable double compression (PSNR {psnr3:.1f} dB)")
        elif psnr3 > 24:
            print(f"⚠ Step 3: Challenging compression (PSNR {psnr3:.1f} dB) - significant quality loss")
        else:
            print(f"✗ Step 3: Failed compression (PSNR {psnr3:.1f} dB) - bottleneck far too narrow")
    
    print()


def main():
    """
    Compare results across all trained steps.
    Usage: python compare_steps.py
    """
    
    # Try loading all steps
    histories = {}
    for step in [1, 2, 3]:
        hist = load_history(step, dim="2D")
        if hist:
            histories[step] = hist
    
    if not histories:
        print("❌ No training results found!")
        print("   Make sure you've run training with at least one step.")
        return
    
    print(f"✓ Found results for steps: {list(histories.keys())}")
    
    # Generate comparison plot
    plot_comparison(histories, dim="2D", save_path="stepped_comparison_2D.png")
    
    # Print metrics table
    print_final_metrics(histories, dim="2D")
    
    # Check for 3D results
    histories_3d = {}
    for step in [1, 2, 3]:
        hist = load_history(step, dim="3D")
        if hist:
            histories_3d[step] = hist
    
    if histories_3d:
        print(f"\n✓ Found 3D results for steps: {list(histories_3d.keys())}")
        plot_comparison(histories_3d, dim="3D", save_path="stepped_comparison_3D.png")
        print_final_metrics(histories_3d, dim="3D")


if __name__ == "__main__":
    main()