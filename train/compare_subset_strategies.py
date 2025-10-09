"""
Compare Data Subset Strategies for Optuna Tuning
=================================================
Quick script to estimate runtime and help decide on subset size.
"""

from dataloader_train import create_data_pairs, load_phase_mapping
from ct_phase_training import CTPhaseDataset
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def estimate_tuning_time(
    data_dir: str,
    labels_csv: str,
    patch_size=(64, 64),
    patch_depth=7,
    batch_size=4,
    optuna_epochs=15,
    n_trials=50,
    subset_fractions=[0.1, 0.25, 0.33, 0.5, 1.0]
):
    """
    Estimate Optuna tuning time for different subset sizes.
    """
    
    print("\n" + "="*80)
    print("OPTUNA TUNING TIME ESTIMATOR")
    print("="*80)
    
    # Load data
    phase_mapping = None
    if Path(labels_csv).exists():
        phase_mapping = load_phase_mapping(labels_csv)
    
    data_splits = create_data_pairs(data_dir, phase_mapping=phase_mapping)
    
    print(f"\nDataset Size:")
    print(f"  Training pairs: {len(data_splits['train'])}")
    print(f"  Validation pairs: {len(data_splits['val'])}")
    
    # Create full dataset to count patches
    full_train_dataset = CTPhaseDataset(
        data_splits['train'],
        patch_size=patch_size,
        patch_depth=patch_depth,
        overlap_ratio=0.5,
        augment=False
    )
    
    full_val_dataset = CTPhaseDataset(
        data_splits['val'],
        patch_size=patch_size,
        patch_depth=patch_depth,
        overlap_ratio=0.5,
        augment=False
    )
    
    total_train_patches = len(full_train_dataset)
    total_val_patches = len(full_val_dataset)
    
    print(f"\nTotal Patches (with overlap=0.5):")
    print(f"  Training patches: {total_train_patches:,}")
    print(f"  Validation patches: {total_val_patches:,}")
    
    # Estimate time per batch (rough estimate: 0.5-1.0 sec per batch on modern GPU)
    seconds_per_batch_train = 0.75  # Adjust based on your GPU
    seconds_per_batch_val = 0.5     # Validation is faster (no backprop)
    
    print(f"\n{'='*80}")
    print(f"ESTIMATED TUNING TIME COMPARISON")
    print(f"{'='*80}")
    print(f"\nConfiguration: {n_trials} trials, {optuna_epochs} epochs/trial, batch_size={batch_size}")
    print(f"GPU assumption: ~{seconds_per_batch_train}s/batch (train), ~{seconds_per_batch_val}s/batch (val)\n")
    
    print(f"{'Subset':<10} {'Train':<15} {'Val':<15} {'Time/Trial':<15} {'Total Time':<15} {'Recommendation'}")
    print("-"*100)
    
    recommendations = {
        0.1: "Quick test only",
        0.25: "Fast iteration",
        0.33: "⭐ RECOMMENDED",
        0.5: "Thorough search",
        1.0: "Maximum (slow)"
    }
    
    for fraction in subset_fractions:
        # Calculate subset sizes
        subset_train_patches = int(total_train_patches * fraction)
        subset_val_patches = int(total_val_patches * fraction)
        
        # Calculate batches per epoch
        train_batches = (subset_train_patches + batch_size - 1) // batch_size
        val_batches = (subset_val_patches + batch_size - 1) // batch_size
        
        # Time per trial
        train_time_per_trial = train_batches * optuna_epochs * seconds_per_batch_train
        val_time_per_trial = val_batches * optuna_epochs * seconds_per_batch_val
        total_time_per_trial = train_time_per_trial + val_time_per_trial
        
        # Total time for all trials (accounting for ~30% pruning)
        pruning_factor = 0.7  # Assume 30% of trials get pruned early
        total_time = total_time_per_trial * n_trials * pruning_factor
        
        # Format output
        subset_str = f"{fraction:.0%}"
        train_str = f"{subset_train_patches:,}"
        val_str = f"{subset_val_patches:,}"
        time_per_trial_str = f"{total_time_per_trial/60:.1f} min"
        total_time_str = f"{total_time/3600:.1f} hrs"
        rec_str = recommendations.get(fraction, "")
        
        print(f"{subset_str:<10} {train_str:<15} {val_str:<15} {time_per_trial_str:<15} {total_time_str:<15} {rec_str}")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    # Smart recommendations based on dataset size
    if total_train_patches < 1000:
        print("\n📊 Small dataset detected (< 1000 patches)")
        print("   Recommendation: Use 50-100% of data (subset_fraction=0.5 or 1.0)")
        print("   Reason: Small datasets need more data for reliable tuning")
    elif total_train_patches < 5000:
        print("\n📊 Medium dataset detected (1K-5K patches)")
        print("   Recommendation: Use 33-50% of data (subset_fraction=0.33 or 0.5)")
        print("   Reason: Good balance between speed and reliability")
    else:
        print("\n📊 Large dataset detected (> 5K patches)")
        print("   Recommendation: Use 10-33% of data (subset_fraction=0.1 to 0.33)")
        print("   Reason: Subset is representative enough, saves significant time")
    
    print("\n💡 General Tips:")
    print("   • Start with 10% for initial testing")
    print("   • Use 33% for final hyperparameter search (best balance)")
    print("   • Only use 100% if dataset is very small or heterogeneous")
    print("   • More trials on subset > fewer trials on full data")
    print("   • Hyperparameters from 33% subset transfer well to full training")
    
    print("\n🎯 Suggested Workflow:")
    print("   1. Quick test: 10% data, 10 trials (~15-30 min)")
    print("   2. Standard tuning: 33% data, 50 trials (~2-4 hrs)")
    print("   3. Full training: 100% data with best hyperparameters")
    
    print("\n" + "="*80 + "\n")
    
    return {
        'total_train_patches': total_train_patches,
        'total_val_patches': total_val_patches,
        'subset_estimates': {
            f: {
                'train_patches': int(total_train_patches * f),
                'val_patches': int(total_val_patches * f),
                'time_hours': (
                    (int(total_train_patches * f) // batch_size * optuna_epochs * seconds_per_batch_train +
                     int(total_val_patches * f) // batch_size * optuna_epochs * seconds_per_batch_val) *
                    n_trials * 0.7
                ) / 3600
            }
            for f in subset_fractions
        }
    }


def main():
    """Run the comparison."""
    
    # Your configuration
    config = {
        'data_dir': '../ncct_cect/vindr_ds/registered_cases',
        'labels_csv': '../ncct_cect/vindr_ds/labels.csv',
        'patch_size': (96, 128),
        'patch_depth': 7,
        'batch_size': 4,
        'optuna_epochs': 15,
        'n_trials': 50
    }
    
    # Check if data exists
    if not Path(config['data_dir']).exists():
        print(f"\n❌ Error: Data directory not found: {config['data_dir']}")
        print("Please update the 'data_dir' path in this script.")
        return
    
    try:
        results = estimate_tuning_time(
            data_dir=config['data_dir'],
            labels_csv=config['labels_csv'],
            patch_size=config['patch_size'],
            patch_depth=config['patch_depth'],
            batch_size=config['batch_size'],
            optuna_epochs=config['optuna_epochs'],
            n_trials=config['n_trials']
        )
        
        print("✅ Analysis complete!")
        print("\nNext step: Edit optuna_tuning.py and set your preferred subset_fraction")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()