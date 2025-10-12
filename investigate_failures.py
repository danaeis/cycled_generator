"""
Investigate Optuna Trial Failures
==================================
Debug why 55% of trials are failing.
"""

import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def investigate_failures(
    trials_csv: str = "trials_ct_phase_loss_tuning.csv",
    output_dir: str = "../ncct_cect/vindr_ds/optuna_tuning"
):
    """Investigate why trials are failing."""
    
    df = pd.read_csv(output_dir+"/optuna_results/"+trials_csv)
    output_dir = Path(output_dir)
    
    print("="*80)
    print("INVESTIGATING TRIAL FAILURES")
    print("="*80)
    
    # Get failed trials
    failed = df[df['state'] == 'FAIL']
    completed = df[df['state'] == 'COMPLETE']
    
    print(f"\n📊 FAILURE STATISTICS:")
    print(f"   Total trials: {len(df)}")
    print(f"   Failed: {len(failed)} ({len(failed)/len(df)*100:.1f}%)")
    print(f"   Completed: {len(completed)} ({len(completed)/len(df)*100:.1f}%)")
    print(f"   Pruned: {len(df[df['state'] == 'PRUNED'])}")
    
    # Analyze duration patterns
    print(f"\n⏱️  DURATION ANALYSIS:")
    
    if len(failed) > 0:
        failed_durations = pd.to_timedelta(failed['duration'].dropna())
        if len(failed_durations) > 0:
            avg_fail_duration = failed_durations.mean()
            print(f"   Average time before failure: {avg_fail_duration}")
            
            # Quick failures (< 5 min) vs slow failures
            quick_fails = sum(failed_durations < pd.Timedelta(minutes=5))
            print(f"   Quick failures (<5 min): {quick_fails}/{len(failed_durations)}")
            print(f"      → Likely data loading or initialization errors")
            
            slow_fails = sum(failed_durations > pd.Timedelta(minutes=30))
            print(f"   Slow failures (>30 min): {slow_fails}/{len(failed_durations)}")
            print(f"      → Likely OOM or training instabilities")
    
    if len(completed) > 0:
        completed_durations = pd.to_timedelta(completed['duration'])
        avg_complete_duration = completed_durations.mean()
        print(f"   Average completion time: {avg_complete_duration}")
    
    # Check for parameter patterns in failures
    print(f"\n🔍 PARAMETER ANALYSIS:")
    print("   Comparing failed vs completed trials...\n")
    
    param_cols = [col for col in df.columns if col.startswith('params_')]
    
    for col in param_cols:
        param_name = col.replace('params_', '')
        
        if len(failed) > 0 and len(completed) > 0:
            fail_mean = failed[col].mean()
            complete_mean = completed[col].mean()
            diff_pct = abs(fail_mean - complete_mean) / complete_mean * 100
            
            if diff_pct > 20:  # Significant difference
                print(f"   ⚠️  {param_name}:")
                print(f"      Failed trials avg:    {fail_mean:.4f}")
                print(f"      Completed trials avg: {complete_mean:.4f}")
                print(f"      Difference: {diff_pct:.1f}%")
                print()
    
    # Look for trial output directories
    print(f"\n📁 CHECKING TRIAL OUTPUT DIRECTORIES:")
    
    if output_dir.exists():
        trial_dirs = sorted(output_dir.glob("trial_*"))
        print(f"   Found {len(trial_dirs)} trial directories")
        
        # Check for log files
        failed_with_logs = []
        for trial_num in failed['number'].values:
            trial_dir = output_dir / f"trial_{trial_num:04d}"
            if trial_dir.exists():
                # Look for error logs
                log_files = list(trial_dir.glob("*.log")) + list(trial_dir.glob("*.txt"))
                if log_files:
                    failed_with_logs.append((trial_num, trial_dir, log_files))
        
        if failed_with_logs:
            print(f"\n   ✅ Found logs for {len(failed_with_logs)} failed trials:")
            for trial_num, trial_dir, log_files in failed_with_logs[:5]:  # Show first 5
                print(f"\n   Trial {trial_num}:")
                print(f"      Directory: {trial_dir}")
                print(f"      Logs: {[f.name for f in log_files]}")
                
                # Try to read last few lines of log
                for log_file in log_files:
                    if log_file.stat().st_size > 0:
                        try:
                            with open(log_file, 'r') as f:
                                lines = f.readlines()
                                if lines:
                                    print(f"\n      Last 5 lines from {log_file.name}:")
                                    for line in lines[-5:]:
                                        print(f"         {line.rstrip()}")
                        except Exception as e:
                            print(f"      Could not read {log_file.name}: {e}")
    else:
        print(f"   ⚠️  Output directory not found: {output_dir}")
    
    # Common failure patterns to check
    print(f"\n🔍 COMMON FAILURE PATTERNS TO CHECK:")
    print("   1. Out of Memory (OOM):")
    print("      - Check if GPU memory is sufficient")
    print("      - Try reducing batch_size to 2 or 1")
    print("      - Monitor with: watch -n 1 nvidia-smi")
    print()
    print("   2. Data Loading Errors:")
    print("      - Missing files or corrupted data")
    print("      - Incorrect file paths")
    print("      - Run: python -m dataloader_train")
    print()
    print("   3. Numerical Instabilities:")
    print("      - NaN/Inf in loss values")
    print("      - Gradient explosion")
    print("      - Check discriminator balance")
    print()
    print("   4. Disk Space Issues:")
    print("      - Root partition full")
    print("      - Cannot write checkpoints")
    print("      - Run: df -h")
    
    # Recommendations
    print(f"\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    print("\n1. 🔧 DEBUG ONE TRIAL MANUALLY:")
    print("   Run a single trial with the most common failed parameters")
    print("   to see the exact error message:")
    
    if len(failed) > 0:
        # Get most common parameter combination from failures
        common_fail = failed.iloc[0]
        print("\n   python -c \"")
        print("   from ct_phase_training import CTPhaseTrainer")
        print("   config = {")
        print(f"       'learning_rate': {common_fail['params_disc_lr_multiplier']*2e-4:.6f},")
        print(f"       'lambda_cycle': {common_fail['params_lambda_cycle']:.3f},")
        print(f"       'lambda_mse': {common_fail['params_lambda_mse']:.3f},")
        print("       # ... other params ...")
        print("   }")
        print("   # Run training and check for errors")
        print("   \"")
    
    print("\n2. 📊 CHECK SYSTEM RESOURCES:")
    print("   - GPU memory: nvidia-smi")
    print("   - Disk space: df -h")
    print("   - RAM usage: free -h")
    print("   - Monitor during trial: watch -n 1 'nvidia-smi && df -h'")
    
    print("\n3. 🧪 SIMPLIFY TO DEBUG:")
    print("   - Use smaller patch_size (32x32)")
    print("   - Use batch_size=1")
    print("   - Reduce trial_epochs to 5")
    print("   - Disable augmentation")
    
    print("\n4. 📝 ENABLE VERBOSE LOGGING:")
    print("   - Set logging level to DEBUG")
    print("   - Add try-except blocks with detailed error messages")
    print("   - Save error stack traces to files")
    
    # Create a debug script
    debug_script_path = output_dir / "debug_single_trial.py"
    debug_script = '''
"""Debug a single Optuna trial to find errors"""
import torch
from ct_phase_training import CTPhaseTrainer
from dataloader_train import create_data_pairs, load_phase_mapping
from torch.utils.data import DataLoader
from ct_phase_training import CTPhaseDataset

# Use parameters from a failed trial
config = {
    'data_dir': '../ncct_cect/vindr_ds/registered_cases',
    'labels_csv': '../ncct_cect/vindr_ds/labels.csv',
    'output_dir': '../ncct_cect/vindr_ds/debug_trial',
    'patch_size': (64, 64),
    'patch_depth': 7,
    'overlap_ratio': 0.75,
    'batch_size': 1,  # Small for debugging
    'learning_rate': 2e-4,
    'epochs': 5,  # Just a few epochs
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Default parameters
    'lambda_cycle': 10.0,
    'lambda_mse': 100.0,
    'lambda_focal': 5.0,
    'lambda_adv': 1.0,
}

print("Loading data...")
phase_mapping = load_phase_mapping(config['labels_csv'])
data_splits = create_data_pairs(config['data_dir'], phase_mapping)

print("Creating datasets...")
train_dataset = CTPhaseDataset(
    data_splits['train'][:1],  # Just one pair for debugging
    patch_size=config['patch_size'],
    patch_depth=config['patch_depth'],
    overlap_ratio=0.5,
    augment=False
)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

print("Initializing trainer...")
trainer = CTPhaseTrainer(config)

print("Training for a few steps...")
try:
    for i, batch in enumerate(train_loader):
        if i >= 5:  # Just 5 batches
            break
        losses = trainer.train_step(batch)
        print(f"Batch {i}: {losses}")
    print("✅ Debug training successful!")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
'''
    
    try:
        with open(debug_script_path, 'w') as f:
            f.write(debug_script)
        print(f"\n💾 Created debug script: {debug_script_path}")
        print(f"   Run: python {debug_script_path}")
    except Exception as e:
        print(f"⚠️  Could not create debug script: {e}")


if __name__ == "__main__":
    investigate_failures()