"""
Data Loading Helper for CT Phase Training
==========================================
Helper functions to load and prepare your CT data for training.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


def load_phase_mapping(labels_csv_path: str) -> Dict[str, Dict[str, str]]:
    """
    Load phase mapping from CSV file.
    
    Args:
        labels_csv_path: Path to labels CSV
        
    Returns:
        Mapping: {case_uid: {series_uid: phase_label}}
    """
    df = pd.read_csv(labels_csv_path)
    
    phase_mapping = {}
    for _, row in df.iterrows():
        case_uid = row['StudyInstanceUID']
        series_uid = row['SeriesInstanceUID']
        phase_label = row['Label'].lower()
        
        if case_uid not in phase_mapping:
            phase_mapping[case_uid] = {}
        phase_mapping[case_uid][series_uid] = phase_label
    
    return phase_mapping


def infer_phase_from_filename(filename: str) -> str:
    """Infer CT phase from filename."""
    filename = filename.lower()
    
    phase_keywords = {
        'non-contrast': ['noncontrast', 'non-contrast', 'pre', 'baseline', 'native', 'nc'],
        'arterial': ['arterial', 'art', 'early', 'phase1', 'p1'],
        'portal': ['portal', 'venous', 'pv', 'phase2', 'p2', 'late'],
        'delayed': ['delayed', 'delay', 'equilibrium', 'phase3', 'p3']
    }
    
    for phase, keywords in phase_keywords.items():
        if any(keyword in filename for keyword in keywords):
            return phase
    
    return 'unknown'


def create_data_pairs(
    data_dir: str,
    phase_mapping: Dict = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> Dict[str, List[Dict]]:
    """
    Create training/validation/test data pairs from directory structure.
    
    Args:
        data_dir: Root directory with case folders
        phase_mapping: Optional phase mapping from CSV
        test_size: Test set proportion
        val_size: Validation set proportion
        random_state: Random seed
        
    Returns:
        Dictionary with 'train', 'val', 'test' data pairs
    """
    data_dir = Path(data_dir)
    
    # Collect all cases and their available files
    cases = {}
    
    for case_dir in data_dir.iterdir():
        if not case_dir.is_dir():
            continue
            
        case_id = case_dir.name
        cases[case_id] = {}
        
        # Find all registered image files
        for nii_file in case_dir.glob("*_registered.nii.gz"):
            if "_seg" in str(nii_file):
                continue  # Skip segmentation files
                
            # Extract series ID and phase
            filename = nii_file.stem.replace("_registered", "").replace(".nii", "")
            parts = filename.split('_')
            
            if len(parts) >= 2:
                series_id = parts[1]
                
                # Determine phase
                if phase_mapping and case_id in phase_mapping and series_id in phase_mapping[case_id]:
                    phase = phase_mapping[case_id][series_id]
                else:
                    phase = infer_phase_from_filename(filename)
                
                cases[case_id][phase] = {
                    'image': nii_file,
                    'series_id': series_id
                }
                
                # Check for segmentation
                seg_file = case_dir / f"{filename}_registered_seg.nii.gz"
                if seg_file.exists():
                    cases[case_id][phase]['segmentation'] = seg_file
    
    # Filter valid cases (need non-contrast + at least one contrast phase)
    valid_cases = []
    for case_id, phases in cases.items():
        if 'non-contrast' in phases:
            contrast_phases = [p for p in phases.keys() if p != 'non-contrast']
            if contrast_phases:
                valid_cases.append((case_id, phases))
    
    logger.info(f"Found {len(valid_cases)} valid cases")
    
    # Split cases into train/val/test
    case_ids = [case[0] for case in valid_cases]
    train_ids, temp_ids = train_test_split(
        case_ids, 
        test_size=test_size+val_size, 
        random_state=random_state
    )
    val_ids, test_ids = train_test_split(
        temp_ids, 
        test_size=test_size/(test_size+val_size), 
        random_state=random_state
    )
    
    # Create data pairs for each split
    def make_pairs(case_subset):
        pairs = []
        for case_id in case_subset:
            case_data = dict(valid_cases)[case_id]
            nc_info = case_data['non-contrast']
            
            # Create pairs: non-contrast -> each contrast phase
            for phase, phase_info in case_data.items():
                if phase != 'non-contrast':
                    pairs.append({
                        'source_path': nc_info['image'],
                        'target_path': phase_info['image'],
                        'source_phase': 'non-contrast',
                        'target_phase': phase,
                        'case_id': case_id,
                        'source_series': nc_info['series_id'],
                        'target_series': phase_info['series_id'],
                        'source_seg': nc_info.get('segmentation'),
                        'target_seg': phase_info.get('segmentation')
                    })
        return pairs
    
    splits = {
        'train': make_pairs(train_ids),
        'val': make_pairs(val_ids),
        'test': make_pairs(test_ids)
    }
    
    logger.info(f"Data splits created:")
    logger.info(f"  Train: {len(train_ids)} cases, {len(splits['train'])} pairs")
    logger.info(f"  Val: {len(val_ids)} cases, {len(splits['val'])} pairs")
    logger.info(f"  Test: {len(test_ids)} cases, {len(splits['test'])} pairs")
    
    return splits


# ============================================================================
# DEBUGGING UTILITIES
# ============================================================================

def debug_data_loading(config: Dict):
    """Debug data loading to ensure everything works."""
    
    print("=" * 80)
    print("DEBUGGING DATA LOADING")
    print("=" * 80)
    
    # Load phase mapping
    phase_mapping = None
    if Path(config['labels_csv']).exists():
        try:
            phase_mapping = load_phase_mapping(config['labels_csv'])
            print(f"✓ Loaded phase mapping for {len(phase_mapping)} cases")
        except Exception as e:
            print(f"✗ Could not load phase mapping: {e}")
    
    # Create data pairs
    try:
        data_splits = create_data_pairs(
            config['data_dir'],
            phase_mapping=phase_mapping
        )
        print(f"✓ Created data splits successfully")
        
        # Print sample pair
        if data_splits['train']:
            sample = data_splits['train'][0]
            print(f"\nSample training pair:")
            print(f"  Case: {sample['case_id']}")
            print(f"  Source: {sample['source_phase']} -> {sample['source_path'].name}")
            print(f"  Target: {sample['target_phase']} -> {sample['target_path'].name}")
            if sample.get('target_seg'):
                print(f"  Has segmentation: ✓")
        
        return data_splits
        
    except Exception as e:
        print(f"✗ Error creating data splits: {e}")
        import traceback
        traceback.print_exc()
        return None


def debug_dataset(dataset, num_samples: int = 3):
    """Debug dataset by loading a few samples."""
    
    print("\n" + "=" * 80)
    print("DEBUGGING DATASET")
    print("=" * 80)
    
    print(f"Dataset length: {len(dataset)}")
    
    for i in range(min(num_samples, len(dataset))):
        try:
            sample = dataset[i]
            print(f"\nSample {i}:")
            print(f"  Source shape: {sample['source'].shape}")
            print(f"  Target shape: {sample['target'].shape}")
            print(f"  Source phase: {sample['source_phase']}")
            print(f"  Target phase: {sample['target_phase']}")
            print(f"  Phase indices: {sample['source_phase_idx']} -> {sample['target_phase_idx']}")
            print(f"  Masks: {list(sample['masks'].keys())}")
            print(f"  Case ID: {sample['case_id']}")
            
            # Check value ranges
            print(f"  Source range: [{sample['source'].min():.2f}, {sample['source'].max():.2f}]")
            print(f"  Target range: [{sample['target'].min():.2f}, {sample['target'].max():.2f}]")
            
        except Exception as e:
            print(f"✗ Error loading sample {i}: {e}")
            import traceback
            traceback.print_exc()


def debug_model(config: Dict):
    """Debug model architecture."""
    import torch
    from ct_phase_training import PhaseConditionedGenerator, Discriminator3D
    
    print("\n" + "=" * 80)
    print("DEBUGGING MODEL")
    print("=" * 80)
    
    device = torch.device(config['device'])
    
    try:
        # Test generator
        generator = PhaseConditionedGenerator(num_phases=4).to(device)
        
        batch_size = 2
        depth = config['patch_depth']
        height, width = config['patch_size']
        
        test_input = torch.randn(batch_size, 1, depth, height, width).to(device)
        test_phase = torch.tensor([1, 2]).to(device)  # arterial, portal
        
        print(f"Testing Generator:")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Phase indices: {test_phase}")
        
        output = generator(test_input, test_phase)
        print(f"  Output shape: {output.shape}")
        print(f"  ✓ Generator works!")
        
        # Test discriminator
        discriminator = Discriminator3D().to(device)
        disc_output = discriminator(output)
        print(f"\nTesting Discriminator:")
        print(f"  Discriminator output shape: {disc_output.shape}")
        print(f"  ✓ Discriminator works!")
        
        # Test full forward pass
        print(f"\nTesting full forward pass:")
        generated = generator(test_input, test_phase)
        reconstructed = generator(generated, torch.tensor([0, 0]).to(device))  # back to non-contrast
        
        print(f"  Source -> Target: {test_input.shape} -> {generated.shape}")
        print(f"  Target -> Reconstructed: {generated.shape} -> {reconstructed.shape}")
        print(f"  ✓ Cycle consistency works!")
        
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# COMPLETE TRAINING SCRIPT
# ============================================================================

def run_full_training():
    """Complete training script with all debugging."""
    import torch
    from torch.utils.data import DataLoader
    from ct_phase_training import CTPhaseDataset, CTPhaseTrainer
    
    # Configuration
    config = {
        'data_dir': '../ncct_cect/vindr_ds/test_registered_cases',
        'labels_csv': '../ncct_cect/vindr_ds/labels.csv',
        'output_dir': '../ncct_cect/vindr_ds/loss_handled_training',
        
        'patch_size': (64, 64),
        'patch_depth': 16,
        'overlap_ratio': 0.75,
        'disc_lr_multiplier':2.0,

        'batch_size': 1,
        'learning_rate': 2e-4,
        'epochs': 100,
        
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("=" * 80)
    print("CT PHASE GENERATION - COMPLETE TRAINING PIPELINE")
    print("=" * 80)
    
    # Step 1: Debug data loading
    print("\nStep 1: Data Loading")
    data_splits = debug_data_loading(config)
    if not data_splits:
        print("✗ Data loading failed. Exiting.")
        return
    
    # Step 2: Create datasets
    print("\nStep 2: Creating Datasets")
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
    
    # Step 3: Debug dataset
    debug_dataset(train_dataset, num_samples=2)
    
    # Step 4: Debug model
    print("\nStep 3: Model Architecture")
    if not debug_model(config):
        print("✗ Model test failed. Exiting.")
        return
    
    # Step 5: Create dataloaders
    print("\nStep 4: Creating DataLoaders")
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
    
    print(f"✓ Train loader: {len(train_loader)} batches")
    print(f"✓ Val loader: {len(val_loader)} batches")
    
    # Step 6: Initialize trainer
    print("\nStep 5: Initializing Trainer")
    trainer = CTPhaseTrainer(config)
    
    # Step 7: Start training
    print("\nStep 6: Starting Training")
    print("=" * 80)
    
    try:
        trainer.train(train_loader, val_loader, config['epochs'])
        print("\n✓ Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
        trainer.save_checkpoint(float('inf'), is_best=False)
        
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_full_training()