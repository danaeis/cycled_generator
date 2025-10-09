"""
Debug DataLoader Collation Issues
==================================
Quick diagnostic script to identify and fix the DataLoader error.
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def custom_collate_fn(batch):
    """Safe collate function that handles varying mask dictionaries."""
    try:
        # Stack image tensors (ensure contiguous)
        sources = torch.stack([item['source'].contiguous() for item in batch])
        targets = torch.stack([item['target'].contiguous() for item in batch])
        
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
                    organ_masks.append(item['masks'][organ].contiguous())
                else:
                    organ_masks.append(torch.zeros_like(item['source']))
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


def test_dataloader(config):
    """Test dataloader with diagnostic output."""
    
    print("\n" + "="*80)
    print("DATALOADER DIAGNOSTIC TEST")
    print("="*80)
    
    # Import your modules
    try:
        from ct_phase_training import CTPhaseDataset
        from dataloader_train import create_data_pairs, load_phase_mapping
    except ImportError as e:
        print(f"❌ Error importing modules: {e}")
        return False
    
    # Load data
    print("\n1. Loading data...")
    try:
        phase_mapping = None
        if Path(config['labels_csv']).exists():
            phase_mapping = load_phase_mapping(config['labels_csv'])
        
        data_splits = create_data_pairs(config['data_dir'], phase_mapping=phase_mapping)
        
        if not data_splits['train']:
            print("❌ No training data found!")
            return False
        
        print(f"✓ Found {len(data_splits['train'])} training pairs")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Create dataset
    print("\n2. Creating dataset...")
    try:
        # Use small subset for testing
        test_pairs = data_splits['train'][:5]  # Only 5 pairs for quick test
        
        dataset = CTPhaseDataset(
            test_pairs,
            patch_size=config['patch_size'],
            patch_depth=config['patch_depth'],
            overlap_ratio=0.5,
            augment=False
        )
        
        print(f"✓ Dataset created with {len(dataset)} patches")
        
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test individual samples
    print("\n3. Testing individual samples...")
    try:
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            print(f"\n  Sample {i}:")
            print(f"    source: {sample['source'].shape}, dtype={sample['source'].dtype}")
            print(f"    target: {sample['target'].shape}, dtype={sample['target'].dtype}")
            print(f"    source_phase: {sample['source_phase']}")
            print(f"    target_phase: {sample['target_phase']}")
            print(f"    masks: {list(sample['masks'].keys())}")
            print(f"    is_contiguous (source): {sample['source'].is_contiguous()}")
            print(f"    is_contiguous (target): {sample['target'].is_contiguous()}")
            
            # Check for issues
            if not sample['source'].is_contiguous():
                print("    ⚠️  WARNING: source tensor is not contiguous!")
            if not sample['target'].is_contiguous():
                print("    ⚠️  WARNING: target tensor is not contiguous!")
        
        print("\n✓ Individual samples work correctly")
        
    except Exception as e:
        print(f"❌ Error loading individual samples: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test DataLoader WITHOUT custom collate (to reproduce error)
    print("\n4. Testing DataLoader WITHOUT custom collate_fn...")
    try:
        loader_default = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            drop_last=True
        )
        
        batch = next(iter(loader_default))
        print(f"✓ Default collate works!")
        print(f"  Batch source shape: {batch['source'].shape}")
        
    except Exception as e:
        print(f"⚠️  Default collate FAILED (expected): {e}")
        print("   This is the error you're seeing. Now trying custom collate...")
    
    # Test DataLoader WITH custom collate
    print("\n5. Testing DataLoader WITH custom collate_fn...")
    try:
        loader_custom = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            drop_last=True,
            collate_fn=custom_collate_fn
        )
        
        batch = next(iter(loader_custom))
        print(f"✓ Custom collate works!")
        print(f"  Batch source shape: {batch['source'].shape}")
        print(f"  Batch target shape: {batch['target'].shape}")
        print(f"  Batch masks: {list(batch['masks'].keys())}")
        
        # Try loading a few more batches
        for i, batch in enumerate(loader_custom):
            if i >= 2:
                break
            print(f"  Batch {i+1} loaded successfully")
        
        print("\n✓ All batches loaded successfully with custom collate!")
        
    except Exception as e:
        print(f"❌ Custom collate FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test with multiple workers
    print("\n6. Testing with num_workers=2...")
    try:
        loader_workers = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=2,
            drop_last=True,
            collate_fn=custom_collate_fn,
            persistent_workers=False
        )
        
        for i, batch in enumerate(loader_workers):
            if i >= 2:
                break
            print(f"  Batch {i} loaded with workers")
        
        print("✓ Multiple workers work correctly")
        
    except Exception as e:
        print(f"⚠️  Multiple workers failed: {e}")
        print("   Recommendation: Use num_workers=0 for now")
    
    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)
    print("\n✅ SOLUTION: Use custom_collate_fn in your DataLoader!")
    print("\nTo fix the issue in your code:")
    print("1. Copy the custom_collate_fn from dataset_collate_fix.py")
    print("2. Update DataLoader initialization to include: collate_fn=custom_collate_fn")
    print("3. Set num_workers=0 initially, increase to 2-4 once stable")
    print("4. Always use drop_last=True for training")
    
    return True


def main():
    """Run diagnostic test."""
    
    config = {
        'data_dir': '../ncct_cect/vindr_ds/registered_cases',
        'labels_csv': '../ncct_cect/vindr_ds/labels.csv',
        'patch_size': (64, 64),
        'patch_depth': 7
    }
    
    print("\nChecking if data directory exists...")
    if not Path(config['data_dir']).exists():
        print(f"❌ Data directory not found: {config['data_dir']}")
        print("Please update the 'data_dir' in this script to point to your data.")
        return
    
    success = test_dataloader(config)
    
    if success:
        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print("\n1. The custom_collate_fn has been verified to work")
        print("2. Update your ct_phase_training.py to use it:")
        print("\n   from dataset_collate_fix import custom_collate_fn")
        print("\n   train_loader = DataLoader(..., collate_fn=custom_collate_fn)")
        print("\n3. Run optuna_tuning.py again - it should work now!")
    else:
        print("\n❌ Diagnostics failed. Please check the error messages above.")


if __name__ == "__main__":
    main()