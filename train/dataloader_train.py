"""
Data Loading Helper for CT Phase Training
==========================================
Helper functions to load and prepare your CT data for training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple
import logging
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from torch.utils.data import DataLoader
import re

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
        for nii_file in case_dir.glob("*_deformable.nii.gz"):
            if "_seg" in str(nii_file):
                continue  # Skip segmentation files
                
            # Extract series ID and phase
            filename = nii_file.stem.replace("_deformable", "").replace(".nii", "")
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
                seg_file = case_dir / f"{filename}_deformable_seg.nii.gz"
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

def visualize_single_patch(
    sample: Dict,
    save_path: Path,
    show_all_slices: bool = False
):
    """
    Visualize a single patch with detailed information.
    
    Args:
        sample: Sample dictionary from dataset
        save_path: Path to save visualization
        show_all_slices: If True, show all depth slices, else just middle slice
    """
    source = sample['source'].squeeze().numpy()  # [D, H, W]
    target = sample['target'].squeeze().numpy()  # [D, H, W]
    
    depth, height, width = source.shape
    
    if show_all_slices:
        # Show all slices in grid
        n_slices = depth
        fig, axes = plt.subplots(2, n_slices, figsize=(3*n_slices, 6))
        
        for i in range(n_slices):
            # Source
            im0 = axes[0, i].imshow(source[i], cmap='gray', vmin=-3, vmax=3)
            axes[0, i].set_title(f'Source Slice {i}', fontsize=10)
            axes[0, i].axis('off')
            plt.colorbar(im0, ax=axes[0, i], fraction=0.046)
            
            # Target
            im1 = axes[1, i].imshow(target[i], cmap='gray', vmin=-3, vmax=3)
            axes[1, i].set_title(f'Target Slice {i}', fontsize=10)
            axes[1, i].axis('off')
            plt.colorbar(im1, ax=axes[1, i], fraction=0.046)
    else:
        # Show only middle slice with more detail
        mid_slice = depth // 2
        
        fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Row 1: Source and Target middle slices
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(source[mid_slice], cmap='gray', vmin=-3, vmax=3)
        ax1.set_title(f'Source: {sample["source_phase"]}', fontsize=12, fontweight='bold')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(target[mid_slice], cmap='gray', vmin=-3, vmax=3)
        ax2.set_title(f'Target: {sample["target_phase"]}', fontsize=12, fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        
        # Difference map
        ax3 = fig.add_subplot(gs[0, 2])
        diff = np.abs(source[mid_slice] - target[mid_slice])
        im3 = ax3.imshow(diff, cmap='hot', vmin=0, vmax=2)
        ax3.set_title('Absolute Difference', fontsize=12, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046)
        
        # Row 2: Histogram comparisons
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.hist(source[mid_slice].flatten(), bins=50, alpha=0.7, label='Source', color='blue')
        ax4.hist(target[mid_slice].flatten(), bins=50, alpha=0.7, label='Target', color='red')
        ax4.set_xlabel('Intensity Value')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Intensity Distributions', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Center profile (vertical)
        ax5 = fig.add_subplot(gs[1, 1])
        center_x = width // 2
        ax5.plot(source[mid_slice, :, center_x], label='Source', linewidth=2)
        ax5.plot(target[mid_slice, :, center_x], label='Target', linewidth=2)
        ax5.set_xlabel('Y Position')
        ax5.set_ylabel('Intensity')
        ax5.set_title(f'Vertical Profile (X={center_x})', fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Center profile (horizontal)
        ax6 = fig.add_subplot(gs[1, 2])
        center_y = height // 2
        ax6.plot(source[mid_slice, center_y, :], label='Source', linewidth=2)
        ax6.plot(target[mid_slice, center_y, :], label='Target', linewidth=2)
        ax6.set_xlabel('X Position')
        ax6.set_ylabel('Intensity')
        ax6.set_title(f'Horizontal Profile (Y={center_y})', fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Row 3: All slices montage
        ax7 = fig.add_subplot(gs[2, :])
        montage_source = np.hstack([source[i] for i in range(depth)])
        montage_target = np.hstack([target[i] for i in range(depth)])
        montage = np.vstack([montage_source, montage_target])
        im7 = ax7.imshow(montage, cmap='gray', vmin=-3, vmax=3, aspect='auto')
        ax7.set_title('All Slices Montage (Top: Source, Bottom: Target)', fontweight='bold')
        ax7.set_ylabel('Source (top) / Target (bottom)')
        ax7.set_xlabel('Concatenated Slices')
        slice_positions = [i * width + width//2 for i in range(depth)]
        ax7.set_xticks(slice_positions)
        ax7.set_xticklabels([f'S{i}' for i in range(depth)], fontsize=8)
        plt.colorbar(im7, ax=ax7, fraction=0.02)
    
    # Add overall title with metadata
    title = (f'Patch Visualization - Case: {sample["case_id"]}\n'
             f'{sample["source_phase"]} → {sample["target_phase"]} | '
             f'Shape: {source.shape} | '
             f'Patch Center: {sample.get("patch_center", "N/A")}\n'
             f'Value Range - Source: [{source.min():.2f}, {source.max():.2f}], '
             f'Target: [{target.min():.2f}, {target.max():.2f}]')
    
    fig.suptitle(title, fontsize=12, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved patch visualization to {save_path}")

def debug_dataset_with_visualization(
    dataset, 
    num_samples: int = 5,
    output_dir: str = './debug_patches'
):
    """
    Debug dataset by loading and visualizing samples.
    
    Args:
        dataset: CTPhaseDataset instance
        num_samples: Number of samples to visualize
        output_dir: Directory to save visualizations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("DEBUGGING DATASET WITH VISUALIZATION")
    print("=" * 80)
    
    print(f"Dataset length: {len(dataset)}")
    print(f"Saving visualizations to: {output_dir}")
    
    # Statistics accumulators
    all_source_ranges = []
    all_target_ranges = []
    all_shapes = []
    
    for i in range(min(num_samples, len(dataset))):
        try:
            sample = dataset[i]
            
            print(f"\n{'='*60}")
            print(f"Sample {i}:")
            print(f"  Case ID: {sample['case_id']}")
            print(f"  Source shape: {sample['source'].shape}")
            print(f"  Target shape: {sample['target'].shape}")
            print(f"  Source phase: {sample['source_phase']}")
            print(f"  Target phase: {sample['target_phase']}")
            print(f"  Phase indices: {sample['source_phase_idx']} → {sample['target_phase_idx']}")
            print(f"  Patch center: {sample.get('patch_center', 'N/A')}")
            
            # Check value ranges
            source_min = sample['source'].min().item()
            source_max = sample['source'].max().item()
            target_min = sample['target'].min().item()
            target_max = sample['target'].max().item()
            
            print(f"  Source range: [{source_min:.3f}, {source_max:.3f}]")
            print(f"  Target range: [{target_min:.3f}, {target_max:.3f}]")
            
            all_source_ranges.append((source_min, source_max))
            all_target_ranges.append((target_min, target_max))
            all_shapes.append(sample['source'].shape)
            
            # Check masks
            if sample['masks']:
                print(f"  Available masks: {list(sample['masks'].keys())}")
                for organ, mask in sample['masks'].items():
                    mask_sum = mask.sum().item()
                    if mask_sum > 0:
                        print(f"    - {organ}: {mask_sum:.0f} voxels ({100*mask_sum/mask.numel():.1f}%)")
            else:
                print(f"  No masks available")
            
            # Visualize this patch
            save_path = output_dir / f'patch_{i:03d}_case_{sample["case_id"]}.png'
            visualize_single_patch(sample, save_path, show_all_slices=False)
            
            # Also save all slices version for first few samples
            if i < 2:
                save_path_all = output_dir / f'patch_{i:03d}_allslices_case_{sample["case_id"]}.png'
                visualize_single_patch(sample, save_path_all, show_all_slices=True)
            
        except Exception as e:
            print(f"✗ Error loading sample {i}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print overall statistics
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    
    if all_source_ranges:
        print(f"\nValue Ranges across all samples:")
        print(f"  Source min: {min(r[0] for r in all_source_ranges):.3f}")
        print(f"  Source max: {max(r[1] for r in all_source_ranges):.3f}")
        print(f"  Target min: {min(r[0] for r in all_target_ranges):.3f}")
        print(f"  Target max: {max(r[1] for r in all_target_ranges):.3f}")
    
    if all_shapes:
        unique_shapes = set(all_shapes)
        print(f"\nUnique patch shapes: {unique_shapes}")
    
    print(f"\n✓ Saved {min(num_samples, len(dataset))} patch visualizations to {output_dir}")

def create_patch_coverage_map(
    dataset,
    case_idx: int = 0,
    output_dir: str = './debug_patches'
):
    """
    Visualize where patches are extracted from in the full volume.
    
    Args:
        dataset: CTPhaseDataset instance
        case_idx: Which case to visualize (index into data_pairs)
        output_dir: Directory to save visualization
    """
    import nibabel as nib
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print(f"CREATING PATCH COVERAGE MAP FOR CASE {case_idx}")
    print("=" * 80)
    
    # Get all patches for this case
    case_patches = [
        (idx, coords) for idx, coords in enumerate(dataset.patch_coords)
        if coords[0] == case_idx
    ]
    
    if not case_patches:
        print(f"No patches found for case {case_idx}")
        return
    
    print(f"Found {len(case_patches)} patches for this case")
    
    # Load the volume to get dimensions
    pair_data = dataset.data_pairs[case_idx]
    source_vol = nib.load(pair_data['source_path']).get_fdata()
    depth, height, width = source_vol.shape
    
    print(f"Volume shape: {source_vol.shape}")
    
    # Create coverage map
    coverage_map = np.zeros((height, width), dtype=np.int32)
    
    # Mark each patch location
    for _, (_, center_z, y_start, x_start) in case_patches:
        y_end = y_start + dataset.patch_size[0]
        x_end = x_start + dataset.patch_size[1]
        coverage_map[y_start:y_end, x_start:x_end] += 1
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Middle slice of volume
    mid_slice = depth // 2
    source_slice = source_vol[mid_slice]
    
    ax1 = axes[0]
    im1 = ax1.imshow(source_slice, cmap='gray', vmin=-100, vmax=300)
    ax1.set_title(f'Original Volume (Slice {mid_slice}/{depth})', fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    # Coverage map
    ax2 = axes[1]
    im2 = ax2.imshow(coverage_map, cmap='hot', interpolation='nearest')
    ax2.set_title(f'Patch Coverage Map\n(Max overlap: {coverage_map.max()})', 
                  fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    # Overlay
    ax3 = axes[2]
    ax3.imshow(source_slice, cmap='gray', vmin=-100, vmax=300, alpha=0.7)
    overlay = np.ma.masked_where(coverage_map == 0, coverage_map)
    im3 = ax3.imshow(overlay, cmap='hot', alpha=0.5, interpolation='nearest')
    
    # Draw patch rectangles for first few patches
    for i, (_, (_, center_z, y_start, x_start)) in enumerate(case_patches[:20]):
        y_end = y_start + dataset.patch_size[0]
        x_end = x_start + dataset.patch_size[1]
        color = 'green' if i < 5 else 'yellow'
        rect = plt.Rectangle((x_start, y_start), 
                            dataset.patch_size[1], dataset.patch_size[0],
                            fill=False, edgecolor=color, linewidth=1)
        ax3.add_patch(rect)
        
        if i < 5:
            # Add patch number
            ax3.text(x_start, y_start, str(i), color='white', 
                    fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    ax3.set_title(f'Coverage Overlay\n(First 20 patches shown)', fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    fig.suptitle(f'Patch Extraction Coverage - Case {case_idx}: {pair_data["case_id"]}\n'
                 f'Patch Size: {dataset.patch_size}, Depth: {dataset.patch_depth}, '
                 f'Overlap: {dataset.overlap_ratio}',
                 fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    save_path = output_dir / f'patch_coverage_case_{case_idx}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved coverage map to {save_path}")
    
    # Print statistics
    print(f"\nCoverage Statistics:")
    print(f"  Total patches: {len(case_patches)}")
    print(f"  Max overlap: {coverage_map.max()}x")
    print(f"  Mean overlap: {coverage_map[coverage_map > 0].mean():.2f}x")
    print(f"  Coverage: {100 * (coverage_map > 0).sum() / coverage_map.size:.1f}%")

def debug_model(config: Dict):
    """Debug model architecture."""
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
    """Complete training script with all debugging and checkpoint resuming."""
    from ct_phase_training import CTPhaseDataset, CTPhaseTrainer
    
    # Configuration
    config = {
        'data_dir': '../ncct_cect/vindr_ds/deformable_registered_bspline',
        'labels_csv': '../ncct_cect/vindr_ds/labels.csv',
        'output_dir': '../ncct_cect/vindr_ds/patch_bspline_training',
        
        'patch_size': (128, 192),
        'patch_depth': 15,
        'overlap_ratio': 0.5,

        'pad_mode': 'constant',            # Padding mode
        'save_nifti': True,               # Enable NIfTI export
    
        'disc_lr_multiplier':1.599,
        'lambda_cycle': 10,
        
        "lambda_mse_initial": 1.0,      
        "lambda_mse_final": 100.0,      
        "mse_warmup_epochs": 50,

        'lambda_focal': 5,
        'lambda_adv': 10,
        # Training stability
        'adv_warmup_epochs': 8,
        'disc_updates_per_gen': 2,
        'real_label_smoothing': 0.896,
        'fake_label_smoothing': 0.118,

        'batch_size': 8,
        'learning_rate': 2e-4,
        'epochs': 100,
        
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        # Debug options
        'debug_patches': False,
        'debug_output_dir': './debug_patches',
        'keep_last_n_checkpoints': 3,  # Only keep last 3 checkpoints
        'save_samples_interval': 1,     # Save samples every 5 epochs instead of 1
        'keep_last_n_sample_epochs': 5, # Keep samples from last 5 epochs only
    }
    
    print("=" * 80)
    print("CT PHASE GENERATION - COMPLETE TRAINING PIPELINE")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for existing checkpoints
    start_epoch = 0
    checkpoint_path = None
    checkpoint_files = list(output_dir.glob('checkpoint_epoch_*.pth'))
    
    if checkpoint_files:
        # Find the latest checkpoint based on epoch number
        epoch_numbers = []
        for ckpt in checkpoint_files:
            match = re.search(r'checkpoint_epoch_(\d+)\.pth', ckpt.name)
            if match:
                epoch_numbers.append(int(match.group(1)))
        
        if epoch_numbers:
            latest_epoch = max(epoch_numbers)
            checkpoint_path = output_dir / f'checkpoint_epoch_{latest_epoch}.pth'
            start_epoch = latest_epoch + 1
            print(f"✓ Found checkpoint at epoch {latest_epoch}, resuming from epoch {start_epoch}")
        else:
            print("✓ No valid checkpoint files found, starting from scratch")
    else:
        print("✓ No checkpoint files found, starting from scratch")
    
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
        augment=False
    )
    
    val_dataset = CTPhaseDataset(
        data_splits['val'],
        patch_size=config['patch_size'],
        patch_depth=config['patch_depth'],
        overlap_ratio=0.5,
        augment=False
    )
    
    # Step 3: Debug dataset
    if config.get('debug_patches', True):
        print("\nStep 3: Visualizing Patches (NEW!)")
        print("=" * 80)
        
        debug_dataset_with_visualization(
            train_dataset,
            num_samples=config.get('num_debug_samples', 5),
            output_dir=config.get('debug_output_dir', './debug_patches')
        )
        
        # Also create coverage map for first case
        if len(train_dataset.data_pairs) > 0:
            create_patch_coverage_map(
                train_dataset,
                case_idx=0,
                output_dir=config.get('debug_output_dir', './debug_patches')
            )
        
        print("\n" + "=" * 80)
        print("PATCH DEBUGGING COMPLETE!")
        print("=" * 80)
        print(f"\nCheck the debug output directory: {config.get('debug_output_dir', './debug_patches')}")
        print("Files generated:")
        print("  - patch_XXX_case_YYY.png: Individual patch visualizations")
        print("  - patch_XXX_allslices_case_YYY.png: All depth slices")
        print("  - patch_coverage_case_X.png: Spatial coverage map")
        print("\n" + "=" * 80)
    
    # Step 4: Debug model
    # print("\nStep 4: Model Architecture")
    # if not debug_model(config):
    #     print("✗ Model test failed. Exiting.")
    #     return
    
    # Step 5: Create dataloaders
    print("\nStep 5: Creating DataLoaders")
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
    print("\nStep 6: Initializing Trainer")
    trainer = CTPhaseTrainer(config)
    
    # Load checkpoint if available
    if checkpoint_path and start_epoch > 0:
        try:
            is_loaded = trainer.load_checkpoint(checkpoint_path)
            if not is_loaded:
                raise
            print(f"✓ Successfully loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"✗ Failed to load checkpoint: {e}, starting from scratch")
            start_epoch = 0
            trainer = CTPhaseTrainer(config)  # Re-initialize trainer if loading fails
    
    # Step 7: Start training
    print("\nStep 7: Starting Training")
    print("=" * 80)
    
    try:
        trainer.train(train_loader, val_loader, config['epochs'], start_epoch=start_epoch)
        print("\n✓ Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
        trainer.save_checkpoint(start_epoch or float('inf'), is_best=False)
        
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        trainer.save_checkpoint(start_epoch or float('inf'), is_best=False)

if __name__ == "__main__":
    run_full_training()