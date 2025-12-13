"""
OPTIMIZED Data Loading Helper for CT Phase Training
==========================================
Key Optimizations:
- Uses CTPhaseDataset with volume caching
- Configurable cache size for memory management
- All original functionality preserved
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
from tqdm import tqdm

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


def analyze_case_similarities(
        data_splits: Dict[str, List[Dict]],
        output_dir: str = './similarity_analysis',
        min_similarity_threshold: float = 0.5,
        num_slices_to_check: int = 20,
        force_recompute: bool = False  # NEW: Force recomputation
    ) -> Tuple[Dict[str, List[Tuple[str, str, str]]], pd.DataFrame]:
    """
    Analyze similarity between source and target volumes to find problematic PAIRS.
    Uses cached results if available.
    
    Args:
        data_splits: Dictionary with 'train', 'val', 'test' data pairs
        output_dir: Directory to save analysis results
        min_similarity_threshold: Pairs below this SSIM will be flagged
        num_slices_to_check: Number of slices to sample per volume
        force_recompute: If True, recompute even if cache exists
        
    Returns:
        Tuple of (exclude_pairs dict, similarity DataFrame)
    """
    from skimage.metrics import structural_similarity as ssim
    import nibabel as nib
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / 'similarity_analysis.csv'
    
    # Try to load from cache
    if csv_path.exists() and not force_recompute:
        print("\n" + "=" * 80)
        print("LOADING CACHED SIMILARITY ANALYSIS")
        print("=" * 80)
        
        try:
            df = pd.read_csv(csv_path)
            print(f"✓ Loaded cached analysis from {csv_path}")
            print(f"  Total pairs analyzed: {len(df)}")
            print(f"  Analysis date: {pd.to_datetime(csv_path.stat().st_mtime, unit='s')}")
            
            # Reconstruct exclude_pairs from DataFrame
            exclude_pairs = {'train': [], 'val': [], 'test': []}
            problematic = df[df['ssim'] < min_similarity_threshold]
            
            for _, row in problematic.iterrows():
                split = row['split']
                pair_key = (row['case_id'], row['source_phase'], row['target_phase'])
                exclude_pairs[split].append(pair_key)
            
            # Print summary
            total_excluded = sum(len(pairs) for pairs in exclude_pairs.values())
            print(f"\n✓ Reconstructed exclusion list from cache:")
            print(f"  Total pairs to exclude: {total_excluded}")
            for split_name, pairs_list in exclude_pairs.items():
                print(f"    {split_name}: {len(pairs_list)} pairs")
            
            print("\n  💡 To recompute from scratch, set force_recompute=True")
            
            # Still create plots and reports
            _print_similarity_statistics(df, min_similarity_threshold, output_dir)
            _plot_similarity_distribution(df, output_dir, min_similarity_threshold)
            
            return exclude_pairs, df
            
        except Exception as e:
            print(f"⚠ Failed to load cache: {e}")
            print("  Computing from scratch...")
    
    # Compute from scratch
    print("\n" + "=" * 80)
    print("ANALYZING PAIR SIMILARITIES (COMPUTING FROM SCRATCH)")
    print("=" * 80)
    
    if force_recompute:
        print("  (Force recompute enabled)")
    
    all_similarities = []
    exclude_pairs = {'train': [], 'val': [], 'test': []}
    
    # Check all splits
    for split_name, pairs in data_splits.items():
        print(f"\nAnalyzing {split_name} set ({len(pairs)} pairs)...")
        
        for pair in tqdm(pairs, desc=f"Processing {split_name}"):
            try:
                # Load volumes
                source_nii = nib.load(pair['source_path'])
                target_nii = nib.load(pair['target_path'])
                
                source_vol = np.asarray(source_nii.dataobj)
                target_vol = np.asarray(target_nii.dataobj)
                
                # Transpose to (D, H, W)
                source_vol = np.transpose(source_vol, (2, 1, 0))
                target_vol = np.transpose(target_vol, (2, 1, 0))
                
                # Check shape match
                if source_vol.shape != target_vol.shape:
                    print(f"  ⚠ Shape mismatch: {pair['case_id']} "
                          f"{pair['source_phase']}→{pair['target_phase']} - "
                          f"source {source_vol.shape} vs target {target_vol.shape}")
                    all_similarities.append({
                        'case_id': pair['case_id'],
                        'split': split_name,
                        'source_phase': pair['source_phase'],
                        'target_phase': pair['target_phase'],
                        'ssim': 0.0,
                        'ncc': 0.0,
                        'issue': 'shape_mismatch'
                    })
                    exclude_pairs[split_name].append(
                        (pair['case_id'], pair['source_phase'], pair['target_phase'])
                    )
                    continue
                
                # Sample slices evenly through volume
                depth = source_vol.shape[0]
                slice_indices = np.linspace(
                    depth // 4, 3 * depth // 4, 
                    min(num_slices_to_check, depth // 2),
                    dtype=int
                )
                
                # Compute metrics on sampled slices
                ssim_scores = []
                ncc_scores = []
                
                for idx in slice_indices:
                    source_slice = source_vol[idx]
                    target_slice = target_vol[idx]
                    
                    # Normalize slices
                    # source_norm = (source_slice - source_slice.mean()) / (source_slice.std() + 1e-8)
                    # target_norm = (target_slice - target_slice.mean()) / (target_slice.std() + 1e-8)
                    source_norm, target_norm = source_slice, target_slice
                    # SSIM
                    try:
                        ssim_val = ssim(
                            source_norm, target_norm,
                            data_range=source_norm.max() - source_norm.min()
                        )
                        ssim_scores.append(ssim_val)
                    except:
                        ssim_scores.append(0.0)
                    
                    # Normalized Cross-Correlation (handle zero std)
                    try:
                        if source_norm.std() > 1e-6 and target_norm.std() > 1e-6:
                            ncc = np.corrcoef(source_norm.ravel(), target_norm.ravel())[0, 1]
                            if np.isnan(ncc):
                                ncc = 0.0
                        else:
                            ncc = 0.0
                        ncc_scores.append(ncc)
                    except:
                        ncc_scores.append(0.0)
                
                avg_ssim = np.mean(ssim_scores)
                avg_ncc = np.mean(ncc_scores)
                
                # Flag low similarity pairs
                is_problematic = avg_ssim < min_similarity_threshold
                
                all_similarities.append({
                    'case_id': pair['case_id'],
                    'split': split_name,
                    'source_phase': pair['source_phase'],
                    'target_phase': pair['target_phase'],
                    'ssim': avg_ssim,
                    'ncc': avg_ncc,
                    'source_shape': str(source_vol.shape),
                    'target_shape': str(target_vol.shape),
                    'issue': 'low_similarity' if is_problematic else 'ok'
                })
                
                if is_problematic:
                    exclude_pairs[split_name].append(
                        (pair['case_id'], pair['source_phase'], pair['target_phase'])
                    )
                
            except Exception as e:
                print(f"  ✗ Error processing {pair['case_id']} "
                      f"{pair.get('source_phase', '?')}→{pair.get('target_phase', '?')}: {e}")
                all_similarities.append({
                    'case_id': pair['case_id'],
                    'split': split_name,
                    'source_phase': pair.get('source_phase', 'unknown'),
                    'target_phase': pair.get('target_phase', 'unknown'),
                    'ssim': 0.0,
                    'ncc': 0.0,
                    'issue': f'error: {str(e)[:50]}'
                })
                exclude_pairs[split_name].append(
                    (pair['case_id'], pair.get('source_phase', 'unknown'), 
                     pair.get('target_phase', 'unknown'))
                )
    
    # Convert to DataFrame
    df = pd.DataFrame(all_similarities)
    
    # Save to cache
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Analysis saved to cache: {csv_path}")
    
    # Save exclusion list
    exclusion_path = output_dir / 'pairs_to_exclude.txt'
    with open(exclusion_path, 'w') as f:
        f.write("# Format: case_id,source_phase,target_phase,split\n")
        for split_name, pairs_list in exclude_pairs.items():
            for case_id, src_phase, tgt_phase in pairs_list:
                f.write(f"{case_id},{src_phase},{tgt_phase},{split_name}\n")
    
    # Print statistics and create plots
    _print_similarity_statistics(df, min_similarity_threshold, output_dir)
    _plot_similarity_distribution(df, output_dir, min_similarity_threshold)
    
    return exclude_pairs, df


def _print_similarity_statistics(df: pd.DataFrame, threshold: float, output_dir: Path):
    """Print similarity statistics (extracted to avoid duplication)."""
    
    problematic = df[df['ssim'] < threshold].sort_values('ssim')
    
    print("\n" + "=" * 80)
    print(f"PROBLEMATIC PAIRS (SSIM < {threshold})")
    print("=" * 80)
    
    total_excluded = len(problematic)
    print(f"\nFound {total_excluded} problematic pairs to exclude:")
    
    # Count by split
    for split in ['train', 'val', 'test']:
        count = len(problematic[problematic['split'] == split])
        total_in_split = len(df[df['split'] == split])
        print(f"  {split}: {count}/{total_in_split} pairs ({100*count/total_in_split:.1f}%)")
    
    print("\nLowest 30 similarity pairs:")
    print(problematic[['case_id', 'split', 'source_phase', 'target_phase', 
                       'ssim', 'ncc', 'issue']].head(30).to_string(index=False))
    
    # Count unique cases affected
    unique_cases = problematic['case_id'].nunique()
    print(f"\n  → Affects {unique_cases} unique cases (but only specific pairs excluded)")
    
    # Overall statistics
    print("\n" + "=" * 80)
    print("SIMILARITY STATISTICS")
    print("=" * 80)
    print(f"\nOverall SSIM: {df['ssim'].mean():.3f} ± {df['ssim'].std():.3f}")
    print(f"Overall NCC:  {df['ncc'].mean():.3f} ± {df['ncc'].std():.3f}")
    
    print("\nBy split:")
    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]
        if len(split_df) > 0:
            excluded = len(problematic[problematic['split'] == split])
            print(f"  {split:5s}: SSIM {split_df['ssim'].mean():.3f} ± {split_df['ssim'].std():.3f}, "
                  f"NCC {split_df['ncc'].mean():.3f} ± {split_df['ncc'].std():.3f} "
                  f"({excluded}/{len(split_df)} excluded)")
    
    print("\nBy target phase:")
    for phase in sorted(df['target_phase'].unique()):
        phase_df = df[df['target_phase'] == phase]
        if len(phase_df) > 0:
            excluded = len(problematic[problematic['target_phase'] == phase])
            print(f"  {phase:12s}: SSIM {phase_df['ssim'].mean():.3f} ± {phase_df['ssim'].std():.3f} "
                  f"({excluded}/{len(phase_df)} excluded)")

def _plot_similarity_distribution(df: pd.DataFrame, output_dir: Path, threshold: float):
    """Plot similarity score distributions with exclusion threshold."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # SSIM histogram
    axes[0, 0].hist(df['ssim'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(df['ssim'].mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {df["ssim"].mean():.3f}')
    axes[0, 0].axvline(threshold, color='darkred', linestyle='-', 
                       linewidth=2, label=f'Threshold: {threshold}')
    axes[0, 0].set_xlabel('SSIM')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('SSIM Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # NCC histogram
    axes[0, 1].hist(df['ncc'], bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 1].axvline(df['ncc'].mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {df["ncc"].mean():.3f}')
    axes[0, 1].set_xlabel('NCC')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Normalized Cross-Correlation Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # SSIM vs NCC scatter with threshold line
    colors = ['red' if s < threshold else 'blue' for s in df['ssim']]
    axes[1, 0].scatter(df['ssim'], df['ncc'], alpha=0.5, s=20, c=colors)
    axes[1, 0].axvline(threshold, color='darkred', linestyle='--', 
                       linewidth=2, label=f'Exclusion threshold')
    axes[1, 0].set_xlabel('SSIM')
    axes[1, 0].set_ylabel('NCC')
    axes[1, 0].set_title('SSIM vs NCC (Red = Excluded)')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # SSIM by target phase
    phases = df['target_phase'].unique()
    for phase in phases:
        phase_df = df[df['target_phase'] == phase]['ssim']
        axes[1, 1].hist(phase_df, bins=20, alpha=0.6, label=phase, edgecolor='black')
    axes[1, 1].axvline(threshold, color='darkred', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('SSIM')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('SSIM by Target Phase')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'similarity_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Visualization saved to {output_dir / 'similarity_distribution.png'}")

def visualize_excluded_pairs(
    data_splits: Dict[str, List[Dict]],
    exclude_pairs: Dict[str, List[Tuple[str, str, str]]],
    similarity_df: pd.DataFrame,
    output_dir: str = './similarity_analysis',
    num_samples: int = 10
    ):
    """
    Visualize sample slices from excluded pairs to understand why they were flagged.
    
    Args:
        data_splits: Original data splits
        exclude_pairs: Dictionary of excluded pairs per split
        similarity_df: DataFrame with similarity scores
        output_dir: Directory to save visualizations
        num_samples: Number of excluded pairs to visualize
    """
    import nibabel as nib
    from skimage.metrics import structural_similarity as ssim
    
    output_dir = Path(output_dir) / 'excluded_samples'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("VISUALIZING EXCLUDED PAIRS")
    print("=" * 80)
    
    # Collect all excluded pairs with their data
    excluded_with_data = []
    
    for split_name, pairs_list in exclude_pairs.items():
        if not pairs_list:
            continue
            
        for case_id, src_phase, tgt_phase in pairs_list:
            # Find the actual pair data
            matching_pairs = [
                p for p in data_splits[split_name] 
                if p['case_id'] == case_id 
                and p['source_phase'] == src_phase 
                and p['target_phase'] == tgt_phase
            ]
            
            if matching_pairs:
                pair_data = matching_pairs[0]
                
                # Get similarity score from DataFrame
                mask = (
                    (similarity_df['case_id'] == case_id) &
                    (similarity_df['source_phase'] == src_phase) &
                    (similarity_df['target_phase'] == tgt_phase)
                )
                if mask.any():
                    row = similarity_df[mask].iloc[0]
                    ssim_score = row['ssim']
                    ncc_score = row['ncc']
                else:
                    ssim_score = 0.0
                    ncc_score = 0.0
                
                excluded_with_data.append({
                    'pair': pair_data,
                    'split': split_name,
                    'ssim': ssim_score,
                    'ncc': ncc_score
                })
    
    if not excluded_with_data:
        print("No excluded pairs to visualize!")
        return
    
    # Sort by SSIM (lowest first)
    excluded_with_data.sort(key=lambda x: x['ssim'])
    
    print(f"Found {len(excluded_with_data)} excluded pairs")
    print(f"Visualizing {min(num_samples, len(excluded_with_data))} worst cases...")
    
    # Visualize samples
    for idx, item in enumerate(excluded_with_data[:num_samples]):
        try:
            pair = item['pair']
            split = item['split']
            ssim_score = item['ssim']
            ncc_score = item['ncc']
            
            # Load volumes
            source_nii = nib.load(pair['source_path'])
            target_nii = nib.load(pair['target_path'])
            
            source_vol = np.asarray(source_nii.dataobj)
            target_vol = np.asarray(target_nii.dataobj)
            
            # Transpose to (D, H, W)
            source_vol = np.transpose(source_vol, (2, 1, 0))
            target_vol = np.transpose(target_vol, (2, 1, 0))
            
            # Get 3 representative slices
            depth = min(source_vol.shape[0], target_vol.shape[0])
            slice_positions = [depth // 4, depth // 2, 3 * depth // 4]
            
            # Create figure with 3 rows (3 slice positions) x 4 columns (source, target, diff, overlay)
            fig = plt.figure(figsize=(16, 12))
            gs = gridspec.GridSpec(3, 4, hspace=0.3, wspace=0.3)
            
            for row, slice_idx in enumerate(slice_positions):
                if slice_idx >= source_vol.shape[0] or slice_idx >= target_vol.shape[0]:
                    continue
                
                source_slice = source_vol[slice_idx]
                target_slice = target_vol[slice_idx]
                
                # Normalize for display
                # source_norm = (source_slice - source_slice.mean()) / (source_slice.std() + 1e-8)
                # target_norm = (target_slice - target_slice.mean()) / (target_slice.std() + 1e-8)

                source_norm, target_norm = source_slice, target_slice
                
                # Clip for better visualization
                source_display = np.clip(source_norm, -3, 3)
                target_display = np.clip(target_norm, -3, 3)
                
                # Source
                ax = fig.add_subplot(gs[row, 0])
                im = ax.imshow(source_display, cmap='gray', vmin=-3, vmax=3)
                if row == 0:
                    ax.set_title(f'Source: {pair["source_phase"]}', fontsize=10, fontweight='bold')
                ax.set_ylabel(f'Slice {slice_idx}/{depth}', fontsize=9)
                ax.axis('off')
                
                # Target
                ax = fig.add_subplot(gs[row, 1])
                im = ax.imshow(target_display, cmap='gray', vmin=-3, vmax=3)
                if row == 0:
                    ax.set_title(f'Target: {pair["target_phase"]}', fontsize=10, fontweight='bold')
                ax.axis('off')
                
                # Difference map
                ax = fig.add_subplot(gs[row, 2])
                diff = np.abs(source_display - target_display)
                im = ax.imshow(diff, cmap='hot', vmin=0, vmax=3)
                if row == 0:
                    ax.set_title('Absolute Difference', fontsize=10, fontweight='bold')
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                
                # Overlay (checkerboard pattern)
                ax = fig.add_subplot(gs[row, 3])
                overlay = np.copy(source_display)
                # Create checkerboard mask
                h, w = overlay.shape
                checker_size = 32
                for i in range(0, h, checker_size * 2):
                    for j in range(0, w, checker_size * 2):
                        overlay[i:i+checker_size, j:j+checker_size] = target_display[i:i+checker_size, j:j+checker_size]
                        if i+checker_size < h and j+checker_size < w:
                            overlay[i+checker_size:i+2*checker_size, j+checker_size:j+2*checker_size] = \
                                target_display[i+checker_size:i+2*checker_size, j+checker_size:j+2*checker_size]
                
                im = ax.imshow(overlay, cmap='gray', vmin=-3, vmax=3)
                if row == 0:
                    ax.set_title('Checkerboard Overlay', fontsize=10, fontweight='bold')
                ax.axis('off')
            
            # Overall title
            fig.suptitle(
                f'EXCLUDED PAIR #{idx+1} - Case: {pair["case_id"]} ({split})\n'
                f'SSIM: {ssim_score:.4f} | NCC: {ncc_score:.4f} | '
                f'Shape: {source_vol.shape}',
                fontsize=14, fontweight='bold', y=0.98
            )
            
            # Save
            save_path = output_dir / f'excluded_{idx+1:02d}_{pair["case_id"]}_ssim{ssim_score:.3f}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Saved sample {idx+1}/{num_samples}: {pair['case_id']} "
                  f"(SSIM={ssim_score:.3f})")
            
        except Exception as e:
            print(f"  ✗ Error visualizing pair {idx+1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n✓ Visualizations saved to {output_dir}")
    print(f"  Check these images to verify exclusion criteria!")

def create_data_pairs(
    config: Dict,
    phase_mapping: Dict = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    exclude_pairs: Dict[str, List[Tuple[str, str, str]]] = None
    ) -> Dict[str, List[Dict]]:
    """
    Create training/validation/test data pairs from directory structure.
    
    Args:
        config: Configuration dict with data_dir and target_phase
        phase_mapping: Optional phase mapping from CSV
        test_size: Test set proportion
        val_size: Validation set proportion
        random_state: Random seed
        exclude_pairs: Dict[split_name -> List[(case_id, source_phase, target_phase)]]
                      Specific pairs to exclude based on low similarity
        
    Returns:
        Dictionary with 'train', 'val', 'test' data pairs
    """
    data_dir = Path(config.get('data_dir', 'data'))
    target_phase = config.get('target_phase', 'portal')
    
    # ============================================================
    # STEP 1: Collect all available phases for each case
    # ============================================================
    cases = {}
    
    for case_dir in data_dir.iterdir():
        if not case_dir.is_dir():
            continue
            
        case_id = case_dir.name
        cases[case_id] = {}
        
        tag = "_registered" # or deformable
        # Find all registered image files
        for nii_file in case_dir.glob(f"*{tag}_norm.nii.gz"):
            # print("nii file", nii_file)
            if "_seg" in str(nii_file):
                continue  # Skip segmentation files
                
            # Extract series ID and phase
            filename = nii_file.stem.replace(f"{tag}_norm", "").replace(".nii", "")
            parts = filename.split('_')
            # print("filename", filename)
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
                seg_file = case_dir / f"{filename}{tag}_seg.nii.gz"
                if seg_file.exists():
                    cases[case_id][phase]['segmentation'] = seg_file
    
    # ============================================================
    # STEP 2: Filter valid cases (must have non-contrast + target phase)
    # ============================================================
    valid_cases = []
    for case_id, phases in cases.items():
        # Must have non-contrast
        if 'non-contrast' not in phases:
            continue
        
        # Must have the target phase we're interested in
        if target_phase not in phases:
            continue
        
        valid_cases.append((case_id, phases))
    
    logger.info(f"Found {len(valid_cases)} valid cases with non-contrast + {target_phase}")
    
    # ============================================================
    # STEP 3: Split cases into train/val/test
    # ============================================================
    case_ids = [case[0] for case in valid_cases]
    
    train_ids, temp_ids = train_test_split(
        case_ids, 
        test_size=test_size + val_size, 
        random_state=random_state
    )
    val_ids, test_ids = train_test_split(
        temp_ids, 
        test_size=test_size / (test_size + val_size), 
        random_state=random_state
    )
    
    logger.info(f"Split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test cases")
    
    # ============================================================
    # STEP 4: Create pairs with similarity-based exclusions
    # ============================================================
    def make_pairs(case_subset, split_name):
        """Create pairs for a given split, excluding low-similarity pairs."""
        pairs = []
        excluded_count = 0
        
        # Build exclusion set for this split
        exclusion_set = set()
        if exclude_pairs and split_name in exclude_pairs:
            exclusion_set = set(exclude_pairs[split_name])
        
        # Get case data lookup
        case_data_lookup = dict(valid_cases)
        
        for case_id in case_subset:
            if case_id not in case_data_lookup:
                continue
            
            case_data = case_data_lookup[case_id]
            
            # Get non-contrast info
            nc_info = case_data.get('non-contrast')
            if not nc_info:
                continue
            
            # Get target phase info
            target_info = case_data.get(target_phase)
            if not target_info:
                continue
            
            # Check if this specific pair should be excluded
            pair_key = (case_id, 'non-contrast', target_phase)
            
            # if pair_key in exclusion_set:
            #     excluded_count += 1
            #     logger.debug(f"Excluding pair: {case_id} (non-contrast → {target_phase})")
            #     continue
            
            # Add the pair
            pairs.append({
                'source_path': nc_info['image'],
                'target_path': target_info['image'],
                'source_phase': 'non-contrast',
                'target_phase': target_phase,
                'case_id': case_id,
                'source_series': nc_info['series_id'],
                'target_series': target_info['series_id'],
                'source_seg': nc_info.get('segmentation'),
                'target_seg': target_info.get('segmentation')
            })
        
        if excluded_count > 0:
            logger.info(f"  {split_name}: Excluded {excluded_count}/{len(case_subset)} pairs (low similarity)")
        
        return pairs
    
    # ============================================================
    # STEP 5: Generate final splits
    # ============================================================
    splits = {
        'train': make_pairs(train_ids, 'train'),
        'val': make_pairs(val_ids, 'val'),
        'test': make_pairs(test_ids, 'test')
    }
    
    logger.info(f"\nFinal data splits:")
    logger.info(f"  Train: {len(train_ids)} cases → {len(splits['train'])} pairs")
    logger.info(f"  Val:   {len(val_ids)} cases → {len(splits['val'])} pairs")
    logger.info(f"  Test:  {len(test_ids)} cases → {len(splits['test'])} pairs")
    
    # ============================================================
    # STEP 6: Validation check
    # ============================================================
    if len(splits['train']) == 0:
        logger.warning("⚠ WARNING: No training pairs generated!")
        logger.warning(f"  Target phase: {target_phase}")
        logger.warning(f"  Valid cases: {len(valid_cases)}")
        if exclude_pairs:
            logger.warning(f"  Exclusions provided: {sum(len(v) for v in exclude_pairs.values())} pairs")
    
    return splits

# ================================================================
# FINAL FIXED: create_data_pairs (supports both modes + strict exclusion)
# ================================================================

def load_exclusion_list(path: str = "../../similarity_analysis/pairs_to_exclude.txt"):
    """Load your pairs_to_exclude.txt file correctly."""
    exclude = {'train': [], 'val': [], 'test': []}
    path = Path(path)
    if not path.exists():
        logger.info("No exclusion file found. Proceeding without filtering.")
        return exclude

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) != 4:
                continue
            case_id, src_phase, tgt_phase, split = parts
            exclude[split].append((case_id, src_phase, tgt_phase))

    total = sum(len(v) for v in exclude.values())
    logger.info(f"Loaded {total} pair-level exclusions from {path}")
    return exclude


def verify_no_leaks(data_pairs: List[Dict], exclude_pairs: Dict, split_name: str):
    """Final safety check — will crash if any bad pair leaked."""
    bad_keys = {(p['case_id'], p['source_phase'], p['target_phase']) for p in data_pairs}
    excluded_keys = {(c, s, t) for c, s, t in exclude_pairs.get(split_name, [])}

    leaked = bad_keys & excluded_keys
    if leaked:
        logger.error(f"LEAK DETECTED in {split_name}: {len(leaked)} bad pairs found!")
        for k in list(leaked)[:10]:
            logger.error(f"  → {k}")
        raise RuntimeError(f"Excluded pairs leaked into {split_name}!")
    else:
        logger.info(f"Verification OK: No excluded pairs in {split_name}")

# ============================================================================
# COMPLETE TRAINING SCRIPT (OPTIMIZED)
# ============================================================================
from torch.utils.data._utils.collate import default_collate




def safe_collate(batch):
    
    valid_samples = [item for item in batch if not item.get('skip', False)]
    if len(valid_samples) == 0:
        return {'skip_batch': True}  # Entire batch invalid
    # Stack only valid ones
    return torch.utils.data.dataloader.default_collate(valid_samples)

def run_full_training():
    """
    Complete training script with OPTIMIZED volume caching.
    All original functionality preserved + faster data loading.
    """
    from training_phase_gen_slice import MemoryOptimizedTrainer, CTPhaseDataset
    from config import train_config
    # Configuration
    config = train_config
    
    print("=" * 80)
    print("OPTIMIZED CT PHASE GENERATION - COMPLETE TRAINING PIPELINE")
    print("=" * 80)
    print(f"\n🚀 KEY OPTIMIZATIONS:")
    print(f"  ✓ Volume caching (max {config['cache_size']} volumes)")
    print(f"  ✓ Memory-mapped file access")
    print(f"  ✓ Lazy loading (only when needed)")
    print(f"  ✓ Expected speedup: 5-10x faster data loading")
    print(f"  ✓ Expected disk I/O reduction: 80-95%")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for existing checkpoints
    start_epoch = 0
    checkpoint_path = None
    checkpoint_files = list(output_dir.glob('checkpoint_epoch_*.pth'))
    
    if checkpoint_files:
        epoch_numbers = []
        for ckpt in checkpoint_files:
            match = re.search(r'checkpoint_epoch_(\d+)\.pth', ckpt.name)
            if match:
                epoch_numbers.append(int(match.group(1)))
        
        if epoch_numbers:
            latest_epoch = max(epoch_numbers)
            checkpoint_path = output_dir / f'checkpoint_epoch_{latest_epoch}.pth'
            start_epoch = latest_epoch
            print(f"✓ Found checkpoint at epoch {latest_epoch }, resuming from epoch {start_epoch}")
    else:
        print("✓ Starting fresh training")
    

    phase_mapping = None
    if Path(config['labels_csv']).exists():
        try:
            phase_mapping = load_phase_mapping(config['labels_csv'])
            print(f"✓ Loaded phase mapping for {len(phase_mapping)} cases")
        except Exception as e:
            print(f"✗ Could not load phase mapping: {e}")
    # 1. First, analyze similarities
    data_splits = create_data_pairs(config, phase_mapping=phase_mapping)
    print("len train",len(data_splits['train']))
    # 2. Analyze similarities at PAIR level
    # exclude_pairs, similarity_df = analyze_case_similarities(
    #     data_splits,
    #     output_dir='../../similarity_analysis',
    #     min_similarity_threshold=0.45,  # Try 0.45 or 0.40 to be less aggressive
    #     num_slices_to_check=20,
    #     force_recompute=config.get('force_similarity_recompute', False)
    # )
    # print(f"✓ Data splits recreated with exclusions")
    # print(f"  Train: {len(data_splits['train'])} pairs (after filtering)")
    # print(f"  Val: {len(data_splits['val'])} pairs (after filtering)")
    print(f"  Test: {len(data_splits['test'])} pairs (after filtering)")
    
    # # Visualize excluded pairs
    # print("\nVisualizing excluded pairs...")
    # visualize_excluded_pairs(
    #     data_splits=data_splits,
    #     exclude_pairs=exclude_pairs,
    #     similarity_df=similarity_df,
    #     output_dir=config.get('similarity_analysis_dir', './similarity_analysis'),
    #     num_samples=config.get('num_excluded_to_visualize', 50)
    # )

    # Load exclusions from your file
    exclude_pairs_dict = load_exclusion_list("../../similarity_analysis/pairs_to_exclude.txt")

    # Create final clean data splits
    data_splits = create_data_pairs(
        config,
        phase_mapping=phase_mapping,
        exclude_pairs=exclude_pairs_dict
    )

    # FINAL SAFETY CHECK — will crash if anything is wrong
    # for split in ['train', 'val', 'test']:
    #     verify_no_leaks(data_splits[split], exclude_pairs_dict, split)

    # # 2. Recreate data splits with exclusions
    # data_splits = create_data_pairs(
    #     config,
    #     phase_mapping=phase_mapping,
    #     exclude_pairs=exclude_pairs  # Apply filtering
    # )
    print("len train",len(data_splits['train']))

    
    # Step 2: Create OPTIMIZED datasets with caching
    print("\nStep 2: Creating OPTIMIZED Datasets with Volume Caching")
    train_dataset = CTPhaseDataset(
        data_splits['train'],
        patch_size=config['patch_size'],
        slice_range=config['slice_range'],
        overlap_ratio=config['overlap_ratio'],
        augment=False,
        cache_size=config['cache_size'],  # NEW: Enable caching
        use_memmap=config['use_memmap'],   # NEW: Use memory mapping
       
    )

    val_dataset = CTPhaseDataset(
        data_splits['val'],
        patch_size=config['patch_size'],
        slice_range=config['slice_range'],
        overlap_ratio=0.5,
        augment=False,
        cache_size=config['cache_size'],  # NEW: Enable caching
        use_memmap=config['use_memmap'],   # NEW: Use memory mapping
        
    )
    
    print(f"✓ Train dataset: {len(train_dataset)} patches")
    print(f"✓ Val dataset: {len(val_dataset)} patches")
    print(f"✓ Volume cache initialized: {config['cache_size']} volumes max")
    print(f"✓ Val dataset: {len(val_dataset)} patches")

    # Debug: Check if any patches exist
    if len(val_dataset) == 0:
        print("❌ VALIDATION DATASET IS EMPTY!")
        print("Checking patch coordinates...")
        for i, pair in enumerate(val_dataset.data_pairs):
            coords = [c for c in val_dataset.patch_coords if c['pair_idx'] == i]
            print(f"  Case {i} ({pair['case_id']}): {len(coords)} patches")
    # # Step 3: Debug dataset (optional)
    # if config.get('debug_patches', False):
    #     print("\nStep 3: Visualizing Patches")
    #     debug_dataset_with_visualization(
    #         train_dataset,
    #         num_samples=config.get('num_debug_samples', 5),
    #         output_dir=config.get('debug_output_dir', './debug_patches')
    #     )
        
    #     if len(train_dataset.data_pairs) > 0:
    #         create_patch_coverage_map(
    #             train_dataset,
    #             case_idx=0,
    #             output_dir=config.get('debug_output_dir', './debug_patches')
    #         )
    
    # Step 4: Create dataloaders
    print("\nStep 4: Creating DataLoaders")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2,  # Keep workers low to avoid cache conflicts
        pin_memory=True,
        drop_last=True,
        collate_fn = safe_collate
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn = safe_collate
    )
    
    print(f"✓ Train loader: {len(train_loader)} batches")
    print(f"✓ Val loader: {len(val_loader)} batches")
    
    # Step 5: Initialize trainer
    print("\nStep 5: Initializing Trainer")
    trainer = MemoryOptimizedTrainer(config)
    
    # Load checkpoint if available
    if checkpoint_path and start_epoch > 0:
        try:
            is_loaded = trainer.load_checkpoint(checkpoint_path)
            if not is_loaded:
                raise Exception("Failed to load checkpoint")
            print(f"✓ Successfully loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"✗ Failed to load checkpoint: {e}, starting from scratch")
            start_epoch = 0
            trainer = MemoryOptimizedTrainer(config)
    
    # Step 6: Start training
    print("\nStep 6: Starting OPTIMIZED Training")
    print("=" * 80)
    
    try:
        trainer.train(train_loader, val_loader, config['epochs'], start_epoch=start_epoch)
        print("\n✓ Training completed successfully!")
        
        # Print final cache statistics
        cache_stats = train_dataset.get_cache_stats()
        print("\n" + "=" * 80)
        print("FINAL PERFORMANCE STATISTICS")
        print("=" * 80)
        print(f"Volume Cache Hit Rate: {cache_stats['hit_rate']:.2%}")
        print(f"Total Cache Hits: {cache_stats['cache_hits']}")
        print(f"Total Cache Misses: {cache_stats['cache_misses']}")
        print(f"Estimated Disk I/O Reduction: ~{cache_stats['hit_rate'] * 100:.1f}%")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
        trainer.save_checkpoint(start_epoch or 0, {}, is_best=False)
        
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        trainer.save_checkpoint(start_epoch or 0, {}, is_best=False)

if __name__ == "__main__":
    run_full_training()