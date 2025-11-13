"""
Find patches with large empty regions (like your screenshot)
"""
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

def analyze_empty_patches(data_pairs, patch_size=(128, 192), patch_depth=15, 
                         overlap_ratio=0.5, empty_threshold=0.7):
    """
    Find patches where >70% is background (like your screenshot).
    
    Args:
        empty_threshold: If this fraction is background, mark as bad
    """
    
    def normalize(volume):
        volume = np.clip(volume, -1000, 1000)
        volume = (volume + 1000) / 2000.0 * 2.0 - 1.0
        return volume
    
    def extract_patch(volume, z, y, x):
        half_depth = patch_depth // 2
        z_start = max(0, z - half_depth)
        z_end = min(volume.shape[0], z + half_depth + 1)
        
        patch = volume[z_start:z_end, y:y+patch_size[0], x:x+patch_size[1]].copy()
        
        if patch.shape[0] < patch_depth:
            pad_before = (patch_depth - patch.shape[0]) // 2
            pad_after = patch_depth - patch.shape[0] - pad_before
            patch = np.pad(patch, ((pad_before, pad_after), (0, 0), (0, 0)), mode='edge')
        
        return normalize(patch)
    
    bad_patches = []
    case_stats = {}
    
    for pair_idx, pair_data in enumerate(tqdm(data_pairs, desc="Scanning volumes")):
        try:
            nii = nib.load(pair_data['source_path'])
            volume = nii.get_fdata()
            volume = np.transpose(volume, (2, 1, 0))
            
            depth, height, width = volume.shape
            if depth < patch_depth + 2 or height < patch_size[0] or width < patch_size[1]:
                continue
            
            case_id = pair_data['case_id']
            case_stats[case_id] = {'total': 0, 'bad': 0, 'bad_positions': []}
            
            # Sample patches from this volume
            padding = patch_depth // 2
            z_start = padding + patch_depth
            z_end = depth - padding - patch_depth
            
            if z_end <= z_start:
                continue
            
            step_y = max(1, int(patch_size[0] * (1 - overlap_ratio)))
            step_x = max(1, int(patch_size[1] * (1 - overlap_ratio)))
            
            for z in range(z_start, z_end, max(1, patch_depth // 2)):  # Sample every few slices
                for y in range(0, height - patch_size[0] + 1, step_y):
                    for x in range(0, width - patch_size[1] + 1, step_x):
                        case_stats[case_id]['total'] += 1
                        
                        patch = extract_patch(volume, z, y, x)
                        
                        # Check if mostly empty (like your screenshot)
                        # Background is very dark (< -0.8)
                        empty_ratio = (patch < -0.8).mean()
                        
                        if empty_ratio > empty_threshold:
                            case_stats[case_id]['bad'] += 1
                            case_stats[case_id]['bad_positions'].append((z, y, x))
                            
                            bad_patches.append({
                                'case_id': case_id,
                                'pair_idx': pair_idx,
                                'position': (z, y, x),
                                'empty_ratio': empty_ratio,
                                'mean': patch.mean(),
                                'std': patch.std()
                            })
        
        except Exception as e:
            print(f"Error in pair {pair_idx}: {e}")
            continue
    
    return bad_patches, case_stats


def visualize_problematic_cases(bad_patches, case_stats, data_pairs, 
                                output_dir='./empty_patch_analysis',
                                num_examples=10):
    """Visualize worst cases."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sort by empty_ratio
    sorted_bad = sorted(bad_patches, key=lambda x: x['empty_ratio'], reverse=True)
    
    print(f"\n{'='*70}")
    print(f"FOUND {len(bad_patches)} PATCHES LIKE YOUR SCREENSHOT")
    print(f"{'='*70}\n")
    
    # Statistics
    total_patches = sum(s['total'] for s in case_stats.values())
    total_bad = len(bad_patches)
    
    print(f"Total patches scanned: {total_patches:,}")
    print(f"Bad patches found: {total_bad:,} ({total_bad/total_patches*100:.1f}%)")
    print(f"\nWorst {num_examples} patches:")
    
    for i, bad in enumerate(sorted_bad[:num_examples]):
        print(f"  {i+1}. Case {bad['case_id']}: {bad['empty_ratio']*100:.1f}% empty "
              f"(pos: z={bad['position'][0]}, y={bad['position'][1]}, x={bad['position'][2]})")
    
    # Per-case analysis
    print(f"\n{'='*70}")
    print("CASES WITH MOST BAD PATCHES:")
    print(f"{'='*70}")
    
    case_bad_rates = []
    for case_id, stats in case_stats.items():
        if stats['total'] > 0:
            bad_rate = stats['bad'] / stats['total']
            case_bad_rates.append((case_id, stats['bad'], stats['total'], bad_rate))
    
    case_bad_rates.sort(key=lambda x: x[3], reverse=True)
    
    print(f"\nTop 20 worst cases:")
    for i, (case_id, bad, total, rate) in enumerate(case_bad_rates[:20]):
        print(f"  {i+1}. {case_id}: {bad}/{total} bad ({rate*100:.1f}%)")
    
    # Save report
    report = {
        'summary': {
            'total_patches': int(total_patches),
            'bad_patches': int(total_bad),
            'bad_percentage': float(total_bad/total_patches*100)
        },
        'worst_patches': [
            {
                'rank': i+1,
                'case_id': bad['case_id'],
                'position': bad['position'],
                'empty_ratio': float(bad['empty_ratio']),
                'mean': float(bad['mean']),
                'std': float(bad['std'])
            }
            for i, bad in enumerate(sorted_bad[:100])
        ],
        'worst_cases': [
            {
                'rank': i+1,
                'case_id': case_id,
                'bad_patches': int(bad),
                'total_patches': int(total),
                'bad_percentage': float(rate*100)
            }
            for i, (case_id, bad, total, rate) in enumerate(case_bad_rates[:50])
        ]
    }
    
    with open(output_dir / 'empty_patch_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved to {output_dir}/empty_patch_report.json")
    
    # Visualize examples
    print(f"\nVisualizing {min(num_examples, len(sorted_bad))} examples...")
    
    for i, bad in enumerate(sorted_bad[:num_examples]):
        pair_data = data_pairs[bad['pair_idx']]
        nii = nib.load(pair_data['source_path'])
        volume = nii.get_fdata()
        volume = np.transpose(volume, (2, 1, 0))
        
        # Extract patch
        def extract_for_viz(volume, z, y, x):
            half_depth = 15 // 2
            z_start = max(0, z - half_depth)
            z_end = min(volume.shape[0], z + half_depth + 1)
            patch = volume[z_start:z_end, y:y+128, x:x+192].copy()
            if patch.shape[0] < 15:
                pad_before = (15 - patch.shape[0]) // 2
                pad_after = 15 - patch.shape[0] - pad_before
                patch = np.pad(patch, ((pad_before, pad_after), (0, 0), (0, 0)), mode='edge')
            return np.clip((patch + 1000) / 2000.0 * 2.0 - 1.0, -1, 1)
        
        z, y, x = bad['position']
        patch = extract_for_viz(volume, z, y, x)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Show slices
        slices = [0, patch.shape[0]//2, patch.shape[0]-1]
        for j, slice_idx in enumerate(slices):
            axes[0, j].imshow(patch[slice_idx], cmap='gray', vmin=-1, vmax=1)
            axes[0, j].set_title(f'Slice {slice_idx}')
            axes[0, j].axis('off')
        
        # Histogram
        axes[1, 0].hist(patch.ravel(), bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(-0.8, color='red', linestyle='--', label='Empty threshold')
        axes[1, 0].set_xlabel('Intensity')
        axes[1, 0].set_title('Intensity Distribution')
        axes[1, 0].legend()
        
        # Stats
        stats_text = (
            f"Empty ratio: {bad['empty_ratio']*100:.1f}%\n"
            f"Mean: {bad['mean']:.3f}\n"
            f"Std: {bad['std']:.3f}\n"
            f"Case: {bad['case_id']}\n"
            f"Position: z={z}, y={y}, x={x}"
        )
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12,
                       verticalalignment='center', family='monospace')
        axes[1, 1].axis('off')
        
        # Empty map
        empty_map = (patch[patch.shape[0]//2] < -0.8).astype(float)
        axes[1, 2].imshow(empty_map, cmap='Reds', vmin=0, vmax=1)
        axes[1, 2].set_title('Empty Regions (red)')
        axes[1, 2].axis('off')
        
        plt.suptitle(f'Bad Patch #{i+1} - {bad["empty_ratio"]*100:.1f}% Empty',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f'bad_patch_{i+1:03d}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Visualizations saved to {output_dir}/")


if __name__ == "__main__":
    import sys
    sys.path.append('/mnt/project')
    from dataloader_train import create_data_pairs, load_phase_mapping
    from pathlib import Path
    
    # Config
    data_dir = '../ncct_cect/vindr_ds/deformable_registered_bspline'
    labels_csv = '../ncct_cect/vindr_ds/labels.csv'
    
    print("Loading data...")
    phase_mapping = None
    if Path(labels_csv).exists():
        phase_mapping = load_phase_mapping(labels_csv)
    
    data_splits = create_data_pairs(data_dir, phase_mapping=phase_mapping)
    data_pairs = data_splits['train']
    
    print(f"Analyzing {len(data_pairs)} training pairs...")
    
    # Find bad patches (>70% empty)
    bad_patches, case_stats = analyze_empty_patches(
        data_pairs,
        empty_threshold=0.7  # Adjust this if needed
    )
    
    # Visualize
    visualize_problematic_cases(
        bad_patches,
        case_stats,
        data_pairs,
        num_examples=15
    )