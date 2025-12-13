#!/usr/bin/env python3
"""
Multi-Series Z-Alignment Example
Aligns all CT phases in a study to a common coordinate system.
"""

import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
from pathlib import Path

# Import the functions (assuming they're in the same file or imported)
from align_data import (
    align_all_series_in_study,
    verify_study_alignment
)


def find_cropped_volume_file(study_id: str, series_id: str, input_dir: Path, suffix: str = "_crop.nii.gz"):
    """
    Safely find a cropped volume in a flat directory.
    Searches for: {study_id}_{series_id}{suffix}
    Returns Path or None.
    """
    pattern = f"{study_id}_{series_id}{suffix}"
    matches = list(input_dir.glob(pattern))
    
    if len(matches) == 0:
        return None
    elif len(matches) > 1:
        print(f"Multiple matches found for {pattern} — taking first")
    
    return matches[0]


def find_seg_file(study_id: str, series_id: str, input_dir: Path):
    return find_cropped_volume_file(study_id, series_id, input_dir, suffix="_seg.nii.gz")

def find_existing_noncontrast_series(study_rows, input_dir, study_id):
    """
    Find the first Non-contrast series that actually exists on disk.
    Returns SeriesInstanceUID or raises clear error.
    """
    nc_candidates = study_rows[study_rows["Label"] == "Non-contrast"]
    
    if nc_candidates.empty:
        raise ValueError(f"No 'Non-contrast' series labeled in CSV for study {study_id}")
    
    # print(f"   Found {len(nc_candidates)} Non-contrast candidates in CSV...")
    
    for _, row in nc_candidates.iterrows():
        series_id = row["SeriesInstanceUID"]
        expected_file = input_dir / f"{study_id}_{series_id}_crop.nii.gz"
        
        if expected_file.exists():
            # print(f"   Using existing Non-contrast: {series_id[:20]}...")
            return series_id
        # else:
        #     print(f"   Missing on disk: {series_id[:20]}... → skipping")
    
    # raise FileNotFoundError(
    #     f"None of the {len(nc_candidates)} Non-contrast series exist on disk for study {study_id}\n"
    #     f"   Checked paths in: {input_dir}\n"
    #     f"   Tip: Re-run cropping step or check file naming."
    # )

def process_single_study_example():
    """
    Example: Process one study with multiple phases.
    """
    
    # ============================================================
    # CONFIGURATION
    # ============================================================
    MAIN_PATH = "../../../ncct_cect/vindr_ds/"
    INPUT_DIR = Path(MAIN_PATH + "cropped_volumes")
    OUTPUT_DIR = Path(MAIN_PATH + "aligned_volumes")
    LABELS_CSV = MAIN_PATH + "labels.csv"
    
    # Study to process (replace with your actual study ID)
    STUDY_ID = "1.2.840.113619.2.359.3.2831208971.47.1588634888.707"
    
    print(f"\n{'='*80}")
    print(f"MULTI-SERIES Z-ALIGNMENT EXAMPLE")
    print(f"{'='*80}")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Study ID: {STUDY_ID}")
    
    # ============================================================
    # STEP 1: Load labels and find series for this study
    # ============================================================
    labels_df = pd.read_csv(LABELS_CSV)
    study_rows = labels_df[labels_df["StudyInstanceUID"] == STUDY_ID]
    
    if study_rows.empty:
        print(f"❌ No series found for study {STUDY_ID}")
        return
    
    print(f"\n✓ Found {len(study_rows)} series in this study:")
    for _, row in study_rows.iterrows():
        print(f"  - {row['SeriesInstanceUID'][:20]}... ({row['Label']})")
    
    # ============================================================
    # STEP 2: Find non-contrast (reference) series
    # ============================================================
    try:
        reference_series_id = find_existing_noncontrast_series(
            study_rows=study_rows,
            input_dir=INPUT_DIR,
            study_id=STUDY_ID  # or study_id in batch mode
        )
    except Exception as e:
        print(f"   Cannot find valid Non-contrast volume: {e}")
        return  # or continue in batch mode
    # ============================================================
    # STEP 3: Build series_paths dictionary
    # ============================================================
    series_paths = {}
    
    for _, row in study_rows.iterrows():
        series_id = row["SeriesInstanceUID"]
        phase_label = row["Label"]
        
        # Image path
        # image_file = INPUT_DIR / f"{STUDY_ID}_{series_id}_crop.nii.gz"
        
        # if not image_file.exists():
        #     print(f"⚠️  Skipping {series_id}: file not found")
        #     continue

        # === SAFEST WAY: Use glob to find actual file on disk ===
        image_path = find_cropped_volume_file(STUDY_ID, series_id, INPUT_DIR, "_crop.nii.gz")
        seg_path = find_cropped_volume_file(STUDY_ID, series_id, INPUT_DIR, "_seg.nii.gz")

        if image_path is None:
            print(f"   Missing image: {STUDY_ID}_{series_id}_crop.nii.gz")
            continue

        print(f"   Found: {image_path.name}")
        
        # Segmentation path (optional)
        seg_file = INPUT_DIR / f"{STUDY_ID}_{series_id}_seg.nii.gz"
        
        series_paths[series_id] = {
            "image": str(image_file),
            "phase": phase_label,
            "seg": str(seg_file) if seg_file.exists() else None
        }
        
        print(f"  ✓ Added {phase_label}: {series_id[:20]}...")
    
    print(f"\n✓ Prepared {len(series_paths)} series for alignment")
    
    # ============================================================
    # STEP 4: Run multi-series alignment
    # ============================================================
    study_output_dir = OUTPUT_DIR / STUDY_ID
    study_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        alignment_result = align_all_series_in_study(
            study_id=STUDY_ID,
            series_paths=series_paths,
            reference_series_id=reference_series_id,
            output_dir=str(study_output_dir),
            body_threshold=-600,
            min_overlap_slices=30
        )
        
        # ============================================================
        # STEP 5: Check results
        # ============================================================
        print(f"\n{'='*80}")
        print("ALIGNMENT RESULTS")
        print(f"{'='*80}")
        
        aligned_volumes = alignment_result['aligned_volumes']
        metadata = alignment_result['metadata']
        
        print(f"\n✓ Aligned {len(aligned_volumes)} volumes")
        print(f"✓ Common overlap: {metadata['common_overlap']['depth']} slices")
        print(f"✓ Range: [{metadata['common_overlap']['start']}:{metadata['common_overlap']['end']}]")
        
        # Print details for each series
        print(f"\nPer-series details:")
        for series_id, info in metadata['series'].items():
            print(f"\n  {series_id[:30]}...")
            print(f"    Phase: {info['phase']}")
            print(f"    Original shape: {info['original_shape']}")
            print(f"    Aligned shape: {info['aligned_shape']}")
            print(f"    Z-offset: {info['z_offset']} slices")
            print(f"    Similarity score: {info['similarity_score']:.4f}")
            print(f"    Cropped: [{info['crop_range'][0]}:{info['crop_range'][1]}]")
        
        # Verify all match
        if alignment_result['all_match']:
            print(f"\n✓✓✓ SUCCESS: All volumes have matching dimensions!")
        else:
            print(f"\n⚠️  WARNING: Some volumes don't match!")
        
        # ============================================================
        # STEP 6: Save segmentations (if available)
        # ============================================================
        print(f"\n{'='*80}")
        print("PROCESSING SEGMENTATIONS")
        print(f"{'='*80}")
        
        for series_id, series_info in series_paths.items():
            seg_path = series_info.get("seg")
            
            if seg_path is None or not os.path.exists(seg_path):
                print(f"  ⊘ No segmentation for {series_id[:20]}...")
                continue
            
            print(f"\n  Processing seg for {series_id[:20]}...")
            
            # Load segmentation
            seg_sitk = sitk.ReadImage(seg_path)
            seg_np = sitk.GetArrayFromImage(seg_sitk)
            
            # Get crop range from metadata
            crop_info = metadata['series'][series_id]
            crop_start, crop_end = crop_info['crop_range']
            
            # Crop segmentation the same way as image
            seg_cropped_np = seg_np[crop_start:crop_end]
            
            # Convert back to SimpleITK
            seg_cropped = sitk.GetImageFromArray(seg_cropped_np.astype(np.uint8))
            
            # Copy metadata from aligned volume
            aligned_vol = aligned_volumes[series_id]
            seg_cropped.SetSpacing(aligned_vol.GetSpacing())
            seg_cropped.SetDirection(aligned_vol.GetDirection())
            seg_cropped.SetOrigin(aligned_vol.GetOrigin())
            
            # Save aligned segmentation
            seg_output_path = study_output_dir / f"{STUDY_ID}_{series_id}_aligned_seg.nii.gz"
            sitk.WriteImage(seg_cropped, str(seg_output_path))
            
            print(f"    ✓ Saved: {seg_output_path.name}")
        
        # ============================================================
        # STEP 7: Create verification visualizations
        # ============================================================
        print(f"\n{'='*80}")
        print("CREATING VISUALIZATIONS")
        print(f"{'='*80}")
        
        create_alignment_verification_plots(
            aligned_volumes=aligned_volumes,
            metadata=metadata,
            output_dir=study_output_dir
        )
        
        print(f"\n✓ All results saved to: {study_output_dir}")
        
    except Exception as e:
        print(f"\n❌ Alignment failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ============================================================
    # STEP 8: Final verification
    # ============================================================
    print(f"\n{'='*80}")
    print("FINAL VERIFICATION")
    print(f"{'='*80}")
    
    verify_study_alignment(str(study_output_dir))


def create_alignment_verification_plots(aligned_volumes, metadata, output_dir):
    """Create comprehensive verification plots."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    output_dir = Path(output_dir)
    
    # Get reference
    ref_id = metadata['reference_series']
    ref_vol = aligned_volumes[ref_id]
    ref_np = sitk.GetArrayFromImage(ref_vol)
    
    num_series = len(aligned_volumes)
    
    # ============================================================
    # Plot 1: Compare all phases at same slice
    # ============================================================
    mid_slice = ref_np.shape[0] // 2
    
    fig, axes = plt.subplots(1, num_series, figsize=(5 * num_series, 5))
    if num_series == 1:
        axes = [axes]
    
    for idx, (series_id, vol) in enumerate(aligned_volumes.items()):
        vol_np = sitk.GetArrayFromImage(vol)
        phase = metadata['series'][series_id]['phase']
        
        axes[idx].imshow(vol_np[mid_slice], cmap='gray', vmin=-1000, vmax=500)
        axes[idx].set_title(f'{phase}\n{series_id[:15]}...', fontsize=10)
        axes[idx].axis('off')
    
    plt.suptitle(f'All Phases Aligned - Slice {mid_slice}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'alignment_comparison_same_slice.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: alignment_comparison_same_slice.png")
    
    # ============================================================
    # Plot 2: Shape and overlap summary
    # ============================================================
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = 0
    colors = plt.cm.Set3(range(num_series))
    
    for idx, (series_id, info) in enumerate(metadata['series'].items()):
        phase = info['phase']
        original_depth = info['original_shape'][0]
        aligned_depth = info['aligned_shape'][0]
        crop_start, crop_end = info['crop_range']
        z_offset = info['z_offset']
        
        # Draw original volume
        ax.barh(y_pos, original_depth, height=0.3, left=0, 
                color=colors[idx], alpha=0.3, label=f'{phase} (original)')
        
        # Draw aligned region
        ax.barh(y_pos, aligned_depth, height=0.3, left=crop_start,
                color=colors[idx], alpha=0.8, edgecolor='black', linewidth=2)
        
        # Add text
        ax.text(-5, y_pos, f'{phase}\n{series_id[:15]}...', 
                va='center', ha='right', fontsize=9)
        ax.text(original_depth + 5, y_pos, 
                f'offset: {z_offset:+d}\n{aligned_depth} slices',
                va='center', ha='left', fontsize=8)
        
        y_pos += 1
    
    # Mark common overlap region
    common_start = metadata['common_overlap']['start']
    common_end = metadata['common_overlap']['end']
    ax.axvspan(common_start, common_end, alpha=0.2, color='green', 
               label='Common overlap')
    
    ax.set_xlabel('Slice Index (in reference coordinates)', fontsize=12)
    ax.set_title('Multi-Series Z-Alignment Overview', fontsize=14, fontweight='bold')
    ax.set_yticks([])
    ax.legend(loc='upper right')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'alignment_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: alignment_overview.png")
    
    # ============================================================
    # Plot 3: Difference maps between phases
    # ============================================================
    if num_series > 1:
        fig, axes = plt.subplots(1, num_series - 1, figsize=(5 * (num_series - 1), 5))
        if num_series == 2:
            axes = [axes]
        
        other_series = [s for s in aligned_volumes.keys() if s != ref_id]
        
        for idx, series_id in enumerate(other_series):
            vol_np = sitk.GetArrayFromImage(aligned_volumes[series_id])
            phase = metadata['series'][series_id]['phase']
            
            diff = np.abs(ref_np[mid_slice] - vol_np[mid_slice])
            
            im = axes[idx].imshow(diff, cmap='hot', vmin=0, vmax=500)
            axes[idx].set_title(f'|Non-contrast - {phase}|', fontsize=10)
            axes[idx].axis('off')
            plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        
        plt.suptitle(f'Difference Maps - Slice {mid_slice}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'difference_maps.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: difference_maps.png")


def process_all_studies(force_recompute: bool = False):
    """Process all studies with skip logic."""
    MAIN_PATH = "../../ncct_cect/vindr_ds/"
    INPUT_DIR = Path(MAIN_PATH + "cropped_volumes")
    OUTPUT_DIR = Path(MAIN_PATH + "aligned_volumes")
    LABELS_CSV = MAIN_PATH + "labels.csv"
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    labels_df = pd.read_csv(LABELS_CSV)
    study_ids = labels_df["StudyInstanceUID"].unique()
    
    print(f"\n{'='*80}")
    print(f"BATCH PROCESSING: {len(study_ids)} STUDIES")
    print(f"Force recompute: {force_recompute}")
    print(f"{'='*80}")
    
    successful = 0
    failed = 0
    skipped = 0
    failed_studies = []
    
    for idx, study_id in enumerate(study_ids, 1):
        # print(f"\n[{idx}/{len(study_ids)}] {study_id[:30]}...")
        
        # Get series for this study
        study_rows = labels_df[labels_df["StudyInstanceUID"] == study_id]
        
        # Find non-contrast
        
        # nc_row = study_rows[study_rows["Label"] == "Non-contrast"]
        # if nc_row.empty:
        #     print(f"  ⊘ No non-contrast, skipping")
        #     failed += 1
        #     failed_studies.append((study_id, "No non-contrast"))
        #     continue
        
        # reference_series_id = nc_row.iloc[0]["SeriesInstanceUID"]
        try:
            reference_series_id = find_existing_noncontrast_series(
                study_rows=study_rows,
                input_dir=INPUT_DIR,
                study_id=study_id  # or study_id in batch mode
            )
        except Exception as e:
            print(f"   Cannot find valid Non-contrast volume: {e}")
            continue  # or continue in batch mode

        if not reference_series_id:
            # print(f"  ⊘ No valid non-contrast series found, skipping")
            failed += 1
            failed_studies.append((study_id, "No valid non-contrast"))
            continue

        # Build series paths
        series_paths = {}
        for _, row in study_rows.iterrows():
            series_id = row["SeriesInstanceUID"]
            # image_file = INPUT_DIR / f"{study_id}_{series_id}_crop.nii.gz"
            # === SAFEST WAY: Use glob to find actual file on disk ===
            image_file = find_cropped_volume_file(study_id, series_id, INPUT_DIR, "_crop.nii.gz")
            seg_path = find_cropped_volume_file(study_id, series_id, INPUT_DIR, "_seg.nii.gz")

            if image_file is None:
                print(f"   Missing image: {study_id}_{series_id}_crop.nii.gz")
                continue

            print(f"   Found: {image_file.name}")

            if image_file.exists():
                series_paths[series_id] = {
                    "image": str(image_file),
                    "phase": row["Label"],
                    "seg": seg_path
                }
        
        if len(series_paths) < 2:
            print(f"  ⊘ Only {len(series_paths)} series, skipping")
            failed += 1
            failed_studies.append((study_id, f"Only {len(series_paths)} series"))
            continue
        
        # Align
        study_output_dir = OUTPUT_DIR / study_id
        study_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            alignment_result = align_all_series_in_study(
                study_id=study_id,
                series_paths=series_paths,
                reference_series_id=reference_series_id,
                output_dir=str(study_output_dir),
                body_threshold=-600,
                min_overlap_slices=30,
                force_recompute=force_recompute  # NEW
            )
            
            if alignment_result.get('skipped', False):
                skipped += 1
            elif alignment_result['all_match']:
                print(f"  ✓ Success")
                successful += 1
            else:
                print(f"  ⚠️  Completed but sizes don't match")
                failed += 1
                failed_studies.append((study_id, "Size mismatch"))
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            failed += 1
            failed_studies.append((study_id, str(e)[:50]))
    
    # Summary
    print(f"\n{'='*80}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"✓ Successful: {successful}/{len(study_ids)}")
    print(f"⊘ Skipped (already done): {skipped}/{len(study_ids)}")
    print(f"✗ Failed: {failed}/{len(study_ids)}")
    
    if failed_studies:
        print(f"\nFailed studies:")
        for study_id, reason in failed_studies[:20]:  # Show first 20
            print(f"  - {study_id[:30]}... → {reason}")
        if len(failed_studies) > 20:
            print(f"  ... and {len(failed_studies) - 20} more")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        # Process all studies
        if len(sys.argv) > 2 and sys.argv[2] == "--force-recompute":
            process_all_studies(force_recompute=True)
        else:
            process_all_studies(force_recompute=False)
    else:
        # Process single study example
        process_single_study_example()