#!/usr/bin/env python3
"""
Simple Crop Pipeline - Organize cropped volumes by study_id
"""

import os
import sys
import subprocess
import pandas as pd
from pathlib import Path


# =================== CONFIG ===================
MAIN_PATH = "../../ncct_cect/vindr_ds/"
INPUT_DIR = Path(MAIN_PATH + "nifti_unprocessed_volumes")
SEG_DIR = Path(MAIN_PATH + "ts_segmentations")
OUTPUT_DIR = Path(MAIN_PATH + "cropped_volumes")
LABELS_CSV = MAIN_PATH + "labels.csv"
MARGIN = 10
# ==============================================


def crop_volume(vol_path: str, seg_path: str, output_dir: str, output_name: str, margin: int = 10) -> bool:
    """
    Call crop_with_mask.py to crop a volume.
    
    Args:
        vol_path: Path to input volume
        seg_path: Path to segmentation mask
        output_dir: Output directory
        output_name: Base name for output (without extension)
        margin: Margin in voxels
        
    Returns:
        True if successful
    """
    try:
        cmd = [
            'python3', './crop_with_mask.py',
            vol_path,
            seg_path,
            output_dir,
            output_name,
            str(margin)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            return True
        else:
            print(f"      ❌ Error: {result.stderr.strip()}")
            return False
            
    except Exception as e:
        print(f"      ❌ Exception: {e}")
        return False


def process_all_studies(force_recompute: bool = False):
    """Process all studies and organize outputs by study_id."""
    
    print(f"\n{'='*80}")
    print(f"CROP PIPELINE - Organize by Study")
    print(f"{'='*80}")
    print(f"Input volumes:     {INPUT_DIR}")
    print(f"Segmentations:     {SEG_DIR}")
    print(f"Output directory:  {OUTPUT_DIR}")
    print(f"Labels CSV:        {LABELS_CSV}")
    print(f"Margin:            {MARGIN} voxels")
    print(f"Force recompute:   {force_recompute}")
    print(f"{'='*80}\n")
    
    # Check directories exist
    if not INPUT_DIR.exists():
        print(f"❌ Input directory not found: {INPUT_DIR}")
        sys.exit(1)
    
    if not SEG_DIR.exists():
        print(f"❌ Segmentation directory not found: {SEG_DIR}")
        sys.exit(1)
    
    if not os.path.exists(LABELS_CSV):
        print(f"❌ Labels CSV not found: {LABELS_CSV}")
        sys.exit(1)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load labels
    labels_df = pd.read_csv(LABELS_CSV)
    study_ids = labels_df["StudyInstanceUID"].unique()
    
    print(f"Found {len(study_ids)} studies in labels.csv\n")
    
    # Counters
    successful = 0
    skipped = 0
    failed = 0
    failed_studies = []
    
    # Process each study
    for idx, study_id in enumerate(study_ids, 1):
        print(f"[{idx}/{len(study_ids)}] Processing study: {study_id[:50]}...")
        
        # Get all series for this study
        study_rows = labels_df[labels_df["StudyInstanceUID"] == study_id]
        
        # Create study output directory
        study_output_dir = OUTPUT_DIR / study_id
        study_output_dir.mkdir(parents=True, exist_ok=True)
        
        study_successful = 0
        study_skipped = 0
        study_failed = 0
        
        # Process each series
        for _, row in study_rows.iterrows():
            series_id = row["SeriesInstanceUID"]
            phase = row["Label"]
            
            # Build file paths
            vol_filename = f"{study_id}_{series_id}.nii.gz"
            seg_filename = f"{study_id}_{series_id}_seg.nii.gz"
            
            vol_path = INPUT_DIR / vol_filename
            seg_path = SEG_DIR / seg_filename
            
            output_vol = study_output_dir / f"{study_id}_{series_id}_crop.nii.gz"
            output_seg = study_output_dir / f"{study_id}_{series_id}_seg.nii.gz"
            
            # Check if volume exists
            if not vol_path.exists():
                print(f"  ⚠️  Volume not found: {vol_filename}")
                study_failed += 1
                continue
            
            # Check if segmentation exists
            if not seg_path.exists():
                print(f"  ⚠️  Segmentation not found: {seg_filename}")
                study_failed += 1
                continue
            
            # Skip if already processed (unless force recompute)
            if not force_recompute and output_vol.exists() and output_seg.exists():
                print(f"  ⏭️  {phase} - already processed, skipping")
                study_skipped += 1
                continue
            
            # Crop the volume
            print(f"  🔄 {phase} - cropping...")
            success = crop_volume(
                str(vol_path),
                str(seg_path),
                str(study_output_dir),
                f"{study_id}_{series_id}",
                MARGIN
            )
            
            if success:
                print(f"      ✅ Success")
                study_successful += 1
            else:
                study_failed += 1
        
        # Study summary
        total_series = len(study_rows)
        print(f"  Study summary: {study_successful} success, {study_skipped} skipped, {study_failed} failed (of {total_series} series)")
        
        # Update global counters
        successful += study_successful
        skipped += study_skipped
        failed += study_failed
        
        if study_failed > 0 and study_successful == 0:
            failed_studies.append((study_id, f"{study_failed} series failed"))
        
        print()
    
    # Final summary
    print(f"{'='*80}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*80}")
    total_processed = successful + skipped + failed
    print(f"Total series processed: {total_processed}")
    print(f"  ✅ Successful: {successful}")
    print(f"  ⏭️  Skipped:    {skipped}")
    print(f"  ❌ Failed:     {failed}")
    print(f"\nStudies processed: {len(study_ids)}")
    
    if failed_studies:
        print(f"\nFailed studies ({len(failed_studies)}):")
        for study_id, reason in failed_studies[:10]:
            print(f"  - {study_id[:50]}... → {reason}")
        if len(failed_studies) > 10:
            print(f"  ... and {len(failed_studies) - 10} more")
    
    # Count final outputs
    num_studies_with_output = sum(1 for p in OUTPUT_DIR.iterdir() if p.is_dir())
    num_crops = sum(1 for p in OUTPUT_DIR.rglob("*_crop.nii.gz"))
    num_segs = sum(1 for p in OUTPUT_DIR.rglob("*_seg.nii.gz"))
    
    print(f"\nFinal output:")
    print(f"  Studies with output: {num_studies_with_output}")
    print(f"  Cropped volumes:     {num_crops}")
    print(f"  Cropped masks:       {num_segs}")
    print(f"{'='*80}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Crop CT volumes and organize by study')
    parser.add_argument('--force', action='store_true', 
                       help='Force recompute even if outputs exist')
    
    args = parser.parse_args()
    
    process_all_studies(force_recompute=args.force)
