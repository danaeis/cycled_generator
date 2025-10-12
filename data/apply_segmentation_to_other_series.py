#!/usr/bin/env python3
"""
Apply segmentation to other series in pars-ct dataset.

This script applies the existing segmentation masks (from ASSESSORS) to other series
that don't have segmentation, using TotalSegmentator for inference.
"""

import os
import sys
import pandas as pd
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List

def load_phase_labels(phase_labels_csv: str) -> pd.DataFrame:
    """
    Load and clean the phase labels CSV file.
    
    Args:
        phase_labels_csv: Path to the phase labels CSV file
        
    Returns:
        Cleaned DataFrame with phase labels
    """
    print(f"Loading phase labels from: {phase_labels_csv}")
    
    # Read CSV with proper handling of empty cells
    df = pd.read_csv(phase_labels_csv)
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Remove completely empty rows
    df = df.dropna(how='all')
    
    # Fill empty case numbers with forward fill
    df['Case Number'] = df['Case Number'].fillna(method='ffill')
    
    # Filter out rows with no case number
    df = df.dropna(subset=['Case Number'])
    
    # Convert case numbers to string for consistency
    df['Case Number'] = df['Case Number'].astype(str)
    
    # Clean phase labels
    df['Phase Label'] = df['Phase Label'].fillna('')
    
    print(f"Loaded {len(df)} series entries")
    return df

def get_cases_with_segmentation(output_dir: str) -> List[str]:
    """
    Get list of cases that have segmentation masks available.
    
    Args:
        output_dir: Output directory containing processed pars-ct data
        
    Returns:
        List of case names that have segmentation
    """
    pars_ct_dir = os.path.join(output_dir, 'pars_ct_processed')
    if not os.path.exists(pars_ct_dir):
        return []
        
    cases_with_seg = []
    for case_dir in os.listdir(pars_ct_dir):
        case_path = os.path.join(pars_ct_dir, case_dir)
        if os.path.isdir(case_path):
            # Check if it has segmentation masks
            nifti_dir = os.path.join(case_path, 'NIFTI')
            if os.path.exists(nifti_dir):
                seg_files = [f for f in os.listdir(nifti_dir) if 'ON_' in f and f.endswith('.nii')]
                if seg_files:
                    cases_with_seg.append(case_dir)
                    
    print(f"Found {len(cases_with_seg)} cases with segmentation masks")
    return cases_with_seg

def apply_totalsegmentator_to_series(nifti_path: str, output_dir: str) -> bool:
    """
    Apply TotalSegmentator to a NIfTI volume.
    
    Args:
        nifti_path: Path to the input NIfTI file
        output_dir: Directory to save segmentation results
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create output directory for this series
        base_name = os.path.splitext(os.path.basename(nifti_path))[0]
        seg_dir = os.path.join(output_dir, f"{base_name}_segs")
        os.makedirs(seg_dir, exist_ok=True)
        
        # Run TotalSegmentator
        cmd = ['TotalSegmentator', '-i', nifti_path, '-o', seg_dir]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"TotalSegmentator completed for {base_name}")
            return True
        else:
            print(f"TotalSegmentator failed for {base_name}: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Error running TotalSegmentator on {nifti_path}: {e}")
        return False

def combine_segmentation_masks(seg_dir: str, output_path: str) -> bool:
    """
    Combine individual segmentation masks into a single mask.
    
    Args:
        seg_dir: Directory containing individual segmentation masks
        output_path: Path for the combined mask output
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Import the combine_masks function from the existing script
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__)))
        
        # Run the combine_masks script
        cmd = ['python3', 'data/combine_masks.py', seg_dir, output_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Combined masks saved to: {output_path}")
            return True
        else:
            print(f"Failed to combine masks: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Error combining masks: {e}")
        return False

def apply_segmentation_to_other_series(output_dir: str, phase_labels_csv: str):
    """
    Main function to apply segmentation to series without existing segmentation.
    
    Args:
        output_dir: Output directory containing processed data
        phase_labels_csv: Path to phase labels CSV file
    """
    # Load phase labels
    phase_df = load_phase_labels(phase_labels_csv)
    
    # Get cases with segmentation
    cases_with_seg = get_cases_with_segmentation(output_dir)
    
    # Create segmentation output directory
    seg_output_dir = os.path.join(output_dir, 'segmentations')
    os.makedirs(seg_output_dir, exist_ok=True)
    
    # Process each series
    processed_series = 0
    for _, row in phase_df.iterrows():
        case_number = str(row['Case Number'])
        series_number = str(row['Series Number'])
        phase_label = row['Phase Label']
        
        # Check if this case has segmentation
        case_with_seg = None
        for case in cases_with_seg:
            if case_number in case:
                case_with_seg = case
                break
                
        if not case_with_seg:
            print(f"Warning: No segmentation found for case {case_number}")
            continue
            
        # Check if NIfTI volume exists
        nifti_filename = f"{case_number}_{series_number}_{phase_label}.nii.gz"
        nifti_path = os.path.join(output_dir, 'nifti_volumes', nifti_filename)
        
        if not os.path.exists(nifti_path):
            print(f"Warning: NIfTI volume not found: {nifti_filename}")
            continue
            
        # Check if segmentation already exists
        seg_filename = f"{case_number}_{series_number}_{phase_label}_seg.nii.gz"
        seg_path = os.path.join(seg_output_dir, seg_filename)
        
        if os.path.exists(seg_path):
            print(f"Segmentation already exists: {seg_filename}")
            continue
            
        # Apply TotalSegmentator
        print(f"Applying TotalSegmentator to case {case_number}, series {series_number} ({phase_label})...")
        
        if apply_totalsegmentator_to_series(nifti_path, seg_output_dir):
            # Combine the masks
            base_name = os.path.splitext(nifti_filename)[0]
            seg_dir = os.path.join(seg_output_dir, f"{base_name}_segs")
            
            if combine_segmentation_masks(seg_dir, seg_path):
                # Clean up temporary directory
                shutil.rmtree(seg_dir, ignore_errors=True)
                processed_series += 1
                print(f"Successfully created segmentation: {seg_filename}")
            else:
                print(f"Failed to combine masks for {seg_filename}")
        else:
            print(f"Failed to apply TotalSegmentator to {nifti_filename}")
            
    print(f"Processed {processed_series} series successfully")

def main():
    if len(sys.argv) != 3:
        print("Usage: python apply_segmentation_to_other_series.py <output_dir> <phase_labels_csv>")
        sys.exit(1)
        
    output_dir = sys.argv[1]
    phase_labels_csv = sys.argv[2]
    
    if not os.path.exists(output_dir):
        print(f"Error: Output directory does not exist: {output_dir}")
        sys.exit(1)
        
    if not os.path.exists(phase_labels_csv):
        print(f"Error: Phase labels CSV does not exist: {phase_labels_csv}")
        sys.exit(1)
        
    apply_segmentation_to_other_series(output_dir, phase_labels_csv)

if __name__ == "__main__":
    main()

