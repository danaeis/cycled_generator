#!/usr/bin/env python3
"""
Apply cropping pipeline to pars-ct volumes.

This script applies the cropping pipeline to pars-ct volumes using their segmentation masks.
"""

import os
import sys
import pandas as pd
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

def crop_volume_with_mask(volume_path: str, mask_path: str, output_dir: str, output_name: str, margin: int = 10) -> bool:
    """
    Crop a volume using a mask with specified margin.
    
    Args:
        volume_path: Path to the input volume NIfTI file
        mask_path: Path to the segmentation mask NIfTI file
        output_dir: Directory to save cropped volume
        output_name: Name for the output file
        margin: Margin in voxels around the mask
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Import the crop_with_mask function from the existing script
        import subprocess
        
        # Run the crop_with_mask script
        cmd = ['python3', 'data/crop_with_mask.py', volume_path, mask_path, output_dir, output_name, str(margin)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Cropped volume saved: {output_name}")
            return True
        else:
            print(f"Failed to crop volume: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Error cropping volume: {e}")
        return False

def crop_pars_ct_volumes(output_dir: str, phase_labels_csv: str):
    """
    Main function to crop pars-ct volumes using their segmentation masks.
    
    Args:
        output_dir: Output directory containing processed data
        phase_labels_csv: Path to phase labels CSV file
    """
    # Load phase labels
    phase_df = load_phase_labels(phase_labels_csv)
    
    # Create output directories
    cropped_output_dir = os.path.join(output_dir, 'cropped_volumes')
    os.makedirs(cropped_output_dir, exist_ok=True)
    
    # Get available volumes and masks
    nifti_dir = os.path.join(output_dir, 'nifti_volumes')
    seg_dir = os.path.join(output_dir, 'segmentations')
    
    if not os.path.exists(nifti_dir):
        print(f"Error: NIfTI volumes directory not found: {nifti_dir}")
        return
        
    if not os.path.exists(seg_dir):
        print(f"Error: Segmentations directory not found: {seg_dir}")
        return
    
    # Process each series
    processed_series = 0
    for _, row in phase_df.iterrows():
        case_number = str(row['Case Number'])
        series_number = str(row['Series Number'])
        phase_label = row['Phase Label']
        
        # Skip if no phase label
        if not phase_label:
            continue
            
        # Check if volume exists
        volume_filename = f"{case_number}_{series_number}_{phase_label}.nii.gz"
        volume_path = os.path.join(nifti_dir, volume_filename)
        
        if not os.path.exists(volume_path):
            print(f"Warning: Volume not found: {volume_filename}")
            continue
            
        # Check if segmentation exists
        seg_filename = f"{case_number}_{series_number}_{phase_label}_seg.nii.gz"
        seg_path = os.path.join(seg_dir, seg_filename)
        
        if not os.path.exists(seg_path):
            print(f"Warning: Segmentation not found: {seg_filename}")
            continue
            
        # Check if cropped volume already exists
        cropped_filename = f"{case_number}_{series_number}_{phase_label}_crop.nii.gz"
        cropped_path = os.path.join(cropped_output_dir, cropped_filename)
        
        if os.path.exists(cropped_path):
            print(f"Cropped volume already exists: {cropped_filename}")
            continue
            
        # Crop the volume
        print(f"Cropping case {case_number}, series {series_number} ({phase_label})...")
        
        output_name = f"{case_number}_{series_number}_{phase_label}"
        
        if crop_volume_with_mask(volume_path, seg_path, cropped_output_dir, output_name, margin=10):
            processed_series += 1
            print(f"Successfully cropped: {cropped_filename}")
        else:
            print(f"Failed to crop {volume_filename}")
            
    print(f"Processed {processed_series} series successfully")

def main():
    if len(sys.argv) != 3:
        print("Usage: python crop_pars_ct_volumes.py <output_dir> <phase_labels_csv>")
        sys.exit(1)
        
    output_dir = sys.argv[1]
    phase_labels_csv = sys.argv[2]
    
    if not os.path.exists(output_dir):
        print(f"Error: Output directory does not exist: {output_dir}")
        sys.exit(1)
        
    if not os.path.exists(phase_labels_csv):
        print(f"Error: Phase labels CSV does not exist: {phase_labels_csv}")
        sys.exit(1)
        
    crop_pars_ct_volumes(output_dir, phase_labels_csv)

if __name__ == "__main__":
    main()
