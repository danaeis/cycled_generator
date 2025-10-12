#!/usr/bin/env python3
"""
Process pars-ct dataset phase-labeled series.

This script processes the pars-ct dataset to extract series with valid phase labels
from the phase_labels.csv file and converts them to NIfTI format.
"""

import os
import sys
import pandas as pd
import pydicom
import nibabel as nib
import numpy as np
from pathlib import Path
import json
import shutil
from typing import Dict, List, Tuple, Optional

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
    
    # Clean phase labels - only keep valid ones
    valid_phases = ['w/o', 'art', 'por', 'del']
    df['Phase Label'] = df['Phase Label'].fillna('')
    df = df[df['Phase Label'].isin(valid_phases)]
    
    print(f"Loaded {len(df)} valid phase-labeled series")
    return df

def find_case_directories(data_path: str) -> Dict[str, str]:
    """
    Find all case directories in the pars-ct dataset.
    
    Args:
        data_path: Path to the pars-ct dataset root
        
    Returns:
        Dictionary mapping case numbers to directory paths
    """
    case_dirs = {}
    
    for item in os.listdir(data_path):
        item_path = os.path.join(data_path, item)
        if os.path.isdir(item_path):
            # Check if it has SCANS and ASSESSORS directories
            if os.path.exists(os.path.join(item_path, 'SCANS')) and os.path.exists(os.path.join(item_path, 'ASSESSORS')):
                case_dirs[item] = item_path
                
    print(f"Found {len(case_dirs)} case directories")
    return case_dirs

def get_series_info_from_scans(scans_path: str) -> Dict[str, Dict]:
    """
    Extract series information from SCANS directory.
    
    Args:
        scans_path: Path to the SCANS directory
        
    Returns:
        Dictionary mapping series numbers to series info
    """
    series_info = {}
    
    for series_dir in os.listdir(scans_path):
        series_path = os.path.join(scans_path, series_dir)
        if os.path.isdir(series_path):
            dicom_path = os.path.join(series_path, 'DICOM')
            if os.path.exists(dicom_path):
                # Read a sample DICOM file to get series info
                dicom_files = [f for f in os.listdir(dicom_path) if f.lower().endswith('.dcm')]
                if dicom_files:
                    sample_file = os.path.join(dicom_path, dicom_files[0])
                    try:
                        ds = pydicom.dcmread(sample_file, stop_before_pixels=True)
                        series_info[series_dir] = {
                            'series_number': getattr(ds, 'SeriesNumber', series_dir),
                            'series_description': getattr(ds, 'SeriesDescription', ''),
                            'series_uid': getattr(ds, 'SeriesInstanceUID', ''),
                            'dicom_path': dicom_path
                        }
                    except Exception as e:
                        print(f"Warning: Could not read DICOM from {sample_file}: {e}")
                        
    return series_info

def convert_dicom_series_to_nifti(dicom_path: str, output_path: str) -> bool:
    """
    Convert a DICOM series to NIfTI format.
    
    Args:
        dicom_path: Path to the DICOM series directory
        output_path: Path for the output NIfTI file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get all DICOM files
        dicom_files = [f for f in os.listdir(dicom_path) if f.lower().endswith('.dcm')]
        if not dicom_files:
            return False
            
        # Read and sort DICOM files
        slices = []
        for dcm_file in dicom_files:
            dcm_path = os.path.join(dicom_path, dcm_file)
            try:
                ds = pydicom.dcmread(dcm_path, force=True)
                slices.append(ds)
            except Exception as e:
                print(f"Warning: Could not read {dcm_path}: {e}")
                
        if not slices:
            return False
            
        # Sort by InstanceNumber
        slices.sort(key=lambda x: getattr(x, 'InstanceNumber', 0))
        
        # Build 3D volume
        sample_ds = slices[0]
        rows = sample_ds.Rows
        cols = sample_ds.Columns
        n_slices = len(slices)
        
        volume = np.zeros((rows, cols, n_slices), dtype=np.int16)
        
        for i, ds in enumerate(slices):
            volume[:, :, i] = ds.pixel_array
            
        # Create affine matrix
        pixel_spacing = getattr(sample_ds, 'PixelSpacing', [1.0, 1.0])
        slice_thickness = getattr(sample_ds, 'SliceThickness', 1.0)
        
        # Simple affine (can be improved)
        affine = np.eye(4)
        affine[0, 0] = pixel_spacing[0]
        affine[1, 1] = pixel_spacing[1]
        affine[2, 2] = slice_thickness
        
        # Save as NIfTI
        nifti_img = nib.Nifti1Image(volume, affine)
        nib.save(nifti_img, output_path)
        
        return True
        
    except Exception as e:
        print(f"Error converting DICOM series to NIfTI: {e}")
        return False

def process_pars_ct_phases(data_path: str, output_dir: str, phase_labels_csv: str):
    """
    Main function to process pars-ct phase-labeled series.
    
    Args:
        data_path: Path to the pars-ct dataset
        output_dir: Output directory for processed files
        phase_labels_csv: Path to phase labels CSV file
    """
    # Load phase labels
    phase_df = load_phase_labels(phase_labels_csv)
    
    # Find case directories
    case_dirs = find_case_directories(data_path)
    
    # Create output directories
    nifti_output_dir = os.path.join(output_dir, 'nifti_volumes')
    os.makedirs(nifti_output_dir, exist_ok=True)
    
    # Process each case
    processed_cases = 0
    for _, row in phase_df.iterrows():
        case_number = str(row['Case Number'])
        series_number = str(row['Series Number'])
        phase_label = row['Phase Label']
        
        # Find case directory
        case_dir = None
        for case_name, case_path in case_dirs.items():
            if case_number in case_name:
                case_dir = case_path
                break
                
        if not case_dir:
            print(f"Warning: Case directory not found for case {case_number}")
            continue
            
        # Get series info
        scans_path = os.path.join(case_dir, 'SCANS')
        series_info = get_series_info_from_scans(scans_path)
        
        if series_number not in series_info:
            print(f"Warning: Series {series_number} not found in case {case_number}")
            continue
            
        # Convert to NIfTI
        dicom_path = series_info[series_number]['dicom_path']
        output_filename = f"{case_number}_{series_number}_{phase_label}.nii.gz"
        output_path = os.path.join(nifti_output_dir, output_filename)
        
        if os.path.exists(output_path):
            print(f"NIfTI already exists: {output_filename}")
            continue
            
        print(f"Converting case {case_number}, series {series_number} ({phase_label})...")
        
        if convert_dicom_series_to_nifti(dicom_path, output_path):
            print(f"Successfully created: {output_filename}")
            processed_cases += 1
        else:
            print(f"Failed to convert case {case_number}, series {series_number}")
            
    print(f"Processed {processed_cases} series successfully")

def main():
    if len(sys.argv) != 4:
        print("Usage: python process_pars_ct_phases.py <data_path> <output_dir> <phase_labels_csv>")
        sys.exit(1)
        
    data_path = sys.argv[1]
    output_dir = sys.argv[2]
    phase_labels_csv = sys.argv[3]
    
    if not os.path.exists(data_path):
        print(f"Error: Data path does not exist: {data_path}")
        sys.exit(1)
        
    if not os.path.exists(phase_labels_csv):
        print(f"Error: Phase labels CSV does not exist: {phase_labels_csv}")
        sys.exit(1)
        
    process_pars_ct_phases(data_path, output_dir, phase_labels_csv)

if __name__ == "__main__":
    main()

