#!/usr/bin/env python3
"""
Apply registration pipeline to pars-ct volumes.

This script applies the registration pipeline to pars-ct volumes, registering all series
to the non-contrast (w/o) volume for each case.
"""

import os
import sys
import pandas as pd
import SimpleITK as sitk
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

def register_study_pars_ct(case_number: str, cropped_dir: str, output_dir: str, phase_df: pd.DataFrame):
    """
    Register all series in a pars-ct study to the non-contrast volume.
    
    Args:
        case_number: Case number
        cropped_dir: Path to cropped volumes directory
        output_dir: Where to save registered results
        phase_df: DataFrame with phase labels information
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find the non-contrast series for this case
    case_data = phase_df[phase_df['Case Number'] == case_number]
    nc_series = case_data[case_data['Phase Label'] == 'w/o']
    
    if nc_series.empty:
        print(f"⚠️ No non-contrast series found for case {case_number}, skipping...")
        return
    
    # Get the first non-contrast series
    nc_series_number = str(nc_series.iloc[0]['Series Number'])
    nc_filename = f"{case_number}_{nc_series_number}_w/o_crop.nii.gz"
    nc_file = os.path.join(cropped_dir, nc_filename)
    
    if not os.path.exists(nc_file):
        print(f"⚠️ Non-contrast file not found: {nc_file}")
        return
    
    # Load the fixed (non-contrast) image
    fixed = sitk.ReadImage(nc_file)
    print(f"📊 Fixed image shape: {fixed.GetSize()}")
    
    # Process all series for this case
    for _, row in case_data.iterrows():
        series_number = str(row['Series Number'])
        phase_label = row['Phase Label']
        
        # Skip non-contrast (it's our reference)
        if phase_label == 'w/o':
            continue
            
        # Check if cropped volume exists
        moving_filename = f"{case_number}_{series_number}_{phase_label}_crop.nii.gz"
        moving_file = os.path.join(cropped_dir, moving_filename)
        
        if not os.path.exists(moving_file):
            print(f"⚠️ Moving file not found: {moving_file}")
            continue
        
        try:
            # Load moving image
            moving = sitk.ReadImage(moving_file)
            
            # Initialize transform
            initial_transform = sitk.CenteredTransformInitializer(
                fixed, moving, sitk.Euler3DTransform(),
                sitk.CenteredTransformInitializerFilter.GEOMETRY
            )
            
            # Registration method
            registration = sitk.ImageRegistrationMethod()
            registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
            registration.SetMetricSamplingStrategy(registration.RANDOM)
            registration.SetMetricSamplingPercentage(0.2)
            registration.SetInterpolator(sitk.sitkLinear)
            registration.SetOptimizerAsGradientDescent(
                learningRate=1.0,
                numberOfIterations=100,
                convergenceMinimumValue=1e-6,
                convergenceWindowSize=10
            )
            registration.SetOptimizerScalesFromPhysicalShift()
            registration.SetInitialTransform(initial_transform, inPlace=False)
            
            # Execute registration
            final_transform = registration.Execute(fixed, moving)
            
            # Print metrics
            print(f"📊 {case_number} | {series_number} ({phase_label}): {registration.GetMetricValue():.6f}")
            
            # Resample moving image
            registered = sitk.Resample(
                moving, fixed, final_transform,
                sitk.sitkLinear, 0.0, moving.GetPixelID()
            )
            
            # Save registered image
            out_vol_path = os.path.join(output_dir, f"{case_number}_{series_number}_{phase_label}_registered.nii.gz")
            sitk.WriteImage(registered, out_vol_path)
            print(f"   ✅ Saved registered volume: {out_vol_path}")
            
            # Apply same transform to segmentation mask if it exists
            seg_filename = f"{case_number}_{series_number}_{phase_label}_seg_crop.nii.gz"
            seg_file = os.path.join(cropped_dir, seg_filename)
            
            if os.path.exists(seg_file):
                seg = sitk.ReadImage(seg_file)
                
                registered_seg = sitk.Resample(
                    seg, fixed, final_transform,
                    sitk.sitkNearestNeighbor, 0, seg.GetPixelID()
                )
                
                out_seg_path = os.path.join(output_dir, f"{case_number}_{series_number}_{phase_label}_registered_seg.nii.gz")
                sitk.WriteImage(registered_seg, out_seg_path)
                print(f"   ✅ Saved registered mask: {out_seg_path}")
                
        except Exception as e:
            print(f"❌ Error registering {case_number}_{series_number}_{phase_label}: {e}")
    
    # Copy reference (non-contrast) volume and mask
    nc_out = os.path.join(output_dir, f"{case_number}_{nc_series_number}_w/o_registered.nii.gz")
    shutil.copy(nc_file, nc_out)
    
    # Copy non-contrast segmentation if it exists
    nc_seg_filename = f"{case_number}_{nc_series_number}_w/o_seg_crop.nii.gz"
    nc_seg_file = os.path.join(cropped_dir, nc_seg_filename)
    if os.path.exists(nc_seg_file):
        nc_seg_out = os.path.join(output_dir, f"{case_number}_{nc_series_number}_w/o_registered_seg.nii.gz")
        shutil.copy(nc_seg_file, nc_seg_out)
    
    print(f"   Copied reference non-contrast for {case_number}")

def register_pars_ct_volumes(output_dir: str, phase_labels_csv: str):
    """
    Main function to register pars-ct volumes.
    
    Args:
        output_dir: Output directory containing processed data
        phase_labels_csv: Path to phase labels CSV file
    """
    # Load phase labels
    phase_df = load_phase_labels(phase_labels_csv)
    
    # Create output directories
    registered_output_dir = os.path.join(output_dir, 'registered_volumes')
    os.makedirs(registered_output_dir, exist_ok=True)
    
    # Get cropped volumes directory
    cropped_dir = os.path.join(output_dir, 'cropped_volumes')
    
    if not os.path.exists(cropped_dir):
        print(f"Error: Cropped volumes directory not found: {cropped_dir}")
        return
    
    # Get unique case numbers
    case_numbers = phase_df['Case Number'].unique()
    
    print(f"Processing {len(case_numbers)} cases for registration...")
    
    # Process each case
    processed_cases = 0
    for case_number in case_numbers:
        case_number = str(case_number)
        print(f"\nProcessing case: {case_number}")
        
        case_out = os.path.join(registered_output_dir, case_number)
        register_study_pars_ct(case_number, cropped_dir, case_out, phase_df)
        processed_cases += 1
        
    print(f"\n✅ Registration completed for {processed_cases} cases")

def main():
    if len(sys.argv) != 3:
        print("Usage: python register_pars_ct_volumes.py <output_dir> <phase_labels_csv>")
        sys.exit(1)
        
    output_dir = sys.argv[1]
    phase_labels_csv = sys.argv[2]
    
    if not os.path.exists(output_dir):
        print(f"Error: Output directory does not exist: {output_dir}")
        sys.exit(1)
        
    if not os.path.exists(phase_labels_csv):
        print(f"Error: Phase labels CSV does not exist: {phase_labels_csv}")
        sys.exit(1)
        
    register_pars_ct_volumes(output_dir, phase_labels_csv)

if __name__ == "__main__":
    main()

