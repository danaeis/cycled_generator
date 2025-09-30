#!/usr/bin/env python3
"""
Emergency fix script to reprocess your corrupted NIfTI files.
This will fix orientation, spacing, and HU value issues.
"""

import os
import SimpleITK as sitk
import numpy as np
from pathlib import Path
import logging

def fix_corrupted_nifti(input_path: str, output_path: str, target_spacing: list = [1.5, 1.5, 1.5]) -> bool:
    """
    Fix a corrupted NIfTI file by:
    1. Reading and validating the image
    2. Fixing HU values if needed
    3. Reorienting to standard orientation
    4. Resampling to target spacing
    5. Saving with clean headers
    """
    try:
        print(f"Processing: {os.path.basename(input_path)}")
        
        # Step 1: Read image (may have warnings but continue)
        image = sitk.ReadImage(input_path)
        
        # Step 2: Check and fix HU values
        array = sitk.GetArrayFromImage(image)
        original_min, original_max = array.min(), array.max()
        
        # Detect and fix corrupted HU values
        if original_min < -1500 or original_max > 4000:
            print(f"  Fixing HU values: {original_min:.0f} to {original_max:.0f}")
            
            # Clip extreme values
            array = np.clip(array, -1024, 3071)
            
            # If values are in wrong range (e.g., 0-4095), convert to HU
            if original_min >= 0 and original_max > 3000:
                # Assuming 12-bit DICOM converted incorrectly
                array = array.astype(np.float32) - 1024
                print(f"  Converted to HU range: {array.min():.0f} to {array.max():.0f}")
            
            # Create new image with fixed values
            fixed_image = sitk.GetImageFromArray(array)
            fixed_image.CopyInformation(image)
            image = fixed_image
        
        # Step 3: Fix orientation to LPS (Left-Posterior-Superior)
        print(f"  Original spacing: {image.GetSpacing()}")
        print(f"  Original size: {image.GetSize()}")
        
        try:
            # Reorient to standard LPS orientation
            image = sitk.DICOMOrient(image, 'LPS')
            print(f"  Reoriented to LPS")
        except Exception as e:
            print(f"  Warning: Could not reorient: {e}")
        
        # Step 4: Resample to target spacing
        current_spacing = image.GetSpacing()
        if any(abs(c - t) / t > 0.2 for c, t in zip(current_spacing, target_spacing)):
            print(f"  Resampling to {target_spacing}")
            
            # Calculate new size
            original_size = image.GetSize()
            new_size = [
                int(round(original_size[i] * current_spacing[i] / target_spacing[i]))
                for i in range(3)
            ]
            
            # Ensure minimum size
            new_size = [max(s, 64) for s in new_size]
            
            # Create resampler
            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputSpacing(target_spacing)
            resampler.SetSize(new_size)
            resampler.SetOutputDirection(image.GetDirection())
            resampler.SetOutputOrigin(image.GetOrigin())
            resampler.SetTransform(sitk.Transform())
            resampler.SetDefaultPixelValue(-1000)  # Air HU value
            resampler.SetInterpolator(sitk.sitkLinear)
            
            image = resampler.Execute(image)
            print(f"  New size: {image.GetSize()}")
        
        # Step 5: Save with clean headers
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Force clean NIfTI header by using specific writer settings
        writer = sitk.ImageFileWriter()
        writer.SetFileName(output_path)
        writer.SetUseCompression(True)
        writer.Execute(image)
        
        # Validate the saved file
        try:
            test_read = sitk.ReadImage(output_path)
            final_spacing = test_read.GetSpacing()
            final_size = test_read.GetSize()
            
            test_array = sitk.GetArrayFromImage(test_read)
            final_hu_min, final_hu_max = test_array.min(), test_array.max()
            
            print(f"  ✅ Success - Final spacing: {final_spacing}")
            print(f"  ✅ Final size: {final_size}")
            print(f"  ✅ Final HU range: {final_hu_min:.0f} to {final_hu_max:.0f}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Validation failed: {e}")
            return False
            
    except Exception as e:
        print(f"  ❌ Failed to process: {e}")
        return False

def fix_all_nifti_files(input_dir: str, output_dir: str, target_spacing: list = [1.5, 1.5, 1.5]):
    """Fix all NIfTI files in the input directory."""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Find all NIfTI files
    nifti_files = list(input_path.rglob("*.nii.gz"))
    
    print(f"Found {len(nifti_files)} NIfTI files to fix")
    print(f"Target spacing: {target_spacing}")
    print("=" * 50)
    
    success_count = 0
    failure_count = 0
    
    for nifti_file in nifti_files:
        # Create output path maintaining directory structure
        relative_path = nifti_file.relative_to(input_path)
        output_file = output_path / relative_path
        
        if fix_corrupted_nifti(str(nifti_file), str(output_file), target_spacing):
            success_count += 1
        else:
            failure_count += 1
        
        print("-" * 30)
    
    print(f"\n=== SUMMARY ===")
    print(f"✅ Successfully fixed: {success_count}")
    print(f"❌ Failed to fix: {failure_count}")
    print(f"📁 Fixed files saved to: {output_dir}")

def quick_validation(nifti_dir: str):
    """Quick validation of the fixed files."""
    print("\n=== VALIDATING FIXED FILES ===")
    
    nifti_files = list(Path(nifti_dir).rglob("*.nii.gz"))
    
    spacing_issues = 0
    hu_issues = 0
    total_files = len(nifti_files)
    
    for nifti_file in nifti_files[:10]:  # Check first 10
        try:
            image = sitk.ReadImage(str(nifti_file))
            spacing = image.GetSpacing()
            
            array = sitk.GetArrayFromImage(image)
            hu_min, hu_max = array.min(), array.max()
            
            # Check spacing (should be close to 1.5mm)
            if any(abs(s - 1.5) > 0.3 for s in spacing):
                spacing_issues += 1
            
            # Check HU range
            if hu_min < -1500 or hu_max > 4000:
                hu_issues += 1
                
        except Exception as e:
            print(f"Error checking {nifti_file}: {e}")
    
    print(f"Checked {min(10, total_files)} files:")
    print(f"  Spacing issues: {spacing_issues}")
    print(f"  HU range issues: {hu_issues}")
    
    if spacing_issues == 0 and hu_issues == 0:
        print("🎉 All checked files look good!")
    else:
        print("⚠️  Some issues remain - may need manual review")

if __name__ == "__main__":
    # Configure paths
    INPUT_DIR = "../ncct_cect/vindr_ds/nifti_unprocessed_volumes"  # Your current corrupted NIfTI directory
    OUTPUT_DIR = "../ncct_cect/vindr_ds/nifti_fixed_volumes"       # Where to save fixed files
    
    # Target spacing for TotalSegmentator
    TARGET_SPACING = [1.5, 1.5, 1.5]
    
    # Fix all files
    fix_all_nifti_files(INPUT_DIR, OUTPUT_DIR, TARGET_SPACING)
    
    # Quick validation
    quick_validation(OUTPUT_DIR)
    
    print(f"\n✅ NEXT STEPS:")
    print(f"1. Update your dataset path to point to: {OUTPUT_DIR}")
    print(f"2. Re-run your segmentation pipeline")
    print(f"3. The results should be dramatically better!")