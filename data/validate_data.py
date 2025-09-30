import SimpleITK as sitk
import numpy as np
import os

def validate_nifti_for_segmentation(nifti_path: str, target_spacing: list = [1.5, 1.5, 1.5]) -> dict:
    """
    Quick validation function to check if NIfTI is suitable for segmentation.
    Add this to your existing pipeline to diagnose issues.
    """
    try:
        image = sitk.ReadImage(nifti_path)
        
        # Get image properties
        spacing = image.GetSpacing()
        size = image.GetSize()
        direction = image.GetDirection()
        origin = image.GetOrigin()
        
        # Get HU values
        array = sitk.GetArrayFromImage(image)
        hu_min, hu_max, hu_mean = array.min(), array.max(), array.mean()
        
        # Check orientation
        direction_matrix = np.array(direction).reshape(3, 3)
        is_standard_orientation = np.allclose(direction_matrix, np.eye(3), atol=0.1)
        
        # Check spacing
        spacing_ok = all(abs(s - t) / t < 0.3 for s, t in zip(spacing, target_spacing))
        
        # Check HU range
        hu_range_ok = -1500 <= hu_min and hu_max <= 4000
        
        # Check size
        size_ok = all(s >= 64 for s in size)
        
        results = {
            'file': os.path.basename(nifti_path),
            'spacing': spacing,
            'size': size,
            'hu_range': (float(hu_min), float(hu_max)),
            'hu_mean': float(hu_mean),
            'orientation_ok': is_standard_orientation,
            'spacing_ok': spacing_ok,
            'hu_range_ok': hu_range_ok,
            'size_ok': size_ok,
            'overall_ok': all([is_standard_orientation, spacing_ok, hu_range_ok, size_ok])
        }
        
        return results
        
    except Exception as e:
        return {'file': os.path.basename(nifti_path), 'error': str(e)}

def check_all_nifti_files(nifti_root_dir: str):
    """Check all NIfTI files in your output directory."""
    print("=== VALIDATING EXISTING NIFTI FILES ===")
    
    issues = []
    total_files = 0
    
    for root, dirs, files in os.walk(nifti_root_dir):
        for file in files:
            if file.endswith('.nii.gz'):
                total_files += 1
                nifti_path = os.path.join(root, file)
                result = validate_nifti_for_segmentation(nifti_path)
                
                if 'error' in result:
                    print(f"❌ {result['file']}: {result['error']}")
                    issues.append(result)
                elif not result['overall_ok']:
                    print(f"⚠️  {result['file']}:")
                    if not result['orientation_ok']:
                        print(f"   - Orientation issue (non-standard)")
                    if not result['spacing_ok']:
                        print(f"   - Spacing: {result['spacing']} (expected ~1.5mm)")
                    if not result['hu_range_ok']:
                        print(f"   - HU range: {result['hu_range']} (unusual)")
                    if not result['size_ok']:
                        print(f"   - Size: {result['size']} (too small)")
                    issues.append(result)
                else:
                    print(f"✅ {result['file']}: OK")
    
    print(f"\n=== SUMMARY ===")
    print(f"Total files checked: {total_files}")
    print(f"Files with issues: {len(issues)}")
    
    if issues:
        print(f"\n=== RECOMMENDED ACTIONS ===")
        print("1. Reprocess DICOM files with proper standardization")
        print("2. Use SimpleITK instead of dcm2niix for better control")
        print("3. Apply image reorientation and resampling")
        
    return issues
from configs import ORIGINAL_DIR
# Run this on your existing NIfTI files to diagnose issues
if __name__ == "__main__":
    # Replace with your actual path
    nifti_root_dir = ORIGINAL_DIR
    issues = check_all_nifti_files(nifti_root_dir)