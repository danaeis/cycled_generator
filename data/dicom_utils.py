import os
import pickle
import pydicom
import pandas as pd
import SimpleITK as sitk
import nibabel as nib
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import json
import subprocess


# def get_slice_thickness(dicom_path: str) -> float:
#     """Get slice thickness from DICOM series (in mm)."""
#     try:
#         first_file = next(f for f in os.listdir(dicom_path) if f.endswith('.dcm'))
#         ds = pydicom.dcmread(os.path.join(dicom_path, first_file))
#         return float(getattr(ds, 'SliceThickness', float('inf')))  # Use infinity as fallback
#     except:
#         return float('inf')  # Return infinity if error occurs



def get_slice_thickness(dicom_path: str) -> float:
    """Get slice thickness from DICOM series (in mm)."""
    try:
        # Get all DICOM files
        dcm_files = [f for f in os.listdir(dicom_path) if f.endswith('.dcm') or f.endswith('.dicom')]
        if not dcm_files:
            print("dcm_files", dcm_files, dicom_path)
            return float('inf')
        
        # Read first file
        first_file_path = os.path.join(dicom_path, dcm_files[0])
        ds = pydicom.dcmread(first_file_path)
        
        # Method 1: Direct SliceThickness tag (0018,0050)
        try:
            if (0x0018, 0x0050) in ds:
                thickness = float(ds[0x0018, 0x0050].value)
                if thickness > 0:
                    return thickness
                else:
                    print("thickness", thickness)

        except (ValueError, TypeError, AttributeError):
            print("except")
            metadata_dict = {elem.name: elem.value for elem in ds}
            print(metadata_dict)
        
        # Method 2: SpacingBetweenSlices tag (0018,0088)
        try:
            if (0x0018, 0x0088) in ds:
                spacing = float(ds[0x0018, 0x0088].value)
                if spacing > 0:
                    return spacing
        except (ValueError, TypeError, AttributeError):
            pass
        
        metadata_dict = {elem.name: elem.value for elem in ds}
        print(metadata_dict)
        # Method 3: Calculate from ImagePositionPatient (for multi-slice series)
        if len(dcm_files) > 1:
            thickness = calculate_slice_thickness_from_positions(dicom_path, dcm_files)
            if thickness is not None and thickness > 0:
                return thickness
        
        # Method 4: Calculate from SliceLocation (for multi-slice series)
        if len(dcm_files) > 1:
            thickness = calculate_slice_thickness_from_slice_locations(dicom_path, dcm_files)
            if thickness is not None and thickness > 0:
                return thickness
        
        # Method 5: For single slice, use SpacingBetweenSlices if available
        if len(dcm_files) == 1:
            if hasattr(ds, 'SpacingBetweenSlices') and ds.SpacingBetweenSlices is not None:
                try:
                    return float(ds.SpacingBetweenSlices)
                except (ValueError, TypeError):
                    pass
            # Use a default value for single slice rather than excluding
            return 5.0  # Conservative default for single slice
        
        return float('inf')  # Truly failed
        
    except Exception as e:
        print(f"Error processing {dicom_path}: {e}")
        return float('inf')

def calculate_slice_thickness_from_positions(dicom_path: str, dcm_files: List[str]) -> Optional[float]:
    """Calculate slice thickness using ImagePositionPatient from multiple slices."""
    try:
        positions = []
        
        # Sample up to 20 files for calculation
        sample_files = dcm_files[:min(20, len(dcm_files))]
        
        for dcm_file in sample_files:
            try:
                ds = pydicom.dcmread(os.path.join(dicom_path, dcm_file))
                # ImagePositionPatient tag (0020,0032)
                if (0x0020, 0x0032) in ds:
                    ipp = ds[0x0020, 0x0032].value
                    if len(ipp) >= 3:
                        z_pos = float(ipp[2])
                        positions.append(z_pos)
            except:
                continue
        
        if len(positions) < 2:
            return None
        
        # Sort positions and calculate differences
        positions = sorted(set(positions))  # Remove duplicates and sort
        if len(positions) < 2:
            return None
        
        # Calculate differences between consecutive slices
        differences = []
        for i in range(len(positions) - 1):
            diff = abs(positions[i+1] - positions[i])
            if diff > 0.001:  # Filter out very small differences (likely noise)
                differences.append(diff)
        
        if not differences:
            return None
        
        # Return median difference as slice thickness
        return np.median(differences)
        
    except Exception:
        return None

def calculate_slice_thickness_from_slice_locations(dicom_path: str, dcm_files: List[str]) -> Optional[float]:
    """Calculate slice thickness from SliceLocation."""
    try:
        locations = []
        
        sample_files = dcm_files[:min(20, len(dcm_files))]
        
        for dcm_file in sample_files:
            try:
                ds = pydicom.dcmread(os.path.join(dicom_path, dcm_file))
                # SliceLocation tag (0020,1041)
                if (0x0020, 0x1041) in ds:
                    locations.append(float(ds[0x0020, 0x1041].value))
            except:
                continue
        
        if len(locations) < 2:
            return None
        
        unique_locations = sorted(set(locations))
        if len(unique_locations) < 2:
            return None
        
        differences = []
        for i in range(len(unique_locations) - 1):
            diff = abs(unique_locations[i+1] - unique_locations[i])
            if diff > 0.001:
                differences.append(diff)
        
        if not differences:
            return None
        
        return np.median(differences)
        
    except Exception:
        return None

def get_orientation_from_direction(direction_matrix):
    """
    Determine orientation code from direction matrix.
    """
    # Convert direction matrix to numpy array and reshape to 3x3
    direction = np.array(direction_matrix).reshape(3, 3)
    
    # Find the dominant axis for each dimension
    orientations = []
    axis_labels = [['L', 'R'], ['P', 'A'], ['I', 'S']]  # [Left/Right, Posterior/Anterior, Inferior/Superior]
    
    for col in range(3):
        axis_vector = direction[:, col]
        max_component = np.argmax(np.abs(axis_vector))
        if axis_vector[max_component] > 0:
            orientations.append(axis_labels[max_component][1])  # Positive direction
        else:
            orientations.append(axis_labels[max_component][0])  # Negative direction
    
    return ''.join(orientations)


def create_lps_direction_matrix():
    """
    Create a direction matrix for LPS orientation.
    LPS = Left-Posterior-Superior
    """
    # LPS direction matrix:
    # X-axis: Left to Right (negative X is left, positive X is right) -> [-1, 0, 0]
    # Y-axis: Posterior to Anterior (negative Y is posterior, positive Y is anterior) -> [0, -1, 0]  
    # Z-axis: Inferior to Superior (negative Z is inferior, positive Z is superior) -> [0, 0, 1]
    return (-1.0, 0.0, 0.0,   # X-axis direction
            0.0, -1.0, 0.0,   # Y-axis direction  
            0.0, 0.0, 1.0)    # Z-axis direction

def calculate_oriented_origin_and_size(sitk_image, target_direction, target_spacing):
    """
    Calculate the correct origin and size for reoriented image.
    """
    # Get original image properties
    original_size = sitk_image.GetSize()
    original_spacing = sitk_image.GetSpacing()
    original_origin = sitk_image.GetOrigin()
    original_direction = np.array(sitk_image.GetDirection()).reshape(3, 3)
    target_direction_matrix = np.array(target_direction).reshape(3, 3)
    
    # Calculate all 8 corner points in physical space
    corners_index = []
    for i in [0, original_size[0] - 1]:
        for j in [0, original_size[1] - 1]:
            for k in [0, original_size[2] - 1]:
                corners_index.append([i, j, k])
    
    # Transform corner indices to physical coordinates
    corners_physical = []
    for corner_idx in corners_index:
        phys_point = sitk_image.TransformIndexToPhysicalPoint(corner_idx)
        corners_physical.append(phys_point)
    
    corners_physical = np.array(corners_physical)
    
    # Find the bounding box in physical space
    min_phys = np.min(corners_physical, axis=0)
    max_phys = np.max(corners_physical, axis=0)
    
    # Calculate the physical extent
    extent = max_phys - min_phys
    
    # Calculate new size based on extent and target spacing
    new_size = [max(1, int(np.round(extent[i] / target_spacing[i]))) for i in range(3)]
    
    # Calculate new origin - this is the key fix!
    # The origin should be positioned so that when we apply the target direction matrix,
    # we cover the same physical extent as the original image
    
    # For LPS orientation, we want the origin at the corner that, when multiplied by
    # the direction matrix and voxel sizes, gives us the minimum physical coordinates
    # after considering the direction matrix orientation
    
    new_origin = []
    for axis in range(3):
        direction_col = target_direction_matrix[:, axis]  # Direction vector for this axis
        
        # If direction is negative, we want to start from the maximum physical coordinate
        # If direction is positive, we want to start from the minimum physical coordinate
        if np.sum(direction_col * target_direction_matrix[:, axis]) < 0:
            # Negative direction: start from max to go towards min
            new_origin.append(max_phys[np.argmax(np.abs(direction_col))])
        else:
            # Positive direction: start from min to go towards max  
            new_origin.append(min_phys[np.argmax(np.abs(direction_col))])
    
    return tuple(new_origin), new_size, (min_phys, max_phys)


def standardize_to_lps_orientation_and_spacing(
    sitk_image: sitk.Image, 
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    target_orientation: str = 'LPS'
) -> Tuple[sitk.Image, Dict]:
    """
    Standardize image to LPS orientation and specified spacing.
    
    Args:
        sitk_image (SimpleITK.Image): Input image
        target_spacing (tuple): Desired spacing (x, y, z) in mm
        target_orientation (str): Target orientation code (default: 'LPS')
        
    Returns:
        tuple: (standardized_image, transformation_info)
    """
    # Store original properties for logging
    original_spacing = sitk_image.GetSpacing()
    original_size = sitk_image.GetSize()
    original_origin = sitk_image.GetOrigin()
    original_direction = sitk_image.GetDirection()

    # Determine current orientation
    current_orientation = get_orientation_from_direction(original_direction)
    
    print(f"Original image properties:")
    print(f"  Spacing: {original_spacing}")
    print(f"  Size: {original_size}")
    print(f"  Origin: {original_origin}")
    print(f"  Direction: {original_direction}")
    # Determine target direction matrix
    if target_orientation == 'LPS':
        target_direction = create_lps_direction_matrix()
    else:
        raise ValueError(f"Orientation {target_orientation} not implemented yet")
    

    # Step 1: Try DICOMOrient first if only orientation change is needed
    oriented_image = None
    use_manual_resampling = True
    
    if current_orientation != target_orientation:
        print(f"Attempting reorientation from {current_orientation} to {target_orientation}...")
        try:
            oriented_image = sitk.DICOMOrient(sitk_image, target_orientation)
            intermediate_orientation = get_orientation_from_direction(oriented_image.GetDirection())
            print(f"After DICOMOrient - Orientation: {intermediate_orientation}")
            
            # Check if DICOMOrient worked and spacing is already correct
            if (intermediate_orientation == target_orientation and 
                np.allclose(oriented_image.GetSpacing(), target_spacing, rtol=1e-3)):
                print("DICOMOrient succeeded and spacing is correct - no further resampling needed")
                use_manual_resampling = False
                final_image = oriented_image
            elif intermediate_orientation == target_orientation:
                print("DICOMOrient succeeded but spacing needs adjustment")
                # Use oriented image as base for spacing adjustment
                base_image = oriented_image
            else:
                print("DICOMOrient failed to achieve target orientation, using manual resampling")
                base_image = sitk_image
                
        except Exception as e:
            print(f"DICOMOrient failed: {e}, using manual resampling...")
            base_image = sitk_image
    else:
        print("Image already in target orientation")
        base_image = sitk_image

    # Step 2: Manual resampling if needed
    if use_manual_resampling:
        print("Performing manual resampling...")
        
        # Use the base image (either original or oriented from DICOMOrient)
        new_origin, new_size, bbox = calculate_oriented_origin_and_size(
            base_image, target_direction, target_spacing
        )
        
        print(f"Calculated parameters:")
        print(f"  New origin: {new_origin}")
        print(f"  New size: {new_size}")
        print(f"  Physical bbox: {bbox}")
        print(f"  Target direction: {target_direction}")
        
        # Setup resampler
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(target_direction)
        resampler.SetOutputOrigin(new_origin)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        
        # Execute resampling
        print("Executing resampling...")
        final_image = resampler.Execute(base_image)
    

    # Verify final properties
    final_spacing = final_image.GetSpacing()
    final_size = final_image.GetSize()
    final_origin = final_image.GetOrigin()
    final_direction = final_image.GetDirection()
    final_orientation = get_orientation_from_direction(final_direction)
    
    print(f"Final image properties:")
    print(f"  Spacing: {final_spacing}")
    print(f"  Size: {final_size}")
    print(f"  Origin: {final_origin}")
    print(f"  Direction: {final_direction}")
    print(f"  Final orientation: {final_orientation}")

    # Verify results
    orientation_success = final_orientation == target_orientation
    spacing_success = np.allclose(final_spacing, target_spacing, rtol=1e-3)
    
    print(f"Orientation transformation successful: {orientation_success}")
    print(f"Spacing transformation successful: {spacing_success}")
    
    if not orientation_success:
        print(f"WARNING: Failed to achieve target orientation {target_orientation}, got {final_orientation}")
    if not spacing_success:
        print(f"WARNING: Failed to achieve target spacing {target_spacing}, got {final_spacing}")
    
    # Create transformation info for logging
    transformation_info = {
        'original_spacing': original_spacing,
        'original_size': original_size,
        'original_orientation': current_orientation,
        'final_spacing': final_spacing,
        'final_size': final_size,
        'final_orientation': final_orientation,
        'target_orientation': target_orientation,
        'orientation_success': orientation_success,
        'spacing_success': spacing_success,
        'spacing_changed': not np.allclose(original_spacing, final_spacing, rtol=1e-3),
        'size_changed': original_size != tuple(final_size),
        'used_manual_resampling': use_manual_resampling,
    }
    return final_image, transformation_info

    # # Step 2: Calculate physical extent in world coordinates
    # # Get corner points of the original image in physical space
    # original_corners = []
    # for i in range(2):
    #     for j in range(2):
    #         for k in range(2):
    #             index = (i * (original_size[0] - 1), 
    #                     j * (original_size[1] - 1), 
    #                     k * (original_size[2] - 1))
    #             point = sitk_image.TransformIndexToPhysicalPoint(index)
    #             original_corners.append(point)

    # # Find bounding box in world coordinates
    # min_coords = [min(corner[i] for corner in original_corners) for i in range(3)]
    # max_coords = [max(corner[i] for corner in original_corners) for i in range(3)]
    
    # print(f"Physical bounding box: min={min_coords}, max={max_coords}")
    
    # # Step 3: Calculate new image parameters
    # if use_manual_orientation or target_orientation == 'LPS':
    #     # Use LPS direction matrix
    #     target_direction = create_lps_direction_matrix()
        
    #     # Calculate new origin to maintain the same anatomical coverage
    #     # For LPS: origin should be at the left-posterior-inferior corner
    #     new_origin = (max_coords[0],  # Leftmost (max X in LPS)
    #                  max_coords[1],  # Posterior-most (max Y in LPS) 
    #                  min_coords[2])  # Inferior-most (min Z in LPS)
    # else:
    #     target_direction = oriented_image.GetDirection()
    #     new_origin = oriented_image.GetOrigin()


    # # Calculate new size based on physical extent and target spacing
    # physical_size = [max_coords[i] - min_coords[i] for i in range(3)]
    # new_size = [max(1, int(round(abs(physical_size[i]) / target_spacing[i]))) for i in range(3)]
    
    # print(f"Calculated new size: {new_size}")
    # print(f"Target direction: {target_direction}")
    # print(f"New origin: {new_origin}")
    
    # # Step 4: Setup resampler with explicit parameters
    # resampler = sitk.ResampleImageFilter()
    # resampler.SetOutputSpacing(target_spacing)
    # resampler.SetSize(new_size)
    # resampler.SetOutputDirection(target_direction)
    # resampler.SetOutputOrigin(new_origin)
    # resampler.SetInterpolator(sitk.sitkLinear)
    # resampler.SetDefaultPixelValue(0)
    
    # # Execute resampling
    # print("Executing resampling...")
    # resampled_image = resampler.Execute(sitk_image)  # Use original image
    
    # # Verify final properties
    # final_spacing = resampled_image.GetSpacing()
    # final_size = resampled_image.GetSize()
    # final_origin = resampled_image.GetOrigin()
    # final_direction = resampled_image.GetDirection()
    # final_orientation = get_orientation_from_direction(final_direction)
    
    # print(f"Final image properties:")
    # print(f"  Spacing: {final_spacing}")
    # print(f"  Size: {final_size}")
    # print(f"  Origin: {final_origin}")
    # print(f"  Direction: {final_direction}")
    # print(f"  Final orientation: {final_orientation}")
    
    # # Verify orientation is correct
    # orientation_success = final_orientation == target_orientation
    # print(f"Orientation transformation successful: {orientation_success}")
    
    # # Create transformation info for logging
    # transformation_info = {
    #     'original_spacing': original_spacing,
    #     'original_size': original_size,
    #     'original_orientation': current_orientation,
    #     'final_spacing': final_spacing,
    #     'final_size': final_size,
    #     'final_orientation': final_orientation,
    #     'target_orientation': target_orientation,
    #     'orientation_success': orientation_success,
    #     'spacing_changed': abs(sum(original_spacing) - sum(final_spacing)) > 0.001,
    #     'size_changed': original_size != final_size,
    #     'used_manual_orientation': use_manual_orientation,
    #     'physical_bounding_box': {'min': min_coords, 'max': max_coords}
    # }
    
    # return resampled_image, transformation_info

    # # Step 1: Reorient to target orientation (LPS)
    # print(f"Reorienting to {target_orientation}...")
    # oriented_image = sitk.DICOMOrient(sitk_image, target_orientation)
    # Step 2: Resample to target spacing
    # print(f"Resampling to spacing {target_spacing}...")
    
    # # Calculate new size based on target spacing
    # current_spacing = oriented_image.GetSpacing()
    # current_size = oriented_image.GetSize()
    
    # new_size = [
    #     int(round(curr_size * curr_spacing / target_spacing[i]))
    #     for i, (curr_size, curr_spacing) in enumerate(zip(current_size, current_spacing))
    # ]
    
    # # Setup resampler
    # resampler = sitk.ResampleImageFilter()
    # resampler.SetOutputSpacing(target_spacing)
    # resampler.SetSize(new_size)
    # resampler.SetOutputDirection(oriented_image.GetDirection())
    # resampler.SetOutputOrigin(oriented_image.GetOrigin())
    # resampler.SetInterpolator(sitk.sitkLinear)
    # resampler.SetDefaultPixelValue(0)
    
    # # Execute resampling
    # resampled_image = resampler.Execute(oriented_image)
    
    # # Verify final properties
    # final_spacing = resampled_image.GetSpacing()
    # final_size = resampled_image.GetSize()
    # final_origin = resampled_image.GetOrigin()
    # final_direction = resampled_image.GetDirection()
    
    # print(f"Final image properties:")
    # print(f"  Spacing: {final_spacing}")
    # print(f"  Size: {final_size}")
    # print(f"  Origin: {final_origin}")
    # print(f"  Direction: {final_direction}")
    
    # # Create transformation info for logging
    # transformation_info = {
    #     'original_spacing': original_spacing,
    #     'original_size': original_size,
    #     'final_spacing': final_spacing,
    #     'final_size': final_size,
    #     'target_orientation': target_orientation,
    #     'spacing_changed': abs(sum(original_spacing) - sum(final_spacing)) > 0.001,
    #     'size_changed': original_size != final_size
    # }
    
    # return resampled_image, transformation_info

def validate_image_properties(sitk_image: sitk.Image, expected_orientation: str = 'LPS', target_spacing: Optional[Tuple[float, float, float]] = None, tolerance: float = 1e-3) -> Dict:
    """
    Validate that the image has the expected properties.
    
    Args:
        sitk_image (SimpleITK.Image): Image to validate
        expected_orientation (str): Expected orientation
        
    Returns:
        dict: Validation results
    """
    spacing = sitk_image.GetSpacing()
    size = sitk_image.GetSize()
    direction = sitk_image.GetDirection()
    
    # Get actual orientation from direction matrix
    actual_orientation = get_orientation_from_direction(direction)
    
    # Check if orientation matches expected
    orientation_matches = actual_orientation == expected_orientation
    
    # Check if direction matrix corresponds to expected LPS matrix
    expected_lps_direction = create_lps_direction_matrix()
    is_lps_direction = np.allclose(direction, expected_lps_direction, atol=0.1)
    
    # Check spacing uniformity
    spacing_uniform = (abs(spacing[0] - spacing[1]) < 0.001 and 
                      abs(spacing[1] - spacing[2]) < 0.001 and
                      abs(spacing[0] - spacing[2]) < 0.001)
    
    validation_results = {
        'spacing': spacing,
        'size': size,
        'direction': direction,
        'actual_orientation': actual_orientation,
        'expected_orientation': expected_orientation,
        'orientation_matches': orientation_matches,
        'is_lps_direction_matrix': is_lps_direction,
        'spacing_uniform': spacing_uniform,
        'volume_ml': (size[0] * size[1] * size[2] * spacing[0] * spacing[1] * spacing[2]) / 1000,  # Convert mm³ to ml
        'direction_determinant': np.linalg.det(np.array(direction).reshape(3, 3))  # Should be 1 or -1
    }
    # Add spacing validation if target_spacing is provided
    if target_spacing is not None:
        spacing_correct = np.allclose(spacing, target_spacing, rtol=tolerance)
        validation_results.update({
            'spacing_correct': spacing_correct,
            'target_spacing': target_spacing,
            'spacing_difference': np.array(spacing) - np.array(target_spacing)
        })
        
    return validation_results


# def validate_image_properties(sitk_image: sitk.Image, expected_orientation: str = 'LPS') -> Dict:
#     """
#     Validate that the image has the expected properties.
    
#     Args:
#         sitk_image (SimpleITK.Image): Image to validate
#         expected_orientation (str): Expected orientation
        
#     Returns:
#         dict: Validation results
#     """
#     spacing = sitk_image.GetSpacing()
#     size = sitk_image.GetSize()
#     direction = sitk_image.GetDirection()
    
#     # Check if direction matrix corresponds to LPS
#     # LPS means: Left-to-right (L), Posterior-to-anterior (P), Superior-to-inferior (S)
#     # The direction matrix should be close to identity for LPS in SimpleITK
#     is_lps_oriented = np.allclose(direction, (1, 0, 0, 0, 1, 0, 0, 0, 1), atol=0.1)
    
#     validation_results = {
#         'spacing': spacing,
#         'size': size,
#         'direction': direction,
#         'is_standard_orientation': is_lps_oriented,
#         'spacing_uniform': abs(spacing[0] - spacing[1]) < 0.001 and abs(spacing[1] - spacing[2]) < 0.001,
#         'volume_ml': (size[0] * size[1] * size[2] * spacing[0] * spacing[1] * spacing[2]) / 1000  # Convert mm³ to ml
#     }
    
#     return validation_results




def save_dicom_paths(batch_dir: str, labels_csv: str, output_pkl: str) -> List[Dict]:
    """Select one optimal series per phase per study based on slice thickness."""
    labels_df = pd.read_csv(labels_csv)
    phase_lookup = {
        (row['StudyInstanceUID'], row['SeriesInstanceUID']): row['Label'].lower()
        for _, row in labels_df.iterrows()
    }

    case_count = 0
    # {(study_uid, phase): (best_thickness, series_info)}
    best_series = defaultdict(lambda: (float('inf'), None))
    failed_count = 0
    
    print(f"Processing batches in {batch_dir}...")
    
    for batch in os.listdir(batch_dir):
        batch_path = os.path.join(batch_dir, batch)
        if not os.path.isdir(batch_path):
            continue
            
        for study in os.listdir(batch_path):
            study_path = os.path.join(batch_path, study)
            if not os.path.isdir(study_path):
                continue
                
            case_count += 1
            
            for series in os.listdir(study_path):
                series_path = os.path.join(study_path, series)
                if not os.path.isdir(series_path):
                    continue
                    
                key = (study, series)
                
                if key not in phase_lookup:
                    continue
                
                phase = phase_lookup[key]
                thickness = get_slice_thickness(series_path)
                
                if thickness == float('inf'):
                    failed_count += 1
                    print("inf thickness")
                    continue
                
                # Select thinner slices (smaller thickness is better)
                current_best_thickness = best_series[(study, phase)][0]
                print("thickness , current_best_thickness", thickness, " ", current_best_thickness)
                if thickness < current_best_thickness:
                    best_series[(study, phase)] = (
                        thickness,
                        {
                            'study_uid': study,
                            'series_uid': series,
                            'series_path': series_path,
                            'phase': phase,
                            'slice_thickness': thickness
                        }
                    )

    # Extract results
    final_series_data = [info for (thickness, info) in best_series.values() if info is not None]
    
    # Save results
    with open(output_pkl, 'wb') as f:
        pickle.dump(final_series_data, f)
    
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Total cases processed: {case_count}")
    print(f"Series with infinite thickness: {failed_count}")
    print(f"Final series selected: {len(final_series_data)}")
    print(f"Success rate: {((case_count - failed_count) / case_count * 100):.1f}%")
    
    return final_series_data


def convert_dicom_to_nifti_dcm2niix(
    dicom_path: str,
    output_path: str,
    overwrite: bool = False
) -> None:
    """
    Convert DICOM series to NIfTI using dcm2niix.
    
    Args:
        dicom_path (str): Path to the DICOM directory
        output_path (str): Path where the NIfTI file should be saved
        overwrite (bool): Whether to overwrite existing files
    """
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Get base filename without any extensions
        base_filename = os.path.splitext(os.path.splitext(os.path.basename(output_path))[0])[0]
        
        # Prepare dcm2niix command - simplified for compatibility
        cmd = [
            "dcm2niix",
            "-z", "y",
            "-o", output_dir,  # output directory
            "-f", base_filename,  # output filename without extension
            dicom_path  # input DICOM folder
        ]
        
        # Debug: Print the full command
        print(f"Executing command: {' '.join(cmd)}")
        
        # Debug: Check if input directory exists and has DICOM files
        if not os.path.exists(dicom_path):
            raise Exception(f"Input directory does not exist: {dicom_path}")
        
        dicom_files = [f for f in os.listdir(dicom_path) if f.endswith('.dcm') or f.endswith('.dicom')]
        if not dicom_files:
            raise Exception(f"No DICOM files found in {dicom_path}")
        print(f"Found {len(dicom_files)} DICOM files in input directory")
        
        # Run dcm2niix
        result = subprocess.run(cmd, capture_output=True, text=True)
    
        if result.returncode != 0:
            raise Exception(f"dcm2niix failed: {result.stderr}")
        
        # Debug: Check if output file was created
        expected_output = os.path.join(output_dir, f"{base_filename}.nii.gz")
        if not os.path.exists(expected_output):
            raise Exception(f"Expected output file not created: {expected_output}")
            
        print(f"✓ Successfully converted {dicom_path} to {output_path}")
        
    except Exception as e:
        print(f"✗ Failed to convert {dicom_path}: {str(e)}")
        raise

import SimpleITK as sitk

# def reorient_and_resample(sitk_image, target_spacing=(1.0, 1.0, 1.0)):
#     """
#     Reorient the image to canonical orientation (RAS) and resample to target spacing.
    
#     Args:
#         sitk_image (SimpleITK.Image): Input image
#         target_spacing (tuple): Desired spacing (x, y, z)
        
#     Returns:
#         SimpleITK.Image: Reoriented and resampled image
#     """
#     # Step 1: Reorient to RAS
#     sitk_image = sitk.DICOMOrient(sitk_image, 'RAS')
    
#     # Step 2: Resample to target spacing
#     original_spacing = sitk_image.GetSpacing()
#     original_size = sitk_image.GetSize()
    
#     new_size = [
#         int(round(osz * ospc / tspc))
#         for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
#     ]
    
#     resampler = sitk.ResampleImageFilter()
#     resampler.SetOutputSpacing(target_spacing)
#     resampler.SetSize(new_size)
#     resampler.SetOutputDirection(sitk_image.GetDirection())
#     resampler.SetOutputOrigin(sitk_image.GetOrigin())
#     resampler.SetInterpolator(sitk.sitkLinear)
    
#     resampled_image = resampler.Execute(sitk_image)
    
#     return resampled_image

def process_series(
    series_data: List[Dict],
    nifti_root_dir: str,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    target_orientation: str = 'LPS',
    overwrite: bool = False,
    save_transformation_log: bool = True
) -> None:
    """
    Process DICOM series in either tuple or dictionary format.
    Tuple format: (study_uid, series_uid, sitk_image, metadata)
    Dict format: {'study_uid': ..., 'series_uid': ..., ...}
    
    Args:
        series_data (List): List of series data in either tuple or dict format
        nifti_root_dir (str): Root directory for NIfTI output
        overwrite (bool): Whether to overwrite existing files
        use_dcm2niix (bool): Whether to use dcm2niix for conversion (True) or SimpleITK (False)
    """
    transformation_log = []
    successful_conversions = 0
    failed_conversions = 0
    
    print(f"Processing {len(series_data)} series...")
    print(f"Target spacing: {target_spacing}")
    print(f"Target orientation: {target_orientation}")
    
    for i, series in enumerate(series_data):
        try:
            # Handle both tuple and dictionary formats
            if isinstance(series, tuple):
                study_uid, series_uid, sitk_image, metadata = series
                dicom_path = None  # Not available in tuples
            else:  # Dictionary format
                study_uid = series['study_uid']
                series_uid = series['series_uid']
                dicom_path = series.get('series_path')
                phase = series['phase']
            
            print(f"\n--- Processing {i+1}/{len(series_data)}: {study_uid}/{series_uid} ({phase}) ---")
            
            # Create output directory
            output_dir = os.path.join(nifti_root_dir, study_uid)
            os.makedirs(output_dir, exist_ok=True)
            
            # Define output path
            output_path = os.path.join(output_dir, f"{series_uid}.nii.gz")
            
            # Skip if exists and not overwriting
            if not overwrite and os.path.exists(output_path):
                print(f"✓ Skipping {output_path} (already exists)")
                continue
            
            if dicom_path:
                convert_dicom_to_nifti_dcm2niix(dicom_path, output_path, overwrite)
                # Load the NIfTI to enforce orientation/spacing
                sitk_image = sitk.ReadImage(output_path)
                # Step 3: Standardize orientation and spacing
                print("Applying standardization...")
                standardized_image, transform_info = standardize_to_lps_orientation_and_spacing(
                    sitk_image, target_spacing, target_orientation
                )
                # Step 4: Validate the result
                validation_results = validate_image_properties(standardized_image, target_orientation, target_spacing)
                print(f"Validation results: {validation_results}")

                # Step 5: Save the standardized image
                sitk.WriteImage(standardized_image, output_path)
                print(f"✓ Saved standardized image: {output_path}")
                
            else:
                print(f"✗ Skipping {series_uid}: no DICOM path provided.")
        
            # Log transformation details
            if save_transformation_log:
                log_entry = {
                    'study_uid': study_uid,
                    'series_uid': series_uid,
                    'phase': phase,
                    'dicom_path': dicom_path,
                    'output_path': output_path,
                    'transformation': transform_info,
                    'validation': validation_results,
                    'success': True
                }
                transformation_log.append(log_entry)
            
            successful_conversions += 1
            
        except Exception as e:
            print(f"✗ Failed to process {series_uid if 'series_uid' in locals() else 'unknown'}: {str(e)}")
            failed_conversions += 1
            
            # Log failure
            if save_transformation_log:
                log_entry = {
                    'study_uid': series.get('study_uid', 'unknown'),
                    'series_uid': series.get('series_uid', 'unknown'),
                    'phase': series.get('phase', 'unknown'),
                    'dicom_path': series.get('series_path', 'unknown'),
                    'error': str(e),
                    'success': False
                }
                transformation_log.append(log_entry)
    
    # Save transformation log
    if save_transformation_log:
        log_path = os.path.join(nifti_root_dir, 'transformation_log.json')
        with open(log_path, 'w') as f:
            json.dump(transformation_log, f, indent=2, default=str)
        print(f"\n✓ Saved transformation log: {log_path}")
    
    print(f"\n=== PROCESSING SUMMARY ===")
    print(f"Successful conversions: {successful_conversions}")
    print(f"Failed conversions: {failed_conversions}")
    print(f"Success rate: {(successful_conversions / len(series_data) * 100):.1f}%")


def process_original_volumes(
    batch_dir: str,
    labels_csv: str,
    nifti_root_dir: str,
    pkl_path: str = "dicom_paths.pkl",
    target_spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5),
    target_orientation: str = 'LPS',
    overwrite_nifti: bool = False
) -> None:
    """Orchestrate the entire pipeline with caching."""
    # Check if NIfTI output directory is empty
    if not os.path.exists(nifti_root_dir) or not os.listdir(nifti_root_dir) or overwrite_nifti:
        print("NIfTI directory empty/overwrite requested. Processing DICOMs...")
        
        # Load or generate DICOM paths cache
        if os.path.exists(pkl_path) and not overwrite_nifti:
            with open(pkl_path, 'rb') as f:
                series_data = pickle.load(f)
            print(f"Loaded {len(series_data)} series from cache")
        else:
            print("Starting to load DICOM series with metadata...")
            series_data = save_dicom_paths(batch_dir, labels_csv, pkl_path)
            # series_data = load_and_cache_dicom_series(batch_dir, labels_csv, pkl_path)
        print("converting")
        # Convert to NIfTI
        process_series(
            series_data, 
            nifti_root_dir, 
            target_spacing=target_spacing,
            target_orientation=target_orientation,
            overwrite=overwrite_nifti
        )
    else:
        print(f"NIfTI files already exist in {nifti_root_dir}. Skipping conversion.")

