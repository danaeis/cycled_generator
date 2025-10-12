"""
Unified configuration file for both vindr_ds and pars-ct datasets.
"""

import os
from pathlib import Path

# Base configuration
BASE_DATA_DIR = '../data'

# Dataset-specific configurations
DATASET_CONFIGS = {
    'vindr_ds': {
        'main_path': '../ncct_cect/vindr_ds/',
        'batch_dir': '../ncct_cect/vindr_ds/main_batches',
        'labels_csv': '../ncct_cect/vindr_ds/labels.csv',
        'cache_path': '../ncct_cect/vindr_ds/cached_vindr_dicom.pkl',
        'original_dir': '../ncct_cect/vindr_ds/nifti_unprocessed_volumes',
        'segmentation_dir': '../ncct_cect/vindr_ds/ts_segmentations',
        'cropped_dir': '../ncct_cect/vindr_ds/cropped_volumes',
        'registered_dir': '../ncct_cect/vindr_ds/registered_cases',
        'structure': {
            'has_nifti_volumes': True,
            'has_segmentations': False,  # Needs TotalSegmentator
            'has_cropped': False,
            'has_registered': False
        }
    },
    'pars-ct': {
        'main_path': '../pars-ct/',
        'phase_labels_csv': '../pars-ct/phase_labels.csv',
        'structure': {
            'has_nifti_volumes': False,  # Needs conversion from DICOM
            'has_segmentations': True,   # Has manual segmentations in ASSESSORS
            'has_cropped': False,
            'has_registered': False
        }
    }
}

# Processing options
PROCESSING_OPTIONS = {
    'target_spacing': (1.5, 1.5, 1.5),
    'target_orientation': 'LPS',
    'crop_margin': 10,
    'registration_params': {
        'learning_rate': 1.0,
        'iterations': 100,
        'convergence_minimum_value': 1e-6,
        'convergence_window_size': 10,
        'sampling_percentage': 0.2,
        'histogram_bins': 50
    }
}

# Valid phase labels for pars-ct
VALID_PHASE_LABELS = ['w/o', 'art', 'por', 'del']

# Phase label mapping
PHASE_LABEL_MAPPING = {
    'w/o': 'Non-contrast',
    'art': 'Arterial',
    'por': 'Portal-venous',
    'del': 'Delayed'
}

def get_dataset_config(dataset_name: str) -> dict:
    """
    Get configuration for a specific dataset.
    
    Args:
        dataset_name: Name of the dataset ('vindr_ds' or 'pars-ct')
        
    Returns:
        Dictionary containing dataset configuration
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")
    
    return DATASET_CONFIGS[dataset_name]

def validate_dataset_paths(dataset_name: str, data_path: str) -> bool:
    """
    Validate that the dataset paths exist and have the expected structure.
    
    Args:
        dataset_name: Name of the dataset
        data_path: Path to the dataset root directory
        
    Returns:
        True if valid, False otherwise
    """
    if not os.path.exists(data_path):
        return False
    
    config = get_dataset_config(dataset_name)
    
    if dataset_name == 'vindr_ds':
        # Check for nifti_unprocessed_volumes directory
        nifti_dir = os.path.join(data_path, 'nifti_unprocessed_volumes')
        return os.path.exists(nifti_dir)
    
    elif dataset_name == 'pars-ct':
        # Check for case directories with SCANS and ASSESSORS
        found_cases = 0
        for item in os.listdir(data_path):
            item_path = os.path.join(data_path, item)
            if os.path.isdir(item_path):
                if os.path.exists(os.path.join(item_path, 'SCANS')) and os.path.exists(os.path.join(item_path, 'ASSESSORS')):
                    found_cases += 1
        
        return found_cases > 0
    
    return False

def setup_output_directories(output_base_dir: str, dataset_name: str) -> dict:
    """
    Setup output directories for processing.
    
    Args:
        output_base_dir: Base output directory
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary with output directory paths
    """
    output_dirs = {
        'base': output_base_dir,
        'nifti_volumes': os.path.join(output_base_dir, 'nifti_volumes'),
        'segmentations': os.path.join(output_base_dir, 'segmentations'),
        'cropped_volumes': os.path.join(output_base_dir, 'cropped_volumes'),
        'registered_volumes': os.path.join(output_base_dir, 'registered_volumes')
    }
    
    if dataset_name == 'pars-ct':
        output_dirs['pars_ct_processed'] = os.path.join(output_base_dir, 'pars_ct_processed')
    
    # Create directories
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return output_dirs

def get_processing_options() -> dict:
    """
    Get processing options.
    
    Returns:
        Dictionary containing processing options
    """
    return PROCESSING_OPTIONS

def get_valid_phase_labels() -> list:
    """
    Get list of valid phase labels for pars-ct.
    
    Returns:
        List of valid phase labels
    """
    return VALID_PHASE_LABELS

def get_phase_label_mapping() -> dict:
    """
    Get phase label mapping.
    
    Returns:
        Dictionary mapping phase labels to descriptions
    """
    return PHASE_LABEL_MAPPING

