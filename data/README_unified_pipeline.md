# Unified Medical Imaging Pipeline

This unified pipeline processes medical imaging datasets with support for both `vindr_ds` and `pars-ct` dataset formats. It handles the complete workflow from DICOM files to registered volumes with segmentation masks.

## Features

- **Multi-dataset support**: Handles both `vindr_ds` and `pars-ct` datasets
- **Complete workflow**: DICOM to NIfTI conversion, segmentation, cropping, and registration
- **Flexible processing**: Skip or overwrite individual steps as needed
- **Phase-aware processing**: For pars-ct, processes series based on contrast phases
- **Robust error handling**: Continues processing even if individual cases fail

## Dataset Structures

### vindr_ds Dataset
```
vindr_ds/
├── nifti_unprocessed_volumes/     # NIfTI volumes (created by pre_dcm2nifti.py)
├── ts_segmentations/              # TotalSegmentator results
├── cropped_volumes/               # Cropped volumes
└── registered_cases/              # Registered volumes
```

### pars-ct Dataset
```
pars-ct-batch1/
├── case_id_1/
│   ├── SCANS/
│   │   ├── series_name/
│   │   │   └── DICOM/
│   │   │       └── *.dcm
│   │   └── ...
│   └── ASSESSORS/
│       ├── SEG_*/                 # Segmentation folders
│       │   └── SEG/
│       │       ├── *.dcm          # Segmentation DICOM
│       │       └── *.xml          # Metadata
│       └── ...
└── case_id_2/
    └── ...
```

## Installation

### Prerequisites
- Python 3.7+
- Required Python packages:
  ```bash
  pip install pydicom nibabel pandas numpy SimpleITK openpyxl
  ```
- TotalSegmentator (for segmentation):
  ```bash
  pip install TotalSegmentator
  ```

### Setup
1. Make the pipeline script executable:
   ```bash
   chmod +x unified_pipeline.sh
   ```

## Usage

### Basic Usage

#### For vindr_ds dataset:
```bash
./unified_pipeline.sh vindr_ds /path/to/vindr_ds
```

#### For pars-ct dataset:
```bash
./unified_pipeline.sh pars-ct /path/to/pars-ct-batch1 --phase-labels-csv /path/to/phase_labels.csv
```

### Advanced Usage

#### With custom output directory:
```bash
./unified_pipeline.sh vindr_ds /path/to/vindr_ds --output-dir /path/to/output
```

#### Skip specific steps:
```bash
./unified_pipeline.sh vindr_ds /path/to/vindr_ds --skip-segmentation --skip-cropping
```

#### Force reprocessing:
```bash
./unified_pipeline.sh vindr_ds /path/to/vindr_ds --overwrite-nifti --overwrite-segmentation
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--output-dir DIR` | Output base directory (default: data_path/processed) |
| `--phase-labels-csv FILE` | Path to phase labels CSV for pars-ct (required for pars-ct) |
| `--overwrite-nifti` | Force reprocess NIfTI conversion |
| `--overwrite-segmentation` | Force reprocess segmentation |
| `--overwrite-cropping` | Force reprocess cropping |
| `--overwrite-registration` | Force reprocess registration |
| `--skip-segmentation` | Skip segmentation step |
| `--skip-cropping` | Skip cropping step |
| `--skip-registration` | Skip registration step |
| `--help` | Show help message |

## Phase Labels for pars-ct

The pars-ct dataset requires a phase labels CSV file with the following columns:
- `Case Number`: Case identifier
- `Series Number`: Series identifier
- `Phase Label`: Contrast phase (`w/o`, `art`, `por`, `del`)
- `timing`: Timing information
- `Imaging protocole`: Protocol information
- `comments`: Additional comments

### Valid Phase Labels:
- `w/o`: Non-contrast (baseline)
- `art`: Arterial phase
- `por`: Portal-venous phase
- `del`: Delayed phase

## Processing Steps

### 1. DICOM to NIfTI Conversion
- **vindr_ds**: Uses existing `pre_dcm2nifti.py` script
- **pars-ct**: Converts DICOM series to NIfTI format with proper orientation

### 2. Segmentation
- **vindr_ds**: Applies TotalSegmentator to all volumes
- **pars-ct**: Uses existing manual segmentations from ASSESSORS, applies TotalSegmentator to other series

### 3. Cropping
- Crops volumes to abdomen region using segmentation masks
- Adds configurable margin around the region of interest

### 4. Registration
- Registers all series to the non-contrast volume
- Uses mutual information-based registration
- Applies same transform to segmentation masks

## Output Structure

After processing, the output directory will contain:

```
processed/
├── nifti_volumes/                 # Converted NIfTI files
├── segmentations/                 # Segmentation masks
├── cropped_volumes/               # Cropped volumes
├── registered_volumes/            # Registered volumes
└── pars_ct_processed/             # (pars-ct only) Intermediate processing files
    └── case_id/
        ├── StudySeries_info.json
        ├── Segmentations_info.json
        ├── segmentations_pickles/
        └── NIFTI/
```

## File Naming Conventions

### vindr_ds:
- Volumes: `study_name_series_name.nii.gz`
- Segmentations: `study_name_series_name_seg.nii.gz`
- Cropped: `study_name_series_name_crop.nii.gz`
- Registered: `study_name_series_name_registered.nii.gz`

### pars-ct:
- Volumes: `case_number_series_number_phase_label.nii.gz`
- Segmentations: `case_number_series_number_phase_label_seg.nii.gz`
- Cropped: `case_number_series_number_phase_label_crop.nii.gz`
- Registered: `case_number_series_number_phase_label_registered.nii.gz`

## Error Handling

The pipeline includes robust error handling:
- Continues processing if individual cases fail
- Provides detailed error messages
- Logs processing status for each step
- Validates dataset structure before processing

## Troubleshooting

### Common Issues:

1. **Missing phase labels CSV for pars-ct**:
   ```
   Error: Phase labels CSV is required for pars-ct dataset
   ```
   Solution: Provide the `--phase-labels-csv` parameter

2. **Invalid dataset structure**:
   ```
   Error: Dataset structure not found
   ```
   Solution: Ensure the dataset follows the expected directory structure

3. **TotalSegmentator not found**:
   ```
   Error: TotalSegmentator command not found
   ```
   Solution: Install TotalSegmentator: `pip install TotalSegmentator`

4. **Permission denied**:
   ```
   Error: Permission denied
   ```
   Solution: Make sure the script is executable: `chmod +x unified_pipeline.sh`

## Performance Tips

1. **Parallel processing**: The pipeline processes cases sequentially. For large datasets, consider running multiple instances on different subsets.

2. **Storage requirements**: Ensure sufficient disk space for intermediate files, especially for large datasets.

3. **Memory usage**: Registration can be memory-intensive. Monitor system resources during processing.

## Contributing

To extend the pipeline for new dataset types:

1. Add dataset configuration to `unified_configs.py`
2. Implement dataset-specific processing functions
3. Update the main pipeline script to handle the new dataset type
4. Add appropriate validation and error handling

## License

This pipeline is part of the cycled_generator project. Please refer to the project license for usage terms.

