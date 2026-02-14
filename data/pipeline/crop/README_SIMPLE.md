# Crop Pipeline - Simple Update

## What Changed

**Only the output directory structure** - all the cropping logic stays the same.

### Before:
```
cropped_volumes/
├── study1_series1_crop.nii.gz
├── study1_series1_seg.nii.gz
├── study1_series2_crop.nii.gz
├── study1_series2_seg.nii.gz
└── ...
```

### After:
```
cropped_volumes/
├── study1/
│   ├── study1_series1_crop.nii.gz
│   ├── study1_series1_seg.nii.gz
│   ├── study1_series2_crop.nii.gz
│   └── study1_series2_seg.nii.gz
├── study2/
│   └── ...
```

## Usage

### Python (Recommended):
```bash
python3 crop_pipeline_simple.py
```

Or with force recompute:
```bash
python3 crop_pipeline_simple.py --force
```

### Bash:
```bash
chmod +x crop_pipeline.sh
./crop_pipeline.sh
```

## Input Files

The pipeline expects:
- **Volumes**: `nifti_unprocessed_volumes/{study_id}_{series_id}.nii.gz`
- **Segmentations**: `ts_segmentations/{study_id}_{series_id}_seg.nii.gz`
- **Labels**: `labels.csv` with StudyInstanceUID, SeriesInstanceUID, and Label columns

## Output Files

Creates:
- `cropped_volumes/{study_id}/{study_id}_{series_id}_crop.nii.gz`
- `cropped_volumes/{study_id}/{study_id}_{series_id}_seg.nii.gz`

## Changes to crop_with_mask.py

Only one fix: properly updates the affine matrix after cropping so the origin is correct.

## Integration with Your Code

Your existing code works with this structure:

```python
def find_volume_file(study_id: str, series_id: str, input_dir: Path, suffix: str = "_crop.nii.gz"):
    """
    Find a cropped volume in the organized directory structure.
    Searches in: {input_dir}/{study_id}/{study_id}_{series_id}{suffix}
    """
    pattern = f"{study_id}/{study_id}_{series_id}{suffix}"
    file_path = input_dir / pattern
    
    if file_path.exists():
        return file_path
    
    return None
```

Or use glob:
```python
def find_volume_file(study_id: str, series_id: str, input_dir: Path, suffix: str = "_crop.nii.gz"):
    """Glob version for flexibility."""
    pattern = f"{study_id}/*{series_id}{suffix}"
    matches = list(input_dir.glob(pattern))
    
    if len(matches) == 0:
        return None
    elif len(matches) > 1:
        print(f"Multiple matches found for {pattern} — taking first")
    
    return matches[0]
```

## That's It!

No complex parsing, no changes to cropping logic - just organized output directories.
