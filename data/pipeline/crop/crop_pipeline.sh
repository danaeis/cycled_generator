#!/bin/bash
set -e

# Configuration
MAIN_PATH="../ncct_cect/vindr_ds/"
INPUT_DIR="${MAIN_PATH}nifti_unprocessed_volumes"
SEG_DIR="${MAIN_PATH}ts_segmentations"
OUTPUT_DIR="${MAIN_PATH}cropped_volumes"
LABELS_CSV="${MAIN_PATH}labels.csv"
MARGIN=10

echo "================================================================================"
echo "CROP PIPELINE - Organize by Study"
echo "================================================================================"
echo "Input volumes:     $INPUT_DIR"
echo "Segmentations:     $SEG_DIR"
echo "Output directory:  $OUTPUT_DIR"
echo "Labels CSV:        $LABELS_CSV"
echo "Margin:            $MARGIN voxels"
echo "================================================================================"
echo ""

# Check directories exist
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ Input directory not found: $INPUT_DIR"
    exit 1
fi

if [ ! -d "$SEG_DIR" ]; then
    echo "❌ Segmentation directory not found: $SEG_DIR"
    exit 1
fi

if [ ! -f "$LABELS_CSV" ]; then
    echo "❌ Labels CSV not found: $LABELS_CSV"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Get unique study IDs from labels.csv
study_ids=$(python3 -c "
import pandas as pd
df = pd.read_csv('$LABELS_CSV')
for study_id in df['StudyInstanceUID'].unique():
    print(study_id)
")

total_studies=$(echo "$study_ids" | wc -l)
current_study=0

# Process each study
for study_id in $study_ids; do
    ((current_study++))
    
    # echo "[$current_study/$total_studies] Processing study: ${study_id:0:55}..."
    
    # Create study output directory
    study_output_dir="${OUTPUT_DIR}/${study_id}"
    
    
    # Get all series for this study
    series_data=$(python3 -c "
import pandas as pd
df = pd.read_csv('$LABELS_CSV')
study_df = df[df['StudyInstanceUID'] == '$study_id']
for _, row in study_df.iterrows():
    print(f\"{row['SeriesInstanceUID']}|{row['Label']}\")
")
    
    study_successful=0
    study_skipped=0
    study_failed=0
    
    # Process each series
    while IFS='|' read -r series_id phase; do
        # Build file paths
        vol_filename="${study_id}_${series_id}_standardized.nii.gz"
        seg_filename="${study_id}_${series_id}_seg.nii.gz"
        
        vol_path="${INPUT_DIR}/${vol_filename}"
        seg_path="${SEG_DIR}/${seg_filename}"
        
        output_vol="${study_output_dir}/${study_id}_${series_id}_crop.nii.gz"
        output_seg="${study_output_dir}/${study_id}_${series_id}_seg.nii.gz"
        
        # Check if volume exists
        if [ ! -f "$vol_path" ]; then
            # echo "  ⚠️  Volume not found: $vol_path"
            ((study_failed++))
            continue
        fi
        
        # Check if segmentation exists
        if [ ! -f "$seg_path" ]; then
            # echo "  ⚠️  Segmentation not found: $seg_filename"
            ((study_failed++))
            continue
        fi
        
        # Skip if already processed
        if [ -f "$output_vol" ] && [ -f "$output_seg" ]; then
            echo "  ⏭️  $phase - already processed, skipping"
            ((study_skipped++))
            continue
        fi
        
        mkdir -p "$study_output_dir"
        # Crop the volume
        echo "  🔄 $phase - cropping..."
        if python3 ./data/pipeline/crop/crop_with_mask.py "$vol_path" "$seg_path" "$study_output_dir" "${study_id}_${series_id}" "$MARGIN" 2>&1 | grep -q "✅"; then
            echo "      ✅ Success"
            ((study_successful++))
        else
            echo "      ❌ Failed"
            ((study_failed++))
        fi
        
    done <<< "$series_data"
    
    # Study summary
    total_series=$((study_successful + study_skipped + study_failed))
    echo "  Study summary: $study_successful success, $study_skipped skipped, $study_failed failed (of $total_series series)"
    echo ""
done

# Final summary
echo "================================================================================"
echo "BATCH PROCESSING COMPLETE"
echo "================================================================================"

# Count outputs
num_studies=$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
num_crops=$(find "$OUTPUT_DIR" -name "*_crop.nii.gz" | wc -l)
num_segs=$(find "$OUTPUT_DIR" -name "*_seg.nii.gz" | wc -l)

echo "Studies processed: $total_studies"
echo ""
echo "Final output:"
echo "  Studies with output: $num_studies"
echo "  Cropped volumes:     $num_crops"
echo "  Cropped masks:       $num_segs"
echo "================================================================================"
