#!/bin/bash
set -e

# Unified Pipeline for Medical Imaging Data Processing
# Supports both vindr_ds and pars-ct datasets
# Usage: ./unified_pipeline.sh <dataset_name> <data_path> [options]

# Default configuration
DATASET_NAME="pars-ct"
DATA_PATH=""
OUTPUT_BASE_DIR=""
PHASE_LABELS_CSV=""
OVERWRITE_NIFTI=false
OVERWRITE_SEGMENTATION=false
OVERWRITE_CROPPING=false
OVERWRITE_REGISTRATION=false
SKIP_SEGMENTATION=false
SKIP_CROPPING=false
SKIP_REGISTRATION=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 <dataset_name> <data_path> [options]"
    echo ""
    echo "Arguments:"
    echo "  dataset_name    Dataset type: 'vindr_ds' or 'pars-ct'"
    echo "  data_path       Path to the dataset root directory"
    echo ""
    echo "Options:"
    echo "  --output-dir DIR                Output base directory (default: data_path/processed)"
    echo "  --phase-labels-csv FILE         Path to phase labels CSV for pars-ct (required for pars-ct)"
    echo "  --overwrite-nifti              Force reprocess NIfTI conversion"
    echo "  --overwrite-segmentation       Force reprocess segmentation"
    echo "  --overwrite-cropping           Force reprocess cropping"
    echo "  --overwrite-registration       Force reprocess registration"
    echo "  --skip-segmentation            Skip segmentation step"
    echo "  --skip-cropping                Skip cropping step"
    echo "  --skip-registration            Skip registration step"
    echo "  --help                         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 vindr_ds /path/to/vindr_ds"
    echo "  $0 pars-ct /path/to/pars-ct-batch1 --phase-labels-csv /path/to/phase_labels.csv"
    echo "  $0 vindr_ds /path/to/vindr_ds --skip-segmentation --overwrite-cropping"
}

# Function to validate dataset structure
validate_dataset_structure() {
    local dataset_name="$1"
    local data_path="$2"
    
    print_status "Validating dataset structure for $dataset_name..."
    
    if [ ! -d "$data_path" ]; then
        print_error "Data path does not exist: $data_path"
        exit 1
    fi
    
    case "$dataset_name" in
        "vindr_ds")
            # Validate vindr_ds structure
            if [ ! -d "$data_path/nifti_unprocessed_volumes" ]; then
                print_error "vindr_ds structure not found. Expected nifti_unprocessed_volumes directory."
                exit 1
            fi
            print_success "vindr_ds structure validated"
            ;;
        "pars-ct")
            # Validate pars-ct structure
            local found_cases=0
            for case_dir in "$data_path"/*; do
                if [ -d "$case_dir" ] && [ -d "$case_dir/SCANS" ] && [ -d "$case_dir/ASSESSORS" ]; then
                    found_cases=$((found_cases + 1))
                fi
            done
            
            if [ $found_cases -eq 0 ]; then
                print_error "pars-ct structure not found. Expected case directories with SCANS and ASSESSORS folders."
                exit 1
            fi
            
            print_success "pars-ct structure validated ($found_cases cases found)"
            ;;
        *)
            print_error "Unsupported dataset name: $dataset_name. Use 'vindr_ds' or 'pars-ct'"
            exit 1
            ;;
    esac
}

# Function to setup output directories
setup_output_dirs() {
    local dataset_name="$1"
    local data_path="$2"
    local output_dir="$3"
    
    print_status "Setting up output directories..."
    
    mkdir -p "$output_dir"
    
    case "$dataset_name" in
        "vindr_ds")
            mkdir -p "$output_dir/nifti_volumes"
            mkdir -p "$output_dir/segmentations"
            mkdir -p "$output_dir/cropped_volumes"
            mkdir -p "$output_dir/registered_volumes"
            ;;
        "pars-ct")
            mkdir -p "$output_dir/nifti_volumes"
            mkdir -p "$output_dir/segmentations"
            mkdir -p "$output_dir/cropped_volumes"
            mkdir -p "$output_dir/registered_volumes"
            mkdir -p "$output_dir/pars_ct_processed"
            ;;
    esac
    
    print_success "Output directories created"
}

# Function to process vindr_ds dataset
process_vindr_ds() {
    local data_path="$1"
    local output_dir="$2"
    
    print_status "Processing vindr_ds dataset..."
    
    # Step 1: DICOM to NIfTI conversion (if needed)
    if [ ! -d "$data_path/nifti_unprocessed_volumes" ] || [ "$OVERWRITE_NIFTI" = true ]; then
        print_status "Converting DICOM to NIfTI..."
        python3 data/pre_dcm2nifti.py
        print_success "DICOM to NIfTI conversion completed"
    else
        print_status "NIfTI volumes already exist, skipping conversion"
    fi
    
    # Step 2: Segmentation
    if [ "$SKIP_SEGMENTATION" = false ]; then
        if [ ! -d "$data_path/ts_segmentations" ] || [ "$OVERWRITE_SEGMENTATION" = true ]; then
            print_status "Running segmentation..."
            bash data/run_segmentation.sh
            print_success "Segmentation completed"
        else
            print_status "Segmentations already exist, skipping"
        fi
    fi
    
    # Step 3: Cropping
    if [ "$SKIP_CROPPING" = false ]; then
        if [ ! -d "$data_path/cropped_volumes" ] || [ "$OVERWRITE_CROPPING" = true ]; then
            print_status "Running cropping pipeline..."
            bash data/crop_pipeline.sh
            print_success "Cropping completed"
        else
            print_status "Cropped volumes already exist, skipping"
        fi
    fi
    
    # Step 4: Registration
    if [ "$SKIP_REGISTRATION" = false ]; then
        if [ ! -d "$data_path/registered_cases" ] || [ "$OVERWRITE_REGISTRATION" = true ]; then
            print_status "Running registration pipeline..."
            python3 data/register.py
            print_success "Registration completed"
        else
            print_status "Registered volumes already exist, skipping"
        fi
    fi
}

# Function to process pars-ct dataset
process_pars_ct() {
    local data_path="$1"
    local output_dir="$2"
    local phase_labels_csv="$3"
    
    print_status "Processing pars-ct dataset..."
    
    # Validate phase labels CSV
    if [ -z "$phase_labels_csv" ] || [ ! -f "$phase_labels_csv" ]; then
        print_error "Phase labels CSV is required for pars-ct dataset"
        exit 1
    fi
    
    # Process each case
    for case_dir in "$data_path"/*; do
        if [ -d "$case_dir" ] && [ -d "$case_dir/SCANS" ] && [ -d "$case_dir/ASSESSORS" ]; then
            local case_name=$(basename "$case_dir")
            print_status "Processing case: $case_name"
            
            # Create case output directory
            local case_output_dir="$output_dir/pars_ct_processed/$case_name"
            mkdir -p "$case_output_dir"
            
            # Step 1: Parse folder structure and extract metadata
            print_status "Parsing folder structure for $case_name..."
            python3 data/Pars-CT_processor/Code/Step_2_1_ParseFolder.py "$case_dir" "$case_output_dir"
            
            # Step 2: Select segmentations (use all for now)
            print_status "Selecting segmentations for $case_name..."
            python3 data/Pars-CT_processor/Code/Step_2_2_SelectSegmentations.py "$case_output_dir/Segmentations_info.json" --manual <<< "all"
            
            # Step 3: Decode segmentations
            print_status "Decoding segmentations for $case_name..."
            python3 data/Pars-CT_processor/Code/Step_2_3_DecodeSegmentation.py "$case_output_dir/SelectedSegmentations_info.json"
            
            # Step 4: Generate NIfTI files
            print_status "Generating NIfTI files for $case_name..."
            python3 data/Pars-CT_processor/Code/Step_2_4_NiftiGeneration.py "$case_output_dir/PreparedSegmentations_info.json" "$case_output_dir/StudySeries_info.json"
            
            print_success "Case $case_name processed"
        fi
    done
    
    # Now process phase-labeled series using the CSV
    print_status "Processing phase-labeled series..."
    python3 data/process_pars_ct_phases.py "$data_path" "$output_dir" "$phase_labels_csv"
    
    # Apply segmentation to other series (if needed)
    if [ "$SKIP_SEGMENTATION" = false ]; then
        print_status "Applying segmentation to other series..."
        python3 data/apply_segmentation_to_other_series.py "$output_dir" "$phase_labels_csv"
    fi
    
    # Apply cropping pipeline
    if [ "$SKIP_CROPPING" = false ]; then
        print_status "Applying cropping to pars-ct volumes..."
        python3 data/crop_pars_ct_volumes.py "$output_dir" "$phase_labels_csv"
    fi
    
    # Apply registration pipeline
    if [ "$SKIP_REGISTRATION" = false ]; then
        print_status "Applying registration to pars-ct volumes..."
        python3 data/register_pars_ct_volumes.py "$output_dir" "$phase_labels_csv"
    fi
    
    print_success "pars-ct dataset processing completed"
}

# Parse command line arguments
if [ $# -lt 2 ]; then
    show_usage
    exit 1
fi

DATASET_NAME="$1"
DATA_PATH="$2"
shift 2

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_BASE_DIR="$2"
            shift 2
            ;;
        --phase-labels-csv)
            PHASE_LABELS_CSV="$2"
            shift 2
            ;;
        --overwrite-nifti)
            OVERWRITE_NIFTI=true
            shift
            ;;
        --overwrite-segmentation)
            OVERWRITE_SEGMENTATION=true
            shift
            ;;
        --overwrite-cropping)
            OVERWRITE_CROPPING=true
            shift
            ;;
        --overwrite-registration)
            OVERWRITE_REGISTRATION=true
            shift
            ;;
        --skip-segmentation)
            SKIP_SEGMENTATION=true
            shift
            ;;
        --skip-cropping)
            SKIP_CROPPING=true
            shift
            ;;
        --skip-registration)
            SKIP_REGISTRATION=true
            shift
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Set default output directory if not provided
if [ -z "$OUTPUT_BASE_DIR" ]; then
    OUTPUT_BASE_DIR="$DATA_PATH/processed"
fi

# Main execution
print_status "Starting unified pipeline for $DATASET_NAME dataset"
print_status "Data path: $DATA_PATH"
print_status "Output directory: $OUTPUT_BASE_DIR"

# Validate dataset structure
validate_dataset_structure "$DATASET_NAME" "$DATA_PATH"

# Setup output directories
setup_output_dirs "$DATASET_NAME" "$DATA_PATH" "$OUTPUT_BASE_DIR"

# Process based on dataset type
case "$DATASET_NAME" in
    "vindr_ds")
        process_vindr_ds "$DATA_PATH" "$OUTPUT_BASE_DIR"
        ;;
    "pars-ct")
        process_pars_ct "$DATA_PATH" "$OUTPUT_BASE_DIR" "$PHASE_LABELS_CSV"
        ;;
esac

print_success "Pipeline completed successfully!"
print_status "Results saved to: $OUTPUT_BASE_DIR"
