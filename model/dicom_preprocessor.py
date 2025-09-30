import os
import pandas as pd
import numpy as np
import pydicom
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path
import json
import subprocess
import warnings
from tqdm import tqdm
from collections import defaultdict
import logging
from datetime import datetime
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DICOMProcessor:
    """Handles DICOM file processing and metadata extraction"""
    
    def __init__(self):
        self.phase_keywords = {
            'noncontrast': ['noncontrast', 'non-contrast', 'pre', 'baseline', 'native'],
            'arterial': ['arterial', 'art', 'early', 'phase1', 'p1'],
            'portal': ['portal', 'venous', 'pv', 'phase2', 'p2', 'late'],
            'delayed': ['delayed', 'delay', 'equilibrium', 'phase3', 'p3']
        }
    
    def read_dicom_series(self, series_path):
        """Read DICOM series and extract metadata"""
        try:
            series_path = Path(series_path)
            dicom_files = list(series_path.glob("*.dcm"))
            
            if not dicom_files:
                # Try other extensions
                dicom_files = list(series_path.glob("*"))
                dicom_files = [f for f in dicom_files if f.is_file()]
            
            if not dicom_files:
                logger.warning(f"No DICOM files found in {series_path}")
                return None, None
            
            # Read first DICOM to get metadata
            try:
                first_dicom = pydicom.dcmread(dicom_files[0], force=True)
            except Exception as e:
                logger.error(f"Failed to read DICOM {dicom_files[0]}: {e}")
                return None, None
            
            # Extract metadata
            metadata = {
                'series_uid': getattr(first_dicom, 'SeriesInstanceUID', 'unknown'),
                'series_description': getattr(first_dicom, 'SeriesDescription', 'unknown').lower(),
                'protocol_name': getattr(first_dicom, 'ProtocolName', 'unknown').lower(),
                'slice_thickness': float(getattr(first_dicom, 'SliceThickness', 0)),
                'pixel_spacing': getattr(first_dicom, 'PixelSpacing', [1.0, 1.0]),
                'num_slices': len(dicom_files),
                'acquisition_time': getattr(first_dicom, 'AcquisitionTime', 'unknown'),
                'manufacturer': getattr(first_dicom, 'Manufacturer', 'unknown'),
                'model': getattr(first_dicom, 'ManufacturerModelName', 'unknown')
            }
            
            # Use SimpleITK to read the series properly
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(str(series_path))
            
            if not dicom_names:
                logger.warning(f"No valid DICOM series found in {series_path}")
                return None, metadata
            
            reader.SetFileNames(dicom_names)
            reader.MetaDataDictionaryArrayUpdateOn()
            reader.LoadPrivateTagsOn()
            
            try:
                image = reader.Execute()
                return image, metadata
            except Exception as e:
                logger.error(f"Failed to read DICOM series {series_path}: {e}")
                return None, metadata
                
        except Exception as e:
            logger.error(f"Error processing DICOM series {series_path}: {e}")
            return None, None
    
    def identify_phase(self, metadata):
        """Identify CT phase from metadata"""
        # Combine description and protocol name for phase identification
        text_to_search = f"{metadata['series_description']} {metadata['protocol_name']}".lower()
        
        # Check each phase
        for phase, keywords in self.phase_keywords.items():
            if any(keyword in text_to_search for keyword in keywords):
                return phase
        
        return 'unknown'
    
    def convert_sitk_to_nifti(self, sitk_image, output_path):
        """Convert SimpleITK image to NIfTI format"""
        try:
            sitk.WriteImage(sitk_image, str(output_path))
            return True
        except Exception as e:
            logger.error(f"Failed to save NIfTI {output_path}: {e}")
            return False

class RegistrationProcessor:
    """Handles image registration between CT phases"""
    
    def __init__(self, target_spacing=(1.0, 1.0, 3.0)):
        self.target_spacing = target_spacing
        
    def resample_image(self, image, target_spacing=None):
        """Resample image to target spacing"""
        if target_spacing is None:
            target_spacing = self.target_spacing
            
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        
        # Calculate new size
        new_size = [
            int(round(osz * ospc / tspc)) 
            for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
        ]
        
        # Setup resampler
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(image.GetPixelIDValue())
        resampler.SetInterpolator(sitk.sitkLinear)
        
        return resampler.Execute(image)
    
    def normalize_intensity(self, image, intensity_range=(-100, 300)):
        """Normalize image intensity for CT"""
        # Convert to numpy for processing
        array = sitk.GetArrayFromImage(image)
        
        # Clip to HU range
        array = np.clip(array, intensity_range[0], intensity_range[1])
        
        # Robust normalization using percentiles
        valid_pixels = array[array > intensity_range[0]]
        if len(valid_pixels) > 0:
            p1, p99 = np.percentile(valid_pixels, [1, 99])
            array = np.clip(array, p1, p99)
            
            # Z-score normalization
            mean_val = np.mean(array)
            std_val = np.std(array)
            if std_val > 0:
                array = (array - mean_val) / std_val
        
        # Convert back to SimpleITK
        normalized_image = sitk.GetImageFromArray(array)
        normalized_image.CopyInformation(image)
        
        return normalized_image
    
    def register_images(self, fixed_image, moving_image, registration_type='rigid'):
        """Register moving image to fixed image"""
        try:
            # Normalize intensities
            fixed_normalized = self.normalize_intensity(fixed_image)
            moving_normalized = self.normalize_intensity(moving_image)
            
            # Setup registration method
            registration_method = sitk.ImageRegistrationMethod()
            
            # Metric
            registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
            registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
            registration_method.SetMetricSamplingPercentage(0.01)
            
            # Interpolator
            registration_method.SetInterpolator(sitk.sitkLinear)
            
            # Optimizer
            registration_method.SetOptimizerAsRegularStepGradientDescent(
                learningRate=1.0,
                minStep=1e-6,
                numberOfIterations=200,
                gradientMagnitudeTolerance=1e-8
            )
            registration_method.SetOptimizerScalesFromPhysicalShift()
            
            # Setup initial transform
            if registration_type == 'rigid':
                initial_transform = sitk.CenteredTransformInitializer(
                    fixed_normalized, moving_normalized, 
                    sitk.Euler3DTransform(),
                    sitk.CenteredTransformInitializerFilter.GEOMETRY
                )
            else:  # affine
                initial_transform = sitk.CenteredTransformInitializer(
                    fixed_normalized, moving_normalized,
                    sitk.AffineTransform(3),
                    sitk.CenteredTransformInitializerFilter.GEOMETRY
                )
            
            registration_method.SetInitialTransform(initial_transform, inPlace=False)
            
            # Multi-resolution framework
            registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
            registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
            registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
            
            # Execute registration
            final_transform = registration_method.Execute(fixed_normalized, moving_normalized)
            
            # Apply transform to original moving image
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(fixed_image)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(-1000)  # Air HU value
            resampler.SetTransform(final_transform)
            
            registered_image = resampler.Execute(moving_image)
            
            # Calculate registration quality metrics
            metric_value = registration_method.GetMetricValue()
            optimizer_iteration = registration_method.GetOptimizerIteration()
            
            registration_info = {
                'metric_value': metric_value,
                'iterations': optimizer_iteration,
                'transform_parameters': final_transform.GetParameters()
            }
            
            return registered_image, registration_info
            
        except Exception as e:
            logger.error(f"Registration failed: {e}")
            return moving_image, {'error': str(e)}

class SegmentationProcessor:
    """Handles organ segmentation using TotalSegmentator"""
    
    def __init__(self, temp_dir="./temp_segmentation"):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True, parents=True)
        
        # Check if TotalSegmentator is available
        try:
            result = subprocess.run(['TotalSegmentator', '--help'], 
                                  capture_output=True, text=True)
            self.totalseg_available = result.returncode == 0
        except FileNotFoundError:
            self.totalseg_available = False
            logger.warning("TotalSegmentator not found. Segmentation will be skipped.")
    
    def segment_image(self, image, case_id, phase_name):
        """Segment organs using TotalSegmentator"""
        if not self.totalseg_available:
            logger.warning("TotalSegmentator not available, skipping segmentation")
            return None
        
        try:
            # Create temporary files
            input_file = self.temp_dir / f"{case_id}_{phase_name}_input.nii.gz"
            output_dir = self.temp_dir / f"{case_id}_{phase_name}_output"
            
            # Save input image
            sitk.WriteImage(image, str(input_file))
            
            # Run TotalSegmentator
            cmd = [
                'TotalSegmentator',
                '-i', str(input_file),
                '-o', str(output_dir),
                '--ml',  # Use machine learning model
                '--fast'  # Faster inference
            ]
            
            logger.info(f"Running TotalSegmentator for {case_id}_{phase_name}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                logger.error(f"TotalSegmentator failed: {result.stderr}")
                return None
            
            # Load segmentation result
            seg_file = output_dir / "segmentations.nii.gz"
            if seg_file.exists():
                segmentation = sitk.ReadImage(str(seg_file))
                
                # Cleanup temporary files
                input_file.unlink(missing_ok=True)
                shutil.rmtree(output_dir, ignore_errors=True)
                
                return segmentation
            else:
                logger.error(f"Segmentation output not found: {seg_file}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error(f"TotalSegmentator timeout for {case_id}_{phase_name}")
            return None
        except Exception as e:
            logger.error(f"Segmentation failed for {case_id}_{phase_name}: {e}")
            return None
    
    def cleanup(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

class CTPreprocessingPipeline:
    """Main preprocessing pipeline for CT phase generation"""
    
    def __init__(self, data_root, labels_csv, output_dir, temp_dir="./temp"):
        self.data_root = Path(data_root)
        self.labels_csv = Path(labels_csv)
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir)
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.temp_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize processors
        self.dicom_processor = DICOMProcessor()
        self.registration_processor = RegistrationProcessor()
        self.segmentation_processor = SegmentationProcessor(self.temp_dir / "segmentation")
        
        # Load labels
        self.labels_df = self.load_labels()
        
        # Processing statistics
        self.stats = {
            'total_cases': 0,
            'processed_cases': 0,
            'failed_cases': [],
            'registration_quality': {},
            'segmentation_success': {}
        }
    
    def load_labels(self):
        """Load and validate labels CSV"""
        try:
            df = pd.read_csv(self.labels_csv)
            logger.info(f"Loaded labels CSV with {len(df)} entries")
            
            # Ensure required columns exist
            required_columns = ['case_uid', 'series_uid', 'phase_label']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in labels CSV")
            
            return df
        except Exception as e:
            logger.error(f"Failed to load labels CSV: {e}")
            raise
    
    def find_case_series(self, case_uid):
        """Find all series for a case and match with labels"""
        case_path = self.data_root / case_uid
        
        if not case_path.exists():
            logger.warning(f"Case directory not found: {case_path}")
            return {}
        
        # Get labels for this case
        case_labels = self.labels_df[self.labels_df['case_uid'] == case_uid]
        
        if case_labels.empty:
            logger.warning(f"No labels found for case: {case_uid}")
            return {}
        
        series_info = {}
        
        # Find series directories
        for series_dir in case_path.iterdir():
            if not series_dir.is_dir():
                continue
            
            series_uid = series_dir.name
            
            # Check if this series is in our labels
            series_label = case_labels[case_labels['series_uid'] == series_uid]
            
            if series_label.empty:
                logger.debug(f"Series {series_uid} not in labels, skipping")
                continue
            
            # Read DICOM metadata
            image, metadata = self.dicom_processor.read_dicom_series(series_dir)
            
            if image is None or metadata is None:
                logger.warning(f"Failed to read series: {series_dir}")
                continue
            
            phase_label = series_label.iloc[0]['phase_label'].lower()
            
            series_info[series_uid] = {
                'image': image,
                'metadata': metadata,
                'phase_label': phase_label,
                'series_path': series_dir
            }
        
        return series_info
    
    def select_best_series_per_phase(self, series_info):
        """Select the series with largest slice thickness for each phase"""
        phase_series = defaultdict(list)
        
        # Group series by phase
        for series_uid, info in series_info.items():
            phase = info['phase_label']
            phase_series[phase].append((series_uid, info))
        
        # Select best series for each phase (largest slice thickness)
        selected_series = {}
        
        for phase, series_list in phase_series.items():
            if not series_list:
                continue
            
            # Sort by slice thickness (descending) - larger thickness first
            series_list.sort(key=lambda x: x[1]['metadata']['slice_thickness'], reverse=True)
            
            best_series_uid, best_info = series_list[0]
            selected_series[phase] = best_info
            
            logger.info(f"Selected {phase} series {best_series_uid} "
                       f"(slice thickness: {best_info['metadata']['slice_thickness']:.2f}mm)")
            
            if len(series_list) > 1:
                logger.info(f"  Skipped {len(series_list)-1} other {phase} series")
        
        return selected_series
    
    def register_case_phases(self, selected_series, case_uid):
        """Register all phases to the reference phase (non-contrast preferred)"""
        if not selected_series:
            return {}, {}
        
        # Determine reference phase (prefer non-contrast, then arterial)
        reference_phase = None
        if 'noncontrast' in selected_series:
            reference_phase = 'noncontrast'
        elif 'arterial' in selected_series:
            reference_phase = 'arterial'
        else:
            reference_phase = list(selected_series.keys())[0]
        
        logger.info(f"Using {reference_phase} as reference phase for case {case_uid}")
        
        reference_image = selected_series[reference_phase]['image']
        
        # Resample reference image to target spacing
        reference_resampled = self.registration_processor.resample_image(reference_image)
        
        registered_images = {reference_phase: reference_resampled}
        registration_info = {reference_phase: {'is_reference': True}}
        
        # Register other phases to reference
        for phase, series_info in selected_series.items():
            if phase == reference_phase:
                continue
            
            logger.info(f"Registering {phase} to {reference_phase} for case {case_uid}")
            
            # Resample moving image to same spacing as reference
            moving_resampled = self.registration_processor.resample_image(series_info['image'])
            
            # Register
            registered_image, reg_info = self.registration_processor.register_images(
                reference_resampled, moving_resampled, registration_type='rigid'
            )
            
            registered_images[phase] = registered_image
            registration_info[phase] = reg_info
            
            # Log registration quality
            if 'metric_value' in reg_info:
                logger.info(f"  Registration metric: {reg_info['metric_value']:.4f}")
        
        return registered_images, registration_info
    
    def segment_contrast_phases(self, registered_images, case_uid):
        """Segment organs on contrast phases (arterial preferred, then portal)"""
        segmentations = {}
        
        # Priority order for segmentation
        segmentation_priority = ['arterial', 'portal', 'delayed']
        
        for phase in segmentation_priority:
            if phase in registered_images:
                logger.info(f"Segmenting {phase} phase for case {case_uid}")
                
                segmentation = self.segmentation_processor.segment_image(
                    registered_images[phase], case_uid, phase
                )
                
                if segmentation is not None:
                    segmentations[phase] = segmentation
                    logger.info(f"Successfully segmented {phase} phase")
                    
                    # For now, we only segment one phase per case
                    # (can be extended to segment multiple phases)
                    break
                else:
                    logger.warning(f"Segmentation failed for {phase} phase")
        
        return segmentations
    
    def save_processed_case(self, case_uid, registered_images, segmentations, registration_info):
        """Save all processed data for a case"""
        case_output_dir = self.output_dir / case_uid
        case_output_dir.mkdir(exist_ok=True)
        
        # Save registered images
        for phase, image in registered_images.items():
            output_path = case_output_dir / f"{case_uid}_{phase}_registered.nii.gz"
            
            if self.dicom_processor.convert_sitk_to_nifti(image, output_path):
                logger.info(f"Saved {phase} image: {output_path}")
            else:
                logger.error(f"Failed to save {phase} image: {output_path}")
        
        # Save segmentations
        for phase, segmentation in segmentations.items():
            seg_output_path = case_output_dir / f"{case_uid}_{phase}_segmentation.nii.gz"
            
            if self.dicom_processor.convert_sitk_to_nifti(segmentation, seg_output_path):
                logger.info(f"Saved {phase} segmentation: {seg_output_path}")
            else:
                logger.error(f"Failed to save {phase} segmentation: {seg_output_path}")
        
        # Save registration info
        reg_info_path = case_output_dir / f"{case_uid}_registration_info.json"
        
        # Convert numpy arrays to lists for JSON serialization
        json_safe_info = {}
        for phase, info in registration_info.items():
            json_safe_info[phase] = {}
            for key, value in info.items():
                if isinstance(value, np.ndarray):
                    json_safe_info[phase][key] = value.tolist()
                else:
                    json_safe_info[phase][key] = value
        
        try:
            with open(reg_info_path, 'w') as f:
                json.dump(json_safe_info, f, indent=2)
            logger.info(f"Saved registration info: {reg_info_path}")
        except Exception as e:
            logger.error(f"Failed to save registration info: {e}")
    
    def process_case(self, case_uid):
        """Process a single case"""
        logger.info(f"Processing case: {case_uid}")
        
        try:
            # Find and validate series
            series_info = self.find_case_series(case_uid)
            
            if not series_info:
                logger.warning(f"No valid series found for case {case_uid}")
                self.stats['failed_cases'].append((case_uid, "No valid series"))
                return False
            
            # Select best series per phase
            selected_series = self.select_best_series_per_phase(series_info)
            
            if not selected_series:
                logger.warning(f"No series selected for case {case_uid}")
                self.stats['failed_cases'].append((case_uid, "No series selected"))
                return False
            
            logger.info(f"Found {len(selected_series)} phases: {list(selected_series.keys())}")
            
            # Register phases
            registered_images, registration_info = self.register_case_phases(selected_series, case_uid)
            
            if not registered_images:
                logger.error(f"Registration failed for case {case_uid}")
                self.stats['failed_cases'].append((case_uid, "Registration failed"))
                return False
            
            # Segment contrast phases
            segmentations = self.segment_contrast_phases(registered_images, case_uid)
            
            # Save results
            self.save_processed_case(case_uid, registered_images, segmentations, registration_info)
            
            # Update statistics
            self.stats['registration_quality'][case_uid] = registration_info
            self.stats['segmentation_success'][case_uid] = len(segmentations) > 0
            
            logger.info(f"Successfully processed case {case_uid}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing case {case_uid}: {e}")
            self.stats['failed_cases'].append((case_uid, str(e)))
            return False
    
    def run(self):
        """Run the complete preprocessing pipeline"""
        logger.info("Starting CT preprocessing pipeline")
        logger.info(f"Data root: {self.data_root}")
        logger.info(f"Labels CSV: {self.labels_csv}")
        logger.info(f"Output directory: {self.output_dir}")
        
        # Get list of cases to process
        case_uids = self.labels_df['case_uid'].unique()
        self.stats['total_cases'] = len(case_uids)
        
        logger.info(f"Found {len(case_uids)} unique cases to process")
        
        # Process each case
        for case_uid in tqdm(case_uids, desc="Processing cases"):
            success = self.process_case(case_uid)
            if success:
                self.stats['processed_cases'] += 1
        
        # Final statistics
        self.print_summary()
        self.save_statistics()
        
        # Cleanup
        self.segmentation_processor.cleanup()
    
    def print_summary(self):
        """Print processing summary"""
        logger.info("=" * 60)
        logger.info("PREPROCESSING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total cases: {self.stats['total_cases']}")
        logger.info(f"Successfully processed: {self.stats['processed_cases']}")
        logger.info(f"Failed: {len(self.stats['failed_cases'])}")
        
        if self.stats['failed_cases']:
            logger.info("\nFailed cases:")
            for case_uid, reason in self.stats['failed_cases']:
                logger.info(f"  {case_uid}: {reason}")
        
        # Registration quality summary
        reg_metrics = []
        for case_uid, reg_info in self.stats['registration_quality'].items():
            for phase, info in reg_info.items():
                if 'metric_value' in info:
                    reg_metrics.append(info['metric_value'])
        
        if reg_metrics:
            logger.info(f"\nRegistration quality (MI metric):")
            logger.info(f"  Mean: {np.mean(reg_metrics):.4f}")
            logger.info(f"  Std: {np.std(reg_metrics):.4f}")
            logger.info(f"  Min: {np.min(reg_metrics):.4f}")
            logger.info(f"  Max: {np.max(reg_metrics):.4f}")
        
        # Segmentation success rate
        seg_success = sum(self.stats['segmentation_success'].values())
        seg_total = len(self.stats['segmentation_success'])
        
        if seg_total > 0:
            logger.info(f"\nSegmentation success rate: {seg_success}/{seg_total} ({100*seg_success/seg_total:.1f}%)")
    
    def save_statistics(self):
        """Save detailed statistics"""
        stats_file = self.output_dir / "preprocessing_statistics.json"
        
        # Prepare statistics for JSON serialization
        json_stats = {
            'processing_date': datetime.now().isoformat(),
            'total_cases': self.stats['total_cases'],
            'processed_cases': self.stats['processed_cases'],
            'failed_cases': self.stats['failed_cases'],
            'segmentation_success': self.stats['segmentation_success']
        }
        
        # Add registration quality metrics
        reg_summary = {}
        for case_uid, reg_info in self.stats['registration_quality'].items():
            reg_summary[case_uid] = {}
            for phase, info in reg_info.items():
                reg_summary[case_uid][phase] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in info.items()
                }
        
        json_stats['registration_quality'] = reg_summary
        
        try:
            with open(stats_file, 'w') as f:
                json.dump(json_stats, f, indent=2)
            logger.info(f"Statistics saved to: {stats_file}")
        except Exception as e:
            logger.error(f"Failed to save statistics: {e}")

def main():
    """Main execution function"""
    
    # Configuration
    config = {
        'data_root': '/path/to/your/dicom/data',  # Update this path
        'labels_csv': '/path/to/your/labels.csv',  # Update this path  
        'output_dir': '/path/to/your/output',      # Update this path
        'temp_dir': './temp_preprocessing'
    }
    
    # Validate paths
    for key, path in config.items():
        if key != 'temp_dir' and not Path(path).exists():
            logger.error(f"{key} does not exist: {path}")
            return
    
    # Initialize and run pipeline
    pipeline = CTPreprocessingPipeline(
        data_root=config['data_root'],
        labels_csv=config['labels_csv'],
        output_dir=config['output_dir'],
        temp_dir=config['temp_dir']
    )
    
    pipeline.run()

if __name__ == "__main__":
    main()