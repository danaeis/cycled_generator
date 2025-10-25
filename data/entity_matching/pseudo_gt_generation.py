"""
Pseudo-Ground Truth Generation for Supervised Training
Creates perfectly aligned target phase images for GAN/Diffusion training
"""

import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
from pathlib import Path
import json
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset


class PseudoGroundTruthGenerator:
    """
    Generate pseudo-GT from registered and aligned volumes
    """
    
    def __init__(self, quality_threshold: float = 0.85):
        """
        Args:
            quality_threshold: Minimum Dice score to accept as pseudo-GT
        """
        self.quality_threshold = quality_threshold
        self.generated_pairs = []
    
    def verify_alignment_quality(
        self,
        fixed_seg: sitk.Image,
        moving_seg: sitk.Image
    ) -> Dict[str, float]:
        """
        Check if alignment is good enough for pseudo-GT
        
        Returns dict with metrics
        """
        from registration_validation import RegistrationValidator
        
        validator = RegistrationValidator()
        dice = validator.compute_dice(fixed_seg, moving_seg)
        hausdorff = validator.compute_hausdorff(fixed_seg, moving_seg)
        
        is_good = dice >= self.quality_threshold
        
        return {
            'dice': dice,
            'hausdorff': hausdorff,
            'is_good_quality': is_good
        }
    
    def generate_paired_dataset(
        self,
        study_id: str,
        aligned_dir: str,
        labels_df: pd.DataFrame,
        output_dir: str,
        source_phase: str = "Non-contrast",
        target_phases: List[str] = None
    ):
        """
        Generate paired source→target datasets for supervised training
        
        Args:
            source_phase: Source contrast phase (e.g., "Non-contrast")
            target_phases: List of target phases (e.g., ["Arterial", "Venous"])
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if target_phases is None:
            target_phases = ["Arterial", "Portal-venous", "Venous"]
        
        # Get source volume
        source_row = labels_df[
            (labels_df["StudyInstanceUID"] == study_id) & 
            (labels_df["Label"] == source_phase)
        ]
        
        if source_row.empty:
            print(f"⚠️ Source phase '{source_phase}' not found for {study_id}")
            return
        
        source_series = source_row.iloc[0]["SeriesInstanceUID"]
        source_vol_path = os.path.join(
            aligned_dir, f"{study_id}_{source_series}_aligned.nii.gz"
        )
        source_seg_path = os.path.join(
            aligned_dir, f"{study_id}_{source_series}_aligned_seg.nii.gz"
        )
        
        # If source is reference (non-contrast), it might not have _aligned suffix
        if not os.path.exists(source_vol_path):
            source_vol_path = os.path.join(
                aligned_dir, f"{study_id}_{source_series}_deformable.nii.gz"
            )
            source_seg_path = os.path.join(
                aligned_dir, f"{study_id}_{source_series}_deformable_seg.nii.gz"
            )
        
        if not os.path.exists(source_vol_path):
            print(f"⚠️ Source volume not found: {source_vol_path}")
            return
        
        source_vol = sitk.ReadImage(source_vol_path)
        source_seg = sitk.ReadImage(source_seg_path) if os.path.exists(source_seg_path) else None
        
        # Process each target phase
        for target_phase in target_phases:
            target_row = labels_df[
                (labels_df["StudyInstanceUID"] == study_id) & 
                (labels_df["Label"] == target_phase)
            ]
            
            if target_row.empty:
                continue
            
            target_series = target_row.iloc[0]["SeriesInstanceUID"]
            target_vol_path = os.path.join(
                aligned_dir, f"{study_id}_{target_series}_aligned.nii.gz"
            )
            target_seg_path = os.path.join(
                aligned_dir, f"{study_id}_{target_series}_aligned_seg.nii.gz"
            )
            
            if not os.path.exists(target_vol_path):
                continue
            
            try:
                target_vol = sitk.ReadImage(target_vol_path)
                target_seg = sitk.ReadImage(target_seg_path) if os.path.exists(target_seg_path) else None
                
                # Verify quality
                if source_seg is not None and target_seg is not None:
                    quality = self.verify_alignment_quality(source_seg, target_seg)
                    print(f"   Quality check: Dice={quality['dice']:.4f}, "
                          f"HD={quality['hausdorff']:.2f}mm")
                    
                    if not quality['is_good_quality']:
                        print(f"   ⚠️ Skipping {target_phase}: Quality below threshold")
                        continue
                
                # Save paired data
                pair_info = self._save_training_pair(
                    source_vol, target_vol,
                    study_id, source_phase, target_phase,
                    output_dir
                )
                
                self.generated_pairs.append(pair_info)
                print(f"   ✅ Generated pair: {source_phase} → {target_phase}")
            
            except Exception as e:
                print(f"   ❌ Error processing {target_phase}: {e}")
        
        # Save metadata
        metadata_path = os.path.join(output_dir, f"{study_id}_pairs_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.generated_pairs, f, indent=2)
    
    def _save_training_pair(
        self,
        source: sitk.Image,
        target: sitk.Image,
        study_id: str,
        source_phase: str,
        target_phase: str,
        output_dir: str
    ) -> Dict:
        """
        Save a source-target pair for training
        """
        # Create subdirectories
        source_dir = os.path.join(output_dir, "source")
        target_dir = os.path.join(output_dir, "target")
        os.makedirs(source_dir, exist_ok=True)
        os.makedirs(target_dir, exist_ok=True)
        
        # Generate filename
        pair_name = f"{study_id}_{source_phase}_to_{target_phase}"
        
        # Save volumes
        source_path = os.path.join(source_dir, f"{pair_name}_source.nii.gz")
        target_path = os.path.join(target_dir, f"{pair_name}_target.nii.gz")
        
        sitk.WriteImage(source, source_path)
        sitk.WriteImage(target, target_path)
        
        return {
            'study_id': study_id,
            'source_phase': source_phase,
            'target_phase': target_phase,
            'source_path': source_path,
            'target_path': target_path,
            'pair_name': pair_name
        }
    
    def create_slice_pairs(
        self,
        study_id: str,
        aligned_dir: str,
        labels_df: pd.DataFrame,
        output_dir: str,
        source_phase: str = "Non-contrast",
        target_phases: List[str] = None,
        save_as_2d: bool = True
    ):
        """
        Generate 2D slice pairs for 2D model training
        
        Args:
            save_as_2d: If True, save individual PNG/NPY slices
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if target_phases is None:
            target_phases = ["Arterial", "Portal-venous", "Venous"]
        
        # Get source
        source_row = labels_df[
            (labels_df["StudyInstanceUID"] == study_id) & 
            (labels_df["Label"] == source_phase)
        ]
        
        if source_row.empty:
            return
        
        source_series = source_row.iloc[0]["SeriesInstanceUID"]
        source_vol_path = os.path.join(
            aligned_dir, f"{study_id}_{source_series}_aligned.nii.gz"
        )
        
        if not os.path.exists(source_vol_path):
            source_vol_path = os.path.join(
                aligned_dir, f"{study_id}_{source_series}_deformable.nii.gz"
            )
        
        if not os.path.exists(source_vol_path):
            return
        
        source_vol = sitk.ReadImage(source_vol_path)
        source_array = sitk.GetArrayFromImage(source_vol)
        
        # Process targets
        for target_phase in target_phases:
            target_row = labels_df[
                (labels_df["StudyInstanceUID"] == study_id) & 
                (labels_df["Label"] == target_phase)
            ]
            
            if target_row.empty:
                continue
            
            target_series = target_row.iloc[0]["SeriesInstanceUID"]
            target_vol_path = os.path.join(
                aligned_dir, f"{study_id}_{target_series}_aligned.nii.gz"
            )
            
            if not os.path.exists(target_vol_path):
                continue
            
            target_vol = sitk.ReadImage(target_vol_path)
            target_array = sitk.GetArrayFromImage(target_vol)
            
            if save_as_2d:
                # Save individual slices
                slice_dir = os.path.join(
                    output_dir, 
                    f"{source_phase}_to_{target_phase}",
                    study_id
                )
                os.makedirs(os.path.join(slice_dir, "source"), exist_ok=True)
                os.makedirs(os.path.join(slice_dir, "target"), exist_ok=True)
                
                for z in range(min(source_array.shape[0], target_array.shape[0])):
                    source_slice = source_array[z]
                    target_slice = target_array[z]
                    
                    # Save as numpy
                    np.save(
                        os.path.join(slice_dir, "source", f"slice_{z:04d}.npy"),
                        source_slice
                    )
                    np.save(
                        os.path.join(slice_dir, "target", f"slice_{z:04d}.npy"),
                        target_slice
                    )
                
                print(f"   ✅ Saved {source_array.shape[0]} slice pairs: "
                      f"{source_phase} → {target_phase}")


class PairedCTDataset(Dataset):
    """
    PyTorch Dataset for paired CT phase translation
    """
    
    def __init__(
        self,
        pairs_dir: str,
        transform=None,
        normalize: bool = True,
        return_3d: bool = True,
        depth: int = 64
    ):
        """
        Args:
            pairs_dir: Directory with source/ and target/ subdirs
            transform: Optional transforms
            normalize: Apply intensity normalization
            return_3d: If True, return 3D volumes; else 2D slices
            depth: Number of slices for 3D volumes
        """
        self.pairs_dir = pairs_dir
        self.transform = transform
        self.normalize = normalize
        self.return_3d = return_3d
        self.depth = depth
        
        # Find all pairs
        self.pairs = self._find_pairs()
    
    def _find_pairs(self) -> List[Dict]:
        """Find all source-target pairs"""
        source_dir = os.path.join(self.pairs_dir, "source")
        target_dir = os.path.join(self.pairs_dir, "target")
        
        pairs = []
        
        for source_file in Path(source_dir).glob("*.nii.gz"):
            # Find corresponding target
            target_file = Path(target_dir) / source_file.name.replace("_source", "_target")
            
            if target_file.exists():
                pairs.append({
                    'source': str(source_file),
                    'target': str(target_file)
                })
        
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Load volumes
        source_sitk = sitk.ReadImage(pair['source'])
        target_sitk = sitk.ReadImage(pair['target'])
        
        source = sitk.GetArrayFromImage(source_sitk).astype(np.float32)
        target = sitk.GetArrayFromImage(target_sitk).astype(np.float32)
        
        # Normalize
        if self.normalize:
            source = self._normalize_intensity(source)
            target = self._normalize_intensity(target)
        
        if self.return_3d:
            # Crop/pad to fixed depth
            source = self._adjust_depth(source, self.depth)
            target = self._adjust_depth(target, self.depth)
            
            # Add channel dimension
            source = torch.from_numpy(source).unsqueeze(0)  # (1, D, H, W)
            target = torch.from_numpy(target).unsqueeze(0)
        else:
            # Return random 2D slice
            z = np.random.randint(0, min(source.shape[0], target.shape[0]))
            source = torch.from_numpy(source[z]).unsqueeze(0)  # (1, H, W)
            target = torch.from_numpy(target[z]).unsqueeze(0)
        
        if self.transform:
            source = self.transform(source)
            target = self.transform(target)
        
        return {'source': source, 'target': target}
    
    def _normalize_intensity(self, volume: np.ndarray) -> np.ndarray:
        """Normalize to [-1, 1]"""
        volume = volume.clip(-100, 400)  # CT Hounsfield units
        volume = (volume - volume.mean()) / (volume.std() + 1e-8)
        return volume
    
    def _adjust_depth(self, volume: np.ndarray, target_depth: int) -> np.ndarray:
        """Crop or pad to target depth"""
        current_depth = volume.shape[0]
        
        if current_depth > target_depth:
            # Center crop
            start = (current_depth - target_depth) // 2
            return volume[start:start+target_depth]
        elif current_depth < target_depth:
            # Zero pad
            pad_before = (target_depth - current_depth) // 2
            pad_after = target_depth - current_depth - pad_before
            return np.pad(volume, ((pad_before, pad_after), (0, 0), (0, 0)))
        else:
            return volume


# ============================================================================
# Data Quality Assessment
# ============================================================================

def assess_pseudo_gt_quality(
    pairs_dir: str,
    validation_results_dir: str,
    output_report: str
):
    """
    Assess quality of generated pseudo-GT pairs
    
    Args:
        pairs_dir: Directory with generated pairs
        validation_results_dir: Directory with validation metrics
        output_report: Path to save quality report
    """
    import glob
    
    # Load all validation metrics
    metrics_files = glob.glob(
        os.path.join(validation_results_dir, "**/*_metrics.json"),
        recursive=True
    )
    
    quality_data = []
    
    for metrics_file in metrics_files:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        quality_data.append({
            'file': os.path.basename(metrics_file),
            'dice': metrics.get('dice_overall', 0),
            'hausdorff': metrics.get('hausdorff_95', 999),
            'quality_grade': 'High' if metrics.get('dice_overall', 0) > 0.85 else 
                           'Medium' if metrics.get('dice_overall', 0) > 0.75 else 'Low'
        })
    
    df = pd.DataFrame(quality_data)
    
    # Summary statistics
    print("\n" + "="*60)
    print("Pseudo-Ground Truth Quality Assessment")
    print("="*60)
    print(f"\nTotal pairs: {len(df)}")
    print(f"High quality (Dice > 0.85): {len(df[df['quality_grade']=='High'])}")
    print(f"Medium quality (0.75 < Dice ≤ 0.85): {len(df[df['quality_grade']=='Medium'])}")
    print(f"Low quality (Dice ≤ 0.75): {len(df[df['quality_grade']=='Low'])}")
    print(f"\nMean Dice: {df['dice'].mean():.4f} ± {df['dice'].std():.4f}")
    print(f"Mean Hausdorff: {df['hausdorff'].mean():.2f} ± {df['hausdorff'].std():.2f} mm")
    
    # Save report
    df.to_csv(output_report, index=False)
    print(f"\n📊 Quality report saved to: {output_report}")
    
    return df


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    from configs import MAIN_PATH
    
    labels_csv = MAIN_PATH + "labels.csv"
    labels_df = pd.read_csv(labels_csv)
    
    aligned_dir = MAIN_PATH + "deformable_registered"
    pseudo_gt_dir = MAIN_PATH + "pseudo_ground_truth"
    
    os.makedirs(pseudo_gt_dir, exist_ok=True)
    
    generator = PseudoGroundTruthGenerator(quality_threshold=0.85)
    
    # Generate pseudo-GT for all studies
    for study_id in labels_df["StudyInstanceUID"].unique():
        study_aligned = os.path.join(aligned_dir, study_id)
        study_output = os.path.join(pseudo_gt_dir, study_id)
        
        if not os.path.exists(study_aligned):
            continue
        
        print(f"\n🔄 Generating pseudo-GT for {study_id}")
        
        # Generate 3D volume pairs
        generator.generate_paired_dataset(
            study_id,
            study_aligned,
            labels_df,
            study_output,
            source_phase="Non-contrast",
            target_phases=["Arterial", "Portal-venous", "Venous"]
        )
        
        # Also generate 2D slice pairs
        generator.create_slice_pairs(
            study_id,
            study_aligned,
            labels_df,
            os.path.join(pseudo_gt_dir, "slices_2d"),
            source_phase="Non-contrast",
            target_phases=["Arterial", "Portal-venous", "Venous"],
            save_as_2d=True
        )
    
    # Assess quality
    validation_dir = MAIN_PATH + "validation_results"
    quality_report = os.path.join(pseudo_gt_dir, "quality_assessment.csv")
    
    assess_pseudo_gt_quality(pseudo_gt_dir, validation_dir, quality_report)
    
    print("\n✅ Pseudo-ground truth generation completed")
    
    # Example: Create PyTorch dataset
    print("\n" + "="*60)
    print("Creating PyTorch Dataset")
    print("="*60)
    
    dataset = PairedCTDataset(
        pairs_dir=os.path.join(pseudo_gt_dir, labels_df["StudyInstanceUID"].iloc[0]),
        normalize=True,
        return_3d=True,
        depth=64
    )
    
    print(f"Dataset size: {len(dataset)} pairs")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Source shape: {sample['source'].shape}")
        print(f"Target shape: {sample['target'].shape}")
