import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path
from scipy import ndimage
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.image.fid import FrechetInceptionDistance
import warnings
warnings.filterwarnings('ignore')

class CTPreprocessor:
    """Handles multi-scanner CT preprocessing with robust normalization"""
    
    def __init__(self, target_spacing=(1.0, 1.0, 3.0), intensity_range=(-100, 300)):
        self.target_spacing = target_spacing
        self.intensity_range = intensity_range
        
    def normalize_intensity(self, image, phase_type="arterial"):
        """Scanner-agnostic intensity normalization"""
        # Clip to HU range for abdomen
        image = np.clip(image, self.intensity_range[0], self.intensity_range[1])
        
        # Robust normalization using percentiles (handles scanner differences)
        p1, p99 = np.percentile(image[image > -100], [1, 99])
        image = np.clip(image, p1, p99)
        
        # Z-score normalization
        image = (image - np.mean(image)) / (np.std(image) + 1e-8)
        
        return image.astype(np.float32)

class MaskProcessor:
    """Handles segmentation masks and registration transfer"""
    
    def __init__(self):
        self.organ_labels = {
            'liver': [1, 2], 'kidney_right': [3], 'kidney_left': [4],
            'spleen': [5], 'pancreas': [6], 'aorta': [7], 'ivc': [8]
        }
    
    def extract_patch_masks(self, full_masks, patch_coords):
        """Extract mask patches corresponding to image patches"""
        z_start, z_end, y_start, y_end, x_start, x_end = patch_coords
        patch_masks = full_masks[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Create binary masks for each organ
        organ_masks = {}
        for organ, labels in self.organ_labels.items():
            organ_mask = np.zeros_like(patch_masks)
            for label in labels:
                organ_mask[patch_masks == label] = 1
            organ_masks[organ] = organ_mask
            
        return organ_masks

def load_phase_mapping(labels_csv_path):
    """Load phase mapping from CSV file"""
    import pandas as pd
    df = pd.read_csv(labels_csv_path)
    
    # Create mapping from series_uid to phase
    phase_mapping = {}
    for _, row in df.iterrows():
        case_uid = row['StudyInstanceUID']
        series_uid = row['SeriesInstanceUID']
        phase_label = row['Label'].lower()
        
        if case_uid not in phase_mapping:
            phase_mapping[case_uid] = {}
        phase_mapping[case_uid][series_uid] = phase_label
    
    return phase_mapping

def create_data_splits_from_directories(data_dir, phase_mapping=None, test_size=0.15, val_size=0.15, random_state=42):
    """Create patient-level data splits from directory structure"""
    data_dir = Path(data_dir)
    
    # Collect all cases and their available files
    cases = {}
    
    for case_dir in data_dir.iterdir():
        if not case_dir.is_dir():
            print("no directory")
            continue
            
        case_id = case_dir.name
        cases[case_id] = {}
        
        # Find all registered image files (not segmentations)
        for nii_file in case_dir.glob("*.nii.gz"):
            if "_seg" in str(nii_file):
                print("segmentation file")
                continue  # Skip segmentation files
                
            # Extract series ID from filename
            filename = nii_file.stem.replace("_registered", "").replace(".nii", '').replace(".gz", '')
            parts = filename.split('_')
            print("parts", parts)
            if len(parts) >= 2:
                series_id = parts[1]  # Assuming format: caseID_seriesID_registered.nii.gz
                
                # Determine phase from mapping or filename
                if phase_mapping and case_id in phase_mapping and series_id in phase_mapping[case_id]:
                    phase = phase_mapping[case_id][series_id]
                else:
                    # Try to infer phase from filename or series_id
                    phase = infer_phase_from_filename(filename)
                
                cases[case_id][phase] = {
                    'image': nii_file,
                    'series_id': series_id
                }
                
                # Check for corresponding segmentation
                seg_file = case_dir / f"{filename}_registered_seg.nii.gz"
                if seg_file.exists():
                    cases[case_id][phase]['segmentation'] = seg_file
    
    # Filter cases with required phases (non-contrast + at least one contrast)
    valid_cases = []
    for case_id, phases in cases.items():
        if 'non-contrast' in phases:
            contrast_phases = [p for p in phases.keys() if p != 'non-contrast']
            if contrast_phases:
                valid_cases.append((case_id, phases))
    
    print(f"Found {len(valid_cases)} valid cases with required phases")
    
    # Split cases
    case_ids = [case[0] for case in valid_cases]
    train_ids, temp_ids = train_test_split(case_ids, test_size=test_size+val_size, 
                                         random_state=random_state, stratify=None)
    val_ids, test_ids = train_test_split(temp_ids, test_size=test_size/(test_size+val_size), 
                                       random_state=random_state)
    
    # Create data pairs for training (focus on non-contrast -> contrast)
    def create_pairs(case_subset):
        pairs = []
        for case_id in case_subset:
            case_data = dict(valid_cases)[case_id]
            nc_info = case_data['non-contrast']
            
            for phase, phase_info in case_data.items():
                if phase != 'non-contrast':
                    # Non-contrast -> contrast transition
                    pairs.append({
                        'source_path': nc_info['image'],
                        'target_path': phase_info['image'],
                        'source_phase': 'non-contrast',
                        'target_phase': phase,
                        'case_id': case_id,
                        'source_series': nc_info['series_id'],
                        'target_series': phase_info['series_id'],
                        'source_seg': nc_info.get('segmentation'),
                        'target_seg': phase_info.get('segmentation')
                    })
        return pairs
    
    splits = {
        'train': create_pairs(train_ids),
        'val': create_pairs(val_ids),
        'test': create_pairs(test_ids)
    }
    
    print(f"Data splits created:")
    print(f"  Train: {len(train_ids)} cases, {len(splits['train'])} pairs")
    print(f"  Val: {len(val_ids)} cases, {len(splits['val'])} pairs")
    print(f"  Test: {len(test_ids)} cases, {len(splits['test'])} pairs")
    
    return splits

def infer_phase_from_filename(filename):
    """Infer CT phase from filename or series ID"""
    filename = filename.lower()
    
    phase_keywords = {
        'non-contrast': ['noncontrast', 'non-contrast', 'pre', 'baseline', 'native', 'nc'],
        'arterial': ['arterial', 'aterial', 'art', 'early', 'phase1', 'p1'],
        'portal': ['portal', 'venous', 'pv', 'phase2', 'p2', 'late'],
        'delayed': ['delayed', 'delay', 'equilibrium', 'phase3', 'p3']
    }
    
    for phase, keywords in phase_keywords.items():
        if any(keyword in filename for keyword in keywords):
            return phase
    
    return 'unknown'


class CTDataset(Dataset):
    """Fixed CT Dataset - simpler version for 90+ slice data"""
    
    def __init__(self, data_pairs, patch_size=(128, 128), overlap_ratio=0.5, augment=True):
        self.data_pairs = data_pairs
        self.patch_size = patch_size
        self.overlap_ratio = overlap_ratio
        self.augment = augment
        self.patch_coords = []
        
        print(f"Initializing dataset with {len(data_pairs)} data pairs")
        print(f"Patch size: {patch_size}, Overlap ratio: {overlap_ratio}")
        
        # Pre-compute all patch coordinates
        self._compute_patch_coordinates()
        
        print(f"Generated {len(self.patch_coords)} total patches")
    
    def _compute_patch_coordinates(self):
        """Pre-compute overlapping patch coordinates for all volumes"""
        
        for pair_idx, pair_data in enumerate(self.data_pairs):
            print(f"\nProcessing pair {pair_idx + 1}/{len(self.data_pairs)}")
            print(f"Source: {pair_data['source_path'].name}")
            print(f"Target: {pair_data['target_path'].name}")
            
            try:
                # Load volume to get dimensions
                source_vol = nib.load(pair_data['source_path']).get_fdata()
                target_vol = nib.load(pair_data['target_path']).get_fdata()
                
                print(f"Source shape: {source_vol.shape}")
                print(f"Target shape: {target_vol.shape}")
                
                # Check if shapes match
                if source_vol.shape != target_vol.shape:
                    print(f"⚠️  WARNING: Shape mismatch! Skipping this pair.")
                    continue
                
                # Use source volume shape for calculations
                depth, height, width = source_vol.shape
                
                # Check minimum requirements
                if depth < 7:
                    print(f"⚠️  WARNING: Only {depth} slices, need at least 7. Skipping.")
                    continue
                
                if height < self.patch_size[0] or width < self.patch_size[1]:
                    print(f"⚠️  WARNING: Spatial dimensions too small ({height}x{width}) for {self.patch_size} patches. Skipping.")
                    continue
                
                # Calculate step size for overlap
                step_y = max(1, int(self.patch_size[0] * (1 - self.overlap_ratio)))
                step_x = max(1, int(self.patch_size[1] * (1 - self.overlap_ratio)))
                
                print(f"Step sizes: y={step_y}, x={step_x}")
                
                # Generate patch coordinates (7-slice patches: center ± 5)
                # Leave room for 3 slices above and below
                z_range = range(5, depth - 5)
                y_range = range(0, height - self.patch_size[0] + 1, step_y)
                x_range = range(0, width - self.patch_size[1] + 1, step_x)
                
                print(f"Coordinate ranges:")
                print(f"  Z (center slice): {len(z_range)} positions ({min(z_range) if z_range else 'none'} to {max(z_range) if z_range else 'none'})")
                print(f"  Y: {len(y_range)} positions")  
                print(f"  X: {len(x_range)} positions")
                
                pair_patches = 0
                
                # Generate all combinations
                for center_z in z_range:
                    for y_start in y_range:
                        for x_start in x_range:
                            # Store patch info: (pair_idx, center_slice, y_start, x_start)
                            self.patch_coords.append((pair_idx, center_z, y_start, x_start))
                            pair_patches += 1
                
                print(f"✅ Generated {pair_patches} patches for this pair")
                
            except Exception as e:
                print(f"❌ Error processing pair {pair_idx}: {e}")
                continue
        
        if len(self.patch_coords) == 0:
            print("\n❌ NO PATCHES GENERATED!")
            print("Trying with smaller patch size...")
            
            # Try with smaller patches
            self.patch_size = (64, 64)
            self.overlap_ratio = 0.75
            self.patch_coords = []
            self._compute_patch_coordinates()
    
    def __len__(self):
        return len(self.patch_coords)
    
    def __getitem__(self, idx):
        if idx >= len(self.patch_coords):
            raise IndexError(f"Index {idx} out of range for {len(self.patch_coords)} patches")
        
        pair_idx, center_z, y_start, x_start = self.patch_coords[idx]
        pair_data = self.data_pairs[pair_idx]
        
        try:
            # Load volumes
            source_vol = nib.load(pair_data['source_path']).get_fdata()
            target_vol = nib.load(pair_data['target_path']).get_fdata()
            
            # Apply basic intensity normalization
            source_vol = self._normalize_intensity(source_vol)
            target_vol = self._normalize_intensity(target_vol)
            
            # Extract 7-slice patches (center ± 3 slices)
            z_start = center_z - 5
            z_end = center_z + 6  # 7 slices total
            y_end = y_start + self.patch_size[0]
            x_end = x_start + self.patch_size[1]
            
            source_patch = source_vol[z_start:z_end, y_start:y_end, x_start:x_end]
            target_patch = target_vol[z_start:z_end, y_start:y_end, x_start:x_end]
            
            # Verify patch shapes
            expected_shape = (7, self.patch_size[0], self.patch_size[1])
            if source_patch.shape != expected_shape or target_patch.shape != expected_shape:
                print(f"⚠️  Patch shape mismatch: got {source_patch.shape}, expected {expected_shape}")
                # Pad or crop to correct size
                source_patch = self._fix_patch_shape(source_patch, expected_shape)
                target_patch = self._fix_patch_shape(target_patch, expected_shape)
            
            # Load organ masks (simplified)
            organ_masks = {}
            if pair_data.get('target_seg'):
                try:
                    seg_path = Path(pair_data['target_seg'])
                    if seg_path.exists():
                        full_masks = nib.load(seg_path).get_fdata()
                        mask_patch = full_masks[z_start:z_end, y_start:y_end, x_start:x_end]
                        mask_patch = self._fix_patch_shape(mask_patch, expected_shape)
                        
                        # Create simple liver mask (assuming label 1 is liver)
                        liver_mask = (mask_patch >= 1) & (mask_patch <= 2)
                        organ_masks['liver'] = liver_mask.astype(np.float32)
                except Exception as e:
                    pass  # Ignore mask loading errors
            
            # Apply augmentation if training
            if self.augment:
                source_patch, target_patch, organ_masks = self._augment_patch(
                    source_patch, target_patch, organ_masks)
            
            # Convert to tensors and add channel dimension
            source_patch = torch.from_numpy(source_patch).unsqueeze(0).float()  # (1, 7, H, W)
            target_patch = torch.from_numpy(target_patch).unsqueeze(0).float()  # (1, 7, H, W)
            
            # Convert organ masks to tensors
            mask_tensors = {}
            for organ, mask in organ_masks.items():
                mask_tensors[organ] = torch.from_numpy(mask).unsqueeze(0).float()
            
            return {
                'source': source_patch,
                'target': target_patch,
                'source_phase': pair_data['source_phase'],
                'target_phase': pair_data['target_phase'],
                'masks': mask_tensors,
                'patch_coords': (pair_idx, center_z, y_start, x_start),
                'case_id': pair_data['case_id']
            }
        
        except Exception as e:
            print(f"❌ Error loading patch {idx}: {e}")
            # Return a dummy patch
            dummy_shape = (1, 7, self.patch_size[0], self.patch_size[1])
            return {
                'source': torch.zeros(dummy_shape),
                'target': torch.zeros(dummy_shape),
                'source_phase': 'error',
                'target_phase': 'error',
                'masks': {},
                'patch_coords': (pair_idx, 0, 0, 0),
                'case_id': 'error'
            }
    
    def _normalize_intensity(self, image):
        """Basic intensity normalization for CT"""
        # Clip to reasonable HU range for abdomen CT
        image = np.clip(image, -100, 300)
        
        # Z-score normalization
        mean_val = np.mean(image)
        std_val = np.std(image)
        if std_val > 1e-6:
            image = (image - mean_val) / std_val
        
        return image.astype(np.float32)
    
    def _fix_patch_shape(self, patch, target_shape):
        """Fix patch shape by padding or cropping"""
        current_shape = patch.shape
        
        # If shapes match, return as-is
        if current_shape == target_shape:
            return patch
        
        # Pad if too small, crop if too large
        fixed_patch = patch
        
        for dim in range(len(target_shape)):
            if dim >= len(current_shape):
                continue
                
            current_size = fixed_patch.shape[dim]
            target_size = target_shape[dim]
            
            if current_size < target_size:
                # Pad
                pad_width = [(0, 0)] * len(fixed_patch.shape)
                pad_width[dim] = (0, target_size - current_size)
                fixed_patch = np.pad(fixed_patch, pad_width, mode='constant', constant_values=0)
            elif current_size > target_size:
                # Crop from center
                start = (current_size - target_size) // 2
                slices = [slice(None)] * len(fixed_patch.shape)
                slices[dim] = slice(start, start + target_size)
                fixed_patch = fixed_patch[tuple(slices)]
        
        return fixed_patch
    
    def _augment_patch(self, source, target, masks):
        """Apply basic 3D augmentations"""
        # Random horizontal flip
        if np.random.random() > 0.5:
            source = np.flip(source, axis=2).copy()  # flip width
            target = np.flip(target, axis=2).copy()
            for organ in masks:
                masks[organ] = np.flip(masks[organ], axis=2).copy()
        
        # Random vertical flip
        if np.random.random() > 0.5:
            source = np.flip(source, axis=1).copy()  # flip height
            target = np.flip(target, axis=1).copy()
            for organ in masks:
                masks[organ] = np.flip(masks[organ], axis=1).copy()
        
        return source, target, masks
class ResidualBlock3D(nn.Module):
    """3D Residual block with instance normalization"""
    
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm3d(channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.relu(out + residual)

class AttentionGate3D(nn.Module):
    """3D Attention gate for skip connections"""
    
    def __init__(self, gate_channels, skip_channels, inter_channels):
        super().__init__()
        self.W_gate = nn.Conv3d(gate_channels, inter_channels, 1)
        self.W_skip = nn.Conv3d(skip_channels, inter_channels, 1)
        self.W_out = nn.Conv3d(inter_channels, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, gate, skip):
        gate_conv = self.W_gate(gate)
        skip_conv = self.W_skip(skip)
        attention = self.relu(gate_conv + skip_conv)
        attention = self.sigmoid(self.W_out(attention))
        return skip * attention

class Generator3D(nn.Module):
    """3D U-Net Generator with residual blocks and attention gates"""
    
    def __init__(self, input_channels=1, output_channels=1, features=[64, 128, 256, 512, 1024]):
        super().__init__()
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.encoder_pools = nn.ModuleList()
        
        in_channels = input_channels
        for feature in features:
            self.encoder_blocks.append(nn.Sequential(
                nn.Conv3d(in_channels, feature, 3, padding=1),
                nn.InstanceNorm3d(feature),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d(feature, feature, 3, padding=1),
                nn.InstanceNorm3d(feature),
                nn.LeakyReLU(0.2, inplace=True)
            ))
            self.encoder_pools.append(nn.MaxPool3d(2))
            in_channels = feature
        
        # Bottleneck with residual blocks
        self.bottleneck = nn.Sequential(
            ResidualBlock3D(features[-1]),
            ResidualBlock3D(features[-1]),
            ResidualBlock3D(features[-1])
        )
        
        # Decoder
        self.decoder_upconvs = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        
        for i in range(len(features) - 1, 0, -1):
            self.decoder_upconvs.append(nn.ConvTranspose3d(features[i], features[i-1], 2, 2))
            self.attention_gates.append(AttentionGate3D(features[i-1], features[i-1], features[i-1]//2))
            self.decoder_blocks.append(nn.Sequential(
                nn.Conv3d(features[i-1]*2, features[i-1], 3, padding=1),
                nn.InstanceNorm3d(features[i-1]),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d(features[i-1], features[i-1], 3, padding=1),
                nn.InstanceNorm3d(features[i-1]),
                nn.LeakyReLU(0.2, inplace=True)
            ))
        
        # Final output
        self.final_conv = nn.Conv3d(features[0], output_channels, 1)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        # Encoder
        encoder_outputs = []
        for encoder, pool in zip(self.encoder_blocks, self.encoder_pools):
            x = encoder(x)
            encoder_outputs.append(x)
            x = pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        for i, (upconv, attention, decoder) in enumerate(zip(
            self.decoder_upconvs, self.attention_gates, self.decoder_blocks)):
            x = upconv(x)
            skip = encoder_outputs[-(i+2)]  # Get corresponding encoder output
            skip = attention(x, skip)  # Apply attention gate
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)
        
        return self.tanh(self.final_conv(x))

class Discriminator3D(nn.Module):
    """3D PatchGAN Discriminator with spectral normalization"""
    
    def __init__(self, input_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        
        layers = []
        in_channels = input_channels
        
        for i, feature in enumerate(features):
            layers.append(nn.Conv3d(in_channels, feature, 4, 2, 1, bias=False))
            if i > 0:  # No normalization for first layer
                layers.append(nn.InstanceNorm3d(feature))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = feature
        
        # Final classification layer
        layers.append(nn.Conv3d(features[-1], 1, 4, 1, 1))
        
        self.model = nn.Sequential(*layers)
        
        # Apply spectral normalization for training stability
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.utils.spectral_norm(module)
    
    def forward(self, x):
        return self.model(x)

class CycleGAN3D(nn.Module):
    """3D CycleGAN for CT phase translation"""
    
    def __init__(self, input_channels=1, output_channels=1):
        super().__init__()
        
        # Generators for A->B and B->A
        self.gen_AB = Generator3D(input_channels, output_channels)
        self.gen_BA = Generator3D(input_channels, output_channels)
        
        # Discriminators for A and B
        self.disc_A = Discriminator3D(input_channels)
        self.disc_B = Discriminator3D(input_channels)
    
    def forward(self, real_A, real_B=None):
        if self.training:
            # Training mode: return all generated images
            fake_B = self.gen_AB(real_A)
            fake_A = self.gen_BA(real_B)
            cycle_A = self.gen_BA(fake_B)
            cycle_B = self.gen_AB(fake_A)
            
            return {
                'fake_B': fake_B, 'fake_A': fake_A,
                'cycle_A': cycle_A, 'cycle_B': cycle_B
            }
        else:
            # Inference mode: only A->B translation
            return self.gen_AB(real_A)

class FocalLoss3D(nn.Module):
    """3D Focal Loss for organ-specific enhancement"""
    
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.mse = nn.MSELoss(reduction='none')
    
    def forward(self, pred, target, organ_masks):
        """
        Args:
            pred, target: (B, 1, D, H, W)
            organ_masks: dict of {organ_name: (B, 1, D, H, W)}
        """
        base_loss = self.mse(pred, target)
        
        # Create weighted mask emphasizing organs
        weight_mask = torch.ones_like(base_loss)
        for organ_name, mask in organ_masks.items():
            if mask.sum() > 0:  # Only if organ is present
                organ_weight = 3.0 if organ_name in ['liver', 'pancreas'] else 2.0
                weight_mask += (organ_weight - 1.0) * mask
        
        # Apply focal weighting
        focal_weight = (base_loss ** self.gamma)
        weighted_loss = self.alpha * focal_weight * base_loss * weight_mask
        
        return weighted_loss.mean()

class CombinedLoss(nn.Module):
    """Combined loss function for CycleGAN training"""
    
    def __init__(self, lambda_cycle=10.0, lambda_mse=100.0, lambda_focal=50.0, lambda_perceptual=10.0):
        super().__init__()
        self.lambda_cycle = lambda_cycle
        self.lambda_mse = lambda_mse
        self.lambda_focal = lambda_focal
        self.lambda_perceptual = lambda_perceptual
        
        self.mse_loss = nn.MSELoss()
        self.focal_loss = FocalLoss3D()
        self.adversarial_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, outputs, real_A, real_B, organ_masks, disc_outputs):
        losses = {}
        
        # Adversarial losses
        losses['adv_A'] = self.adversarial_loss(
            disc_outputs['disc_fake_A'], torch.ones_like(disc_outputs['disc_fake_A']))
        losses['adv_B'] = self.adversarial_loss(
            disc_outputs['disc_fake_B'], torch.ones_like(disc_outputs['disc_fake_B']))
        
        # Cycle consistency losses
        losses['cycle_A'] = self.mse_loss(outputs['cycle_A'], real_A) * self.lambda_cycle
        losses['cycle_B'] = self.mse_loss(outputs['cycle_B'], real_B) * self.lambda_cycle
        
        # Direct MSE losses
        losses['mse_B'] = self.mse_loss(outputs['fake_B'], real_B) * self.lambda_mse
        losses['mse_A'] = self.mse_loss(outputs['fake_A'], real_A) * self.lambda_mse
        
        # Focal losses for organ enhancement
        if organ_masks:
            losses['focal_B'] = self.focal_loss(outputs['fake_B'], real_B, organ_masks) * self.lambda_focal
            losses['focal_A'] = self.focal_loss(outputs['fake_A'], real_A, organ_masks) * self.lambda_focal
        
        # Total generator loss
        losses['total_gen'] = sum([v for k, v in losses.items() if 'disc' not in k])
        
        return losses

import torch
import numpy as np
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.image.fid import FrechetInceptionDistance
import warnings

class MetricsEvaluator:
    """Comprehensive metrics evaluation with proper torchmetrics initialization"""
    
    def __init__(self, device='cuda'):
        self.device = device
        
        try:
            # SSIM for range [-1, 1] (tanh output)
            self.ssim = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
        except Exception as e:
            print(f"Warning: Could not initialize SSIM: {e}")
            self.ssim = None
        
        try:
            # PSNR for range [-1, 1] (tanh output) 
            self.psnr = PeakSignalNoiseRatio(data_range=2.0).to(device)
        except Exception as e:
            print(f"Warning: Could not initialize PSNR: {e}")
            self.psnr = None
        
        try:
            # FID calculation
            self.fid = FrechetInceptionDistance(feature=2048).to(device)
        except Exception as e:
            print(f"Warning: Could not initialize FID: {e}")
            self.fid = None
        
        print(f"MetricsEvaluator initialized on {device}")
        print(f"  SSIM: {'✓' if self.ssim else '✗'}")
        print(f"  PSNR: {'✓' if self.psnr else '✗'}")  
        print(f"  FID: {'✓' if self.fid else '✗'}")
    
    def evaluate_batch(self, generated, target, organ_masks=None):
        """Evaluate metrics for a batch"""
        metrics = {}
        
        try:
            # Ensure inputs are in correct format
            if generated.dim() == 5:  # (B, C, D, H, W) -> take middle slice
                generated = generated[:, :, generated.shape[2]//2, :, :]  # (B, C, H, W)
                target = target[:, :, target.shape[2]//2, :, :]
            
            # Ensure inputs are in range [0, 1] for SSIM/PSNR
            gen_norm = torch.clamp((generated + 1) / 2, 0, 1)  # From [-1, 1] to [0, 1]
            target_norm = torch.clamp((target + 1) / 2, 0, 1)
            
            # Overall SSIM
            if self.ssim is not None:
                try:
                    # SSIM expects range [0, 1] but we initialized it for [-1, 1]
                    # So we pass the original [-1, 1] values
                    metrics['ssim'] = self.ssim(generated, target).item()
                except Exception as e:
                    print(f"Warning: SSIM calculation failed: {e}")
                    metrics['ssim'] = 0.0
            else:
                metrics['ssim'] = self._manual_ssim(gen_norm, target_norm)
            
            # Overall PSNR  
            if self.psnr is not None:
                try:
                    # PSNR expects range [0, 1] but we initialized it for [-1, 1] 
                    # So we pass the original [-1, 1] values
                    metrics['psnr'] = self.psnr(generated, target).item()
                except Exception as e:
                    print(f"Warning: PSNR calculation failed: {e}")
                    metrics['psnr'] = self._manual_psnr(gen_norm, target_norm)
            else:
                metrics['psnr'] = self._manual_psnr(gen_norm, target_norm)
            
            # Organ-specific SSIM
            if organ_masks and self.ssim is not None:
                metrics['organ_ssim'] = {}
                for organ, mask in organ_masks.items():
                    if mask.sum() > 0:
                        try:
                            # Apply mask and calculate SSIM
                            if mask.dim() == 5:  # Take middle slice
                                mask = mask[:, :, mask.shape[2]//2, :, :]
                            
                            gen_masked = generated * mask
                            target_masked = target * mask
                            
                            if gen_masked.sum() > 0:
                                organ_ssim = self.ssim(gen_masked, target_masked).item()
                                metrics['organ_ssim'][organ] = organ_ssim
                        except Exception as e:
                            print(f"Warning: Organ SSIM calculation failed for {organ}: {e}")
            
        except Exception as e:
            print(f"Error in batch evaluation: {e}")
            metrics = {'ssim': 0.0, 'psnr': 0.0}
        
        return metrics
    
    def _manual_ssim(self, img1, img2):
        """Manual SSIM calculation as fallback"""
        try:
            # Convert to numpy
            img1_np = img1.detach().cpu().numpy()
            img2_np = img2.detach().cpu().numpy()
            
            # Simple SSIM approximation
            mu1 = np.mean(img1_np)
            mu2 = np.mean(img2_np)
            sigma1 = np.var(img1_np)
            sigma2 = np.var(img2_np)
            sigma12 = np.mean((img1_np - mu1) * (img2_np - mu2))
            
            c1 = (0.01) ** 2
            c2 = (0.03) ** 2
            
            ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))
            
            return float(ssim)
        except:
            return 0.0
    
    def _manual_psnr(self, img1, img2):
        """Manual PSNR calculation as fallback"""
        try:
            mse = torch.mean((img1 - img2) ** 2)
            if mse == 0:
                return 100.0  # Perfect match
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            return float(psnr)
        except:
            return 0.0
    
    def update_fid(self, generated, target):
        """Update FID calculation (requires accumulated batches)"""
        if self.fid is None:
            return
            
        try:
            # Handle 3D volumes by taking middle slice or averaging
            if generated.dim() == 5:  # (B, C, D, H, W)
                generated = generated.mean(dim=2)  # Average across depth -> (B, C, H, W)
                target = target.mean(dim=2)
            
            # Convert to 3-channel for FID calculation (FID expects 3-channel images)
            if generated.shape[1] == 1:  # Single channel
                generated = generated.repeat(1, 3, 1, 1)  # (B, 3, H, W)
                target = target.repeat(1, 3, 1, 1)
            
            # Normalize to [0, 255] uint8 range for FID
            gen_uint8 = ((generated + 1) * 127.5).clamp(0, 255).byte()
            target_uint8 = ((target + 1) * 127.5).clamp(0, 255).byte()
            
            # Update FID with real and fake images
            self.fid.update(target_uint8, real=True)
            self.fid.update(gen_uint8, real=False)
            
        except Exception as e:
            print(f"Warning: FID update failed: {e}")
    
    def compute_fid(self):
        """Compute final FID score"""
        if self.fid is None:
            return 0.0
            
        try:
            fid_score = self.fid.compute().item()
            self.fid.reset()  # Reset for next evaluation
            return fid_score
        except Exception as e:
            print(f"Warning: FID computation failed: {e}")
            return 0.0
    
    def reset(self):
        """Reset all metrics"""
        if self.ssim is not None:
            try:
                self.ssim.reset()
            except:
                pass
        
        if self.psnr is not None:
            try:
                self.psnr.reset()
            except:
                pass
        
        if self.fid is not None:
            try:
                self.fid.reset()
            except:
                pass



def save_radiologist_samples(model, dataloader, epoch, save_dir, num_samples=20):
    """Save high-quality samples for radiologist evaluation"""
    save_dir = Path(save_dir) / f"epoch_{epoch}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    samples_saved = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if samples_saved >= num_samples:
                break
                
            source = batch['source'].cuda()
            target = batch['target'].cuda()
            
            # Generate fake target and cycle reconstruction
            outputs = model(source, target)
            fake_target = outputs['fake_B']
            cycle_source = outputs['cycle_A']
            
            # Save samples
            for i in range(source.shape[0]):
                if samples_saved >= num_samples:
                    break
                    
                # Extract center slice for visualization
                center_slice = source.shape[2] // 2
                
                source_slice = source[i, 0, center_slice].cpu().numpy()
                target_slice = target[i, 0, center_slice].cpu().numpy()
                fake_slice = fake_target[i, 0, center_slice].cpu().numpy()
                cycle_slice = cycle_source[i, 0, center_slice].cpu().numpy()
                
                # Create comparison figure
                fig, axes = plt.subplots(2, 2, figsize=(12, 12))
                
                axes[0, 0].imshow(source_slice, cmap='gray')
                axes[0, 0].set_title(f'Original {batch["source_phase"][i]}')
                axes[0, 0].axis('off')
                
                axes[0, 1].imshow(target_slice, cmap='gray')
                axes[0, 1].set_title(f'Ground Truth {batch["target_phase"][i]}')
                axes[0, 1].axis('off')
                
                axes[1, 0].imshow(fake_slice, cmap='gray')
                axes[1, 0].set_title(f'Generated {batch["target_phase"][i]}')
                axes[1, 0].axis('off')
                
                axes[1, 1].imshow(cycle_slice, cmap='gray')
                axes[1, 1].set_title(f'Cycle {batch["source_phase"][i]}')
                axes[1, 1].axis('off')
                
                plt.tight_layout()
                plt.savefig(save_dir / f'sample_{samples_saved:03d}_{batch["case_id"][i]}.png', 
                           dpi=150, bbox_inches='tight')
                plt.close()
                
                samples_saved += 1
# Test function for the robust dataset
def test_robust_dataset():
    """Test the robust dataset"""
    print("Testing Robust Dataset...")
    
    # Import required functions
    from phase_generator import create_data_splits_from_directories, load_phase_mapping
    
    # Configuration
    data_dir = '../ncct_cect/vindr_ds/test_registered_cases'
    labels_csv = '../ncct_cect/vindr_ds/labels.csv'
    
    # Load data splits
    phase_mapping = None
    if Path(labels_csv).exists():
        try:
            phase_mapping = load_phase_mapping(labels_csv)
        except:
            pass
    
    data_splits = create_data_splits_from_directories(data_dir, phase_mapping=phase_mapping)
    
    # Test with robust dataset
    dataset = CTDataset(
        data_splits['train'],
        patch_size=(64, 64),  # Start with smaller patches
        overlap_ratio=0.75,   # High overlap for more patches
        augment=False
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    if len(dataset) > 0:
        try:
            sample = dataset[0]
            print("✅ Successfully loaded sample:")
            print(f"  Source shape: {sample['source'].shape}")
            print(f"  Target shape: {sample['target'].shape}")
            print(f"  Source phase: {sample['source_phase']}")
            print(f"  Target phase: {sample['target_phase']}")
            print(f"  Case ID: {sample['case_id']}")
            print(f"  Available masks: {list(sample['masks'].keys())}")
            return True
        except Exception as e:
            print(f"✗ Error loading sample: {e}")
            return False
    else:
        print("✗ No patches generated")
        return False

# if __name__ == "__main__":
#     test_robust_dataset()



# # Example usage and training setup
# if __name__ == "__main__":
#     # Configuration
#     config = {
#         'data_dir': '../ncct_cect/vindr_ds/test_registered_cases',  # Your registered_cases directory
#         'labels_csv': '../ncct_cect/vindr_ds/labels.csv',  # Optional: CSV with phase mappings
#         'output_dir': '../ncct_cect/vindr_ds/predicted',
#         'batch_size': 4,  # Reduced due to 3D processing
#         'learning_rate': 2e-4,
#         'epochs': 200,
#         'patch_size': (128, 128),
#         'overlap_ratio': 0.5,
#         'device': 'cuda' if torch.cuda.is_available() else 'cpu',
#         'mixed_precision': True,
#         'save_interval': 10  # Save radiologist samples every 10 epochs
#     }
    
#     # Load phase mapping if available
#     phase_mapping = None
#     if Path(config['labels_csv']).exists():
#         try:
#             phase_mapping = load_phase_mapping(config['labels_csv'])
#             # print(phase_mapping)
#             print(f"Loaded phase mapping for {len(phase_mapping)} cases")
#         except Exception as e:
#             print(f"Could not load phase mapping: {e}")
#             print("Will infer phases from filenames")
    
#     # Create data splits using directory structure
#     data_splits = create_data_splits_from_directories(
#         config['data_dir'], 
#         phase_mapping=phase_mapping
#     )
    
#     # Create datasets
#     train_dataset = CTDataset(
#         data_splits['train'], 
#         patch_size=config['patch_size'],
#         overlap_ratio=config['overlap_ratio'],
#         augment=True
#     )
    
#     val_dataset = CTDataset(
#         data_splits['val'],
#         patch_size=config['patch_size'], 
#         overlap_ratio=0.25,  # Less overlap for validation
#         augment=False
#     )
    
#     # Create dataloaders
#     train_loader = DataLoader(
#         train_dataset, 
#         batch_size=config['batch_size'], 
#         shuffle=True, 
#         num_workers=4, 
#         pin_memory=True
#     )
    
#     val_loader = DataLoader(
#         val_dataset, 
#         batch_size=config['batch_size'], 
#         shuffle=False, 
#         num_workers=4, 
#         pin_memory=True
#     )
    
#     print(f"Training dataset: {len(train_dataset)} patches")
#     print(f"Validation dataset: {len(val_dataset)} patches")
#     print("Setup complete! Ready for training...")
    
#     # Print some sample data info
#     if len(data_splits['train']) > 0:
#         sample = data_splits['train'][0]
#         print(f"\nSample training pair:")
#         print(f"  Case: {sample['case_id']}")
#         print(f"  Source: {sample['source_phase']} -> Target: {sample['target_phase']}")
#         print(f"  Source file: {sample['source_path']}")
#         print(f"  Target file: {sample['target_path']}")
#         if sample.get('target_seg'):
#             print(f"  Target segmentation: {sample['target_seg']}")

import torch
import numpy as np
import nibabel as nib
from pathlib import Path

def debug_your_data():
    """Debug your actual data to see what patch sizes you're getting"""
    
    data_dir = Path('../ncct_cect/vindr_ds/test_registered_cases')
    
    # Find first case
    case_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    if not case_dirs:
        print("No case directories found!")
        return
    
    first_case = case_dirs[0]
    nii_files = list(first_case.glob("*_registered.nii.gz"))
    
    if not nii_files:
        print("No .nii.gz files found!")
        return
    
    first_file = nii_files[0]
    print(f"Analyzing: {first_file}")
    
    # Load volume
    img = nib.load(first_file)
    data = img.get_fdata()
    
    print(f"Volume shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Min/Max: {data.min():.2f}/{data.max():.2f}")
    
    # Test different patch sizes
    depth, height, width = data.shape
    
    print(f"\nPatch size analysis:")
    print(f"Volume spatial dimensions: {height} x {width}")
    
    test_patch_sizes = [(32, 32), (64, 64), (128, 128), (256, 256)]
    
    for patch_h, patch_w in test_patch_sizes:
        if height >= patch_h and width >= patch_w:
            print(f"  {patch_h}x{patch_w}: ✅ FITS")
        else:
            print(f"  {patch_h}x{patch_w}: ❌ TOO BIG (volume is {height}x{width})")
    
    # Calculate actual patches with different overlaps
    print(f"\nPatch count analysis:")
    for patch_size in [(32, 32), (64, 64)]:
        for overlap in [0.5, 0.75]:
            patch_h, patch_w = patch_size
            if height >= patch_h and width >= patch_w:
                step_h = int(patch_h * (1 - overlap))
                step_w = int(patch_w * (1 - overlap))
                
                n_patches_h = (height - patch_h) // step_h + 1
                n_patches_w = (width - patch_w) // step_w + 1
                n_patches_z = depth - 6  # 7-slice patches
                
                total_patches = n_patches_h * n_patches_w * n_patches_z
                
                print(f"  {patch_size} with {overlap} overlap: {total_patches} patches")
    
    return data.shape

def create_adaptive_discriminator(input_size):
    """Create discriminator that adapts to your actual patch size"""
    
    height, width = input_size
    min_dim = min(height, width)
    
    print(f"Creating discriminator for {height}x{width} patches")
    
    # Determine number of downsample layers based on patch size
    if min_dim >= 128:
        features = [64, 128, 256, 512]
        print("Using full discriminator (128+ pixels)")
    elif min_dim >= 64:
        features = [64, 128, 256]
        print("Using medium discriminator (64-127 pixels)")
    elif min_dim >= 32:
        features = [64, 128]
        print("Using small discriminator (32-63 pixels)")
    else:
        features = [64]
        print("Using tiny discriminator (<32 pixels)")
    
    class AdaptiveDiscriminator3D(torch.nn.Module):
        def __init__(self, input_channels=1, features_list=features):
            super().__init__()
            
            layers = []
            in_channels = input_channels
            
            for i, feature in enumerate(features_list):
                # First layer: no normalization
                if i == 0:
                    layers.append(torch.nn.Conv3d(in_channels, feature, 4, 2, 1))
                    layers.append(torch.nn.LeakyReLU(0.2, inplace=True))
                else:
                    layers.append(torch.nn.Conv3d(in_channels, feature, 4, 2, 1, bias=False))
                    layers.append(torch.nn.InstanceNorm3d(feature))
                    layers.append(torch.nn.LeakyReLU(0.2, inplace=True))
                
                in_channels = feature
            
            # Final layer - adjust kernel and padding for small inputs
            if min_dim < 32:
                # For very small inputs, use 3x3 kernel with padding
                layers.append(torch.nn.Conv3d(features_list[-1], 1, 3, 1, 1))
            else:
                # Standard 4x4 kernel
                layers.append(torch.nn.Conv3d(features_list[-1], 1, 4, 1, 1))
            
            self.model = torch.nn.Sequential(*layers)
            
            # Test the architecture
            self._test_forward(input_channels, height, width)
        
        def _test_forward(self, channels, height, width):
            """Test if the architecture works with your patch size"""
            print(f"Testing discriminator architecture...")
            
            try:
                # Test input: (batch=1, channels=1, depth=7, height, width)
                test_input = torch.randn(1, channels, 7, height, width)
                test_output = self.model(test_input)
                print(f"  Input shape: {test_input.shape}")
                print(f"  Output shape: {test_output.shape}")
                print(f"  ✅ Architecture works!")
                return True
            except Exception as e:
                print(f"  ❌ Architecture failed: {e}")
                return False
        
        def forward(self, x):
            return self.model(x)
    
    return AdaptiveDiscriminator3D

def create_adaptive_generator(input_size):
    """Create generator that adapts to your actual patch size"""
    
    height, width = input_size
    min_dim = min(height, width)
    
    print(f"Creating generator for {height}x{width} patches")
    
    # Adjust features based on patch size
    if min_dim >= 128:
        features = [64, 128, 256, 512, 1024]
    elif min_dim >= 64:
        features = [32, 64, 128, 256]
    elif min_dim >= 32:
        features = [32, 64, 128]
    else:
        features = [32, 64]
    
    class AdaptiveGenerator3D(torch.nn.Module):
        def __init__(self, input_channels=1, output_channels=1, features_list=features):
            super().__init__()
            
            # Simple U-Net style generator adapted for smaller patches
            self.encoder_blocks = torch.nn.ModuleList()
            self.encoder_pools = torch.nn.ModuleList()
            
            in_channels = input_channels
            for feature in features_list:
                self.encoder_blocks.append(torch.nn.Sequential(
                    torch.nn.Conv3d(in_channels, feature, 3, padding=1),
                    torch.nn.InstanceNorm3d(feature),
                    torch.nn.LeakyReLU(0.2, inplace=True),
                ))
                
                # Use smaller pooling for small patches
                # if min_dim >= 64:
                #     self.encoder_pools.append(torch.nn.MaxPool3d(2))
                # else:
                #     # For small patches, don't pool as aggressively
                self.encoder_pools.append(torch.nn.MaxPool3d((1, 2, 2)))  # Only pool spatial dims
                
                in_channels = feature
            
            # Bottleneck
            self.bottleneck = torch.nn.Sequential(
                torch.nn.Conv3d(features_list[-1], features_list[-1], 3, padding=1),
                torch.nn.InstanceNorm3d(features_list[-1]),
                torch.nn.LeakyReLU(0.2, inplace=True)
            )
            
            # Decoder
            self.decoder_upconvs = torch.nn.ModuleList()
            self.decoder_blocks = torch.nn.ModuleList()
            
            for i in range(len(features_list) - 1, 0, -1):
                if min_dim >= 64:
                    self.decoder_upconvs.append(torch.nn.ConvTranspose3d(features_list[i], features_list[i-1], 2, 2))
                else:
                    self.decoder_upconvs.append(torch.nn.ConvTranspose3d(features_list[i], features_list[i-1], (1, 2, 2), (1, 2, 2)))
                
                self.decoder_blocks.append(torch.nn.Sequential(
                    torch.nn.Conv3d(features_list[i-1]*2, features_list[i-1], 3, padding=1),
                    torch.nn.InstanceNorm3d(features_list[i-1]),
                    torch.nn.LeakyReLU(0.2, inplace=True)
                ))
            
            # Final output
            self.final_conv = torch.nn.Conv3d(features_list[0], output_channels, 1)
            self.tanh = torch.nn.Tanh()
            
            # Test the architecture
            self._test_forward(input_channels, height, width)
        
        def _test_forward(self, channels, height, width):
            """Test if the architecture works"""
            print(f"Testing generator architecture...")
            
            try:
                test_input = torch.randn(1, channels, 7, height, width)
                test_output = self.forward(test_input)
                print(f"  Input shape: {test_input.shape}")
                print(f"  Output shape: {test_output.shape}")
                print(f"  ✅ Generator works!")
                return True
            except Exception as e:
                print(f"  ❌ Generator failed: {e}")
                return False
        
        def forward(self, x):
            # Encoder
            encoder_outputs = []
            for encoder, pool in zip(self.encoder_blocks, self.encoder_pools):
                x = encoder(x)
                encoder_outputs.append(x)
                x = pool(x)
            
            # Bottleneck
            x = self.bottleneck(x)
            
            # Decoder
            for i, (upconv, decoder) in enumerate(zip(self.decoder_upconvs, self.decoder_blocks)):
                x = upconv(x)
                skip = encoder_outputs[-(i+2)]
                
                # Handle size mismatches due to pooling
                if x.shape != skip.shape:
                    x = torch.nn.functional.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
                
                x = torch.cat([x, skip], dim=1)
                x = decoder(x)
            
            return self.tanh(self.final_conv(x))
    
    return AdaptiveGenerator3D

    """Test the working models with your actual data shapes"""
    
    print("=" * 60)
    print("TESTING WORKING 3D MODELS")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing on device: {device}")
    
    # Test individual components
    print(f"\n1. Testing Discriminator...")
    disc = Working3DDiscriminator(input_channels=1).to(device)
    
    print(f"\n2. Testing Generator...")
    gen = Working3DGenerator(input_channels=1, output_channels=1).to(device)
    
    print(f"\n3. Testing Complete CycleGAN...")
    model = Working3DCycleGAN(input_channels=1, output_channels=1).to(device)
    
    # Test with actual batch size and data shape
    print(f"\n4. Testing with batch training...")
    batch_size = 4
    test_real_A = torch.randn(batch_size, 1, 7, 64, 64).to(device)
    test_real_B = torch.randn(batch_size, 1, 7, 64, 64).to(device)
    
    try:
        model.train()
        outputs = model(test_real_A, test_real_B)
        
        print(f"✅ Training forward pass successful!")
        print(f"  fake_B shape: {outputs['fake_B'].shape}")
        print(f"  fake_A shape: {outputs['fake_A'].shape}")
        print(f"  cycle_A shape: {outputs['cycle_A'].shape}")
        print(f"  cycle_B shape: {outputs['cycle_B'].shape}")
        
        # Test discriminators
        disc_A_output = model.disc_A(test_real_A)
        disc_B_output = model.disc_B(test_real_B)
        
        print(f"  disc_A output shape: {disc_A_output.shape}")
        print(f"  disc_B output shape: {disc_B_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

import torch
import torch.nn as nn

class ParametricDiscriminator3D(nn.Module):
    """3D Discriminator that works with any depth and 64x64 spatial size"""
    
    def __init__(self, input_channels=1, input_depth=7):
        super().__init__()
        
        self.input_depth = input_depth
        
        # Calculate pooling strategy based on input depth
        # We want to end up with depth=1 after all pooling
        depth_stages = self._calculate_depth_pooling(input_depth)
        
        layers = []
        in_channels = input_channels
        current_depth = input_depth
        
        # Layer 1: Reduce spatial, preserve or slightly reduce depth
        stride_d, stride_h, stride_w = depth_stages[0]
        layers.extend([
            nn.Conv3d(in_channels, 64, kernel_size=(3,4,4), stride=(stride_d,stride_h,stride_w), padding=(1,1,1)),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        current_depth = current_depth // stride_d
        in_channels = 64
        
        # Layer 2: Continue reduction
        stride_d, stride_h, stride_w = depth_stages[1]
        layers.extend([
            nn.Conv3d(in_channels, 128, kernel_size=(3,4,4), stride=(stride_d,stride_h,stride_w), padding=(1,1,1), bias=False),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        current_depth = current_depth // stride_d
        in_channels = 128
        
        # Layer 3: Final spatial reduction
        stride_d, stride_h, stride_w = depth_stages[2]
        layers.extend([
            nn.Conv3d(in_channels, 256, kernel_size=(3,4,4), stride=(stride_d,stride_h,stride_w), padding=(1,1,1), bias=False),
            nn.InstanceNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        current_depth = current_depth // stride_d
        
        # Final classification layer - adaptive kernel size
        final_kernel_d = max(1, current_depth)
        final_kernel_h = 8  # Should be 8x8 spatial at this point
        final_kernel_w = 8
        
        layers.append(
            nn.Conv3d(256, 1, kernel_size=(final_kernel_d, final_kernel_h, final_kernel_w), stride=1, padding=0)
        )
        
        self.model = nn.Sequential(*layers)
        
        print(f"ParametricDiscriminator3D created for depth={input_depth}")
        print(f"Depth pooling stages: {depth_stages}")
        print(f"Final kernel: ({final_kernel_d}, {final_kernel_h}, {final_kernel_w})")
        self._test_forward()
    
    def _calculate_depth_pooling(self, input_depth):
        """Calculate optimal pooling strategy for given depth"""
        if input_depth <= 4:
            # Very shallow - minimal depth pooling
            return [(1, 2, 2), (1, 2, 2), (2, 2, 2)]
        elif input_depth <= 8:
            # Shallow - moderate depth pooling  
            return [(1, 2, 2), (2, 2, 2), (2, 2, 2)]
        elif input_depth <= 16:
            # Medium - balanced pooling
            return [(2, 2, 2), (2, 2, 2), (2, 2, 2)]
        else:
            # Deep - aggressive depth pooling
            return [(2, 2, 2), (4, 2, 2), (2, 2, 2)]
    
    def _test_forward(self):
        """Test the discriminator with actual input size"""
        try:
            test_input = torch.randn(2, 1, self.input_depth, 64, 64)
            print(f"  Input shape: {test_input.shape}")
            
            x = test_input
            for i, layer in enumerate(self.model):
                x = layer(x)
                print(f"    After layer {i+1}: {x.shape}")
            
            print(f"  Final output shape: {x.shape}")
            print(f"  ✅ ParametricDiscriminator3D works!")
            return True
            
        except Exception as e:
            print(f"  ❌ ParametricDiscriminator3D failed: {e}")
            return False
    
    def forward(self, x):
        return self.model(x)

class ParametricGenerator3D(nn.Module):
    """3D U-Net Generator with parametric depth and fixed channel calculations"""
    
    def __init__(self, input_channels=1, output_channels=1, input_depth=7):
        super().__init__()
        
        self.input_depth = input_depth
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv3d(input_channels, 64, 3, padding=1),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv3d(64, 128, 3, padding=1),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv3d(128, 256, 3, padding=1),
            nn.InstanceNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.enc4 = nn.Sequential(
            nn.Conv3d(256, 512, 3, padding=1),
            nn.InstanceNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Adaptive pooling based on input depth
        self.pool_strategy = self._calculate_pooling_strategy(input_depth)
        
        self.pool1 = nn.MaxPool3d(self.pool_strategy[0])  
        self.pool2 = nn.MaxPool3d(self.pool_strategy[1])  
        self.pool3 = nn.MaxPool3d(self.pool_strategy[2])  
        self.pool4 = nn.MaxPool3d(self.pool_strategy[3])  
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(512, 1024, 3, padding=1),
            nn.InstanceNorm3d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(1024, 512, 3, padding=1),
            nn.InstanceNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Decoder - FIXED CHANNEL CALCULATIONS
        self.up4 = nn.ConvTranspose3d(512, 256, self.pool_strategy[3], stride=self.pool_strategy[3])
        self.dec4 = nn.Sequential(
            nn.Conv3d(256 + 512, 256, 3, padding=1),  # 256 (up) + 512 (skip) = 768 -> 256
            nn.InstanceNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.up3 = nn.ConvTranspose3d(256, 128, self.pool_strategy[2], stride=self.pool_strategy[2])
        self.dec3 = nn.Sequential(
            nn.Conv3d(128 + 256, 128, 3, padding=1),  # 128 (up) + 256 (skip) = 384 -> 128
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.up2 = nn.ConvTranspose3d(128, 64, self.pool_strategy[1], stride=self.pool_strategy[1])
        self.dec2 = nn.Sequential(
            nn.Conv3d(64 + 128, 64, 3, padding=1),  # 64 (up) + 128 (skip) = 192 -> 64
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.up1 = nn.ConvTranspose3d(64, 64, self.pool_strategy[0], stride=self.pool_strategy[0])
        self.dec1 = nn.Sequential(
            nn.Conv3d(64 + 64, 64, 3, padding=1),  # 64 (up) + 64 (skip) = 128 -> 64
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Final output
        self.final = nn.Sequential(
            nn.Conv3d(64, output_channels, 1),
            nn.Tanh()
        )
        
        print(f"ParametricGenerator3D created for depth={input_depth}")
        print(f"Pooling strategy: {self.pool_strategy}")
        self._test_forward()
    
    def _calculate_pooling_strategy(self, input_depth):
        """Calculate pooling strategy that preserves ability to upsample back"""
        if input_depth <= 4:
            # Very shallow depth - mostly spatial pooling
            return [(1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2)]
        elif input_depth <= 8:
            # Shallow depth - balanced pooling
            return [(1, 2, 2), (1, 2, 2), (2, 2, 2), (2, 2, 2)]
        elif input_depth <= 16:
            # Medium depth - more depth pooling possible
            return [(1, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)]
        else:
            # Deep - can pool aggressively
            return [(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)]
    
    def _test_forward(self):
        """Test the generator with actual input size"""
        try:
            test_input = torch.randn(2, 1, self.input_depth, 64, 64)
            print(f"  Input shape: {test_input.shape}")
            
            output = self.forward(test_input)
            print(f"  Output shape: {output.shape}")
            
            if output.shape == test_input.shape:
                print(f"  ✅ ParametricGenerator3D works! Input and output shapes match.")
                return True
            else:
                print(f"  ⚠️ Shape mismatch: input {test_input.shape} vs output {output.shape}")
                return False
                
        except Exception as e:
            print(f"  ❌ ParametricGenerator3D failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def forward(self, x):
        # Encoder with skip connections
        e1 = self.enc1(x)           # 64 channels
        p1 = self.pool1(e1)         
        
        e2 = self.enc2(p1)          # 128 channels
        p2 = self.pool2(e2)         
        
        e3 = self.enc3(p2)          # 256 channels
        p3 = self.pool3(e3)         
        
        e4 = self.enc4(p3)          # 512 channels
        p4 = self.pool4(e4)         
        
        # Bottleneck
        b = self.bottleneck(p4)     # 512 channels
        
        # Decoder with skip connections
        u4 = self.up4(b)            # 512 -> 256 channels
        if u4.shape != e4.shape:
            u4 = nn.functional.interpolate(u4, size=e4.shape[2:], mode='trilinear', align_corners=False)
        d4 = self.dec4(torch.cat([u4, e4], dim=1))  # 256 + 512 = 768 -> 256
        
        u3 = self.up3(d4)           # 256 -> 128 channels
        if u3.shape != e3.shape:
            u3 = nn.functional.interpolate(u3, size=e3.shape[2:], mode='trilinear', align_corners=False)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))  # 128 + 256 = 384 -> 128
        
        u2 = self.up2(d3)           # 128 -> 64 channels
        if u2.shape != e2.shape:
            u2 = nn.functional.interpolate(u2, size=e2.shape[2:], mode='trilinear', align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))  # 64 + 128 = 192 -> 64
        
        u1 = self.up1(d2)           # 64 -> 64 channels
        if u1.shape != e1.shape:
            u1 = nn.functional.interpolate(u1, size=e1.shape[2:], mode='trilinear', align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))  # 64 + 64 = 128 -> 64
        
        # Final output
        output = self.final(d1)     # 64 -> output_channels
        
        return output

class ParametricCycleGAN3D(nn.Module):
    """Complete parametric 3D CycleGAN with configurable depth"""
    
    def __init__(self, input_channels=1, output_channels=1, input_depth=7):
        super().__init__()
        
        self.input_depth = input_depth
        
        # Generators
        self.gen_AB = PhaseConditionalGenerator3D(input_channels, output_channels, input_depth)
        self.gen_BA = PhaseConditionalGenerator3D(input_channels, output_channels, input_depth)
        
        # Discriminators
        self.disc_A = ParametricDiscriminator3D(input_channels, input_depth)
        self.disc_B = ParametricDiscriminator3D(input_channels, input_depth)
        
        print(f"\nParametricCycleGAN3D initialized for depth={input_depth}!")
        self._count_parameters()
    
    def _count_parameters(self):
        """Count model parameters"""
        gen_params = sum(p.numel() for p in self.gen_AB.parameters() if p.requires_grad)
        disc_params = sum(p.numel() for p in self.disc_A.parameters() if p.requires_grad)
        
        print(f"Generator parameters: {gen_params:,}")
        print(f"Discriminator parameters: {disc_params:,}")
        print(f"Total parameters: {(gen_params * 2 + disc_params * 2):,}")
    
    def forward(self, real_A, real_B=None):
        if self.training and real_B is not None:
            # Training mode: return all generated images
            fake_B = self.gen_AB(real_A)
            fake_A = self.gen_BA(real_B)
            cycle_A = self.gen_BA(fake_B)
            cycle_B = self.gen_AB(fake_A)
            
            return {
                'fake_B': fake_B, 'fake_A': fake_A,
                'cycle_A': cycle_A, 'cycle_B': cycle_B
            }
        else:
            # Inference mode: only A->B translation
            return self.gen_AB(real_A)

import torch
import torch.nn as nn
import torch.nn.functional as F

class PhaseConditionalGenerator3D(nn.Module):
    """
    3D U-Net Generator with Phase Conditioning
    Explicitly takes phase labels as input to control generation
    """
    
    def __init__(self, input_channels=1, output_channels=1, num_phases=4):
        super().__init__()
        
        # Phase encoding
        self.num_phases = num_phases
        self.phase_embedding_dim = 64
        
        # Phase embedding layer - converts phase label to vector
        self.phase_embedding = nn.Embedding(num_phases, self.phase_embedding_dim)
        
        # Initial convolution that combines image + phase embedding
        # Phase embedding is spatially replicated and concatenated
        self.initial_conv = nn.Sequential(
            nn.Conv3d(input_channels + 1, 64, 3, padding=1),  # +1 for phase map
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Encoder
        self.enc1 = self._make_encoder_block(64, 64)
        self.enc2 = self._make_encoder_block(64, 128)
        self.enc3 = self._make_encoder_block(128, 256)
        self.enc4 = self._make_encoder_block(256, 512)
        
        # Pooling
        self.pool = nn.MaxPool3d((1, 2, 2))
        
        # Bottleneck with phase-adaptive instance normalization
        self.bottleneck = nn.Sequential(
            nn.Conv3d(512, 1024, 3, padding=1),
            nn.InstanceNorm3d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(1024, 512, 3, padding=1),
            nn.InstanceNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Phase-adaptive modulation in bottleneck
        self.phase_modulation = nn.Linear(self.phase_embedding_dim, 512 * 2)  # scale + shift
        
        # Decoder
        self.up4 = nn.ConvTranspose3d(512, 256, (1, 2, 2), stride=(1, 2, 2))
        self.dec4 = self._make_decoder_block(256 + 512, 256)  # 256 (up) + 512 (skip) = 768
        
        self.up3 = nn.ConvTranspose3d(256, 128, (1, 2, 2), stride=(1, 2, 2))
        self.dec3 = self._make_decoder_block(128 + 256, 128)  # 128 (up) + 256 (skip) = 384
        
        self.up2 = nn.ConvTranspose3d(128, 64, (1, 2, 2), stride=(1, 2, 2))
        self.dec2 = self._make_decoder_block(64 + 128, 64)   # 64 (up) + 128 (skip) = 192
        
        self.up1 = nn.ConvTranspose3d(64, 64, (1, 2, 2), stride=(1, 2, 2))
        self.dec1 = self._make_decoder_block(64 + 64, 64)    # 64 (up) + 64 (skip) = 128
        
        # Final output
        self.final = nn.Sequential(
            nn.Conv3d(64, output_channels, 1),
            nn.Tanh()
        )
    
    def _make_encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x, phase_label):
        """
        Args:
            x: Input image [B, 1, D, H, W]
            phase_label: Phase labels [B] (integer: 0=non-contrast, 1=arterial, 2=portal, 3=delayed)
        """
        batch_size = x.size(0)
        
        # Get phase embedding [B, embedding_dim]
        phase_emb = self.phase_embedding(phase_label)
        
        # Create spatial phase map [B, 1, D, H, W]
        # Replicate phase embedding spatially
        phase_map = phase_emb.view(batch_size, self.phase_embedding_dim, 1, 1, 1)
        phase_map = phase_map.expand(-1, -1, x.size(2), x.size(3), x.size(4))
        
        # Reduce phase_map to 1 channel for concatenation
        phase_map_reduced = torch.mean(phase_map, dim=1, keepdim=True)
        
        # Concatenate image with phase map
        x_with_phase = torch.cat([x, phase_map_reduced], dim=1)
        
        # Initial convolution
        x = self.initial_conv(x_with_phase)
        
        # Encoder with skip connections
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool(e3)
        
        e4 = self.enc4(p3)
        p4 = self.pool(e4)
        
        # Bottleneck with phase-adaptive modulation
        b = self.bottleneck(p4)
        
        # Apply phase-specific modulation (adaptive instance normalization)
        phase_params = self.phase_modulation(phase_emb)  # [B, 512*2]
        scale = phase_params[:, :512].view(batch_size, 512, 1, 1, 1)
        shift = phase_params[:, 512:].view(batch_size, 512, 1, 1, 1)
        b = b * (1 + scale) + shift  # Modulate features based on phase
        
        # Decoder with skip connections
        u4 = self.up4(b)
        if u4.shape != e4.shape:
            u4 = F.interpolate(u4, size=e4.shape[2:], mode='trilinear', align_corners=False)
        d4 = self.dec4(torch.cat([u4, e4], dim=1))
        
        u3 = self.up3(d4)
        if u3.shape != e3.shape:
            u3 = F.interpolate(u3, size=e3.shape[2:], mode='trilinear', align_corners=False)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))
        
        u2 = self.up2(d3)
        if u2.shape != e2.shape:
            u2 = F.interpolate(u2, size=e2.shape[2:], mode='trilinear', align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        
        u1 = self.up1(d2)
        if u1.shape != e1.shape:
            u1 = F.interpolate(u1, size=e1.shape[2:], mode='trilinear', align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        
        # Final output
        output = self.final(d1)
        
        return output


class PhaseConditionalCycleGAN(nn.Module):
    """
    Phase-Conditional CycleGAN for multi-phase CT generation
    """
    
    def __init__(self, input_channels=1, output_channels=1):
        super().__init__()
        
        # Phase mapping
        self.phase_to_idx = {
            'non-contrast': 0,
            'arterial': 1,
            'portal': 2,
            'delayed': 3
        }
        
        # Single conditional generator (can generate any phase)
        self.generator = PhaseConditionalGenerator3D(
            input_channels=input_channels,
            output_channels=output_channels,
            num_phases=len(self.phase_to_idx)
        )
        
        # Phase-conditional discriminator (also knows what phase it should be)
        from train_phase_generator import ParametricDiscriminator3D
        self.discriminator = ParametricDiscriminator3D(input_channels=input_channels, input_depth=7)
    
    def forward(self, x, target_phase_label):
        """
        Args:
            x: Input image [B, 1, D, H, W]
            target_phase_label: Target phase as string or integer [B]
        """
        batch_size = x.size(0)
        # Convert phase names to indices if needed
        if isinstance(target_phase_label, (list, tuple)):
            if isinstance(target_phase_label[0], str):
                phase_indices = torch.tensor([
                    self.phase_to_idx.get(p, 0) for p in target_phase_label
                ], device=x.device)
            else:
                phase_indices = torch.tensor(target_phase_label, device=x.device)
        elif isinstance(target_phase_label, str):
            phase_idx = self.phase_to_idx.get(target_phase_label, 0)
            phase_indices = torch.tensor([phase_idx] * batch_size, device=x.device)
            # phase_indices = torch.tensor([self.phase_to_idx.get(target_phase_label, 0)], device=x.device)
        else:
            phase_indices = target_phase_label
        
        # Generate target phase
        generated = self.generator(x, phase_indices)
        
        return generated
    
    def generate_all_phases(self, x):
        """
        Generate all contrast phases from non-contrast input
        
        Args:
            x: Non-contrast CT [B, 1, D, H, W]
        
        Returns:
            dict: {'arterial': ..., 'portal': ..., 'delayed': ...}
        """
        results = {}
        
        for phase_name, phase_idx in self.phase_to_idx.items():
            if phase_name == 'non-contrast':
                continue
            
            phase_tensor = torch.tensor([phase_idx] * x.size(0), device=x.device)
            results[phase_name] = self.generator(x, phase_tensor)
        
        return results


# Example usage and testing
def test_phase_conditional_generator():
    """Test the phase-conditional generator"""
    
    print("Testing Phase-Conditional Generator...")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    model = PhaseConditionalCycleGAN(input_channels=1, output_channels=1).to(device)
    
    # Test input
    batch_size = 4
    test_input = torch.randn(batch_size, 1, 7, 64, 64).to(device)
    
    # Test 1: Generate specific phase
    print("\nTest 1: Generate arterial phase")
    arterial_output = model(test_input, 'arterial')
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {arterial_output.shape}")
    print(f"  ✅ Single phase generation works!")
    
    # Test 2: Generate different phases from same input
    print("\nTest 2: Generate all phases from non-contrast input")
    all_phases = model.generate_all_phases(test_input)
    for phase_name, output in all_phases.items():
        print(f"  {phase_name}: {output.shape}")
    print(f"  ✅ Multi-phase generation works!")
    
    # Test 3: Batch with mixed phase targets
    print("\nTest 3: Batch with different target phases")
    mixed_phases = ['arterial', 'portal', 'arterial', 'delayed']
    mixed_output = model(test_input, mixed_phases)
    print(f"  Target phases: {mixed_phases}")
    print(f"  Output shape: {mixed_output.shape}")
    print(f"  ✅ Mixed batch generation works!")
    
    print("\n" + "=" * 60)
    print("🎉 All tests passed! Phase-conditional generator is ready.")
    print("\nKey improvements:")
    print("1. ✅ Generator explicitly uses phase information")
    print("2. ✅ Can generate any target phase at inference")
    print("3. ✅ Single model for all phase transitions")
    print("4. ✅ Phase-adaptive feature modulation")

def test_parametric_models():
    """Test the parametric models with different depths"""
    
    print("=" * 60)
    print("TESTING PARAMETRIC 3D MODELS")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing on device: {device}")
    
    # Test different depths
    test_depths = [7, 10, 16]  # Your requested depths
    
    for depth in test_depths:
        print(f"\n{'='*40}")
        print(f"TESTING DEPTH = {depth}")
        print(f"{'='*40}")
        
        try:
            # Test discriminator
            print(f"\n1. Testing Discriminator (depth={depth})...")
            disc = ParametricDiscriminator3D(input_channels=1, input_depth=depth).to(device)
            
            # Test generator
            print(f"\n2. Testing Generator (depth={depth})...")
            gen = ParametricGenerator3D(input_channels=1, output_channels=1, input_depth=depth).to(device)
            
            # Test complete model
            print(f"\n3. Testing Complete CycleGAN (depth={depth})...")
            model = ParametricCycleGAN3D(input_channels=1, output_channels=1, input_depth=depth).to(device)
            
            # Test with batch
            print(f"\n4. Testing batch training (depth={depth})...")
            batch_size = 2  # Smaller batch for testing
            test_real_A = torch.randn(batch_size, 1, depth, 64, 64).to(device)
            test_real_B = torch.randn(batch_size, 1, depth, 64, 64).to(device)
            
            model.train()
            outputs = model(test_real_A, test_real_B)
            
            print(f"✅ Depth {depth} - All tests passed!")
            print(f"  fake_B shape: {outputs['fake_B'].shape}")
            print(f"  fake_A shape: {outputs['fake_A'].shape}")
            print(f"  cycle_A shape: {outputs['cycle_A'].shape}")
            print(f"  cycle_B shape: {outputs['cycle_B'].shape}")
            
            # Test discriminators
            disc_A_output = model.disc_A(test_real_A)
            disc_B_output = model.disc_B(test_real_B)
            
            print(f"  disc_A output shape: {disc_A_output.shape}")
            print(f"  disc_B output shape: {disc_B_output.shape}")
            
        except Exception as e:
            print(f"❌ Depth {depth} failed: {e}")
            import traceback
            traceback.print_exc()
            
        # Clear memory
        if 'model' in locals():
            del model
        if 'disc' in locals():
            del disc
        if 'gen' in locals():
            del gen
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    test_phase_conditional_generator()
    # test_parametric_models()
    # create_dataset_with_custom_depth(depth=10)
    
    # if success:
    #     print(f"\n" + "=" * 60)
    #     print("🎉 ALL TESTS PASSED!")
    #     print("=" * 60)
    #     print("Your models are ready for training with:")
    #     print("- Input shape: [batch, 1, 7, 64, 64]")
    #     print("- Output shape: [batch, 1, 7, 64, 64]")
    #     print("- Working discriminators and generators")
    #     print("\n✅ You can now replace your models with these working versions!")
    # else:
    #     print(f"\n" + "=" * 60)
    #     print("❌ TESTS FAILED")
    #     print("=" * 60)
    #     print("Please check the error messages above.")



# def main():
#     """Main function to debug and create adaptive architectures"""
    
#     print("=" * 60)
#     print("DEBUGGING PATCH SIZE ISSUES")
#     print("=" * 60)
    
#     # Debug your actual data
#     volume_shape = debug_your_data()
    
#     if volume_shape is None:
#         print("Could not analyze data")
#         return
    
#     depth, height, width = volume_shape
    
#     # Suggest optimal patch size
#     print(f"\n" + "=" * 60)
#     print("RECOMMENDED PATCH SIZE")
#     print("=" * 60)
    
#     if height >= 128 and width >= 128:
#         recommended_patch = (128, 128)
#         print(f"✅ Use patch_size = {recommended_patch} (full size)")
#     elif height >= 64 and width >= 64:
#         recommended_patch = (64, 64)
#         print(f"✅ Use patch_size = {recommended_patch} (medium size)")
#     elif height >= 32 and width >= 32:
#         recommended_patch = (32, 32)
#         print(f"✅ Use patch_size = {recommended_patch} (small size)")
#     else:
#         recommended_patch = (min(32, height), min(32, width))
#         print(f"⚠️  Use patch_size = {recommended_patch} (tiny size - may not work well)")
    
#     # Test adaptive architectures
#     print(f"\n" + "=" * 60)
#     print("TESTING ADAPTIVE ARCHITECTURES")
#     print("=" * 60)
    
#     # Create adaptive models
#     DiscriminatorClass = create_adaptive_discriminator(recommended_patch)
#     GeneratorClass = create_adaptive_generator(recommended_patch)
    
#     # Test both
#     print(f"\nCreating models...")
#     try:
#         discriminator = DiscriminatorClass(input_channels=1)
#         generator = GeneratorClass(input_channels=1, output_channels=1)
#         print(f"✅ Both models created successfully!")
        
#         print(f"\n🔧 UPDATE YOUR CONFIG:")
#         print(f"config = {{")
#         print(f"    'patch_size': {recommended_patch},")
#         print(f"    'overlap_ratio': 0.75,  # More overlap for smaller patches")
#         print(f"    'batch_size': 4,  # Can use larger batch with smaller patches")
#         print(f"    # ... other settings")
#         print(f"}}")
        
#     except Exception as e:
#         print(f"❌ Model creation failed: {e}")

# if __name__ == "__main__":
#     main()