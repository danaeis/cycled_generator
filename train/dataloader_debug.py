# ============================================================================
# DEBUGGING UTILITIES
# ============================================================================

def debug_data_loading(config: Dict):
    """Debug data loading to ensure everything works."""
    
    print("=" * 80)
    print("DEBUGGING DATA LOADING")
    print("=" * 80)
    
    # Load phase mapping
    phase_mapping = None
    if Path(config['labels_csv']).exists():
        try:
            phase_mapping = load_phase_mapping(config['labels_csv'])
            print(f"✓ Loaded phase mapping for {len(phase_mapping)} cases")
        except Exception as e:
            print(f"✗ Could not load phase mapping: {e}")
    
    # Create data pairs
    try:
        data_splits = create_data_pairs(
            config['data_dir'],
            phase_mapping=phase_mapping
        )
        print(f"✓ Created data splits successfully")
        
        # Print sample pair
        if data_splits['train']:
            sample = data_splits['train'][0]
            print(f"\nSample training pair:")
            print(f"  Case: {sample['case_id']}")
            print(f"  Source: {sample['source_phase']} -> {sample['source_path'].name}")
            print(f"  Target: {sample['target_phase']} -> {sample['target_path'].name}")
            if sample.get('target_seg'):
                print(f"  Has segmentation: ✓")
        
        return data_splits
        
    except Exception as e:
        print(f"✗ Error creating data splits: {e}")
        import traceback
        traceback.print_exc()
        return None

def debug_dataset(dataset, num_samples: int = 3):
    """Debug dataset by loading a few samples."""
    
    print("\n" + "=" * 80)
    print("DEBUGGING DATASET")
    print("=" * 80)
    
    print(f"Dataset size: {len(dataset)} patches")
    print(f"Testing first {num_samples} samples...")
    
    for i in range(min(num_samples, len(dataset))):
        try:
            sample = dataset[i]
            print(f"\nSample {i+1}:")
            print(f"  Source shape: {sample['source'].shape}")
            print(f"  Target shape: {sample['target'].shape}")
            print(f"  Source phase: {sample['source_phase'].item()}")
            print(f"  Target phase: {sample['target_phase'].item()}")
            print(f"  Case ID: {sample['case_id']}")
            print(f"  ✓ Sample loaded successfully")
            
        except Exception as e:
            print(f"  ✗ Error loading sample {i+1}: {e}")
            import traceback
            traceback.print_exc()
    
    # Test cache statistics if available
    if hasattr(dataset, 'get_cache_stats'):
        cache_stats = dataset.get_cache_stats()
        print(f"\n" + "=" * 80)
        print("CACHE STATISTICS")
        print("=" * 80)
        print(f"Cache size: {cache_stats['cache_size']} volumes")
        print(f"Cache hits: {cache_stats['cache_hits']}")
        print(f"Cache misses: {cache_stats['cache_misses']}")
        print(f"Hit rate: {cache_stats['hit_rate']:.2%}")
        print(f"Disk I/O reduction: ~{cache_stats['hit_rate'] * 100:.1f}%")

def debug_dataset_with_visualization(dataset, num_samples: int = 5, output_dir: str = './debug_patches'):
    """
    Debug dataset with visualization of patches.
    
    Args:
        dataset: CTPhaseDataset instance
        num_samples: Number of samples to visualize
        output_dir: Directory to save visualizations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("DEBUGGING DATASET WITH VISUALIZATION")
    print("=" * 80)
    
    print(f"Dataset size: {len(dataset)} patches")
    print(f"Visualizing {num_samples} samples...")
    print(f"Output directory: {output_dir}")
    
    for i in range(min(num_samples, len(dataset))):
        try:
            sample = dataset[i]
            
            # Get middle slice from each volume
            source_vol = sample['source'][0].numpy()  # Remove channel dim
            target_vol = sample['target'][0].numpy()
            mid_slice = source_vol.shape[0] // 2
            
            # Create visualization
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            axes[0].imshow(source_vol[mid_slice], cmap='gray', vmin=-1, vmax=1)
            axes[0].set_title(f'Source (Phase {sample["source_phase"].item()})')
            axes[0].axis('off')
            
            axes[1].imshow(target_vol[mid_slice], cmap='gray', vmin=-1, vmax=1)
            axes[1].set_title(f'Target (Phase {sample["target_phase"].item()})')
            axes[1].axis('off')
            
            plt.suptitle(f'Patch {i} - Case: {sample["case_id"]}')
            plt.tight_layout()
            plt.savefig(output_dir / f'patch_{i:03d}_case_{sample["case_id"]}.png', 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Sample {i+1} visualized")
            
        except Exception as e:
            print(f"  ✗ Error visualizing sample {i+1}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✓ Visualizations saved to {output_dir}")
    
    # Test cache statistics
    if hasattr(dataset, 'get_cache_stats'):
        cache_stats = dataset.get_cache_stats()
        print(f"\n" + "=" * 80)
        print("CACHE STATISTICS AFTER LOADING")
        print("=" * 80)
        print(f"Cache size: {cache_stats['cache_size']} volumes")
        print(f"Cache hits: {cache_stats['cache_hits']}")
        print(f"Cache misses: {cache_stats['cache_misses']}")
        print(f"Hit rate: {cache_stats['hit_rate']:.2%}")

def create_patch_coverage_map(dataset, case_idx: int = 0, output_dir: str = './debug_patches'):
    """
    Create a coverage map showing where patches are extracted from.
    
    Args:
        dataset: CTPhaseDataset instance
        case_idx: Index of case to visualize
        output_dir: Directory to save visualization
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all patches for this case
    case_patches = [coord for coord in dataset.patch_coords if coord['pair_idx'] == case_idx]
    
    if not case_patches:
        print(f"No patches found for case {case_idx}")
        return
    
    # Get volume shape
    pair_data = dataset.data_pairs[case_idx]
    if hasattr(dataset, 'volume_cache'):
        shape = dataset.volume_cache.get_volume_shape(pair_data['source_path'])
    else:
        import nibabel as nib
        nii = nib.load(pair_data['source_path'])
        shape = (nii.shape[2], nii.shape[1], nii.shape[0])
    
    # Create coverage map
    coverage = np.zeros((shape[1], shape[2]))  # H x W
    
    for coord in case_patches:
        y, x = coord['y'], coord['x']
        h, w = dataset.patch_size
        coverage[y:y+h, x:x+w] += 1
    
    # Visualize
    plt.figure(figsize=(12, 8))
    plt.imshow(coverage, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Patch Coverage Count')
    plt.title(f'Patch Coverage Map - Case {pair_data["case_id"]}\n'
             f'Total patches: {len(case_patches)}')
    plt.xlabel('Width')
    plt.ylabel('Height')
    
    # Mark patch centers
    for coord in case_patches[::max(1, len(case_patches)//20)]:  # Show subset of centers
        y = coord['y'] + dataset.patch_size[0] // 2
        x = coord['x'] + dataset.patch_size[1] // 2
        plt.plot(x, y, 'b.', markersize=3, alpha=0.5)
    
    plt.savefig(output_dir / f'patch_coverage_case_{case_idx}.png', 
               dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Coverage map saved for case {case_idx}")
    print(f"  Total patches: {len(case_patches)}")
    print(f"  Max coverage: {int(coverage.max())}x")

def debug_model(config: Dict):
    """Test model architecture."""
    from training_phase_gen_optimized import Generator3D, Discriminator3D
    
    print("\n" + "=" * 80)
    print("DEBUGGING MODEL ARCHITECTURE")
    print("=" * 80)
    
    device = config['device']
    
    try:
        # Create test input
        batch_size = 2
        test_input = torch.randn(
            batch_size, 1, 
            config['patch_depth'], 
            config['patch_size'][0], 
            config['patch_size'][1]
        ).to(device)
        test_phase = torch.tensor([1, 2]).to(device)
        
        # Test generator
        generator = Generator3D().to(device)
        
        print(f"Testing Generator:")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Phase indices: {test_phase}")
        
        output = generator(test_input, test_phase)
        print(f"  Output shape: {output.shape}")
        print(f"  ✓ Generator works!")
        
        # Test discriminator
        discriminator = Discriminator3D().to(device)
        disc_output = discriminator(output)
        print(f"\nTesting Discriminator:")
        print(f"  Discriminator output shape: {disc_output.shape}")
        print(f"  ✓ Discriminator works!")
        
        # Test full forward pass
        print(f"\nTesting full forward pass:")
        generated = generator(test_input, test_phase)
        reconstructed = generator(generated, torch.tensor([0, 0]).to(device))
        
        print(f"  Source -> Target: {test_input.shape} -> {generated.shape}")
        print(f"  Target -> Reconstructed: {generated.shape} -> {reconstructed.shape}")
        print(f"  ✓ Cycle consistency works!")
        
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
