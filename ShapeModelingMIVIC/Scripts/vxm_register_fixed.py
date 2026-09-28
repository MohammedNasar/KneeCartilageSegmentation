#!/usr/bin/env python3
"""
VoxelMorph Registration Wrapper with Compatibility Fix
Alternative to modifying the model file - wraps the loading process
"""

import sys
import argparse
import numpy as np
import voxelmorph as vxm
import tensorflow as tf

def load_model_with_fix(model_path):
    """
    Load VoxelMorph model with compatibility fix for atlas_feats parameter
    """
    import h5py
    import json
    import tempfile
    import shutil
    
    print(f"Loading model with compatibility fix: {model_path}")
    
    # Create a temporary fixed model
    temp_model = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
    temp_model.close()
    
    try:
        # Read and fix the config
        with h5py.File(model_path, 'r') as f_in:
            # Get the config
            if 'model_config' in f_in.attrs:
                config_str = f_in.attrs['model_config']
                if isinstance(config_str, bytes):
                    config_str = config_str.decode('utf-8')
                
                config = json.loads(config_str)
                
                # Remove problematic parameters
                if 'config' in config and isinstance(config['config'], dict):
                    # List of parameters not supported in VoxelMorph 0.2+
                    params_to_remove = [
                        'atlas_feats', 
                        'atlas_feat_device',
                        'mean_cap',
                        'use_probs',
                        'nb_conv_per_level',
                        'conv_image_shape'
                    ]
                    for param in params_to_remove:
                        if param in config['config']:
                            print(f"Removing incompatible parameter: {param}")
                            del config['config'][param]
                
                # Write temporary fixed model
                shutil.copy2(model_path, temp_model.name)
                
                with h5py.File(temp_model.name, 'r+') as f_out:
                    # Update the model_config
                    del f_out.attrs['model_config']
                    f_out.attrs['model_config'] = json.dumps(config)
                
                # Load the fixed model
                model = vxm.networks.VxmDense.load(temp_model.name)
                print("✓ Model loaded successfully with compatibility fix")
                return model
            else:
                # No config to fix, load normally
                return vxm.networks.VxmDense.load(model_path)
                
    finally:
        # Clean up temp file
        import os
        if os.path.exists(temp_model.name):
            os.unlink(temp_model.name)

def register_images(fixed_path, moving_path, model_path, 
                   moved_path, warp_path, 
                   moving_labels_path=None, moved_labels_path=None,
                   gpu=0):
    """
    Register moving image to fixed image using VoxelMorph
    """
    
    # Set GPU
    if gpu >= 0:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.set_visible_devices(gpus[gpu], 'GPU')
                print(f"Using GPU {gpu}: {gpus[gpu].name}")
            except RuntimeError as e:
                print(f"GPU setup error: {e}")
    
    # Load images
    print(f"Loading fixed image: {fixed_path}")
    fixed = vxm.py.utils.load_volfile(fixed_path)
    
    print(f"Loading moving image: {moving_path}")
    moving = vxm.py.utils.load_volfile(moving_path)
    
    # Add batch and channel dimensions
    fixed_input = fixed[np.newaxis, ..., np.newaxis]
    moving_input = moving[np.newaxis, ..., np.newaxis]
    
    # Load model with fix
    model = load_model_with_fix(model_path)
    
    # Perform registration
    print("Performing registration...")
    moved, warp = model.register(moving_input, fixed_input)
    
    # Remove batch dimension
    moved = np.squeeze(moved)
    warp = np.squeeze(warp)
    
    # Save results
    print(f"Saving moved image: {moved_path}")
    vxm.py.utils.save_volfile(moved, moved_path)
    
    print(f"Saving warp field: {warp_path}")
    vxm.py.utils.save_volfile(warp, warp_path)
    
    # Warp labels if provided
    if moving_labels_path and moved_labels_path:
        print(f"Loading moving labels: {moving_labels_path}")
        moving_labels = vxm.py.utils.load_volfile(moving_labels_path)
        
        print("Warping labels...")
        # Apply the warp field to labels using nearest neighbor
        moved_labels = vxm.networks.Transform(
            moving_labels.shape, 
            interp_method='nearest'
        ).predict([moving_labels[np.newaxis, ..., np.newaxis], warp[np.newaxis, ...]])
        
        moved_labels = np.squeeze(moved_labels)
        
        print(f"Saving moved labels: {moved_labels_path}")
        vxm.py.utils.save_volfile(moved_labels, moved_labels_path)
    
    print("✓ Registration complete!")

def main():
    parser = argparse.ArgumentParser(description='VoxelMorph Registration with Compatibility Fix')
    parser.add_argument('--fixed', required=True, help='Fixed image path')
    parser.add_argument('--moving', required=True, help='Moving image path')
    parser.add_argument('--model', required=True, help='VoxelMorph model path')
    parser.add_argument('--moved', required=True, help='Output moved image path')
    parser.add_argument('--warp', required=True, help='Output warp field path')
    parser.add_argument('--moving_labels', help='Moving labels path (optional)')
    parser.add_argument('--moved_labels', help='Output moved labels path (optional)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device number (default: 0)')
    
    args = parser.parse_args()
    
    try:
        register_images(
            args.fixed, args.moving, args.model,
            args.moved, args.warp,
            args.moving_labels, args.moved_labels,
            args.gpu
        )
        return 0
    except Exception as e:
        print(f"Error during registration: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
