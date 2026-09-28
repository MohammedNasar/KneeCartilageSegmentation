#!/usr/bin/env python3
"""
VoxelMorph Model Compatibility Fixer
Diagnoses and fixes the 'atlas_feats' parameter issue
"""

import h5py
import json
import sys

def diagnose_model(model_path):
    """Check what parameters are stored in the model"""
    print(f"Analyzing model: {model_path}")
    print("=" * 60)
    
    try:
        with h5py.File(model_path, 'r') as f:
            # Check if config exists
            if 'model_config' in f.attrs:
                config_str = f.attrs['model_config']
                if isinstance(config_str, bytes):
                    config_str = config_str.decode('utf-8')
                
                config = json.loads(config_str)
                print("\nModel Configuration:")
                print(json.dumps(config, indent=2))
                
                # Check for problematic parameters
                if 'config' in config and isinstance(config['config'], dict):
                    # Parameters not supported in VoxelMorph 0.2+
                    problematic_params = [
                        'atlas_feats', 
                        'atlas_feat_device',
                        'mean_cap',
                        'use_probs',
                        'nb_conv_per_level',
                        'conv_image_shape'
                    ]
                    found_issues = []
                    
                    for param in problematic_params:
                        if param in config['config']:
                            found_issues.append(param)
                    
                    if found_issues:
                        print(f"\n⚠ Found problematic parameters: {found_issues}")
                        print("These parameters are not compatible with current VoxelMorph version")
                        return config, found_issues
                    else:
                        print("\n✓ No problematic parameters found")
                        return config, []
            else:
                print("No model_config found in HDF5 file")
                return None, []
                
    except Exception as e:
        print(f"Error reading model: {e}")
        return None, []

def fix_model_config(model_path, output_path=None):
    """Remove problematic parameters from model config"""
    
    if output_path is None:
        output_path = model_path.replace('.h5', '_fixed.h5')
    
    print(f"\nAttempting to fix model...")
    print(f"Input:  {model_path}")
    print(f"Output: {output_path}")
    
    try:
        with h5py.File(model_path, 'r') as f_in:
            if 'model_config' not in f_in.attrs:
                print("No model_config to fix")
                return False
            
            config_str = f_in.attrs['model_config']
            if isinstance(config_str, bytes):
                config_str = config_str.decode('utf-8')
            
            config = json.loads(config_str)
            
            # Remove problematic parameters
            removed = []
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
                        del config['config'][param]
                        removed.append(param)
            
            if not removed:
                print("No parameters to remove")
                return False
            
            # Write fixed model
            with h5py.File(output_path, 'w') as f_out:
                # Copy all datasets and groups
                def copy_item(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        f_in.copy(name, f_out)
                    elif isinstance(obj, h5py.Group):
                        f_out.create_group(name)
                
                f_in.visititems(copy_item)
                
                # Copy all attributes except model_config
                for key in f_in.attrs:
                    if key != 'model_config':
                        f_out.attrs[key] = f_in.attrs[key]
                
                # Write fixed model_config
                f_out.attrs['model_config'] = json.dumps(config)
            
            print(f"\n✓ Successfully removed parameters: {removed}")
            print(f"✓ Fixed model saved to: {output_path}")
            return True
            
    except Exception as e:
        print(f"Error fixing model: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_vxm_model.py <model_path> [output_path]")
        print("\nExample:")
        print("  python fix_vxm_model.py /path/to/model/0960.h5")
        print("  python fix_vxm_model.py /path/to/model/0960.h5 /path/to/fixed/0960_fixed.h5")
        sys.exit(1)
    
    model_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Diagnose
    config, issues = diagnose_model(model_path)
    
    # Fix if needed
    if issues:
        print("\n" + "=" * 60)
        success = fix_model_config(model_path, output_path)
        
        if success:
            print("\n" + "=" * 60)
            print("NEXT STEPS:")
            print("=" * 60)
            print(f"1. Test the fixed model:")
            print(f"   python vxm_register.py --model {output_path or model_path.replace('.h5', '_fixed.h5')} ...")
            print(f"\n2. If it works, replace the original:")
            print(f"   mv {output_path or model_path.replace('.h5', '_fixed.h5')} {model_path}")
            print(f"\n3. Or update your script to use the fixed model")
    else:
        print("\n✓ Model appears to be compatible")

if __name__ == "__main__":
    main()
