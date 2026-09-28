"""
Verify that predicted segmentations overlay correctly with input images.
"""

import nibabel as nib
import numpy as np
import os

def verify_alignment(input_path, pred_path):
    """
    Check if prediction has same metadata as input.
    """
    input_nii = nib.load(input_path)
    pred_nii = nib.load(pred_path)
    
    print(f"\n{'='*60}")
    print(f"Checking alignment:")
    print(f"  Input: {os.path.basename(input_path)}")
    print(f"  Pred:  {os.path.basename(pred_path)}")
    print(f"{'='*60}")
    
    # Check shapes
    print(f"\nShapes:")
    print(f"  Input: {input_nii.shape}")
    print(f"  Pred:  {pred_nii.shape}")
    match_shape = input_nii.shape == pred_nii.shape
    print(f"  Match: {' ✓' if match_shape else '✗ MISMATCH'}")
    
    # Check affines
    print(f"\nAffine matrices:")
    print(f"  Input:\n{input_nii.affine}")
    print(f"  Pred:\n{pred_nii.affine}")
    match_affine = np.allclose(input_nii.affine, pred_nii.affine, atol=1e-3)
    print(f"  Match: {'✓' if match_affine else '✗ MISMATCH'}")
    
    # Check voxel spacing
    input_spacing = input_nii.header.get_zooms()[:3]
    pred_spacing = pred_nii.header.get_zooms()[:3]
    print(f"\nVoxel spacing:")
    print(f"  Input: {input_spacing}")
    print(f"  Pred:  {pred_spacing}")
    match_spacing = np.allclose(input_spacing, pred_spacing, atol=1e-3)
    print(f"  Match: {'✓' if match_spacing else '✗ MISMATCH'}")
    
    # Overall
    print(f"\n{'='*60}")
    if match_shape and match_affine and match_spacing:
        print("✅ PERFECT ALIGNMENT - Images will overlay correctly!")
    else:
        print("❌ MISALIGNMENT - Overlay will NOT work properly!")
    print(f"{'='*60}\n")
    
    return match_shape and match_affine and match_spacing


# Example usage
if __name__ == "__main__":
    input_img = "C:\\Users\\mnibrahim\\Documents\\Projects\\KneeCartilage\\Data\\SBIR_GrantData\\Images_half\\1\\participant_5.nii"
    pred_seg = "C:\\Users\\mnibrahim\\Documents\\Projects\\KneeCartilage\\Code\\Deep_Nets_Segmentation_MIVIC\\3d_segmentation\\3dunet_B12\\output\\participant_5_label_pred.nii.gz"
    
    verify_alignment(input_img, pred_seg)