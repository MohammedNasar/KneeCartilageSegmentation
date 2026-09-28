# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 12:21:00 2024

@author: lliu
"""

import os
import nibabel as nib
import numpy as np

def align_images_in_folder(folder_path, template_image_path, output_folder):
    #example
    #folder_path = r'C:\Users\lliu\OneDrive - Delaware State University\DesktopLab\BLSA_fs_copy'
    #template_image_path = r'C:\Users\lliu\OneDrive - Delaware State University\DesktopLab\TIDAQ\template\template_int16_V7_fold1.nii'
    #output_folder = r'C:\Users\lliu\OneDrive - Delaware State University\DesktopLab\BLSA_TEMPLATE'

    # List all files in the folder
    files = os.listdir(folder_path)

    # Filter the list to only include NIfTI files
    nifti_files = [file for file in files if file.endswith('.nii')]

    # Load the template MRI image
    template_img = nib.load(template_image_path)
    template_affine = template_img.affine

    for nifti_file in nifti_files:
        # Construct the full file path to the NIfTI file
        file_path = os.path.join(folder_path, nifti_file)

        # Load the target MRI image
        target_img = nib.load(file_path)
        target_data = target_img.get_fdata()
        target_affine = target_img.affine

        # Calculate the center of the target image in voxel coordinates
        target_center_voxel = np.array(target_data.shape) / 2

        # Calculate the center of the template image in voxel coordinates
        template_center_voxel = np.array(template_img.shape) / 2

        # Convert voxel centers to world coordinates using the respective affine matrices
        target_center_world = nib.affines.apply_affine(target_affine, target_center_voxel)
        template_center_world = nib.affines.apply_affine(template_affine, template_center_voxel)

        # Calculate the offset between the target center and the template center
        offset = template_center_world - target_center_world

        # Adjust the target affine matrix to align its center with the template center
        new_affine = target_affine.copy()
        new_affine[:3, 3] += offset

        # Create a new nibabel image using the modified data and affine matrix
        aligned_image = nib.Nifti1Image(target_data, new_affine)

        # Set the file path for the output image
        output_image_path = os.path.join(output_folder, f"{nifti_file[:-4]}_sp.nii")

        # Save the aligned image to a new file
        nib.save(aligned_image, output_image_path)

        print(f"Aligned image saved to {output_image_path}")
