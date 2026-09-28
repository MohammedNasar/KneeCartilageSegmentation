#!/usr/bin/env python
# coding: utf-8
"""
Apply intensity normalization to all images in data path.

@author: S. Makrogiannis, MIVIC/PEMaCS/DSU (smakrogiannis@desu.edu)
"""

# Import modules.
import os,sys,glob
import itk
import numpy as np
import matplotlib.pyplot as plt


# Set data path and display all nifti files therein.
data_path = './'
all_nii_files = glob.glob(data_path + '*.nii')
print(all_nii_files)


# For all nii files: read file, apply intensity normalization, and write image to new file.
for filename in all_nii_files:
    print(filename)
    input_image = itk.imread(filename)
    input_image_np = itk.GetArrayFromImage(input_image).astype(float)
    max_intensity = np.max(input_image_np)
    input_image_np = input_image_np / max_intensity
    output_image = itk.GetImageFromArray(input_image_np)
    output_image.SetOrigin(input_image.GetOrigin())
    output_image.SetDirection(input_image.GetDirection())
    output_image.SetSpacing(input_image.GetSpacing())
    subject_id = os.path.splitext(filename)[0]
    itk.imwrite(output_image, subject_id + "_fl.nii")

