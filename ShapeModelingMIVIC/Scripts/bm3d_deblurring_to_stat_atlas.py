#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply BM3D deblurring to statistical atlas.

@author: S. Makrogiannis, MIVIC/PEMaCS/DSU (smakrogiannis@desu.edu)
"""

import itk
import numpy as np
import matplotlib.pyplot as plt
import bm3d


def apply_bm3d_3d(input_image_np):
        
    # Determine sigma and v.
    experiment_number = 1

    array_size = input_image_np.shape
    n_slices = array_size[0]
    filtered_image_np = np.zeros(array_size, np.float32)    

    if experiment_number == 1:
        bsnr = 40
        sigma = .05  # if "sigma=-1", then the value of sigma deps on the BSNR
        v = np.ones((9, 9))
        v = v / np.sum(v)
    else:
        sigma = 8 / 255
        v = bm3d.gaussian_kernel((25, 25), 0.4)

        
    for i in range(0, n_slices):    
        print('Slice #:', i)
        input_image_slice = input_image_np[i,:,:]
#        input_image_slice = input_image_slice / np.max(input_image_slice)
        input_image_slice = np.atleast_3d(input_image_slice)

        if sigma == -1:
            sigma = np.sqrt(np.linalg.norm(np.ravel(input_image_slice - np.mean(input_image_slice)), 2) ** 2 / (input_image_slice.shape[0] * input_image_slice.shape[1] * 10 ** (bsnr / 10)))
        
        filtered_image_np[i,:,:] =  bm3d.bm3d_deblurring(input_image_slice,sigma,v) 

        psnr = get_psnr(input_image_slice, filtered_image_np[i,:,:])
        print("PSNR:", psnr)
        
        difference_image = filtered_image_np[i,:,:] - np.squeeze(input_image_slice)
        
        fig, (ax1, ax2) = plt.subplots(1, 2)
#         im = plt.imshow(np.concatenate((np.squeeze(input_image_slice), filtered_image_np[i,:,:],difference_image),axis=1), cmap='gray')
        ax1.imshow(np.concatenate((np.squeeze(input_image_slice), filtered_image_np[i,:,:]),axis=1), cmap='gray')
        im2 = ax2.imshow(difference_image, cmap='gray')
        plt.colorbar(im2, ax=ax2, shrink=0.6)
        fig.suptitle('Original, Deblurred, Difference')
        plt.show()

    return(filtered_image_np)


   
def get_psnr(y_est: np.ndarray, y_ref: np.ndarray) -> float:
    """
    Return PSNR value for y_est and y_ref presuming the noise-free maximum is 1.
    :param y_est: Estimate array
    :param y_ref: Noise-free reference
    :return: PSNR value
    """
    return 10 * np.log10(1 / np.mean(((y_est - y_ref).ravel()) ** 2))
    





# data_path = '/Users/sokratis/OneDrive - Delaware State University/TIDAQU/ShapeModelingWorkspace/Myo_SegmenTUM_thigh3D/StatisticalAtlasGeneration_FFD3/'
data_path = 'C:/Users/smakrogiannis/OneDriveDSU/TIDAQU/ShapeModelingWorkspace/Myo_SegmenTUM_thigh3D/StatisticalAtlasGeneration_FFD3/'
input_image = itk.imread(data_path + 'Myo_SegmenTUM_thigh3D_SubjectList_Warp_HV006_1_WATER_stack2.nii')
input_image_np = itk.GetArrayFromImage(input_image)
#input_image_np = np.asarray(input_image)

max_intensity = np.max(input_image_np)
input_image_np = input_image_np / max_intensity

filtered_image_np = apply_bm3d_3d(input_image_np)

filtered_image_np = max_intensity * filtered_image_np  
filtered_image_np = np.short(filtered_image_np)

filtered_image = itk.GetImageFromArray(filtered_image_np)

# Copy header info to filtered_image.
filtered_image.SetOrigin(input_image.GetOrigin())
filtered_image.SetDirection(input_image.GetDirection())
filtered_image.SetSpacing(input_image.GetSpacing())

itk.imwrite(filtered_image, 'Myo_SegmenTUM_thigh3D_SubjectList_Warp_HV006_1_WATER_stack2_BM3D.nii')

