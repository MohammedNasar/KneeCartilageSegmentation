#!/usr/bin/env python3.6

#==========================================================================
#   Advanced Normalization tools (ANTs) Example script.
#   Author:
#
#   Edited by:
#   A.M. Okorie , MIVIC/PEMACS/DSU (amokorie14@students.desu.edu)
#   S. Makrogiannis, MIVIC/PEMACS/DSU (smakrogiannis@desu.edu)
#==========================================================================

from sys import argv,exit
import os
import ants

# fixedImage_filename = I0022091.M03.Input.nii, movingImage_filename = I0021036.M03.Input.nii, registeredImage_filename = registered.nii, deformationField_filename = deformation.mhd, movingLabels_filename = I0021036.M38.Labels.GT.nii

# Check for command syntax.
if len(argv) < 8:
    print('Usage:\n' + str(argv[0]) + ' <fixedImage_filename> <movingImage_filename> <registeredImage_filename> <deformationField_filename> <movingLabels_filename> <atlas_labels_suffix> <type_of_transform> \n')
    exit(0)
	
# Parameter definitions.
fixedImageFilename = argv[1]
movingImageFilename = argv[2]
registeredImageFilename = argv[3]
deformationFieldFilename = argv[4]
movingImageLabelsFilename = argv[5]
atlasLabelsSuffix = argv[6]
type_of_transform = argv[7]

print("\n fixedImage: " + fixedImageFilename + ",\t movingImage: " + movingImageFilename + ",\t registeredImageFilename: " + registeredImageFilename + ",\t deformationFieldFilename: " + deformationFieldFilename + ",\t movingImageLabelsFilename: " + movingImageLabelsFilename + ",\t atlasLabelsSuffix: " + atlasLabelsSuffix + "\n")

# print("\n Subject: " + fixedImageFilename + ",\t Atlas: " + movingImageFilename + "\n")



#  Register a pair of images either through the full or simplified interface to the ANTs registration method

# fixedImageFilename =  '~/ShapeModelingWorkspace/38PCT/OriginalImages/I0021036.M03.Input.nii'

# movingImageFilename = '~/ShapeModelingWorkspace/38PCT/OriginalImages/I0022560.M03.Input.nii'

fixedImage = ants.image_read(fixedImageFilename)
movingImage = ants.image_read(movingImageFilename)

# Set min intensity to zero
fixedImage = fixedImage - fixedImage.min()
movingImageMin = movingImage.min()
movingImage = movingImage - movingImageMin


#  “SyN”: Symmetric normalization: Affine + deformable transformation, with mutual information as optimization metric.
mytx = ants.registration(fixed=fixedImage, moving=movingImage, type_of_transform = type_of_transform )

# Apply a transform list to map atlas labels from one atlas space to subject space 
# In image registration, one computes mappings between (usually) pairs of images. 
# These transforms are often a sequence of increasingly complex maps, e.g. from translation, 
# to rigid, to affine to deformation. The list of such transforms is passed to this function to 
# interpolate one image domain into the next image domain, as below. The order matters strongly 
# and the user is advised to familiarize with the standards established in examples.

# mywarpedimage = ants.apply_transforms(fixed=fixedImage, moving=movingImage,transformlist=mytx['fwdtransforms'] )
print(movingImageLabelsFilename)
movingImageLabels = ants.image_read(movingImageLabelsFilename)
warpedMovingImageLabels = ants.apply_transforms(fixed=fixedImage,interpolator='genericLabel',moving=movingImageLabels,transformlist=mytx['fwdtransforms'] )


# Save registration/segmentation results

# get warped and write moving image to file (save image of atlas warped to subject space)
warpedMovingImage = mytx.get('warpedmovout')
# print("Moving Image Mininum Intensity:" + str(movingImageMin) + "\n")
warpedMovingImage = warpedMovingImage + movingImageMin
ants.image_write(warpedMovingImage, registeredImageFilename)

# read and write forward transform parameters to file 
# ants.write_transform(ants.read_transform(fwdtransformsParameters), 'fwdtransformsParameters.mat')
ants.image_write(ants.image_read(mytx.get('fwdtransforms')[0]), deformationFieldFilename)

registration_suffix = "_Reg_From_"
warping_suffix = "_Warp_From_"
fixed_image_name = os.path.splitext(fixedImageFilename)[0]

nonlinearly_registered_atlas_labels_filename = fixed_image_name + warping_suffix + registration_suffix + atlasLabelsSuffix  + ".nii"
print(nonlinearly_registered_atlas_labels_filename + "\n")

# write warped atlas labels to file (save atlas label image warped to subject space)
ants.image_write(warpedMovingImageLabels, nonlinearly_registered_atlas_labels_filename)
