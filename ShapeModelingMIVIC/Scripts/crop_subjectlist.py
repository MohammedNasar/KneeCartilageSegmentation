#!/usr/bin/env python

"""
Created on Fri Sep  9 06:46:34 2022

@author: smakrogiannis MIVIC Lab
"""


import itk
import os
from cropping_script import symmetric_crop

no_of_pixels_x = 24
no_of_pixels_y = 120
no_of_pixels_z = 0

subject_list_filename = 'C:/Users/amokorie/Delaware State University/Sokratis Makrogiannis - TIDAQU/ShapeModelingWorkspace/Myo_SegmenTUM_thigh3D/Segmented_Labels_croppadSubjectList.txt'

# Read subject list and run segmentation.
f = open(subject_list_filename, 'r')
#print(f)

for line in f:
    subject_filename = line.rstrip('\n')
    subject = os.path.splitext(subject_filename)[0]
    input_image = itk.imread(subject_filename)
    cropped_image = symmetric_crop(input_image, no_of_pixels_x, no_of_pixels_y, no_of_pixels_z)
    output_filename = 'Cropped/' + subject + '_crop.nii'
    itk.imwrite(cropped_image, output_filename)
    
f.close()