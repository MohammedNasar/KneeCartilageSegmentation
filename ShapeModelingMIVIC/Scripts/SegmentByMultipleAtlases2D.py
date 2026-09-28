#!/usr/bin/env python

#==========================================================================
#   Multi-atlas-based segmentation script.
#   S. Makrogiannis, MIVIC/DPCS/DSU (smakrogiannis@desu.edu)
#==========================================================================

import itk
from sys import argv,exit
from STAPLEMultiLabelFusion import STAPLEMultiLabelFusion2D
import os, csv
import time

# get the start time
start_time = time.time()


# Check for command syntax.
if len(argv) < 3:
    print('Usage:\n' + str(argv[0]) + ' <Subject> <Atlas List>\n')
    exit(0)


# Parse arguments and assign them to local variables.
subject_filename = argv[1]
atlas_list_filename = argv[2]

subject = os.path.splitext(subject_filename)[0]
atlas_list = os.path.splitext(atlas_list_filename)[0]


# Segmentation program using single atlas.
project_path = "~/Codes/src/gitrepos/ShapeModelingMIVIC/"
single_atlas_segmentation_program = project_path + "Scripts/SegmentBySingleAtlas.py"
image_calculator_program = "/usr/local/ITKApps_x86_64_dyn_rel/ITKApps-build/ImageCalculator/ImageCalculator"
calculator_options = " --intype FLOAT --outtype FLOAT -d 2"
registration_suffix = "_Reg_From_"
warping_suffix = "_Warp_From_"


# Read atlases list.
with open(atlas_list_filename, 'r') as f:
    reader = csv.reader(f)
    segmented_subject_string = "" 

    for row in reader:
        print( row )
        # For each member in the lists.
        atlas_filename = row[0]
        atlas = os.path.splitext(atlas_filename)[0]
        atlas_labels_filename = row[1]
        atlas_labels = os.path.splitext(atlas_labels_filename)[0]

        # Segmentation by single atlas.
        # ~/Software/src/sokrepo/ShapeModeling/Scripts/SegmentBySingleAtlas.py <Subject> <Atlas> <Atlas Labels>
        command = "python3" + " " + single_atlas_segmentation_program + " " + subject_filename + " " + atlas_filename + " " + atlas_labels_filename
        print( command + "\n")
        os.system( command )

        # Form the string for segmentation fusion.
        nonlinearly_registered_atlas_labels_filename = subject + warping_suffix + registration_suffix + atlas_labels  + ".nii"
        segmented_subject_string = segmented_subject_string + " " + nonlinearly_registered_atlas_labels_filename
        # End the for loop.


# Fuse labels from different atlas mappings.
# Compute average.
fused_segmented_subject_filename = subject + warping_suffix + registration_suffix + atlas_list + "_Mean.nii"
command = image_calculator_program + " --in " + segmented_subject_string + " --avg " + " --out " + fused_segmented_subject_filename + calculator_options 
print( command + "\n")
os.system( command )

# Use itk's label voting.
voter = itk.LabelVotingImageFilter.IUC2IUC2.New()

# Initialize counter of atlases.
counter = 0 

# Read atlas list and add label images to voter.
with open(atlas_list_filename, 'r') as f:
    reader = csv.reader(f)
    segmented_subject_string = "" 

    for row in reader:
        atlas_filename = row[0]
        atlas = os.path.splitext(atlas_filename)[0]
        atlas_labels_filename = row[1]
        atlas_labels = os.path.splitext(atlas_labels_filename)[0]
        
        # Form the string for segmentation fusion.
        nonlinearly_registered_atlas_labels_filename = subject + warping_suffix + registration_suffix + atlas_labels  + ".nii"
        imageReader = itk.ImageFileReader.IF2.New()
        imageReader.SetFileName( nonlinearly_registered_atlas_labels_filename )
        imageReader.Update()
        # Cast to IUC2.
        caster = itk.CastImageFilter.IF2IUC2.New()
        caster.SetInput( imageReader.GetOutput() )
        caster.Update()

        # Add label map to voter.
        voter.SetInput(counter, caster.GetOutput())

        # Increase label map index.
        counter += 1

        # Get the labels from the first image
        if counter==1:
            converter = itk.LabelImageToLabelMapFilter.IUC2LM2.New()
            converter.SetInput(caster.GetOutput())
            converter.Update()
            labelMap = converter.GetOutput()
            labelVector = labelMap.GetLabels()
            

# Run voting.
voter.Update()

# Write voting results to file.
fused_segmented_subject_filename = subject + warping_suffix + registration_suffix + atlas_list + "_Voting.nii"
imageWriter = itk.ImageFileWriter.IUC2.New()
imageWriter.SetInput( voter.GetOutput() )
imageWriter.SetFileName( fused_segmented_subject_filename )
imageWriter.Update()

# Apply STAPLE method for label fusion.
filename_prefix = subject + warping_suffix + registration_suffix
STAPLEMultiLabelFusion2D(filename_prefix, atlas_list_filename, labelVector)

# get the end time
end_time = time.time()

# get execution time 
elapsed_time = end_time - start_time

print('Subject: {0},  start time: {1:5.3f}, end time: {2:5.3f}, elapsed time: {3:5.3f}' .format(subject, start_time, end_time, elapsed_time) + "\n")