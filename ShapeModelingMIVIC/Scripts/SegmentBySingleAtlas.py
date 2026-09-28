#!/usr/bin/env python

#==========================================================================
#   Single atlas-based segmentation script.
#   S. Makrogiannis, MIVIC/DPCS/DSU (smakrogiannis@desu.edu)
#==========================================================================

from sys import argv,exit
import os
import time, csv


# get the start time
start_time = time.time()


# Check for command syntax.
if len(argv) < 4:
    print('Usage:\n' + str(argv[0]) + ' <Subject> <Atlas> <Atlas Labels>\n')
    exit(0)


# Parameter definitions.
subject_filename = argv[1]
atlas_filename = argv[2]
atlas_labels_filename = argv[3]
# segmented_subject_filename = argv[4]

print("\nSubject: " + subject_filename + ",\t Atlas: " + atlas_filename + "\n")

project_path = "/home/users/mibrahim/segmentation/knee/ge_scanner/ShapeModelingMIVIC/"


# 2D programs.
# linear_registration_program = project_path + "Scripts/LinearRegistration2D.py"
# linear_transform_program = project_path + "Scripts/ApplyTransform2D.py"
# ffd_registration_program = "/usr/local/DeformableRegistration2D_x86_64_dyn_rel/bin/DeformableRegistration2D_DSUMIVIC" 
# sdd_registration_program = "/usr/local/DeformableRegistration2D_x86_64_dyn_rel/bin/DeformableRegistration2D_17"
# warping_program = "/usr/local/DeformableRegistration2D_x86_64_dyn_rel/bin/WarpImageFilter2D"

# 3D programs.
linear_registration_3d_python = project_path + "Scripts/LinearRegistration3D.py"
linear_registration_multistage_3d = "/usr/local/LinearRegistration3D_x86_64_dyn_rel/bin/MultiStageImageRegistration3D_2"
# ImageRegistration3D_8,ImageRegistration3D_20,MultiResImageRegistration3D_3,MultiStageImageRegistration3D_2
linear_transform_program = project_path + "Scripts/ApplyTransform3D.py"

ants_registration_program = project_path + "Scripts/DeformableRegistration_ANTs.py"
ffd_registration_program = "/usr/local/DeformableRegistration3D_x86_64_dyn_rel/bin/DeformableRegistration3D_DSUMIVIC"
sdd_registration_program = "/usr/local/DeformableRegistration3D_x86_64_dyn_rel/bin/DeformableRegistration3D_17"
vxm_registration_program = project_path + "Scripts/vxm_register.py"
warping_program = "/usr/local/DeformableRegistration3D_x86_64_dyn_rel/bin/WarpImageFilter3D"


# Settings
ants_type_of_transform = "SyN" # Options: SyN, SyNOnly
use_MIVIC_linear_transformation = True  # set to False for SyN
vxm_net_filename = "/home/users/mibrahim/segmentation/Knee/GE_Scanner_10/ShapeModelingMIVIC/Scripts/vxm_knee_25Mar2026/models/vxm_template/0100.h5" 
# "3d_thigh_mri_myo_segmentum_v6.h5"

linear_registration_program = linear_registration_multistage_3d
elastic_registration_program = sdd_registration_program

registration_suffix = "_Reg_From_"
warping_suffix = "_Warp_From_"
execution_options = " >> log.txt"
# diffeomorphic_demons_options = " -i 20x15x10x5 >> log.txt"

subject = os.path.splitext(subject_filename)[0]
atlas = os.path.splitext(atlas_filename)[0]
atlas_labels = os.path.splitext(atlas_labels_filename)[0]
print(subject)
print(atlas)
print(atlas_labels)


# Linear registration of atlas to subject's space.

# ~/Software/src/thalesrepos/trunk/ShapeModeling/Scripts/LinearRegistration2D.py I0022703.M01.Input.nii SixtySixPCT_SubjectList_Corrected4.txt_Warp_I0022284.M01.Input.nii test.linear.nii
if use_MIVIC_linear_transformation == True:
    linearly_registered_atlas_filename = subject + registration_suffix + atlas + ".nii"
    linearly_registered_atlas_labels_filename =  subject + registration_suffix + atlas_labels + ".nii"
    if linear_registration_program == linear_registration_3d_python:
        command = linear_registration_program + " " + subject_filename + " " + atlas_filename + " " + linearly_registered_atlas_filename + " " + execution_options 
        print( command + "\n")
        os.system( command )
        # Transform atlas labels to subject's space.
        affine_transform_filename = linearly_registered_atlas_filename + ".tfm"
        linearly_registered_atlas_labels_filename =  subject + registration_suffix + atlas_labels + ".nii"
        command =  "python" + " " + linear_transform_program + " " + atlas_labels_filename + " " + subject_filename + " " + affine_transform_filename + " " + linearly_registered_atlas_labels_filename + " NN" + execution_options 
        print( command + "\n")
        os.system( command )
    else:
        command = linear_registration_program + " " + subject_filename + " " + atlas_filename + " " + linearly_registered_atlas_filename + " " + atlas_labels_filename + " " + linearly_registered_atlas_labels_filename + " " + execution_options 
        print( command + "\n")
        os.system( command )
else: 
    linearly_registered_atlas_filename = atlas + ".nii"
    print(linearly_registered_atlas_filename)
    linearly_registered_atlas_labels_filename = atlas_labels + ".nii"
    print(linearly_registered_atlas_labels_filename)


# Elastic registration of linearly-registered atlas to subject's space.

nonlinearly_registered_atlas_filename =  subject + warping_suffix + registration_suffix + atlas + ".nii"
nonlinearly_registered_atlas_field_filename =  subject + warping_suffix + registration_suffix + atlas + "_Field.nii.gz";
nonlinearly_registered_atlas_labels_filename = subject + warping_suffix + registration_suffix + atlas_labels  + ".nii"

if elastic_registration_program == ants_registration_program:
    command = "python3" + " " + ants_registration_program + " " + subject_filename + " " + linearly_registered_atlas_filename + " " + nonlinearly_registered_atlas_filename + " " + nonlinearly_registered_atlas_field_filename + " " + linearly_registered_atlas_labels_filename + " " + atlas_labels + " " + ants_type_of_transform
elif elastic_registration_program == vxm_registration_program:
    command = "python3" + " " + elastic_registration_program + " --fixed " + subject_filename + " --moving " + linearly_registered_atlas_filename + " --moved " + nonlinearly_registered_atlas_filename + " --warp " + nonlinearly_registered_atlas_field_filename + " --moving_labels " + linearly_registered_atlas_labels_filename  + " --moved_labels " + nonlinearly_registered_atlas_labels_filename  + " --model " + vxm_net_filename + " --gpu 0"
else:
    command = elastic_registration_program + " " + subject_filename + " " + linearly_registered_atlas_filename + " " + nonlinearly_registered_atlas_filename + " " + nonlinearly_registered_atlas_field_filename + execution_options

print( command + "\n")
os.system( command )


# Warp linearly-registered atlas labels to subject's space.
if (elastic_registration_program == sdd_registration_program) or  (elastic_registration_program == ffd_registration_program):
    # /usr/local/DeformableRegistration3D_x86_64_dyn_rel/bin/WarpImageFilter3D test.label.linear2.nii test.warp.field.mhd test.label.warp2.nii NN
    command = warping_program + " " + linearly_registered_atlas_labels_filename + " " + nonlinearly_registered_atlas_field_filename + " " + nonlinearly_registered_atlas_labels_filename + " NN" + execution_options 
    print( command + "\n")
    os.system( command )			

# get the end time
end_time = time.time()

# get execution time 
elapsed_time = end_time - start_time

print('Subject: {0}, Atlas: {1:5s}, start time: {2:5.3f}, end time: {3:5.3f}, elapsed time: {4:5.3f}' .format(subject, atlas, start_time, end_time, elapsed_time) + "\n")

# time = [atlas, start_time, end_time, elapsed_time]

# with open('execution_time.csv', 'a') as csvfile:
# 	writer = csv.writer(csvfile)
# 	writer.writerow(time)
	
