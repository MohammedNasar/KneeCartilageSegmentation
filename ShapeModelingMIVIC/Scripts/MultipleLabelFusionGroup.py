#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 13:20:45 2023

@author: mivic lab
"""

# import itk
import os, sys, csv
from JointMultiLabelFusion import JointMultiLabelFusion3D
from STAPLEMultiLabelFusion import STAPLEMultiLabelFusion3D
# from ReadITKTransform import CopyImageHeaderInfo3D


def ApplyMultiLabelFusionGroup(dir_list_filename, subject_list_filename, atlas_list_filename, label_fusion_method):
    '''
    

    Parameters
    ----------
    dir_list_filename : .CSV FILE (string)
        A list of the root folder and folders containing segmentations to be fused.
    subject_list_filename : .TXT FILE (string)
        List of all the subjects.
    atlas_list_filename : .CSV FILE (string)
        List of atlases.
    label_fusion_method : STR (string)
        STAPLE, or JLF.

    Returns
    -------
    None.

    '''
    
    # specify the folder containg the results of MAS segmentation
    # data_root_dir = os.path.expanduser(os.path.join('~', 'Data')) # 'ShapeModelingWorkspace', 'Myo_SegmenTUM_thigh3D'))
    # working_dir = os.getcwd()
    # label_fusion_method = 'STAPLE' # JLF
    # result_dir = 'MultiLabelFusion_' + label_fusion_method
    # os.mkdir(result_dir)
    # os.chdir(result_dir)
    # dir_list_filename = 'FolderList.csv'
    # atlas_list_filename = 'AtlasList3.csv'
    # subject_list_filename = 'SubjectList.txt'
    dir_list = list()
    registration_suffix = "_Reg_From_"
    warping_suffix = "_Warp_From_"
    # tissueLabels = [1,2,3,4,5,6,7,8]
    tissueLabels = [1]
    print('dir_list_filename',dir_list_filename)
    # Read-in folder list from csv and copy to python list.
    with open(dir_list_filename, 'r') as d:
        dir_reader = csv.reader(d)
        print("dir_reader",dir_reader)
        data_root_dir = next(dir_reader)[0]
        print(data_root_dir)
        for dir_row in dir_reader:
            dir_name = os.path.join(data_root_dir, dir_row[0])
            print(dir_name)
            dir_list.append(dir_name)
    
    # Read subject list and run segmentation.
    f = open(subject_list_filename, 'r')
    print(f)
    
    for line in f:
        subject_filename = line.rstrip('\n')
        subject = os.path.splitext(subject_filename)[0]
        filename_prefix = subject + warping_suffix + registration_suffix
        
        if label_fusion_method == 'STAPLE':
            # Apply STAPLE method for label fusion.
            STAPLEMultiLabelFusion3D(filename_prefix, atlas_list_filename, tissueLabels, subject_filename, dir_list)
        elif label_fusion_method == 'JLF':
            # Apply Joint Label Fusion.
            # subject_filename = os.path.join(data_root_dir, subject_filename)
            JointMultiLabelFusion3D(filename_prefix, atlas_list_filename, tissueLabels, subject_filename, dir_list)
        else:
            print('Unknown label fusion method. Exiting ...')
            exit(1)
        
    return 0
    
    
    
# Main function.
if __name__ == "__main__":

    # Check for command syntax.
    if len(sys.argv) < 5:
        print('Usage:\n' + str(sys.argv[0]) + ' <dir_list_filename> <subject_list_filename> <atlas_list_filename> <label_fusion_method>\n')
        sys.exit(0)

    # Get/Set parameters.
    dir_list_filename = sys.argv[1]
    subject_list_filename = sys.argv[2]
    atlas_list_filename = sys.argv[3]
    label_fusion_method = sys.argv[4]

    # Call the fusion function.
    ApplyMultiLabelFusionGroup(dir_list_filename, subject_list_filename, atlas_list_filename, label_fusion_method)
