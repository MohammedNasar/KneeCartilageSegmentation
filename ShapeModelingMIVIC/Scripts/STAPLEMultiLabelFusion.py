#!/usr/bin/env python

#==========================================================================
#   Apply STAPLE to make decision for tissue segmentation.
#   S. Makrogiannis, MIVIC/DPCS/DSU (smakrogiannis@desu.edu)
#==========================================================================*/

import os, csv
import numpy as np
import itk
from ReadITKTransform import CopyImageHeaderInfo3D


def STAPLEMultiLabelFusion3D(subjectID, atlas_list_filename, tissueLabels, subject_filename=[""], dir_list=[""]):
    '''
    '''

    # Export atlas list identifier.
    atlas_list = os.path.splitext(atlas_list_filename)[0]
    
    nTissueLabels = len(tissueLabels)
    tissueLabelswithAir = itk.VariableLengthVector[itk.SS]()
    tissueLabelswithAir.SetSize(nTissueLabels+1)
    tissueLabelswithAir.Fill(0)
    for i in range(1, nTissueLabels+1):
        tissueLabelswithAir.SetElement(i, tissueLabels[i-1])
    tissueLabels = tissueLabelswithAir

    # Identify tissue labels.

    # Fuse labels from different atlas mappings.
    # Use itk's STAPLE.
    stapler = itk.STAPLEImageFilter.IF3IF3.New()

    label_counter = 0
    # For each tissue label:
    for i in range(0, nTissueLabels+1):

        tissueLabel = tissueLabels.GetElement(i)
        print(tissueLabel)

        stapler.SetForegroundValue(tissueLabel)

        # Initialize counter of atlases.
        atlas_counter = 0 

        # Read atlas list and add label images to voter.
        for k in range(0, len(dir_list)):
            dir_name = dir_list[k]
            uses_atlas = "Unet" not in dir_name

            # If atlas-based technique
            if uses_atlas:
                with open(atlas_list_filename, 'r') as f:
                    reader = csv.reader(f)
                    # segmented_subject_string = "" 
                    for row in reader:
                        atlas_filename = row[0]
                        atlas = os.path.splitext(atlas_filename)[0]
                        atlas_labels_filename = row[1]
                        atlas_labels = os.path.splitext(atlas_labels_filename)[0]

                        # Form the string for segmentation fusion.
                        nonlinearly_registered_atlas_labels_filename = os.path.join(dir_name, subjectID + atlas_labels_filename)
                        imageReader = itk.ImageFileReader.IF3.New()
                        imageReader.SetFileName( nonlinearly_registered_atlas_labels_filename )
                        imageReader.Update()
                        nonlinearly_registered_atlas_labels = imageReader.GetOutput()
                        
                        # Ensure consistent volume geometry.
                        if k==0:
                            subject_origin = nonlinearly_registered_atlas_labels.GetOrigin()
                            subject_direction = nonlinearly_registered_atlas_labels.GetDirection()
                            subject_spacing = nonlinearly_registered_atlas_labels.GetSpacing()
                        else:
                            nonlinearly_registered_atlas_labels.SetOrigin(subject_origin)
                            nonlinearly_registered_atlas_labels.SetDirection(subject_direction)
                            nonlinearly_registered_atlas_labels.SetSpacing(subject_spacing)

                        # Add label map to stapler.
                        stapler.SetInput(atlas_counter, nonlinearly_registered_atlas_labels)

                        # Increase label map index.
                        atlas_counter += 1
            else:
                    # Read label map.
                    no_atlas_subjectID = os.path.splitext(subject_filename)[0]
                    subject_labels_filename = os.path.join(dir_name, no_atlas_subjectID + "_prediction.nii")
                    imageReader = itk.ImageFileReader.IF3.New()
                    imageReader.SetFileName( subject_labels_filename )
                    imageReader.Update()
                    subject_label_map = imageReader.GetOutput()

                    # Ensure consistent volume geometry.
                    if k==0:
                        subject_origin = subject_label_map.GetOrigin()
                        subject_direction = subject_label_map.GetDirection()
                        subject_spacing = subject_label_map.GetSpacing()
                    else:
                        subject_label_map.SetOrigin(subject_origin)
                        subject_label_map.SetDirection(subject_direction)
                        subject_label_map.SetSpacing(subject_spacing)

                    # Add label map to stapler.
                    stapler.SetInput(atlas_counter, subject_label_map)
                    
                    # Increase label map index.
                    atlas_counter += 1
                
        # Run STAPLE
        print("Running STAPLE...\n")
        stapler.Update()
        LikelihoodMap = stapler.GetOutput()

        # Write tissue likelihood image to file.
        save_likelihood_maps = False
        if save_likelihood_maps:
            staple_likelihood_map_filename = subjectID + atlas_list +"_STAPLE" + str(tissueLabel) + ".nii"
            imageWriter = itk.ImageFileWriter.IF3.New()
            imageWriter.SetInput( LikelihoodMap )
            imageWriter.SetFileName( staple_likelihood_map_filename )
            imageWriter.Update()

        # To debug read from file.
#         imageReader = itk.ImageFileReader.IF3.New()
#         imageReader.SetFileName( staple_likelihood_map_filename )
#         imageReader.Update()
#         LikelihoodMap = imageReader.GetOutput()

        if label_counter == 0:
            #  copy tissue likelihood map to a new array.
            size = LikelihoodMap.GetLargestPossibleRegion().GetSize()
            new_size = np.append(size, nTissueLabels+1)
            new_size = np.flip(new_size)
            print(new_size)
            LikelihoodArray4D = np.zeros(new_size, np.float32)
        # else:
            # Concatenate to previous array
            # LikelihoodArray4D = np.concatenate((LikelihoodArray4D, itk.GetArrayFromImage(LikelihoodMap)), axis=0)

        LikelihoodArray4D[label_counter,:,:,:] = itk.GetArrayFromImage(LikelihoodMap)

        print(LikelihoodArray4D.shape)
        label_counter += 1

    itk.imwrite(itk.GetImageViewFromArray(LikelihoodArray4D), "test.nii")

    # Set label of maximum probability as the final decision.
    MaximumLikelihoodLabel = np.argmax(LikelihoodArray4D, axis=0)
    MaximumLikelihoodLabel = MaximumLikelihoodLabel.astype(np.float32)
    print(MaximumLikelihoodLabel.dtype)
    print(MaximumLikelihoodLabel.shape)

    MaximumLikelihoodLabelImage = itk.GetImageFromArray(MaximumLikelihoodLabel)

    # Cast image to UC2 type.
    caster = itk.CastImageFilter.IF3IUC3.New()
    caster.SetInput( MaximumLikelihoodLabelImage )
    caster.Update()

    # Change labels to original.
    labelChanger = itk.ChangeLabelImageFilter.IUC3IUC3.New()
    labelChanger.SetInput(caster.GetOutput())
    for i in range(0, nTissueLabels+1):
        labelChanger.SetChange(i, tissueLabelswithAir.GetElement(i))
    labelChanger.Update()
    FusedLabelImage = labelChanger.GetOutput()

    # Copy header info to fused label image.
    FusedLabelImage.SetOrigin(LikelihoodMap.GetOrigin())
    FusedLabelImage.SetDirection(LikelihoodMap.GetDirection())
    FusedLabelImage.SetSpacing(LikelihoodMap.GetSpacing())

    # Write tissue likelihood image to file.
    atlas_list_id = os.path.split(atlas_list)[1]
    staple_segmented_subject_filename = subjectID + atlas_list_id +"_STAPLE" + ".nii"
    imageWriter = itk.ImageFileWriter.IUC3.New()
    imageWriter.SetInput( FusedLabelImage )
    imageWriter.SetFileName( staple_segmented_subject_filename )
    imageWriter.Update()

    return 0


def STAPLEMultiLabelFusion2D(subjectID, atlas_list_filename, tissueLabels):
    '''
    '''

    # Export atlas list identifier.
    atlas_list = os.path.splitext(atlas_list_filename)[0]
    
    nTissueLabels = len(tissueLabels)
    tissueLabelswithAir = itk.VariableLengthVector[itk.SS]()
    tissueLabelswithAir.SetSize(nTissueLabels+1)
    tissueLabelswithAir.Fill(0)
    for i in range(1, nTissueLabels+1):
        tissueLabelswithAir.SetElement(i, tissueLabels[i-1])
    tissueLabels = tissueLabelswithAir

    # Identify tissue labels.

    # Fuse labels from different atlas mappings.
    # Use itk's STAPLE.
    stapler = itk.STAPLEImageFilter.IF2IF2.New()

    label_counter = 0
    # For each tissue label:
    for i in range(0, nTissueLabels+1):

        tissueLabel = tissueLabels.GetElement(i)
        print(tissueLabel)

        stapler.SetForegroundValue(tissueLabel)

        # Initialize counter of atlases.
        atlas_counter = 0 

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
                nonlinearly_registered_atlas_labels_filename = subjectID + atlas_labels  + ".nii"
                imageReader = itk.ImageFileReader.IF2.New()
                imageReader.SetFileName( nonlinearly_registered_atlas_labels_filename )
                imageReader.Update()

                # Add label map to stapler.
                stapler.SetInput(atlas_counter, imageReader.GetOutput())

                # Increase label map index.
                atlas_counter += 1

        # Run STAPLE
        stapler.Update()
        LikelihoodMap = stapler.GetOutput()

        # Write tissue likelihood image to file.
        staple_likelihood_map_filename = subjectID + atlas_list +"_STAPLE" + str(tissueLabel) + ".nii"
        imageWriter = itk.ImageFileWriter.IF2.New()
        imageWriter.SetInput( LikelihoodMap )
        imageWriter.SetFileName( staple_likelihood_map_filename )
        imageWriter.Update()

        if label_counter == 0:
            #  copy tissue likelihood map to a new array.
            size = LikelihoodMap.GetLargestPossibleRegion().GetSize()
            LikelihoodArray3D = np.zeros(size, np.float32)
            LikelihoodArray3D = itk.GetArrayFromImage(LikelihoodMap)
        else:
            # Concatenate to previous array
            LikelihoodArray3D = np.dstack((LikelihoodArray3D, itk.GetArrayFromImage(LikelihoodMap)))

        # print(LikelihoodArray3D.shape)
        label_counter += 1

    itk.imwrite(itk.GetImageViewFromArray(LikelihoodArray3D), "test.nii")

    # Set label of maximum probability as the final decision.
    MaximumLikelihoodLabel = np.argmax(LikelihoodArray3D, axis=2)
    MaximumLikelihoodLabel = MaximumLikelihoodLabel.astype(np.float32)
    # print(MaximumLikelihoodLabel.dtype)

    MaximumLikelihoodLabelImage = itk.GetImageFromArray(MaximumLikelihoodLabel)

    # Cast image to UC2 type.
    caster = itk.CastImageFilter.IF2IUC2.New()
    caster.SetInput( MaximumLikelihoodLabelImage )
    caster.Update()

    # Change labels to original.
    labelChanger = itk.ChangeLabelImageFilter.IUC2IUC2.New()
    labelChanger.SetInput(caster.GetOutput())
    for i in range(0, nTissueLabels+1):
        labelChanger.SetChange(i, tissueLabelswithAir.GetElement(i))
    labelChanger.Update()

    # Write tissue likelihood image to file.
    staple_segmented_subject_filename = subjectID + atlas_list +"_STAPLE" + ".nii"
    imageWriter = itk.ImageFileWriter.IUC2.New()
    imageWriter.SetInput( labelChanger.GetOutput() )
    imageWriter.SetFileName( staple_segmented_subject_filename )
    imageWriter.Update()

    # Return final label map.

    return 0
