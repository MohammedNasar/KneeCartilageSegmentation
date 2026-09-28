#!/usr/bin/env python

#==========================================================================
#   Apply JLF to make decision for tissue segmentation.
#   S. Makrogiannis, MIVIC/PEMACS/DSU (smakrogiannis@desu.edu)
#==========================================================================*/

import os, csv
import numpy as np
import itk
from ReadITKTransform import CopyImageHeaderInfo3D
import ants


def JointMultiLabelFusion3D(subjectID, atlas_list_filename, tissueLabels, subject_filename, dir_list=[""]):
    ''' Example:
	import ants
	ref = ants.image_read( ants.get_ants_data('r16'))
	ref = ants.resample_image(ref, (50,50),1,0)
	ref = ants.iMath(ref,'Normalize')
	mi = ants.image_read( ants.get_ants_data('r27'))
	mi2 = ants.image_read( ants.get_ants_data('r30'))
	mi3 = ants.image_read( ants.get_ants_data('r62'))
	mi4 = ants.image_read( ants.get_ants_data('r64'))
	mi5 = ants.image_read( ants.get_ants_data('r85'))
	refmask = ants.get_mask(ref,low_thresh=0.01,cleanup=1)
	refmask = ants.iMath(refmask,'ME',2) # just to speed things up
	ilist = [mi,mi2,mi3,mi4,mi5]
	seglist = [None]*len(ilist)
	for i in range(len(ilist)):
		ilist[i] = ants.iMath(ilist[i],'Normalize')
		mytx = ants.registration(fixed=ref , moving=ilist[i] ,
			typeofTransform = ('Affine') )
		mywarpedimage = ants.apply_transforms(fixed=ref,moving=ilist[i],
				transformlist=mytx['fwdtransforms'])
		ilist[i] = mywarpedimage
		seg = ants.threshold_image(ilist[i],'Otsu', 3)
		seglist[i] = seg
	r = 2
	pp = ants.joint_label_fusion(ref, refmask, ilist, r_search=2,
						label_list=seglist, rad=[r]*ref.dimension )
	pp = ants.joint_label_fusion(ref,refmask,ilist, r_search=2, rad=[r]*ref.dimension)
    '''

    # Export atlas list identifier.
    atlas_list = os.path.splitext(atlas_list_filename)[0]
    
    # Identify tissue labels.
    nTissueLabels = len(tissueLabels)
    tissueLabelswithAir = itk.VariableLengthVector[itk.SS]()
    tissueLabelswithAir.SetSize(nTissueLabels+1)
    tissueLabelswithAir.Fill(0)
    for i in range(1, nTissueLabels+1):
        tissueLabelswithAir.SetElement(i, tissueLabels[i-1])
    tissueLabels = tissueLabelswithAir

    # Fuse labels from different atlas mappings.
    # Use ANT's Joint Label Fusion.
	
    # Initialize counter of atlases.
    ilist = list()
    seglist = list()

    # Read atlas list and add label images to JLF.
    for k in range(0, len(dir_list)):
        dir_name = dir_list[k]
        uses_atlas = "Unet" not in dir_name


        # Read subject and subject labels.
        subject_path = os.path.join(dir_name, subject_filename)
        subject = ants.image_read( subject_path )
        subject = ants.iMath(subject, 'Normalize')

        if k==0:
            subject_origin = ants.get_origin(subject)
            subject_direction = ants.get_direction(subject)
            subject_spacing = ants.get_spacing(subject)
        else:
            subject.set_origin(subject_origin)
            subject.set_direction(subject_direction)
            subject.set_spacing(subject_spacing)


        # Create leg mask.
        # subject_mask = ants.get_mask(subject, low_thresh=0.01, cleanup=0)
        subject_mask = ants.threshold_image(subject, 'Otsu', 2)
        # subject_mask = ants.iMath(subject_mask,'ME',2) # just to speed things up
        subject_mask.set_origin(subject_origin)
        subject_mask.set_direction(subject_direction)
        subject_mask.set_spacing(subject_spacing)
        ants.image_write( subject_mask, 'jlf_target_mask.nii.gz' )


        # If atlas-based technique.
        if uses_atlas:
            # Read atlas list and add label images to voter.
            with open(atlas_list_filename, 'r') as f:
                reader = csv.reader(f)
                segmented_subject_string = "" 
                for row in reader:
                    atlas_filename = row[0]
                    atlas_name = os.path.splitext(atlas_filename)[0]
                    atlas_labels_filename = row[1]
                    atlas_labels_name = os.path.splitext(atlas_labels_filename)[0]

                    # Form filename strings.
                    # Form the string for segmentation fusion.
                    nonlinearly_registered_atlas_filename = os.path.join(dir_name, subjectID + atlas_filename)
                    nonlinearly_registered_atlas_labels_filename = os.path.join(dir_name, subjectID + atlas_labels_filename)

                    # Read atlas and atlas labels.
                    mapped_atlas = ants.image_read(nonlinearly_registered_atlas_filename)
                    mapped_atlas.set_origin(subject_origin)
                    mapped_atlas.set_direction(subject_direction)
                    mapped_atlas.set_spacing(subject_spacing)

                    mapped_atlas_labels = ants.image_read(nonlinearly_registered_atlas_labels_filename)
                    mapped_atlas_labels.set_origin(subject_origin)
                    mapped_atlas_labels.set_direction(subject_direction)
                    mapped_atlas_labels.set_spacing(subject_spacing)

                    # Add images to lists.
                    ilist.append(ants.iMath(mapped_atlas,'Normalize'))
                    seglist.append(mapped_atlas_labels)
        else:
            # Add the subject image.
            ilist.append(subject)
            
            # Add the label image.
            subjectID = os.path.splitext(subject_filename)[0]
            subject_labels_filename = os.path.join(dir_name, subjectID + "_prediction.nii")
            subject_labels = ants.image_read(subject_labels_filename)
            subject_labels.set_origin(subject_origin)
            subject_labels.set_direction(subject_direction)
            subject_labels.set_spacing(subject_spacing)
            seglist.append(subject_labels)



    # Run JLF
    r = 3
    print("Running JLF...\n")
    fusion_output = ants.joint_label_fusion(subject, subject_mask, ilist, r_search=3, label_list=seglist, 
                                            rad=[r]*subject.dimension, no_zeroes=False, max_lab_plus_one=True)	
	
    # Write fused label image to file.
    atlas_list_id = os.path.split(atlas_list)[1]
    jlf_segmented_subject_filename = subjectID + atlas_list_id +"_JLF" + ".nii"
    ants.image_write( fusion_output['segmentation'], jlf_segmented_subject_filename )

    # Write fused intensity image to file.
    jlf_fused_subject_filename = "Intensity_Image_JLF" + ".nii"
    ants.image_write( fusion_output['intensity'], jlf_fused_subject_filename )
    

    return 0
