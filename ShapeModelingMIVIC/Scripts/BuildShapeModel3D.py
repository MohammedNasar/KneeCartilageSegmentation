#!/usr/bin/env python

#==========================================================================
#   Build atlas from multiple subjects in 3D space.
#   S. Makrogiannis, MIVIC/DPCS/DSU
#==========================================================================

import sys,os,itk,csv
from JointMultiLabelFusion import JointMultiLabelFusion3D

# Linearly register all subjects to template.
def AlignAllImagesToTemplate3D(subject_list_filename, template_filename, project_path):

    # Settings and options.
    registration_options = " >> log.txt" # " -3D"
    # calculator_options = " --intype SHORT --outtype SHORT -d 3"
    calculator_options = " --intype FLOAT --outtype FLOAT -d 3"
    nIterations = 1
    linear_registration_3d_python = project_path + "Scripts/LinearRegistration3D.py"
    linear_registration_multistage_3d = "/usr/local/LinearRegistration3D_x86_64_dyn_rel/bin/MultiStageImageRegistration3D_2"
    # ImageRegistration3D_8,ImageRegistration3D_20,MultiResImageRegistration3D_3,MultiStageImageRegistration3D_2

    # String conventions.
    registration_suffix = "_Reg_From_"
    warping_suffix = "_Warp_From_"
    variance_suffix = "_Var_"

    # Used programs.
    linear_registration_program = linear_registration_multistage_3d

    image_calculator_program = "/usr/local/ITKApps_x86_64_dyn_rel/ITKApps-build/ImageCalculator/ImageCalculator"

    # Select filename before file extension to be used
    # as subject id.
    template = os.path.splitext(template_filename)[0]
    subject_list = os.path.splitext(subject_list_filename)[0]
    average_filename = subject_list + registration_suffix + template + ".nii"
    variance_filename = subject_list + registration_suffix + variance_suffix + template + ".nii"

    # Apply iterative scheme to maximize overlapping.
    counter = 0
    while counter < nIterations:
        # Read subject list.
        f = open(subject_list_filename, 'r')
        reader = csv.reader(f)
        print("\nIteration #\t" + str(counter) + "\n")
        print(f)
        
        transformed_subject_string = ""
        # For each subject in the list:
        for row in reader:
            subject_filename = row[0]
            subject = os.path.splitext(subject_filename)[0]
            subject_labels_filename = row[1]
            subject_labels = os.path.splitext(subject_labels_filename)[0]

            # Linearly align to template and
            # store aligned image and transformation.
            transformed_filename = template + registration_suffix + subject + ".nii"
            transformed_labels_filename = template + registration_suffix + subject_labels + ".nii"

            # In first loop use the original template and subject images.
            if counter == 0:
                command = linear_registration_program + " " + template_filename +  " " + subject_filename + " " + transformed_filename + " " + subject_labels_filename + " " + transformed_labels_filename + registration_options
            # In subsequent loops use the previous average as template and the transformed images as
            # subjects.
            else:
                command = linear_registration_program + " " + average_filename +  " " + transformed_filename + " " + transformed_filename + " " +  subject_filename + " " + transformed_labels_filename + registration_options

            print( command + "\n")
            os.system( command )

            # ToDo: apply registration parameters to segmentation mask.

            # Create long string with all filenames.
            transformed_subject_string = transformed_subject_string + " " + transformed_filename

        # Compute average.
        command = image_calculator_program + " --in " + transformed_subject_string + " --avg " + " --out " + average_filename + calculator_options 
        print( command + "\n")
        os.system( command )
            
        # Compute variance.
        command = image_calculator_program + " --in " + transformed_subject_string + " --var " + " --out " + variance_filename + calculator_options 
        print( command + "\n")
        os.system( command )

        counter += 1

        # Close subject list.
        f.close()

    # Compute warped subject distances from average atlas.
    ComputeSubjectDistancesFromAverageAtlas3D(subject_list_filename, template_filename, registration_suffix)

    return 0


# Non-linearly warp subjects to template.
def WarpAllImagesToTemplate3D(subject_list_filename, template_filename, project_path):

    # Settings and options.
    deformable_registration_options = " >> log.txt"  
    calculator_options = " --intype FLOAT --outtype FLOAT -d 3"
    nIterations = 1 #5,10
    vxm_net_filename = project_path + "Scripts/test/models/vxm_template_fold1/0040.h5"
    # vxm_net_filename = "3d_thigh_mri_myo_segmentum_v5.h5"
    ants_registration_program = project_path + "Scripts/DeformableRegistration_ANTs.py"
    ffd_registration_program = "/usr/local/DeformableRegistration3D_x86_64_dyn_rel/bin/DeformableRegistration3D_DSUMIVIC"
    sdd_registration_program = "/usr/local/DeformableRegistration3D_x86_64_dyn_rel/bin/DeformableRegistration3D_17"
    vxm_registration_program = project_path + "Scripts/vxm_register.py"

    # String conventions.
    registration_suffix = "_Reg_From_"
    warping_suffix = "_Warp_From_"
    variance_suffix = "_Var_"
    warped_subject_string = ""

    # Used programs.
    deformable_registration_program = vxm_registration_program
    image_calculator_program = "/usr/local/ITKApps_x86_64_dyn_rel/ITKApps-build/ImageCalculator/ImageCalculator"
#     calculator_options = " --intype SHORT --outtype SHORT -d 2"

    # Select filename before file extension to be used
    # as subject id.
    template = os.path.splitext(template_filename)[0]
    subject_list = os.path.splitext(subject_list_filename)[0]
    linear_average_filename = subject_list + registration_suffix + template + ".nii"
    average_filename = subject_list + warping_suffix + registration_suffix + template + ".nii" 
    variance_filename = subject_list + warping_suffix + registration_suffix + variance_suffix + template + ".nii" 

    # Apply iterative scheme to maximize overlapping.
    counter = 0
    while counter < nIterations:
        # Read subject list.
        f = open(subject_list_filename, 'r')
        reader = csv.reader(f)
        print("\nIteration #\t" + str(counter) + "\n")
        print(f)

        warped_subject_string = ""
        # For each subject in the list:
        for row in reader:
            subject_filename = row[0]
            subject = os.path.splitext(subject_filename)[0]
            subject_labels_filename = row[1]
            subject_labels = os.path.splitext(subject_labels_filename)[0]
            transformed_filename = template + registration_suffix + subject + ".nii"

            # Warp to template.
            # Store warped image and transformation.
            # fnirt --ref=<some template> --in=<some image>
            warped_filename =  template + warping_suffix + registration_suffix + subject + ".nii"
            field =  template + warping_suffix + registration_suffix + subject + "_Field.nii"
            transformed_labels_filename = template + registration_suffix + subject_labels + ".nii"
            warped_labels_filename = template + warping_suffix + registration_suffix + subject_labels + ".nii"

            # linearly_registered_atlas_filename = subject + registration_suffix + atlas + ".nii"
            # linearly_registered_atlas_labels_filename =  subject + registration_suffix + atlas_labels + ".nii"
            # nonlinearly_registered_atlas_filename =  subject + warping_suffix + registration_suffix + atlas + ".nii"
            # nonlinearly_registered_atlas_field_filename =  subject + warping_suffix + registration_suffix + atlas + "_Field.nii";
            # nonlinearly_registered_atlas_labels_filename = subject + warping_suffix + registration_suffix + atlas_labels  + ".nii"

            if counter==0:
                if deformable_registration_program == vxm_registration_program:
                    command = "python3" + " " + deformable_registration_program + " --fixed " + linear_average_filename + " --moving " + transformed_filename + " --moved " + warped_filename + " --warp " + field + " --moving_labels " + transformed_labels_filename + " --moved_labels " + warped_labels_filename  + " --model " + vxm_net_filename + " --gpu 0"
                else:
                    command = deformable_registration_program + " " + linear_average_filename + " " + transformed_filename + " " + warped_filename + " " + field +  deformable_registration_options
            else:
                if deformable_registration_program == vxm_registration_program:
                    command = "python3" + " " + deformable_registration_program + " --fixed " + average_filename + " --moving " + warped_filename + " --moved " + warped_filename + " --moving_labels " + transformed_labels_filename + " --moved_labels " + warped_labels_filename + " --warp " + field + " --model " + vxm_net_filename + " --gpu 0"
                else:
                    command = deformable_registration_program + " " + average_filename + " " + warped_filename + " " + warped_filename + " " + field + deformable_registration_options

            print( command + "\n")
            os.system( command )

            # Create long string with all filenames.
            warped_subject_string = warped_subject_string + " " + warped_filename

        # Compute average.
        command = image_calculator_program + " --in " + warped_subject_string + " --avg " + " --out " + average_filename + calculator_options 
        print( command + "\n")
        os.system( command )

        # Compute variance.
        command = image_calculator_program + " --in " + warped_subject_string + " --var " + " --out " + variance_filename + calculator_options 
        print( command + "\n")
        os.system( command )

        counter += 1
            
        # Close subject list.
        f.close()

    # Compute warped subject distances from average atlas.
    ComputeSubjectDistancesFromAverageAtlas3D(subject_list_filename, template_filename, warping_suffix + registration_suffix)

    return 0


# Compute distances between the subjects in list from an atlas image.
# This will be used for atlas selection.
def ComputeSubjectDistancesFromAverageAtlas3D(subject_list_filename, template_filename, transform_suffix):

    # Variable naming and initializations.
    image_calculator_program = "/usr/local/ITKApps_x86_64_dyn_rel/ITKApps-build/ImageCalculator/ImageCalculator"
    # warping_suffix = "_Warp_"
    calculator_options = " --intype FLOAT --outtype FLOAT -d 3"

    subject_list = os.path.splitext(subject_list_filename)[0]
    template = os.path.splitext(template_filename)[0]
    average = subject_list + transform_suffix + template
    average_filename = subject_list + transform_suffix + template + ".nii" 
    output_filename = "Distances_" + subject_list + "_" + template + ".csv"

    # Read subject list line-by-line.
    input_file = open(subject_list_filename, 'r')
    reader = csv.reader(input_file)
    print(input_file)

    # Open distance file for writing
    output_file = open(output_filename, 'w')

    # Read average atlas image.
    atlasImageReader = itk.ImageFileReader.IF3.New()
    atlasImageReader.SetFileName( average_filename )
    atlasImageReader.Update()
    atlasImage = atlasImageReader.GetOutput()

    # Instantiate Squared Differences image filter.
    differenceFilter = itk.SquaredDifferenceImageFilter.IF3IF3IF3.New()
    differenceFilter.SetInput1( atlasImage )

    # Instantiate Mattes Mutual Information filter.
    miMetric     = itk.MattesMutualInformationImageToImageMetric.IF3IF3.New()
    miMetric.SetNumberOfHistogramBins = 50
    # transform = itk.IdentityTransform[itk.D,3].New()
    transform = itk.AffineTransform[itk.D,3].New()
    transform.SetIdentity()
    # print transform
    miMetric.SetTransform( transform )
    interpolator = itk.LinearInterpolateImageFunction.IF3D.New()
    miMetric.SetInterpolator( interpolator )
    numberOfSamples = 0.8 * atlasImage.GetLargestPossibleRegion().GetNumberOfPixels() 
    miMetric.SetNumberOfSpatialSamples( int(numberOfSamples) );
    miMetric.SetFixedImage( atlasImage )
    miMetric.SetFixedImageRegion(  atlasImage.GetBufferedRegion()  )

    # For each subject in the list:
    for row in reader:
        subject_filename = row[0]
        subject = os.path.splitext(subject_filename)[0]

        # Compose warped subject and atlas filenames.
        warped = template + transform_suffix + subject
        warped_filename =  warped + ".nii"

        # Write image pair to output file.
        line_string = warped + "," + average + ","
        output_file.write(line_string)

        # Compute pair-wise distances and write results to new file.
        # command = image_calculator_program + " --in " + warped_filename + " " + average_filename + " --out Distance.nii --ofsqr --statAVG" + calculator_options + " >> " + output_filename
        # print( command + "\n")
        # os.system( command )

        # Small itk pipeline for computing squared distance averages.
        # Read subject image.
        subjectImageReader = itk.ImageFileReader.IF3.New()
        subjectImageReader.SetFileName( warped_filename )
        subjectImageReader.Update()

        # Compute squared differences.
        differenceFilter.SetInput2( subjectImageReader.GetOutput() )
        differenceFilter.Update()

        # Write difference image to file.
        differenceImageWriter = itk.ImageFileWriter.IF3.New()
        differenceImageWriter.SetInput( differenceFilter.GetOutput() )
        differenceFileName = warped + "_" + average + "_Diff.nii"
        differenceImageWriter.SetFileName( differenceFileName )
        differenceImageWriter.Update()

        # MI
        miMetric.SetMovingImage( subjectImageReader.GetOutput() )
        miMetric.Initialize()
        # miMetric.Update()

        # Compute average over image.
        statisticsFilter = itk.StatisticsImageFilter.IF3.New()
        statisticsFilter.SetInput(differenceFilter.GetOutput())
        statisticsFilter.Update()
        mean = statisticsFilter.GetMean()
        # stdev = statisticsFilter.GetSigma()
        parameters = transform.GetParameters()
        # print transform
        mi = miMetric.GetValue( parameters )

        # Write average to output file.
        output_file.write( '%.3f,%.3f\n' % (mean,mi) )
        print('%s, %s, Mean Sq. Err.: %.3f, MI: %.3f' % (warped,average,mean,mi)) 

        
    input_file.close()
    output_file.close()

    return 0


def ApplyJLFToAllWarpedImages(subject_list_filename, template_filename, project_path):

    # String conventions.
    registration_suffix = "_Reg_From_"
    warping_suffix = "_Warp_From_"

    # Select filename before file extension to be used
    # as subject id.
    template = os.path.splitext(template_filename)[0]
    subject_list = os.path.splitext(subject_list_filename)[0]

    # Read subject list.
    in_file = open(subject_list_filename, 'r')
    reader = csv.reader(in_file)
    print(in_file)

    # Open file for writing.
    warped_subject_list_filename = 'WarpedSubjectList.csv'
    csv_file = open(warped_subject_list_filename, 'w')    
    print(csv_file)

    # For each subject in the list:
    for row in reader:
        subject_filename = row[0]
        subject = os.path.splitext(subject_filename)[0]
        subject_labels_filename = row[1]
        subject_labels = os.path.splitext(subject_labels_filename)[0]

        # Store warped image and transformation.
        warped_filename =  template + warping_suffix + registration_suffix + subject + ".nii"
        warped_labels_filename = template + warping_suffix + registration_suffix + subject_labels + ".nii"
        line = warped_filename + "," + warped_labels_filename + '\n'
        csv_file.write(line)

    # Close subject list.
    in_file.close()
    csv_file.close()

    # String conventions.
    registration_suffix = "_Reg_From_"
    warping_suffix = "_Warp_From_"

    # Apply joint label fusion.
    template_id = "" #template + warping_suffix + registration_suffix
    tissue_labels = [1,2,3,4,5,6,7,8]
    JointMultiLabelFusion3D(template_id, warped_subject_list_filename, tissue_labels, template_filename)

    return 0


# Main function.
if __name__ == "__main__":

    # Check for command syntax.
    if len(sys.argv) < 3:
        print('Usage:\n' + str(sys.argv[0]) + ' <Template Filename> <Subject List Filename>\n')
        sys.exit(0)

    # Get/Set parameters.
    template_filename = sys.argv[1]
    subject_list_filename = sys.argv[2]
    # project_path = "~/Codes/src/gitrepos/ShapeModelingMIVIC/"
    project_path = "/home/users/mibrahim/segmentation/Knee/ShapeModelingMIVIC/"

    # First pass: create average image using a subject as template.
    AlignAllImagesToTemplate3D(subject_list_filename, template_filename, project_path)

    # Then: Iteratively align all subject images to template to generate an average image (atlas).
    WarpAllImagesToTemplate3D(subject_list_filename, template_filename, project_path)

    # Apply Joint Label Fusion to all warped images.
    ApplyJLFToAllWarpedImages(subject_list_filename, template_filename, project_path)
