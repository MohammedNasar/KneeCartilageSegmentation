#!/usr/bin/env python

#==========================================================================
#   Build atlas from multiple subjects.
#   S. Makrogiannis, MIVIC/DPCS/DSU
#==========================================================================

import sys,os,itk


# Linearly register all subjects to template.
def AlignAllImagesToTemplate(subject_list_filename, template_filename):

    # Used programs.
    registration_program = "~/Codes/src/gitrepos/ShapeModelingMIVIC/Scripts/LinearRegistration2D.py"
    # registration_program = "~/Software/bin/LinearRegistration2D_x86_64/bin/ImageRegistration2D_14" # ImageRegistration2D_9,ImageRegistration2D_14,MultiResImageRegistration2D_2
    image_calculator_program = "/usr/local/ITKApps_x86_64_dyn_rel/ITKApps-build/ImageCalculator/ImageCalculator"
    registration_options = " >> log.txt" # " -2D"
    # calculator_options = " --intype SHORT --outtype SHORT -d 2"
    calculator_options = " --intype FLOAT --outtype FLOAT -d 2"
    nIterations = 1

    # String conventions.
    registration_suffix = "_Reg_"
    variance_suffix = "_Var_"

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
        print("\nIteration #\t" + str(counter) + "\n")
        print f
        
        transformed_subject_string = ""
        # For each subject in the list:
        for line in f:
            subject_filename = line.rstrip('\n')
            subject = os.path.splitext(subject_filename)[0]

            # Linearly align to template and
            # store aligned image and transformation.
            # flirt [options] -in <inputvol> -ref <refvol> -out <outputvol> -omat <outputmatrix> -2D
            transformed_filename = subject + registration_suffix + template + ".nii"

            # In first loop use the original template and subject images.
            if counter == 0:
                command = registration_program + " " + template_filename +  " " + subject_filename + " " + transformed_filename + registration_options
            # In subsequent loops use the previous average as template and the transformed images as
            # subjects.
            else:
                command = registration_program + " " + average_filename +  " " + transformed_filename + " " + transformed_filename + registration_options

            print( command + "\n")
            os.system( command )

            # Apply registration parameters to segmentation mask.

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
    ComputeSubjectDistancesFromAverageAtlas(subject_list_filename, template_filename, registration_suffix)

    return 0


# Non-linearly warp subjects to template.
def WarpAllImagesToTemplate(subject_list_filename, template_filename):

    # Used programs.
    warping_program = "/usr/local/DeformableRegistration2D_x86_64_dyn_rel/bin/DeformableRegistration2D_DSUMIVIC" 
    # FFD: DeformableRegistration2D_DSUMIVIC,DeformableRegistration2D_13,DeformableRegistration2D_12,DeformableRegistration2D_6,DeformableRegistration2D_14,DeformableRegistration2D_15
    # DD: DeformableRegistration2D_17,DeformableRegistration2D_3,DeformableRegistration2D_2,DeformableRegistration2D_16

    warping_options = " >> log.txt"  

    image_calculator_program = "/usr/local/ITKApps_x86_64_dyn_rel/ITKApps-build/ImageCalculator/ImageCalculator"
#     calculator_options = " --intype SHORT --outtype SHORT -d 2"
    calculator_options = " --intype FLOAT --outtype FLOAT -d 2"
    nIterations = 5 #5,10

    # String conventions.
    registration_suffix = "_Reg_"
    warping_suffix = "_Warp_"
    variance_suffix = "_Var_"
    warped_subject_string = ""

    # Select filename before file extension to be used
    # as subject id.
    template = os.path.splitext(template_filename)[0]
    subject_list = os.path.splitext(subject_list_filename)[0]
    linear_average_filename = subject_list + registration_suffix + template + ".nii"
    average_filename = subject_list + warping_suffix + template + ".nii" 
    variance_filename = subject_list + warping_suffix + variance_suffix + template + ".nii" 

    # Apply iterative scheme to maximize overlapping.
    counter = 0
    while counter < nIterations:
        # Read subject list.
        f = open(subject_list_filename, 'r')
        print("\nIteration #\t" + str(counter) + "\n")
        print f

        warped_subject_string = ""
        # For each subject in the list:
        for line in f:
            subject_filename = line.rstrip('\n')
            subject = os.path.splitext(subject_filename)[0]
            transformed_filename = subject + registration_suffix + template + ".nii"

            # Warp to template.
            # Store warped image and transformation.
            # fnirt --ref=<some template> --in=<some image>
            warped_filename =  subject + warping_suffix + template + ".nii"
            field =  subject + warping_suffix + template + ".field.mhd"
            if counter==0:
#                if warping_program==diffeomorphic_demons_program:
#                    command = warping_program + " -f " + linear_average_filename + " -m " + transformed_filename + " -o " + warped_filename + " -O " + field + diffeomorphic_demons_options
#                    command = warping_program + " " + linear_average_filename + " " + transformed_filename + " " + warped_filename + " " + field + warping_options
#                else:
                command = warping_program + " " + linear_average_filename + " " + transformed_filename + " " + warped_filename + " " + field + warping_options

            else:
#                if warping_program==diffeomorphic_demons_program:
#                    command = warping_program + " -f " + average_filename + " -m " + warped_filename + " -o " + warped_filename + " -O " + field + diffeomorphic_demons_options
#                    command = warping_program + " " + average_filename + " " + warped_filename + " " + warped_filename + " " +  field + warping_options
#                else:
                command = warping_program + " " + average_filename + " " + warped_filename + " " + warped_filename + " " + field + warping_options
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
    ComputeSubjectDistancesFromAverageAtlas(subject_list_filename, template_filename, warping_suffix)

    return 0


# Compute distances between the subjects in list from an atlas image.
# This will be used for atlas selection.
def ComputeSubjectDistancesFromAverageAtlas(subject_list_filename, template_filename, transform_suffix):

    # Variable naming and initializations.
    image_calculator_program = "/usr/local/ITKApps_x86_64_dyn_rel/ITKApps-build/ImageCalculator/ImageCalculator"
    # warping_suffix = "_Warp_"
    calculator_options = " --intype FLOAT --outtype FLOAT -d 2"

    subject_list = os.path.splitext(subject_list_filename)[0]
    template = os.path.splitext(template_filename)[0]
    average = subject_list + transform_suffix + template
    average_filename = subject_list + transform_suffix + template + ".nii" 
    output_filename = "Distances_" + subject_list + "_" + template + ".csv"

    # Read subject list line-by-line.
    input_file = open(subject_list_filename, 'r')
    print input_file

    # Open distance file for writing
    output_file = open(output_filename, 'w')

    # Read average atlas image.
    atlasImageReader = itk.ImageFileReader.IF2.New()
    atlasImageReader.SetFileName( average_filename )
    atlasImageReader.Update()
    atlasImage = atlasImageReader.GetOutput()

    # Instantiate Squared Differences image filter.
    differenceFilter = itk.SquaredDifferenceImageFilter.IF2IF2IF2.New()
    differenceFilter.SetInput1( atlasImage )

    # Instantiate Mattes Mutual Information filter.
    miMetric     = itk.MattesMutualInformationImageToImageMetric.IF2IF2.New()
    miMetric.SetNumberOfHistogramBins = 50
    # transform = itk.IdentityTransform[itk.D,2].New()
    transform = itk.AffineTransform[itk.D,2].New()
    transform.SetIdentity()
    # print transform
    miMetric.SetTransform( transform )
    interpolator = itk.LinearInterpolateImageFunction.IF2D.New()
    miMetric.SetInterpolator( interpolator )
    numberOfSamples = 0.8 * atlasImage.GetLargestPossibleRegion().GetNumberOfPixels() 
    miMetric.SetNumberOfSpatialSamples( long(numberOfSamples) );
    miMetric.SetFixedImage( atlasImage )
    miMetric.SetFixedImageRegion(  atlasImage.GetBufferedRegion()  )

    # For each subject in the list:
    for line in input_file:
        subject_filename = line.rstrip('\n')
        subject = os.path.splitext(subject_filename)[0]

        # Compose warped subject and atlas filenames.
        warped = subject + transform_suffix + template
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
        subjectImageReader = itk.ImageFileReader.IF2.New()
        subjectImageReader.SetFileName( warped_filename )
        subjectImageReader.Update()

        # COmpute squared differences.
        differenceFilter.SetInput2( subjectImageReader.GetOutput() )
        differenceFilter.Update()

        # Write difference image to file.
        differenceImageWriter = itk.ImageFileWriter.IF2.New()
        differenceImageWriter.SetInput( differenceFilter.GetOutput() )
        differenceFileName = warped + "_" + average + "_Diff.nii"
        differenceImageWriter.SetFileName( differenceFileName )
        differenceImageWriter.Update()

        # MI
        miMetric.SetMovingImage( subjectImageReader.GetOutput() )
        miMetric.Initialize()
        # miMetric.Update()

        # Compute average over image.
        statisticsFilter = itk.StatisticsImageFilter.IF2.New()
        statisticsFilter.SetInput(differenceFilter.GetOutput())
        statisticsFilter.Update()
        mean = statisticsFilter.GetMean()
        # stdev = statisticsFilter.GetSigma()
        parameters = transform.GetParameters()
        # print transform
        mi = miMetric.GetValue( parameters )

        # Write average to output file.
        output_file.write( '%.3f,%.3f\n' % (mean,mi) )
        print '%s, %s, Mean Sq. Err.: %.3f, MI: %.3f' % (warped,average,mean,mi) 

        
    input_file.close()
    output_file.close()

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

    # First pass: create average image using a subject as template.
    AlignAllImagesToTemplate(subject_list_filename, template_filename)

    # Then: Iteratively align all subject images to template to generate an average image (atlas).
    WarpAllImagesToTemplate(subject_list_filename, template_filename)

