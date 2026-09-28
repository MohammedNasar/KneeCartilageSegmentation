#!/usr/bin/env python

#==========================================================================
#   Read input image and affine transform and apply resampling.
#   Adapted from ITK examples.
#   S. Makrogiannis, MIVIC/DPCS/DSU (smakrogiannis@desu.edu)
#==========================================================================*/

import numpy
import itk

def ReadITKTransform3D( transform_file ):
    '''
    '''
    # read the transform
    transform = None
    with open( transform_file, 'r' ) as f:
        for line in f:
        
        # check for Parameters:
            if line.startswith( 'Parameters:' ):
                values = line.split( ': ' )[1].split( ' ' )
                # filter empty spaces and line breaks
                parameter_values = [float( e ) for e in values if ( e != '' and e != '\n' )]
                # create the upper left of the matrix
                transform_upper_left = numpy.reshape( parameter_values[0:9], ( 3, 3 ) )
                # grab the translation as well
                translation = parameter_values[9:]
                
            # check for FixedParameters:
            if line.startswith( 'FixedParameters:' ):
                values = line.split( ': ' )[1].split( ' ' )
                # filter empty spaces and line breaks
                fixed_parameter_values = [float( e ) for e in values if ( e != '' and e != '\n' )]
                # setup the center
                center = fixed_parameter_values
                
    # compute the offset
    offset = numpy.ones( 4 )
    for i in range( 0, 3 ):
        offset[i] = translation[i] + center[i];
        for j in range( 0, 3 ):
            offset[i] -= transform_upper_left[i][j] * center[i]

    # add the [0, 0, 0] line
    # transform = numpy.vstack( ( transform_upper_left, [0, 0, 0] ) )
    # and the [offset, 1] column
    # transform = numpy.hstack( ( transform, numpy.reshape( offset, ( 4, 1 ) ) ) )
   
    transform = [parameter_values, fixed_parameter_values]
    return transform


def ReadITKTransform2D( transform_file ):
    '''
    '''
    # read the transform
    with open( transform_file, 'r' ) as f:
        for line in f:

            # check for Parameters:
            if line.startswith( 'Parameters:' ):
                values = line.split( ': ' )[1].split( ' ' )

                # filter empty spaces and line breaks
                parameter_values = [float( e ) for e in values if ( e != '' and e != '\n' )]
                # create the upper left of the matrix
                transform_upper_left = numpy.reshape( parameter_values[0:4], ( 2, 2 ) )
                # grab the translation as well
                translation = parameter_values[4:]

            # check for FixedParameters:
            if line.startswith( 'FixedParameters:' ):
                values = line.split( ': ' )[1].split( ' ' )

                # filter empty spaces and line breaks
                fixed_parameter_values = [float( e ) for e in values if ( e != '' and e != '\n' )]
                # setup the center
                center = fixed_parameter_values

    # compute the offset
    offset = numpy.ones( 2 )
    for i in range( 0, 1 ):
        offset[i] = translation[i] + center[i];
        for j in range( 0, 1 ):
            offset[i] -= transform_upper_left[i][j] * center[i]

#    # add the [0, 0, 0] line
#    transform = numpy.vstack( ( transform_upper_left, [0, 0] ) )
#    # and the [offset, 1] column
#    transform = numpy.hstack( ( transform, numpy.reshape( offset, ( 2, 1 ) ) ) )
    
    transform = [parameter_values, fixed_parameter_values]
    
    return transform

def CopyImageHeaderInfo2D(filename1,filename2):
    '''
    '''
    reader = itk.ImageFileReader.ISS2.New()
    reader.SetFileName(filename1)
    InputImage1 = reader.Update()
    InputImage1 = reader.GetOutput()

    reader2 = itk.ImageFileReader.ISS2.New()
    reader2.SetFileName(filename2)
    InputImage2 = reader2.Update()
    InputImage2 = reader2.GetOutput()

    InputImage2.SetOrigin(InputImage1.GetOrigin())
    InputImage2.SetDirection(InputImage1.GetDirection())
    InputImage2.SetSpacing(InputImage1.GetSpacing())

    writer = itk.ImageFileWriter.ISS2.New()
    writer.SetFileName(filename2)
    writer.SetInput(InputImage2)
    writer.Update()

    return 0

def CopyImageHeaderInfo3D(filename1,filename2):
    '''
    '''
    reader = itk.ImageFileReader.ISS3.New()
    reader.SetFileName(filename1)
    InputImage1 = reader.Update()
    InputImage1 = reader.GetOutput()

    reader2 = itk.ImageFileReader.ISS3.New()
    reader2.SetFileName(filename2)
    InputImage2 = reader2.Update()
    InputImage2 = reader2.GetOutput()

    InputImage2.SetOrigin(InputImage1.GetOrigin())
    InputImage2.SetDirection(InputImage1.GetDirection())
    InputImage2.SetSpacing(InputImage1.GetSpacing())

    writer = itk.ImageFileWriter.ISS3.New()
    writer.SetFileName(filename2)
    writer.SetInput(InputImage2)
    writer.Update()

    return 0