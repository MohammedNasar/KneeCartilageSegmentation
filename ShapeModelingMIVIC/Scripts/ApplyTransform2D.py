#!/usr/bin/env python

#==========================================================================
#   Read input image and affine transform and apply resampling.
#   Adapted from ITK examples.
#==========================================================================*/

# import sys
# sys.path.insert(0,'/home/sokratis/Codes/bin/ITK_x86_64_dyn_rel/lib')

import itk
from ReadITKTransform import ReadITKTransform2D
# import readITKtransform
from sys import argv,exit


# Check for command syntax.
if len(argv) < 5:
    print('Usage:\n' + str(argv[0]) + ' <Source Image> <Target Image> <Transform File> <Transformed Image> [interpolation(LINEAR,NN)]\n')
    exit(0)


# Read source image.
sourceImageReader = itk.ImageFileReader.ISS2.New()
sourceImageReader.SetFileName(  argv[1] )
sourceImageReader.Update()
sourceImage = sourceImageReader.GetOutput() 

# Read target image.
targetImageReader = itk.ImageFileReader.ISS2.New()
targetImageReader.SetFileName(  argv[2] )
targetImageReader.Update()
targetImage = targetImageReader.GetOutput() 

# Read transform.
transformFromFile = ReadITKTransform2D( argv[3] )
#print( transformFromFile )

#transformReader = itk.TransformFileReaderTemplate[itk.F].New()
#transformReader.SetFileName( argv[3] )
#transformReader.Update()
#transformList = transformReader.GetTransformList()
#print( transformList )

# Resample source image.
resampler = itk.ResampleImageFilter.ISS2ISS2.New()
transform = itk.AffineTransform[itk.D,2].New()
resampler.SetInput( sourceImage )
resampler.SetTransform( transform )
resampler.SetDefaultPixelValue( 0 )  # -400 for pQCT.

parameters = transform.GetParameters()
for i in range (6):
    parameters.SetElement(i, transformFromFile[0][i])
    # print(parameters.GetElement(i))

transform.SetParameters(parameters)

fixed_parameters = transform.GetFixedParameters()
for i in range (2):
    fixed_parameters.SetElement(i, transformFromFile[1][i])
    # print(fixed_parameters.GetElement(i))

transform.SetFixedParameters(fixed_parameters)

# Set interpolator if made explicit (default is linear).
nn_interpolator = itk.NearestNeighborInterpolateImageFunction.ISS2D.New() 
linear_interpolator = itk.LinearInterpolateImageFunction.ISS2D.New() 

if len(argv)==6:
    if str(argv[5]).upper() == 'NN':
        print("Using nearest neighbor interpolator.")
        resampler.SetInterpolator( nn_interpolator )
        resampler.SetDefaultPixelValue( 0 )
    elif str(argv[5]).upper() == 'LINEAR':
        print("Using linear interpolator.")
        resampler.SetInterpolator( linear_interpolator )
        resampler.SetDefaultPixelValue( 0 )  # -400 for pQCT.
    else:
        print("Unknown interpolation function, exiting")
        exit(0)

# Set geometrical attibutes and update.
region = targetImage.GetLargestPossibleRegion()
resampler.SetSize( region.GetSize() )
resampler.SetOutputSpacing( targetImage.GetSpacing() )
resampler.SetOutputDirection( targetImage.GetDirection() )
resampler.SetOutputOrigin(  targetImage.GetOrigin() )
resampler.Update()

# Copy target header attributes to resampled image.
resampledImage = resampler.GetOutput()

#
# Cast for output
#
# outputCast = itk.RescaleIntensityImageFilter.IF2IF2.New()
# outputCast.SetInput( resampler.GetOutput() )
# outputCast.SetOutputMinimum( 0 )
# outputCast.SetOutputMaximum( 65535 )
# outputCast = itk.CastImageFilter.IF2ISS2.New()
# outputCast.SetInput( resampler.GetOutput() )
# outputCast.Update()


# Write transormed image.
writer = itk.ImageFileWriter.ISS2.New()
writer.SetFileName( argv[4] )
writer.SetInput(resampler.GetOutput())
writer.Update()
