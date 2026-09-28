#!/usr/bin/env python

#==========================================================================
#   Adapted from ITK examples.
#   S. Makrogiannis, MIVIC/PEMACS/DSU (smakrogiannis@desu.edu)
#==========================================================================*/

# from InsightToolkit import *
import itk
from sys import argv,exit


# Check for command syntax.
if len(argv) < 4:
    print('Usage:\n' + str(argv[0]) + ' <Template Image> <Subject Image> <Registered Image>\n')
    exit(0)

#
# Read the fixed and moving images using filenames
# from the command line arguments
#
fixedImageReader = itk.ImageFileReader.IF3.New()
movingImageReader = itk.ImageFileReader.IF3.New()

fixedImageReader.SetFileName(  argv[1] )
movingImageReader.SetFileName( argv[2] )

fixedImageReader.Update()
movingImageReader.Update()

fixedImage = fixedImageReader.GetOutput() 
movingImage = movingImageReader.GetOutput()

# Get pixel intensity minimum to use as background in resampling.
minmaxCalculator = itk.MinimumMaximumImageCalculator.IF3.New()
minmaxCalculator.SetImage(movingImage)
minmaxCalculator.Compute()
minMovingImage = minmaxCalculator.GetMinimum()
# Get pixel intensity maximum

# Normalize intensities of moving and target image.


#
#  Instantiate the classes for the registration framework
#
registration    = itk.ImageRegistrationMethod.IF3IF3.New()
imageMetric     = itk.MattesMutualInformationImageToImageMetric.IF3IF3.New()
# imageMetric     = itk.MeanSquaresImageToImageMetric.IF2IF2.New()
transform       = itk.AffineTransform[itk.D,3].New()
# transform       = itk.CenteredSimilarity2DTransform.D.New()
optimizer       = itk.RegularStepGradientDescentOptimizer.New()
interpolator    = itk.LinearInterpolateImageFunction.IF3D.New()

registration.SetOptimizer(      optimizer )
registration.SetTransform(      transform )
registration.SetInterpolator(   interpolator )
registration.SetMetric(         imageMetric )
registration.SetFixedImage(  fixedImage )
registration.SetMovingImage( movingImage )
registration.SetFixedImageRegion(  fixedImage.GetBufferedRegion() )

# float_image_type     = itk.Image[itk.F, 2]
# transform_type = itk.CenteredSimilarity2DTransform
# initializer    = itk.CenteredTransformInitializer[transform_type,float_image_type,float_image_type].New()
# initializer.SetTransform( transform )
# initializer.SetFixedImage( fixedImage )
# initializer.SetMovingImage( movingImage )
# initializer.MomentsOn()
# initializer.InitializeTransform()

#
# Initial transform parameters 
#
# transform.SetAngle( 0.0 );
# transform.SetScale( 1.0 );

# registration.SetInitialTransformParameters( transform.GetParameters() )

# center of the fixed image
fixedSpacing = fixedImage.GetSpacing()
fixedOrigin = fixedImage.GetOrigin()
fixedSize = fixedImage.GetLargestPossibleRegion().GetSize()

centerFixed = ( fixedOrigin.GetElement(0) + fixedSpacing.GetElement(0) * fixedSize.GetElement(0) / 3.0,
                fixedOrigin.GetElement(1) + fixedSpacing.GetElement(1) * fixedSize.GetElement(1) / 3.0,
                fixedOrigin.GetElement(2) + fixedSpacing.GetElement(2) * fixedSize.GetElement(2) / 3.0)

# center of the moving image 
movingSpacing = movingImage.GetSpacing()
movingOrigin = movingImage.GetOrigin()
movingSize = movingImage.GetLargestPossibleRegion().GetSize()

centerMoving = ( movingOrigin.GetElement(0) + movingSpacing.GetElement(0) * movingSize.GetElement(0) / 3.0,
                 movingOrigin.GetElement(1) + movingSpacing.GetElement(1) * movingSize.GetElement(1) / 3.0,  
                 movingOrigin.GetElement(2) + movingSpacing.GetElement(2) * movingSize.GetElement(2) / 3.0)

# transform center
center = transform.GetCenter()
center.SetElement( 0, centerFixed[0] )
center.SetElement( 1, centerFixed[1] )
center.SetElement( 2, centerFixed[2] )

# transform translation
translation = transform.GetTranslation()
translation.SetElement( 0, centerMoving[0] - centerFixed[0] )
translation.SetElement( 1, centerMoving[1] - centerFixed[1] )
translation.SetElement( 2, centerMoving[2] - centerFixed[2] )

initialParameters = transform.GetParameters()

print ("Initial Parameters: ")
# print "Angle: %f" % (initialParameters.GetElement(1), )
print ("Center: %f, %f, %f" % ( transform.GetCenter().GetElement(0), transform.GetCenter().GetElement(1), transform.GetCenter().GetElement(2) ))
print ("Translation: %f, %f, %f" % (initialParameters.GetElement(9), initialParameters.GetElement(10), initialParameters.GetElement(11)))

registration.SetInitialTransformParameters( initialParameters )

#
# Define optimizer parameters
#

# optimizer scale
translationScale = 1.0 / 1000.0  # 100.0

optimizerScales = itk.Array.D( transform.GetNumberOfParameters() )
# optimizerScales.SetElement(0, 10.0)
# optimizerScales.SetElement(1, 1.0)
# optimizerScales.SetElement(2, translationScale)
# optimizerScales.SetElement(3, translationScale)
# optimizerScales.SetElement(4, translationScale)
# optimizerScales.SetElement(5, translationScale)

optimizerScales.SetElement(0, 1.0)
optimizerScales.SetElement(1, 1.0)
optimizerScales.SetElement(2, 1.0)
optimizerScales.SetElement(3, 1.0)
optimizerScales.SetElement(4, 1.0)
optimizerScales.SetElement(5, 1.0)
optimizerScales.SetElement(6, 1.0)
optimizerScales.SetElement(7, 1.0)
optimizerScales.SetElement(8, 1.0)
optimizerScales.SetElement(9, translationScale)
optimizerScales.SetElement(10, translationScale)
optimizerScales.SetElement(11, translationScale)

optimizer.SetScales( optimizerScales )
optimizer.SetMaximumStepLength( 0.1 )  # Previously: 0.1, 1.0
optimizer.SetMinimumStepLength( 0.0001 )
optimizer.SetNumberOfIterations( 700 ) # Previously: 350

# Set image metric parameters.
imageMetric.SetNumberOfHistogramBins( 50 )
sampleNumber = int( 0.6 * fixedImage.GetBufferedRegion().GetNumberOfPixels() )
imageMetric.SetNumberOfSpatialSamples( sampleNumber )

#
# Iteration Observer
#
def iterationUpdate():
    currentParameter = transform.GetParameters()
    print ("M: %f   P: %f %f %f %f %f %f %f %f %f %f %f %f" % ( optimizer.GetValue(),
                                 currentParameter.GetElement(0),
                                 currentParameter.GetElement(1),
                                 currentParameter.GetElement(2),
                                 currentParameter.GetElement(3),
                                 currentParameter.GetElement(4),
                                 currentParameter.GetElement(5),
                                 currentParameter.GetElement(6),
                                 currentParameter.GetElement(7),
                                 currentParameter.GetElement(8),
                                 currentParameter.GetElement(9),
                                 currentParameter.GetElement(10),
                                 currentParameter.GetElement(11)))
 
iterationCommand = itk.PyCommand.New()
iterationCommand.SetCommandCallable( iterationUpdate )
optimizer.AddObserver( itk.IterationEvent(), iterationCommand )

print ("Starting registration")

#
# Start the registration process
#

# registration.StartRegistration()
registration.Update()

#
# Get the final parameters of the transformation
#
finalParameters = registration.GetLastTransformParameters()

print ("Final Registration Parameters ")
print ("Matrix(0,0)  = %f" % finalParameters.GetElement(0))
print ("Matrix(0,1)  = %f" % finalParameters.GetElement(1))
print ("Matrix(0,2)  = %f" % finalParameters.GetElement(2))
print ("Matrix(1,0)  = %f" % finalParameters.GetElement(3))
print ("Matrix(1,1)  = %f" % finalParameters.GetElement(4))
print ("Matrix(1,2)  = %f" % finalParameters.GetElement(5))
print ("Matrix(2,0)  = %f" % finalParameters.GetElement(6))
print ("Matrix(2,1)  = %f" % finalParameters.GetElement(7))
print ("Matrix(2,2)  = %f" % finalParameters.GetElement(8))
print ("Translation in  Z = %f" % finalParameters.GetElement(9))
print ("Translation in  Y = %f" % finalParameters.GetElement(10))
print ("Translation in  X = %f" % finalParameters.GetElement(11))

# Now, we use the final transform for resampling the moving image.
resampler = itk.ResampleImageFilter.IF3IF3.New()

# print( transform )
resampler.SetTransform( transform )
resampler.SetInput( movingImage )

region = fixedImage.GetLargestPossibleRegion()

resampler.SetSize( region.GetSize() )
resampler.SetOutputSpacing( fixedImage.GetSpacing() )
resampler.SetOutputDirection( fixedImage.GetDirection() )
resampler.SetOutputOrigin(  fixedImage.GetOrigin() )
resampler.SetDefaultPixelValue( minMovingImage )   # 0 for MRI, -400 for pQCT, -1000 for CT

# Map pixel intensities back to original scale.


#
# Cast for output
#
outputCast = itk.CastImageFilter.IF3ISS3.New()
outputCast.SetInput( resampler.GetOutput() )
outputCast.Update()

# Copy the transformed image pointer.
# transformedImage = resampler.GetOutput()
transformedImage = outputCast.GetOutput()
# transformedImage.CopyInformation(fixedImage)

# Write affine transformation to file.
transformWriter = itk.TransformFileWriterTemplate[itk.F].New()
transformWriter.SetInput( transform )
transformWriter.SetFileName( argv[3]+".tfm" )
transformWriter.Update()

# Write transformed image.
writer = itk.ImageFileWriter.ISS3.New()
# writer = itk.ImageFileWriter.IF2.New()
writer.SetFileName( argv[3] )
writer.SetInput( transformedImage )

writer.Update()
