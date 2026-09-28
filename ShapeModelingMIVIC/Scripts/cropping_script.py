#!/usr/bin/env python

import itk 


def symmetric_crop(input_image, no_of_pixels_x, no_of_pixels_y, no_of_pixels_z):
    '''

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    no_of_pixels_x : TYPE
        DESCRIPTION.
    no_of_pixels_y : TYPE
        DESCRIPTION.
    no_of_pixels_z : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    # no_of_pixels_x = 80
    # no_of_pixels_y = 80
    # no_of_pixels_z = 12
    
    
    
    input_region = input_image.GetBufferedRegion()
    
    size = input_region.GetSize()
    start = input_region.GetIndex()
    start[0] = start[0] + no_of_pixels_x
    start[1] = start[1] + no_of_pixels_y
    start[2] = start[2] + no_of_pixels_z
    size[0] = size[0]-2*no_of_pixels_x
    size[1] = size[1]-2*no_of_pixels_y
    size[2] = size[2]-2*no_of_pixels_z
    
    desired_region = input_region
    desired_region.SetSize(size)
    desired_region.SetIndex(start)
    
    extractFilter = itk.ExtractImageFilter.New(input_image)
    extractFilter.SetExtractionRegion(desired_region)
    extractFilter.Update()
    cropped_image = extractFilter.GetOutput()
    return cropped_image
    
