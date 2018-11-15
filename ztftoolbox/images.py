#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# collection of python functions to process and manipulate ztf image data.
#
# Author: M. Giomi (matteo.giomi@desy.de)


import os
import numpy as np
from scipy.signal import medfilt
from astropy.io import fits
from astropy.visualization import ZScaleInterval, ImageNormalize

import logging
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)


def bin_ndarray(ndarray, new_shape, operation='sum'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions and 
        new axes must divide old ones.

    taken from:
    https://stackoverflow.com/questions/8090229/resize-with-averaging-or-rebin-a-numpy-2d-array
    """
    operation = operation.lower()
    if not operation in ['sum', 'mean', 'median']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                  ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1*(i+1))
    return ndarray

def preproc(infile, quadrant=None, medfilter_kernel_size=11, texp=10, 
    doubling_temp_deg=7., reftemp=150.):
    """
        given a raw image file, applies ovserscan and dark current correction. 
        
        The overscan correction removes from each pixel column 
        in the images the median of the corresponding overscan region. 
        
        The darkcurrent contribution is computed using the given 
        doubling and reference temperatures.
        
        Parameters:
        -----------
        
        infile: `str`
            fits file with the raw images for the 4 readout quadrants.
        
        quandrant: `int`
            number of readout quadrant you need to process. If None, all the quadrants
            are used. The ordering of the quadrant on the chips is as follows:
                                            2 1
                                            3 4
        
        medfilter_kernel_size: `int` (odd)
            size (in pixel) of the median filter used to smooth the overscan profile.
            Needs to be an odd number!
        
        texp: `float`
            exposure time in second, if None then read it from the fits header.
            
        doubling_temp: `float`
            doubling temperature for the CCDs
            
        reftemp: `float`:
            reference temperature where the dark current is known.
        
        Return:
        -------
            
            np.array(s) with the pre-proc image. If quadrant!=None, just the desired
            quadrant will be returned, else all four.
    """
    
    # open the file
    hudl=fits.open(infile)
    logger.debug("Applying overscan and dark-current correction to: %s"%infile)
    
    # get the temperatures for this ccd
    ccd=infile.split("_c")[-1].split("_")[0]
    ccdtemp=float(hudl[0].header['CCDTMP'+ccd])
    logger.debug("temperature for CCD %s: %.2f K"%(ccd, ccdtemp))
    
    # get the exposure time for the image
    if texp is None:
        texp=float(hudl[0].header['EXPOSURE'])
    logger.debug("exposure time for image: %.2f sec"%texp)
    
    # select the quadrants
    if quadrant is None:
        quadrants == list(range(1, 5))
    else:
        quadrants = [quadrant]
    
    # remove the ovserscan for each readout quadrant
    preprocessed = []
    for iimg in quadrants:
        iosc=iimg+4
        logger.debug("reading readout quadrant: %d, corresponding overscan %d"%
            (iimg, iosc))
        img=hudl[iimg].data
        oscan=hudl[iosc].data
        
        # compute dark current
        gain=float(hudl[iimg].header['GAIN'])
        dc_e=float(hudl[iimg].header['DARKCUR'])
        darkc = texp * ( dc_e / ( 2.**((reftemp-ccdtemp)/doubling_temp_deg) ) ) / gain
        logger.debug("gain for readout quadrant %d: %.2f e-/ADU"%(iimg, gain))
        logger.debug("dark current @ 150 K: %.2f"%dc_e)
        logger.debug("dark current at %.2f K: %.2f"%(ccdtemp, darkc))
        
        # compute overscan (fit the median overscan)
        xovs=range(len(oscan))
        median=np.median(oscan, axis=1)
        ovs_filt=medfilt(median, medfilter_kernel_size)
        ovs_corr=ovs_filt # or ovs_fit
        
        # remove the ovserscan and dark
        corr_img=img-ovs_corr[:, None] - darkc
        preprocessed.append(np.asarray(corr_img, dtype=np.float64))
    
    if len(preprocessed)==1:
        preprocessed=preprocessed[0]
    
    # return
    return preprocessed



