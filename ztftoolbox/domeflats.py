#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# collection of python functions to process ZTF high-freq flats.
#
# Author: M. Giomi (matteo.giomi@desy.de)


import os
import numpy as np
from scipy.signal import medfilt
from astropy.io import fits
from astropy.visualization import ZScaleInterval, ImageNormalize

from ztftoolbox.pipes import get_logger
from ztftoolbox.ztfdb import ztfdb
from ztftoolbox.mosaique import ccdqid


led_weights = {1: 1, 2: 1}  # nominal weights for each LED.


def create_highfreq_flats(fid, rcid, night_date=None, raw_flats=None, bias_frame=None, logger=None):
    """
        produce calibrated high freq flat image for one readout channel by stacking
        together monochromatic flats. This emulate what the hiFreqFlat.pl script
        is doing, that is, the following:
            
            * find the raw quadrant-wise flats (via getRawImagesForCalibration), 
            e.g.:
                ...
                ztf_20180219076215_000000_zg_c07_f_q2.fits
                ztf_20180219075914_000000_zg_c07_f_q2.fits
                ...
            
            * get the calibrated master bias frame (via getCalFiles), e.g:
                ztf_20180219_00_c07_q2_bias.fits
            
            * subtract this bias frame from all the raw quadrant-wise flats using 
            subtractImages.
            
            * normalize each of these bias-subtracted images using normimage.
            
            * finally stack them via 'stack'
        
        Parameters:
        -----------
            
            fid: `int`
                ZTF filter id (1='g', 2='r', 3='i').
            
            rcid: `int`
                readout channel number. If None, get all of them.
            
            night_date: `str`
                date of night used to get the calibration files, e.g. '2018-11-21'. If
                None, you must provide bias frames and raw flats.
            
            raw_flats: `list`
                collection of raw flat images for a given RC that will go into the 
                calibrated flat. If None, try and use night_date to get them from the
                SO database.
            
            bias_frame: `list`
                calibrated master bias to use. If None, use night_date to get it from SOBB.
    """
    
    logger = get_logger(logger)
    
    # get CCD and q numbers
    ccdid, q = ccdqid(rcid)
    
    # if raw flats are not given, use night date to get them.
    db_handle = ztfdb(logger=logger)
    if raw_flats is None:
        if night_date is None:
            raise ValueError("you must provide a night_date if you don't specify the raw flats.")
        raw_flats = db_handle.get_rawflats(date=night_date, fid=fid, ccdid=ccdid)['filename'].tolist()
    
    # same for the bias frame:
    if bias_frame is None:
        if night_date is None:
            raise ValueError("you must provide a night_date if you don't specify the bias_frame")
        bias_frame = db_handle.get_calfiles('bias', date=night_date, rcid=rcid)['filename'].tolist()
    
    logger.info("creating stacked calibrated hifreqflat using %d raw flat images:\n%s"%
        (len(raw_flats), "\t\n".join(raw_flats)))
    logger.info("using bias frame: %s"%bias_frame)
    
    
