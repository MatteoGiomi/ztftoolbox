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


import ztftoolbox.images as ztfim
from ztftoolbox.paths import ztf_filter_names, get_static_calimages
from ztftoolbox.pipes import get_logger
from ztftoolbox.ztfdb import ztfdb
from ztftoolbox.mosaique import ccdqid



led_weights = dict([(n, 1) for n in range(16)])  # nominal weigths for LEDs: all to 1


def create_highfreq_flats(fid, rcid, outfile=None, night_date=None, raw_flats=None, bias_frame=None, wdir="/tmp",  logger=None):
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
            
            outfile: `str`
                path for the output highfreq flat file. If None, use filter and dates.
            
            night_date: `str`
                date of night used to get the calibration files, e.g. '2018-11-21'. If
                None, you must provide bias frames and raw flats.
            
            raw_flats: `list`
                collection of raw flat images for a given RC that will go into the 
                calibrated flat. If None, try and use night_date to get the raw, uncompressed
                images from the SO database, and then uncompress them.
            
            bias_frame: `list`
                calibrated master bias to use. If None, use night_date to get it from SOBB.
            
            wdir: `str`
                path to working directory (used to store uncompressed raw flats)
    """
    
    logger = get_logger(logger)
    
    # create substructre in wdir
    split_wdir = os.path.join(wdir, 'split')
    biass_wdir = os.path.join(wdir, 'biassub')
    norma_wdir = os.path.join(wdir, 'normalized')
    stack_wdir = os.path.join(wdir, 'stack')
    
    # get CCD and q numbers
    ccdid, q = ccdqid(rcid)
    
    # create output file name:
    if outfile is None and (not night_date is None):
        outfile = os.path.join(stack_wdir, "ztf_%s_%s_c%02d_q%d_hifreqflat.fits"%(
            night_date.replace('-',''), ztf_filter_names[fid], ccdid, q))

    # ------------------------------------------------- #
    #           COLLECT AND UNPACK RAW FLATS            #
    # ------------------------------------------------- #
    
    # if raw flats are not given, use night date to get them.
    db_handle = ztfdb(logger=logger)
    if raw_flats is None:
        if night_date is None:
            raise ValueError("you must provide a night_date if you don't specify the raw flats.")
        
        # get the raw CCD-wise images
        raw_ccd_flats = db_handle.get_rawflats(date=night_date, fid=fid, ccdid=ccdid)['filename'].tolist()
        
        # uncompress and split (use subdirectory of the wdir)
        if not os.path.isdir(split_wdir):
            os.makedirs(split_wdir)
        qflats = ztfim.uncompress_and_split(
            raw_ccd_flats, dest_dir=split_wdir, logger=logger, nw=4, cleanup=True, overwrite=False)
        
        # select those for this readout channel
        raw_flats = [qf for qf in qflats if ('c%02d'%ccdid in qf) and ('q%d'%q in qf)]
        
    # same for the bias frame:
    if bias_frame is None:
        if night_date is None:
            raise ValueError("you must provide a night_date if you don't specify the bias_frame")
        bias_frame = db_handle.get_calfiles('bias', date=night_date, rcid=rcid)['filename'].tolist()[0]
    
    raw_flats_names = [os.path.split(fn)[-1] for fn in raw_flats]
    logger.info("creating stacked calibrated hifreqflat using %d raw flat images:\n%s"%
        (len(raw_flats), "\t\n".join(raw_flats)))
    logger.info("using bias frame: %s"%bias_frame)


    # -------------------------------------------- #
    #           PREPROCESS THE FLATS               #
    # -------------------------------------------- #
    
    # A) subtract bias frames from all of them and save them to subdir of wdir
    
    if not os.path.isdir(biass_wdir):
        os.makedirs(biass_wdir)
    bias_sub_flats = [os.path.join(
        biass_wdir, rfn.replace(".fits", "_biassub.fits")) for rfn in raw_flats_names]
    ztfim.subtract_images(raw_flats, bias_frame, bias_sub_flats, nw=4, logger=logger, overwrite=False)
    
    # B) normalize the images: you need to get the bias cmask file
    
    if not os.path.isdir(norma_wdir):
        os.makedirs(norma_wdir)
    norm_biassub_flats = [os.path.join(
        norma_wdir, rfn.replace(".fits", "_normalized_biassub.fits")) for rfn in raw_flats_names]
    bias_cmask = bias_frame.replace(".fits", "cmask.fits")
    ztfim.normalize_image(bias_sub_flats, bias_cmask, norm_biassub_flats)
    
    # C) apply LED weights (before stacking)
    pass
    
    # D) stack them all
    if not os.path.isdir(stack_wdir):
        os.makedirs(stack_wdir)
    stacked_prod = ztfim.stack_images(
        norm_biassub_flats, sigma=2.5, output_stacked=outfile, logger=logger)
    
    # make a copy of the raw stacked file
    os.system("cp %s %s"%(outfile, outfile.replace(".fits", "_raw.fits")))
    
    # optional ??:
    # imageStatistics -i ztf_20180220_zg_c05_q2_hifreqflatstddev.fits >& imageStatistics.out
    
    # E) mask noisy pixels (TODO: set the threshold!
    cmask = ztfim.mask_noisy_pixels(stacked_prod['output_std'], frames=len(raw_flats), min_frames=4, threshold=0.00381655)
    
    # F) apply lampcorr image
    lampcorr_img = get_static_calimages(rcid, fid, 'lampcorr', date='latest')
    stacked_lampcorr = outfile.replace(".fits", "_lampcorr.fits")
    ztfim.divide_images(outfile, lampcorr_img, output=stacked_lampcorr, logger=logger, overwrite=False)
    
    # G) normalize again
    ztfim.normalize_image(stacked_lampcorr, cmask, outfile)
    logger.info("DONE FOR GOOD: here is your high-freq flat: %s"%outfile)
    
    # TODO: move result in wdir (back one level from wdir/stack)
    return outfile
    
    
