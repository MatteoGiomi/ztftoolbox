#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# collection of python functions to process ZTF high-freq flats.
#
# Author: M. Giomi (matteo.giomi@desy.de)

import os
import concurrent.futures
import pandas as pd
from astropy.io import fits

import ztftoolbox.images as ztfim
from ztftoolbox.paths import ztf_filter_names, get_static_calimages, check_same_exposure
from ztftoolbox.pipes import get_logger
from ztftoolbox.ztfdb import ztfdb
from ztftoolbox.mosaique import ccdqid


flat_led_weights = dict([(n, 0.25) for n in range(16)])  # nominal weigths for LEDs: all to 1


def read_ilum_from_head(flat):
    """
        given a raw fomeflat file, read its header and return the ILUM*
        info such as LED wlen, ID, and so as a dictionary.
    """
    out = {}
    head = fits.getheader(flat, 0)
    for k, v in head.items():
        if 'ILUM' in k or (k in ['OBSJD', 'FILTERID', 'CCD_ID']):
            out[k]=v
    out['path']=os.path.abspath(flat)
    return out


def get_led_info(flats, nw=4, logger=None):
    """
        given one or more fits files, read their headers and return the ILUM*
        info such as LED wlen, ID, and so on.
        
        Parameters:
        -----------
            
            flats: `str` or `list`
                path(s) to one or more raw flats with the LED info in the header.
            
            nw: `int`
                number of processes in the process pool.
            
        Returns:
        --------
            
            pandas.DataFrame.
    """
    logger = get_logger(logger)
    
    if type(flats) is str:
        flats = [flats]
    logger.info("reading ILUM parameters for %d raw flats:\n%s"%(len(flats), "\n".join(flats)))
    
    with concurrent.futures.ProcessPoolExecutor(max_workers = nw) as executor:
        res_it = executor.map(read_ilum_from_head, flats)
    return pd.DataFrame(list(res_it))


def scale_flats_for_color(flats, led_weights, outdir, raw_flats=None, logger=None):
    """
        given a dictionary specifying the weights of each LED color and a set 
        of flat images, apply the weights and save the images in new file(s).
        
        if each LED color has a weight w so that: w in [0, 1] and (w1+w2+..) = 1
        then the i-th flat image belowinging to LED color 1 will be scaled by 
        the coefficient:
        
            c_1_i = w1 * N_tot / N1
            
        where N_tot is the total number of flats (len(flats)) and N1 is the number 
        of flats with LED color 1.
        
        Parameters:
        -----------
            
            flats: `str` or `list`
                path(s) to one or more raw flats with the LED info in the header.
            
            led_weigths: `dict`
                where keys are LED numbers and values are weights. Each weight must take
                values between 0 and 1 and the sum of all weights for the colors of the 
                given flats must be equal to 1.
            
            outdir: `str`
                path to output directory. If given, scaled flats will be saved to this
                directory with a standard filename.
            
            raw_flats: `str` or `list`
                since only the raw flats have the LED ILUM keywords in the header, 
                use this option to provide the raw flats from which to read the LED colors, 
                but apply the scaling to the flats argument.
    """
    logger = get_logger(logger)
    
    # read in the LED params for each file
    read_led_from = flats if raw_flats is None else raw_flats
    led_info = get_led_info(read_led_from, logger=logger)
    
    # check that the LEDs weights sums to 1
    unique_led = led_info['ILUM_LED'].unique().tolist()
    logger.info("Found the following LED IDs: %s"%unique_led)
    
    sum_w = sum(led_weights[ll] for ll in unique_led)
    if sum_w != 1:
        raise ValueError("Weights for LEDs %s must sum to 1. Got %s"%
            (unique_led, led_weights))
    
    # apply the right scaling for each group of flats teken with a given LED.
    n_tot, full_output = len(flats), []
    for led, monochrome in led_info.groupby('ILUM_LED'):
        
        # find the weight for this LED color
        led_w = led_weights.get(led)
        if led_w is None:
            raise KeyError("cannot find weights for LED # %d in %s"%(led, repr(led_weights)))
        
        # compute the scaling you apply to all the files for this LED
        scale = led_w * n_tot / float(len(monochrome))
        logger.info("Applying coeff %f to the %d flats with LED #%d (w=%f)"%
            (scale, len(monochrome), led, led_w))
        
        # go back and pair the right file of the input list to 
        # the raw flats from which you have read the color.
        if not raw_flats is None:
            inputs = []
            for inf in flats:
                for path in monochrome['path']:
                    if check_same_exposure(inf, path):
                        inputs.append(inf)
                        logger.debug("input file: %s corresponds to raw flat %s"%
                            (os.path.split(inf)[-1], os.path.split(path)[-1]))
                        break
            if len(inputs) != len(monochrome):
                raise RuntimeError("Cannot match all input flats to the raw_flats.")
        else:
            inputs = monochrome['path'].tolist()
        
        # now figure the name of the output files
        outputs = []
        for inp in inputs:
            outf = os.path.join(outdir, os.path.split(inp)[-1].replace(".fits", "_cscaled.fits"))
            outputs.append(outf)
        
        # finally multiply them bastirds
        ztfim.multiply_scalar(inputs, scale, outputs, overwrite=False, logger=logger)
        
        # and append the results
        full_output += outputs
    
    # greetings to the fambly
    logger.info("done multiplying images.")
    return full_output


def get_uncomp_raw_flats(fid, ccdid, night_date, dest_dir, q=None, db_handle=None, logger=None):
    """
        query the SODB for the raw flats taken on a given night, uncompress and
        split them in quandrant-wise files, and save the results in given directory.
        
        Returns a list with the files produced.
        
        NOTE: this will gather and plit all the files for the given CCD, for all the
        quandrants.
        
        Parameters:
        -----------
            
            fid: `int`
                ZTF filter id (1='g', 2='r', 3='i').
            
            ccdid: `int`
                ID for the CCD (1 to 16).
            
            night_date: `str`
                date of night used to get the calibration files, e.g. '2018-11-21'.
            
            dest_dir: `str`
                path to directory where the uncompressed files will be stored. It will
                be created if not existing.
            
            q: `int`
                quandrant ID (1 to 4). If given it will return a list with just the 
                files for this specific quadrant (although the files for all the q 
                of the given CCDs will be processed)
            
            db_handle: `ztftoolbox.ztfdb.ztfdb` instance
                client to query the SODB.
    """
    logger = get_logger(logger)
    
    # get the raw CCD-wise images
    if db_handle is None:
        db_handle = ztfdb(logger=logger)
    raw_ccd_flats = db_handle.get_rawflats(date=night_date, fid=fid, ccdid=ccdid)['filename'].tolist()
    
    # uncompress and split (use subdirectory of the wdir)
    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)
    qflats = ztfim.uncompress_and_split(
        raw_ccd_flats, dest_dir=dest_dir, logger=logger, nw=4, cleanup=True, overwrite=False)
    
    # select those for this readout channel (if given) and return a list of them
    if q is None:
        raw_flats = [qf for qf in qflats if ('c%02d'%ccdid in qf)]
    else:
        raw_flats = [qf for qf in qflats if ('c%02d'%ccdid in qf) and ('q%d'%q in qf)]
    return raw_flats


def create_highfreq_flats(fid, rcid, outfile=None, night_date=None, led_weights=None,
    raw_flats=None, bias_frame=None, wdir="/tmp", logger=None):
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
            
            led_weights: `dict`
                where keys are LED numbers and values are weights. Each weight must take
                values between 0 and 1 and the sum of all weights for the colors of the 
                given flats must be equal to 1. If None, no weights are applied.
            
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
    split_wdir      = os.path.join(wdir, 'split')
    biass_wdir      = os.path.join(wdir, 'biassub')
    norma_wdir      = os.path.join(wdir, 'normalized')
    cscaled_wdir    = os.path.join(wdir, 'cscaled')
    stack_wdir      = os.path.join(wdir, 'stack')
    
    # get CCD and q numbers
    ccdid, q = ccdqid(rcid)
    
    # create output file name:
    if outfile is None:
        if not night_date is None:
            outfile = os.path.join(stack_wdir, "ztf_%s_%s_c%02d_q%d_hifreqflat.fits"%(
                night_date.replace('-',''), ztf_filter_names[fid], ccdid, q))
        else:
            raise RuntimeError("If night_date is not specified, you must provide a name for the output file.")

    # ------------------------------------------------- #
    #           COLLECT AND UNPACK RAW FLATS            #
    # ------------------------------------------------- #
    
    # if raw flats are not given, use night date to get them.
    db_handle = ztfdb(logger=logger)
    if raw_flats is None:
        if night_date is None:
            raise ValueError("you must provide a night_date if you don't specify the raw flats.")
        
        raw_flats = get_uncomp_raw_flats(
            fid=fid, ccdid=ccdid, night_date=night_date, dest_dir=split_wdir, q=q, db_handle=db_handle, logger=logger)
        
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
    norm_flats = ztfim.normalize_image(bias_sub_flats, bias_cmask, norm_biassub_flats, b=16, sigmas=3)
    
    # C) apply LED weights (before stacking)
    to_stack = norm_biassub_flats
    if not led_weights is None:
        if not os.path.isdir(cscaled_wdir):
            os.makedirs(cscaled_wdir)
        cscaled_flats = scale_flats_for_color(
            norm_biassub_flats, led_weights=led_weights, outdir=cscaled_wdir, raw_flats=raw_flats, logger=logger)
        to_stack = cscaled_flats
    
    # D) stack them all
    if not os.path.isdir(stack_wdir):
        os.makedirs(stack_wdir)
    stacked_out = os.path.join(stack_wdir, os.path.split(outfile)[-1].replace(".fits", "_stack.fits"))
    stacked_prod = ztfim.stack_images(
        to_stack, sigma=2.5, output_stacked=stacked_out, logger=logger)
    
    # compute stats on flat std (needed to set the th for mask noisy pixels)
    img_stats = ztfim.image_statistics(stacked_prod['output_std'])
    mask_th = 1.85 * img_stats['median']
    logger.info("noisy pixel mask threshold: %f"%mask_th)
    
    # E) mask noisy pixels
    cmask = ztfim.mask_noisy_pixels(stacked_prod['output_std'], frames=len(raw_flats), min_frames=4, threshold=mask_th)
    
    # F) apply lampcorr image
    lampcorr_img = get_static_calimages(rcid, fid, 'lampcorr', date='latest')
    stacked_lampcorr = stacked_out.replace(".fits", "_lampcorr.fits")
    ztfim.divide_images(stacked_out, lampcorr_img, output=stacked_lampcorr, logger=logger, overwrite=False)
    
    # G) normalize again
    ztfim.normalize_image(stacked_lampcorr, cmask, outfile, b=16, sigmas=3, r_th=0.01, overwrite=True)
    logger.info("DONE FOR GOOD: here is your high-freq flat: %s"%outfile)
    
    # TODO: move result in wdir (back one level from wdir/stack)
    return outfile
    
    
