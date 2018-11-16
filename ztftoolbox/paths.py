#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# collection of python functions to manipulate ZTF paths and files names.
#
# Author: M. Giomi (matteo.giomi@desy.de)

import os

archive_base_sci = "/ztf/archive/sci/"
archive_base_cal = "/ztf/archive/cal/"

# file produced by instrphotcal.pl have the following extensions
calprod_exts = [
    "_findtracks.tbl",
    "_inpsciimg.fits",
    "_mskimg.fits",
    "_psfcat.fits",
    "_psfcat.reg",
    "_psfcat.tbl",
    "_sciimg_astromqa.txt",
    "_sciimgdao.psf",
    "_sciimgdaopsfcent.fits",
    "_sciimgdaopsf.fits",
    "_sciimg.fits",
    "_sciimgqa.txt",
    "_sciunc.fits",
    "_sexcat.addkeys",
    "_sexcat.fits",
    "_sexcat_ldac.fits",
    "_sexcat.txt"
    ]


def parse_filefracday(filefracday):
    """
        extract info from the filefracday string used in file names, where year 
        is the first four characters of filefracday, month is characters 5--6 
        of filefracday, day is characters 7--8 of filefracday, and fracday 
        is characters 9--14 of filefracday. Returns the results as a dictionary.
    """
    return dict(
                year     = int(filefracday[:4]),
                month    = int(filefracday[4:6]),
                day      = int(filefracday[6:8]),
                fracday  = int(filefracday[8:])
                )


def parse_filename(filename, which='sci'):
    """
        given the name of a file for some ZTF pipeline data product, reverse the naming
        scheme and extract the info.
        
        Parameters:
        -----------
            
            filename: `str`
                name of file for some ZTF pipeline data product, e.g.:
                ztf_20180313148067_001561_zr_c02_o_q2_log.txt   (if which == 'sci')
                ztf_20180220_zg_c03_q4_hifreqflat.fits          (if which == 'cal')
        
        Returns:
        --------
            
            dictionary with the following keys extracted from the file name:
            year, month, day, filefrac, filefracday, field, filter, ccdid, and
            if the file is not a raw ccd-wise one, the quadrant id.
    """
    
    location, name = os.path.split(filename)
    name = name.split(".")[:-1][0]   # remove extension (e.g .fits, .fits.fz, ecc)
    pieces = [p.strip() for p in name.split("_")]
    if which == 'sci':
        out                 = parse_filefracday(pieces[1])
        out['field']        = int(pieces[2])
        out['filter']       = pieces[3]
        out['ccdid']        = int(pieces[4].replace("c", ""))
        if len(pieces)>6:   # raw files are CCD wise, they don't have the quadrant id
            out['q']        = int(pieces[6].replace("q", ""))
    elif which == 'cal':
        out                 = dict(year = pieces[1][:4])
        out['month']        = int(pieces[1][4:6])
        out['day']          = int(pieces[1][6:8])
        out['filter']       = pieces[2]
        out['ccdid']        = int(pieces[3].replace("c", ""))
        out['q']            = int(pieces[4].replace("q", ""))
    else:
        raise ValueError("which flag should be either 'sci' or 'cal'. got %s instead"%which)
    return out


def get_instrphot_log(raw_quadrant_image):
    """
        given a fits file containng an uncompressed raw image for a given quadrant, 
        go and look for the pipeline processing log and return a path for that file.
        
        Parameters:
        -----------
            
            raw_quadrant_image: `str`
                path to a fits file containing the raw quadrant wise image, e.g.:
                ztf_20180313148067_001561_zr_c02_o_q2_log.txt
        
        Returns:
        --------
            
            str, path to the logfile
    """
    
    # get the path from the info in the filename
    fn_info = parse_filename(raw_quadrant_image)
    log_fn = os.path.split(raw_quadrant_image)[-1].split(".")[:-1][0]+"_log.txt"
    logfile = os.path.join(
                            archive_base_sci, 
                            str(fn_info['year']), 
                            "%02d"%fn_info['month'] + "%02d"%fn_info['day'], 
                            str(fn_info['fracday']),
                            log_fn
                        )
    if not os.path.isfile(logfile):
        raise FileNotFoundError("logfile %s for img %s does not exists. my clues are: %s"%
            (logfile, raw_quadrant_image, repr(log_fn)))
    return logfile


def get_domeflat_log(cal_flat_image):
    """
        given a fits file containng an uncompressed calibrated domeflat image, 
        go and look for the pipeline processing log and return a path for that file.
        
        Parameters:
        -----------
            
            cal_flat_image: `str`
                path to a fits file containing the calibrated hifreq flat for a given RC.
                E.g.: ztf_20180220_zg_c03_q4_hifreqflat.fits
        
        Returns:
        --------
            
            str, path to the logfile
    """
    
    # parse the filename to get the date and info
    fn_info = parse_filename(cal_flat_image, 'cal')
    log_fn = os.path.split(cal_flat_image)[-1].replace(".fits", "log.txt")
    logfile = os.path.join(
                            archive_base_cal, 
                            str(fn_info['year']), 
                            "%02d"%fn_info['month'] + "%02d"%fn_info['day'], 
                            'hifreqflat',
                            fn_info['filter'],
                            "ccd%02d"%fn_info['ccdid'],
                            "q%d"%fn_info['q'],
                            log_fn
                        )
    if not os.path.isfile(logfile):
        raise FileNotFoundError("logfile %s for img %s does not exists. my clues are: %s"%
            (logfile, cal_flat_image, repr(log_fn)))
    return logfile
