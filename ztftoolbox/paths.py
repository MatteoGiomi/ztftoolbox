#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# collection of python functions to manipulate ZTF paths and files names.
#
# Author: M. Giomi (matteo.giomi@desy.de)

import os, glob, datetime

archive_base_sci    = "/ztf/archive/sci/"
archive_base_cal    = "/ztf/archive/cal/"
archive_base_raw    = "/ztf/archive/raw/"
archive_base_static = "/ztf/ops/static/"

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

# filter ID to names:
ztf_filter_names = {1: 'zg', 2: 'zr', 3: 'zi'}

def check_same_exposure(file1, file2):
    """
        using info in the file names, check if the two files derives from 
        the same exposure.
    """
    meta1, meta2 = parse_filename(file1), parse_filename(file2)
    return (meta1 == meta2)
    
    

def get_static_calimages(rcid, fid, which, date='latest'):
    """
        return the static calibration image file corresponding to a 
        given readout channel and filter, i.e.:
        /ztf/ops/static/rc17/calimg/ztf_20170101_zg_c05_q2_lampcorr.fits
        
        Parameters:
        -----------
            
            rcid: `int`
                readout-channel ID.
            
            fid: `int`
                ZTf filter ID.
            
            which: `str`
                type of calimg file to return. Allowed values are:
                ['pmask', 'hifreqflat', 'lownuflat', 'lampcorr']
    """
    
    allowed = ['pmask', 'hifreqflat', 'lownuflat', 'lampcorr']
    if not which in allowed:
        raise ValueError("static calimage file have types %s. got %s"%
            (",".join(allowed), which))
    
    # get all the files for the specified filter and type
    base_dir = os.path.join(archive_base_static, "rc%02d"%rcid, "calimg")
    all_files = glob.glob(base_dir+"/ztf_*_%s_*_%s.fits"%(ztf_filter_names[fid], which))
    
    # select the one you like depending on the date
    def get_date(fn):
        """ extract date from filename """
        datestr = os.path.split(fn)[-1].split("_")[1]
        return datetime.datetime.strptime(datestr, "%Y%m%d")
    all_files = sorted(all_files, key=get_date)
    if date == 'latest':
        my_file = all_files[-1]
    else:
        my_file = [ff for ff in all_files if date in ff][0]
    return my_file


def parse_filefracday(filefracday):
    """
        extract info from the filefracday string used in file names, where year 
        is the first four characters of filefracday, month is characters 5--6 
        of filefracday, day is characters 7--8 of filefracday, and fracday 
        is characters 9--14 of filefracday. Returns the results as a dictionary.
    """
    return dict(
                year        = int(filefracday[:4]),
                month       = int(filefracday[4:6]),
                day         = int(filefracday[6:8]),
                fracday     = filefracday[8:],
                filefracday = filefracday
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
            
            which: `str`
                specify if the path is a science image or a flat/bias one.
                
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
        raise ValueError("'which' flag should be either 'sci' or 'cal'. got %s instead"%which)
    return out


def get_instrphot_log(raw_quadrant_image):
    """
        given a fits file containng an uncompressed raw image for a given quadrant, 
        go and look for the pipeline processing log and return a path for that file.
        
        Parameters:
        -----------
            
            raw_quadrant_image: `str`
                path to a fits file containing the raw quadrant wise image, e.g.:
                ztf_20180313148067_001561_zr_c02_o_q2.fits
        
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


def get_raw_monochrome_flats(cal_image, which = 'sci'):
    """
        given a fits file containng either calibrated domeflat image or a science
        image, return the list of raw, monochromatic, CCD-wise flats that went into it.
        
        Parameters:
        -----------
            
            cal_image: `str`
                path to a fits file containing the calibrated hifreq flat for a given RC
                or the corresponding sci image. E.g.: 
                    ztf_20180220_zg_c03_q4_hifreqflat.fits (if which == 'cal')
                    ztf_20180313148067_001561_zr_c02_o_q2.fits (if which == 'sci')
            
            which: `str`
                specify if the path is a science image or a flat/bias one. If the image 
                provided is a sci image, look in its log to figure out which flats were used.
            
        Returns:
        --------
            
            list of paths to the raw domeflat files.
    """
    
    # either look in the log of the science image or you're done.
    cal_flat_image = None
    if which == 'sci':
        with open(get_instrphot_log(cal_image)) as log:
            for l in log:
                if "hiflatimage=" in l:
                    cal_flat_image = l.split("=")[-1].strip()
                    break
    elif which == 'cal':
        cal_flat_image = cal_image
    else:
        raise ValueError("'which' flag should be either 'sci' or 'cal'. got %s instead"%which)
    
    # now process the file name to get the log of the hiFreqFlat.pl script
    location, name = os.path.split(cal_flat_image)
    raw_flats = []
    with open(get_domeflat_log(cal_flat_image)) as log:
        for l in log:
            if "i, rId, filename =" in l:
                fname = l.split(" ")[-1].strip()
                fn_info = parse_filename(fname)
                
                # build up the path to the file:
                raw_flat_path = os.path.join(
                                        archive_base_raw,
                                        str(fn_info['year']),
                                        "%02d"%fn_info['month'] + "%02d"%fn_info['day'],
                                        str(fn_info['fracday']))
                
                # build up the raw flat name:
                raw_flat_name = "ztf_%s%02d%02d%s_000000_%s_c%02d_f.fits.fz"%(
                    str(fn_info['year']), fn_info['month'], fn_info['day'], 
                    str(fn_info['fracday']), fn_info['filter'], fn_info['ccdid']
                    )
                
                # combine them
                raw_flat = os.path.join(raw_flat_path, raw_flat_name)
                
                # check and add it to the list
                if not os.path.isfile(raw_flat):
                    raise FileNotFoundError("raw flat %s for cal flat %s does not exists. my clues are: %s"%
                        (raw_flat, cal_flat_image, repr(fn_info)))
                else:
                    raw_flats.append(raw_flat)
    return raw_flats



