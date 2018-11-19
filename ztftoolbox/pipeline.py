#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# collection of python wrappers to run the ZTF photometric pipeline from python.
#
# Author: M. Giomi (matteo.giomi@desy.de)

import os, glob, time, os
import subprocess
import concurrent.futures

#import logging
#logging.basicConfig(level = logging.DEBUG)

from ztftoolbox.paths import get_instrphot_log, parse_filename
from ztftoolbox.pipes import get_logger, execute

split_cmd = '/ztf/ops/sw/stable/ztf/src/pl/perl/UncompressAndSplitRaw.pl'
photcal_cmd = '/ztf/ops/sw/stable/ztf/src/pl/perl/instrphotcal.pl'

def uncompress_and_split(raw_img, dest_dir, logger=None, nw=4, cleanup=True, overwrite=False):
    """
        Split a raw CCD exposure(s) file into the four quadrants
        and save the results in the desired directory.
        
        Parameters:
        -----------
            
            raw_img: `str` or `list`
                path (or paths if list) of all the raw images to be processed.
            
            dest_dir: `str`
                path to the directory where the split images should go.
            
            nw: `int`
                number of processes used to do the job.
            
            cleanup: `bool`
                if True, automatically generated QA files are removed.
            
            overwrite: `bool`
                if True overwrite existing files.
        
        Returns:
        --------
            
            list of names for the split q-wise images obtained.
    """
    start = time.time()
    logger = get_logger(logger)
    
    # either you work just for one image or you submit jobs
    if type(raw_img) == str:
        raw_img = [raw_img]
    
    # check if you have already split the files and in case don't overwrite them
    to_do = []
    if not overwrite:
        done = glob.glob(dest_dir+"/ztf*_q*.fits")
        for img in raw_img:
            if not any([parse_filename(img)['filefracday'] in qf for qf in done]):
                to_do.append(img)
    else:
        to_do = raw_img
    logger.debug("de-compressing and splitting %d image(s) to %s:\n%s"%
        (len(raw_img), dest_dir, "\n".join(raw_img)))
    
    # now submit
    exe_args = [['sh', split_cmd, img, dest_dir] for img in to_do]
    with concurrent.futures.ProcessPoolExecutor(max_workers = nw) as executor:
        executor.map(execute, exe_args)
    
    # cleanup if necessary
    if cleanup and len(to_do)>0:
        logger.debug("removing QA files from %s"%dest_dir)
        for qaf in glob.glob(dest_dir+"/ztf_*_qa.txt"):
            os.remove(qaf)
    
    # retrieve the files you just split
    done = []
    for qimg in glob.glob(dest_dir+"/ztf*_q*.fits"):
        if any([parse_filename(qimg)['filefracday'] in ri for ri in raw_img]):
            done.append(qimg)
    end = time.time()
    print ("done uncompress and splitting raw images. took %.2e sec"%(end-start))
    return done


def run_instrphot(raw_quadrant_image, wdir=None, logfile=None, logger=None, keep="all", expid=None, **kwargs):
    """
        execute the instrumental calibration pipeline on *one* of the split RC files.
        This script search in the official pipeline archieves for the log file relative
        to the processing of the given image and parse the pipeline arguments from
        there. These arguments can be modified using the kwargs. Possible are:
        
        
        Parameters:
        -----------
            
            raw_quadrant_image: `str`
                path to a fits file containing the raw quadrant wise image.
            
            wdir: `str`
                path to the working directory you want to run the analysis from. 
                It will be created if not existing. If None, a default wdir will be
                created using the name of the input image. Use 'no' to run on the 
                current directory.
            
            logfile: `str`
                path to the logfile where instrphotcal.pl output will be stored. If
                None, a default one will be created in wdir following IPAC naming schemes.
                Use 'no' to disabling logging to file and get stdout and stderr on the screen.
            
            keep: `str` or `list`
                specifies the extension(s) of the product files to be kept. The others
                will be removed at the end of the job. Set to 'all' to keep all the files.
                e.g. keep = ['psfcat', 'sciimg'] will leave in wdir only the psfcat and sciimg files.
        
        Kwargs:
        -------
        
            -inpquadimg: `str`
                Required input FITS filename of readout-channel (quadrant) image; 
                already floating-bias corrected upstream.
                
            -inpmask: `str`
                Required calibration pixel-mask FITS image file (e.g., "cmask" from 
                bias-calibration pipeline).
                
            -inppmask: `str`
                Required prior hardware bad-pixel pixel-mask FITS image file.
                
            -inpbias: `str`
                Required bias-calibration FITS image file.
                
            -inphiflat: `str`
                Required high-frequency flat-calibration FITS image file.
                
            -inploflat: `str`
                Required low-frequency flat-calibration FITS image file.
                
            -inpphotcat: `str`
                Required catalog file of external sources to support photometric 
                calibration; in ascii format.
                
            -inpastrcal: `str`
                Required matching catalog file of external sources to support 
                astrometric calibration; in FITS LDAC format.
                
            -inpcfgfile: `str`
                Required input configuraton file in ASCII format containing 
                instrphotcal.pl specific processing parameters; listed as "keyword value" pairs.
            
            Optional switches (fixed)
            -d: switch to generate debug diagnostics and files UNSET
            -v: switch to increase verbosity to stdout SET
        
    """
    
    start = time.time()
    logger = get_logger(logger)
    logger.info("running ZTF pipeline on image %s"%(raw_quadrant_image))
    
    # look into the logs (if any) to gather the standard arguments 
    # for the pipeline processing and the exposure ID
    cmd_args = {}
    try:
        log = get_instrphot_log(raw_quadrant_image)
        logger.debug("read standard pipeline arguments from %s"%log)
        pipeline_cmd = None
        with open(log) as lf:
            for line in lf:
                if ("Exposure ID: expId=" in line) and (expid is None):
                    expid = int(line.replace("Exposure ID: expId=", "").strip())
                if ("Executing command=" in line) and ("instrphotcal.pl" in line):
                    pipeline_cmd = line
                    break
        # parse the pipeline command to get all the arguments
        pieces = [pp.strip() for pp in pipeline_cmd.split(" -") if (
                    ("Executing" not in pp) and ("log.txt" not in pp))]
        for pp in pieces:
            key, val = [x.strip() for x in pp.split(" ")]
            cmd_args[key] = val
    except FileNotFoundError as e:
        logger.warning(e)
    
    # create the name for the logfile and working directory unless specified otherwise
    img_path, img_name = os.path.split(raw_quadrant_image)
    if wdir is None:
        wdir = img_name.replace(".fits", "")
    if wdir.lower() == 'no':
        wdir = "./"
    if logfile is None:
        logfile = img_name.replace(".fits", "_log.txt")
        logfile = os.path.join(wdir, logfile)
    if logfile.lower() == 'no':
        logfile = None
    
    # replace any of these with keyword arguments and run on the given image
    cmd_args.update(kwargs)
    cmd_args.update({'inpquadimg': raw_quadrant_image})
    msg = "\n".join(["\t-%s: %s"%(k, v) for k, v in cmd_args.items()])
    logger.debug("running instrphotcal.pl with arguments:\n%s"%msg)
    
    # exectute the command
    cmd = ['sh', photcal_cmd, '-v']
    for k, v in cmd_args.items():
        cmd.append('-%s'%k)
        cmd.append('%s'%str(v))
    
    # either use the EXPID from the log or one passed to this function
    if not expid is None:
        env = {'EXPID': str(expid)}
    else:
        env = None
    rcode = execute(cmd, logfile=logfile, wdir=wdir, env=env)
    
    # eventually cleanup the wdir
    if keep != 'all':
        if type(keep) == str:
            keep = [keep]
        keep += ['log.txt']
        logger.debug("removing files not containing %s from wdir"%repr(keep))
        # select all the files with the correct name
        all_prod_files = glob.glob(wdir+"/*%s*"%img_name.replace(".fits", ""))   
        for ff in all_prod_files:
            if not any([k in ff for k in keep]):
                if os.path.isfile(ff):
                    os.remove(ff)
    # done
    end = time.time()
    logger.debug("done processing %s. took %.2e sec"%(raw_quadrant_image, (end-start)))
    return rcode

def proc_img(img, wdir_base):
    """ wrapper function for process pool submission."""
    my_wdir = os.path.join(wdir_base, "tmp_"+img.split("/")[-1].replace(".fits", ""))
    run_instrphot(img, wdir=my_wdir, keep='psfcat.fits')


def calibrate_many(imgs, wdir_base, nw=4, logger=None):
    """
        run instrumental photometric calibration for a set of images using multiprocessing
        and store the psfcat and the logs produced.
        
        Parameters:
        -----------
            
            imgs: `list`
                list of path to the raw, quadrant-wise images to calibrate.
            
            wdir_base: `str`
                path specifying the root of the working directories. Files produced by the
                calibration will be stored there. Each image is processed into a separate 
                subdirectory of wdir_base. At the end of the processing we check in 
                the log for sucessfull termination and move all the psfcat files together 
                in a sepatate directory, e.g.:

                say, wdir_base = "./pipeline_wdirs/rc11"
                image 'fuffa666.fits' will be processed inside: "./pipeline_wdirs/rc11/fuffa666/"
                after processing this directory will contain:
                "./pipeline_wdirs/rc11/fuffa666/fuffa666_psfcat.fits"
                "./pipeline_wdirs/rc11/fuffa666/fuffa666_log.txt"

                if fuffa666_log.txt indicates sucessful termination, the file ./pipeline_wdirs/rc11/fuffa666/fuffa666_psfcat.fits
                is moved to ./pipeline_wdirs/rc11 and the directory ./pipeline_wdirs/rc11/fuffa666/ is deleted.
            
            nw: `int`
                number of processes in the process pool.
        
        Returns:
        --------
            
            list fo jobs that failed.
    """
    
    logger = get_logger(logger)
    logger.info("processing %d images using %d workers and saving results saved to %s"%
        (len(imgs), nw, wdir_base))
    
    start = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers = nw) as executor:
        executor.map(proc_img, imgs, [wdir_base]*len(imgs))
    end = time.time()
    logger.info("done processing %d images. Took %.2e sec"%
        (len(imgs), (end-start)))

    # check in the log files, and move the files back 
    logger.info("checking file integrity and moving files..")
    failed = []
    for wd in glob.glob(wdir_base+"/tmp_*"):
        bad = True
        logfiles = glob.glob(wd+"/*_log.txt")

        # parse the file and check for sucessful termination. If job is successful, 
        # move the files out of the wdir and clean it
        if len(logfiles) == 1:
            logfile =logfiles[0]
            tail = subprocess.check_output(['tail', '-10', logfile], universal_newlines=True)
            if 'Successfully executed "instrphotcal.pl"' in tail:
                psfcat_file = glob.glob(wd+"/*_psfcat.fits")
                subprocess.call("mv %s/* %s" %(wd, wdir_base), shell=True)
                os.rmdir(wd)
                bad = False
        if bad:
            logger.warning("job in %s not terminated correctly"%wd)
            failed.append(wd)
    logger.info("%d jobs have failed"%(len(failed)))
    return failed

