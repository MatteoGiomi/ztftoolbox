#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# collection of python functions and pipeline wrappers to process 
# and manipulate ztf image data.
#
# Author: M. Giomi (matteo.giomi@desy.de)

import os, glob, time
import subprocess
import concurrent.futures
from astropy.io import fits

from ztftoolbox.paths import parse_filename
from ztftoolbox.pipes import get_logger, execute

split_cmd       = '/ztf/ops/sw/stable/ztf/src/pl/perl/UncompressAndSplitRaw.pl'
subtract_cmd    = '/ztf/ops/sw/180116/ztf/bin/subtractImages'
normalize_cmd   = '/ztf/ops/sw/180116/ztf/bin/normimage'
stack_cmd       = '/ztf/ops/sw/180116/ztf/bin/stack'
mask_noisy_cmd  = '/ztf/ops/sw/180116/ztf/bin/maskNoisyPixels'
divide_cmd      = '/ztf/ops/sw/180116/ztf/bin/divideImages'
imgstat_cmd     = '/ztf/ops/sw/180116/ztf/bin/imageStatistics'


def multiply_scalar_one(infile, outfile, coeff, ext=0, overwrite=False):
    """
        given fits file multiply the image by coeff and save the result to outfile.
        
        Parameters:
        -----------
            
            in[out]file: `str`
                path to input and output files.
            
            coeff: `float`
                coefficient of the product
            
            ext: `int` or `str`
                name of the FITS file extension (if more than one are present) you
                want to multiply.
            
            overwrite: `bool`
                passed to astropy.io.fits.writeto.
    """
    
    # check if you don't want to overwtite
    if os.path.isfile(outfile) and (not overwrite):
        return

    with fits.open(infile) as hdul:
        head, data = hdul[ext].header, hdul[ext].data
        head['ORIGFILE'] = infile
        head['MULCOEFF'] = coeff
        hdul[ext].data = data * coeff
        hdul[ext].header = head
        hdul.writeto(outfile, overwrite=overwrite)


def multiply_scalar(images, coeffs, outputs, ext=0, nw=4, overwrite=False, logger=None):
    """
        multiply a set of images by a set of coefficients.
        
        Parameters:
        -----------
            
            images: `str` or `list`
                path (or paths if list) of all the images to be multiplied.
            
            coeffs: `float`
                coefficient for multiplication
            
            outputs: `str` or `list`
                path (or paths if list) of all resulting images. 
            
            ext: `int` or `str`
                name of the FITS file extension (if more than one are present) you
                want to multiply.
            
            nw: `int`
                number of processes used to do the job.
            
            overwrite: `bool`
                if True overwrite existing files.
        
    """
    
    start = time.time()
    logger = get_logger(logger)
    
    # either you work just for one image or you submit jobs
    if type(images) == str:
        images = [images]
    if type(outputs) == str:
        output_files = [output_files]
    if type(coeffs) in [int, float]:
        coeffs = [coeffs]
    
    # you can multiply many images by the same coeff, or the same image
    # by many different coefficients
    if len(coeffs) == 1: coeffs *= len(images)
    if len(images) == 1: images *= len(coeffs)
    
    # check:
    if not (len(coeffs) == len(images) == len(outputs)):
        raise ValueError("Number of output files must match those of images and / or coefficients.")
    
    # wrap the args together
    args = list(zip(images, coeffs, outputs))
    logger.debug("Multiplying %d images using %d workers:"%(len(args), nw))
    for pp in args: logger.debug("%s * %f --> %s"%(pp[0], pp[1], pp[2]))
    
    # now submit
    with concurrent.futures.ProcessPoolExecutor(max_workers = nw) as executor:
        executor.map(
            multiply_scalar_one, images, outputs, coeffs, [ext]*len(images), [overwrite]*len(images))
    
    # return files actually processed
    end = time.time()
    logger.info("done multiplying images. took %.2e sec"%(end-start))
    return outputs



def image_statistics(img, logger=None):
    """
        wrapper around imageStatistic. Returns image stats as dictionary
    """
    
    logger = get_logger(logger)
    logger.info("computing image statistics on: %s"%img)
    
    # run command and format output
    stat_str = subprocess.check_output([imgstat_cmd, '-i', img], universal_newlines=True)
    logger.debug(stat_str)
    out = {}
    for l in stat_str.split("\n"):
        if "Image " in l and "=" in l:
            key, value = [pp.strip() for pp in l.split("=")]
            key = key.replace("Image ", "").replace("number of", "n").replace(" ", "_")
            out[key] = float(value)
    logger.info("%s"%repr(out))
    return out


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


def subtract_images(image_a, image_b, output, nw=4, logger=None, overwrite=False):
    """
        subtract image_b from image_a and store output to desired file.
        Wrapper around /ztf/ops/sw/180116/ztf/bin/subtractImages.
        
        Parameters:
        -----------
        
            image_a: `str` or `list`
                input minuend FITS-image file.
            
            image_b: `str` or `list`
                subtrahend FITS-image file.
            
            output: `str` or `list`
                path to output files.
            
            nw: `int`
                number of workers.
            
            overwrite: `bool`
                if True overwrite existing files.
    """
    start = time.time()
    logger = get_logger(logger)
    
    # either you work just for one image or you submit jobs
    if type(image_a) == str:
        image_a = [image_a]
    if type(image_b) == str:
        image_b = [image_b]
    if type(output) == str:
        output = [output]
    
    # you can subtract the same image from may other ones,
    # or subtract many diff images from the same. Output names must always be uniques
    if len(image_a) == 1: image_a *= len(output)
    if len(image_b) == 1: image_b *= len(output)
    pairs = list(zip(image_a, image_b, output))
    
    # select jobs to be run only if output does not exist or if overwrite
    to_do = [pp for pp in pairs if (not os.path.isfile(pp[2])) or overwrite]
    logger.debug("Subtracting %d image pair(s) with %d workers:"%(len(pairs), nw))
    for pp in pairs: logger.debug("%s - %s --> %s"%(pp[0], pp[1], pp[2]))
    
    # now submit
    exe_args = [[subtract_cmd, '-i', pp[0], '-s', pp[1], '-o', pp[2]] for pp in to_do]
    with concurrent.futures.ProcessPoolExecutor(max_workers = nw) as executor:
        executor.map(execute, exe_args)
    
    # return files actually processed
    end = time.time()
    print ("done subtracting images. took %.2e sec"%(end-start))
    return [pp[2] for pp in to_do]


def divide_images(image_a, image_b, output, nw=4, logger=None, overwrite=False):
    """
        divide image_a from image_b and store output to desired file.
        Wrapper around /ztf/ops/sw/180116/ztf/bin/divideImages.
        
        Parameters:
        -----------
        
            image_a: `str` or `list`
                input dividend FITS-image file
            
            image_b: `str` or `list`
                input divisor FITS-image file
            
            output: `str` or `list`
                path to output files.
            
            nw: `int`
                number of workers.
            
            overwrite: `bool`
                if True overwrite existing files.
    """
    start = time.time()
    logger = get_logger(logger)
    
    # either you work just for one image or you submit jobs
    if type(image_a) == str:
        image_a = [image_a]
    if type(image_b) == str:
        image_b = [image_b]
    if type(output) == str:
        output = [output]
    
    # you can subtract the same image from may other ones,
    # or subtract many diff images from the same. Output names must always be uniques
    if len(image_a) == 1: image_a *= len(output)
    if len(image_b) == 1: image_b *= len(output)
    pairs = list(zip(image_a, image_b, output))
    
    # select jobs to be run only if output does not exist or if overwrite
    to_do = [pp for pp in pairs if (not os.path.isfile(pp[2])) or overwrite]
    logger.debug("Dividing %d image pair(s) with %d workers:"%(len(pairs), nw))
    for pp in pairs: logger.debug("%s / %s --> %s"%(pp[0], pp[1], pp[2]))
    
    # now submit
    exe_args = [[divide_cmd, '-i', pp[0], '-s', pp[1], '-o', pp[2]] for pp in to_do]
    with concurrent.futures.ProcessPoolExecutor(max_workers = nw) as executor:
        executor.map(execute, exe_args)
    
    # return files actually processed
    end = time.time()
    print ("done dividing images. took %.2e sec"%(end-start))
    return [pp[2] for pp in to_do]


def normalize_image(image, cmask, output, b=16, sigmas=3, r_th=None, nw=4, logger=None, overwrite=False):
    """
        Normalize input images and apply bias mask. 
        Wrapper around /ztf/ops/sw/180116/ztf/bin/normimage
        
        Parameters:
        -----------
        
            image: `str` or `list`
                fits image(s) to normalize
            
            cmask: `str` or `list`
                mask frame to apply to each image.
            
            output: `str` or `list`
                path to output file(s).
            
            b: `float`
                bit-mask value to test against mask image (default = 0).
            
            sigmas: `float`
                number of "sigmas" for outlier rejection.
            
            r_th: `float`
                threshold for resetting result to unity (optional)
            
            nw: `int`
                number of workers.
            
            overwrite: `bool`
                if True overwrite existing files.
    """
    start = time.time()
    logger = get_logger(logger)
    
    # either you work just for one image or you submit jobs
    if type(image) == str:
        image = [image]
    if type(cmask) == str:
        cmask = [cmask]
    if type(output) == str:
        output = [output]
    
    # you can subtract the same image from may other ones,
    # or subtract many diff images from the same. Output names must always be uniques
    if len(image) == 1: image *= len(output)
    if len(cmask) == 1: cmask *= len(output)
    pairs = list(zip(image, cmask, output))
    
    # select jobs to be run only if output does not exist or if overwrite
    to_do = [pp for pp in pairs if (not os.path.isfile(pp[2])) or overwrite]
    logger.debug("Normalizing %d images with %d workers:\n %s"%
        (len(pairs), nw, "\n".join(image)))
    if len(cmask)>1:
        logger.debug("Using c mask: %s"%"\n".join(cmask))
    else:
        logger.debug("Using c mask: %s"%cmask[0])
    
    # create the list of args for each command
    exe_args = []
    for pp in to_do:
        cmd = [normalize_cmd, '-i', pp[0], '-m', pp[1], '-o', pp[2], '-b', "%f"%b, '-s', "%f"%sigmas]
        if not r_th is None:
            cmd += ['-r', "%f"%r_th]
        exe_args.append(cmd)
        
    # now submit
    with concurrent.futures.ProcessPoolExecutor(max_workers = nw) as executor:
        executor.map(execute, exe_args)
    
    # return files actually processed
    end = time.time()
    print ("done normalizing images. took %.2e sec"%(end-start))
    return [pp[2] for pp in to_do]


def stack_images(images, output_stacked, sigma=2.5, output_unc=None, 
    output_nsampl=None, output_av=None, output_std=None, logger=None, overwrite=False):
    """
        given a list of RC-wise fits file, run the stack routine on them. E.g.
        
        stack -i images.lst -s 2.5 -t 36 -a hifreqflatavg.fits -d ztf_20180220_zg_c05_q2_hifreqflatstddev.fits 
        -o ztf_20180220_zg_c05_q2_stack.fits -u ztf_20180220_zg_c05_q2_hifreqflatunc.fits 
        -n ztf_20180220_zg_c05_q2_hifreqflatnsamps.fits >& stack.out
        
        where:
        -i <input list-of-images file>
        -o <output mean FITS-image file>
        -u <output mean-uncertainty FITS-image file>
        -n <output number-of-samples FITS-image file>
        -s <number of "sigmas" for outlier rejection> (default = 5.0)
        -a <output average FITS-image file, before outlier rejection> (required with -d option; otherwise optional)
        -d <output standard-deviation FITS-image file, before outlier rejection> (required with -a option; otherwise optional)
        -t <number of processing threads> (default = 25)
    
        Parameters:
        -----------
            
            images: `list`
                input images to be stacked.
            
            output_stacked: `str`
                path to output image file.
            
            sigma: `float`
                number of "sigmas" for outlier rejection.
            
            output_unc: `str` or None
                path to mean-uncertainty FITS-image file. If None, this is derived
                from output_stacked
            
            output_nsampl: `str` or None
                path to umber-of-samples FITS-image file. If None, this is derived
                from output_stacked
            
            output_av: `str` or None
                output average FITS-image file, before outlier rejection. 
                Required with output_std option; otherwise optional.
            
            output_std: `str` or None
                output standard-deviation FITS-image file, before outlier rejection. 
                Required with output_av option; otherwise optional.
            
            overwrite: `bool`
                destroy file if existsing and re-run.
        
        Returns:
        --------
            
            dictionary with produced files.
    """
    
    start = time.time()
    logger = get_logger(logger)
    
    # create auxiliary file names and ather them in a dict
    if output_unc is None:
        output_unc = output_stacked.replace(".fits", "unc.fits")
    if output_nsampl is None:
        output_nsampl = output_stacked.replace(".fits", "nsamps.fits")
    if output_av is None:
        output_av = output_stacked.replace(".fits", "avg.fits")
    if output_std is None:
        output_std = output_stacked.replace(".fits", "stddev.fits")
    outfiles = {}
    outfiles['output_stacked']=output_stacked
    outfiles['output_unc']=output_unc
    outfiles['output_nsampl']=output_nsampl
    outfiles['output_av']=output_av
    outfiles['output_std']=output_std
    
    # if you want to re-run, you better remove the old files
    if any([os.path.isfile(ff) for ff in outfiles.values()]):
        if not overwrite:
            logger.info("output files already exist and we respect them.")
            return outfiles
        else:
            logger.info("removing output files to run anew!")
            for ff in outfiles.values():
                print (ff)
                if os.path.isfile(ff):
                    os.remove(ff)
    
    logger.info("Stacking %d images into: %s"%(len(images), output_stacked))
    logger.info("number of sigmas for outlier rejection: %f"%sigma)
    logger.info("output mean-uncertainty FITS-image file: %s"%output_unc)
    logger.info("output number-of-samples FITS-image file: %s"%output_nsampl)
    logger.info("output average FITS-image file, before outlier rejection: %s"%output_av)
    logger.info("output standard-deviation FITS-image file, before outlier rejection: %s"%output_std)
    
    # create file containing the list of images
    list_file = output_stacked.replace(".fits", "_input.lst")
    with open(list_file, 'w') as lf:
        for img in images:
            lf.write("%s\n"%img)
    logger.info("created input list %s"%list_file)
    
    # create the command and run it:
    cmd = [stack_cmd]
    cmd += ['-i', list_file]
    cmd += ['-s', "%f"%sigma]
    cmd += ['-a', output_av]
    cmd += ['-d', output_std]
    cmd += ['-o', output_stacked]
    cmd += ['-u', output_unc]
    cmd += ['-n', output_nsampl]
    execute(cmd, logger=logger)
    end = time.time()
    
    # return paths of the files
    print ("done stacking images. took %.2e sec"%(end-start))
    return outfiles

def mask_noisy_pixels(input_std, frames, min_frames, threshold, output_mask=None, logger=None, overwrite=False):
    """
        run maskNoisyPixels on an image. E.g.:
            
        maskNoisyPixels -f 12 -m 4 -s 0.00333185 -d ztf_20180220_zg_c05_q2_hifreqflatstddev.fits 
        -o ztf_20180220_zg_c05_q2_hifreqflatcmask.fits >& maskNoisyPixels.out
        
        where:
        -i <input pmask FITS-image file> (optional)
        -d <input standard-deviation FITS-image file>
        -u <input uncertainty FITS-image file> (optional; required if -d option is not given)
        -n <input number-of-samples FITS-image file> (optional; required if -u option is given)
        -o <output cmask FITS-image file>
        -s <noisy pixel threshold> (default = 5.0)
        -f <number of frames> (default = 1)
        -m <minimum number of frames> (default = 1)
        
        Parameters:
        -----------
            
            input_std: `str`
                input standard-deviation FITS-image file.
            
            output_mask: `str` or None
                <output cmask FITS-image file. If None, created from input_std
            
            frames: `int`
                number of frames.
            
            min_frames: `int`
                minimum number of frames.
            
            threshold: `float`
                noisy pixel threshold.
    """
    
    start = time.time()
    logger = get_logger(logger)
    
    # create out file name if not given
    if output_mask is None:
        output_mask = input_std.replace("stddev.fits", "cmask.fits")
    logger.info("Creating cmask file %s from %s"%(output_mask, input_std))
    
    # check before trying to overwrite
    if os.path.isfile(output_mask):
        if overwrite:
            logger.info("removing output files to run anew!")
            os.remove(output_mask)
        else:
            logger.info("output file already exist and we respect it.")
            return output_mask
    
    # run baby
    cmd = [mask_noisy_cmd, '-f', "%f"%frames, '-m', "%d"%min_frames, '-s', 
        "%f"%threshold, '-d', input_std, '-o', output_mask]
    execute(cmd, logger=logger)
    end = time.time()
    print ("done masking noisy pixels. took %.2e sec"%(end-start))
    return output_mask

