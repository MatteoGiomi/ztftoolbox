# collection of python functions to manipulate ZTF paths and files names.
#
# Author: matteo.giomi@desy.de

import os

archive_base = "/ztf/archive/sci/"

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

def rqid(ccd, q):
    """
        given the ccd ID number (from 1 to 16)
        and the CCD-wise readout channel number (from 1 to 4)
        compute the ID of that readout quadrant (from 1 to 64)
    """
    if type(ccd) in [str, float, int]:
        ccd=int(ccd)
    else:
        ccd=ccd.astype(int)
    if type(q) in [str, float, int]:
        q=int(q)
    else:
        q=q.astype(int)
    rqid=(ccd-1)*4 + q - 1
    return rqid


def ccdqid(rc):
    """
        given the readout quadrant ID (0 to 63), 
        computes the ccd (1 to 16) and quadrant (1 to 4) ID
    """
    ccd_id = rc//4 + 1
    q = rc%4 + 1
    return ccd_id, q


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


def parse_filename(filename):
    """
        given the name of a file for some ZTF pipeline data product, reverse the naming
        scheme and extract the info.
        
        Parameters:
        -----------
            
            filename: `str`
                name of file for some ZTF pipeline data product, e.g.:
                ztf_20180313148067_001561_zr_c02_o_q2_log.txt
        
        Returns:
        --------
            
            dictionary with the following keys extracted from the file name:
            year, month, day, filefrac, filefracday, field, filter, ccdid, and
            if the file is not a raw ccd-wise one, the quadrant id.
    """
    
    location, name = os.path.split(filename)
    name = name.split(".")[:-1][0]   # remove extension (e.g .fits, .fits.fz, ecc)
    pieces = [p.strip() for p in name.split("_")]
    out                 = parse_filefracday(pieces[1])
    out['field']        = int(pieces[2])
    out['filter']       = pieces[3]
    out['ccdid']        = int(pieces[4].replace("c", ""))
    if len(pieces)>6:   # raw files are CCD wise, they don't have the quadrant id
        out['q']        = int(pieces[6].replace("q", ""))
    return out


def get_instrphot_log(raw_quadrant_image):
    """
        given a fits file containng an uncompressed raw image for a give quadrant, 
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
                            archive_base, 
                            str(fn_info['year']), 
                            "%02d"%fn_info['month'] + "%02d"%fn_info['day'], 
                            str(fn_info['fracday']),
                            log_fn
                        )
    if not os.path.isfile(logfile):
        raise FileNotFoundError("logfile %s for img %s does not exists. my clues are: %s"%
            (logfile, raw_quadrant_image, repr(log_fn)))
    return logfile
