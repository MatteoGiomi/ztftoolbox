# collection of functions to manipulate ZTF image data

import numpy as np
from scipy.signal import medfilt
from astropy.io import fits

import logging
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)


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

def getaxesxy(ccd, q):
    """
        given the ccd number (from 1 to 16) and the 
        CCD-wise readout channel number (from 1 to 4), 
        this function return the x-y position of this 
        readout quadrant on the 8 x 8 grid of the full ZTF field.
    """
    yplot=7-( 2*((ccd-1)//4) + 1*(q==1 or q==2) )
    xplot=2*( 4-(ccd-1)%4)-1 - 1*(q==2 or q==3) 
    return int(xplot), int(yplot)


def getaxesxy_ccd(ccdid):
    """
        given the ID number of a CCD (1 to 16) return the
        x/y position of the ccd on a 4x4 matrix, so that, i.e.:
        
        fig, axes = plt.subplots(4, 4, sharex=True, sharey=True)
        ax = axes[row][col]
    """
    col = 3 - (iccd - 1)%4
    row = 3 - (iccd - 1)//4
    return (row, col)
    
    
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
    if not operation in ['sum', 'mean']:
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


def combine_to_ccd(quadrants, rotate=True):
    """
        given a list of 4 np array containing the image of each readout channel, 
        returns a ccd-wise array, that is:
                            
                            2   |   1
                            ---------
                            3   |   4
        
        Parameters:
        -----------
        
            quandrants: `array-like`
                list / tuple / array with the 4 rc images. Quadrants should be 
                supplied in the right order: [q1, q2, q3, q4].
    """
    
    q1, q2, q3, q4 = quadrants[0], quadrants[1], quadrants[2], quadrants[3]
    
    if rotate:
        row_top = np.hstack((np.rot90(q2, 2), np.rot90(q1, 2)))
        row_bottom = np.hstack((np.rot90(q3, 2), np.rot90(q4, 2)))
    else:
        print ("we are not rotating anything")
        row_top = np.hstack((q2, q1))
        row_bottom = np.hstack((q3, q4))
    return np.vstack((row_bottom, row_top))
    
    
    
