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


def combineimages(files, outfile, plot=True, gapX = 462, gapY = 645, 
    fill_value = 0, newshape=None, overwrite=False):  #, newshape=(616, 512))
    """ 
        Combine CCD images into a full frame and save the output to a file. 
        Adapted from Matthew's combine_cal_files
  
        Parameters
        ----------
        files: `list`
            list of 16 fits files (one for ccd), that has to be combined together.
            IMPORTANT: this list has to be sorted according to ccd ID, otherwise the 
            trick won't work!
        outfile: `str`
            name of the output file that will be produced.
        plot: `bool`
            if true, a plot will be created. Only works if newshape is not None, 
            that is, if the data has been downsampled.
        gapX : int
            The separation between CCDs in the x (RA) direction
        gapY : int
            The separation between CCDs in the y (Dec) direction
        fill_value : int
            The value for pixels in the CCD gaps
        newshape : tuple
            if not None, the data array is downsampled to the newshape using bin_ndarray.
    """
    
    # here is the modified Matthew's function
    if not os.path.isfile(outfile) or overwrite:
        print ("combinig images...")
        rows = []
        for ccdrow in tqdm.tqdm(range(4)):
            for qrow in range(2):
                chunks = []
                for ccd in range(4): 
                    hdulist = fits.open(files[4 * ccdrow + (4 - ccd) - 1])
                    if qrow == 0:
                        img_data_1 = hdulist[3].data   
                        img_data_2 = hdulist[4].data
                    else:
                        img_data_1 = hdulist[2].data
                        img_data_2 = hdulist[1].data
                    
                    if not newshape is None:
                        img_data_1=bin_ndarray(img_data_1, newshape, 'mean')
                        img_data_2=bin_ndarray(img_data_2, newshape, 'mean')
                    
                    # Rotate by 180 degrees
                    img_data_1 = np.rot90(img_data_1, 2)
                    img_data_2 = np.rot90(img_data_2, 2)
                    x_gap = np.zeros((img_data_1.shape[0], gapX)) + fill_value
                    chunks.append(np.hstack((img_data_1, img_data_2)))
                row_data = np.hstack((chunks[0], x_gap, chunks[1], x_gap, chunks[2], x_gap, chunks[3]))
                rows.append(row_data)
            if ccdrow < 3: rows.append(np.zeros((gapY, row_data.shape[1])) + fill_value)
        array = np.vstack(rows)
        fits.writeto(
            outfile, array, header = fits.getheader(files[0], 0), overwrite=overwrite)
        print ("combined image saved to: %s"%outfile)

    # eventually plot it
    if (plot is True) and os.path.isfile(outfile):  #not (newshape is None) 
    
        fig, ax=plt.subplots(figsize=(10, 10))
        try:
            norm=ImageNormalize(array, interval=ZScaleInterval())
        except UnboundLocalError:
            array=fits.getdata(outfile, 0)
            norm=ImageNormalize(array, interval=ZScaleInterval())
        im=ax.imshow(
            array, origin='lower', aspect='auto', 
            cmap='nipy_spectral', norm=norm, 
            interpolation='none')
        cb=plt.colorbar(im, ax=ax)
        cb.set_label("mean $\Delta$t [s] in each pixel group")
        ax.set_title(outfile.split("/")[-1].replace(".fits.fz", ""))
        outplot=outfile.replace(".fits", "").replace(".fz", "")+".png"
        fig.savefig(outplot)
        print ("plot saved to: %s"%outplot)


def plot_ccd_image(ccdfile, outfile, rotate = True, cmap = "Greys", clabel = ""):
    """
        make a plot with the 4 rc in one raw image file
    """
    
    # read in the data and eventually rotate it
    data = []
    for irc in range(1, 5):
        if rotate:
            rc_img = np.rot90(fits.getdata(ccdfile, irc), 2)
        else:
            rc_img = fits.getdata(ccdfile, irc)
        data.append(np.float32(rc_img))
    
    # prepare the plot
    fig, ((ax2, ax1), (ax3, ax4)) = plt.subplots(
        2, 2, figsize=(8, 8), sharex=True, sharey=True)
    
    # combine all the data to have a global scaling
    flatdata = np.array(data).flatten()
    norm=ImageNormalize(flatdata, interval=ZScaleInterval())
    
    # plot 
    for irc in range(1, 5):
        # pick the right axes
        if irc==1:
            ax=ax1
        elif irc==2:
            ax=ax2
        elif irc==3:
            ax=ax3
        else:
            ax=ax4
        
        im=ax.imshow(data[irc -1], origin='lower', norm=norm, aspect='auto', cmap = cmap)
        ax.set_xticks([])
        ax.set_yticks([])
        
    # add colorbar and save plot
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.0, hspace=0.0)
    cb=fig.colorbar(im, ax=[ax1, ax2, ax3, ax4], pad=0.02)
    cb.set_label(clabel)
    fig.savefig(outfile)
    print ("plot saved to:", outfile)
    plt.close(fig)

def split_in_rc(infile, outfile_tmpl = None, overwrite = False, 
        dtype  = 'float32', rm_original = False, compress = True, **writeto_kwargs):
    """
        split a ZTF raw image file (CCD wise) into the 4 files correspoding to the 
        readout channels.  
    
        Parameters:
        -----------
        
            infile: `str`
                path to the input CCD, raw image file.
            
            outile_tmpl: `str` 
                name template for the output files. It has to contain a sequence 
                that can be formatted via a single integer, specifiyng the readout channel.
                If None defaults to infile+'_rc%02d_'.
            
            overwrite: `bool`
                self explaining
            
            dtype: `str` or built-in data type
                data type to case the images to
            
            rm_original: `bool`
                weather or not the original file has to be removed.
            
            compress: `bool`
                if True, write the file as a CompressedHDU, docs at:
                http://docs.astropy.org/en/stable/io/fits/api/images.html#astropy.io.fits.CompImageHDU
            
            writeto_kwargs: `astropy.io.fits.writeto kwargs`
                additional arguments to pass the astropy.io.fits.writeto method
            
    """
    
    if outfile_tmpl is None:
        pieces = os.path.basename(infile).split(".fits")
        pieces.insert(len(pieces)-1, "_rc%02d.fits")
        outfile_tmpl =  os.path.join(os.path.dirname(infile), "".join(pieces))
    
    hudl = fits.open(infile)
    iccd = hudl[0].header['CCD_ID']
    for irq in range(1, 5):
        rcid =  rqid(iccd, irq)
        outfile = outfile_tmpl%rcid
        if os.path.isfile(outfile) and not overwrite:
            continue
        hudl[irq].header['EXTNAME'] = str(1)
        if compress:
            chdu = fits.CompImageHDU(hudl[irq].data.astype(dtype), hudl[irq].header)
            chdu.writeto(outfile, overwrite = overwrite, **writeto_kwargs)
        else:
            fits.writeto(
            outfile, 
            data = hudl[irq].data.astype(dtype), 
            header = hudl[irq].header, overwrite = overwrite, **writeto_kwargs)
            
    hudl.close()
    if rm_original:
        os.remove(infile)
