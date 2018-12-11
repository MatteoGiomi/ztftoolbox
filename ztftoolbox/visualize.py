#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# functions to plot and manipulate starflat data
#
# Author: M. Giomi (matteo.giomi@desy.de)

import os
import pandas as pd
import numpy as np
from scipy.stats import binned_statistic_2d


xmin, xmax = 0, 3072
ymin, ymax = 0, 3080

def scatter_to_array_resize(x, y, z, bins = 35):
    """
        given a set of points, create a 2d array of desired dimensions
        by first binning it and the resizing.
    """
    # to compare compute binned stat
    binned = binned_statistic_2d(
        x, y, z, statistic='mean', bins = bins, range = [[xmin, xmax], [ymin, ymax]])
    hist = binned.statistic.T
    
    # now resize the array to the pixel shape
    from skimage.transform import resize
    resized = resize(hist, (ymax, xmax), order = 3, mode = 'reflect')
    return resized


def bin_csv(xname, yname, zname, df=None, csv_file=None, rotate=True, statistic='mean', bins=50, compression='gzip', **kwargs):
    """
        given a csv file with a dataframe, or a dataframe per se, bin the data according to 
        x and y, and compute some bin-wise statistic. Return the corresponding
        2D array.
        
        Uses scipy.binned_statistic_2d:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic_2d.html
        
        Parameters:
        -----------
            
            x[y][z]name: `str`
                name of the variables (x, and y) which you want to bin on, and of the
                quantity (z) you want to aggregate in each bin.
            
            df: `pandas.DataFrame`
                dataframe to use. If None, you must pass the file.
            
            csv_file: `str` or None
                path to a csv file that can be parsed with pandas.read_csv. If None, 
                you can pass a df directly.
            
            rotate: `bool`
                if True, output array will be rotated 180 deg.
            
            statistic: `str`
                which kind of aggredation function to apply to each bin.
            
            bins: `int` or [`int`, `int`] or `array_like` or [`array`, `array`]
                how to bin in each dimension
            
            kwargs: `args`
                to be passed to scipy.binned_statistic_2d. See:
                
        
        Returns:
        --------
            
            numpy array
        
    """
    
    # read the file if no df is passed
    if df is None:
        df = pd.read_csv(csv_file, compression = 'gzip')
    
    # add defaults to kwargs (standard RC size)
    bs_range = kwargs.get('range', [[0, 3072], [0, 3080]])
    binned = binned_statistic_2d(
        df[xname], df[yname], df[zname], statistic=statistic, bins=bins, range=bs_range, **kwargs)
    hist = binned.statistic.T
    
    # rotate eventually and return
    if rotate:
        hist = np.rot90(hist, 2)
    return hist


def radial_profile(x, y, z, nbins = 80, rmin = 0, rmax = 4500):
    """
        given a set of 3D points (x,y,z) compute the radial profile
        of z as a function of the radius in the xy plane.
    """
    r = np.sqrt(x**2 + y**2)
    rbins = np.linspace(rmin, rmax, nbins)
    rbinc, av_mag_radial, err_mag_radial = [], [], []
    for ib in range(1, len(rbins)):
        mask = np.logical_and(r > rbins[ib-1], r < rbins[ib])
#        print ("computing radial proile between %.2f and %.2f pixels from the center. %d points"%
#            (rbins[ib-1], rbins[ib], sum(mask)))
        data = z[mask]
        if len(data) == 0:
            continue
        av_mag_radial.append(np.average(data))
        err_mag_radial.append(np.std(data)/np.sqrt(len(data)))
        rbinc.append((rbins[ib-1] + rbins[ib])/2.)     #np.sqrt(rbins[ib-1]*rbins[ib])
    r = np.array(rbinc)
    y = np.array(av_mag_radial)
    yerr = np.array(err_mag_radial)
    return (r, y, yerr)

#def do_starflat_map(df_files, df_cut = None, name = None, overwrite = False):
#    """
#    Parameters:
#    -----------
#    
#        files: `list`
#            a list 64 csv files containing the processed catalogs (as DataFrames).
#            One file for each RC.
#        
#        df_cut: `str` or None
#            string to select entries in the dataframe.
#        
#        name: `str` or None
#            if given, the numpy array with the full camera will be saved to file.
#            if given and df_cut is not None, the df_cut expression will be added to
#            the name.
#        
#        overwrite: `bool`
#            if a file name is given and the file exist, the starflat map is loaded
#            from this file, unless ovewrite
#    
#    Returns:
#    --------
#        
#        array of the full starflat map.
#    """
#    
#    # add cut to name
#    if not name is None:
#        saveto = name
#        if not df_cut is None:
#            saveto = name+df_cut.replace(" ", "")
#    
#    # if a file is already there, read it
#    if (not name is None) and os.path.isfile(saveto+".npy") and (not overwrite):
#        print ("reading array from: %s"%saveto)
#        return np.load(saveto+".npy")

#    full_camera = np.array([ None for _ in range(64)]).reshape(8, 8)
#    for f in tqdm.tqdm(df_files):
##        print ("computing starflat map for file: %s"%f)
#        
#        # figure out readout channel
#        rcid = int(f.split('rc')[-1].replace('.csv', ''))
#        
#        # read and eventually select and then bin
#        df = pd.read_csv(f, compression = 'gzip', engine = 'c', memory_map = True)
#        if not df_cut is None:
#            df = df.query(df_cut)
#        
#        print ("mean mag diff:", df['mag_diff'].mean())
#        print ("std:", df['mag_diff'].std())
#        print ("skew:", df['mag_diff'].skew())
#        plt.hist(df['mag_diff'], bins = 100, log = True)
#        plt.show()
#        
#        # get the starflat map from the df
#        arr = scatter_to_array(df, 'xpos', 'ypos', 'mag_diff')
#        arr = np.rot90(arr, 2)
#        
##        arr = np.random.random()*np.ones((8, 8))
##        if rcid == 61:
##            arr = 10*np.ones((8, 8))

#        # figure out where to plot
#        plotx, ploty = getaxesxy(*ccdqid(rcid))
#        full_camera[plotx][ploty] = arr
#    
#    # combine together the quadrants
#    rows = []   
#    for irow in range(8):
#        rows.append( np.hstack([full_camera[icol][7-irow] for icol in range(8)]) )
#    full_camera = np.vstack(rows)

#    # save it
#    if not name is None:
#        print ("saving camera wide starflat map into: %s"%saveto)
#        np.save(saveto, full_camera)
#    return full_camera

