#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# functions to plot and manipulate starflat data
#
# Author: M. Giomi (matteo.giomi@desy.de)

import inspect
import os
import pandas as pd
import numpy as np
from scipy.stats import binned_statistic_2d
from skimage.transform import resize


xsize, ysize = 3072, 3080


def to_array(x, y, z, resize_hist=True, **kwargs):
    """
        given a set of points, create a 2d array of desired dimensions
        by first binning it and the resizing.
    """

    # separate kwargs and merge with default
    binstat_pars = inspect.signature(binned_statistic_2d).parameters.values()
    binstat_adict = {par.name:par.default for par in binstat_pars}
    binstat_adict.update({
                        'x': x,
                        'y': y,
                        'values': z,
                        'range': [[0, xsize], [0, ysize]]
                    })
    binstat_adict.update({k:v for k, v in kwargs.items() if k in binstat_adict.keys()})

    # to compare compute binned stat
    binned = binned_statistic_2d(**binstat_adict)
    hist = binned.statistic.T
    
    # now resize the array to the pixel shape
    if resize_hist:
        resize_pars = inspect.signature(resize).parameters.values()
        resize_adict = {par.name:par.default for par in resize_pars}
        resize_adict.update({'image': hist, 'output_shape': (ysize, xsize)})
        resize_adict.update({k:v for k, v in kwargs.items() if k in resize_adict.keys()})
        return resize(**resize_adict)
    else:
        return hist


def to_chip_array(x, y, z, resize_h=True, full_ccd=False):
    """
        wrapper around to_array() function with built-in ZTF chip
        geometry.
    """
    full_range = ((-xsize, +xsize), (-ysize, +ysize))
    binsx, binsy = 64, 56 # bins of 96 pixels (x) and 110 pixels (y) for the full ccd
    out_shape=(2*ysize, 2*xsize)
    if not full_ccd:
        full_range = ((0, +xsize), (0, +ysize))
        binsx, binsy = 32, 28 
        out_shape=(ysize, ysize)
    return to_array(
                    x, y, z, 
                    bins=(binsx, binsy), stats='mean', 
                    range=full_range, mode='edge', 
                    output_shape=out_shape,
                    resize_hist=resize_h
                )

#def bin_csv(xname, yname, zname, df=None, csv_file=None, rotate=True, statistic='mean', bins=50, compression='gzip', **kwargs):
#    """
#        given a csv file with a dataframe, or a dataframe per se, bin the data according to
#        x and y, and compute some bin-wise statistic. Return the corresponding
#        2D array.

#        Uses scipy.binned_statistic_2d:
#        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic_2d.html

#        Parameters:
#        -----------

#            x[y][z]name: `str`
#                name of the variables (x, and y) which you want to bin on, and of the
#                quantity (z) you want to aggregate in each bin.

#            df: `pandas.DataFrame`
#                dataframe to use. If None, you must pass the file.

#            csv_file: `str` or None
#                path to a csv file that can be parsed with pandas.read_csv. If None,
#                you can pass a df directly.

#            rotate: `bool`
#                if True, output array will be rotated 180 deg.

#            statistic: `str`
#                which kind of aggredation function to apply to each bin.

#            bins: `int` or [`int`, `int`] or `array_like` or [`array`, `array`]
#                how to bin in each dimension

#            kwargs: `args`
#                to be passed to scipy.binned_statistic_2d. See:


#        Returns:
#        --------

#            numpy array

#    """

#    # read the file if no df is passed
#    if df is None:
#        df = pd.read_csv(csv_file, compression = 'gzip')

#    # add defaults to kwargs (standard RC size)
#    bs_range = kwargs.get('range', [[0, 3072], [0, 3080]])
#    binned = binned_statistic_2d(
#        df[xname], df[yname], df[zname], statistic=statistic, bins=bins, range=bs_range, **kwargs)
#    hist = binned.statistic.T

#    # rotate eventually and return
#    if rotate:
#        hist = np.rot90(hist, 2)
#    return hist


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
