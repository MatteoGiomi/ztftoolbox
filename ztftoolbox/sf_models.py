#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 2D models to use with starflat fits
#
# Author: M. Giomi (matteo.giomi@desy.de)

import numpy as np
from numpy import sqrt
from ztftoolbox.visualize import xsize, ysize
from scipy.interpolate import SmoothBivariateSpline, RectBivariateSpline

def spline_fit(x, y, z, evaluate=False, full_ccd=True, **interp_kwargs):
    """
        spline fit to unstructured 3d data using scipy SmoothBivariateSpline.
        return callable and function evaluated over the pixels.
        
        Parameters:
        -----------
            
            x, y, z: `array-like`
                data points to fit: f(x,y) = z
            
            evaluate: `bool`
                if True, evaluate the function over the pixels
            
            full_ccd: `bool`
                if you want to evaluate the model over a full CCD or 
                just a RC.
        
        Returns:
        --------
        
            spline function (+array)
    """
    # smooth spline fit
    spline = SmoothBivariateSpline(x, y, z, **interp_kwargs)
    
    # eval spline over the pixels
    if evaluate:
        xmin, ymin = 0, 0
        if full_ccd:
            xmin, ymin = -xsize, -ysize
        x_pix, y_pix = np.arange(xmin, xsize)[np.newaxis,], np.arange(ymin, ysize)[:,np.newaxis]
        spline_arr = spline.ev(x_pix, y_pix)
        return spline, spline_arr
    else:
        return spline

def interp_array(arr):
    """
        interpolate 2d array. E.g.:
        func = interp(arr)
        func(x, y) = arr[y, x]
    """
    x_px, y_px = np.arange(arr.shape[1]), np.arange(arr.shape[0])
    return RectBivariateSpline(x_px, y_px, arr.T, kx=1, ky=1)

def sur4_rad4(xy, *coef):
    """
        4th order polinomial in 2 variables + 4th order radial term
        
        Parameters:
        -----------
            
            xy: `2darray`
                your data point in shape: (k,M), M number of points, k dimension (2 in this case)
            
            coeffs: `array-like`
                you matrix with the coefficients.
    """
    x = xy[0, :]
    y = xy[1, :]
    r = sqrt(x**2 + y**2)
    poly = (coef[0] + 
            # pure 1d poly
            coef[1]*(x**4) + coef[2]*(x**3) + coef[3]*(x**2) + coef[4]*x + 
            coef[5]*(y**4) + coef[6]*(y**3) + coef[7]*(y**2) + coef[8]*y +
            # mixed terms 4 order
            coef[9]*(x**3)*y + coef[10]*(x**2)*(y**2) + coef[11]*(y**3)*x +
            # mixed terms 3 order
            coef[12]*(x**2)*y + coef[13]*(y**2)*x +
            # mixed terms 2 order
            coef[14]*(x*y)
            )
    radial = (coef[15]*(r**4) +
              coef[16]*(r**3) +
              coef[17]*(r**2) +
              coef[18]*(r)
              )
    return (poly+radial)


def evaluate_model(model, model_params, xrange=(0, 3072), 
    yrange=(0, 3080), npp=500, px=None, py=None):
    """
        evaluate a function on the pixels. 
        Use px, py or ranges to create grid points.
    """
    
    if type(npp) in [float, int]:
        npp = (npp, npp)
    
    if px is None:
        px = np.linspace(xrange[0], xrange[1], npp[0])
    if py is None:
        py = np.linspace(yrange[0], yrange[1], npp[1])
    g_xy = np.array(np.meshgrid(px, py)).T.reshape(-1, 2).T
    return model(g_xy, *model_params).reshape(npp[0], npp[1]).T

