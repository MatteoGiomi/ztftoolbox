#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# functions to create starflats. To run for the full camera, use
#
#        import concurrent.futures
#        with concurrent.futures.ProcessPoolExecutor(max_workers = 4) as executor:
#            executor.map(starflat_rcwise, list(range(64)))
# 
#
# Author: M. Giomi (matteo.giomi@desy.de)

import os
import numpy as np
import pandas as pd

import logging
logging.basicConfig()

from scipy.optimize import curve_fit

from dataslicer.dataset import dataset
from dataslicer.metadata import metadata
from dataslicer.objtable import objtable
from dataslicer.srcdf import srcdf

from ztftoolbox.sf_models import sur4_rad4, evaluate_model, spline_fit, interp_array
from ztftoolbox.visualize import radial_profile, to_chip_array
from ztftoolbox.mosaique import split_in_quadrants, rqid

def rotate_rc_xypos(xpos, ypos, rcid=None, q=None, xmax=3072, ymax=3080):
    """
        Transform two lists of RC-wise pixel corrdinates so that the 
        radial excess sit at the origin. hint: (3k, 3k) is, in the raw 
        coordinates, always in the upper left corner.
        
        Parameters:
        -----------
            
            x[y]pos: `array-like`
                pixel coordinates.
            
            rcid: `int`
                Id of readout channel of xpos and ypos (0 to 63). If None, you 
                must provide q.
            
            q: `int`
                quadrant identifier (1 to 4).
        
        Returns:
            new rotated positions.
        
    """
    if q is None:
        q = (rcid % 4) + 1
    if q == 1:
        # radial origin in upper left
        x = - (xpos - xmax)
        y = - (ypos - ymax)
    elif q == 2:
        # radial origin in upper right
        x = xpos
        y = - (ypos - ymax)
    elif q == 3:
        # radial origin in lower right
        x = xpos
        y = ypos
    elif q == 4:
        # radial origin in lower left
        x = - (xpos - xmax)
        y = ypos
    else:
        raise ValueError("Values of q must be 1, 2, 3, or 4. Got %d instead"%q)
    return x, y

class starflatter():
    """
        holds together analysis parameters and dataslicer routines 
        to make starflat maps from PSF fit catalogs
    """
    #plotdirbase = "./diag_plots"
    
    def __init__(self, logger=None):
        """
            set default values for analysis parameters
        """
        
        self.logger = logger if not logger is None else logging.getLogger(__name__)
        
        # names of columns
        self.xcoord         = "ra"
        self.ycoord         = "dec"
        self.magkey         = "mag"
        self.errmagkey      = "sigmag"
        
        # keys from header file to read and include in the dataslicer tables
        self.meta_keys = ['MAGZP', 'EXPID', 'MAGZPUNC', 'CLRCOEFF', 'CLRCOUNC', 'OBSMJD', 'FIELDID', 'RCID']
        self.meta_keys_2join = self.meta_keys

        # clustering params
        self.cluster_size_arcsec = 3
        self.ps1cp_rs_arcsec = 3
        self.min_sources_in_cluster = 0#5

        # selection cuts
        self.preselection = "flags == 0 & snr > 10"
        self.hard_magdiff_cut = "abs(cal_mag - gmag) < 0.3"
        self.dist2ps1_cluster_cut = "dist2ps1<0.3"
        self.iqr_mag_cut = 2.5


    def starflat_rc(self, rcid, datadir, plotdir, test=False, metadata_file=None, save_csvs=False):
        """
            run the starflat analsysis on a single RC for the files.
            
            Parameters:
            -----------
                
                rcid: `int`
                    Identifier number from the readout channel (0 to 64) 
                
                datadir: `str`
                    path to flder containing the PSF catalogs for the starflat dataset
                
                test: `bool`
                    if True, run the entire analysis with 5% of the sources only.
                
                metadata_file: `str`
                    path to a file containing the metadata from fits file header for
                    all the files in the dataset. 
                
                save_csv: `bool`
                    save source table and metatdata to csv files in datadir. These 
                    makes repeating the analysis faster.
        """
        
        # load the stuff
        ds = dataset("starflat_rc%d"%rcid, datadir, logger=self.logger)
        ds.load(metadata_ext = 0, objtable_ext = 'PSF_CATALOG',
            header_keys = self.meta_keys, 
            metadata_file = metadata_file,#os.path.join(datadir, 'starflat_metadata.csv'),
            force_reload = False,
            expr = 'RCID == %d'%rcid, downcast=False, obj_tocsv=save_csvs, meta_tocsv=save_csvs)
        ds.set_plot_dir(plotdir)

        # preselecyion
        ds.select_objects(self.preselection)
        if test:
            ds.objtable.df = ds.objtable.df.sample(frac=0.05, replace=True)
        
        # add meta to sources
        ds.merge_metadata_to_sources(metadata_cols = self.meta_keys_2join, join_on = 'OBSID')
        
        # cluster and PS1 match
        ds.objtable.cluster_sources(
            cluster_size_arcsec = self.cluster_size_arcsec,
            min_samples = self.min_sources_in_cluster,
            xname = self.xcoord, yname = self.ycoord, purge_df = True)
        
        ds.objtable.match_to_PS1cal(rs_arcsec = self.ps1cp_rs_arcsec, 
            xname = self.xcoord, yname = self.ycoord, use = 'clusters', plot = True)
        
        # now remove cluster if they contain a source too far away from the PS1 cal
        ds.objtable.select_clusters(self.dist2ps1_cluster_cut, plot_x = 'dist2ps1', log = True)

        # calibrate photometry
        ds.objtable.calmag(
            self.magkey, err_mag_col = self.errmagkey, calmag_col = 'cal_mag', 
            clrcoeff_name = 'CLRCOEFF', ps1_color1 = 'gmag', ps1_color2 = 'rmag')
        
        # cut based on PS1
        ds.objtable.ps1based_outlier_rm_iqr('cal_mag', self.iqr_mag_cut, ps1mag_name='gmag', n_mag_bins=10, plot = True)
        
        # remove sources belonging to cluster with outlier
        ds.objtable.select_clusters(self.hard_magdiff_cut, plot_x = 'cal_mag', plot_y = 'gmag')
        
        # add nice column and return
        df = pd.concat(
            [ds.objtable.df,
            ( ds.objtable.df['cal_mag'] - ds.objtable.df['gmag'] ).rename('mag_diff')],
            axis =1)
        return df


class starfitter():
    """
        fit the starflat maps created by the starflatter with a given model
    """
    
    
    def __init__(self, logger=None):
        """
        """
        self.logger = logger if not logger is None else logging.getLogger(__name__)
    
    def set_model(self, model, guess='plane'):
        """
            set the model to be fitted to the maps and eventually it's guess.
            
            Parameters:
            -----------
                
                model: `str` or `callable`.
                    if string, must be one of the models defined in ztftoolbox.sf_models
                    
                guess: `array-like` or `str`
                    starting point of the fit.
        """
        
        #TODO: this is a phony method for now.
        self.model_func     = sur4_rad4
        self.model_npar     = 19
        self.model_guess    = np.zeros(19)
        self.model_guess[0] = 1
        
        
    def fit_analytical_model_rc(self, x, y, z, zerr=None, rcid=None, rotate_radial=False, plot_dir=None):
        """
            given a starflat map, fit the map of magnitude residual vs x,y position.
            
            Parameters:
            -----------
                
                x, y, z: `array-like`
                    values of the position (x, y) and magnitude residuals (z)
                
                rcid: `int`
                    ID of the readout channel (0 to 63)
                
                rotate_radial: `bool`
                    if True, the points are rotated in the x-y plane so that
                    the origin of the radial term is always in the same place. 
                    For this to work, you need to pass the quadrant identifier.
        """
        
        if rcid is None and rotate_radial:
            raise ValueError("must provide RC ID if you want to rotate the image")
        
        # transform the coordinates so that the radial excess sit at the origin
        # hint: (3k, 3k) is, in the raw coordinates, always in the upper left corner.
        if rotate_radial:
            q = (rcid % 4) + 1
            if q == 1:
                # radial origin in upper left
                x = - (x - xmax)
                y = - (y - ymax)
            elif q == 2:
                # radial origin in upper right
                x = x
                y = - (y - ymax)
            elif q == 3:
                # radial origin in lower right, nothing to do
                x = x
                y = y
            else:
                # radial origin in lower left
                x = - (x - xmax)
                y = y

        self.logger.info("Now fitting")
        xy = np.array((x, y))
        
        self.model_guess[0] = z.mean()
        
        if zerr is None:
            popt, pcov = curve_fit(
                self.model_func, xdata = xy, ydata = z, p0=self.model_guess)
        else:
            popt, pcov = curve_fit(
                self.model_func, xdata=xy, ydata=z, p0=self.model_guess, sigma=zerr, absolute_sigma=True)
        
        self.logger.debug("Least Square Fit results:")
        for ip, pp in enumerate(popt):
            self.logger.debug("param #%d = %.3e"%(ip, pp))

        # evaluate the fitted model on the pixels and at the source coordinates
        fit_surf = evaluate_model(self.model_func, model_params=popt)
        fit_residuals = z - self.model_func(xy, *popt)

        ## compute the radial profile of the image
        rad, profile, profile_e = radial_profile(x, y, z)
        res_rad, res_profile, res_profile_e = radial_profile(x, y, fit_residuals)
        self.logger.info("DATA MEAN & STD:", z.mean(), z.std())
        self.logger.info("FIT RESIDUAL MEAN & STD:", fit_residuals.mean(), fit_residuals.std())
        
        # plot the stuff
        if not plot_dir is None:
            
            import matplotlib.pyplot as plt
            vmin, vmax = -0.04, +0.04
            
            # plot fit results
            fig, axes = plt.subplots(1, 3, figsize = (15, 5))
            axes = axes.flatten()
            imshow_args = {'aspect': 'auto', 'origin': 'lower'}

            ax = axes[0]
            h_data = to_chip_array(x, y, z, bins = 35)
            im = ax.imshow(h_data, **imshow_args, vmin = vmin, vmax = vmax)
            cb = plt.colorbar(im, ax = ax)
            ax.set_title("Binned data")

            ax = axes[1]
            im = ax.imshow(fit_surf, **imshow_args, vmin = vmin, vmax = vmax)
            cb = plt.colorbar(im, ax = ax)
            ax.set_title("Fitted function")

            ax = axes[2]
            h_res = to_chip_array(x, y, fit_residuals, bins = 35)
            im = ax.imshow(h_res, cmap = 'seismic', **imshow_args, vmin = vmin, vmax = vmax)
            cb = plt.colorbar(im, ax = ax)
            ax.set_title("Fit residuals")
            
            for ax in axes:
                ax.set_xticks([])
                ax.set_yticks([])
            
            fig.tight_layout()
            fig.savefig(os.path.join(plot_dir, "fit_results_rc%d_maps.png"%rcid))
            plt.close(fig)

            # now plot the hitograms (radial profile separately)
            fig, ax = plt.subplots()
            ax.errorbar(rad, profile, xerr = 0, yerr = profile_e, fmt = 'o', label = "starflat data")
            ax.errorbar(res_rad, res_profile, xerr = 0, yerr = res_profile_e, fmt = 'o', label = "2D fit residuals")
        #    ax.plot(rad, radp_fit(rad), label = "1D fit (poly %d order)"%radp_poly_order)
            ax.set_title("radial profile")
            ax.set_xlabel("radius [pixel]")
            ax.set_ylabel("Average $m_{diff}$ in ring [mag]")
            ax.legend(loc = 'best')
            fig.tight_layout()
            fig.savefig(os.path.join(plot_dir, "fit_results_rc%d_radial.png"%rcid))
            plt.close(fig)
            
            
            # now plot the pulls and the residuals
            fig, axes = plt.subplots(1, 2, figsize = (10, 5))
            ax = axes[0]
            ax.set_title("Residuals")
            h = ax.hist(z, bins = 100, histtype = "step", label = "Data")
            h = ax.hist(fit_residuals, bins = h[1], histtype = "step", label = "Fit residuals")
            ax.legend(loc = 'best')
            ax.set_xlabel("magnitude residuals [mag]")
            ax.set_ylabel("number of data points")
            
            # and the pulls
            if not zerr is None:
                ax = axes[1]
                ax.yaxis.tick_right()
                ax.yaxis.set_label_position("right")
                ax.set_title("Pulls")
                h = ax.hist(z/zerr, bins = 100,  histtype = "step", label = "Data")
                h = ax.hist(fit_residuals/zerr, bins = h[1], histtype = "step", label = "Fit residuals")
                print ("STD of data pulls: %.2e"%np.std(z/zerr))
                print ("STD of residual pulls: %.2e"%np.std(fit_residuals/zerr))
            
            # plot a gaussinan
        #    popt, pcov = curve_fit(gauss_function, h[1][:1], h[0], p0 = [sum(h[0]), 0, 1])
        #    ax.plot(h[1],gauss_function(h[1], *popt), 'k')
            ax.set_yscale('log')
            ax.legend(loc = 'best')
            ax.set_xlabel("magnitude residuals pull ($m_{diff}/\Delta m_{diff}$)")
            ax.set_ylabel("number of data points")
            
            # plot the radial profile
            fig.savefig(os.path.join(plot_dir, "fit_results_rc%d_hist.png"%rcid))
            plt.close(fig)
        
        
        # now return the data
        data = evaluate_model(self.model_func, model_params=popt, npp=(3072, 3080))
        return data
    
    def fit_spline_model(self, sf_dfs, ccd_id, ccd_only=False):
        """
            spline fit to the starflat data (the dataframes returned by the 
            starflatter method). This method is using data from a full CCD
            to fit a smooth spline capturing the CCD-wide variations. 
            
            If ccd_only is False, the resulting model (a numpy array) is split back 
            into the 4 quadrants, and the residuals (data-ccd-wide model) for each
            quadrants are then fitted with a spline again, allowing to capture 
            higher-frequency variations and RC specific features. 
            
            The ccd-wide model and the rc-one are then summed together to build
            the full model.
            
            Parameters:
            -----------
                
                sf_dfs: `dict`
                    dictionary with 4 dfs from the starflat data, indexed
                    by their readout channel id, i.e.
                        {42: df_with_sf_data_for_rc_42, ...}
                
                ccd_id: `int`
                    ID of the CCD (1 to 16)
                
                rc_id: `int`
                    ID of the readout channel (0 to 63)
                
                ccd_only: `bool`
                    weather to make just the pass on the CCD-wide fit.
            
            Returns:
            --------
            
                models for the 4 rc given as inputs.
        """
        
        # transform to ccd wide coordinates and get all the data together
        x, y, z = np.array([]), np.array([]), np.array([])
        for rcid, df in sf_dfs.items():
            sdf = srcdf(df)
            xccd, yccd = sdf.compute_ccd_coord('xpos', 'ypos', rotate=True, rcid=rcid)
            x = np.append(x, xccd)
            y = np.append(y, yccd)
            z = np.append(z, sdf['mag_diff'])
        
        # first pass: spline fit over the full CCD
        ccd_fit, ccd_fit_eval = spline_fit(x, y, z, full_ccd=True, evaluate=True, kx=5, ky=5)
        
        # split the CCD back into its quadrants and
        # reindex it according to the RC id
        quadrants = split_in_quadrants(ccd_fit_eval)
        quadrants = {rqid(ccd_id, qid): np.rot90(qq, 2) for qid, qq in quadrants.items()}
        
        # if you are so easily satisfied then..
        if ccd_only:
            return quadrants
        
        # now, we do the second pass: select the RC you want
        out = {}
        for rc_id, rc_df in sf_dfs.items():
            # compute the residuals wrt the CCD wide model
            # and fit them again, now with RC coordinates
            res = rc_df['mag_diff'] - ccd_fit.ev(rc_df['ccd_xpos'], rc_df['ccd_ypos'])
            rc_fit, rc_fit_eval = spline_fit(
                rc_df['xpos'], rc_df['ypos'], res, full_ccd=False, evaluate=True, kx=5, ky=5)
            # add the models and keep aside
            my_mod = rc_fit_eval + quadrants[rc_id]
            out[rc_id] = my_mod
        
        # bye!
        return out
       
    def get_residuals(self, model_array, sf_df):
        """
            compute the residuals of a model
        """
        
        # create an interpolation of that model so that we can call it
        model_interp = interp_array(model_array)
        res = sf_df['mag_diff'].values - model_interp.ev(sf_df['xpos'], sf_df['ypos'])
        res_array = to_chip_array(sf_df['xpos'], sf_df['ypos'], res)
        return res, res_array
        

