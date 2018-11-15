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
import pandas as pd

import logging
logging.basicConfig(level = logging.DEBUG)

from dataslicer.dataset import dataset
from dataslicer.metadata import metadata
from dataslicer.objtable import objtable


# ----- global analysis settings ------ #


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
        
        self.logger if not logger is None else logging.getLogger(__name__)
        
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
        ds = dataset("starflat_rc%d"%rcid, datadir)
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
        
        
        return ds.objtable.df
    
#        # save the dataframe to file
#         = pd.concat(
#            [ds.objtable.df,
#            ( ds.objtable.df['cal_mag'] - ds.objtable.df['gmag'] ).rename('mag_diff')],
#            axis =1)
#        return 
#        


#import concurrent.futures
#with concurrent.futures.ProcessPoolExecutor(max_workers = 4) as executor:
#    executor.map(starflat_rcwise, list(range(64)))
#print ("done!")

