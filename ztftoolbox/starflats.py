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


from dataslicer.dataset import dataset
from dataslicer.metadata import metadata
from dataslicer.objtable import objtable


# ----- global analysis settings ------ #

# naming and paths
xcoord, ycoord = "ra", "dec"
magkey, errmagkey = "mag", "sigmag"
plotdirbase = "./diag_plots"


# keys to keep
meta_keys = ['MAGZP', 'EXPID', 'MAGZPUNC', 'CLRCOEFF', 'CLRCOUNC', 'OBSMJD', 'FIELDID', 'RCID']
meta_keys_2join = meta_keys

# clustering params
cluster_size_arcsec = 3
ps1cp_rs_arcsec = 3
min_sources_in_cluster = 0#5

# selection cuts
preselection = "flags == 0 & snr > 10"
hard_magdiff_cut = "abs(cal_mag - gmag) < 0.3"
dist2ps1_cluster_cut = "dist2ps1<0.3"
iqr_mag_cut = 2.5

test = False


# ------------------------------------- #


def starflat_rcwise(rcid, datadir):
    """
        run the starflat analsysis on a single RC for the files 
    """
    
    # load the stuff
    ds = dataset("starflat_rc%d"%rcid, datadir)
    ds.load(metadata_ext = 0, objtable_ext = 'PSF_CATALOG',
        header_keys = meta_keys, 
#        metadata_file = os.path.join(datadir, 'starflat_metadata.csv'),
        force_reload = False,
        expr = 'RCID == %d'%rcid, downcast=False, obj_tocsv=False, meta_tocsv=False)
    ds.set_plot_dir(plotdirbase)

    # preselecyion
    ds.select_objects(preselection)
    if test:
        ds.objtable.df = ds.objtable.df.sample(frac=0.05, replace=True)
    
    # add meta to sources
    ds.merge_metadata_to_sources(metadata_cols = meta_keys_2join, join_on = 'OBSID')
    
    # cluster and PS1 match
    ds.objtable.cluster_sources(
        cluster_size_arcsec = cluster_size_arcsec,
        min_samples = min_sources_in_cluster,
        xname = xcoord, yname = ycoord, purge_df = True)
    ds.objtable.match_to_PS1cal(rs_arcsec = ps1cp_rs_arcsec, 
        xname = xcoord, yname = ycoord, use = 'clusters', plot = True)
    
    # now remove cluster if they contain a source too far away from the PS1 cal
    ds.objtable.select_clusters(dist2ps1_cluster_cut, plot_x = 'dist2ps1', log = True)

    # calibrate photometry
    ds.objtable.calmag(
        magkey, err_mag_col = errmagkey, calmag_col = 'cal_mag', 
        clrcoeff_name = 'CLRCOEFF', ps1_color1 = 'gmag', ps1_color2 = 'rmag')
    
    # cut based on PS1
    ds.objtable.ps1based_outlier_rm_iqr('cal_mag', iqr_mag_cut, ps1mag_name='gmag', n_mag_bins=10, plot = True)
    
    # remove sources belonging to cluster with outlier
    ds.objtable.select_clusters(hard_magdiff_cut, plot_x = 'cal_mag', plot_y = 'gmag')
    
    
    return ds.objtable.df
    
#    # save the dataframe to file
#     = pd.concat(
#        [ds.objtable.df,
#        ( ds.objtable.df['cal_mag'] - ds.objtable.df['gmag'] ).rename('mag_diff')],
#        axis =1)
#    return 
    
#####################
# now run the stuff #
#####################

#import concurrent.futures
#with concurrent.futures.ProcessPoolExecutor(max_workers = 4) as executor:
#    executor.map(starflat_rcwise, list(range(64)))
#print ("done!")

