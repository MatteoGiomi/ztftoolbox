#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# class to query the SO database for file. 
# Need to have auth file .pgpass in home directory
#
# Author: M. Giomi (matteo.giomi@desy.de)

import pandas as pd
from sqlalchemy import create_engine

from ztftoolbox.pipes import get_logger, execute


class ztfdb():
    """
        connect to the ZTF database
    """
    
    user    = "ztfmg"
    host    = "ztfdb2"
    dbname  = "ztf2"
    port    = 5432
    
    def __init__(self, user=user, dbname=dbname, host=host, port=port, logger=None):
        self.logger = get_logger(logger)
        self.engine = create_engine("postgresql://%s@%s:%s/%s"%(user, host, port, dbname))
    
    def query(self, expr, todf=True, columns=None):
        """
            execute a query on the database and return the results either as 
            dataframe or iterator.
            
            Parameters:
            -----------
                
                expr: `str`
                    sql query expression.
                
                todf: `bool`
                    if True, results are returned as a pandas dataframe. Set to False
                    to have just the iterable.
                
                columns: `list` or `str`
                    names of database table columns to return. If None, get them all. Works
                    only if todf is True.
            
            Returns:
            --------
                
                pandas.DataFrame or iterable with query results.
        """
        res = self.engine.execute(expr)
        if todf:
            df = pd.DataFrame(list(res), columns=res.keys())
            if columns is None:
                return df
            else:
                cols = [columns] if type(columns)==str else columns
                return df[cols]
        else:
            return res


    def get_calfiles(self, caltype, rcid=None, columns=None, date=None):
        """
            query the SODB to get the calibrated bias/flats frames generated 
            on a given night.
            
            Parameters
            ----------
                
                caltype: `str`
                    type of calibration file, e.g. 'bias' or 'hifreqflat'
                
                rcid: `int`
                    readout channel number. If None, get all of them.
                
                columns: `list` or `str`
                    names of database table columns to return. If None, get them all.
                
                date: `str`
                    night where the files where produced. If None, get them all.
            
            Returns:
            --------
                
                pandas dataframe with the query results
        """
        
        self.logger.debug("Query SODB for %s files produced on %s"%(caltype, date))
        
        # buildup query command and execute
        query_cmd = "select * from calfiles where caltype='%s'"%caltype
        if not date is None:
            query_cmd += " and startdate='%s'"%date
        if not rcid is None:
            query_cmd += " and rcid=%d"%rcid
        return self.query(query_cmd, columns=columns)
    
    
    def get_rawflats(self, date, fid, ccdid=None, columns=None):
        """
            query the SODB to get the raw domeflat frames generated on a given night.
            
            Parameters
            ----------
                
                date: `str`
                    night where the files where produced.
                
                fid: `int`
                    ZTF filter id (1='g', 2='r', 3='i').
                
                ccdid: `int`
                    CCD number (1 to 16). If None, get all of them.
                
                columns: `list` or `str`
                    names of database table columns to return. If None, get them all.
            
            Returns:
            --------
                
                pandas dataframe with the query results
        """
        
        self.logger.debug("Query SODB for raw flats produced on %s"%date)
        
        # buildup query command and execute
        query_cmd = "select c.* from ccdfiles c, nights n where fid=%d \
        and itid=4 and c.nid=n.nid and n.nightdate='%s'"%(fid, date)
        if not ccdid is None:
            query_cmd += " and c.ccdid=%d"%ccdid
        return self.query(query_cmd, columns=columns)
        




