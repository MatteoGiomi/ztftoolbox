#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# functions to query the SO database for file. 
# Need to have auth file .pgpass in home directory
#
# Author: M. Giomi (matteo.giomi@desy.de)

import pandas as pd
from ztftoolbox.pipes import get_logger, execute


from sqlalchemy import create_engine, MetaData 

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
    
    def query(self, expr, todf=True):
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
            
            Returns:
            --------
                
                pandas.DataFrame or iterable with query results.
        """
        res = self.engine.execute(expr)
        if todf:
            return pd.DataFrame(list(res), columns=res.keys())
        else:
            return res



def get_calfiles(caltype, startdate=None, enddate=None, logger=None):
    """
        query the SODB to get the raw bias frames generated between startdate 
        and enddate.
        
        Parameters
        ----------
            
            start[end]date: `str`
                time limits for the queries. Can be None to open up the interval
                in one (or both) sides.
        
        Returns:
        --------
    """
    
    logger = get_logger(logger)
    logger.debug("Query SODB for %s files produced bewtween %s and %s"%
        (caltype, startdate, enddate))
    
    # buildup query command
    query_cmd = "select * from calfiles where caltype='%s'"%caltype
    if not startdate is None:
        query_cmd += " and startdate='%s'"%startdate
    if not enddate is None:
        query_cmd += " and enddate='%s'"%enddate
    
    # connect to database and execute the query
    db = ztfdb()
    res = db.query(query_cmd)
    print (res.columns.values)
    return res['filename']




