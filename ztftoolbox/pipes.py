#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# submit commands via python
#
# Author: M. Giomi (matteo.giomi@desy.de)


import os, glob, time, os
import subprocess
import concurrent.futures
import logging
logging.basicConfig(level = logging.DEBUG)

def get_logger(logger):
    return logger if not logger is None else logging.getLogger(__name__)

def execute(cmd, wdir=None, logfile=None, logger=None, env=None, shell=False):
    """
        Execute a system command in a given folder.

        https://codereview.stackexchange.com/questions/6567/ \
        redirecting-subprocesses-output-stdout-and-stderr-to-the-logging-module
        
        Parameters:
        -----------
            
            wdir: `str`
                path to the working directory you want to run the analysis from. 
                It will be created if not existing.
            
            logfile: `str`
                path to the logfile to store command output. If None, display the
                stuff on the console.
            
            env: dict-like
                environment for the command to run in. The provide dictoinary will be used
                to update the current environment.
        
        Also see:
        https://www.endpoint.com/blog/2015/01/28/getting-realtime-output-using-python
    """
    logger = get_logger(logger)
    
    # take care of the wdir and logfile
    if (not wdir is None) and (not os.path.isdir(wdir)):
        logger.debug("creating working directory %s"%wdir)
        os.makedirs(wdir)
    if logfile is None:
        stdout = None
    else:
        logger.debug("command log will be saved to %s"%logfile)
        log_handle = open(logfile, 'w+')
        stdout = log_handle
    
    # need environment
    my_env = os.environ.copy()
    if not env is None:
        logger.debug("Custom environ variables: %s"%repr(env))
        my_env.update(env)
    
    # run the process
    popen = subprocess.Popen(
        cmd, stdout=stdout, stderr=subprocess.STDOUT, cwd=wdir, env=my_env, shell=shell)
    popen.communicate()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)
    return return_code

