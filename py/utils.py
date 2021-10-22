#!/usr/bin/env python

"""utils.py: utility module for data parsing and helping analsysis or plots."""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2021, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import datetime as dt
import numpy as np
import glob
from loguru import logger
import configparser

def read_rbsp_logs(_filestr="config/logs/*.txt"):
    """
    Read all log files and store them each records 
    to a list of dictionary
    """
    files = glob.glob(_filestr)
    files.sort()
    rbsp_modes = []
    for f in files:
        logger.info(f"RBSP log file {f}")
        with open(f, "r") as fp: lines = fp.readlines()
        for l in lines:
            l = l.replace("\n", "").split(" ")
            l = list(filter(None, l))
            r, st, et = l[0], dt.datetime.strptime(l[1]+"T"+l[2], "%Y-%m-%dT%H:%M:%S"),\
                            dt.datetime.strptime(l[3]+"T"+l[4], "%Y-%m-%dT%H:%M:%S")
            b = [int(l[5].replace("[","").replace(",", "")), 
                 int(l[6].replace(",","")), 
                 int(l[7].replace("]","").replace(",", ""))]
            op = [int(l[8]), int(l[9])]
            o = dict(
                rad = r,
                beams = b,
                stime = st,
                etime = et,
                other_params = op
            )
            rbsp_modes.append(o)
    logger.info(f"Total RBSP mode log entry {len(rbsp_modes)}")
    return rbsp_modes

def get_config(key, section="HSR"):
    config = configparser.ConfigParser()
    config.read("config/conf.ini")
    val = config[section][key]
    return val

def get_gridded_parameters(q, xparam="beam", yparam="slist", zparam="v", r=0, rounding=False):
    """
    Method converts scans to "beam" and "slist" or gate
    """
    plotParamDF = q[ [xparam, yparam, zparam] ]
    if rounding:
        plotParamDF.loc[:, xparam] = np.round(plotParamDF[xparam].tolist(), r)
        plotParamDF.loc[:, yparam] = np.round(plotParamDF[yparam].tolist(), r)
    else:
        plotParamDF[xparam] = plotParamDF[xparam].tolist()
        plotParamDF[yparam] = plotParamDF[yparam].tolist()
    plotParamDF = plotParamDF.groupby( [xparam, yparam] ).mean().reset_index()
    plotParamDF = plotParamDF[ [xparam, yparam, zparam] ].pivot( xparam, yparam )
    x = plotParamDF.index.values
    y = plotParamDF.columns.levels[1].values
    X, Y  = np.meshgrid( x, y )
    # Mask the nan values! pcolormesh can't handle them well!
    Z = np.ma.masked_where(
            np.isnan(plotParamDF[zparam].values),
            plotParamDF[zparam].values)
    return X,Y,Z

if __name__ == "__main__":
    "__main__ function"
    read_rbsp_logs()
    pass