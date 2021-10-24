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

from matplotlib.dates import date2num

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

def ribiero_gs_flg(vel, time):
    L = np.abs(time[-1] - time[0]) * 24
    high = np.sum(np.abs(vel) > 15.0)
    low = np.sum(np.abs(vel) <= 15.0)
    if low == 0: R = 1.0  # TODO hmm... this works right?
    else: R = high / low  # High vel / low vel ratio
    # See Figure 4 in Ribiero 2011
    if L > 14.0:
        # Addition by us
        if R > 0.15: return False    # IS
        else: return True     # GS
        # Classic Ribiero 2011
        #return True  # GS
    elif L > 3:
        if R > 0.2: return False
        else: return True
    elif L > 2:
        if R > 0.33: return False
        else: return True
    elif L > 1:
        if R > 0.475: return False
        else: return True
    # Addition by Burrell 2018 "Solar influences..."
    else:
        if R > 0.5: return False
        else: return True
    # Classic Ribiero 2011
    # else:
    #    return False

def _run_riberio_threshold_on_rad(u, flag="gflg_ribiero"):
    df = u.copy()
    clust_flag = np.array(df.cluster_tag)
    gs_flg = np.zeros_like(clust_flag)
    vel = np.hstack(np.abs(df.v))
    t = np.hstack(df.time.apply(lambda x: date2num(x)))
    clust_flag[np.array(df.slist) < 7] = -1
    gs_flg = np.zeros_like(clust_flag)
    for c in np.unique(clust_flag):
        clust_mask = c == clust_flag
        if c == -1: gs_flg[clust_mask] = -1
        else: gs_flg[clust_mask] = ribiero_gs_flg(vel[clust_mask], t[clust_mask])
    df[flag] = gs_flg
    return df

if __name__ == "__main__":
    "__main__ function"
    read_rbsp_logs()
    pass