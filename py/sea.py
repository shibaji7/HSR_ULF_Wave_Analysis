#!/usr/bin/env python

"""sea.py: superposed epoch analysis module for data processing and analsysis."""

__author__ = "Chakraborty, S."
__copyright__ = ""
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import numpy as np
from scipy.signal import resample


def sean(datalist, ynames, xname="t_instance"):
    """
    Run the statistics
    """
    l = max([len(d) for d in datalist])
    df = dict()
    for yname in ynames:
        df[yname] = dict()
        resampled_datalist, time = [], []
        for d in datalist:
            if len(d) != l:
                resampled_datalist.append(resample(d[yname], l))
            else:
                resampled_datalist.append(d[yname].tolist())
                if len(time) == 0:
                    time = d[xname].tolist()
        df[yname][xname] = time
        df[yname]["mean"] = np.nanmean(resampled_datalist, axis=0)
        df[yname]["median"] = np.nanmedian(resampled_datalist, axis=0)
        df[yname]["count"] = len(resampled_datalist[0])
        df[yname]["lq_nan"] = np.nanpercentile(resampled_datalist, 5, axis=0)
        df[yname]["uq_nan"] = np.nanpercentile(resampled_datalist, 95, axis=0)
    return df
