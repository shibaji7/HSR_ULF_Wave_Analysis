#!/usr/bin/env python

"""
    populate_sw_params.py: Populate the rows using datasets.
"""

__author__ = "Chakraborty, S."
__copyright__ = ""
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import numpy as np
import pandas as pd
import glob
import ray

BASE_LOCATION = "tmp/data/"
CSV = "tmp/sd.run.13/analysis/"

def fetch_os_data(row):
    """
    Fetch parameters based on date time range
    """
    stime, etime = row["stime"], row["etime"]
    param_files = BASE_LOCATION + "omni/%04d%02d.csv"%(stime.year,stime.month)
    df = pd.read_csv(param_files, parse_dates=["DATE"])
    df = df[
        (df.DATE >= stime) &
        (df.DATE < etime)
    ]
    row["sym-h"] = np.nanmean(df["SYM-H"])
    row["ae"] = np.nanmean(df["AE"])
    row["v"] = np.nanmean(df["V"])
    row["p_dyn"] = np.nanmean(df["P_DYN"])
    row["bx"] = np.nanmean(df["Bx"])
    row["by_gsm"] = np.nanmean(df["By_GSM"])
    row["bz_gsm"] = np.nanmean(df["Bz_GSM"])
    return row

@ray.remote
def populate_parameter_by_file(f):
    """
    Parallel process each file
    """
    o = pd.read_csv(f, parse_dates=["stime", "etime"])
    if "sym-h" not in o.columns.tolist():
        o = o.apply(lambda r: fetch_os_data(r), axis=1)
        o.to_csv(f, index=False, header=True)
    return 0

def populate_parameters():
    """
    Populate datasets by each files
    """
    # Initialize Ray
    ray.init(num_cpus = 4)
    
    files = glob.glob(CSV + "*.csv")
    files.sort()
    
    # Run Parallel Ray to compute parameters
    params = []
    for f in files:
        params.append(
            populate_parameter_by_file.remote(f)
        )
    oplist = ray.get(params)    
    return

if __name__ == "__main__":
    populate_parameters()