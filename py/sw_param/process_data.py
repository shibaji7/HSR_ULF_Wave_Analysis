#!/usr/bin/env python

"""process_data.py: process data do a multi-stage processing of all the ascii type files and save as .hdf5 file store."""

__author__ = "Chakraborty, S."
__copyright__ = ""
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"


import datetime as dt
import glob

import numpy as np
import pandas as pd

np.random.seed(0)

BASE_LOCATION = "/home/shibaji/OneDrive/SuperDARN-Data-Share/Shi/HSR/data/"


######################################################################################################
## This function reads all the files from omni raw database and convert to .hdf5 files
######################################################################################################
def raw_omni_to_hdf5():
    files = glob.glob(BASE_LOCATION + "omni/raw/*.asc")
    hdf5_base = BASE_LOCATION + "omni/%s.csv"
    header = [
        "DATE",
        "ID_IMF",
        "ID_SW",
        "nIMF",
        "nSW",
        "POINT_RATIO",
        "TIME_SHIFT(sec)",
        "RMS_TIME_SHIFT",
        "RMS_PF_NORMAL",
        "TIME_BTW_OBS",
        "Bfa",
        "Bx",
        "By_GSE",
        "Bz_GSE",
        "By_GSM",
        "Bz_GSM",
        "B_RMS",
        "Bfa_RMS",
        "V",
        "Vx_GSE",
        "Vy_GSE",
        "Vz_GSE",
        "n",
        "T",
        "P_DYN",
        "E",
        "BETA",
        "MACH_A",
        "X_GSE",
        "Y_GSE",
        "Z_GSE",
        "BSN_Xgse",
        "BSN_Ygse",
        "BSN_Zgse",
        "AE",
        "AL",
        "AU",
        "SYM-D",
        "SYM-H",
        "ASY-D",
        "ASY-H",
        "PC-N",
        "MACH_M",
    ]
    for fname in files:
        hdf5_fname = hdf5_base % (fname.split("/")[-1].replace(".asc", ""))
        print(fname, "-to-", hdf5_fname)
        with open(fname, "r") as f:
            lines = f.readlines()
        linevalues = []
        for i, line in enumerate(lines):
            values = list(filter(None, line.replace("\n", "").split(" ")))
            timestamp = dt.datetime(
                int(values[0]), 1, 1, int(values[2]), int(values[3])
            ) + dt.timedelta(days=int(values[1]) - 1)
            linevalues.append(
                [
                    timestamp,
                    int(values[4]),
                    int(values[5]),
                    int(values[6]),
                    int(values[7]),
                    int(values[8]),
                    int(values[9]),
                    int(values[10]),
                    float(values[11]),
                    int(values[12]),
                    float(values[13]),
                    float(values[14]),
                    float(values[15]),
                    float(values[16]),
                    float(values[17]),
                    float(values[18]),
                    float(values[19]),
                    float(values[20]),
                    float(values[21]),
                    float(values[22]),
                    float(values[23]),
                    float(values[24]),
                    float(values[25]),
                    float(values[26]),
                    float(values[27]),
                    float(values[28]),
                    float(values[29]),
                    float(values[30]),
                    float(values[31]),
                    float(values[32]),
                    float(values[33]),
                    float(values[34]),
                    float(values[35]),
                    float(values[36]),
                    float(values[37]),
                    float(values[38]),
                    float(values[39]),
                    float(values[40]),
                    float(values[41]),
                    float(values[42]),
                    float(values[43]),
                    float(values[44]),
                    float(values[45]),
                ]
            )
        _o = pd.DataFrame(linevalues, columns=header)
        _o.to_csv(hdf5_fname, header=True, index=False)
    return


######################################################################################################
## This function reads all the files from symh raw database and convert to .hdf5 files
######################################################################################################
def raw_symh_to_hdf5():
    files = glob.glob(BASE_LOCATION + "geomag/symh/raw/*.asc")
    hdf5_base = BASE_LOCATION + "geomag/symh/%s.csv"
    for fname in files:
        hdf5_fname = hdf5_base % (fname.split("/")[-1].replace(".asc", ""))
        print(fname, "-to-", hdf5_fname)
        with open(fname, "r") as f:
            lines = f.readlines()
        header = []
        linevalues = []
        for i, line in enumerate(lines[14:]):
            values = list(filter(None, line.replace("\n", "").split(" ")))
            if i == 0:
                del values[-1]
                del values[1]
                header = values
            else:
                linevalues.append(
                    [
                        dt.datetime.strptime(
                            values[0] + " " + values[1], "%Y-%m-%d %H:%M:%S.%f"
                        ),
                        int(values[2]),
                        float(values[3]),
                        float(values[4]),
                        float(values[5]),
                        float(values[6]),
                    ]
                )
        _o = pd.DataFrame(linevalues, columns=header)
        _o.to_csv(hdf5_fname, header=True, index=False)
    return


######################################################################################################
## This function reads all the files from Kp raw database and convert to .hdf5 files
######################################################################################################
def raw_Kp_to_hdf5():
    fname = BASE_LOCATION + "geomag/Kp/raw/Kp.asc"
    hdf5_fname = BASE_LOCATION + "geomag/Kp/Kp.csv"
    with open(fname, "r") as f:
        lines = f.readlines()
    linevalues = []
    for line in lines[3:-1]:
        values = line.replace("\n", "")[0:25]
        date = dt.datetime.strptime(values[0:8], "%Y%m%d")
        for i in range(8):
            I = 9 + 2 * i
            linevalues.append(
                [date + dt.timedelta(hours=3 * i), values[I : I + 2].replace(" ", "")]
            )
    header = ["DATE", "Kp"]
    _o = pd.DataFrame(linevalues, columns=header)
    _o.to_csv(hdf5_fname, header=True, index=False)
    return


if __name__ == "__main__":
    raw_Kp_to_hdf5()
    # raw_symh_to_hdf5()
    raw_omni_to_hdf5()
