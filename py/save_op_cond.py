#!/usr/bin/env python

"""
    save_op_cond.py: module to sace opvation prime conductance to superdarn wave event file
"""

__author__ = "Shi, X."
__copyright__ = ""
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Shi, X."
__email__ = "xueling7@vt.edu"
__status__ = "Research"

import datetime
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from geospacepy import satplottools
from ovationpyme.ovation_prime import ConductanceEstimator


def draw_interpolated_conductance(
    new_mlat_grid, new_mlt_grid, dt, hemi, fluxtypes=["diff", "mono"]
):
    """
    Interpolate hall and pedersen conductance
    onto grid described by mlat_grid, mlt_grid
    """
    estimator = ConductanceEstimator(fluxtypes=fluxtypes)

    mlatgrid, mltgrid, pedgrid, hallgrid = estimator.get_conductance(
        dt, hemi=hemi, auroral=True, solar=True
    )

    ped_interpolator = LatLocaltimeInterpolator(mlatgrid, mltgrid, pedgrid)
    new_pedgrid = ped_interpolator.interpolate(new_mlat_grid, new_mlt_grid)

    hall_interpolator = LatLocaltimeInterpolator(mlatgrid, mltgrid, hallgrid)
    new_hallgrid = hall_interpolator.interpolate(new_mlat_grid, new_mlt_grid)

    f = plt.figure(figsize=(11, 5))
    aH = f.add_subplot(121)
    aP = f.add_subplot(122)

    X, Y = satplottools.latlt2cart(
        new_mlat_grid.flatten(), new_mlt_grid.flatten(), hemi
    )
    X = X.reshape(new_mlat_grid.shape)
    Y = Y.reshape(new_mlt_grid.shape)

    satplottools.draw_dialplot(aH)
    satplottools.draw_dialplot(aP)

    mappableH = aH.pcolormesh(X, Y, new_hallgrid, vmin=0.0, vmax=20.0)
    mappableP = aP.pcolormesh(X, Y, new_pedgrid, vmin=0.0, vmax=15.0)

    aH.set_title("Hall Conductance")
    aP.set_title("Pedersen Conductance")

    f.colorbar(mappableH, ax=aH)
    f.colorbar(mappableP, ax=aP)

    f.suptitle(
        "OvationPyme Interpolated Conductance {0} Hemisphere at {1}".format(
            hemi, dt.strftime("%c")
        ),
        fontweight="bold",
    )
    return f, new_pedgrid, new_hallgrid


def draw_conductance(dt, hemi, fluxtypes=["diff", "mono"]):
    """
    Get the hall and pedersen conductance for one date and hemisphere
    """
    estimator = ConductanceEstimator(fluxtypes=fluxtypes)

    mlatgrid, mltgrid, pedgrid, hallgrid = estimator.get_conductance(
        dt, hemi=hemi, auroral=True, solar=True
    )

    f = plt.figure(figsize=(11, 5))
    aH = f.add_subplot(121)
    aP = f.add_subplot(122)

    X, Y = satplottools.latlt2cart(mlatgrid.flatten(), mltgrid.flatten(), hemi)
    X = X.reshape(mlatgrid.shape)
    Y = Y.reshape(mltgrid.shape)

    satplottools.draw_dialplot(aH)
    satplottools.draw_dialplot(aP)

    mappableH = aH.pcolormesh(X, Y, hallgrid, vmin=0.0, vmax=20.0)
    mappableP = aP.pcolormesh(X, Y, pedgrid, vmin=0.0, vmax=15.0)

    aH.set_title("Hall Conductance")
    aP.set_title("Pedersen Conductance")

    f.colorbar(mappableH, ax=aH)
    f.colorbar(mappableP, ax=aP)

    f.suptitle(
        "OvationPyme Conductance Output {0} Hemisphere at {1} \n".format(
            hemi, dt.strftime("%c")
        ),
        fontweight="bold",
    )
    return f, mlatgrid, mltgrid, pedgrid, hallgrid


def save_conductance_csv(dt, hemi, fn_cond, fluxtypes=["diff", "mono", "wave"]):
    """
    Get the hall and pedersen conductance for one date and hemisphere
    and save to csv file
    """
    estimator = ConductanceEstimator(fluxtypes=fluxtypes)

    mlatgrid, mltgrid, pedgrid, hallgrid = estimator.get_conductance(
        dt, hemi=hemi, auroral=True, solar=True
    )
    df = pd.DataFrame(
        {
            "mlatgrid": mlatgrid.flatten(),
            "mltgrid": mltgrid.flatten(),
            "pedgrid": pedgrid.flatten(),
            "hallgrid": hallgrid.flatten(),
        }
    )
    df.to_csv(fn_cond, index=False, float_format="%.4f")


def save_wave_event_cond_batch(
    fn_wave="wave_info_db/201507_v_los_igrf.csv",
    fn_new="wave_info_db/201507_v_los_igrf_cond.csv",
    hemi="N",
    homedir="/home/xueling/",
    fluxtypes=["diff", "mono", "wave"],
):

    df_wave = pd.read_csv(fn_wave)
    df_wave["stime"] = df_wave["stime"].astype("datetime64[ns]")
    mt_tmp = df_wave["stime"] + datetime.timedelta(minutes=30)
    df_mt = [element.to_pydatetime() for element in list(mt_tmp)]
    # df_mt = np.array()

    df_mlat = np.array(df_wave["mlat"])
    df_mlt = np.array(df_wave["mlt"])
    op_mlat = []
    op_mlt = []
    op_ped = []
    op_hall = []
    for ii, tt in enumerate(df_mt):
        fn1 = "op_dt_hemi_cond_{}.csv".format(tt.strftime("%Y%m%d_%H%M%S"))
        fn_cond = os.path.join(homedir, fn1)
        # print(fn_cond)

        if ~os.path.exists(fn_cond):
            save_conductance_csv(tt, hemi, fn_cond, fluxtypes=fluxtypes)

        df_cond = pd.read_csv(fn_cond)
        mlatgrid = np.array(df_cond["mlatgrid"])
        mltgrid = np.array(df_cond["mltgrid"])
        pedgrid = np.array(df_cond["pedgrid"])
        hallgrid = np.array(df_cond["hallgrid"])

        # find the closest cell to locate conductance
        mlat_tmp = df_mlat[ii]
        mlt_tmp = df_mlt[ii]
        dist_tmp = np.sqrt((mlat_tmp - mlatgrid) ** 2 + (15 * (mlt_tmp - mltgrid)) ** 2)
        cell_ind = np.argmin(dist_tmp)

        op_mlat.append(mlatgrid[cell_ind])
        op_mlt.append(mltgrid[cell_ind])
        op_ped.append(pedgrid[cell_ind])
        op_hall.append(hallgrid[cell_ind])

    df_wave["op_mlat"] = op_mlat
    df_wave["op_mlt"] = op_mlt
    df_wave["op_ped"] = op_ped
    df_wave["op_hall"] = op_hall

    df_new = pd.DataFrame(df_wave)
    df_new.to_csv(fn_new, index=False)


if __name__ == "__main__":
    main_dir = "/home/xueling/Dropbox/wave_info_db/"
    event_fnames = [
        "201501_v_los_igrf",
        "201502_v_los_igrf",
        "201503_v_los_igrf",
        "201504_v_los_igrf",
        "201505_v_los_igrf",
        "201506_v_los_igrf",
        "201507_v_los_igrf",
        "201508_v_los_igrf",
        "201509_v_los_igrf",
        "201510_v_los_igrf",
        "201511_v_los_igrf",
        "201512_v_los_igrf",
    ]
    for fn in event_fnames:
        t = time.time()
        fn_wave = main_dir + fn + ".csv"
        fn_new = main_dir + fn + "_new.csv"
        save_wave_event_cond_batch(
            fn_wave=fn_wave,
            fn_new=fn_new,
            hemi="N",
            homedir="/home/xueling/op_cond_meta/",
            fluxtypes=["diff", "mono", "wave"],
        )
        print("Done printing file " + fn_new)
        print(time.time() - t)
