#!/usr/bin/env python

"""extreme_event_analysis.py: plots module for data post processing and analysis."""

__author__ = "Shi, X."
__copyright__ = ""
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

# Import the modules we need.
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use(['ieee', 'science'])


import sys
sys.path.append("py/")
from analytics import Stats, DialPlot, TimeSeriesAnalysis, StackPlots
import numpy as np
import os

def generate_statistics(
    files, Ndpint_threshold, I_sig_threshold,
    bdr_threshold, jhr_lim, 
):
    """
    Populate statistics for all events
    """
    fig_dir = "/".join(files.split("/")[:-1]) + "/ex_figures/"
    os.makedirs(fig_dir, exist_ok=True)
    # Read datasets
    stat = Stats(
        files,
        Ndpint_threshold=Ndpint_threshold,
        I_sig_threshold=I_sig_threshold,
        bdr_threshold=bdr_threshold
    )
    # Fetch Extreme Events
    events, _ = stat.get_valid_extreme_events(jhr_lim=jhr_lim)
    # Generate Dial plot
    dials = DialPlot(
        nrows=1,
        ncols=2,
    )
    dials.plot_scatter(
        np.deg2rad(events.mlt*15-90),
        90-events.mlat,
        events.Erms,
        txt=r"$J_{th}$=%d $mW/m^2$"%jhr_lim,
        size=5,
    )
    dials.plot_scatter(
        np.deg2rad(events.mlt*15-90),
        90-events.mlat,
        events.jhr,
        vlims=[jhr_lim, jhr_lim*4],
        label=r"Joule heating, $mW/m^2$",
        size=5,
    )
    dials.enumerate_figure()
    dials.save(fig_dir + "summary_dial.png")
    dials.close()

    # Generate TS summary plots
    e_events = events.reset_index()
    for i in range(len(e_events)):
        fig_file_name = fig_dir + f"event-{'%02d'%i}.png"
        ts = TimeSeriesAnalysis(e_events.iloc[i])
        sp = StackPlots(ts)
        sp.plot_indexs()
        sp.addParamPlot(pmin=-300, pmax=300)
        sp.add_time_series(ylim=[-1000,1000])
        sp.add_fft()
        sp.save(fig_file_name)
        sp.close()
    return


if __name__ == "__main__":
    files = "tmp/sd.run.14/analysis/*.csv"
    Ndpint_threshold = 420
    I_sig_threshold = 0.6
    bdr_threshold = 3
    jhr_lim = 2
    generate_statistics(
        files, Ndpint_threshold, I_sig_threshold,
        bdr_threshold, jhr_lim, 
    )
    