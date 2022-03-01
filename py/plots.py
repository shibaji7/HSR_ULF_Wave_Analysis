#!/usr/bin/env python

"""plots.py: plots module for data plotting."""

__author__ = "Chakraborty, S."
__copyright__ = ""
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import sys
sys.path.extend(["py/"])

import matplotlib
matplotlib.use("Agg")
matplotlib.style.use(["science", "ieee"])
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, num2date
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator
import matplotlib.patches as mpatches

import pandas as pd
import numpy as np

import utils

class RangeTimeIntervalPlot(object):
    """
    Create plots for velocity, width, power, elevation angle, etc.
    """
    
    def __init__(self, nrang, dates, fig_title="", num_subplots=3):
        self.nrang = nrang
        self.unique_gates = np.linspace(1, nrang, nrang)
        self.dates = dates
        self.num_subplots = num_subplots
        self._num_subplots_created = 0
        self.fig = plt.figure(figsize=(6, 3*num_subplots), dpi=150) # Size for website
        plt.suptitle(fig_title, x=0.075, y=0.99, ha="left", fontweight="bold", fontsize=15)
        self.rc_ax = None
        return
    
    def addParamPlot(self, df, beam, title, p_max=100, p_min=-100, p_step=20, xlabel="Time, UT", zparam="v",
                    label=r"Velocity, $ms^{-1}$", cmap="Spectral"):
        ax = self._add_axis()
        df = df[df.bmnum==beam]
        X, Y, Z = utils.get_gridded_parameters(df, xparam="time", yparam="slist", zparam=zparam, rounding=False)
        bounds = list(range(p_min, p_max+1, p_step))
        cmap = matplotlib.cm.get_cmap(cmap)
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        # cmap.set_bad("w", alpha=0.0)
        # Configure axes
        ax.xaxis.set_major_formatter(DateFormatter(r"$%H^{%M}$"))
        hours = mdates.HourLocator(byhour=range(0, 24, 1))
        ax.xaxis.set_major_locator(hours)
        dtime = (pd.Timestamp(self.dates[-1]).to_pydatetime()-
                 pd.Timestamp(self.dates[0]).to_pydatetime()).total_seconds()/3600.
        if dtime < 4.:
            minutes = mdates.MinuteLocator(byminute=range(0, 60, 10))
            ax.xaxis.set_minor_locator(minutes)
            ax.xaxis.set_minor_formatter(DateFormatter(r"$%H^{%M}$"))
        ax.set_xlabel(xlabel, fontdict={"size":12, "fontweight": "bold"})
        ax.set_xlim([self.dates[0], self.dates[-1]])
        ax.set_ylim([0, self.nrang])
        ax.set_ylabel("Range gate", fontdict={"size":12, "fontweight": "bold"})
        ax.pcolormesh(X, Y, Z.T, lw=0.01, edgecolors="None", cmap=cmap, norm=norm)
        self._add_colorbar(self.fig, ax, bounds, cmap, label=label)
        ax.set_title(title, loc="left", fontdict={"fontweight": "bold"})
        return
    
    def addParamSctr(self, df, beam, title, p_max=100, p_min=-100, p_step=20, xlabel="Time, UT", zparam="v",
                    label=r"Velocity, $ms^{-1}$", cmap="Spectral"):
        ax = self._add_axis()
        df = df[df.bmnum==beam]
        X, Y, Z = utils.get_gridded_parameters(df, xparam="time", yparam="slist", zparam=zparam, rounding=False)
        bounds = list(range(p_min, p_max+1, p_step))
        cmap = matplotlib.cm.get_cmap(cmap)
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        # cmap.set_bad("w", alpha=0.0)
        # Configure axes
        ax.xaxis.set_major_formatter(DateFormatter(r"$%H^{%M}$"))
        hours = mdates.HourLocator(byhour=range(0, 24, 1))
        ax.xaxis.set_major_locator(hours)
        dtime = (pd.Timestamp(self.dates[-1]).to_pydatetime()-
                 pd.Timestamp(self.dates[0]).to_pydatetime()).total_seconds()/3600.
        if dtime < 4.:
            minutes = mdates.MinuteLocator(byminute=range(0, 60, 10))
            ax.xaxis.set_minor_locator(minutes)
            ax.xaxis.set_minor_formatter(DateFormatter(r"$%H^{%M}$"))
        ax.set_xlabel(xlabel, fontdict={"size":12, "fontweight": "bold"})
        ax.set_xlim([self.dates[0], self.dates[-1]])
        ax.set_ylim([0, self.nrang])
        ax.set_ylabel("Range gate", fontdict={"size":12, "fontweight": "bold"})
        ax.scatter(X, Y, s=1, c=Z.T, edgecolors="None", cmap=cmap, norm=norm)
        self._add_colorbar(self.fig, ax, bounds, cmap, label=label)
        ax.set_title(title, loc="left", fontdict={"fontweight": "bold"})
        return
    
    def addGSIS(self, df, beam, title, xlabel="", ylabel="Range gate", zparam="gflg_ribiero"):
        # add new axis
        ax = self._add_axis()
        df = df[df.bmnum==beam]
        X, Y, Z = utils.get_gridded_parameters(df, xparam="time", yparam="slist", zparam=zparam, rounding=False)
        flags = np.array(df[zparam]).astype(int)
        if -1 in flags and 2 in flags:                     # contains noise flag
            cmap = mpl.colors.ListedColormap([(0.0, 0.0, 0.0, 1.0),     # black
                (1.0, 0.0, 0.0, 1.0),     # blue
                (0.0, 0.0, 1.0, 1.0),     # red
                (0.0, 1.0, 0.0, 1.0)])    # green
            bounds = [-1, 0, 1, 2, 3]      # Lower bound inclusive, upper bound non-inclusive
            handles = [mpatches.Patch(color="red", label="IS"), mpatches.Patch(color="blue", label="GS"),
                      mpatches.Patch(color="black", label="US"), mpatches.Patch(color="green", label="SAIS")]
        elif -1 in flags and 2 not in flags:
            cmap = matplotlib.colors.ListedColormap([(0.0, 0.0, 0.0, 1.0),     # black
                                              (1.0, 0.0, 0.0, 1.0),     # blue
                                              (0.0, 0.0, 1.0, 1.0)])    # red
            bounds = [-1, 0, 1, 2]      # Lower bound inclusive, upper bound non-inclusive
            handles = [mpatches.Patch(color="red", label="IS"), mpatches.Patch(color="blue", label="GS"),
                      mpatches.Patch(color="black", label="US")]
        else:
            cmap = matplotlib.colors.ListedColormap([(1.0, 0.0, 0.0, 1.0),  # blue
                                              (0.0, 0.0, 1.0, 1.0)])  # red
            bounds = [0, 1, 2]          # Lower bound inclusive, upper bound non-inclusive
            handles = [mpatches.Patch(color="red", label="IS"), mpatches.Patch(color="blue", label="GS")]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        ax.xaxis.set_major_formatter(DateFormatter(r"$%H^{%M}$"))
        hours = mdates.HourLocator(byhour=range(0, 24, 4))
        ax.xaxis.set_major_locator(hours)
        dtime = (pd.Timestamp(self.dates[-1]).to_pydatetime()-
                 pd.Timestamp(self.dates[0]).to_pydatetime()).total_seconds()/3600.
        if dtime < 4.:
            minutes = mdates.MinuteLocator(byminute=range(0, 60, 10))
            ax.xaxis.set_minor_locator(minutes)
            ax.xaxis.set_minor_formatter(DateFormatter(r"$%H^{%M}$"))
        ax.set_xlabel(xlabel, fontdict={"size":12, "fontweight": "bold"})
        ax.set_xlim([self.dates[0], self.dates[-1]])
        ax.set_ylim([0, self.nrang])
        ax.set_ylabel(ylabel, fontdict={"size":12, "fontweight": "bold"})
        ax.pcolormesh(X, Y, Z.T, lw=0.01, edgecolors="None", cmap=cmap, norm=norm)
        ax.set_title(title,  loc="left", fontdict={"fontweight": "bold"})
        ax.legend(handles=handles, loc=1)
        return ax
    
    def add_range_cell_data(self, df, rc, title="", xlabel="Time, UT", ylabel=r"Velocity, $ms^{-1}$",
                            bounds=True):
        x, y = df[(df.bmnum==rc["bmnum"]) & (df.slist==rc["gate"])].time, df[(df.bmnum==rc["bmnum"]) & 
                                                                             (df.slist==rc["gate"])].v
        ax = self._add_axis() if self.rc_ax is None else self.rc_ax
        ax.xaxis.set_major_formatter(DateFormatter(r"$%H^{%M}$"))
        hours = mdates.HourLocator(byhour=range(0, 24, 1))
        ax.xaxis.set_major_locator(hours)
        dtime = (pd.Timestamp(self.dates[-1]).to_pydatetime()-
                 pd.Timestamp(self.dates[0]).to_pydatetime()).total_seconds()/3600.
        if dtime < 4.:
            minutes = mdates.MinuteLocator(byminute=range(0, 60, 10))
            ax.xaxis.set_minor_locator(minutes)
            ax.xaxis.set_minor_formatter(DateFormatter(r"$%H^{%M}$"))
        ax.set_xlabel(xlabel, fontdict={"size":12, "fontweight": "bold"})
        ax.set_xlim([self.dates[0], self.dates[-1]])
        #ax.set_ylim([0, self.nrang])
        ax.set_ylabel(ylabel, fontdict={"size":12, "fontweight": "bold"})
        ax.set_title(title, loc="left", fontdict={"fontweight": "bold"})
        ax.plot(x, y, color="k", ls="-", lw=0.6)
        if bounds:
            ci = df[(df.bmnum==rc["bmnum"]) & (df.slist==rc["gate"])]["v.sprd"]   
            ax.fill_between(x, (y - ci), (y + ci), color=rc["color"], alpha=0.3, label="Gate=%02d"%rc["gate"])
        self.rc_ax = ax
        return
    
    def _add_axis(self):
        self._num_subplots_created += 1
        ax = self.fig.add_subplot(self.num_subplots, 1, self._num_subplots_created)
        return ax
    
    def _add_colorbar(self, fig, ax, bounds, colormap, label=""):
        """
        Add a colorbar to the right of an axis.
        """
        import matplotlib as mpl
        pos = ax.get_position()
        cpos = [pos.x1 + 0.025, pos.y0 + 0.0125,
                0.015, pos.height * 0.9]                # this list defines (left, bottom, width, height
        cax = fig.add_axes(cpos)
        norm = matplotlib.colors.BoundaryNorm(bounds, colormap.N)
        cb2 = matplotlib.colorbar.ColorbarBase(cax, cmap=colormap,
                                        norm=norm,
                                        ticks=bounds,
                                        spacing="uniform",
                                        orientation="vertical")
        cb2.set_label(label)
        return
    
    def save(self, filepath):
        self.fig.savefig(filepath, bbox_inches="tight")
        return

    def close(self):
        self.fig.clf()
        plt.close()
        return
    
class AnalysisStackPlots(object):
    """
    """
    
    def __init__(self, fig_title="", num_subplots=3):
        self.num_subplots = num_subplots
        self._num_subplots_created = 0
        self.fig = plt.figure(figsize=(6, 3*num_subplots), dpi=150) # Size for website
        plt.suptitle(fig_title, x=0.075, y=0.99, ha="left", fontweight="bold", fontsize=15)
        return
    
    def save(self, filepath):
        self.fig.savefig(filepath, bbox_inches="tight")
        return

    def close(self):
        self.fig.clf()
        plt.close()
        return
    
    def _add_axis(self):
        self._num_subplots_created += 1
        ax = self.fig.add_subplot(self.num_subplots, 1, self._num_subplots_created)
        return ax
    
    def add_TS_axes(self, xtime, yval, title="", xlabel="Time, UT", ylabel="Velo, m/s", 
                    col="r", ls="-", lw=1.0, a=0.7):
        ax = self._add_axis()
        ax.set_xlabel(xlabel, fontdict={"size":12, "fontweight": "bold"})
        ax.set_ylabel(ylabel, fontdict={"size":12, "fontweight": "bold"})
        ax.set_title(title, loc="left", fontdict={"fontweight": "bold"})
        ax.xaxis.set_major_formatter(DateFormatter(r"$%H^{%M}$"))
        hours = mdates.HourLocator(byhour=range(0, 24, 1))
        ax.xaxis.set_major_locator(hours)
        dtime = (pd.Timestamp(xtime.tolist()[-1]).to_pydatetime()-
                 pd.Timestamp(xtime.tolist()[0]).to_pydatetime()).total_seconds()/3600.
        if dtime < 4.:
            minutes = mdates.MinuteLocator(byminute=range(0, 60, 30))
            ax.xaxis.set_minor_locator(minutes)
            ax.xaxis.set_minor_formatter(DateFormatter(r"$%H^{%M}$"))
        ax.plot(xtime, yval, col+"s", ms=0.8, alpha=a)
        ax.set_ylim([-1000, 1000])
        return
    
    def add_FFT_axes(self, freq, amp, title="", xlabel="Freq, Hz", ylabel=r"PSD, $(m/s)^2/Hz$"):
        ax = self._add_axis()
        ax.set_xlabel(xlabel, fontdict={"size":12, "fontweight": "bold"})
        ax.set_ylabel(ylabel, fontdict={"size":12, "fontweight": "bold"})
        ax.set_title(title, loc="left", fontdict={"fontweight": "bold"})
        ax.plot(freq, amp, "ks", ms=0.8)
        ax.semilogx(freq, amp, "b", ls="-", lw=1.0, alpha=0.7)
        return