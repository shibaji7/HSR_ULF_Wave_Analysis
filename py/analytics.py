#!/usr/bin/env python

"""analytics.py: plots module for data post processing and analysis."""

__author__ = "Shi, X."
__copyright__ = ""
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import datetime as dt
import glob

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

plt.style.use(["ieee", "science"])
import matplotlib.dates as mdates
import utils
from get_fit_data import FetchData
from matplotlib.patches import Circle


class Stats(object):
    def __init__(
        self,
        loc,
        parse_dates=["stime", "etime", "bin_time"],
        Ndpint_threshold=420,
        I_sig_threshold=0.6,
        bdr_threshold=3,
    ):
        """
        Parameters:
        -----------
        loc: Location of the files
        parse_dates: Date parameters in csv files needed to be parsed
        """
        self.loc = loc
        self.files = glob.glob(loc)
        self.D = pd.DataFrame()
        for f in self.files:
            self.D = pd.concat([self.D, pd.read_csv(f, parse_dates=parse_dates)])
        self.D["Ndpint"] = self.D.len * self.D.intt
        self.D["jhr"] = self.D.op_ped * self.D.Erms**2 * 1000.0  # mW/m^2
        self.D.Erms = self.D.Erms * 1000.0  # mV
        self.Ndpint_threshold = Ndpint_threshold
        self.I_sig_threshold = I_sig_threshold
        self.bdr_threshold = bdr_threshold
        return

    def get_valid_events(
        self,
        key="jhr",
        Ndpint_threshold=None,
        I_sig_threshold=None,
        bdr_threshold=None,
    ):
        """
        Fetch an event based on Ndpint_threshold and I_sig_trhreshold
        """
        Ndpint_threshold = (
            Ndpint_threshold if Ndpint_threshold else self.Ndpint_threshold
        )
        I_sig_threshold = I_sig_threshold if I_sig_threshold else self.I_sig_threshold
        bdr_threshold = bdr_threshold if bdr_threshold else self.bdr_threshold
        logger.info(
            f"Fetch {key} events with ndp:{Ndpint_threshold} and iSig:{I_sig_threshold} and badPoint:{bdr_threshold}"
        )
        o = self.D.copy()
        o = o[
            (o.Ndpint >= Ndpint_threshold)
            & (o.I_sig > I_sig_threshold)
            & (o.num_bad_data_rsamp <= bdr_threshold)
        ]
        return o, np.array(o[key])

    def get_valid_extreme_events(
        self,
        key="jhr",
        Ndpint_threshold=None,
        I_sig_trhreshold=None,
        jhr_lim=5,  # mW/m^2
    ):
        """
        Fetch an event based on Ndpint_threshold and I_sig_trhreshold
        """
        o, _ = self.get_valid_events()
        logger.info(f"Filter events based on Joule Heating Rate {jhr_lim}")
        o = o[o.jhr >= jhr_lim]
        logger.info(f"Total number of extreme events {len(o)}")
        return o, np.array(o[key])

    def fetch_events_based_on_index(
        self,
        index,
        index_limits,
    ):
        """
        Fetch an event based on geophysical index
        """
        o, _ = self.get_valid_events()
        logger.info(f"Filter events based on {index}")
        if len(index_limits) == 2:
            o = o[(o[index] < index_limits[0]) & (o[index] >= index_limits[1])]
        else:
            o = o[o[index] < index_limits[0]]
        logger.info(f"Total number of events {len(o)}")
        return o

    def fetch_extreme_events_based_on_inedx(
        self,
        index,
        index_limits,
        jhr_lim=5,  # mW/m^2
    ):
        """
        Fetch an event based on geophysical index and jhr
        """
        o = self.fetch_events_based_on_index(index, index_limits)
        logger.info(f"Filter events based on Joule Heating Rate {jhr_lim}")
        o = o[o.jhr >= jhr_lim]
        logger.info(f"Total number of extreme events {len(o)}")
        return o


class DialPlot(object):
    def __init__(self, nrows=1, ncols=1, fig_title=""):
        """
        Create all the Dial plots by number of rows and colums
        """
        self.nrows = nrows
        self.ncols = ncols
        self.num_of_axes = nrows * ncols
        self.num_of_axes_created = 0
        self.dials = []
        self.fig = plt.figure(figsize=(4 * ncols, 4 * nrows), dpi=240)
        plt.suptitle(
            fig_title, x=0.075, y=0.99, ha="left", fontweight="bold", fontsize=15
        )
        return

    def save(self, fname):
        self.fig.savefig(fname, bbox_inches="tight")
        return

    def close(self):
        self.fig.clf()
        plt.close()
        return

    def create_dial(self):
        """
        Create a new dial
        """
        self.num_of_axes_created += 1
        dial = self.fig.add_subplot(
            self.nrows,
            self.ncols,
            self.num_of_axes_created,
            projection="polar",
        )
        dial.set_xticks([0, np.pi / 2, np.pi, np.pi * 3 / 2])
        dial.set_xticklabels([6, 12, 18, 0], fontsize=10)
        dial.set_ylim([0, 30])
        dial.set_yticks([0, 10, 20, 30])
        dial.set_yticklabels([90, 80, 70, 60], fontsize=10)
        self.dials.append(dial)
        return dial

    def plot_scatter(
        self,
        xparam,
        yparam,
        zparam,
        size=10,
        label=r"$E_{rms}$, mV/m",
        vlims=[30, 300],
        cmap=plt.cm.jet,
        add_cbar=True,
        txt=None,
    ):
        """
        Plot data in a dial
        """
        ax = self.create_dial()
        sc = ax.scatter(
            xparam,
            yparam,
            c=zparam,
            alpha=0.7,
            s=size,
            cmap=cmap,
            label=label,
            vmin=vlims[0],
            vmax=vlims[1],
        )
        if add_cbar:
            self.add_cbar(ax, sc, label)
        if txt:
            ax.text(
                0.05,
                1.05,
                txt,
                va="center",
                ha="left",
                transform=ax.transAxes,
                fontdict={"size": 10},
            )
        return ax

    def add_cbar(self, ax, im, label):
        """
        Adding colorbars
        """
        cbar = plt.colorbar(im, ax=ax, shrink=0.5)
        cbar.set_label(label, fontsize=10)
        return cbar

    def enumerate_figure(
        self,
    ):
        """
        Adding text information in the figure
        """
        plt.tight_layout()
        return

    def tag_events_box(
        self,
        axis_id=1,
        xylocs=[],
        radius=1,
        edgecolor="m",
        fill=False,
        lw=0.8,
    ):
        """
        Adding text information in the figure
        """
        ax = self.dials[axis_id - 1]
        for xy in xylocs:
            p = Circle(
                xy,
                radius,
                transform=ax.transData._b,
                edgecolor=edgecolor,
                facecolor=None,
                fill=fill,
                lw=lw,
            )
            ax.add_patch(p)
        return

    def identify_events(
        self,
        xparam,
        yparam,
        size=10,
        axis_id=2,
    ):
        """
        Ovearlay gray patch for a given event
        """
        ax = self.dials[axis_id - 1]
        ax.scatter(
            xparam,
            yparam,
            c="k",
            alpha=0.9,
            s=size,
        )
        return


class TimeSeriesAnalysis(object):
    def __init__(
        self,
        event,
        hours=2,
        load_parse_rbsp=True,
        limit_omni_by_event_time=False,
    ):
        """
        Conduct TS analysis of extreme events
        """
        self.event = event
        self.hours = hours
        self.stime = (self.event.stime - dt.timedelta(hours=self.hours)).replace(
            minute=0, second=0
        )
        self.etime = (self.event.etime + dt.timedelta(hours=self.hours + 1)).replace(
            minute=0, second=0
        )
        self.limit_omni_by_event_time = limit_omni_by_event_time
        self.load_omni()
        if load_parse_rbsp:
            self.load_rbsp_raw_dataset()
            self.load_parsed_rbsp_dataset()
        return

    def load_omni(
        self, loc="/home/shibaji/OneDrive/SuperDARN-Data-Share/Shi/HSR/data/omni/*.csv"
    ):
        """
        Load OMNI dataset
        """
        files = glob.glob(loc)
        files.sort()
        self.omni = pd.DataFrame()
        name_str = [self.stime.strftime("%Y%m"), self.etime.strftime("%Y%m")]
        for f in files:
            if f.split("/")[-1].replace(".csv", "") in name_str:
                logger.info(f"Load {f}")
                self.omni = pd.concat([self.omni, pd.read_csv(f, parse_dates=["DATE"])])

        if self.limit_omni_by_event_time:
            self.omni = self.omni[
                (self.omni.DATE >= self.event.stime)
                & (self.omni.DATE <= self.event.etime)
            ]
        else:
            self.omni = self.omni[
                (self.omni.DATE >= self.stime) & (self.omni.DATE <= self.etime)
            ]
        bz = np.array(self.omni["Bz_GSM"])
        bz[bz > 100] = np.nan
        self.omni.Bz_GSM = bz
        return

    def load_rbsp_raw_dataset(self):
        """
        Load RBSP dataset
        """
        self.fd = FetchData(self.event.rad, [self.stime, self.etime], ftype="fitacf3")
        _, scans, exists = self.fd.fetch_data(by="scan")
        self.sdframe = self.fd.scans_to_pandas(scans)
        return

    def load_parsed_rbsp_dataset(
        self, loc="tmp/sd.run.{run_id}/{date}/{rad}*{kind}.csv", run_id=14
    ):
        """
        Load RBSP data
        """
        self.rbsp_data = {}
        kinds = ["rsamp", "fft", "fill", "dtrnd"]
        for kind in kinds:
            logger.info(f"Loading {kind}-data")
            files = glob.glob(
                loc.format(
                    date=self.event.stime.strftime("%Y-%m-%d"),
                    rad=self.event.rad,
                    kind=kind,
                    run_id=run_id,
                )
            )
            files.sort()
            dat = pd.DataFrame()
            for f in files:
                dat = pd.concat([dat, pd.read_csv(f)])
            self.rbsp_data[kind] = dat
        return


class StackPlots(object):
    def __init__(
        self,
        ts,
    ):
        self.ts = ts
        self.fig = plt.figure(figsize=(8, 3 * 5), dpi=240)
        self.num_of_axes_created = 0
        return

    def date_string(self, label_style="web"):
        # Set the date and time formats
        dfmt = "%d %b %Y" if label_style == "web" else "%d %b %Y,"
        tfmt = "%H:%M"
        stime = self.ts.event.bin_time
        date_str = "{:{dd} {tt}} UT".format(stime, dd=dfmt, tt=tfmt)
        return date_str

    def create_axes(self):
        """
        Create a new dial
        """
        self.num_of_axes_created += 1
        ax = self.fig.add_subplot(5, 1, self.num_of_axes_created)
        return ax

    def set_time_axis(self, ax):
        ax.xaxis.set_major_formatter(mdates.DateFormatter(r"%H^{%M}"))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter(r"%H^{%M}"))
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 1)))
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=range(0, 60, 15)))
        return ax

    def set_labels(self, ax, xlabel, ylabel, title, ycol="k"):
        ax.set_xlabel(xlabel, fontdict={"size": 12, "fontweight": "bold"})
        ax.set_ylabel(
            ylabel, fontdict={"size": 12, "fontweight": "bold", "color": ycol}
        )
        ax.set_title(title, loc="left", fontdict={"fontweight": "bold"})
        return ax

    def plot_indexs(self):
        ax = self.set_labels(
            self.set_time_axis(self.create_axes()),
            "",
            "IMF, nT",
            r"{%s} / $E_{rms}=%.1f$  mV / $\sigma_P=%.1f$ S / $JH_r=%.1f$ $mW/m^2$"
            % (
                self.date_string(),
                self.ts.event.Erms,
                self.ts.event.op_ped,
                self.ts.event.jhr,
            ),
        )
        ax.plot(
            self.ts.omni.DATE,
            self.ts.omni.Bz_GSM,
            "bo",
            ms=0.8,
            ls="None",
            label=r"$B_z$",
        )
        ax.plot(
            self.ts.omni.DATE,
            self.ts.omni.By_GSM,
            "ro",
            ms=0.8,
            ls="None",
            label=r"$B_y$",
        )
        ax.plot(
            self.ts.omni.DATE, self.ts.omni.Bx, "ko", ms=0.8, ls="None", label=r"$B_x$"
        )
        ax.legend(loc=2)
        ax.set_ylim(-20, 20)
        ax.set_xlim(self.ts.stime, self.ts.etime)
        ax.axvline(self.ts.event.stime, color="gray", ls="-", lw=0.4)
        ax.axvline(self.ts.event.etime, color="gray", ls="-", lw=0.4)

        ax = self.set_labels(
            self.set_time_axis(self.create_axes()),
            "",
            "AE, nT",
            "",
            ycol="b",
        )
        ax.plot(self.ts.omni.DATE, self.ts.omni.AE, "b-", lw=0.8, label=r"AE")
        ax.set_ylim(0, 1500)
        ax.set_xlim(self.ts.stime, self.ts.etime)
        ax.axvline(self.ts.event.stime, color="gray", ls="-", lw=0.4)
        ax.axvline(self.ts.event.etime, color="gray", ls="-", lw=0.4)
        ax = self.set_labels(self.set_time_axis(ax.twinx()), "", "SYM-H, nT", "")
        ax.plot(self.ts.omni.DATE, self.ts.omni["SYM-H"], "k-", lw=0.8, label=r"AE")
        ax.set_ylim(-150, 20)
        ax.set_xlim(self.ts.stime, self.ts.etime)
        return

    def addParamPlot(
        self,
        pmax=500,
        pmin=-500,
        xlabel="Time, UT",
        zparam="v",
        label=r"Velocity, $ms^{-1}$",
        cmap="Spectral",
    ):
        ax = self.set_labels(
            self.set_time_axis(self.create_axes()),
            "",
            "Range Gate, km",
            "%s / %02d" % (self.ts.event.rad.upper(), self.ts.event.beam),
        )
        df = self.ts.sdframe[self.ts.sdframe.bmnum == self.ts.event.beam]
        X, Y, Z = utils.get_gridded_parameters(
            df, xparam="time", yparam="slist", zparam=zparam, rounding=False
        )
        cmap = mpl.cm.get_cmap(cmap)
        ax.set_ylim([0, 100])
        im = ax.pcolormesh(
            X, Y, Z.T, lw=0.01, edgecolors="None", cmap=cmap, vmax=pmax, vmin=pmin
        )
        ax.set_xlim(self.ts.stime, self.ts.etime)
        ax.axvline(self.ts.event.stime, color="gray", ls="-", lw=0.4)
        ax.axvline(self.ts.event.etime, color="gray", ls="-", lw=0.4)
        ax.axhline(self.ts.event.gate, color="m", ls="--", lw=0.4)
        self._add_colorbar(ax, im, label=label)
        return

    def _add_colorbar(self, ax, im, label=""):
        """
        Add a colorbar to the right of an axis.
        """

        pos = ax.get_position()
        cpos = [
            pos.x1 + 0.025,
            pos.y0 + 0.0125,
            0.015,
            pos.height * 0.8,
        ]  # this list defines (left, bottom, width, height
        cax = self.fig.add_axes(cpos)
        cb = self.fig.colorbar(
            im,
            cax=cax,
        )
        cb.set_label(label)
        return

    def add_time_series(self, kind="rsamp", ylim=[-1000, 1000]):
        ax = self.set_labels(
            self.set_time_axis(self.create_axes()),
            "Time, UT",
            r"Velocity, $ms^{-1}$",
            "%s / %02d / %02d / %s"
            % (
                self.ts.event.rad.upper(),
                self.ts.event.beam,
                self.ts.event.gate,
                kind,
            ),
        )
        data = self.ts.rbsp_data[kind]
        data = data[
            (data.slist == self.ts.event.gate) & (data.bmnum == self.ts.event.beam)
        ]
        data.time = pd.to_datetime(data.time)
        ax.plot(data.time, data.v, "ko", ls="None", ms=0.5)
        ax.plot(data.time, data.v, "k--", lw=0.5)
        ax.set_xlim(self.ts.stime, self.ts.etime)
        ax.axvline(self.ts.event.stime, color="gray", ls="-", lw=0.4)
        ax.axvline(self.ts.event.etime, color="gray", ls="-", lw=0.4)
        ax.set_ylim(ylim)
        return

    def add_fft(self):
        ax = self.set_labels(self.create_axes(), "Frequency, mHz", r"Power", "")
        data = self.ts.rbsp_data["fft"]
        data.time_window_start = pd.to_datetime(data.time_window_start)
        data.time_window_end = pd.to_datetime(data.time_window_end)
        data = data[
            (data.slist == self.ts.event.gate)
            & (data.bmnum == self.ts.event.beam)
            & (data.time_window_start == self.ts.event.stime)
            # & (data.time_window_end <= self.ts.etime)
        ]
        ax.plot(data.frq * 1e3, data.amp, "k--", lw=0.8)
        ax.set_xlim(0, 30)
        return

    def save(self, filepath):
        self.fig.savefig(filepath, bbox_inches="tight")
        return

    def close(self):
        self.fig.clf()
        plt.close()
        return
