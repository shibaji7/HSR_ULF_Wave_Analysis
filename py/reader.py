#!/usr/bin/env python

"""reader.py: read module for data post-processing and analsysis."""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2022, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import os
import sys
sys.path.extend(["py/"])
import numpy as np
import pandas as pd
import json

from loguru import logger
import utils
from plots import AnalysisStackPlots

class Folder(object):
    """
    Class to hold all the processed files
    from one RBSP log entry
    """
    
    def __init__(self, dirs, rad, stime, etime, params=["raw", "dtrnd", "rsamp", "fft"]):
        self.rad = rad
        self.stime = stime
        self.etime = etime
        self.dirs = dirs
        self.params = params
        self.frames = {}
        self.filters = {}
        stime, etime = self.stime.strftime("%H%M"), self.etime.strftime("%H%M")
        for p in params:
            self.dirs[p] = self.dirs[p].format(rad=rad, stime=stime, etime=etime)
            if os.path.exists(self.dirs[p]) and os.stat(self.dirs[p]).st_size > 1: 
                self.frames[p] = pd.read_csv(self.dirs[p]) if p == "fft" else\
                            pd.read_csv(self.dirs[p], parse_dates=["time"])
                if p=="fft" or p=="rsamp": self.fetch_filter_options(p)
            else: self.frames[p] = pd.DataFrame()
        return
    
    def get_data(self, p, T=None, beams=None, gates=None, Tx=None):
        """
        Fetch data form a specific frame with,
        T: Time window
        beams: Beams
        Tx: FFT run window
        """
        o = self.frames[p].copy()
        if len(o) > 0:
            if (beams is not None) and (len(beams) > 0): o = o[o.bmnum.isin(beams)]
            if (gates is not None) and (len(gates) > 0): o = o[o.slist.isin(gates)]
            if (p != "fft") and (T is not None) and (len(T) == 2): o = o[(o.time>=T[0]) & (o.time<T[1])]
            if (Tx is not None) and (len(Tx) > 0): o = o[o.Tx.isin(Tx)]
        return o
    
    def fetch_filter_options(self, p):
        """
        Find the options of filter selections [valid for FFT and RSAMP]
        """
        self.filters[p] = {}
        o = self.frames[p].copy()
        if len(o) > 0:
            txs = o.Tx.unique()
            for t in txs:
                self.filters[p][t] = {}
                if p=="rsamp":
                    times = o[o.Tx==t].time
                    tmax, tmin = max(times), min(times)
                else: tmax, tmin = None, None
                beams = o[o.Tx==t].bmnum.unique()
                gates = o[o.Tx==t].slist.unique()
                logger.warning(f"Parameter({p}), U-Tx({t}), W({(tmin, tmax)}), U-Beams({beams}), U-Gates({gates})")
                self.filters[p][t]["tmax"] = tmax
                self.filters[p][t]["tmin"] = tmin
                self.filters[p][t]["gates"] = gates
                self.filters[p][t]["beams"] = beams
        return

class Reader(object):
    """
    This class is dedicated to read one / multple 
    outputs generated by the filtering stage.
    """
    
    def __init__(self, _filestr="config/logs/*.txt", select=[10]):
        """
        Params
        ------
        _filestr - Regular expression to search RBSP log files
        select - Select ith entries in the file
        """
        self._filestr = _filestr
        self.rbsp_logs = pd.DataFrame.from_records(utils.read_rbsp_logs(_filestr))
        self.select = select
        logger.info(f"RBSP mode log entry -to- dataframe")
        return
    
    def check_entries(self, T=None, rad=None):
        """
        Check entries by 
        T: time winodw
        rad: radar name
        """
        o = self.rbsp_logs.copy()
        if rad is not None: o = o[o.rad==rad]
        if (T is not None) and (len(T)==2): o = o[(o.stime>=T[0]) & (o.etime<=T[1])]
        logger.info(f"RBSP mode log entries {len(o)}")
        print(o)
        return o
    
    def load_params(self, dates):
        """
        Load parameters from 
        """
        with open("config/params.json") as f: o = json.load(f)
        for k in o["filter"]:
            setattr(self, k, o["filter"][k])
        setattr(self, "run_id", o["run_id"])
        setattr(self, "save", o["save"])
        setattr(self, "files", o["files"])
        self.dirs = {}
        base = self.files["base"].format(run_id=self.run_id, date=dates[0].strftime("%Y-%m-%d"))
        for p in ["raw", "dtrnd", "rsamp", "fft"]:
            self.dirs[p] = base + self.files["csv"]%(p) 
        self.dirs["log"] = base + self.files["log"]%("log") 
        self.dirs["rti_plot"] = base + self.files["rti_plot"]
        return
    
    def parse_files(self, select=None):
        """
        Parse all kinds of files for analysis
        """
        self.file_entries = {}
        select = select if select else self.select
        for i in select:
            row = self.rbsp_logs.iloc[i]
            logger.info(f"Load entry #{i}, R({row.rad}):{row.stime}-{row.etime}")
            self.load_params([row.stime, row.etime])
            self.file_entries[i] = Folder(self.dirs, row.rad, row.stime, row.etime)
        return
    
    def get_stackplot_fname(self, select, Tx=0):
        """
        Create plot filename
        """
        row = self.rbsp_logs.iloc[select]
        base = self.files["base"].format(run_id=self.run_id, date=row.stime.strftime("%Y-%m-%d"))
        fname = base + "{rad}_{stime}_{etime}_{Tx}.png".format(rad=row.rad, stime=row.stime,
                                                               etime=row.etime, Tx=Tx)
        return fname
    
    def generate_stacks(self, frames, fname, fig_title="FFT Analysis"):
        """
        Generate a stackplots of dataset
        """
        asp = AnalysisStackPlots(fig_title=fig_title, num_subplots=2)
        for f in frames:
            p, o = f["p"], f["df"]
            if p == "rsamp": asp.add_TS_axes(o.time, o.v, f["title"])
            if p == "fft": asp.add_FFT_axes(o.frq, o.amp, f["title"])
        logger.info(f"Save to-{fname}")
        asp.fig.subplots_adjust(wspace=0.5,hspace=0.5)
        asp.save(fname)
        asp.close()
        return
    
if __name__ == "__main__":
    # Sample datasets
    select_row = 10
    # Create a reader object that reads all the RBSP mode entries
    r = Reader()
    # Check for entries by radar / datetime interval
    r.check_entries(rad="cly")
    # Read all the files with selected index by passing selected list index
    r.parse_files(select=[select_row])
    # Fetch FFT data by beam, gate, and Tx count
    of = r.file_entries[select_row].get_data(p="fft", beams=[7], gates=[30], Tx=[4])
    print(of.head())
    # Fetch resample data by beam, time interval
    ox = r.file_entries[select_row].get_data(p="rsamp", beams=[7], gates=[30], Tx=[4])
    print(ox.head())
    # Stack plots
    frames = [
        {"p": "rsamp", "df": ox, "title": "Rad: CLY, Beam 7, Gate 33, Tx 2, RSamp"},
        {"p": "fft", "df": of, "title": "Rad: CLY, Beam 7, Gate 33, Tx 2, FFT"}, 
    ]
    r.generate_stacks(frames, r.get_stackplot_fname(select_row))
    # ptint selection criteria
    filt_dict = r.file_entries[select_row].filters
    print(filt_dict)