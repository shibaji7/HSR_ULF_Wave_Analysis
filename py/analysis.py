#!/usr/bin/env python

"""analysis.py: analysis module for data processing and analsysis."""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2021, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import sys
sys.path.extend(["py/"])

from loguru import logger
import datetime as dt
import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial

from plots import RangeTimeIntervalPlot as RTI
from get_fit_data import FetchData
import utils

class Filter(object):
    """
    For each 1-hour interval, select only ionospheric backscatter
    and reject ground scatter by requiring backscatter to satisfy
    one of the following conditions: 
        a) Doppler velocity|VLOS| >= 50 m/s; 
        b) Spectral width W >= 50 m/s; 
        c) Backscatter is flagged as ionospheric scatter by 
            traditional method.
        d) Power is greater than 3 dB
        e) Errors in VLOS or W < 100 m/s
        f) Remove near-range data (slant range <= 765 km)
        g) If the number of data points in 1-hour interval is 
            less than 134 (2/3 of the total number assuming an 
            18-s sampling rate)or the largest data gap in the 
            1-hour interval is greater than 6 minutes 
            (10% of the 1-hour time interval), this 1-hour 
            interval will be discarded.
    """
    
    def __init__(self, rad, dates, beams, filters=["a", "b", "c", "d", "e", "f"], hour_win=1.):
        """
        Parameters
        ----------
        rad - 3 char radar code
        dates - [sdate, edate]
        beams - RBSP beam numbers
        hour_win - Filtering hour window
        filters - combinations of three filtering criteria
        """
        self.rad = rad
        self.dates = dates
        self.beams = beams
        self.hour_win = hour_win
        self.filters = filters
        self._fetch()
        self._filter()
        return
    
    def _fetch(self):
        """
        Fetch radar data from repository
        """
        logger.info(f" Read radar file {self.rad} for {[d.strftime('%Y.%m.%dT%H.%M') for d in self.dates]}")
        self.fd = FetchData(self.rad, self.dates)
        beams, _ = self.fd.fetch_data(by="beams")
        self.frame = self.fd.convert_to_pandas(beams)
        self.frame["srange"] = self.frame.frang + (self.frame.slist*self.frame.rsep)
        return
    
    def _filter(self):
        """
        Filter data based on the beams 
        and given filter criteria
        """
        hour_length = np.rint((self.dates[1]-self.dates[0]).total_seconds()/3600.)
        hours = [(self.dates[0]+dt.timedelta(hours=h),
                  self.dates[0]+dt.timedelta(hours=h+1)) for h in range(int(hour_length)-1)]
        logger.info(f" Toral hour window length {hour_length}")
        logger.info(f" RBSP beams {'-'.join([str(b) for b in self.beams])}")
        logger.info(f" RBSP total data {len(self.frame)}")
        self.fil_frame = pd.DataFrame()
        for hw in hours:
            fil_frame = self.frame.copy()
            fil_frame = fil_frame[fil_frame.bmnum.isin(self.beams)]
            fil_frame = fil_frame[(fil_frame.time>=hw[0]) & (fil_frame.time<hw[1])]
            for fc in self.filters:
                if fc == "a": fil_frame = fil_frame[np.abs(fil_frame.v)>=30.]
                if fc == "b": fil_frame = fil_frame[fil_frame.w_l>=30.]
                if fc == "c": fil_frame = fil_frame[fil_frame.gflg==0]
                if fc == "d": fil_frame = fil_frame[fil_frame.p_l>3.]
                if fc == "e": fil_frame = fil_frame[(fil_frame.v_e<=100.) & (fil_frame.w_l_e<=100.)]
                if fc == "f": fil_frame = fil_frame[fil_frame.srange>765.]
            max_tdiff = np.rint(np.nanmax([t.total_seconds()/60. for t in 
                                           np.diff([hw[0]] + fil_frame.time.tolist() + [hw[1]])]))
            logger.info(f"DLen of {hw[0].strftime('%Y.%m.%dT%H.%M')}-{hw[1].strftime('%Y.%m.%dT%H.%M')} -hour- {len(fil_frame)} & max(td)-{max_tdiff}")
            self.fil_frame = pd.concat([self.fil_frame, fil_frame])
        logger.info(f" RBSP total data after filter {len(self.fil_frame)}")
        return
    
    def _save(self):
        """
        Save data into files for latter accessing.
        """
        time_str = self.dates[0].strftime("%Y.%m.%dT%H:%M") + "-" + self.dates[1].strftime("%H:%M") + " UT"
        mdates = np.unique(self.frame.time)
        rti = RTI(100, mdates, num_subplots=2)
        rti.addParamPlot(self.frame, self.beams[0], xlabel="",
                         title="Date: %s, Rad: %s[Bm: %02d]"%(time_str, self.rad.upper(), self.beams[0]))
        rti.addParamSctr(self.fil_frame, self.beams[0], "Filters: %s"%"-".join(self.filters))
        rti.save("tmp/out.png")
        return
    
class DataFetcherFilter(object):
    """
    For each RBSP log entry in the log files
    this class forks a process to fetch fitacf
    data from the SD repository and filter it 
    based on the given filtering condition.
    """
    
    def __init__(self, _filestr="config/logs/*.txt", cores=24, filters=["a", "d", "e", "f"], run_first=None):
        """
        Params
        ------
        _filestr - Regular expression to search files
        cores - Mutiprocessing cores
        filters - Combinations of three filtering criteria
        run_first - Firt N modes to run
        """
        self._filestr = _filestr
        self.cores = cores
        self.run_first = run_first
        self.filters = filters
        self.rbsp_logs = utils.read_rbsp_logs(_filestr)
        self._run()
        return
    
    def _proc(self, o, filters=["a", "b", "c"]):
        """
        Process method to invoke filter
        """
        rad, dates, beams = o["rad"], [o["stime"], o["etime"]], o["beams"]
        logger.info(f"Filtering radar {rad} for {[d.strftime('%Y.%m.%dT%H.%M') for d in dates]}")
        f = Filter(rad, dates, beams, filters)
        f._save()
        return f
    
    def _run(self):
        """
        Run parallel threads to access and filter data
        """
        if self.run_first: logger.info(f"Start parallel procs for first {self.run_first} entries")
        else: logger.info(f"Start parallel procs for first {len(self.rbsp_logs)} entries")
        self.flist = []
        rlist = self.rbsp_logs if self.run_first is None else self.rbsp_logs[:self.run_first]
        p0 = mp.Pool(self.cores)
        partial_filter = partial(self._proc, filters=self.filters)
        for f in p0.map(partial_filter, rlist):
            self.flist.append(f)
        return

if __name__ == "__main__":
    "__main__ function"
    DataFetcherFilter(run_first=1)
    pass