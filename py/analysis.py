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
from sklearn.cluster import DBSCAN
import multiprocessing as mp
from functools import partial

from plots import RangeTimeIntervalPlot as RTI
from get_fit_data import FetchData
import utils


class DBScan(object):
    """ Class to cluster 2D or 3D data """
    
    def __init__(self, frame, params={"eps_g":1, "eps_s":1, "min_samples":10, "eps":1}):
        self.frame = frame
        for k in params.keys():
            setattr(self, k, params[k])
        self.run_2D_cluster_bybeam()
        return
    
    def run_2D_cluster_bybeam(self):
        o = pd.DataFrame()
        for b in np.unique(self.frame.bmnum):
            _o = self.frame[self.frame.bmnum==b]
            _o.slist, _o.scnum, eps, min_samples = _o.slist/self.eps_g, _o.scnum/self.eps_s, self.eps, self.min_samples
            ds = DBSCAN(eps=self.eps, min_samples=min_samples).fit(_o[["slist", "scnum"]].values)
            _o["cluster_tag"] = ds.labels_
            _o = utils._run_riberio_threshold_on_rad(_o)
            o = pd.concat([o, _o])
        self.frame = o.sort_values("time").copy()
        return

class Filter(object):
    """
    For each 1-hour interval, select only ionospheric backscatter
    and reject ground scatter by requiring backscatter to satisfy
    one of the following conditions: 
        a) Doppler velocity|VLOS| >= 50 m/s or Spectral width W >= 50 m/s or 
            Backscatter is flagged as ionospheric scatter by traditional method.
        b) Power is greater than 3 dB and Errors in VLOS & W < 100 m/s
        c) Remove near-range data (slant range <= 765 km)
        d) If the number of data points in 1-hour interval is 
            less than 134 (2/3 of the total number assuming an 
            18-s sampling rate)or the largest data gap in the 
            1-hour interval is greater than 6 minutes 
            (10% of the 1-hour time interval), this 1-hour 
            interval will be discarded.
    """
    
    def __init__(self, rad, dates, beams, filters=["a", "b", "c", "d"], hour_win=1., 
                 gflg_key="gflg", w_mins=10., param="v"):
        """
        Parameters
        ----------
        rad - 3 char radar code
        dates - [sdate, edate]
        beams - RBSP beam numbers
        hour_win - Filtering hour window
        gflg_key - G-Flag key to access
        filters - combinations of three filtering criteria
        w_mins - Minute window
        param - parameter to be detrend
        """
        self.rad = rad
        self.dates = dates
        self.beams = beams
        self.hour_win = hour_win
        self.gflg_key = gflg_key
        self.filters = filters
        self.w_mins = w_mins
        self.param = param
        self._fetch()
        self._filter()
        return
    
    def _fetch(self):
        """
        Fetch radar data from repository
        """
        dates = [
                    self.dates[0]-dt.timedelta(minutes=self.w_mins/2),
                    self.dates[1]+dt.timedelta(minutes=self.w_mins/2)
                ] if self.w_mins is not None else self.dates
        logger.info(f" Read radar file {self.rad} for {[d.strftime('%Y.%m.%dT%H.%M') for d in self.dates]}")
        self.fd = FetchData(self.rad, dates)
        _, scans = self.fd.fetch_data(by="scan")
        self.frame = self.fd.scans_to_pandas(scans)
        self.frame["srange"] = self.frame.frang + (self.frame.slist*self.frame.rsep)
        if "ribiero" in self.gflg_key: 
            self.db = DBScan(self.frame.copy())
            self.frame = self.db.frame.copy()
        return
    
    def _filter(self):
        """
        Filter data based on the beams 
        and given filter criteria
        """
        time_length = np.rint((self.dates[1]-self.dates[0]).total_seconds()/(self.hour_win*3600.))
        time_windows = [(self.dates[0]+dt.timedelta(hours=h*self.hour_win),
                  self.dates[0]+dt.timedelta(hours=(h+1)*self.hour_win)) for h in range(int(time_length))]
        logger.info(f" Toral time window length {time_length}")
        logger.info(f" RBSP beams {'-'.join([str(b) for b in self.beams])}")
        logger.info(f" RBSP total data {len(self.frame)}")
        self.fil_frame = pd.DataFrame()
        for tw in time_windows:
            add_frame = True
            fil_frame = self.frame.copy()
            fil_frame = fil_frame[fil_frame.bmnum.isin(self.beams)]
            fil_frame = fil_frame[(fil_frame.time>=tw[0]) & (fil_frame.time<tw[1])]
            if "a" in self.filters: 
                if -1 in fil_frame[self.gflg_key].tolist(): 
                    fil_frame = fil_frame[((np.abs(fil_frame.v)>=50.) | (fil_frame.w_l>=50.)) & (fil_frame[self.gflg_key]==0)]
                else: fil_frame = fil_frame[fil_frame[self.gflg_key]==0]
            if "b" in self.filters: fil_frame = fil_frame[(fil_frame.p_l>3.) & (fil_frame.v_e<=100.) & (fil_frame.w_l_e<=100.)]
            if "c" in self.filters: fil_frame = fil_frame[fil_frame.srange>765.]
            if "d" in self.filters:
                max_tdiff = np.rint(np.nanmax([t.total_seconds()/60. for t in 
                                               np.diff([tw[0]] + fil_frame.time.tolist() + [tw[1]])]))
                logger.info(f"DLen of {tw[0].strftime('%Y.%m.%dT%H.%M')}-{tw[1].strftime('%Y.%m.%dT%H.%M')} -hour- {len(fil_frame)} & max(td)-{max_tdiff}")
                if (len(fil_frame) < 134) or (max_tdiff >= 10.): add_frame = False
            if add_frame: self.fil_frame = pd.concat([self.fil_frame, fil_frame])
        logger.info(f" RBSP total data after filter {len(self.fil_frame)}")
        # Detrending
        logger.info(f" Started detreanding data points, {len(self.frame)}")
        self.d_frame = self.frame.apply(self.__trnd_support__, axis=1)
        logger.info(f" Done detreanding data points, {len(self.d_frame)}")
        return
    
    def __trnd_support__(self, row):
        """
        Detrend the velocity data - Support Function.
        """
        pval = np.nan
        b, g, t, p_raw = row["bmnum"], row["slist"], row["time"], row[self.param]
        if (t >= self.dates[0]) & (t <= self.dates[-1]):
            t_start, t_end = t - dt.timedelta(minutes=self.w_mins/2), t + dt.timedelta(minutes=self.w_mins/2)
            x = self.frame[(self.frame.bmnum==b) & (self.frame.slist==g) & 
                           (self.frame.time>=t_start) & (self.frame.time>=t_end)]
            pval = p_raw - np.nanmedian(x[self.param])
        row[self.param] = pval
        return row
    
    def _save(self, rclist=[]):
        """
        Save data into files for latter accessing.
        """
        time_str = self.dates[0].strftime("%Y.%m.%dT%H:%M") + "-" + self.dates[1].strftime("%H:%M") + " UT"
        nplots = 3
        nplots = nplots+1 if "ribiero" in self.gflg_key else nplots
        nplots = nplots+1 if len(rclist)>0 else nplots
        rti = RTI(100, self.dates, num_subplots=nplots)
        rti.addParamPlot(self.frame, self.beams[0], xlabel="",
                         title="Date: %s, Rad: %s[Bm: %02d]"%(time_str, self.rad.upper(), self.beams[0]))
        rti.addParamPlot(self.d_frame, self.beams[0], title=r"Detrend $[T_w=%d minutes]$"%self.w_mins, xlabel="")
        for rc in rclist:
            rti.add_range_cell_data(self.d_frame, rc, title=r"Detrend $[T_w=%d minutes]$"%self.w_mins, xlabel="")
            rti.rc_ax.legend(loc=1)
        rti.addParamSctr(self.fil_frame, self.beams[0], "Filters: %s"%"-".join(self.filters))
        if "ribiero" in self.gflg_key: rti.addGSIS(self.db.frame, self.beams[0], "")
        rti.save("tmp/out.png")
        return
    
    @staticmethod
    def filter_data_by_detrending(rad, dates, beams, filters=["a", "b", "c", "d"], hour_win=1., 
                                  gflg_key="gflg", w_mins=10., param="v", rclist=[]):
        """
        Parameters
        ----------
        rad - 3 char radar code
        dates - [sdate, edate]
        beams - RBSP beam numbers
        hour_win - Filtering hour window
        gflg_key - G-Flag key to access
        filters - combinations of three filtering criteria
        w_mins - Minute window
        param - parameter to be detrend
        rclist - list of range cell plots
        """
        f = Filter(rad, dates, beams, filters, hour_win, gflg_key, w_mins, param)
        f._save(rclist)
        return f
    
class DataFetcherFilter(object):
    """
    For each RBSP log entry in the log files
    this class forks a process to fetch fitacf
    data from the SD repository and filter it 
    based on the given filtering condition.
    """
    
    def __init__(self, _filestr="config/logs/*.txt", cores=24, filters=["a", "b", "c", "d"], run_first=None, 
                 gflg_key=None, param="v", w_mins=10.):
        """
        Params
        ------
        _filestr - Regular expression to search files
        cores - Mutiprocessing cores
        filters - Combinations of three filtering criteria
        gflg_key - G-Flag key to access
        w_mins - Minute window
        param - parameter to be detrend
        run_first - Firt N modes to run
        """
        self._filestr = _filestr
        self.cores = cores
        self.run_first = run_first
        self.param = param
        self.w_mins = w_mins
        self.gflg_key = "gflg" if gflg_key is None else gflg_key
        self.filters = filters
        self.rbsp_logs = utils.read_rbsp_logs(_filestr)
        self._run()
        return
    
    def _proc(self, o):
        """
        Process method to invoke filter
        """
        rad, dates, beams = o["rad"], [o["stime"], o["etime"]], o["beams"]
        logger.info(f"Filtering radar {rad} for {[d.strftime('%Y.%m.%dT%H.%M') for d in dates]}")
        f = Filter(rad, dates, beams, self.filters, gflg_key=self.gflg_key, w_mins=self.w_mins, param=self.param)
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
        partial_filter = partial(self._proc)
        for f in p0.map(partial_filter, rlist):
            self.flist.append(f)
        return

if __name__ == "__main__":
    "__main__ function"
    #DataFetcherFilter(filters=["a"], run_first=1, gflg_key="gflg_ribiero")
    Filter.filter_data_by_detrending("pgr", [dt.datetime(2016,1,25,1), dt.datetime(2016,1,25,1,30)], beams=[12],
                                    hour_win=0.5, rclist=[{"bmnum":12, "gate":13, "color":"r"}, 
                                                          {"bmnum":12, "gate":15, "color":"b"}])
    pass