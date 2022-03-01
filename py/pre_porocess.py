#!/usr/bin/env python

"""pre_process.py: analysis module for data processing and analsysis."""

__author__ = "Chakraborty, S."
__copyright__ = ""
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import os
import sys
sys.path.extend(["py/"])

from loguru import logger
import datetime as dt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import multiprocessing as mp
from functools import partial
import json
from scipy.interpolate import interp1d
from scipy.stats import t as T
import time
import shutil
from scipy.fft import rfft, rfftfreq
from scipy.signal import get_window

import aacgmv2
import pydarn

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
    
    def __init__(self, rad, dates, beams, filters=["a", "b", "c", "d"], hour_win=None, 
                 gflg_key=None, w_mins=None, param=None, ts=None, min_no_echoes=None, 
                 min_pct_echoes=None, max_tdiff=None):
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
        ts - Sampling time interval for interpolation
        min_no_echoes - Number of echoes per hour window per cell
        min_pct_echoes - Minimum precentage of echoes per hour window per cell
        """
        self.proc_start_time = time.time()
        self.load_params(dates)
        self.rad = rad
        self.dates = dates
        self.beams = beams
        self.filters = filters
        if hour_win is not None: self.hour_win = hour_win
        if gflg_key is not None: self.gflg_key = gflg_key
        if w_mins is not None: self.w_mins = w_mins
        if param is not None: self.param = param
        if min_no_echoes is not None: self.min_no_echoes = min_no_echoes
        if min_pct_echoes is not None: self.min_pct_echoes = min_pct_echoes
        if ts is not None: self.ts = ts
        hdw_data = pydarn.read_hdw_file(self.rad)
        self.lats, self.lons = pydarn.radar_fov(hdw_data.stid, coords="geo")
        self.log = f" Done initialization: {self.rad}, {self.dates}\n"
        self._fetch()
        self._filter()
        return
    
    def cacl_nechoes(self, intt):
        """
        For each hour interval caclulate minimum number of echoes
        Parameters
        ----------
        intt: Integration time in sec
        """
        nechoes = int(self.min_pct_echoes * 3600./(6.*intt))
        nechoes = self.min_no_echoes if nechoes > self.min_no_echoes else nechoes
        return nechoes
    
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
        if not os.path.exists(base): os.system("mkdir -p "+base)
        for p in ["raw", "dtrnd", "rsamp", "fft"]:
            self.dirs[p] = base + self.files["csv"]%(p) 
        self.dirs["log"] = base + self.files["log"]%("log") 
        self.dirs["rti_plot"] = base + self.files["rti_plot"]
        return
    
    def _fetch(self):
        """
        Fetch radar data from repository
        """
        self.log += " Fetching data...\n"
        dates = [
                    self.dates[0]-dt.timedelta(minutes=self.w_mins/2),
                    self.dates[1]+dt.timedelta(minutes=self.w_mins/2)
                ] if self.w_mins is not None else self.dates
        logger.info(f" Read radar file {self.rad} for {[d.strftime('%Y.%m.%dT%H.%M') for d in self.dates]}")
        self.log += f" Read radar file {self.rad} for {[d.strftime('%Y.%m.%dT%H.%M') for d in self.dates]}\n"
        self.fd = FetchData(self.rad, dates)
        _, scans = self.fd.fetch_data(by="scan")
        self.frame = self.fd.scans_to_pandas(scans)
        if len(self.frame) > 0:
            self.frame["srange"] = self.frame.frang + (self.frame.slist*self.frame.rsep)
            self.frame["intt"] = self.frame["intt.sc"] + 1.e-6*self.frame["intt.us"]
            if "ribiero" in self.gflg_key: 
                self.log += f" Modify GS flag.\n"
                logger.info(f" Modify GS flag.")
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
        self.log += f" Toral time window length {time_length}\n"
        self.log += f" RBSP beams {'-'.join([str(b) for b in self.beams])}\n"
        self.log += f" RBSP total data {len(self.frame)}\n"
        self.fil_frame = pd.DataFrame()
        for tw in time_windows:
            add_frame = True
            fil_frame = self.frame.copy()
            fil_frame = fil_frame[fil_frame.bmnum.isin(self.beams)]
            fil_frame = fil_frame[(fil_frame.time>=tw[0]) & (fil_frame.time<tw[1])]
            nechoes = self.cacl_nechoes(fil_frame.intt.mean())
            if "a" in self.filters: 
                if -1 in fil_frame[self.gflg_key].tolist(): 
                    fil_frame = fil_frame[(np.abs(fil_frame.v)>=50.) | (fil_frame.w_l>=50.) | (fil_frame[self.gflg_key]==0)]
                else: fil_frame = fil_frame[fil_frame[self.gflg_key]==0]
            if "b" in self.filters: fil_frame = fil_frame[(fil_frame.p_l>3.) & (fil_frame.v_e<=100.) & (fil_frame.w_l_e<=100.)]
            if "c" in self.filters: fil_frame = fil_frame[fil_frame.srange>500.]
            if "d" in self.filters:
                max_tdiff = np.rint(np.nanmax([t.total_seconds()/60. for t in 
                                               np.diff([tw[0]] + fil_frame.time.tolist() + [tw[1]])]))
                logger.info(f"DLen of {tw[0].strftime('%Y.%m.%dT%H.%M')}-{tw[1].strftime('%Y.%m.%dT%H.%M')} -hour- (NE:{nechoes}) {len(fil_frame)} & max(td)-{max_tdiff}")
                self.log += f"DLen of {tw[0].strftime('%Y.%m.%dT%H.%M')}-{tw[1].strftime('%Y.%m.%dT%H.%M')} -hour- (NE:{nechoes}) {len(fil_frame)} & max(td)-{max_tdiff}\n"
                if (len(fil_frame) < nechoes) or (max_tdiff >= self.max_tdiff): add_frame = False
            if add_frame: self.fil_frame = pd.concat([self.fil_frame, fil_frame])
        logger.info(f" RBSP total data after filter {len(self.fil_frame)}")
        self.log += f" RBSP total data after filter {len(self.fil_frame)}\n"
        
        # Detrending the dataset
        logger.info(f" Started detreanding data points, {len(self.fil_frame)}")
        self.log += f" Started detreanding data points, {len(self.fil_frame)}\n"
        self.d_frame = self.fil_frame.apply(self.__trnd_support__, axis=1)
        logger.info(f" Done detreanding data points, {len(self.d_frame)}")
        self.log += f" Done detreanding data points, {len(self.d_frame)}\n"
        
        # Discard, interpolating by 18 sec, and FFT
        self.r_frame = pd.DataFrame()
        self.fft_frame = pd.DataFrame()
        logger.info(f" Discard, interpolate data, and FFT by range-cell.")
        self.log += f" Discard, interpolate data, and FFT by range-cell.\n"
        if len(self.d_frame) > 0: self.__resample_rangecell_data__()
        return
    
    def __resample_rangecell_data__(self):
        """
        Interpolating by ts sec
        """
        beams, gates = np.unique(self.d_frame.bmnum), np.unique(self.d_frame.slist)
        time_length = np.rint((self.dates[1]-self.dates[0]).total_seconds()/(self.hour_win*3600.))
        time_windows = [(self.dates[0]+dt.timedelta(hours=h*self.hour_win),
                  self.dates[0]+dt.timedelta(hours=(h+1)*self.hour_win)) for h in range(int(time_length))]
        fn = lambda z, m, c: m * z + c
        #Very slow range cell wise interpolation
        tx = 1
        for tx, tw in enumerate(time_windows):
            self.log += f" Time window DIF Op: {tw}\n"
            for b in beams:
                for r in gates:
                    self.log += f" Beam-Gate DIF Op: {b}, {r} "
                    add_frame = True
                    o = self.d_frame[(self.d_frame.bmnum == b) & (self.d_frame.slist == r) &
                                     (self.d_frame.time>=tw[0]) & (self.d_frame.time<tw[1])]
                    if "d" in self.filters:
                        max_tdiff = np.rint(np.nanmax([t.total_seconds()/60. for t in 
                                                       np.diff([tw[0]] + o.time.tolist() + [tw[1]])]))
                        if (max_tdiff >= self.max_tdiff): add_frame = False
                        self.log += f" Max(td)-{max_tdiff}\n"
                    if (len(o) > 0) and add_frame:
                        nechoes = self.cacl_nechoes(o.intt.mean())
                        nechoes = int(nechoes * (tw[1]-tw[0]).total_seconds()/(3600.*self.hour_win))
                        self.log += f" Compare DIF Op [ne.len(o)]: {nechoes}, {len(o)}\n"
                        if len(o) >= nechoes:
                            intt = o.intt.mean()
                            tdiff = (tw[1]-tw[0]).total_seconds()/(3600.)
                            x, y = np.array(o.time.apply(lambda t: t.hour*3600 + 
                                                         t.minute*60 + t.second)), np.array(o[self.param])
                            x, y = x[~np.isnan(y)], y[~np.isnan(y)]
                            start = o.time.tolist()[0]
                            xnew = [x[0]+(i*self.ts) for i in range(int(200*tdiff))]
                            tnew = [start+dt.timedelta(seconds=i*self.ts) for i in range(int(200*tdiff))]
                            f = interp1d(x, y, kind=self.fit["kind"], bounds_error=False, fill_value="extrapolate")
                            ynew = f(xnew)
                            o = pd.DataFrame()
                            o["time"], o["bmnum"], o["slist"] = tnew, b, r
                            o["hour"], o[self.param] = np.array([1] + [0]*(len(o)-1)), ynew
                            o[self.param+".dof"], o[self.param+".sig"] = len(x)-self.fit["dod"], self.fit["sig"]
                            sprd, ta = self.estimate_spred(f, x, y, xnew)
                            o[self.param+".sprd"] = sprd*ta
                            o[self.param+".ub"], o[self.param+".lb"] = ynew + sprd*ta, ynew - sprd*ta
                            o["intt"], o["Tx"] = intt, tx
                            self.r_frame = pd.concat([self.r_frame, o])
                            fft = self.__run_fft__(o, b, r, tx)
                            self.fft_frame = pd.concat([self.fft_frame, fft])
        if len(self.r_frame) > 0: 
            self.log += f" Manipulate location information.\n"
            self.r_frame["rad"] = self.rad
            self.r_frame = self.r_frame.apply(self.__get_magnetic_loc__, axis=1)
        return
    
    def estimate_spred(self, f, x, y, xnew):
        """
        Estimate S_pred from fitting.
        """
        xm, n = x.mean(), len(x)
        sx2 = np.mean((x-xm)**2)
        sy2 = np.mean((y-f(x))**2)
        spred = np.sqrt( ( 1 + (1/n) + ((xnew-xm)**2/((n-1)*sx2)) ) )
        ta = 2*(1-T.ppf(self.fit["sig"], len(x)-self.fit["dod"]))
        return spred, ta
    
    def __get_magnetic_loc__(self, row):
        """
        Get AACGMV2 magnetic location
        """
        lat, lon = self.lats[row["slist"],row["bmnum"]], self.lons[row["slist"],row["bmnum"]]
        row["mlat"], row["mlon"], row["mlt"] = aacgmv2.get_aacgm_coord(lat, lon, 300, row["time"])
        return row
    
    def __run_fft__(self, o, b, r, tx):
        """
        Run FFT on velocity data
        """
        fft = pd.DataFrame()
        n = len(o)
        self.log += f" Applying normalization in FFT\n" if self.fft_amp_norm else f" No normalization in FFT\n"
        if self.fft_amp_norm: Baf = 2.0/n * rfft(np.array(o[self.param])*get_window("hanning",Nx=n))
        else: Baf = rfft(np.array(o[self.param])*get_window("hanning",Nx=n))
        frq = rfftfreq(n, self.ts)
        fft["frq"] = frq
        fft[self.param+"_real"], fft[self.param+"_imag"] = np.real(Baf), np.imag(Baf)
        fft["amp"], fft["ang"] = np.absolute(Baf), np.angle(Baf, deg=True)
        fft["bmnum"], fft["slist"], fft["rad"] = b, r, self.rad
        fft["Tx"] = tx
        return fft
    
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
    
    def _plots(self, beams=[], rclist=[]):
        """
        Plot data into PNG files
        """
        time_str = self.dates[0].strftime("%Y.%m.%dT%H:%M") + "-" + self.dates[1].strftime("%H:%M") + " UT"
        nplots = 3
        nplots = nplots+1 if "ribiero" in self.gflg_key else nplots
        nplots = nplots+1 if len(rclist)>0 else nplots
        beams = beams if (beams is not None) and (len(beams) > 0) else self.beams
        for bm in beams:
            rti = RTI(100, self.dates, num_subplots=nplots)
            rti.addParamPlot(self.frame, self.beams[0], xlabel="",
                             title="Date: %s, Rad: %s[Bm: %02d]"%(time_str, self.rad.upper(), self.beams[0]))
            rti.addParamPlot(self.r_frame, self.beams[0], title="Filters: %s \& "%"-".join(self.filters)+\
                                        r"Detrend $[T_w=%d mins]$"%self.w_mins, xlabel="")
            for rc in rclist:
                rti.add_range_cell_data(self.r_frame, rc, title="Filters: %s \& "%"-".join(self.filters)+\
                                        r"Detrend $[T_w=%d mins]$"%self.w_mins)
                rti.rc_ax.legend(loc=1)
            if "ribiero" in self.gflg_key: rti.addGSIS(self.db.frame, self.beams[0], "")
            rti.save(self.dirs["rti_plot"].format(rad=self.rad, bm="%02d"%self.beams[0], 
                                                  stime=self.dates[0].strftime("%H%M"), 
                                                  etime=self.dates[-1].strftime("%H%M")))
            rti.close()
        return
    
    def _save(self):
        """
        Print data into csv file for later processing.
        """
        self.log += f" Done processing, saving data to csv and logs.\n"
        stime, etime = self.dates[0].strftime("%H%M"), self.dates[-1].strftime("%H%M")
        for p in ["raw", "dtrnd", "rsamp", "fft"]:
            if self.save[p]:
                file = self.dirs[p].format(rad=self.rad, stime=stime, etime=etime)
                if p == "raw": self.fil_frame.to_csv(file, index=False, header=True, float_format="%g")
                if p == "dtrnd": self.d_frame.to_csv(file, index=False, header=True, float_format="%g")
                if p == "rsamp": self.r_frame.to_csv(file, index=False, header=True, float_format="%g")
                if p == "fft": self.fft_frame.to_csv(file, index=False, header=True, float_format="%g")
        if self.save["log"]:
            file = self.dirs["log"].format(rad=self.rad, stime=stime, etime=etime)
            logger.info(f" Proc interval time {np.round(time.time() - self.proc_start_time, 2)} sec.")
            self.log += f" Proc interval time {np.round(time.time() - self.proc_start_time, 2)} sec."
            with open(file, "w") as f: f.writelines(self.log)
        base = self.files["base"].format(run_id=self.run_id, date=self.dates[0].strftime("%Y-%m-%d"))
        prm_file = "/".join(base.split("/")[:-2]) + "/params.json"
        if self.save["param"] and \
                (not os.path.exists(prm_file)): shutil.copy("config/params.json", prm_file)
        return
    
    @staticmethod
    def filter_data_by_detrending(rad, dates, beams, filters=["a", "b", "c", "d"], hour_win=1., 
                                  gflg_key="gflg", w_mins=15., param="v", rclist=[], min_no_echoes=60):
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
        f = Filter(rad, dates, beams, filters, hour_win, gflg_key, w_mins, param, min_no_echoes=min_no_echoes)
        f._plots(rclist=rclist)
        f._save()
        return f
    
class DataFetcherFilter(object):
    """
    For each RBSP log entry in the log files
    this class forks a process to fetch fitacf
    data from the SD repository and filter it 
    based on the given filtering condition.
    """
    
    def __init__(self, _filestr="config/logs/*.txt", cores=24, run_first=None):
        """
        Params
        ------
        _filestr - Regular expression to search files
        cores - Mutiprocessing cores
        run_first - Firt N modes to run
        """
        self._filestr = _filestr
        self.cores = cores
        self.run_first = run_first
        self.rbsp_logs = utils.read_rbsp_logs(_filestr)
        self._run()
        return
    
    def _proc(self, o):
        """
        Process method to invoke filter
        """
        f = None
        if (o["etime"]-o["stime"]).total_seconds()/3600. >= 1.:
            rad, dates, beams = o["rad"], [o["stime"], o["etime"]], o["beams"]
            logger.info(f"Filtering radar {rad} for {[d.strftime('%Y.%m.%dT%H.%M') for d in dates]}")
            f = Filter(rad, dates, beams)
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
    start = time.time()
    DataFetcherFilter(run_first=147)
    end = time.time()
    logger.info(f" Interval time {np.round(end - start, 2)} sec.")
#     Filter.filter_data_by_detrending("pgr", [dt.datetime(2016,1,25,1), dt.datetime(2016,1,25,1,30)], beams=[12],
#                                     hour_win=0.5, rclist=[{"bmnum":12, "gate":13, "color":"r"}, 
#                                                           {"bmnum":12, "gate":15, "color":"b"}],
#                                     min_no_echoes=60)
    pass