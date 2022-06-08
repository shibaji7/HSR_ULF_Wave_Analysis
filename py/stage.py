#!/usr/bin/env python

"""
    stage.py: module to stagging data into raw format
"""

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

from get_fit_data import FetchData
from functools import partial
import multiprocessing as mp
import pandas as pd
import numpy as np
import datetime as dt
from loguru import logger
from sklearn.cluster import DBSCAN
import time
import os
import sys
import shutil
import utils as utils
import json


class DBScan(object):
    """Class to cluster 2D or 3D data"""

    def __init__(
        self, frame, params={"eps_g": 1, "eps_s": 1, "min_samples": 10, "eps": 1}
    ):
        self.frame = frame
        for k in params.keys():
            setattr(self, k, params[k])
        self.run_2D_cluster_bybeam()
        return

    def run_2D_cluster_bybeam(self):
        o = pd.DataFrame()
        for b in np.unique(self.frame.bmnum):
            _o = self.frame[self.frame.bmnum == b]
            _o.slist, _o.scnum, eps, min_samples = (
                _o.slist / self.eps_g,
                _o.scnum / self.eps_s,
                self.eps,
                self.min_samples,
            )
            ds = DBSCAN(eps=self.eps, min_samples=min_samples).fit(
                _o[["slist", "scnum"]].values
            )
            _o["cluster_tag"] = ds.labels_
            _o = utils._run_riberio_threshold_on_rad(_o)
            o = pd.concat([o, _o])
        self.frame = o.sort_values("time").copy()
        return


class StagingUnit(object):
    """
    Store all the Raw SuperDARN data to local or
    ARC (remote super computer) for further processing.
    """

    def __init__(self, rad, dates):
        """
        Parameters
        ----------
        rad - 3 char radar code
        dates - [sdate, edate]
        """
        self.proc_start_time = time.time()
        self.rad = rad
        self.dates = dates
        self.load_params(dates)
        self._fetch()
        return

    def load_params(self, dates):
        """
        Load parameters from params.json
        """
        with open("config/params.json") as f:
            o = json.load(f)
        for k in o["filter"]:
            setattr(self, k, o["filter"][k])
        setattr(self, "run_id", o["run_id"])
        setattr(self, "save", o["save"])
        setattr(self, "files", o["files"])
        setattr(self, "ftype", o["ftype"])
        setattr(self, "stage_arc", o["stage_arc"])
        setattr(self, "remove_local", o["remove_local"])
        self.dirs = {}
        self.stage = self.files["stage"].format(date=dates[0].strftime("%Y-%m-%d"))
        self.arc_stage = self.files["arc_stage"].format(
            date=dates[0].strftime("%Y-%m-%d")
        )
        if not os.path.exists(self.stage):
            os.system("mkdir -p " + self.stage)
        self.dirs["raw"] = self.stage + self.files["csv"] % ("raw")
        stime, etime = dates[0].strftime("%H%M"), dates[-1].strftime("%H%M")
        self.dirs["raw"] = self.dirs["raw"].format(
            rad=self.rad, stime=stime, etime=etime
        )
        return

    def _fetch(self):
        """
        Fetch radar data from repository
        """
        if not os.path.exists(self.dirs["raw"]):
            logger.info(" Fetching data...\n")
            dates = (
                [
                    self.dates[0] - dt.timedelta(minutes=self.w_mins / 2),
                    self.dates[1] + dt.timedelta(minutes=self.w_mins / 2),
                ]
                if self.w_mins is not None
                else self.dates
            )
            logger.info(
                f" Read radar file {self.rad} for {[d.strftime('%Y.%m.%dT%H.%M') for d in self.dates]}"
            )
            self.fd = FetchData(self.rad, dates, ftype=self.ftype)
            _, scans, self.data_exists = self.fd.fetch_data(by="scan")
            if self.data_exists:
                self.frame = self.fd.scans_to_pandas(scans)
                if len(self.frame) > 0:
                    self.frame["srange"] = self.frame.frang + (
                        self.frame.slist * self.frame.rsep
                    )
                    self.frame["intt"] = (
                        self.frame["intt.sc"] + 1.0e-6 * self.frame["intt.us"]
                    )
                    if "ribiero" in self.gflg_key:
                        logger.info(f" Modify GS flag.")
                        self.db = DBScan(self.frame.copy())
                        self.frame = self.db.frame.copy()
            else:
                logger.info(f" Radar file does not exist!")
        else:
            self.data_exists = False
        return

    def _save(self):
        """
        Print data into csv file for later processing.
        """
        self.frame.to_csv(self.dirs["raw"], index=False, header=True, float_format="%g")
        ## To remote super computer
        if self.stage_arc:
            conn = utils.get_session(key_filename=utils.get_pubfile())
            utils.to_remote_FS(
                conn, self.dirs["raw"], self.arc_stage, self.remove_local
            )
        return


class StagingHopper(object):
    """
    For each RBSP log entry in the log files
    this class forks a process to fetch fitacf
    data from the SD repository and store to
    local or ARC.
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
        Process method to invoke stagging unit
        """
        s = None
        if (o["etime"] - o["stime"]).total_seconds() / 3600.0 >= 1.0:
            rad, dates = o["rad"], [o["stime"], o["etime"]]
            logger.info(
                f"Filtering radar {rad} for {[d.strftime('%Y.%m.%dT%H.%M') for d in dates]}"
            )
            s = StagingUnit(rad, dates)
            if s.data_exists:
                s._save()
        return s

    def _run(self):
        """
        Run parallel threads to access and save data
        """
        if self.run_first:
            logger.info(f"Start parallel procs for first {self.run_first} entries")
        else:
            logger.info(f"Start parallel procs for first {len(self.rbsp_logs)} entries")
        self.flist = []
        rlist = (
            self.rbsp_logs
            if self.run_first is None
            else self.rbsp_logs[: self.run_first]
        )
        p0 = mp.Pool(self.cores)
        partial_filter = partial(self._proc)
        for f in p0.map(partial_filter, rlist):
            self.flist.append(f)
        return


if __name__ == "__main__":
    "__main__ function"
    start = time.time()
    StagingHopper(run_first=None)
    end = time.time()
    logger.info(f" Interval time {np.round(end - start, 2)} sec.")
    pass
