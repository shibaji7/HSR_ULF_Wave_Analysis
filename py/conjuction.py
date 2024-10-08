#!/usr/bin/env python

"""conjuction.py: Conjuction study - (1) create a list of conjuction events (2) plot survey plots."""

__author__ = "Shi, X."
__copyright__ = ""
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import argparse
import datetime as dt
import glob
import os

import numpy as np
import pandas as pd
from analytics import StackPlots, TimeSeriesAnalysis
from reader import Reader
from tqdm import tqdm

BASE_LOCATION = "/home/shibaji/OneDrive/SuperDARN-Data-Share/Shi/HSR/latest/conjuction/"


def event_analysis(index=0):
    """
    Run individual event analysis
    """
    records = pd.read_csv(
        "config/conjuction-records/conjuction.txt",
        parse_dates=["StartTime", "StopTime"],
    )
    record = records.iloc[index].to_dict()
    EventAnalysis(record, index)
    return


class EventAnalysis(object):
    """
    Read files from databases
    """

    def __init__(self, event, index):
        self.event = event
        self.index = index
        for key in list(event.keys()):
            setattr(self, key, event[key])
        self.has_records = 0
        if not os.path.exists(BASE_LOCATION):
            os.makedirs(BASE_LOCATION, exist_ok=True)
        self.fig_file_name = BASE_LOCATION + f"event-{'%02d'%self.index}.png"
        self.get_radar_observations()
        self.get_satellite_observations()
        self.plot_RTI_panels()
        return

    def get_radar_observations(self):
        self.reader = Reader()
        self.logs = self.reader.check_entries(rad=self.rad, date=self.StartTime.date())
        if len(self.logs) > 0:
            self.has_records += 1
            log = self.logs.iloc[0].to_dict()
            log.update(self.event)
            log["bin_time"] = log["date"]
            log = pd.DataFrame.from_records([log])
            self.timeseries = TimeSeriesAnalysis(log.iloc[0])
        return

    def get_satellite_observations(self):
        self.has_records += 2
        return

    def plot_RTI_panels(self):
        if self.has_records >= 1:
            sp = StackPlots(self.timeseries)
            sp.plot_indexs()
            sp.addParamPlot(pmin=-300, pmax=300)
            sp.add_time_series(ylim=[-1000, 1000])
            sp.add_fft()
            sp.save(self.fig_file_name)
            sp.close()
        return


def load_file(sat, file):
    """
    Load a file name
    """
    lines, o = None, []
    with open(file, "r") as f:
        lines = f.readlines()
    header = list(filter(None, lines[0].replace("\n", "").split(" ")))
    for line in lines[1:]:
        line = list(filter(None, line.replace("\n", "").split(" ")))
        dx = {u: v for u, v in zip(header, line)}
        o.append(dx)
    o = pd.DataFrame.from_records(o)
    o.StartTime = o.StartTime.apply(
        lambda x: dt.datetime.strptime(x, "%Y-%m-%d/%H:%M:%S")
    )
    o.StopTime = o.StopTime.apply(
        lambda x: dt.datetime.strptime(x, "%Y-%m-%d/%H:%M:%S")
    )
    for col in o.columns:
        if col not in ["StartTime", "StopTime"]:
            o[col] = np.array(o[col]).astype(float)
    o["SAT"] = sat
    return o


def compare_config_log_files(folder="config/conjuction-records/", dlat=3.0, dlon=10.0):
    """
    Compare the log files under the folder
    """
    SAT_files = glob.glob(folder + "*footpoints*")
    sd_record = pd.read_csv(
        folder + "SD_RBSP_mode_Pc5_event_list_radar_info.txt",
        parse_dates=["stime", "etime"],
    )
    satellites = [sat.split("/")[-1].split("_")[0] for sat in SAT_files]
    records = []
    sat_records = {
        sat: load_file(sat, file) for sat, file in zip(satellites, SAT_files)
    }
    for i, rec in tqdm(sd_record.iterrows()):
        stime, etime = rec["stime"], rec["etime"]
        mlat, mlon, mlt = rec["mlat"], rec["mlon"], rec["mlt"]
        rad, beam, gate = rec["rad"], rec["beam"], rec["gate"]
        for sat in satellites:
            o = sat_records[sat]
            o = o[(o.StartTime == stime) & (o.StopTime == etime)]
            u1 = o[
                ((o.T89_CGM_LAT1 - dlat / 2) <= mlat)
                & ((o.T89_CGM_LAT1 + dlat / 2) >= mlat)
                & ((o.T89_CGM_LON1 - dlon / 2) <= mlon)
                & ((o.T89_CGM_LON1 + dlon / 2) >= mlon)
            ]
            u2 = o[
                ((o.T89_CGM_LAT2 - dlat / 2) <= mlat)
                & ((o.T89_CGM_LAT2 + dlat / 2) >= mlat)
                & ((o.T89_CGM_LON2 - dlon / 2) <= mlon)
                & ((o.T89_CGM_LON2 + dlon / 2) >= mlon)
            ]
            u3 = o[
                ((o.T89_CGM_LAT3 - dlat / 2) <= mlat)
                & ((o.T89_CGM_LAT3 + dlat / 2) >= mlat)
                & ((o.T89_CGM_LON3 - dlon / 2) <= mlon)
                & ((o.T89_CGM_LON3 + dlon / 2) >= mlon)
            ]
            u = pd.concat([u1, u2, u3])
            if len(u) > 0:
                u["rad"], u["beam"], u["gate"] = rad, beam, gate
                records.append(u)
    records = pd.concat(records)
    records.to_csv(folder + "conjuction.txt", index=False, header=True)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Conjuction Study")
    parser.add_argument(
        "-m",
        "--method",
        default="EA",
        type=str,
        help="CON: Create Conjunction File; EA: Event analysis survey plots",
    )
    parser.add_argument(
        "-dlat",
        "--dlat",
        default=3.0,
        type=float,
        help="Latitude bin to consider conjction",
    )
    parser.add_argument(
        "-dlon",
        "--dlon",
        default=10.0,
        type=float,
        help="Longitude bin to consider conjction",
    )
    parser.add_argument("-i", "--index", default=0, type=int, help="Event analysis")
    args = parser.parse_args()
    for k in vars(args).keys():
        print("     ", k, "->", str(vars(args)[k]))
    if args.method == "CON":
        compare_config_log_files(dlat=args.dlat, dlon=args.dlon)
    elif args.method == "EA":
        event_analysis(index=args.index)
    else:
        print(f"Invalid method / not implemented {args.method}")
