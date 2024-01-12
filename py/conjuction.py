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
import glob
import pandas as pd
import datetime as dt
import numpy as np
from tqdm import tqdm

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
        dx = {u:v for u, v in zip(header, line)}
        o.append(dx)
    o = pd.DataFrame.from_records(o)
    o.StartTime = o.StartTime.apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d/%H:%M:%S"))
    o.StopTime = o.StopTime.apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d/%H:%M:%S"))
    for col in o.columns:
        if col not in ["StartTime", "StopTime"]:
            o[col] = np.array(o[col]).astype(float)
    o["SAT"] = sat
    return o

def compare_config_log_files(folder="config/conjuction-records/", dlat=3., dlon=10.):
    """
    Compare the log files under the folder
    """
    SAT_files = glob.glob(folder + "*footpoints*")
    sd_record = pd.read_csv(folder + "SD_RBSP_mode_Pc5_event_list_radar_info.txt", parse_dates=["stime","etime"])
    satellites = [sat.split("/")[-1].split("_")[0] for sat in SAT_files]
    records = []
    sat_records = {sat:load_file(sat, file) for sat, file in zip(satellites, SAT_files)}
    for i, rec in tqdm(sd_record.iterrows()):
        stime, etime = rec["stime"], rec["etime"]
        mlat, mlon, mlt = rec["mlat"], rec["mlon"], rec["mlt"]
        rad, beam, gate = rec["rad"], rec["beam"], rec["gate"]
        for sat in satellites:
            o = sat_records[sat]
            o = o[
                (o.StartTime==stime)
                & (o.StopTime==etime)
            ]
            u1 = o[
                ((o.T89_CGM_LAT1 - dlat/2) <= mlat)
                & ((o.T89_CGM_LAT1 + dlat/2) >= mlat)
                & ((o.T89_CGM_LON1 - dlon/2) <= mlon)
                & ((o.T89_CGM_LON1 + dlon/2) >= mlon)
            ]
            u2 = o[
                ((o.T89_CGM_LAT2 - dlat/2) <= mlat)
                & ((o.T89_CGM_LAT2 + dlat/2) >= mlat)
                & ((o.T89_CGM_LON2 - dlon/2) <= mlon)
                & ((o.T89_CGM_LON2 + dlon/2) >= mlon)
            ]
            u3 = o[
                ((o.T89_CGM_LAT3 - dlat/2) <= mlat)
                & ((o.T89_CGM_LAT3 + dlat/2) >= mlat)
                & ((o.T89_CGM_LON3 - dlon/2) <= mlon)
                & ((o.T89_CGM_LON3 + dlon/2) >= mlon)
            ]
            u = pd.concat([u1, u2, u3])
            if len(u) > 0:
                u["rad"], u["beam"], u["gate"] = rad, beam, gate
                records.append(u)
    records = pd.concat(records)
    records.to_csv(folder+"conjuction.txt", index=False, header=True)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Conjuction Study")
    parser.add_argument(
        "-m", "--method", default="CON", type=str, help="CON: Create Conjunction File; EA: Event analysis survey plots"
    )
    parser.add_argument(
        "-dlat", "--dlat", default=3., type=float, help="Latitude bin to consider conjction"
    )
    parser.add_argument(
        "-dlon", "--dlon", default=10., type=float, help="Longitude bin to consider conjction"
    )
    args = parser.parse_args()
    for k in vars(args).keys():
        print("     ", k, "->", str(vars(args)[k]))
    if args.method == "CON":
        compare_config_log_files(dlat=args.dlat, dlon=args.dlon)
    elif args.method == "EA":
        pass
    else:
        print(f"Invalid method / not implemented {args.method}")
    