#!/usr/bin/env python

"""utils.py: utility module for data parsing and helping analsysis or plots."""

__author__ = "Chakraborty, S."
__copyright__ = ""
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"


import os
import datetime as dt
import numpy as np
import glob
from loguru import logger
import configparser
from scipy.optimize import curve_fit
from scipy.stats.distributions import t
from matplotlib.dates import date2num
import paramiko
import os
from cryptography.fernet import Fernet
import json
import pydarn

from geo import rad_fov, geoPack


def compute_field_of_view_parameters(rad):
    """
    For a radar code compute
    i) glat, ii) glon, iii) azm
    for each range cell
    """
    hdw = pydarn.read_hdw_file(rad)
    ngates, nbeams = hdw.gates - 10, hdw.beams
    glats, glons = pydarn.Coords.GEOGRAPHIC(hdw.stid)
    azms = np.zeros((nbeams, ngates))
    for b in range(nbeams):
        for g in range(ngates):
            d = geoPack.calcDistPnt(
                glats[g, b],
                glons[g, b],
                300,
                distLat=glats[g + 1, b],
                distLon=glons[g + 1, b],
                distAlt=300,
            )
            azms[b, g] = d["az"]
    return glats, glons, azms


def fitting_curves(x, y, xnew, fn, alpha=0.05):
    """
    Fitting the data with confidance intervals
    """
    pars, pcov = curve_fit(fn, x, y)
    dof = max(0, len(y) - len(pars))
    ynew = fn(np.array(xnew), *pars)
    tval = t.ppf(1.0 - alpha / 2.0, dof)
    cov = np.sqrt((np.diagonal(pcov) ** 2).sum())
    yu, yl = ynew + tval * cov, ynew + tval * cov
    return ynew, yu, yl


def folders(base_fold="tmp/sd/", date=None, create=True):
    """
    Create folder structures

    If create 'T' then creates folders else fetch
    structure only.
    """
    bstr = ""
    folders = base_fold.split("/")
    folders = folders if date is None else folders + [date.strftime("%Y-%m-%d")]
    if create:
        for d in folders:
            bstr += d + "/"
            if not os.path.exists(bstr):
                os.system("mkdir " + bstr)
    _dirs_ = dict(
        base=base_fold,
        event=base_fold + "/" + date.strftime("%Y-%m-%d"),
        radar_dtrnd_file=base_fold
        + "/"
        + date.strftime("%Y-%m-%d")
        + "/{rad}_{stime}_{etime}_d.csv",
        radar_rsamp_file=base_fold
        + "/"
        + date.strftime("%Y-%m-%d")
        + "/{rad}_{stime}_{etime}_r.csv",
        radar_fft_file=base_fold
        + "/"
        + date.strftime("%Y-%m-%d")
        + "/{rad}_{stime}_{etime}_fft.csv",
        radar_log_file=base_fold
        + "/"
        + date.strftime("%Y-%m-%d")
        + "/{rad}_{stime}_{etime}.log",
        radar_rti_plot=base_fold
        + "/"
        + date.strftime("%Y-%m-%d")
        + "/{rad}_{bm}_{stime}_{etime}.png",
    )
    return _dirs_


def read_rbsp_logs(_filestr="config/logs/*.txt"):
    """
    Read all log files and store them each records
    to a list of dictionary
    """
    files = sorted(glob.glob(_filestr))
    rbsp_modes = []
    for f in files:
        logger.info(f"RBSP log file {f}")
        with open(f, "r") as fp:
            lines = fp.readlines()
        for l in lines:
            l = l.replace("\n", "").split(" ")
            l = list(filter(None, l))
            r, st, et = (
                l[0],
                dt.datetime.strptime(l[1] + "T" + l[2], "%Y-%m-%dT%H:%M:%S"),
                dt.datetime.strptime(l[3] + "T" + l[4], "%Y-%m-%dT%H:%M:%S"),
            )
            b = [
                int(l[5].replace("[", "").replace(",", "")),
                int(l[6].replace(",", "")),
                int(l[7].replace("]", "").replace(",", "")),
            ]
            op = [int(l[8]), int(l[9])]
            o = dict(rad=r, beams=b, stime=st, etime=et, other_params=op)
            rbsp_modes.append(o)
    logger.info(f"Total RBSP mode log entry {len(rbsp_modes)}")
    return rbsp_modes


def get_config(key, section="HSR"):
    config = configparser.ConfigParser()
    config.read("config/conf.ini")
    val = config[section][key]
    return val


def get_gridded_parameters(
    q, xparam="beam", yparam="slist", zparam="v", r=0, rounding=False
):
    """
    Method converts scans to "beam" and "slist" or gate
    """
    plotParamDF = q[[xparam, yparam, zparam]]
    if rounding:
        plotParamDF.loc[:, xparam] = np.round(plotParamDF[xparam].tolist(), r)
        plotParamDF.loc[:, yparam] = np.round(plotParamDF[yparam].tolist(), r)
    else:
        plotParamDF[xparam] = plotParamDF[xparam].tolist()
        plotParamDF[yparam] = plotParamDF[yparam].tolist()
    plotParamDF = plotParamDF.groupby([xparam, yparam]).mean().reset_index()
    plotParamDF = plotParamDF[[xparam, yparam, zparam]].pivot(xparam, yparam)
    x = plotParamDF.index.values
    y = plotParamDF.columns.levels[1].values
    X, Y = np.meshgrid(x, y)
    # Mask the nan values! pcolormesh can't handle them well!
    Z = np.ma.masked_where(
        np.isnan(plotParamDF[zparam].values), plotParamDF[zparam].values
    )
    return X, Y, Z


def ribiero_gs_flg(vel, time):
    L = np.abs(time[-1] - time[0]) * 24
    high = np.sum(np.abs(vel) > 15.0)
    low = np.sum(np.abs(vel) <= 15.0)
    if low == 0:
        R = 1.0  # TODO hmm... this works right?
    else:
        R = high / low  # High vel / low vel ratio
    # See Figure 4 in Ribiero 2011
    if L > 14.0:
        # Addition by us
        if R > 0.15:
            return False  # IS
        else:
            return True  # GS
        # Classic Ribiero 2011
        # return True  # GS
    elif L > 3:
        if R > 0.2:
            return False
        else:
            return True
    elif L > 2:
        if R > 0.33:
            return False
        else:
            return True
    elif L > 1:
        if R > 0.475:
            return False
        else:
            return True
    # Addition by Burrell 2018 "Solar influences..."
    else:
        if R > 0.5:
            return False
        else:
            return True
    # Classic Ribiero 2011
    # else:
    #    return False

def get_radar_fov_locations(rad=None, hdw=None):
    """
    Fetch radar fov lats and lons
    """
    if hdw is None: hdw = pydarn.read_hdw_file(rad)
    fov = rad_fov.CalcFov(hdw=hdw)
    lats, lons = fov.latFull.T, fov.lonFull.T
    return lats, lons
    
def _run_riberio_threshold_on_rad(u, flag="gflg_ribiero"):
    df = u.copy()
    clust_flag = np.array(df.cluster_tag)
    gs_flg = np.zeros_like(clust_flag)
    vel = np.hstack(np.abs(df.v))
    t = np.hstack(df.time.apply(lambda x: date2num(x)))
    clust_flag[np.array(df.slist) < 7] = -1
    gs_flg = np.zeros_like(clust_flag)
    for c in np.unique(clust_flag):
        clust_mask = c == clust_flag
        if c == -1:
            gs_flg[clust_mask] = -1
        else:
            gs_flg[clust_mask] = ribiero_gs_flg(vel[clust_mask], t[clust_mask])
    df[flag] = gs_flg
    return df


class Conn2Remote(object):
    def __init__(self, host, user, key_filename, port=22, passcode=None):
        self.host = host
        self.user = user
        self.key_filename = key_filename
        self.passcode = passcode
        self.port = port
        self.con = False
        if passcode:
            self.decrypt()
        self.conn()
        return

    def decrypt(self):
        passcode = bytes(self.passcode, encoding="utf8")
        cipher_suite = Fernet(passcode)
        self.user = cipher_suite.decrypt(bytes(self.user, encoding="utf8")).decode(
            "utf-8"
        )
        self.host = cipher_suite.decrypt(bytes(self.host, encoding="utf8")).decode(
            "utf-8"
        )
        return

    def conn(self):
        if not self.con:
            self.ssh = paramiko.SSHClient()
            self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh.connect(
                hostname=self.host,
                port=self.port,
                username=self.user,
                key_filename=self.key_filename,
            )
            self.scp = paramiko.SFTPClient.from_transport(self.ssh.get_transport())
            self.con = True
        return

    def close(self):
        if self.con:
            self.scp.close()
            self.ssh.close()
        return


def encrypt(host, user, filename="config/passcode.json"):
    passcode = Fernet.generate_key()
    cipher_suite = Fernet(passcode)
    host = cipher_suite.encrypt(bytes(host, encoding="utf8"))
    user = cipher_suite.encrypt(bytes(user, encoding="utf8"))
    with open(filename, "w") as f:
        f.write(
            json.dumps(
                {
                    "user": user.decode("utf-8"),
                    "host": host.decode("utf-8"),
                    "passcode": passcode.decode("utf-8"),
                },
                sort_keys=True,
                indent=4,
            )
        )
    return


def get_session(filename="config/passcode.json", key_filename="", isclose=False):
    with open(filename, "r") as f:
        obj = json.loads("".join(f.readlines()))
        conn = Conn2Remote(
            obj["host"],
            obj["user"],
            key_filename=key_filename,
            passcode=obj["passcode"],
        )
    if isclose:
        conn.close()
    return conn


def chek_remote_file_exists(fname, conn):
    try:
        logger.info("File Check:" + fname)
        conn.scp.stat(fname)
        return True
    except FileNotFoundError:
        return False


def to_remote_FS(conn, local_file, rdir, is_local_remove=False):
    conn.ssh.exec_command("mkdir -p " + rdir)
    remote_file = rdir + local_file.split("/")[-1]
    logger.info(" To file:" + remote_file)
    conn.scp.put(local_file, remote_file)
    if is_local_remove:
        os.remove(local_file)
    return


def from_remote_FS(conn, local_file, LFS):
    remote_file = LFS + local_file.replace("../", "")
    logger.info(" From file:" + remote_file)
    conn.scp.get(remote_file, local_file)
    return


def get_pubfile():
    with open("config/pub.json", "r") as f:
        obj = json.loads("".join(f.readlines()))
        pubfile = obj["pubfile"]
    return pubfile


def fetch_file(conn, local_file, LFS):
    remote_file = LFS + local_file.replace("../", "")
    is_remote = chek_remote_file_exists(remote_file, conn)
    if is_remote:
        from_remote_FS(conn, local_file, LFS)
    return is_remote


if __name__ == "__main__":
    "__main__ function"
    read_rbsp_logs()
    pass
