#!/usr/bin/env python
"""
    post_process.py: Analysis module for data post processing and analsysis.
"""

__author__ = "Shi, X."
__copyright__ = ""
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import sys

sys.path.extend(["py/", "py/sw_param/"])

import glob
import os
import time
from multiprocessing.pool import Pool

import numpy as np
import pandas as pd
from calc_ionospheric_params import ComputeIonosphereicConductivity
from calc_ionospheric_params import ComputeIonosphereicEField as CIE
from loguru import logger
from reader import Reader


def narrowband_wave_finder(
    freqs, psd, phase, fl_s=0.0016, fh_s=0.0067, fl_t=0.000278, fh_t=0.027778
):
    """
    inputs:
    freqs: frequencies in Hz (should have the same shape with psd)
    psd: power spectral density [V**2/Hz] from FFT
    phase: phase in degree from FFT
    fl_s: lower frequency limit of wave power
    fh_s: higher frequency limit of wave power
    fl_t: lower frequency limit of total power
    fh_t: higher frequency limit of total power
    outputs: return None if no event is detected;
    return FWHM, FWHM_left, FWHM_right, peak_psd, peak_freq, S_sig, S_total, I_sig, peak_phase
    if an event is detected based on the criteria below:
    1. Find the largest peak in 1.6-6.7 mHz;
    2. Calculate the FWHM of the peak; if FWHM < 2 mHz, calculate the Pc5 inegrated power
    (S_Pc5) and the spectral power integrated over 0.278-27.8 mHz (S_total);
    3. Calculate the Pc5 index = S_Pc5/S_total.
    """
    from scipy.signal import find_peaks, peak_widths

    peaks, _ = find_peaks(psd)

    peak_f = freqs[peaks]
    peak_psd = psd[peaks]
    peak_phase = phase[peaks]
    sig_ind = np.where((peak_f >= fl_s) & (peak_f <= fh_s))

    if np.size(sig_ind) == 0:
        return None
    else:
        peak_sig = peaks[sig_ind]
        peak_psd_sig = peak_psd[sig_ind]
        peak_f_sig = peak_f[sig_ind]
        peak_phase_sig = peak_phase[sig_ind]
        # print(peak_psd_sig)

        sig_max_ind = np.argmax(peak_psd_sig)
        peak_psd_max = peak_psd_sig[sig_max_ind]
        peak_freq = peak_f_sig[sig_max_ind] * 1000.0
        peak_ang = peak_phase_sig[sig_max_ind]

        peak_ind = peak_sig[sig_max_ind]
        # print(peak_ind)

        results_half = peak_widths(psd, [peak_ind], rel_height=0.5)
        f_res = freqs[1] - freqs[0]
        FWHM = results_half[0][0] * f_res * 1000.0  # mHz
        if FWHM >= 2:
            return None
        else:
            psd_sig = psd[(freqs >= fl_s) & (freqs <= fh_s)]
            psd_total = psd[(freqs >= fl_t) & (freqs <= fh_t)]
            S_sig = np.trapz(psd_sig, dx=f_res)
            S_total = np.trapz(psd_total, dx=f_res)
            I_sig = S_sig / S_total
            return (
                FWHM,
                results_half[2][0],
                results_half[3][0],
                peak_psd_max,
                peak_freq,
                S_sig,
                S_total,
                I_sig,
                peak_ang,
            )


def generate_stack_plot(
    o_ts, o_fft, stack_plot, bm, gt, rad, stime, etime, wave_event, I_min, N, N_min
):
    # Stack plots
    if (
        (stack_plot == True)
        and (wave_event[7] > I_min)
        and (len_df[N] * intt_df[N] >= N_min)
    ):
        frames = [
            {
                "p": "rsamp",
                "df": o_ts,
                "title": rad + ", Beam " + str(bm) + ", Gate " + str(gt) + ", RSamp",
            },
            {
                "p": "fft",
                "df": o_fft,
                "title": rad + ", Beam " + str(bm) + ", Gate " + str(gt) + ", FFT",
            },
        ]

        fig_name = "hpPc5_event_{rad}_{stime}_{etime}_b{beam}_g{gate}.png".format(
            rad=rad,
            stime=stime.strftime("%Y%m%d-%H%M"),
            etime=etime.strftime("%Y%m%d-%H%M"),
            beam=bm,
            gate=gt,
        )
        r.generate_stacks(frames, fig_name)
    return


def compute_Efield_each_entry(event):
    """
    This method invoked from
    `save_event_info` method by parallel proc
    """
    # Set all objects/parameters
    (
        index,
        rad,
        rbsp_log_fn,
        row,
        bm,
        gt,
        Tn,
        stime,
        etime,
        E_method,
        mag_type,
        stack_plot,
        I_min,
        N_min,
    ) = event
    logger.info(f"Running event number: {index}")
    # Create a reader object that reads all the RBSP mode entries
    r = Reader(_filestr="config/logs/" + rbsp_log_fn)
    r.parse_files(select=[row])
    o_fft = r.file_entries[row].get_data(p="fft", beams=[bm], gates=[gt], Tx=[Tn])

    event_dic = {
        "FWHM": None,
        "f_left_ind": None,
        "f_right_ind": None,
        "peak_psd": None,
        "peak_freq": None,
        "peak_ang": None,
        "S_sig": None,
        "S_total": None,
        "I_sig": None,
        "mlat": None,
        "mlon": None,
        "mlt": None,
        "rad": None,
        "beam": None,
        "gate": None,
        "stime": None,
        "etime": None,
        "len": None,
        "intt": None,
        "Erms": None,
        "num_bad_data_rsamp": None,
        "isValid": False,
    }

    if (not o_fft.empty) and (not o_fft["amp"].isnull().values.any()):
        o_ts = r.file_entries[row].get_data(p="rsamp", beams=[bm], gates=[gt], Tx=[Tn])

        # Find narrowband Pc5 event
        freqs = np.array(o_fft.frq)
        psd = np.array(18.0 / 100 * o_fft.amp**2)
        phase = np.array(o_fft.ang)
        wave_event = narrowband_wave_finder(freqs, psd, phase)

        if wave_event:
            event_dic["isValid"] = True
            event_dic["FWHM"] = wave_event[0]
            event_dic["f_left_ind"] = wave_event[1]
            event_dic["f_right_ind"] = wave_event[2]
            event_dic["peak_psd"] = wave_event[3]
            event_dic["peak_freq"] = wave_event[4]
            event_dic["S_sig"] = wave_event[5]
            event_dic["S_total"] = wave_event[6]
            event_dic["I_sig"] = wave_event[7]
            event_dic["peak_ang"] = wave_event[8]

            N = int(len(o_ts.v) / 2)
            mlat_df = np.array(o_ts["mlat"])
            mlon_df = np.array(o_ts["mlon"])
            mlt_df = np.array(o_ts["mlt"])
            len_df = np.array(o_ts["len"])
            intt_df = np.array(o_ts["intt"])
            cip = CIE(rad, o_ts, {"e_field": E_method, "mag_type": mag_type})
            cip.compute_efield()
            r_frame = cip.df.copy()
            E_VXB = np.array(r_frame["E_vlos"])
            E_rms = np.sqrt(np.sum(np.square(E_VXB)) / N)

            event_dic["num_bad_data_rsamp"] = o_ts["num_bad_data_rsamp"].tolist()[0]
            event_dic["Erms"] = E_rms
            event_dic["mlat"] = mlat_df[N]
            event_dic["mlon"] = mlon_df[N]
            event_dic["mlt"] = mlt_df[N]
            event_dic["rad"] = rad
            event_dic["beam"] = bm
            event_dic["gate"] = gt
            event_dic["stime"] = stime
            event_dic["etime"] = etime
            event_dic["len"] = len_df[N]
            event_dic["intt"] = intt_df[N]

            generate_stack_plot(
                o_ts,
                o_fft,
                stack_plot,
                bm,
                gt,
                rad,
                stime,
                etime,
                wave_event,
                I_min,
                N,
                N_min,
            )
    return event_dic


def save_event_info(
    fname="tmp/201501_v_los_igrf.csv",
    stack_plot=False,
    I_min=0.5,
    N_min=420,
    E_method="v_los",
    mag_type="igrf",
    rbsp_log_fn="RBSP_Mode_NH_Radars_Log_201501.txt",
    pcores=8,
):
    """
    save events identified by the narrowband_wave_finder into a csv file
    inputs:
    fname: name of file to be saved
    outputs:
    csv file with event information
    """

    # Create a reader object that reads all the RBSP mode entries
    r = Reader(_filestr="config/logs/" + rbsp_log_fn)
    # Check for all entries
    o = r.check_entries()
    event_dics = {
        "FWHM": [],
        "f_left_ind": [],
        "f_right_ind": [],
        "peak_psd": [],
        "peak_freq": [],
        "peak_ang": [],
        "S_sig": [],
        "S_total": [],
        "I_sig": [],
        "mlat": [],
        "mlon": [],
        "mlt": [],
        "rad": [],
        "beam": [],
        "gate": [],
        "stime": [],
        "etime": [],
        "len": [],
        "intt": [],
        "Erms": [],
        "num_bad_data_rsamp": [],
    }
    pool = Pool(pcores)
    base_events = []

    index = 0
    for row in o.index:
        r.parse_files(select=[row])
        filt_dict = r.file_entries[row].filters

        if filt_dict:
            entry = r.rbsp_logs.iloc[row]
            rad = entry.rad

            for rec in filt_dict["rsamp"]:
                bm = rec["beam"]
                gt = rec["gate"]
                Tn = rec["Tx"]
                stime = rec["tmin"]
                etime = rec["tmax"]

                base_events.append(
                    (
                        index,
                        rad,
                        rbsp_log_fn,
                        row,
                        bm,
                        gt,
                        Tn,
                        stime,
                        etime,
                        E_method,
                        mag_type,
                        stack_plot,
                        I_min,
                        N_min,
                    )
                )
                index += 1
    logger.info(f"Total events to be processed: {len(base_events)}")
    # Running parallel loop
    for proc_event in pool.map(compute_Efield_each_entry, base_events):
        if proc_event["isValid"]:
            for key in list(event_dics.keys()):
                event_dics[key].append(proc_event[key])
    df = pd.DataFrame(event_dics)
    logger.info(f"Processed event details: \n{df.head()}")
    df.to_csv(fname, index=False, header=True, float_format="%g")
    return


def _run_():
    files = glob.glob("config/logs/*.txt")
    files.sort()
    for f in files:
        ud = f.replace(".txt", "").split("_")[-1]
        fname = f"tmp/sd.run.14/analysis/{ud}_v_los_igrf.csv"
        logger.info(f"Running file: {fname}")
        if not os.path.exists(fname):
            t = time.time()
            save_event_info(
                fname=fname,
                stack_plot=False,
                I_min=0.5,
                N_min=420,
                mag_type="igrf",
                E_method="v_los",
                rbsp_log_fn=f.split("/")[-1],
            )
            logger.info(f"Time taken to processes: {time.time() - t}")
    cic = ComputeIonosphereicConductivity()
    cic.compute_conductivities()
    return


if __name__ == "__main__":
    # Running all pre_processed files
    _run_()
