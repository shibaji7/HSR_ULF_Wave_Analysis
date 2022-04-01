#!/usr/bin/env python
"""post_process.py: analysis module for data post processing and analsysis."""

__author__ = ""
__copyright__ = ""
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import os
import sys
import numpy as np
from scipy.signal import find_peaks, peak_widths
sys.path.extend(["py/"])

def narrowband_wave_finder(freqs,psd,fl_s=0.0016,fh_s=0.0067,fl_t=0.000278,fh_t=0.027778):
    """ 
    inputs: 
    freqs: frequencies in Hz (should have the same shape with psd) 
    psd: power spectral density [V**2/Hz] from FFT
    fl_s: lower frequency limit of wave power
    fh_s: higher frequency limit of wave power
    fl_t: lower frequency limit of total power
    fh_t: higher frequency limit of total power
    
    outputs: return None if no event is detected;
    return FWHM, FWHM_left, FWHM_right, peak_psd, peak_freq, S_sig, S_total, I_sig if 
    an event is detected based on the criteria below:
    1. Find the largest peak in 1.6-6.7 mHz; 
    2. Calculate the FWHM of the peak; if FWHM < 2 mHz, calculate the Pc5 inegrated power 
    (S_Pc5) and the spectral power integrated over 0.278-27.8 mHz (S_total); 
    3. Calculate the Pc5 index = S_Pc5/S_total.
    """
    
    peaks, _ = find_peaks(psd)
    
    peak_f = freqs[peaks]
    peak_psd = psd[peaks]
    sig_ind = np.where((peak_f >= fl_s) & (peak_f <= fh_s))
    
    if sig_ind == []:
        return None
    else:
        peak_sig = peaks[sig_ind]
        peak_psd_sig = peak_psd[sig_ind]
        peak_f_sig = peak_f[sig_ind]
        
        sig_max_ind = np.argmax(peak_psd_sig)
        peak_psd_max = peak_psd_sig[sig_max_ind]
        peak_freq = peak_f_sig[sig_max_ind]*1000.
    
        peak_ind = peak_sig[sig_max_ind]
        #print(peak_ind)
    
        results_half = peak_widths(psd, [peak_ind], rel_height=0.5)
        f_res = freqs[1]-freqs[0]
        FWHM = results_half[0][0]*f_res*1000. #mHz
        if  FWHM >= 2:
            return None
        else:
            psd_sig = psd[(freqs >= fl_s) & (freqs <= fh_s)]
            psd_total = psd[(freqs >= fl_t) & (freqs <= fh_t)]
            S_sig = np.trapz(psd_sig, dx=f_res)
            S_total = np.trapz(psd_total, dx=f_res)
            I_sig = S_sig/S_total
            return FWHM,results_half[2][0],results_half[3][0], peak_psd_max,peak_freq,S_sig,S_total,I_sig
