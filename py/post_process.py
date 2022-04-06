#!/usr/bin/env python
"""post_process.py: analysis module for data post processing and analsysis."""

__author__ = "Shi, X."
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
from scipy import stats
from scipy.signal import find_peaks, peak_widths
sys.path.extend(["py/"])
from py.reader import Reader

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
    
    if np.size(sig_ind) == 0:
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

        
def despike_mad(data,num=6, scale='normal'):
    """ 
    despike a time series based on median absolute deviation
    inputs: 
    data: data to be despiked
    num: number of median absoute deviations for despiking, default is 6
    
    outputs: 
    good_data: good data with outliers removed
    good_ind: index of good data in the original data
    """
    mad = stats.median_abs_deviation(data, scale=scale)
    median_data = np.median(data)

    good_ind = (data <= median_data + mad*num) & (data >= median_data - mad*num)
    good_data = data[good_ind]
    
    return good_data, good_ind


def save_event_info(fname='wave_events_info.csv',stack_plot=False,I_min=0.5,N_min=420):
    """ 
    save events identified by the narrowband_wave_finder into a csv file
    
    inputs: 
    fname: name of file to be saved
    
    outputs: 
    csv file with event information
    """
    
    # Create a reader object that reads all the RBSP mode entries
    r = Reader()
    # Check for entries 
    #o = r.check_entries(rad="bks")
    o = r.check_entries()
    event_dic = {'FWHM': [],'f_left_ind':[],'f_right_ind':[],'peak_psd':[],
                 'peak_freq':[],'S_sig':[],'S_total':[],'I_sig':[],'mlat':[],
                 'mlon':[],'mlt':[],'rad':[],'beam':[],'gate':[],'stime':[],
                 'etime':[],'len':[],'intt':[]}

    for row in o.index:
        r.parse_files(select=[row])
        filt_dict = r.file_entries[row].filters
        if filt_dict:
            entry = r.rbsp_logs.iloc[row]
            rad = entry.rad
            #print(entry.rad)
            for rec in filt_dict['rsamp']:
                bm = rec['beam']
                gt = rec['gate']
                Tn = rec['Tx']
                stime = rec['tmin']
                etime = rec['tmax']
                o_fft = r.file_entries[row].get_data(p="fft", beams=[bm], gates=[gt], Tx=[Tn])
                
                if ((not o_fft.empty) and (not o_fft['amp'].isnull().values.any())):
                    o_ts = r.file_entries[row].get_data(p="rsamp", beams=[bm], gates=[gt], Tx=[Tn])
                
                    #Find narrowband Pc5 event
                    freqs = np.array(o_fft.frq)
                    psd = np.array(18./100*o_fft.amp**2)
                    wave_event = narrowband_wave_finder(freqs,psd)
                    
                    if wave_event:
                        event_dic['FWHM'].append(wave_event[0])
                        event_dic['f_left_ind'].append(wave_event[1])
                        event_dic['f_right_ind'].append(wave_event[2])
                        event_dic['peak_psd'].append(wave_event[3])
                        event_dic['peak_freq'].append(wave_event[4])
                        event_dic['S_sig'].append(wave_event[5])
                        event_dic['S_total'].append(wave_event[6])
                        event_dic['I_sig'].append(wave_event[7])
                    
                        N = int(len(o_ts.v)/2)
                        mlat_df = np.array(o_ts['mlat'])
                        mlon_df = np.array(o_ts['mlon'])
                        mlt_df = np.array(o_ts['mlt'])
                        len_df = np.array(o_ts['len'])
                        intt_df = np.array(o_ts['intt'])
                        event_dic['mlat'].append(mlat_df[N])
                        event_dic['mlon'].append(mlon_df[N])
                        event_dic['mlt'].append(mlt_df[N])
                        event_dic['rad'].append(entry.rad)
                        event_dic['beam'].append(bm)
                        event_dic['gate'].append(gt)
                        event_dic['stime'].append(stime)
                        event_dic['etime'].append(etime)
                        event_dic['len'].append(len_df[N])
                        event_dic['intt'].append(intt_df[N])
                        
                        # Stack plots
                        if ((stack_plot == True) and (wave_event[7] > I_min) and (len_df[N] * intt_df[N] >= N_min)):
                            frames = [
                                {"p": "rsamp", "df": o_ts, "title": rad+", Beam "+ str(bm)+", Gate "+str(gt)+", RSamp"},
                                {"p": "fft", "df": o_fft, "title": rad+", Beam "+ str(bm)+", Gate "+str(gt)+", FFT"}, ]
                            
                            fig_name = "hpPc5_event_{rad}_{stime}_{etime}_b{beam}_g{gate}.png".format(rad=entry.rad,
                                                                                                stime=stime.strftime("%Y%m%d-%H%M"),
                                                                                                etime=etime.strftime("%Y%m%d-%H%M"),
                                                                                                beam=bm,gate=gt)
                            r.generate_stacks(frames, fig_name)
                
    df = pd.DataFrame(event_dic)
    df.to_csv(fname, index=False)
