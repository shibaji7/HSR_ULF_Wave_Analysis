#!/usr/bin/env python

"""dump_data.py: dump data downloads all the ascii type files from different FTP repositories."""

__author__ = "Chakraborty, S."
__copyright__ = ""
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import os
import requests
import numpy as np
import datetime as dt
from calendar import monthrange

BASE_LOCATION = "tmp/data/"

def ini():
    os.makedirs( BASE_LOCATION + "omni/raw/", exist_ok=True )
    os.makedirs( BASE_LOCATION + "geomag/symh/raw/", exist_ok=True )
    os.makedirs( BASE_LOCATION + "geomag/Kp/raw/", exist_ok=True )
    return

##############################################################################################
## Download 1m resolution solar wind omni data from NASA GSFC ftp server
##############################################################################################
def download_omni_dataset():
    base_uri = "https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/monthly_1min/omni_min%d%02d.asc"
    base_storage = BASE_LOCATION + "omni/raw/%d%02d.asc"
    for year in range(2012,2018):
        for month in range(1,13):
            f_path = base_storage%(year,month)
            if not os.path.exists(f_path):
                url = base_uri%(year,month)
                response = requests.get(url)
                if response.status_code==200:
                    with open(f_path,"w") as f:
                        f.write(response.text)
    return


##############################################################################################
## Download 1m resolution ASY / SYM data from WDC Kyoto ftp server
##############################################################################################
def download_symH_dataset():
    base_uri = "http://wdc.kugi.kyoto-u.ac.jp/cgi-bin/aeasy-cgi?Tens=%s&Year=%s&Month=%s&Day_Tens=0&Days=1&Hour=00&min=00&Dur_Day_Tens=%s&Dur_Day=%s&Dur_Hour=00&Dur_Min=00&Image+Type=GIF&COLOR=COLOR&AE+Sensitivity=0&ASY/SYM++Sensitivity=0&Output=ASY&Out+format=IAGA2002&Email=shibaji7@vt.edu"
    base_storage = BASE_LOCATION + "geomag/symh/raw/%d%02d.asc"
    for year in range(2012, 2017):
        for month in range(1,13):
            ytens = str(int(year/10))
            yrs = str(np.mod(year,10))
            mnt = "%02d"%month
            _,dur = monthrange(year, month)
            durtens = "%02d"%(int(dur/10))
            dur = str(np.mod(dur,10))
            url = base_uri%(ytens,yrs,mnt,durtens,dur)
            f_path = base_storage%(year,month)
            if not os.path.exists(f_path):
                cmd = "wget -O '%s' '%s'"%(f_path, url)
                os.system(cmd)
    return

##############################################################################################
## Download 3h resolution Kp data from WDC Kyoto ftp server
##############################################################################################
def download_Kp_dataset():
    if not os.path.exists("tmp/data/geomag/Kp/raw/Kp.asc"):
        cmd = "curl -o \"tmp/data/geomag/Kp/raw/Kp.asc\" --data \"SCent=20&STens=1&SYear=2&From=1&ECent=20&ETens=1&EYear=8&To=1&Email=shibaji7%40vt.edu\" http://wdc.kugi.kyoto-u.ac.jp/cgi-bin/kp-cgi"
        os.system(cmd)
    return
 
if __name__ == "__main__":
    ini()
    download_Kp_dataset()
    #download_symH_dataset()
    download_omni_dataset()    
    pass