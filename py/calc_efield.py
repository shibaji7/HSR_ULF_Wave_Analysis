#!/usr/bin/env python

"""
    calc_ionospheric_params.py: module to compute i) e-Field from magnetic 
                                field and SD velocity values, 2) conductivity [profile]
                                from availabel modules, 3) current density [J],
                                and 4) Jule heating. 
"""

__author__ = "Chakraborty, S.; Shi, X."
__copyright__ = ""
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import sys
sys.path.extend(["py/"])
import pyIGRF

def compute_B_field(location, date, kind="igrf", Re=6371.0, B0=3.12*1e-5):
    B = {}
    if kind == "igrf": 
        r, theta_lat, phi_lon = location[0], location[1], location[2]
        B["d"], B["i"], B["h"], B["t"], B["p"],\
            B["r"], B["b"] = pyIGRF.igrf_value(theta_lat, phi_lon, r, date)
    elif kind == "dipole":        
        r, theta_lat = location[0], location[1]
        B["r"] = -2 * B0 * (Re/r)**3 * np.cos(np.deg2rad(theta_lat))
        B["t"] = -B0 * (Re/r)**3 * np.sin(np.deg2rad(theta_lat))
        B["b"] = np.sqrt(B["r"]**2 + B["t"]**2)
    return B

class ComputeIonosphereicProperties(object):
    """
    This class is dedicated to cunsume SD velocity data,
    and compute following ionospheric parameter based on various
    magnetic field and conductivity profiles.
        1. E-field
        2. Conducitivity
        3. Current density
        4. Jule heating
    """
    
    def __init__(self):
        """
        Parameters:
        ----------
        """
        self.Re = 6371.0
        return
    
    def get_magnetic_field_info(self, locations, dates, kind="igrf"):
        """
        Get magnetic field information based on location and 
        magnetic model type [igrf/dipole].
        """
        Bfields = []
        for location, date in zip(locations, dates):
            Bfields.append(compute_B_field(location, date))
        return Bfields

    def compute_efield(self):
        """
        """
        return

    def compute_conducityvity_profile(self):
        """
        """
        return

    def compute_current_density(self):
        """
        """
        return

    def compute_jule_heating(self):
        """
        """
        return