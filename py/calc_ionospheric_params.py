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

import numpy as np
import pyIGRF
import swifter
import sys
from loguru import logger

sys.path.extend(["py/"])


def compute_B_field(location, date, mag_type="igrf", Re=6371.0, B0=3.12e-5):
    """
    Parameters:
    ----------
    location: Geographic location [h, lat, lon]
    date: Datetime of the event
    Re: Earth's radius in km
    mag_type: Magnetic model type
    B0: Magnetic field on the surface of the earth near equator
    """
    FACT = 180.0 / np.pi
    B = {}
    if mag_type == "igrf":
        r, theta_lat, phi_lon = location[0], location[1], location[2]
        B["d"], B["i"], B["h"], B["t"], B["p"], B["r"], B["b"] = pyIGRF.igrf_value(
            theta_lat, phi_lon, r, date.year
        )
        B["h"], B["t"], B["p"], B["r"], B["b"] = (
            B["h"] * 1e-9,
            B["t"] * 1e-9,
            B["p"] * 1e-9,
            B["r"] * 1e-9,
            B["b"] * 1e-9,
        )
    elif mag_type == "dipole":
        r, theta_lat = location[0] + Re, location[1]
        B["r"] = -2 * B0 * (Re / r) ** 3 * np.cos(np.deg2rad(theta_lat))
        B["p"] = 0.0
        B["t"] = -B0 * (Re / r) ** 3 * np.sin(np.deg2rad(theta_lat))
        B["b"] = np.sqrt(B["r"] ** 2 + B["t"] ** 2)
        B["h"] = np.sqrt(B["p"] ** 2 + B["t"] ** 2)
        B["i"] = FACT * np.arctan2(B["r"], B["h"])
        B["d"] = FACT * np.arctan2(B["p"], B["t"])
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

    def __init__(self, df, vel_key="vlos", Re=6371.0, mag_type="igrf", B0=3.12 * 1e-5):
        """
        Parameters:
        ----------
        df: Dataframe containing velocity
        vel_key: Velocity key to compute E
        Re: Earth's radius in km
        mag_type: Magnetic model type
        B0: Magnetic field on the surface of the earth near equator
        """
        self.df = df
        self.vel_key = vel_key
        self.Re = Re
        self.mag_type = mag_type
        self.B0 = B0
        return

    def get_magnetic_field_info(self, location, date):
        """
        Get magnetic field information based on location and
        magnetic model type [igrf/dipole].
        Parameters:
        -----------
        location: Geographic location [h, lat, lon]
        date: Datetime of the event
        """
        return compute_B_field(location, date, self.mag_type, self.Re, self.B0)

    def compute_from_vlos(self, row):
        """
        Compute E=-(V_los X B).
        """
        location, date = [300, row["glat"], row["glon"]], row["time"]
        Bfield = self.get_magnetic_field_info(location, date)
        row["E_computed_from_vlos"] = -(row["v"] * Bfield["b"])
        return row

    def compute_efield(self):
        """
        Compute E=-(V X B).
        Assumption:
        -----------
        1. Reflection is _|_ to B field.
        2. Thus v_los is _|_ to B field.
        3. -(V X B) = [|V|.|B| sin(t)]_t=90 = |V|.|B|.
        """
        func = (
            self.compute_from_vlos if self.vel_key == "vlos" else self.compute_from_vlos
        )
        self.df = self.df.swifter.apply(func, axis=1)
        return

    def compute_conductivity_profile(self):
        """ """
        return

    def compute_current_density(self):
        """ """
        return

    def compute_jule_heating(self):
        """ """
        return


if __name__ == "__main__":
    import datetime as dt

    b = compute_B_field([0, 0, 0], dt.datetime(2015, 1, 1), mag_type="dipole")
    logger.info(f"B-field from dipole-{b}")
