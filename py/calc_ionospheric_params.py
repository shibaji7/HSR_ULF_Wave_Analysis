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

import os
import sys

sys.path.extend(["py/"])
import datetime as dt
import glob
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyIGRF
import ray
import utils as utils
from loguru import logger
from ovationpyme.ovation_prime import ConductanceEstimator
import swifter


@ray.remote
def compute_conductivity_profile(folder, t, hemi, fluxtypes, auroral=True, solar=True):
    """
    Compute conductivity of each event.
    """
    file = os.path.join(
        folder,
        "op_dt_hemi.{}_cond_{}.csv".format(hemi, t.strftime("%Y%m%d_%H%M%S")),
    )
    logger.info(f"Populating {file}")
    if not os.path.exists(file):
        estimator = ConductanceEstimator(fluxtypes=fluxtypes)
        mlatgrid, mltgrid, pedgrid, hallgrid = estimator.get_conductance(
            t, hemi=hemi, auroral=auroral, solar=solar
        )
        o = pd.DataFrame(
            {
                "mlatgrid": mlatgrid.flatten(),
                "mltgrid": mltgrid.flatten(),
                "pedgrid": pedgrid.flatten(),
                "hallgrid": hallgrid.flatten(),
            }
        )
        o.to_csv(file, index=False, header=True, float_format="%g")
    else:
        o = pd.read_csv(file)
    return {"time": t, "op": o}


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
    r, theta_lat, phi_lon = location[0], location[1], location[2]
    FACT = 180.0 / np.pi
    B = {}
    if mag_type == "igrf":
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
        r += Re
        theta_lat = 90 - theta_lat
        B["r"] = 2 * B0 * (Re / r) ** 3 * np.cos(np.deg2rad(theta_lat))
        B["p"] = 0.0
        B["t"] = B0 * (Re / r) ** 3 * np.sin(np.deg2rad(theta_lat))
        B["b"] = np.sqrt(B["r"] ** 2 + B["t"] ** 2)
        B["h"] = np.sqrt(B["p"] ** 2 + B["t"] ** 2)
        B["i"] = FACT * np.arctan2(B["r"], B["h"])
        B["d"] = FACT * np.arctan2(B["p"], B["t"])
    return B


class EfieldMethods(object):
    """
    Class holds all the functions to
    compute electric fields.
    """

    def __init__(self):
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
        return compute_B_field(
            location, date, self.methods["mag_type"], self.Re, self.B0
        )

    def compute_from_vlos(self, row):
        """
        Compute E=-(V_los X B).
        Assumption:
        -----------
        1. Reflection is _|_ to B field.
        2. -(V X B) = [-|V|.|B| sin(t)]_t=90 = -|V|.|B|
        """
        location, date = [300, row["glat"], row["glon"]], row["time"]
        Bfield = self.get_magnetic_field_info(location, date)
        row["E_vlos"] = -(row["v"] * Bfield["b"])
        return row

    def compute_from_vect(self, row):
        """
        Compute E=-(V_los X B).
        Assumption:
        -----------
        1. Decomposing to horizontal component, valid for a range cell.
        2. Northward and eastward directed unit vectors n' and e' respectively.
        2. -(V X B) = -( Vn.Be n' - Ve.Bn e') vector multiplication.
        """
        location, date = [300, row["glat"], row["glon"]], row["time"]
        Bfield = self.get_magnetic_field_info(location, date)
        azm = self.azms[row["bmnum"], row["slist"]]
        Vn, Ve = row["v"] * np.cos(np.rad2deg(azm)), row["v"] * np.sin(np.rad2deg(azm))
        row["E_vector_n"], row["E_vector_e"] = -Vn * Bfield["p"], Ve * Bfield["t"]
        row["E_vector"] = np.sqrt(row["E_vector_n"] ** 2 + row["E_vector_e"] ** 2)
        return row


class ComputeIonosphereicEField(EfieldMethods):
    """
    This class is dedicated to cunsume SD velocity data,
    and compute following ionospheric parameter based on various
    magnetic field profiles.
        1. E-field
    """

    def __init__(
        self,
        rad,
        df,
        methods={
            "e_field": "vlos",
            "mag_type": "igrf",
        },
        Re=6371.0,
        B0=3.12e-5,
    ):
        """
        Parameters:
        ----------
        rad: Radar code
        df: Dataframe containing velocity
        methods: All the methods for various model
        Re: Earth's radius in km
        B0: Magnetic field on the surface of the earth near equator
        """
        self.rad = rad
        self.df = df
        self.methods = methods
        self.Re = Re
        self.B0 = B0
        self.glats, self.glons, self.azms = utils.compute_field_of_view_parameters(rad)
        return

    def compute_efield(self):
        """
        Compute E=-(V X B).
        """
        # Select function based on
        if self.methods["e_field"] == "v_vector":
            func = self.compute_from_vect
        elif self.methods["e_field"] == "v_los":
            func = self.compute_from_vlos
        else:
            func = self.compute_from_vlos
        self.df = self.df.swifter.apply(func, axis=1)
        return


class ComputeIonosphereicConductivity(EfieldMethods):
    """
    This class is dedicated to cunsume SD velocity data,
    and compute following ionospheric parameter based on various
    magnetic field profiles.
        1. Conductivity
    """

    def __init__(
        self,
        files=[],
        hemi="N",
        flux_types=["diff", "mono", "wave"],
    ):
        """
        Parameters:
        ----------
        file_location: Location of the event files
        hemi: Hemisphere
        flux_types: List of all flux types in Ovation Prime
        """
        # Load Config File
        with open("config/params.json") as f:
            self.conf = json.load(f, object_hook=lambda x: SimpleNamespace(**x))
        self.files = (
            files
            if len(files) > 0
            else glob.glob(
                self.conf.files.analysis.format(run_id=self.conf.run_id) + "*.csv"
            )
        )
        self.hemi = hemi
        self.flux_types = flux_types
        # Ovationpime folder name
        self.op_cond_file_location = os.path.join(
            self.conf.files.analysis.format(run_id=self.conf.run_id), "op"
        )
        os.makedirs(self.op_cond_file_location, exist_ok=True)
        # Initialize Ray
        ray.init(num_cpus=4)
        return

    def compute_conductivities(self):
        """
        Compute conducttivities for all events
        """
        for f in self.files:
            records = pd.read_csv(f, parse_dates=["stime", "etime"])
            records["bin_time"] = records.stime + dt.timedelta(
                minutes=60 * self.conf.filter.hour_win / 2
            )
            tmp = pd.DataFrame(
                list(zip(records.bin_time.unique())), columns=["bin_time"]
            )
            tmp.bin_time = tmp.bin_time.apply(lambda t: t.to_pydatetime())
            logger.info(f"Total number of records {len(records)}")
            logger.info(
                f"Total number of unique times {len(records.bin_time.unique())}"
            )
            # Run Parallel Ray to compute Conductivities
            conductivities = []
            for t in tmp.bin_time:
                conductivities.append(
                    compute_conductivity_profile.remote(
                        self.op_cond_file_location, t, self.hemi, self.flux_types
                    )
                )
            oplist = ray.get(conductivities)
            del conductivities
            self.conductivities = {}
            for l in oplist:
                self.conductivities[l["time"]] = l["op"]
            logger.info(
                f"Computed number of unique events {len(self.conductivities.keys())}"
            )
            # TODO: Run Parallel to populate locaion wise conductivity
            records = records.apply(self.populate_conductivity, axis=1)
            logger.info(f"Total number of processed records {len(records)}")
            records.to_csv(f, index=False, header=True, float_format="%g")
        return

    def populate_conductivity(self, row):
        """
        Populate locaion wise conductivity
        """
        # Find the closest cell to locate conductance
        time = row.bin_time
        mlat = row.mlat
        mlt = row.mlt
        o = self.conductivities[time]
        mlatgrid = np.array(o.mlatgrid)
        mltgrid = np.array(o.mltgrid)
        pedgrid = np.array(o.pedgrid)
        hallgrid = np.array(o.hallgrid)
        dist_tmp = np.sqrt((mlat - mlatgrid) ** 2 + (15 * (mlt - mltgrid)) ** 2)
        cell_ind = np.argmin(dist_tmp)
        row["op_mlat"] = mlatgrid[cell_ind]
        row["op_mlt"] = mltgrid[cell_ind]
        row["op_ped"] = pedgrid[cell_ind]
        row["op_hall"] = hallgrid[cell_ind]
        return row


if __name__ == "__main__":
    cic = ComputeIonosphereicConductivity()
    cic.compute_conductivities()
