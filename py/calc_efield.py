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
        return
    
    def get_magnetic_field_info(self):
        """
        """
        return

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