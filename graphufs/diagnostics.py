"""Module for computing diagnostics in the loss function

TODO:
    * Deal with any input/output transformations
"""

import numpy as np

from ufs2arco import Layers2Pressure

def prepare_diagnostic_functions(function_names):
    """Make a dictionary that has the function handles and inputs
    so that evaluation can happen very fast during the loss function.

    The generic template for each function is:

        def my_diagnosed_quantity(dataset):
            '''
            returns my_diagnosed_dataarray

    Args:
        function_names (list, tuple, etc): with diagnostic quantities to compute, that we've precoded

    Returns:
        function_mapping (dict): with {"function_name": (*args)}
    """
    function_mapping = {
        "10m_horizontal_wind_speed": _10m_horizontal_wind_speed,
        "wind_speed": _wind_speed,
        "horizontal_wind_speed": _horizontal_wind_speed,
        "hydrostatic_layer_thickness": _hydrostatic_layer_thickness,
        "hydrostatic_geopotential": _hydrostatic_geopotential,
    }

    required_variables = {
        "10m_horizontal_wind_speed": ("ugrd10m", "vgrd10m"),
        "wind_speed": ("ugrd", "vgrd", "dzdt"),
        "horizontal_wind_speed": ("ugrd", "vgrd"),
        "hydrostatic_layer_thickness": ("pressfc", "tmp", "spfh"), # ak, bk are already present ... should be at least
        "hydrostatic_geopotential": ("pressfc", "tmp", "spfh", "hgtsfc_static"), # ak, bk are already present ... should be at least
    }

    recognized_names = list(function_mapping.keys())
    for name in function_names:
        assert name in recognized_names, \
            f"{__name__}.prepare_diagnostic_functions: did not recognize {name}, has to be one of {recognized_names}"
    # filter to only return what user wants
    return {
        "functions": {key: val for key, val in function_mapping.items() if key in function_names},
        "required_variables": {key: val for key, val in required_variables.items() if key in function_names},
    }


def _wind_speed(xds):
    u = xds["ugrd"]
    v = xds["vgrd"]
    w = xds["dzdt"]
    return np.sqrt(u**2 + v**2 + w**2)

def _horizontal_wind_speed(xds):
    u = xds["ugrd"]
    v = xds["vgrd"]
    return np.sqrt(u**2 + v**2)

def _10m_horizontal_wind_speed(xds):
    u = xds["ugrd10m"]
    v = xds["vgrd10m"]
    return np.sqrt(u**2 + v**2)

def _get_l2p(xds):
    kw = {}
    if "ak" in xds and "bk" in xds:
        kw["ak"] = xds["ak"]
        kw["bk"] = xds["bk"]
    if "pfull" not in xds:
        kw["level_name"] = "level"
    return Layers2Pressure(**kw)


def _pressure_interfaces(xds):
    lp = _get_l2p(xds)
    return lp.calc_pressure_interfaces(xds["pressfc"])

def _hydrostatic_layer_thickness(xds):
    lp = _get_l2p(xds)
    # TODO: potentially remap spfh
    return lp.calc_delz(xds["pressfc"], xds["tmp"], xds["spfh"])

def _hydrostatic_geopotential(xds):
    lp = _get_l2p(xds)
    # TODO: potentially remap spfh
    delz = lp.calc_delz(xds["pressfc"], xds["tmp"], xds["spfh"])
    return lp.calc_geopotential(xds["hgtsfc_static"], delz)
