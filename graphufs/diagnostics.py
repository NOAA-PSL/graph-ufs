"""Module for computing diagnostics in the loss function

TODO:
    * Deal with any input/output transformations
"""

import numpy as np

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
        "wind_speed": _wind_speed,
        "horizontal_wind_speed": _horizontal_wind_speed,
    }

    recognized_names = list(function_mapping.keys())
    for name in function_names:
        assert name in recognized_names, \
            f"{__name__}.prepare_diagnostic_functions: did not recognize {name}, has to be one of {recognized_names}"
    # filter to only return what user wants
    function_mapping = {key: val for key, val in function_mapping.items() if key in function_names}
    return function_mapping


def _wind_speed(xds):
    u = xds["ugrd"]
    v = xds["vgrd"]
    w = xds["dzdt"]
    return np.sqrt(u**2 + v**2 + w**2)

def _horizontal_wind_speed(xds):
    u = xds["ugrd"]
    v = xds["vgrd"]
    return np.sqrt(u**2 + v**2)
