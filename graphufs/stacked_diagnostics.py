"""Module for computing diagnostics in the loss function, for stacked implementation

TODO:
    * Deal with any input/output transformations
"""

import numpy as np
import jax.numpy as jnp


def prepare_diagnostic_functions(input_meta, output_meta, function_names):
    """Make a dictionary that has the function handles and inputs
    so that evaluation can happen very fast during the loss function.

    The generic template for each function is:

        def my_diagnosed_quantity(input_array, output_array, masks):
            '''
            input_array.shape = [sample, lat, lon, input_channels]
            output_array.shape = [sample, lat, lon, output_channels] # note this is predictions or targets
            masks = dict with "inputs" and "outputs" that gives masks for each field
            returns my_diagnosed_quantity

    Args:
        input_meta, output_meta (dict): metadata for input or prediction/target arrays, mapping channel number to varname, level, timeslot etc
            see graphufs.utils.get_channel_index
        function_names (list, tuple, etc): with diagnostic quantities to compute, that we've precoded

    Returns:
        function_mapping (dict): with {"function_name": (*args)}
    """
    masks = {
        "inputs": get_masks(input_meta),
        "outputs": get_masks(output_meta)
    }

    function_mapping = {
        "wind_speed": _wind_speed,
        "horizontal_wind_speed": _horizontal_wind_speed,
    }

    n_levels = 1 + np.max([val.get("level", 0) for val in output_meta.values()])
    shapes = {
        "wind_speed": n_levels,
        "horizontal_wind_speed": n_levels,
    }
    recognized_names = list(function_mapping.keys())
    for name in function_names:
        assert name in recognized_names, \
            f"{__name__}.prepare_diagnostic_functions: did not recognize {name}, has to be one of {recognized_names}"
    # filter to only return what user wants
    function_mapping = {key: val for key, val in function_mapping.items() if key in function_names}
    shapes = {key: val for key, val in shapes.items() if key in function_names}
    return masks, function_mapping, shapes


def get_masks(meta):
    varnames = list(set([cinfo["varname"] for cinfo in meta.values()]))
    masks = {key: [] for key in varnames}
    for channel, cinfo in meta.items():
        masks[cinfo["varname"]].append(channel)
    return masks


def _wind_speed(inputs, outputs, masks):
    u = outputs[..., masks["outputs"]["ugrd"]]
    v = outputs[..., masks["outputs"]["vgrd"]]
    w = outputs[..., masks["outputs"]["dzdt"]]
    return jnp.sqrt(u**2 + v**2 + w**2)

def _horizontal_wind_speed(inputs, outputs, masks):
    u = outputs[..., masks["outputs"]["ugrd"]]
    v = outputs[..., masks["outputs"]["vgrd"]]
    return jnp.sqrt(u**2 + v**2)
