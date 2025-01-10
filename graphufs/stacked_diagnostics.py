"""Module for computing diagnostics in the loss function, for stacked implementation

TODO:
    * Deal with any input/output transformations
"""

import numpy as np
import jax.numpy as jnp
import jax.debug
import xarray as xr


def prepare_diagnostic_functions(input_meta, output_meta, function_names, extra):
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
        "hydrostatic_layer_thickness": _hydrostatic_layer_thickness,
        "hydrostatic_geopotential": _hydrostatic_geopotential,
    }

    n_levels = 1 + np.max([val.get("level", 0) for val in output_meta.values()])
    shapes = {
        "10m_wind_speed": 1,
        "10m_horizontal_wind_speed": 1,
        "wind_speed": n_levels,
        "horizontal_wind_speed": n_levels,
        "hydrostatic_layer_thickness": n_levels,
        "hydrostatic_geopotential": n_levels,
    }

    # check for recognized names
    recognized_names = list(function_mapping.keys())
    for name in function_names:
        assert name in recognized_names, \
            f"{__name__}.prepare_diagnostic_functions: did not recognize {name}, has to be one of {recognized_names}"

    # check extra
    if any(x in ("pressure_interfaces", "hydrostatic_layer_thickness", "hydrostatic_geopotential") for x in function_names):
        assert "ak" in extra
        assert extra["ak"] is not None
        assert "bk" in extra
        assert extra["bk"] is not None

    for key in ["ak", "bk"]:
        if key in extra and isinstance(extra[key], xr.DataArray):
            extra[key] = extra[key].values

    for key in ["input_transforms", "output_transforms"]:
        if key in extra:
            extra[key] = dict() if extra[key] is None else extra[key]

    # filter to only return what user wants
    return {
        "functions": {key: val for key, val in function_mapping.items() if key in function_names},
        "masks": masks,
        "shapes": {key: val for key, val in shapes.items() if key in function_names},
        "extra": extra,
    }

def get_masks(meta):
    varnames = list(set([cinfo["varname"] for cinfo in meta.values()]))
    masks = {key: [] for key in varnames}
    for channel, cinfo in meta.items():
        masks[cinfo["varname"]].append(channel)
    return masks


def _wind_speed(inputs, outputs, masks, extra):
    u = outputs[..., masks["outputs"]["ugrd"]]
    v = outputs[..., masks["outputs"]["vgrd"]]
    w = outputs[..., masks["outputs"]["dzdt"]]
    return jnp.sqrt(u**2 + v**2 + w**2)

def _horizontal_wind_speed(inputs, outputs, masks, extra):
    u = outputs[..., masks["outputs"]["ugrd"]]
    v = outputs[..., masks["outputs"]["vgrd"]]
    return jnp.sqrt(u**2 + v**2)

def _pressure_interfaces(inputs, outputs, masks, extra):
    pressfc = outputs[..., masks["outputs"]["pressfc"]]
    dtype = pressfc.dtype
    shape = pressfc.shape[:-1] + extra["bk"].shape

    ak = jnp.broadcast_to(extra["ak"].astype(dtype), shape)
    bk = jnp.broadcast_to(extra["bk"].astype(dtype), shape)
    return ak + pressfc*bk

def _hydrostatic_layer_thickness(inputs, outputs, masks, extra):
    # constants
    g = 9.80665
    Rd = 287.05
    Rv = 461.5
    q_min = 1e-10
    z_vir = Rv / Rd - 1.

    # handle transforms
    v = {}
    for key in ["spfh", "tmp"]:
        var = outputs[..., masks["outputs"][key]]
        if key in extra["output_transforms"]:
            v[key] = extra["output_transforms"][key](var)
        else:
            v[key] = var

    # pressure interfaces
    prsi = _pressure_interfaces(inputs, outputs, masks, extra)

    # calc dlogp
    logp = jnp.log(prsi)
    dlogp = logp[..., 1:] - logp[..., :-1]
    return -Rd / g * v["tmp"] * (1. + z_vir*v["spfh"])*dlogp


def _hydrostatic_geopotential(inputs, outputs, masks, extra):
    # constants
    g = 9.80665
    Rd = 287.05
    Rv = 461.5
    q_min = 1e-10
    z_vir = Rv / Rd - 1.

    # handle transforms
    hgtsfc_static = outputs[..., masks["inputs"]["hgtsfc_static"]]
    layer_thickness = _hydrostatic_layer_thickness(inputs, outputs, masks, extra)

    # geopotential at the surface
    phi0 = g * hgtsfc_static

    # and in 3D
    dz = g * jnp.abs(layer_thickness)
    phii = jnp.concatenate([dz, phi0], axis=-1)
    phii = phii[..., ::-1]
    phii = jnp.cumsum(phii, axis=-1)
    phii = phii[..., ::-1]

    # now grab all interfaces except surface, and subtract half
    geopotential = phii[..., :-1] - 0.5 * dz
    return geopotential
