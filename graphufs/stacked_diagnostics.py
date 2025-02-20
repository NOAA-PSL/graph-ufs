"""
Module for computing diagnostics in the loss function, for stacked implementation
"""

from typing import Dict, List, Tuple, Union, Any

import numpy as np
import jax.numpy as jnp
import xarray as xr

# constants
_gravity = 9.80665
_dry_air_specific_gas = 287.05
_water_vapor_specific_gas = 461.5
_vapor_to_gas_ratio_minus1 = water_vapor_specific_gas / dry_air_specific_gas - 1.


def prepare_diagnostic_functions(
    input_meta: Dict[int, Dict[str, Any]],
    output_meta: Dict[int, Dict[str, Any]],
    function_names: Union[List[str], Tuple[str, ...]],
    extra: Dict[str, Any]
) -> Dict[str, Any]:
    """Make a dictionary that has the function handles and inputs
    so that evaluation can happen very fast during the loss function.

    Note that the "metadata" referred to throughout can be acquired from
    graphufs.utils.get_channel_index
    for some reason, I called it "channel_index" then, but I think metadata is a
    more appropriate name.

    The generic template for each function is:

        def my_diagnosed_quantity(input_array, output_array, masks):
            '''
            input_array.shape = [sample, lat, lon, input_channels]
            output_array.shape = [sample, lat, lon, output_channels] # note this is predictions or targets
            masks = dict with "inputs" and "outputs" that gives masks for each field
            returns my_diagnosed_quantity

    Args:
        input_meta (dict): Metadata for input arrays, mapping channel number to varname, level, timeslot etc.
        output_meta (dict): Metadata for prediction/target arrays, mapping channel number to varname, level, timeslot etc.
        function_names (list or tuple): List of diagnostic quantities to compute, that we've precoded.
        extra (dict): Additional parameters needed for certain diagnostics.

    Returns:
        dict: Mapping of function names to their respective arguments.
    """
    masks = {
        "inputs": get_masks(input_meta),
        "outputs": get_masks(output_meta)
    }

    function_mapping = {
        "10m_horizontal_wind_speed": _10m_horizontal_wind_speed,
        "wind_speed": _wind_speed,
        "horizontal_wind_speed": _horizontal_wind_speed,
        "hydrostatic_layer_thickness": _hydrostatic_layer_thickness,
        "hydrostatic_geopotential": _hydrostatic_geopotential,
    }

    n_levels = 1 + np.max([val.get("level", 0) for val in output_meta.values()])
    shapes = {
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


def get_masks(
    meta: Dict[int, Dict[str, Any]]
) -> Dict[str, List[int]]:
    """Get masks for each field from metadata.

    Example:
        returned masks dict will look something like:

        >>> masks = {
            "inputs": {
                "ugrd10m": [0, 1], # e.g. we need two for two initial conditions
                "vgrd10m": [1, 2],
                "tmp":     [3, 4, 5, 6, 7, 8], # e.g. 3 vertical levels * 2 initial conditions
            },
            "outputs": {
                "ugrd10m": [0],
                "vgrd10m": [1],
                "tmp":     [2, 3, 4],
            },
        }

        where the returned list for each variable is the channel index for the array,
        so we can grab the right variables from input and output arrays

    Args:
        meta (dict): Metadata for input or prediction/target arrays, mapping channel number to varname, level, timeslot etc.

    Returns:
        dict: Masks for each field.
    """
    varnames = list(set([cinfo["varname"] for cinfo in meta.values()]))
    masks = {key: [] for key in varnames}
    for channel, cinfo in meta.items():
        masks[cinfo["varname"]].append(channel)
    return masks


def _wind_speed(
    inputs: jnp.ndarray,
    outputs: jnp.ndarray,
    masks: Dict[str, Dict[str, List[int]]],
    extra: Dict[str, Any]
) -> jnp.ndarray:
    """Calculate wind speed from output arrays.

    Args:
        inputs (jnp.ndarray): Input array.
        outputs (jnp.ndarray): Output array.
        masks (dict): Masks for input and output fields.
        extra (dict): Additional parameters needed for certain diagnostics.

    Returns:
        jnp.ndarray: Wind speed.
    """
    u = outputs[..., masks["outputs"]["ugrd"]]
    v = outputs[..., masks["outputs"]["vgrd"]]
    w = outputs[..., masks["outputs"]["dzdt"]]
    return jnp.sqrt(u**2 + v**2 + w**2)


def _horizontal_wind_speed(
    inputs: jnp.ndarray,
    outputs: jnp.ndarray,
    masks: Dict[str, Dict[str, List[int]]],
    extra: Dict[str, Any]
) -> jnp.ndarray:
    """Calculate horizontal wind speed from output arrays.

    Args:
        inputs (jnp.ndarray): Input array.
        outputs (jnp.ndarray): Output array.
        masks (dict): Masks for input and output fields.
        extra (dict): Additional parameters needed for certain diagnostics.

    Returns:
        jnp.ndarray: Horizontal wind speed.
    """
    u = outputs[..., masks["outputs"]["ugrd"]]
    v = outputs[..., masks["outputs"]["vgrd"]]
    return jnp.sqrt(u**2 + v**2)


def _10m_horizontal_wind_speed(
    inputs: jnp.ndarray,
    outputs: jnp.ndarray,
    masks: Dict[str, Dict[str, List[int]]],
    extra: Dict[str, Any]
) -> jnp.ndarray:
    """Calculate 10m horizontal wind speed from output arrays.

    Args:
        inputs (jnp.ndarray): Input array.
        outputs (jnp.ndarray): Output array.
        masks (dict): Masks for input and output fields.
        extra (dict): Additional parameters needed for certain diagnostics.

    Returns:
        jnp.ndarray: 10m horizontal wind speed.
    """
    u = outputs[..., masks["outputs"]["ugrd10m"]]
    v = outputs[..., masks["outputs"]["vgrd10m"]]
    return jnp.sqrt(u**2 + v**2)


def _pressure_interfaces(
    inputs: jnp.ndarray,
    outputs: jnp.ndarray,
    masks: Dict[str, Dict[str, List[int]]],
    extra: Dict[str, Any]
) -> jnp.ndarray:
    """Calculate pressure interfaces from output arrays.

    Args:
        inputs (jnp.ndarray): Input array.
        outputs (jnp.ndarray): Output array.
        masks (dict): Masks for input and output fields.
        extra (dict): Additional parameters needed for certain diagnostics.

    Returns:
        jnp.ndarray: Pressure interfaces.
    """
    pressfc = outputs[..., masks["outputs"]["pressfc"]]
    dtype = pressfc.dtype
    shape = pressfc.shape[:-1] + extra["bk"].shape

    ak = jnp.broadcast_to(extra["ak"].astype(dtype), shape)
    bk = jnp.broadcast_to(extra["bk"].astype(dtype), shape)
    return ak + pressfc * bk


def _hydrostatic_layer_thickness(
    inputs: jnp.ndarray,
    outputs: jnp.ndarray,
    masks: Dict[str, Dict[str, List[int]]],
    extra: Dict[str, Any]
) -> jnp.ndarray:
    """Calculate hydrostatic layer thickness from output arrays.

    Args:
        inputs (jnp.ndarray): Input array.
        outputs (jnp.ndarray): Output array.
        masks (dict): Masks for input and output fields.
        extra (dict): Additional parameters needed for certain diagnostics.

    Returns:
        jnp.ndarray: Hydrostatic layer thickness.
    """
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
    return -_dry_air_specific_gas / _gravity * v["tmp"] * (1. + _vapor_to_gas_ratio_minus1 * v["spfh"]) * dlogp


def _hydrostatic_geopotential(
    inputs: jnp.ndarray,
    outputs: jnp.ndarray,
    masks: Dict[str, Dict[str, List[int]]],
    extra: Dict[str, Any]
) -> jnp.ndarray:
    """Calculate hydrostatic geopotential from output arrays.

    Args:
        inputs (jnp.ndarray): Input array.
        outputs (jnp.ndarray): Output array.
        masks (dict): Masks for input and output fields.
        extra (dict): Additional parameters needed for certain diagnostics.

    Returns:
        jnp.ndarray: Hydrostatic geopotential.
    """
    # handle transforms
    hgtsfc_static = outputs[..., masks["inputs"]["hgtsfc_static"]]
    layer_thickness = _hydrostatic_layer_thickness(inputs, outputs, masks, extra)

    # geopotential at the surface
    phi0 = _gravity * hgtsfc_static

    # and in 3D
    dz = _gravity * jnp.abs(layer_thickness)
    phii = jnp.concatenate([dz, phi0], axis=-1)
    phii = phii[..., ::-1]
    phii = jnp.cumsum(phii, axis=-1)
    phii = phii[..., ::-1]

    # now grab all interfaces except surface, and subtract half
    geopotential = phii[..., :-1] - 0.5 * dz
    return geopotential
