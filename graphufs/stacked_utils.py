import itertools
import xarray as xr
from .datasets import Dataset

def product_dict(**kwargs):
    keys = kwargs.keys()
    for instance in itertools.product(*kwargs.values()):
        yield dict(zip(keys, instance))

def get_channel_index(xds, preserved_dims=("batch", "lat", "lon")):
    """For StackedGraphCast, we need to add prediction to last timestamp from the initial conditions.
    To do this, we need a mapping from channel indices to the variables contained in that channel
    with all the collapsed dimensions

    Example:
        >>> inputs, targets, forcings = ...
        >>> get_channel_index(inputs)
        {0: {'varname': 'day_progress_cos', 'time': 0},
         1: {'varname': 'day_progress_cos', 'time': 1},
         2: {'varname': 'day_progress_sin', 'time': 0},
         3: {'varname': 'day_progress_sin', 'time': 1},
         4: {'varname': 'pressfc', 'time': 0},
         5: {'varname': 'pressfc', 'time': 1},
         6: {'varname': 'tmp', 'time': 0, 'level': 0},
         7: {'varname': 'tmp', 'time': 0, 'level': 1},
         8: {'varname': 'tmp', 'time': 0, 'level': 2},
         9: {'varname': 'tmp', 'time': 1, 'level': 0},
         10: {'varname': 'tmp', 'time': 1, 'level': 1},
         11: {'varname': 'tmp', 'time': 1, 'level': 2},
         12: {'varname': 'ugrd10m', 'time': 0},
         13: {'varname': 'ugrd10m', 'time': 1},
         14: {'varname': 'vgrd10m', 'time': 0},
         15: {'varname': 'vgrd10m', 'time': 1},
         16: {'varname': 'year_progress_cos', 'time': 0},
         17: {'varname': 'year_progress_cos', 'time': 1},
         18: {'varname': 'year_progress_sin', 'time': 0},
         19: {'varname': 'year_progress_sin', 'time': 1}}

    Inputs:
        xds (xarray.Dataset): e.g. inputs, targets
        preserved_dims (tuple, optional): same as in graphcast.model_utils.dataset_to_stacked

    Returns:
        mapping (dict): with keys = logical indices 0 -> n_channels-1, and values = a dict
            with "varname" and each dimension, where value corresponds to logical position of that dimension
    """

    mapping = {}
    channel = 0
    for varname in sorted(xds.data_vars):
        stacked_dims = list(set(xds[varname].dims) - set(preserved_dims))
        stacked_dims = sorted(
            stacked_dims,
            key=lambda x: list(xds[varname].dims).index(x),
        )
        stacked_dim_dict = {
            k: list(range(len(xds[k])))
            for k in stacked_dims
        }

        for i, selection in enumerate(product_dict(**stacked_dim_dict), start=channel):
            mapping[i] = {"varname": varname, **selection}
        channel = i+1
    return mapping

def convert_loss_channel2var(Emulator, loss2d):
    """Convert loss by channel to a dataset with loss separated by variable

    Args:
        Emulator: note that it has to be the training emulator
        loss2d (xr.DataArray): second axis just has to be "channel"

    Returns:
        xds (xr.Dataset): with each variable indicating it's loss
    """
    em = Emulator()
    tds = Dataset(em, mode="training")
    xinputs, xtargets, _ = tds.get_xarrays(0)

    tmeta_inp = get_channel_index(xinputs)
    tmeta = get_channel_index(xtargets)
    varloss = {}
    for cidx in loss2d.channel.values:
        mymeta = tmeta[cidx]
        varname = mymeta["varname"]
        this_loss = loss2d.sel(channel=cidx, drop=True)
        this_loss.name = varname
        if "level" in mymeta:
            levelval = xtargets.level.isel(level=mymeta["level"]).values
            this_loss = this_loss.expand_dims({"level": [levelval]})
            if varname not in varloss:
                varloss[varname] = [this_loss]
            else:
                varloss[varname].append(this_loss)
        elif "z_l" in mymeta:
            ocnlevelval = xtargets.z_l.isel(z_l=mymeta["z_l"]).values
            this_loss = this_loss.expand_dims({"z_l": [ocnlevelval]})
            if varname not in varloss:
                varloss[varname] = [this_loss]
            else:
                varloss[varname].append(this_loss)
        else:
            varloss[varname] = this_loss

    for key in xtargets.data_vars:
        if "level" in xtargets[key].dims:
            varloss[key] = xr.concat(varloss[key], dim="level")
        elif "z_l" in xtargets[key].dims:
            varloss[key] = xr.concat(varloss[key], dim="z_l")
    return xr.Dataset(varloss)
