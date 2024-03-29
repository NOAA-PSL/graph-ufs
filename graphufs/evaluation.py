import numpy as np
import xesmf as xe
import xarray as xr


def convert_wb2_format(gufs, ds, inittimes) -> xr.Dataset:
    """Convert a dataset into weatherbench2 compatible format. Details can be
    found in: https://weatherbench2.readthedocs.io/en/latest/evaluation.html.

    Args:
        gufs: emulator class
        ds (xr.Dataset): the xarray dadatset
        inittimes (xr.Dataset): a dataset that contains "inititime", forecast
                initialization time, and lead time coordinate "time".
    """

    # regrid to the obs coordinates
    ds_obs = xr.open_zarr(
        gufs.wb2_obs_url,
        storage_options={"token": "anon"},
    )
    ds_out = xr.Dataset(
        {
            "lat": (["lat"], ds_obs["latitude"].values),
            "lon": (["lon"], ds_obs["longitude"].values),
        }
    )
    regridder = xe.Regridder(
        ds,
        ds_out,
        "conservative",
        periodic=True,
        reuse_weights=False,
        filename="graphufs_regridder",
    )
    ds_out = regridder(ds)

    # rename variables
    ds_out = ds_out.rename_vars(
        {
            "pressfc": "surface_pressure",
            "tmp": "temperature",
            "ugrd10m": "10m_u_component_of_wind",
            "vgrd10m": "10m_v_component_of_wind",
        }
    )

    # fix pressure levels to match obs
    ds_out["level"] = np.array(list(gufs.pressure_levels), dtype=np.float32)

    # remove batch dimension
    ds_out = ds_out.rename({"optim_step": "o", "time": "t", "batch": "b"})
    ds_out = ds_out.stack(time=("o", "b", "t"), create_index=False)
    ds_out = ds_out.drop_vars(["o", "b", "t"])
    init_times = inittimes["datetime"].values.flatten()
    lead_times = inittimes["time"].values.flatten()
    ds_out = ds_out.assign_coords({"lead_time": lead_times, "time": init_times})
    ds_out = ds_out.rename({"lat": "latitude", "lon": "longitude"})

    # transpose the dimensions, and insert lead_time
    ds_out = ds_out.transpose("time", ..., "longitude", "latitude")
    #for var in ds_out.data_vars:
    #    ds_out[var] = ds_out[var].expand_dims({"lead_time": ds_out.lead_time}, axis=1)

    return ds_out


def compute_rmse_bias(
    predictions: xr.Dataset, targets: xr.Dataset, stats: dict, it: int
) -> None:
    """Compute fast metrics (rmse and bias) between predictions and target.

    Args:
        predictions (xr.Dataset): the forecast
        targets (xr.Dataset): the ground trutch
        stats (dict): dictionary containing statistics
        it (int): current chunk iteration id. This is needed for computing a running average
    """
    diff = predictions - targets
    rmse = np.sqrt((diff ** 2).mean())
    bias = diff.mean()

    # compute running average of rmse and bias
    for var_name, _ in rmse.data_vars.items():
        r = rmse[var_name].values
        b = bias[var_name].values
        if var_name in stats.keys():
            rmse_o = stats[var_name][0]
            bias_o = stats[var_name][1]
            r = rmse_o + (r - rmse_o) / (it + 1)
            b = bias_o + (b - bias_o) / (it + 1)
        stats[var_name] = [r, b]
