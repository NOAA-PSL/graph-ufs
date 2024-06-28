"""TODO:

0. Have raw predictions and targets ...
...
just push this through WB2?
...
To compare to ERA5...
1. Compute true pressure level from pk = bk + ak ps
2. Convert ERA5 to these pressure levels or the other way around
3. Regrid to 1.5 degree for WB2
4. Compute climatology for ACC?
"""
from functools import partial
import logging
import os
import sys
import jax
import haiku as hk
import numpy as np
import dask
import xarray as xr
from tqdm import tqdm
import xesmf
import cf_xarray as cfxr

from graphcast import rollout

from graphufs import init_devices, construct_wrapped_graphcast
from graphufs.utils import get_last_input_mapping
from graphufs.batchloader import ExpandedBatchLoader
from graphufs.datasets import Dataset
from graphufs.stacked_training import init_model, optimize

from ufs2arco import Timer
from p1stacked import P1Emulator

from ufs2arco.regrid.gaussian_grid import gaussian_latitudes

def get_bounds(xds, is_gaussian=False):
    xds = xds.cf.add_bounds(["lat", "lon"])

    for key in ["lat", "lon"]:
        corners = cfxr.bounds_to_vertices(
            bounds=xds[f"{key}_bounds"],
            bounds_dim="bounds",
            order=None,
        )
        xds = xds.assign_coords({f"{key}_b": corners})
        xds = xds.drop_vars(f"{key}_bounds")

    if is_gaussian:
        xds = xds.drop_vars("lat_b")
        _, lat_b = gaussian_latitudes(len(xds.lat)//2)
        lat_b = np.concatenate([lat_b[:,0], [lat_b[-1,-1]]])
        if xds["lat"][0] > 0:
            lat_b = lat_b[::-1]
        xds["lat_b"] = xr.DataArray(
            lat_b,
            dims="lat_vertices",
        )
        xds = xds.set_coords("lat_b")
    return xds

def create_output_dataset(lat, lon, is_gaussian):
    xds = xr.Dataset({
        "lat": lat,
        "lon": lon,
    })
    return get_bounds(xds, is_gaussian)

def regrid_and_rename(xds, url):
    """Note that it's assumed the obs dataset is not on a Gaussian grid but input is"""

    obs = xr.open_zarr(url, storage_options={"token":"anon"})
    if "with_poles" in url:
        obs = obs.sel(latitude=slice(-89, 89))


    ds_out = create_output_dataset(
        lat=obs["latitude"].values,
        lon=obs["longitude"].values,
        is_gaussian=False,
    )
    if "lat_b" not in xds and "lon_b" not in xds:
        xds = get_bounds(xds, is_gaussian=True)

    regridder = xesmf.Regridder(
        ds_in=xds,
        ds_out=ds_out,
        method="conservative",
        reuse_weights=False,
    )
    ds_out = regridder(xds, keep_attrs=True)

    rename_dict = {
        "pressfc": "surface_pressure",
        "ugrd10m": "10m_u_component_of_wind",
        "vgrd10m": "10m_v_component_of_wind",
        "tmp2m": "2m_temperature",
        "tmp": "temperature",
        "ugrd": "u_component_of_wind",
        "vgrd": "v_component_of_wind",
        "dzdt": "vertical_velocity",
        "spfh": "specific_humidity",
        "prateb_ave": "total_preciptation_3hr",
        "lat": "latitude",
        "lon": "longitude",
    }
    rename_dict = {k: v for k,v in rename_dict.items() if k in ds_out}
    ds_out = ds_out.rename(rename_dict)
    ds_out = ds_out.transpose("time", ..., "longitude", "latitude")

    # ds_out has the lat/lon boundaries from input dataset
    # remove these because it doesn't make sense anymore
    ds_out = ds_out.drop_vars(["lat_b", "lon_b"])
    return ds_out


def swap_batch_time_dims(predictions, targets, inittimes):

    predictions = predictions.rename({"time": "lead_time"})
    targets = targets.rename({"time": "lead_time"})

    # create "time" dimension = t0
    predictions["time"] = xr.DataArray(
        inittimes,
        coords=predictions["batch"].coords,
        dims=predictions["batch"].dims,
        attrs={
            "description": "Forecast initialization time, last timestep of initial conditions",
        },
    )

    targets["time"] = xr.DataArray(
        inittimes,
        coords=targets["batch"].coords,
        dims=targets["batch"].dims,
        attrs={
            "description": "Forecast initialization time, last timestep of initial conditions",
        },
    )

    # swap logical batch for t0
    predictions = predictions.swap_dims({"batch": "time"}).drop_vars("batch")
    targets = targets.swap_dims({"batch": "time"}).drop_vars("batch")

    return predictions, targets

def make_fake_plevels(xds, plevels):

    xds = xds.rename({"level": "hybrid"})
    xds["level"] = xr.DataArray(
        np.array(list(plevels)),
        coords=xds.hybrid.coords,
        dims=xds.hybrid.dims,
    )
    xds = xds.swap_dims({"hybrid": "level"}).drop_vars("hybrid")

    xds = xds.sel(level=[100, 500, 850])
    return xds



def store_container(path, xds, time, **kwargs):

    if "time" in xds:
        xds = xds.isel(time=0, drop=True)

    container = xr.Dataset()
    for key in xds.coords:
        container[key] = xds[key].copy()

    for key in xds.data_vars:
        dims = ("time",) + xds[key].dims
        coords = {"time": time, **dict(xds[key].coords)}
        shape = (len(time),) + xds[key].shape
        chunks = (1,) + tuple(-1 for _ in xds[key].dims)

        container[key] = xr.DataArray(
            data=dask.array.zeros(
                shape=shape,
                chunks=chunks,
                dtype=xds[key].dtype,
            ),
            coords=coords,
            dims=dims,
            attrs=xds[key].attrs.copy(),
        )
    container.to_zarr(path, compute=False, **kwargs)
    logging.info(f"Stored container at {path}")

def predict(
    params,
    state,
    emulator,
    batchloader,
) -> xr.Dataset:

    @hk.transform_with_state
    def run_forward(inputs, targets_template, forcings):
        predictor = construct_wrapped_graphcast(emulator)
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    def with_params(fn):
        return partial(fn, params=params, state=state)

    def drop_state(fn):
        return lambda **kw: fn(**kw)[0]

    gc = drop_state(with_params(jax.jit(run_forward.apply)))

    hours = int(emulator.forecast_duration.value / 1e9 / 3600)
    pname = f"results/v1/{batchloader.dataset.mode}/predictions.fakeplevel.{hours}h.zarr"
    tname = f"results/v1/{batchloader.dataset.mode}/replay.fakeplevel.{hours}h.zarr"

    all_inittimes = batchloader.dataset.xds.datetime[emulator.n_input-1:-emulator.n_target]
    n_steps = len(batchloader)
    progress_bar = tqdm(total=n_steps, ncols=80, desc="Processing")
    for k in range(n_steps):
        inputs, targets, forcings = batchloader.get_data()

        # retrieve and drop t0
        inittimes = inputs.datetime.isel(time=-1).values
        inputs = inputs.drop_vars("datetime")
        targets = targets.drop_vars("datetime")
        forcings = forcings.drop_vars("datetime")

        predictions = rollout.chunked_prediction(
            gc,
            rng=jax.random.PRNGKey(0),
            inputs=inputs,
            targets_template=np.nan * targets,
            forcings=forcings,
        )

        # Add t0 as new variable, and swap out for logical sample/batch index
        predictions, targets = swap_batch_time_dims(predictions, targets, inittimes)

        # HACK: make fake pressure levels and select only a few
        predictions = make_fake_plevels(predictions, p1.pressure_levels)
        targets = make_fake_plevels(targets, p1.pressure_levels)

        # regrid and rename variables
        predictions = regrid_and_rename(predictions, p1.wb2_obs_url)
        targets = regrid_and_rename(targets, p1.wb2_obs_url)

        # Store to zarr one batch at a time
        if k == 0:
            store_container(pname, predictions, time=all_inittimes.values)
            store_container(tname, targets, time=all_inittimes.values)

        # Get time slice
        bs = batchloader.batch_size
        tslice = slice(k*bs, (k+1)*bs)

        # Store to zarr
        spatial_region = {k: slice(None, None) for k in predictions.dims if k != "time"}
        region = {
            "time": tslice,
            **spatial_region,
        }
        predictions.to_zarr(pname, region=region)
        targets.to_zarr(tname, region=region)

        progress_bar.update()


if __name__ == "__main__":

    timer = Timer()
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
    )
    p1, args = P1Emulator.from_parser()
    init_devices(p1)
    dask.config.set(scheduler="threads", num_workers=p1.dask_threads)

    vds = Dataset(
        p1,
        mode="validation",
        preload_batch=False,
    )

    validator = ExpandedBatchLoader(
        vds,
        batch_size=p1.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=p1.num_workers,
        max_queue_size=p1.max_queue_size,
    )

    # setup weights
    logging.info(f"Reading weights ...")
    # TODO: this looks in local_store_path, but we may want to look somewhere else
    params, state = p1.load_checkpoint(id=50)

    predict(
        params=params,
        state=state,
        emulator=p1,
        batchloader=validator,
    )
