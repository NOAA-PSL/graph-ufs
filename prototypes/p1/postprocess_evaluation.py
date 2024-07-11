import logging
import numpy as np
import dask
import xarray as xr
import xesmf
import cf_xarray as cfxr

from ufs2arco.regrid.gaussian_grid import gaussian_latitudes

from p1stacked import P1Emulator
from stacked_preprocess import setup_log

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

def get_valid_initial_conditions(forecast, truth):

    forecast_valid_time = forecast["time"] + forecast["lead_time"]
    valid_time = list(set(truth["time"].values).intersection(set(forecast_valid_time.values.flatten())))

    initial_times = xr.where(
        [t0 in valid_time and tf in valid_time for t0, tf in zip(
            forecast.time.values,
            forecast_valid_time.isel(lead_time=-1, drop=True).values
        )],
        forecast["time"],
        np.datetime64("NaT"),
    ).dropna("time")
    return initial_times


def regrid_and_rename(xds, url):
    """Note that it's assumed the obs dataset is not on a Gaussian grid but input is"""

    obs = xr.open_zarr(url, storage_options={"token":"anon"})

    # subsample in space to avoid poles
    if "with_poles" in url:
        obs = obs.sel(latitude=slice(-89, 89))

    # subsample forecast dataset in time, only get what we have truth to compare to
    t0 = get_valid_initial_conditions(xds, obs)
    xds = xds.sel(time=t0)

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

if __name__ == "__main__":

    setup_log()
    p1, args = P1Emulator.from_parser()
    dask.config.set(scheduler="threads", num_workers=p1.dask_threads)

    duration = p1.target_lead_time[-1]

    for name in ["predictions", "replay"]:
        ds = xr.open_zarr(f"/p1-evaluation/v1/validation/{name}.{duration}.zarr")

        ds = ds[["pressfc", "tmp2m", "ugrd10m", "vgrd10m"]]

        # HACK: make fake pressure levels and select only a few
        #ds = make_fake_plevels(ds, p1.pressure_levels)
        #logging.info(f"Selected fake pressure levels...")

        # regrid and rename variables
        ds = regrid_and_rename(ds, p1.wb2_obs_url)
        logging.info(f"Done regridding...")

        path = f"/p1-evaluation/v1/validation/{name}.{duration}.postprocessed.zarr"
        ds.to_zarr(path)
        logging.info(f"Done writing to {path}")
