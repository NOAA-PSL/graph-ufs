import numpy as np
import dask
import xarray as xr
import xesmf
import cf_xarray as cfxr
import pandas as pd

from ufs2arco.regrid.gaussian_grid import gaussian_latitudes
from ufs2arco import Layers2Pressure

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

    if "lead_time" in forecast.dims:
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

    else:
        valid_time = list(set(truth["time"].values).intersection(set(forecast.time.values)))
        initial_times = xr.DataArray(
            valid_time,
            coords={"time": valid_time},
            dims=("time",),
        )
    initial_times = initial_times.sortby("time")

    return initial_times


def regrid_and_rename(xds, truth):
    """Note that it's assumed the truth dataset is not on a Gaussian grid but input is"""

    ds_out = create_output_dataset(
        lat=truth["latitude"].values,
        lon=truth["longitude"].values,
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
        "prateb_ave": "total_precipitation_3hr",
        "lat": "latitude",
        "lon": "longitude",
    }
    rename_dict = {k: v for k,v in rename_dict.items() if k in ds_out}
    ds_out = ds_out.rename(rename_dict)

    # ds_out has the lat/lon boundaries from input dataset
    # remove these because it doesn't make sense anymore
    ds_out = ds_out.drop_vars(["lat_b", "lon_b"])
    return ds_out


def interp2pressure(xds, plevels):
    """Assume plevels is in hPa"""

    lp = Layers2Pressure()
    if "delz" not in xds:
        xds["delz"] = lp.calc_delz(xds["pressfc"], xds["tmp"], xds["spfh"])
    prsl = lp.calc_layer_mean_pressure(xds["pressfc"], xds["tmp"], xds["spfh"], xds["delz"])

    vars2d = [f for f in xds.keys() if "level" not in xds[f].dims]
    vars3d = [f for f in xds.keys() if "level" in xds[f].dims]
    pds = xr.Dataset({f: xds[f] for f in vars2d})
    plevels = np.array(list(plevels))
    pds["level"] = xr.DataArray(
        plevels,
        coords={"level": plevels},
        dims=("level",),
        attrs={
            "description": "Pressure level",
            "units": "hPa",
        },
    )
    results = {k: list() for k in vars3d}
    for p in plevels:

        cds = lp.get_interp_coefficients(p*100, prsl)
        mask = (cds["is_right"].sum("level") > 0) & (cds["is_left"].sum("level") > 0)
        for key in vars3d:
            interpolated = lp.interp2pressure(xds[key], p*100, prsl, cds)
            interpolated = interpolated.expand_dims({"level": [p]})
            interpolated = interpolated.where(mask)
            results[key].append(interpolated)

    for key in vars3d:
        pds[key] = xr.concat(results[key], dim="level")

    return pds
