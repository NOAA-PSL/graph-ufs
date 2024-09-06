import numpy as np
import xarray as xr
import pandas as pd

import cmocean
import xmovie

from graphufs.spatialmap import get_extend
from plot_diurnal_movie import open_zarr, swap_dims, movie_func

def calc_wind_speed(xds):
    if "ugrd" in xds:
        ws = np.sqrt(xds["ugrd"]**2 + xds["vgrd"]**2)
    else:
        ws = np.sqrt(xds["u_component_of_wind"]**2 + xds["v_component_of_wind"]**2)
    ws.attrs["units"] = "m/sec"
    ws.attrs["long_name"] = "Wind Speed"
    return ws

def get_truth(name):
    if name.lower() == "era5":
        url = "gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr"
        rename = {}
    elif name.lower() == "replay":
        url = "gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree/03h-freq/zarr/fv3.zarr"
        rename = {"pfull": "level", "grid_yt": "lat", "grid_xt": "lon"}

    truth = xr.open_zarr(
        url,
        storage_options={"token": "anon"},
    )
    truth = truth.rename(rename)
    truth.attrs["name"] = name
    return truth


if __name__ == "__main__":

    # select the date and pressure level we want
    date = "2022-01-02T06"
    level = 50 # hPa

    psl = open_zarr("/p1-evaluation/v1/validation/graphufs.240h.zarr")
    psl = psl.isel(fhr=slice(1, None, 2))
    gdm = open_zarr("/p1-evaluation/gdm-v1/validation/graphufs_gdm.240h.zarr")

    for tname in ["ERA5", "Replay"]:

        truth = get_truth(tname)

        # Compute this
        psl["wind_speed"] = calc_wind_speed(psl)
        gdm["wind_speed"] = calc_wind_speed(gdm)
        truth["wind_speed"] = calc_wind_speed(truth)

        # setup for each variable
        plot_options = {
            "tmp": {
                "cmap": "cmo.thermal",
                "vmin": -70,
                "vmax": -45,
            },
            "spfh": {
                "cmap": "cmo.haline",
                "vmin": 2e-6,
                "vmax": 3e-6,
            },
            "wind_speed": {
                "cmap": "cmo.tempo",
                "vmin": 0,
                "vmax": 50,
            },
        }

        evars = {"tmp": "temperature", "spfh": "specific_humidity", "wind_speed": "wind_speed"}

        for varname, options in plot_options.items():

            ds = xr.Dataset({
                "PSL": psl[varname].sel(time=date).sel(level=level, method="nearest", drop=True).load(),
                "GDM": gdm[varname].sel(time=date).sel(level=level, method="nearest", drop=True).load(),
            })

            # Rename so they all have the same frame dimension
            frame_index = np.arange(len(psl.fhr))
            ds["frame_index"] = xr.DataArray(
                frame_index,
                coords=ds.fhr.coords,
            )
            ds = ds.set_coords("frame_index").swap_dims({"fhr": "frame_index"})

            tvname = evars[varname] if truth.name == "ERA5" else varname
            ds[truth.name] = truth[tvname].sel(
                time=ds["PSL"].valid_time.values,
            ).sel(
                level=level, method="nearest", drop=True,
            ).load()
            ds[truth.name] = ds[truth.name].swap_dims({"time": "frame_index"})

            # Convert to degC
            if varname == "tmp":
                for key in ds.data_vars:
                    ds[key] -= 273.15
                    ds[key].attrs["units"] = "degC"

            ds.attrs["label"] = f"{evars[varname]} ({ds[truth.name].units})"

            mov = xmovie.Movie(
                ds,
                movie_func,
                framedim="frame_index",
                input_check=False,
                add_colorbar=False,
                **options
            )
            mov.save(f"figures/graphufs_vs_{truth.name.lower()}_{varname}_{level}hpa_{date}.mp4", progress=True, overwrite_existing=True)
