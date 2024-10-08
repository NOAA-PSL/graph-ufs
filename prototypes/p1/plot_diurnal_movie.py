import numpy as np
import xarray as xr
import pandas as pd

import cmocean
import xmovie

from graphufs.spatialmap import get_extend

def swap_dims(xds):

    if "prediction_timedelta" in xds.coords and "lead_time" not in xds.coords:
        xds = xds.rename({"prediction_timedelta": "lead_time"})

    xds["fhr"] = (xds.lead_time.astype(int) / 3600 / 1e9).astype(int)
    xds = xds.swap_dims({"lead_time": "fhr"})
    return xds

def open_zarr(*args, **kwargs):
    xds = xr.open_zarr(*args, **kwargs)
    xds = swap_dims(xds)

    xds["valid_time"] = xds.time + xds.lead_time
    xds = xds.set_coords("valid_time")
    return xds

def movie_func(xds, fig, frame_index, *args, **kwargs):

    axs = []
    dalist = []

    truthname = [y for y in list(xds.data_vars) if y not in ("PSL", "GDM")][0]
    # Plot the two forecast datasets
    for i, name in enumerate(["PSL", "GDM"]):
        ax = fig.add_subplot(1, 3, i+1)
        axs.append(ax)

        plotme = xds[name].isel(frame_index=frame_index)
        dalist.append(plotme)
        plotme.plot(ax=ax, **kwargs)
        ax.set(
            xlabel="",
            ylabel="",
            title=f"{name}, fhr = {plotme.fhr.values:03d}h",
        )

    # now ERA5
    ax = fig.add_subplot(1, 3, 3)
    axs.append(ax)
    vtime = plotme.valid_time.values
    plotme = xds[truthname].isel(frame_index=frame_index)
    dalist.append(plotme)
    p = plotme.plot(ax=ax, **kwargs)
    ax.set(
        xlabel="",
        ylabel="",
        title=f"{truthname}, {str(vtime)[:13]}",
    )

    # now the colorbar
    extend, kwargs["vmin"], kwargs["vmax"] = get_extend(
        dalist,
        kwargs.get("vmin", None),
        kwargs.get("vmax,", None),
    )
    fig.colorbar(p, ax=axs, orientation="horizontal", shrink=.6, aspect=35, label=xds.attrs.get("label", ""), extend=extend)

    return None, None

if __name__ == "__main__":

    psl = open_zarr("/p1-evaluation/v1/validation/graphufs.240h.zarr")
    gdm = open_zarr("/p1-evaluation/gdm-v1/validation/graphufs_gdm.240h.zarr")
    era = xr.open_zarr(
        "gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr",
        storage_options={"token": "anon"},
    )

    # select the date we want
    date = "2022-01-02T06"
    ds = xr.Dataset({
        "PSL": psl["tmp2m"].sel(time=date).load(),
        "GDM": gdm["tmp2m"].sel(time=date).load(),
    })

    # Rename so they all have the same frame dimension
    frame_index = np.arange(len(psl.fhr))
    ds["frame_index"] = xr.DataArray(
        frame_index,
        coords=ds.fhr.coords,
    )
    ds = ds.set_coords("frame_index").swap_dims({"fhr": "frame_index"})

    ds["ERA5"] = era["2m_temperature"].sel(time=ds["PSL"].valid_time.values).load()
    ds["ERA5"] = ds["ERA5"].swap_dims({"time": "frame_index"})

    # Convert to degC
    for key in ds.data_vars:
        ds[key] -= 273.15
        ds[key].attrs["units"] = "degC"

    ds.attrs["label"] = r"2m Temperature ($^\circ C$)"

    mov = xmovie.Movie(
        ds,
        movie_func,
        framedim="frame_index",
        input_check=False,
        cmap="cmo.thermal",
        vmin=-10,
        vmax=30,
        add_colorbar=False,
    )
    mov.save(f"figures/diurnal_tmp2m_{date}.mp4", progress=True, overwrite_existing=True)
